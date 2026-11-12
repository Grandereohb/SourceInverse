"""IA-only shared wind transfer: wind audit first, optional fixed two-case pilot."""

import argparse
import gc
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT,OUT,AUDIT,STUDY,read,save,JointBounds
from run_ia_wind_ensemble_pilot import load_wind_members
from paper_affine_wind import fit_incompressible_wind
from paper_sqrt_affine_wind import SquareRootAffineWindHistory
from ia_wind_transfer import fit_shared_wind_transfer
from ia_metec_point_adapter import configure,xy_from_lon_lat,window_mean_release
from paper_beam_joint_baselines import BeamJointResidual
from paper_q_model_selection import fit_profiled_q_models

DEST=STUDY/'ia_shared_wind_pilot_20261028'


def wind_fit(data,meta,raw,sensors):
    names=['E','NW','SW']
    members,missing=load_wind_members(raw,pd.to_datetime(data['time_utc'],utc=True),names,data['wind_xy'])
    stations=sensors.loc[names]
    positions=xy_from_lon_lat(stations.lon,stations.lat,meta['origin'])
    vectors=np.stack(members,axis=1)
    shared=fit_shared_wind_transfer(positions,vectors)
    _,instant_gradient,instant_fitted=fit_incompressible_wind(positions,vectors)
    shared.update(missing=missing,instantaneous_wind_sse=float(np.sum((instant_fitted-vectors)**2)),
        shared_gradient_rms_per_s=float(np.sqrt(np.mean(shared['gradient']**2))),
        instantaneous_gradient_rms_per_s=float(np.sqrt(np.mean(instant_gradient**2))))
    return shared


def run(mode):
    torch.set_num_threads(1)
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('NLS baseline changed: '+name)
    parent=read(OUT/'protocol.json')
    pilot_cases=read(STUDY/'ia_location_capacity_pilot_20261028/protocol.json')['public'][:2]
    sources=[Path(__file__),ROOT/'paper_experiments/ia_wind_transfer.py',
        ROOT/'paper_experiments/ia_metec_point_adapter.py',ROOT/'paper_experiments/ia_physical_beam_adapter.py',
        ROOT/'paper_experiments/paper_sqrt_affine_wind.py',ROOT/'paper_experiments/paper_sqrt_gaussian.py']
    protocol=dict(wind_audit_cases=parent['event_ids'],pilot_cases=pilot_cases,
        selection='Same first two public development panel events, fixed before shared-wind fitting',
        wind_model='A(t)=mean_u_x(t)*Bx+mean_u_y(t)*By, traceless Bx/By; six coefficients shared across event',
        wind_data='All E/NW/SW wind observations; no concentrations or source truth in wind calibration',
        inversion='IA only, all concentrations, original SSE, Q BIC1/3/5, original two starts/bounds/nfev250',
        numerical_orders=parent['numerical_orders'],
        nls='Read frozen results only; no modifications or optimization',
        limitation='Wind fit is not source-location validation; local linear flow is not CFD',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve previous shared-wind protocol')
    save(DEST/'protocol.json',protocol)
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    sensors=pd.read_csv(AUDIT/'raw/input_data/sensor_locations.csv').set_index('name')
    audit=[]
    for identifier in protocol['wind_audit_cases']:
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        shared=wind_fit(data,meta,raw,sensors)
        save(DEST/identifier/'wind_calibration.json',shared)
        audit.append(dict(event_id=identifier,minutes=len(data['concentration']),
            **{key:shared[key] for key in ['wind_sse','uniform_wind_sse','instantaneous_wind_sse',
                'variance_fraction_explained','shared_parameter_rank','design_condition',
                'shared_gradient_rms_per_s','instantaneous_gradient_rms_per_s']}))
    pd.DataFrame(audit).to_csv(DEST/'wind_audit.csv',index=False,encoding='utf-8-sig')
    if mode=='audit':
        save(DEST/'audit_status.json',dict(wind_audit_complete=True,events=len(audit),
            inversion_run=False,nls_called=False,source_truth_read=False))
        print('WIND_AUDIT_COMPLETE',len(audit),flush=True)
        return
    rows=[]
    for identifier in pilot_cases:
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        shared=read(DEST/identifier/'wind_calibration.json')
        history=SquareRootAffineWindHistory(np.arange(len(data['concentration'])+1)*60.,
            shared['velocity'],shared['gradient'],data['wind_xy'])
        output=DEST/identifier/'ia_fit.json'
        if output.exists():
            fit=read(output)
        else:
            print('SHARED_WIND_START',identifier,flush=True)
            orders=protocol['numerical_orders']
            model,xyt,basis,response,observed=configure(data,meta['source_height_m'],device='cuda',
                time_order=orders['time'],release_order=orders['release'],history_minutes=orders['history_minutes'],
                transport_history=history)
            initial=read(OUT/'fits'/identifier/'initialization.json')
            objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
            fit=fit_profiled_q_models(objective,[np.array(start) for start in initial['starts']],max_nfev=250)
            save(output,fit)
            del model,response,objective
            gc.collect()
            torch.cuda.empty_cache()
        truth=read(OUT/'inputs'/identifier/'truth.json')
        xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method,item in [('ia_shared_wind',fit),('original_ia',read(OUT/'fits'/identifier/'ia_vpdi.json')),
                            ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]:
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.asarray(item['source_norm'])*100-xy)),
                q_kg_h=3600*window_mean_release(item['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                rmse_ppm=float(np.sqrt(item['mse'])),converged=item.get('converged')))
        print('SHARED_WIND_DONE',identifier,rows[-3],flush=True)
        pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(pilot_complete=True,nls_called=False,production_promoted=False,
        numerical_replay_pending=True,full_goal_achieved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['audit','pilot'],default='audit')
    run(parser.parse_args().mode)
