"""Independent measured affine wind calibration followed by two matched public fits."""

import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from run_metec_comparison import ROOT,OUT,AUDIT,read,save,load_input,JointBounds
from run_ia_wind_ensemble_pilot import load_wind_members
from ia_metec_point_adapter import configure,xy_from_lon_lat,window_mean_release
from paper_affine_wind import AffineWindHistory,fit_incompressible_wind
from paper_beam_joint_baselines import BeamJointResidual
from paper_joint_baselines import fit_joint_nls
from paper_q_model_selection import fit_profiled_q_models

DEST=OUT.parent/'ia_affine_wind_pilot_20261028'


def calibration(data,meta,raw,sensors):
    names=['E','NW','SW']
    winds,audit=load_wind_members(raw,pd.to_datetime(data['time_utc'],utc=True),names,data['wind_xy'])
    records=sensors.loc[names]
    positions=xy_from_lon_lat(records.lon,records.lat,meta['origin'])
    vectors=np.stack(winds,axis=1)
    velocity,gradient,fitted=fit_incompressible_wind(positions,vectors)
    deviations=vectors-vectors.mean(axis=1,keepdims=True)
    history=AffineWindHistory(np.arange(len(vectors)+1)*60.,velocity,gradient,data['wind_xy'])
    return history,dict(wind_stations=names,missing=audit,
        wind_rms_residual_m_s=float(np.sqrt(np.mean((fitted-vectors)**2))),
        station_variation_explained=1-float(np.sum((fitted-vectors)**2)/max(np.sum(deviations**2),1e-30)),
        max_gradient_per_s=float(np.linalg.norm(gradient,axis=(1,2)).max()),
        velocity_intercepts_m_s=velocity,gradient_per_s=gradient)


def covariance_preflight(plan):
    valid=plan.weighted_basis.abs().sum(-1)>0
    maximum=0.
    for perp,contrast in [(.1,0.),(.1,60.),(60.,0.),(60.,60.)]:
        covariance=2*(perp*(plan.parallel+plan.perpendicular)+contrast*plan.parallel)
        xx,xy,yy=[covariance[...,index][valid] for index in range(3)]
        determinant=xx*yy-xy*xy
        if not bool(torch.all((xx>0)&(yy>0)&(determinant>0)&torch.isfinite(determinant))):
            raise FloatingPointError('Affine covariance lost positive definiteness; no clipping applied')
        maximum=max(maximum,float(((xx+yy)**2/determinant).max()))
    return dict(all_domain_corners_positive=True,max_trace_squared_over_determinant=maximum)


def run():
    torch.set_num_threads(1)
    parent=read(OUT/'protocol.json')
    cases=[]
    heights=set()
    for identifier in parent['event_ids']:
        height=read(OUT/'inputs'/identifier/'metadata.json')['source_height_m']
        if height not in heights:
            cases.append(identifier)
            heights.add(height)
    files=[Path(__file__),Path(__file__).with_name('paper_affine_wind.py'),
        Path(__file__).with_name('metec_point_adapter.py'),Path(__file__).with_name('paper_physical_beam_adapter.py')]
    protocol=dict(version='measured_incompressible_affine_wind_pilot_v1',cases=cases,
        selection='First event per supplied height in frozen chronological order; development only',
        physical_model='Known piecewise-affine divergence-free horizontal velocity, exact Gaussian mean/covariance propagation',
        wind='Traceless affine least squares fit to E/NW/SW vectors only; diffusion orientation follows original mean wind',
        caveat='Affine extrapolation and space-uniform diffusion remain assumptions; no full CFD claim',
        observations='All valid event concentrations; original background and seconds-level release window',
        objective='Original unweighted squared residual; no additional loss or withholding',
        initial='Original common two starts and bounds; matched IA and NLS physical responses',
        max_nfev=250,orders=parent['numerical_orders'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve old affine pilot; changed protocol requires a new directory')
    save(DEST/'protocol.json',protocol)
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    sensors=pd.read_csv(AUDIT/'raw/input_data/sensor_locations.csv').set_index('name')
    rows=[]
    for identifier in cases:
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        history,audit=calibration(data,meta,raw,sensors)
        save(DEST/identifier/'wind_calibration.json',audit)
        begun=time.perf_counter()
        print('AFFINE_PLAN_START',identifier,flush=True)
        orders=protocol['orders']
        model,xyt,basis,response,observed=configure(data,meta['source_height_m'],device='cuda',
            time_order=orders['time'],release_order=orders['release'],history_minutes=orders['history_minutes'],
            transport_history=history)
        check=covariance_preflight(response.plan)
        check['plan_seconds']=time.perf_counter()-begun
        save(DEST/identifier/'numerical_preflight.json',check)
        print('AFFINE_PLAN_DONE',identifier,round(check['plan_seconds'],1),flush=True)
        initial=read(OUT/'fits'/identifier/'initialization.json')
        bounds=JointBounds(**initial['bounds'])
        starts=[np.array(s) for s in initial['starts']]
        for method in ['ia_affine_wind','joint_nls_affine_wind']:
            output=DEST/identifier/(method+'.json')
            if not output.exists():
                print('AFFINE_FIT_START',identifier,method,flush=True)
                objective=BeamJointResidual(response,model,0.,xyt,basis,observed,bounds)
                fit=(fit_profiled_q_models(objective,starts,max_nfev=protocol['max_nfev'])
                    if method.startswith('ia_') else fit_joint_nls(objective,starts,protocol['max_nfev']))
                save(output,fit)
                print('AFFINE_FIT_DONE',identifier,method,round(fit['seconds'],1),flush=True)
        # Evaluation-only truth is not used to calibrate wind or select Q capacity.
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method in ['ia_affine_wind','joint_nls_affine_wind','ia_vpdi','joint_nls_trf']:
            fit=read(DEST/identifier/(method+'.json')) if 'affine' in method else read(OUT/'fits'/identifier/(method+'.json'))
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.asarray(fit['source_norm'])*100-true_xy)),
                q_kg_h=3600*window_mean_release(fit['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'],rmse_ppm=float(np.sqrt(fit['mse'])),
                q_basis_count=fit.get('q_basis_count',5),converged=fit.get('converged')))
        del model,response,objective
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(pilot_fits_complete=True,full_goal_achieved=False,
        high_order_replay_pending=True,physical_model_independent_validation_pending=True))


if __name__=='__main__':
    raise SystemExit('Archived matched-NLS control. NLS is frozen; use IA-only experiments.')
