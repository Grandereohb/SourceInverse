"""Frozen17-event IA-only shared-wind evaluation with optional numerical replay."""

import argparse
import gc
import hashlib
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT,OUT,AUDIT,STUDY,read,save,JointBounds
from run_ia_shared_wind_pilot import DEST as PILOT,wind_fit
from ia_metec_point_adapter import configure,xy_from_lon_lat,window_mean_release
from paper_sqrt_affine_wind import SquareRootAffineWindHistory
from paper_beam_joint_baselines import BeamJointResidual
from paper_q_model_selection import fit_profiled_q_models
from analyze_affine_accuracy import intervals

DEST=STUDY/'ia_shared_wind_full_20261028'


def history_from(data,calibration):
    return SquareRootAffineWindHistory(np.arange(len(data['concentration'])+1)*60.,
        calibration['velocity'],calibration['gradient'],data['wind_xy'])


def run(mode):
    torch.set_num_threads(1)
    parent=read(OUT/'protocol.json')
    pilot=read(PILOT/'protocol.json')
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('NLS baseline changed: '+name)
    for name,digest in parent['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Original input changed: '+name)
    sources=[Path(__file__),*[ROOT/name for name in pilot['source_sha256']],
        ROOT/'paper_experiments/paper_q_model_selection.py',
        ROOT/'paper_experiments/paper_reduced_solver.py',
        ROOT/'pinn_source/reduced_trust_region.py']
    for name,digest in pilot['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Pilot method changed: '+name)
    protocol=dict(event_ids=parent['event_ids'],
        method='Fixed6-coefficient measured-wind transfer, original SSE and IA Q BIC1/3/5',
        initial='Original common two starts, bounds and nfev250; no source truth',
        numerical_orders=parent['numerical_orders'],
        replay_orders=dict(time=32,release=32,history_minutes=40),batch_minutes=8,threshold=.01,
        reuse='Pilot predictions must agree under the unchanged response at relative L2 <=1e-8',
        failure='Keep every event and explicit failures; do not summarize a favourable incomplete subset',
        nls='Frozen7563376 outputs only; no optimization or modification',
        scope='Complete fixed-site development cohort, not independent-site confirmation',
        pilot_protocol_sha256=hashlib.sha256((PILOT/'protocol.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve previous full-cohort results/protocol')
    save(DEST/'protocol.json',protocol)
    for source in sources:
        target=DEST/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source,target)
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    sensors=pd.read_csv(AUDIT/'raw/input_data/sensor_locations.csv').set_index('name')
    for number,identifier in enumerate(protocol['event_ids'],1):
        folder=DEST/identifier
        if (folder/'ia_fit.json').exists() or (folder/'failed.json').exists():
            continue
        if mode=='replay':
            raise ValueError('Complete all IA fits before replay')
        print('SHARED_FULL_START',number,'of17',identifier,flush=True)
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        calibration=wind_fit(data,meta,raw,sensors)
        save(folder/'wind_calibration.json',calibration)
        history=history_from(data,calibration)
        model=response=objective=None
        try:
            orders=protocol['numerical_orders']
            model,xyt,basis,response,observed=configure(data,meta['source_height_m'],device='cuda',
                time_order=orders['time'],release_order=orders['release'],history_minutes=orders['history_minutes'],
                transport_history=history)
            reused=False
            if identifier in pilot['pilot_cases']:
                previous=read(PILOT/identifier/'ia_fit.json')
                theta=model.xs.new_tensor([*previous['source_norm'],previous['diffusion_norm'][1],
                    previous['diffusion_norm'][0]-previous['diffusion_norm'][1]])
                prediction=(response(model,0.,xyt,basis,physical_parameters=theta)@
                    model.xs.new_tensor(previous['coefficients'])).cpu().numpy()
                delta=float(np.linalg.norm(prediction-np.array(previous['prediction']))/
                    max(np.linalg.norm(prediction),1e-30))
                save(folder/'reuse_check.json',dict(relative_l2=delta,passed=delta<=1e-8,
                    source_sha256=hashlib.sha256((PILOT/identifier/'ia_fit.json').read_bytes()).hexdigest()))
                if delta<=1e-8:
                    save(folder/'ia_fit.json',previous)
                    reused=True
            if not reused:
                initial=read(OUT/'fits'/identifier/'initialization.json')
                objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
                fit=fit_profiled_q_models(objective,[np.array(s) for s in initial['starts']],max_nfev=250)
                save(folder/'ia_fit.json',fit)
            print('SHARED_FULL_DONE',number,identifier,'reused' if reused else 'fitted',flush=True)
        except (FloatingPointError,np.linalg.LinAlgError,RuntimeError) as exc:
            save(folder/'failed.json',dict(type=type(exc).__name__,message=str(exc),nls_called=False))
            print('SHARED_FULL_FAILED',identifier,str(exc),flush=True)
        finally:
            del model,response,objective
            gc.collect()
            torch.cuda.empty_cache()
    failed=[identifier for identifier in protocol['event_ids'] if (DEST/identifier/'failed.json').exists()]
    save(DEST/'status.json',dict(expected_events=17,complete_events=17-len(failed),failed_events=failed,
        nls_called=False,numerical_replay_pending=True,production_promoted=False,full_goal_achieved=False))
    if failed:
        raise RuntimeError('Incomplete cohort retained explicitly: '+str(failed))
    rows=[]
    for identifier in protocol['event_ids']:
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method,fit in [('ia_shared_wind',read(DEST/identifier/'ia_fit.json')),
            ('original_ia',read(OUT/'fits'/identifier/'ia_vpdi.json')),
            ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]:
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(fit['source_norm'])*100-true_xy)),
                q_kg_h=3600*window_mean_release(fit['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'],rmse_ppm=float(np.sqrt(fit['mse'])),
                q_basis_count=fit.get('q_basis_count',5),converged=fit.get('converged')))
    frame=pd.DataFrame(rows)
    frame.to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    frame.groupby('method').localization_error_m.agg(['mean','median','count']).to_csv(DEST/'summary.csv',encoding='utf-8-sig')
    wide=frame.pivot(index='event_id',columns='method',values='localization_error_m')
    groups=[read(OUT/'inputs'/identifier/'truth.json')['equipment_group'] for identifier in wide.index]
    paired=intervals(wide.ia_shared_wind-wide.frozen_nls,groups,np.random.default_rng(20261028))
    save(DEST/'paired_accuracy.json',dict(comparison=paired,events=17,equipment_groups=len(set(groups)),
        scope='Exploratory development comparison; changes include measured wind modelling, not pure elimination'))
    print('SHARED_FULL_PAIRED',paired,flush=True)
    if mode=='fit':
        return
    replay=[]
    for identifier in protocol['event_ids']:
        folder=DEST/identifier
        path=folder/'replay32.json'
        if path.exists():
            replay.append(read(path))
            continue
        print('SHARED_REPLAY_START',identifier,flush=True)
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        fit=read(folder/'ia_fit.json')
        history=history_from(data,read(folder/'wind_calibration.json'))
        predictions=[]
        for first in range(0,len(data['concentration']),protocol['batch_minutes']):
            query=np.arange(first,min(first+protocol['batch_minutes'],len(data['concentration'])))
            model,xyt,basis,response,observed=configure(data,meta['source_height_m'],device='cuda',
                time_order=32,release_order=32,history_minutes=40,transport_history=history,observation_minutes=query)
            theta=model.xs.new_tensor([*fit['source_norm'],fit['diffusion_norm'][1],
                fit['diffusion_norm'][0]-fit['diffusion_norm'][1]])
            value=(response(model,0.,xyt,basis,physical_parameters=theta)@
                model.xs.new_tensor(fit['coefficients'])).cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError('Nonfinite shared-wind replay')
            predictions.append(value)
            del model,response
        prediction=np.concatenate(predictions)
        delta=float(np.linalg.norm(prediction-np.array(fit['prediction']))/max(np.linalg.norm(prediction),1e-30))
        item=dict(event_id=identifier,relative_l2=delta,passed=delta<=protocol['threshold'])
        save(path,item)
        np.savez_compressed(folder/'replay32_prediction.npz',prediction=prediction)
        replay.append(item)
        print('SHARED_REPLAY_DONE',item,flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    status=read(DEST/'status.json')
    status.update(numerical_replay_pending=False,numerically_passed=all(item['passed'] for item in replay),replay_rows=replay)
    save(DEST/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['fit','replay','all'],default='fit')
    run(parser.parse_args().mode)
