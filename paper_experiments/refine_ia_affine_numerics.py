"""IA-only numerical repair, triggered by replay error rather than source truth.

Keep every original event and the frozen NLS baseline. Reuse numerically
accepted IA estimates; refine only the three failed replay cases. This does
not identify a pure variable-projection gain over a matched forward model.
"""

import gc
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT,OUT,AUDIT,STUDY,read,save,JointBounds
from run_ia_affine_wind_pilot import calibration
from ia_metec_point_adapter import configure,xy_from_lon_lat,window_mean_release
from paper_sqrt_affine_wind import SquareRootAffineWindHistory
from paper_beam_joint_baselines import BeamJointResidual
from paper_location_capacity import capacity_response
from reduced_trust_region import fit_reduced_trust_region
from analyze_affine_accuracy import intervals

PREVIOUS=STUDY/'ia_affine_wind_sqrt_full_20261028'
DEST=STUDY/'ia_affine_numerical_repair_20261028'


def make_history(data,meta,raw,sensors):
    history,audit=calibration(data,meta,raw,sensors)
    return SquareRootAffineWindHistory(history.edges,history.velocity,
        history.gradient,data['wind_xy']),audit


def run():
    torch.set_num_threads(1)
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('Frozen NLS changed: '+name)
    parent=read(PREVIOUS/'protocol.json')
    replay=read(PREVIOUS/'replay_summary.json')
    cases=parent['cases']
    failed=[row['event_id'] for row in replay['rows']
            if row['method']=='ia_affine_wind' and not row['passed']]
    sources=[Path(__file__),*[ROOT/'paper_experiments'/name for name in
        ['ia_metec_point_adapter.py','ia_physical_beam_adapter.py',
         'paper_affine_wind.py','paper_sqrt_affine_wind.py','paper_sqrt_gaussian.py',
         'paper_location_capacity.py','run_ia_affine_wind_pilot.py',
         'run_ia_wind_ensemble_pilot.py','analyze_affine_accuracy.py']],
         ROOT/'pinn_source/reduced_trust_region.py']
    protocol=dict(event_ids=cases,refine_events=failed,
        trigger='Original IA16/16-history20 versus replay32/32-history40 relative response change exceeds1%; no truth',
        fitting='Only IA, existing selected Q capacity, same SSE/bounds, one previous physical optimum start',
        fit_orders=dict(time=32,release=32,history_minutes=40),max_nfev=100,
        replay_orders=dict(time=64,release=64,history_minutes=60),batch_minutes=8,threshold=.01,
        accepted_reuse='Other14 previous IA estimates already passed32/32-history40 replay; no favourable-event selection',
        nls='7563376 frozen baseline, historical results only; not executed or tuned',
        interpretation='Full-method wind-model comparison, not an isolated solver advantage',
        parent_protocol_sha256=hashlib.sha256((PREVIOUS/'protocol.json').read_bytes()).hexdigest(),
        previous_replay_sha256=hashlib.sha256((PREVIOUS/'replay_summary.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve previous repair protocol/results')
    save(DEST/'protocol.json',protocol)
    for source in sources:
        target=DEST/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source,target)
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    sensors=pd.read_csv(AUDIT/'raw/input_data/sensor_locations.csv').set_index('name')
    # One-minute Eulerian means are not a direct measurement of the Lagrangian
    # velocity autocovariance required to calibrate physical eddy diffusivity.
    wind=raw[raw['wind.speed'].notna() & raw['wind.direction'].notna()]
    cadence={}
    for name,group in wind.groupby('name'):
        clock=group.timestamp.drop_duplicates().sort_values()
        cadence[name]=float(clock.diff().dt.total_seconds().dropna().median())
    save(DEST/'wind_information_audit.json',dict(station_sampling_median_s=cadence,
        directly_identifies_turbulent_diffusion=False,
        reason='Minute-averaged fixed-point Eulerian vectors do not uniquely identify sub-minute Lagrangian decorrelation or turbulent diffusivity',
        action='Do not turn their variance into a fixed D bound; preserve the existing physical fit bounds'))
    for identifier in failed:
        folder=DEST/identifier
        if (folder/'complete.json').exists():
            continue
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        history,audit=make_history(data,meta,raw,sensors)
        save(folder/'wind_calibration.json',audit)
        old=read(PREVIOUS/identifier/'ia_affine_wind.json')
        path=folder/'ia_fit.json'
        if path.exists():
            fit=read(path)
        else:
            print('IA_NUMERIC_START',identifier,flush=True)
            begun=time.perf_counter()
            model,xyt,basis,response,observed=configure(data,meta['source_height_m'],
                device='cuda',time_order=32,release_order=32,history_minutes=40,
                transport_history=history)
            plan_seconds=time.perf_counter()-begun
            initial=read(OUT/'fits'/identifier/'initialization.json')
            objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
            count=old['q_basis_count']
            function,mapping=capacity_response(objective,count)
            start=np.r_[old['source_norm'],old['diffusion_norm'][1],
                         old['diffusion_norm'][0]-old['diffusion_norm'][1]]
            fit=fit_reduced_trust_region(function,observed,[start],
                objective.bounds.lower[:4],objective.bounds.upper[:4],
                q_upper_bound=float(objective.bounds.upper[4]),max_nfev=protocol['max_nfev'])
            theta=fit.pop('theta')
            native=np.array(fit['coefficients'])
            fit.update(source_norm=theta[:2],diffusion_norm=[theta[2]+theta[3],theta[2]],
                native_coefficients=native,coefficients=mapping@native,q_basis_count=count,
                mse=fit['sse']/len(observed),plan_seconds=plan_seconds,
                same_selected_q_capacity=True,nls_called=False)
            save(path,fit)
            print('IA_NUMERIC_FIT_DONE',identifier,fit['sse'],flush=True)
            del model,response,objective,function
            gc.collect()
            torch.cuda.empty_cache()
        predictions=[]
        print('IA_NUMERIC_REPLAY_START',identifier,flush=True)
        for first in range(0,len(data['concentration']),protocol['batch_minutes']):
            query=np.arange(first,min(first+protocol['batch_minutes'],len(data['concentration'])))
            model,xyt,basis,response,observed=configure(data,meta['source_height_m'],
                device='cuda',time_order=64,release_order=64,history_minutes=60,
                transport_history=history,observation_minutes=query)
            theta=model.xs.new_tensor([*fit['source_norm'],fit['diffusion_norm'][1],
                fit['diffusion_norm'][0]-fit['diffusion_norm'][1]])
            with torch.no_grad():
                values=(response(model,0.,xyt,basis,physical_parameters=theta)@
                    model.xs.new_tensor(fit['coefficients'])).cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite high-order IA replay')
            predictions.append(values)
            del model,response
        prediction=np.concatenate(predictions)
        delta=float(np.linalg.norm(prediction-np.asarray(fit['prediction']))/
                    max(np.linalg.norm(prediction),1e-30))
        save(folder/'replay.json',dict(relative_l2=delta,passed=delta<=protocol['threshold'],
            prediction=prediction,orders=protocol['replay_orders']))
        save(folder/'complete.json',dict(fitted=True,numerically_passed=delta<=protocol['threshold']))
        print('IA_NUMERIC_REPLAY_DONE',identifier,delta,flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    rows=[]
    numerical=[]
    for identifier in cases:
        is_refined=identifier in failed
        fit=read(DEST/identifier/'ia_fit.json') if is_refined else read(PREVIOUS/identifier/'ia_affine_wind.json')
        check=(read(DEST/identifier/'replay.json') if is_refined else
               next(row for row in replay['rows'] if row['event_id']==identifier and row['method']=='ia_affine_wind'))
        numerical.append(dict(event_id=identifier,refined=is_refined,
            passed=check['passed'],relative_l2=check['relative_l2']))
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method,item in [('ia_affine_repaired',fit),
            ('ia_affine_previous',read(PREVIOUS/identifier/'ia_affine_wind.json')),
            ('original_ia',read(OUT/'fits'/identifier/'ia_vpdi.json')),
            ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]:
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.asarray(item['source_norm'])*100-true_xy)),
                q_kg_h=3600*window_mean_release(item['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'],equipment_group=truth['equipment_group'],
                rmse_ppm=float(np.sqrt(item['mse'])),refined=is_refined))
    frame=pd.DataFrame(rows)
    frame.to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    frame.groupby('method').localization_error_m.agg(['mean','median','count']).to_csv(DEST/'summary.csv',encoding='utf-8-sig')
    wide=frame.pivot(index='event_id',columns='method',values='localization_error_m')
    groups=[read(OUT/'inputs'/identifier/'truth.json')['equipment_group'] for identifier in wide.index]
    comparison=intervals(wide.ia_affine_repaired-wide.frozen_nls,groups,np.random.default_rng(20261028))
    save(DEST/'paired_accuracy.json',dict(comparison=comparison,
        scope='Exploratory development sample, not independent-site confirmation or pure Q-elimination effect'))
    save(DEST/'status.json',dict(complete_events=len(cases),refined_events=failed,
        numerical_rows=numerical,numerically_passed=all(row['passed'] for row in numerical),
        nls_called=False,production_promoted=False,full_goal_achieved=False))
    print('IA_REPAIR_SUMMARY',comparison,flush=True)


if __name__=='__main__':
    run()
