"""Frozen full17-event affine-wind evaluation, serial with per-event checkpoints."""

import gc
import hashlib
from pathlib import Path
import time
import shutil

import numpy as np
import pandas as pd
import torch

from run_metec_comparison import ROOT,OUT,AUDIT,read,save,JointBounds
from run_ia_affine_wind_pilot import DEST as PILOT,calibration
from ia_metec_point_adapter import configure,xy_from_lon_lat,window_mean_release
from paper_beam_joint_baselines import BeamJointResidual
from paper_joint_baselines import fit_joint_nls
from paper_q_model_selection import fit_profiled_q_models
from paper_sqrt_affine_wind import SquareRootAffineWindHistory
from paper_sqrt_gaussian import combine_diffusion_factors

LEGACY=OUT.parent/'ia_affine_wind_full_20261028'
DEST=OUT.parent/'ia_affine_wind_sqrt_full_20261028'
METHODS=['ia_affine_wind','joint_nls_affine_wind']


def factor_preflight(plan):
    valid=plan.weighted_basis.abs().sum(-1)>0
    lp,lq=plan.parallel_factors[valid],plan.perpendicular_factors[valid]
    largest=0.
    for perp,contrast in [(.1,0.),(.1,60.),(60.,0.),(60.,60.)]:
        a,c,d=combine_diffusion_factors(lp,lq,lp.new_tensor(perp+contrast),lp.new_tensor(perp))
        if not bool(torch.all((a>0)&(d>0)&torch.isfinite(a)&torch.isfinite(c)&torch.isfinite(d))):
            raise FloatingPointError('Square-root covariance is not finite and nonsingular')
        trace=a*a+c*c+d*d
        largest=max(largest,float((2*(torch.log(trace)-torch.log(a)-torch.log(d))).max()))
    return dict(all_domain_corners_positive=True,max_log_trace_squared_over_determinant=largest)


def run():
    torch.set_num_threads(1)
    parent=read(OUT/'protocol.json')
    pilot=read(PILOT/'protocol.json')
    for name,digest in parent['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Original public data changed: '+name)
    files=[Path(__file__),*[ROOT/'paper_experiments'/name for name in ['paper_affine_wind.py',
        'paper_sqrt_affine_wind.py','paper_sqrt_gaussian.py','metec_point_adapter.py',
        'paper_physical_beam_adapter.py','run_ia_affine_wind_pilot.py','paper_q_model_selection.py',
        'paper_reduced_solver.py','paper_joint_baselines.py','paper_beam_joint_baselines.py']],
        ROOT/'pinn_source/reduced_trust_region.py']
    protocol=dict(version='full17_measured_affine_wind_square_root_v2',cases=parent['event_ids'],methods=METHODS,
        initial='Unchanged original common starts and bounds, including original amplitude cap limitations',
        q='IA BIC1/3/5; matched NLS fixed5; original unweighted residual and all finite observations',
        orders=parent['numerical_orders'],max_nfev=250,
        failure='Keep explicit failed events; no clipping, source-region masks or uniform-wind fallback',
        covariance='Exact local closed-form factors and QR accumulation; analytic Givens kernel',
        reuse='Both predecessor predictions must agree under new response within relative L2 of1e-6; otherwise refit both methods',
        scope='Development full-cohort evaluation after two-event pilot; not independent validation',
        pilot_protocol_sha256=hashlib.sha256((PILOT/'protocol.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Changed full-cohort protocol; preserve the earlier version')
    save(DEST/'protocol.json',protocol)
    for source in files:
        target=DEST/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source,target)
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    sensors=pd.read_csv(AUDIT/'raw/input_data/sensor_locations.csv').set_index('name')
    for index,identifier in enumerate(protocol['cases']):
        folder=DEST/identifier
        if (folder/'complete.json').exists() or (folder/'failed.json').exists():
            continue
        print('AFFINE_FULL_START',index+1,'of',len(protocol['cases']),identifier,flush=True)
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        history,audit=calibration(data,meta,raw,sensors)
        history=SquareRootAffineWindHistory(history.edges,history.velocity,history.gradient,data['wind_xy'])
        save(folder/'wind_calibration.json',audit)
        orders=protocol['orders']
        model=response=objective=None
        try:
            begun=time.perf_counter()
            model,xyt,basis,response,observed=configure(data,meta['source_height_m'],device='cuda',
                time_order=orders['time'],release_order=orders['release'],history_minutes=orders['history_minutes'],
                transport_history=history)
            check=factor_preflight(response.plan)
            check['plan_seconds']=time.perf_counter()-begun
            save(folder/'numerical_preflight.json',check)
            initial=read(OUT/'fits'/identifier/'initialization.json')
            bounds=JointBounds(**initial['bounds'])
            starts=[np.array(s) for s in initial['starts']]
            predecessor=(LEGACY/identifier if (LEGACY/identifier/'complete.json').exists() else
                PILOT/identifier if identifier in pilot['cases'] else None)
            if predecessor is not None:
                verified={}
                for method in METHODS:
                    prior=read(predecessor/(method+'.json'))
                    dp,dq=prior['diffusion_norm']
                    theta=model.xs.new_tensor([*prior['source_norm'],dq,dp-dq])
                    prediction=(response(model,0.,xyt,basis,physical_parameters=theta)@
                        model.xs.new_tensor(prior['coefficients'])).cpu().numpy()
                    delta=float(np.linalg.norm(prediction-np.array(prior['prediction']))/
                        max(np.linalg.norm(prediction),1e-30))
                    verified[method]=dict(relative_l2=delta,passed=bool(np.isfinite(delta) and delta<=1e-6),
                        predecessor=str(predecessor.relative_to(ROOT)),
                        source_sha256=hashlib.sha256((predecessor/(method+'.json')).read_bytes()).hexdigest())
                save(folder/'reuse_verification.json',verified)
                if all(item['passed'] for item in verified.values()):
                    for method in METHODS:
                        save(folder/(method+'.json'),read(predecessor/(method+'.json')))
                    save(folder/'complete.json',dict(complete=True,reused_verified_equivalence=True))
                    print('AFFINE_FULL_VERIFIED_REUSE',index+1,identifier,flush=True)
                    continue
            for method in METHODS:
                output=folder/(method+'.json')
                if output.exists():
                    continue
                objective=BeamJointResidual(response,model,0.,xyt,basis,observed,bounds)
                fit=(fit_profiled_q_models(objective,starts,max_nfev=protocol['max_nfev'])
                    if method.startswith('ia_') else fit_joint_nls(objective,starts,protocol['max_nfev']))
                save(output,fit)
                print('AFFINE_FULL_FIT',identifier,method,round(fit['seconds'],1),flush=True)
            save(folder/'complete.json',dict(complete=True,reused_identical_pilot=False))
        except (FloatingPointError,np.linalg.LinAlgError) as exc:
            save(folder/'failed.json',dict(type=type(exc).__name__,message=str(exc),
                complete=False,no_clipping_or_data_deletion=True))
            print('AFFINE_FULL_FAILED',identifier,str(exc),flush=True)
        finally:
            del model,response,objective
            gc.collect()
            torch.cuda.empty_cache()
    # Evaluate only after attempting every event; failures stay explicitly listed.
    rows=[]
    failed=[]
    for identifier in protocol['cases']:
        folder=DEST/identifier
        if (folder/'failed.json').exists():
            failed.append(identifier)
            continue
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method in [*METHODS,'ia_vpdi','joint_nls_trf']:
            path=folder/(method+'.json') if method in METHODS else OUT/'fits'/identifier/(method+'.json')
            fit=read(path)
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.asarray(fit['source_norm'])*100-true_xy)),
                q_kg_h=3600*window_mean_release(fit['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'],rmse_ppm=float(np.sqrt(fit['mse'])),
                q_basis_count=fit.get('q_basis_count',5),converged=fit.get('converged')))
    frame=pd.DataFrame(rows)
    frame.to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    frame.groupby('method').localization_error_m.agg(['median','mean','count']).to_csv(DEST/'summary.csv',encoding='utf-8-sig')
    save(DEST/'status.json',dict(all_events_attempted=True,failed_events=failed,
        complete_events=len(protocol['cases'])-len(failed),expected_events=len(protocol['cases']),
        replay_pending=True,ready=False,full_goal_achieved=False))


if __name__=='__main__':
    raise SystemExit('Archived matched-NLS control. NLS is frozen; use IA-only experiments.')
