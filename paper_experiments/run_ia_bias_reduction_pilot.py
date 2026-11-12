"""Fixed seven-case, IA-only second-order source-bias pilot; no nonlinear refits."""

import gc
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from run_paper_updated_evidence import STUDY,synthetic_setup,prior
from run_metec_comparison import ROOT,OUT,read,save,load_input,JointBounds
from paper_beam_joint_baselines import BeamJointResidual
from paper_location_capacity import capacity_response
from paper_observation_covariance import estimate_background_noise
from ia_bias_reduction import correct_source_estimate,station_ar1_covariance_operator
from metec_point_adapter import xy_from_lon_lat,window_mean_release

DEST=STUDY/'ia_bias_reduction_pilot_20261028'


def iid_noise(residual,jacobian):
    dof=len(residual)-jacobian.shape[1]
    if dof<=0:
        raise ValueError('Residual variance needs positive residual degrees of freedom')
    variance=float(residual@residual/dof)
    return lambda values:variance*values,dict(kind='iid',variance=variance,
        source='Fitted residual SSE divided by N-p; no generator noise truth')


def public_noise(data):
    indexes=np.flatnonzero(np.isfinite(data['concentration'].ravel()))
    stations=data['concentration'].shape[1]
    def noise(residual,jacobian):
        table=np.full(data['concentration'].shape,np.nan)
        table.ravel()[indexes]=residual
        calibration=estimate_background_noise(table)
        factor=np.sqrt(len(residual)/(len(residual)-jacobian.shape[1]))
        sigma=np.array(calibration['station_sigma'])*factor
        operator=station_ar1_covariance_operator(indexes//stations,indexes%stations,
            sigma,calibration['station_rho'])
        calibration.update(kind='independent_station_AR1',station_sigma=sigma,
            source='All event residuals; working covariance, not independent calibration',
            df_scale=factor,station_residual_means=np.nanmean(table,axis=0),
            limitation='Stationarity, zero mean and independence across stations are assumptions; transport bias is not removed')
        return operator,calibration
    return noise


def correction(objective,reference,noise):
    if np.any(np.asarray(reference['active_mask'])!=0):
        return dict(applied=False,reason='Active amplitude constraint: interior expansion is not applicable')
    theta=np.r_[reference['source_norm'],reference['diffusion_norm'][1],
                 reference['diffusion_norm'][0]-reference['diffusion_norm'][1]]
    response,mapping=capacity_response(objective,reference['q_basis_count'])
    coefficients=np.array(reference['native_coefficients'])
    np.testing.assert_allclose(response(theta,False)@coefficients,reference['prediction'],rtol=1e-7,atol=1e-10)
    result=correct_source_estimate(response,objective.observations.cpu().numpy(),theta,coefficients,
        objective.bounds.lower[:4],objective.bounds.upper[:4],float(objective.bounds.upper[4]),
        noise,active_mask=reference['active_mask'],relative_step=1e-4,interior_sigma_margin=3.)
    if result['applied']:
        result['native_coefficients']=result['coefficients'].copy()
        result['coefficients']=mapping@result['coefficients']
        result['source_norm']=result['physical'][:2]
    return result


def assembled(reference,result,length):
    fit={key:reference[key] for key in ['source_norm','coefficients','prediction','q_basis_count','diffusion_norm','sse']}
    if result['applied']:
        fit.update({key:result[key] for key in ['source_norm','coefficients','prediction','sse']})
    fit.update(applied=result['applied'],correction=result,
        source_displacement_m=float(np.linalg.norm(np.array(fit['source_norm'])-reference['source_norm'])*length))
    return fit


def run():
    torch.set_num_threads(1)
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('NLS baseline changed: '+name)
    panel=read(STUDY/'ia_location_capacity_pilot_20261028/protocol.json')
    sources=[Path(__file__),*[ROOT/'paper_experiments'/name for name in
        ['ia_bias_reduction.py','paper_location_capacity.py','paper_observation_covariance.py']]]
    protocol=dict(synthetic=panel['synthetic'],public=panel['public'],diagnostic_public='20220512005',
        selection='Unchanged fixed seven-case development panel; no truth-based eligibility',
        reference='Original mean-wind IA, same forward; isolates post-estimation correction, not wind changes',
        assumptions='Interior full-rank mean; original point and a3-sigma noise neighborhood must remain inside bounds',
        validity='Reject if bias exceeds1 first-order SE, finite-difference change exceeds0.1%, or corrected source leaves original bounds',
        synthetic_noise='Residual SSE/(N-p), iid; no generator truth',
        public_noise='Event-residual station AR1, rho bound0..0.98; working covariance, not independently calibrated',
        action='Subtract source components of second-order bias; D unchanged; Q re-profiled once at fixed source',
        all_observations_retained=True,nls_called=False,nonlinear_refits=False,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve previous bias-correction pilot')
    save(DEST/'protocol.json',protocol)
    for source in sources:
        target=DEST/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source,target)
    rows=[]
    failures=[]
    for entry in panel['synthetic']:
        identifier=entry['case']['case_id']
        old,data,objective,_=synthetic_setup(entry,'cpu')
        reference=read(STUDY/'ia_release_capacity_full_v1'/identifier/'result.json')['methods']['ia_vpdi_release_bic']
        path=DEST/identifier/'result.json'
        if path.exists():
            fit=read(path)
        else:
            begun=time.perf_counter()
            print('BIAS_START',identifier,flush=True)
            try:
                result=correction(objective,reference,iid_noise)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                result=dict(applied=False,analysis_failed=True,reason=str(exc),error_type=type(exc).__name__)
            fit=assembled(reference,result,float(old['metadata']['length_m']))
            fit['incremental_analysis_seconds']=time.perf_counter()-begun
            save(path,fit)
        if fit['correction'].get('analysis_failed'):
            failures.append(identifier)
        metric=prior.formal.metrics(old['metadata'],data,fit['source_norm'],np.array(fit['coefficients']),np.array(fit['prediction']))
        rows.append(dict(dataset='synthetic',case_id=identifier,method='ia_bias_corrected',applied=fit['applied'],
            localization_error_m=metric['localization_error_m'],q_relative_l2=metric['q_relative_l2'],
            displacement_m=fit['source_displacement_m'],reason=fit['correction'].get('reason','applied')))
        for method,item in [('original_ia',reference),('frozen_nls',old['methods']['joint_nls_trf'])]:
            rows.append(dict(dataset='synthetic',case_id=identifier,method=method,
                localization_error_m=item['localization_error_m'],q_relative_l2=item['q_relative_l2']))
        print('BIAS_DONE',identifier,rows[-3],flush=True)
        del objective
        gc.collect()
    for identifier in panel['public']:
        reference=read(OUT/'fits'/identifier/'ia_vpdi.json')
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        path=DEST/identifier/'result.json'
        if path.exists():
            fit=read(path)
        else:
            begun=time.perf_counter()
            print('BIAS_START',identifier,flush=True)
            if np.any(np.array(reference['active_mask'])!=0):
                result=dict(applied=False,reason='Active amplitude constraint: interior expansion is not applicable')
            else:
                data,meta,model,xyt,basis,response,observed=load_input(identifier,'cuda',dict(time=16,release=16,history_minutes=20))
                initial=read(OUT/'fits'/identifier/'initialization.json')
                objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
                try:
                    result=correction(objective,reference,public_noise(data))
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    result=dict(applied=False,analysis_failed=True,reason=str(exc),error_type=type(exc).__name__)
                del objective,response,model
            fit=assembled(reference,result,100.)
            fit['incremental_analysis_seconds']=time.perf_counter()-begun
            save(path,fit)
        if fit['correction'].get('analysis_failed'):
            failures.append(identifier)
        # Truth is attached after eligibility and the deterministic correction.
        truth=read(OUT/'inputs'/identifier/'truth.json')
        xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method,item in [('ia_bias_corrected',fit),('original_ia',reference),
            ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]:
            rows.append(dict(dataset='public',case_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(item['source_norm'])*100-xy)),
                q_kg_h=3600*window_mean_release(item['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                applied=item.get('applied'),displacement_m=item.get('source_displacement_m'),
                reason=item.get('correction',{}).get('reason','applied') if method=='ia_bias_corrected' else '',
                diagnostic_only=identifier==protocol['diagnostic_public']))
        print('BIAS_DONE',identifier,rows[-3],flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(pilot_complete=not bool(failures),analysis_failures=failures,
        nls_called=False,nonlinear_refits=False,production_promoted=False,full_goal_achieved=False))


if __name__=='__main__':
    run()
