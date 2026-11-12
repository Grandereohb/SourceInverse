"""Two-case IA-only likelihood smoke study; NLS is read, never executed."""

import hashlib
import time
import numpy as np
import pandas as pd
import torch

from run_paper_updated_evidence import STUDY,synthetic_setup,prior
from run_metec_comparison import ROOT,OUT,read,save,load_input,JointBounds
from paper_beam_joint_baselines import BeamJointResidual
from paper_location_capacity import capacity_response
from ia_robust_profile import fit_robust_profile
from metec_point_adapter import xy_from_lon_lat

PREVIOUS=STUDY/'ia_location_capacity_pilot_20261028'
DEST=STUDY/'ia_robust_profile_smoke_20261028'


def fit_case(objective,starts,reference):
    count=reference['q_basis_count']
    response,mapping=capacity_response(objective,count)
    theta=np.r_[reference['source_norm'],reference['diffusion_norm'][1],
                reference['diffusion_norm'][0]-reference['diffusion_norm'][1]]
    observed=objective.observations.detach().cpu().numpy()
    initial_prediction=response(theta,False)@np.asarray(reference['native_coefficients'])
    residual=initial_prediction-observed
    scale=1.4826*np.median(np.abs(residual-np.median(residual)))
    scale*=np.sqrt(len(residual)/max(len(residual)-(4+count),1))
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Cannot estimate a fixed positive residual scale')
    begun=time.perf_counter()
    fit=fit_robust_profile(response,observed,
        [np.asarray(start)[:4] for start in starts]+[theta],
        objective.bounds.lower[:4],objective.bounds.upper[:4],
        objective.bounds.upper[4],scale,maxiter=250)
    physical=fit.pop('theta')
    native=fit['coefficients'].copy()
    fit.update(source_norm=physical[:2],diffusion_norm=[physical[2]+physical[3],physical[2]],
        coefficients=mapping@native,native_coefficients=native,q_basis_count=count,
        sse=float(np.sum((fit['prediction']-observed)**2)),
        baseline_sse=float(np.sum(residual**2)),seconds=time.perf_counter()-begun,
        peak_ratio=float(np.max(fit['prediction'])/max(np.max(observed),1e-30)),
        baseline_peak_ratio=float(np.max(initial_prediction)/max(np.max(observed),1e-30)))
    return fit


def run():
    torch.set_num_threads(1)
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('NLS baseline changed: '+name)
    previous=read(PREVIOUS/'protocol.json')
    entry=previous['synthetic'][0]
    identifier=previous['public'][0]
    files=[ROOT/'paper_experiments'/name for name in
        ['ia_robust_profile.py','run_ia_robust_profile_pilot.py','paper_location_capacity.py']]
    protocol=dict(cases=[entry['case']['case_id'],identifier],
        selection='First synthetic and first public event in the prior fixed panel, before new fitting',
        observations='All observations; Huber replaces SSE for geometry, no additional penalty',
        delta=1.345,scale='Fixed baseline residual MAD with residual-df adjustment',
        q_capacity='Same selected baseline Q capacity; no truth-based tuning',
        final_amplitude='Bounded least squares at fixed robust geometry',
        starts='Original common starts plus existing IA geometry',maxiter=250,
        nls='Historical files only; no new calls',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve the existing protocol')
    save(DEST/'protocol.json',protocol)
    rows=[]
    old,data,objective,starts=synthetic_setup(entry,'cpu')
    case=entry['case']['case_id']
    output=DEST/case/'result.json'
    if output.exists():
        fit=read(output)
    else:
        cached=read(PREVIOUS/'synthetic'/case/'result.json')
        selected=next(m for m in cached['release_models'] if m['q_basis_count']==cached['q_basis_count'])
        print('ROBUST_START',case,flush=True)
        fit=fit_case(objective,starts,selected)
        save(output,fit)
    metric=prior.formal.metrics(old['metadata'],data,fit['source_norm'],
        np.array(fit['coefficients']),np.array(fit['prediction']))
    rows.append(dict(case_id=case,method='ia_robust',localization_error_m=metric['localization_error_m'],
        peak_ratio=fit['peak_ratio'],baseline_peak_ratio=fit['baseline_peak_ratio'],
        sse=fit['sse'],baseline_sse=fit['baseline_sse'],converged=fit['converged']))
    original=read(STUDY/'ia_release_capacity_full_v1'/case/'result.json')['methods']['ia_vpdi_release_bic']
    for method,item in [('original_ia',original),('frozen_nls',old['methods']['joint_nls_trf'])]:
        rows.append(dict(case_id=case,method=method,localization_error_m=item['localization_error_m']))
    print('ROBUST_DONE',case,rows[0],flush=True)
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    data,meta,model,xyt,basis,response,observed=load_input(identifier,'cuda',dict(time=16,release=16,history_minutes=20))
    initial=read(OUT/'fits'/identifier/'initialization.json')
    objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
    output=DEST/identifier/'result.json'
    if output.exists():
        fit=read(output)
    else:
        cached=read(PREVIOUS/'public'/identifier/'result.json')
        selected=next(m for m in cached['release_models'] if m['q_basis_count']==cached['q_basis_count'])
        print('ROBUST_START',identifier,flush=True)
        fit=fit_case(objective,[np.array(s) for s in initial['starts']],selected)
        save(output,fit)
    truth=read(OUT/'inputs'/identifier/'truth.json')
    true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
    for method,item in [('ia_robust',fit),('original_ia',read(OUT/'fits'/identifier/'ia_vpdi.json')),
                        ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]:
        rows.append(dict(case_id=identifier,method=method,
            localization_error_m=float(np.linalg.norm(np.array(item['source_norm'])*100-true_xy)),
            peak_ratio=item.get('peak_ratio'),baseline_peak_ratio=item.get('baseline_peak_ratio'),
            sse=item.get('sse'),baseline_sse=item.get('baseline_sse'),converged=item.get('converged')))
    print('ROBUST_DONE',identifier,rows[-3],flush=True)
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(smoke_complete=True,nls_called=False,production_promoted=False,
        full_goal_achieved=False,note='Development smoke only, not full-sample or independent evidence'))


if __name__=='__main__':
    run()
