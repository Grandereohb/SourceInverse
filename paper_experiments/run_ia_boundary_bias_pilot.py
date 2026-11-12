"""IA-only finite-box bias pilot on the original fixed seven-case panel."""

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
from run_ia_bias_reduction_pilot import iid_noise,public_noise,assembled
from ia_boundary_bias import correct_boundary_source
from metec_point_adapter import xy_from_lon_lat,window_mean_release

DEST=STUDY/'ia_boundary_bias_pilot_20261028'


def analyze(objective,reference,noise):
    theta=np.r_[reference['source_norm'],reference['diffusion_norm'][1],
                 reference['diffusion_norm'][0]-reference['diffusion_norm'][1]]
    response,mapping=capacity_response(objective,reference['q_basis_count'])
    coefficients=np.array(reference['native_coefficients'])
    np.testing.assert_allclose(response(theta,False)@coefficients,reference['prediction'],rtol=1e-7,atol=1e-10)
    result=correct_boundary_source(response,objective.observations.cpu().numpy(),theta,coefficients,
        objective.bounds.lower[:4],objective.bounds.upper[:4],float(objective.bounds.upper[4]),
        noise,samples=4096,seed=20261028)
    if result['applied']:
        result['native_coefficients']=result['coefficients'].copy()
        result['coefficients']=mapping@result['coefficients']
    return result


def run():
    torch.set_num_threads(1)
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('Frozen NLS changed: '+name)
    panel=read(STUDY/'ia_location_capacity_pilot_20261028/protocol.json')
    sources=[Path(__file__),*[ROOT/'paper_experiments'/name for name in
        ['ia_boundary_bias.py','ia_bias_reduction.py','run_ia_bias_reduction_pilot.py',
         'paper_location_capacity.py','paper_observation_covariance.py']]]
    protocol=dict(synthetic=panel['synthetic'],public=panel['public'],diagnostic_public='20220512005',
        method='Full-mean linearized Gaussian displacement, metric projection onto original parameter box',
        reference='Original mean-wind IA; no nonlinear refitting, wind changes or new losses',
        integration=dict(samples=4096,seed=20261028,antithetic=True,nested_relative_tolerance=.05),
        validity='Full-rank local mean; same physical/Q bounds; exact corrected-mean remainder <=10%; no clipping source',
        noise='Same working iid/residual-AR1 models as prior interior-bias pilot',
        action='Subtract only source components of local projected displacement; D unchanged; Q re-profiled once',
        limitations=['Finite-box plug-in, not a consistently calibrated boundary bootstrap',
                    'Local linearization does not repair transport mismatch or guarantee error reduction'],
        all_cases_retained=True,nls_called=False,nonlinear_refits=False,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve old boundary pilot protocol/results')
    save(DEST/'protocol.json',protocol)
    for source in sources:
        target=DEST/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source,target)
    rows=[]
    failures=[]
    for kind,entry in [('synthetic',v) for v in panel['synthetic']]+[('public',v) for v in panel['public']]:
        if kind=='synthetic':
            identifier=entry['case']['case_id']
            old,data,objective,_=synthetic_setup(entry,'cpu')
            reference=read(STUDY/'ia_release_capacity_full_v1'/identifier/'result.json')['methods']['ia_vpdi_release_bic']
            noise=iid_noise
            length=float(old['metadata']['length_m'])
        else:
            identifier=entry
            data,meta,model,xyt,basis,response,observed=load_input(identifier,'cuda',dict(time=16,release=16,history_minutes=20))
            initial=read(OUT/'fits'/identifier/'initialization.json')
            objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
            reference=read(OUT/'fits'/identifier/'ia_vpdi.json')
            noise=public_noise(data)
            length=100.
        output=DEST/identifier/'result.json'
        if output.exists():
            fit=read(output)
        else:
            print('BOUNDARY_START',identifier,flush=True)
            begun=time.perf_counter()
            try:
                result=analyze(objective,reference,noise)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                result=dict(applied=False,analysis_failed=True,reason=str(exc),error_type=type(exc).__name__)
            fit=assembled(reference,result,length)
            fit['incremental_analysis_seconds']=time.perf_counter()-begun
            save(output,fit)
        if fit['correction'].get('analysis_failed'):
            failures.append(identifier)
        if kind=='synthetic':
            metric=prior.formal.metrics(old['metadata'],data,fit['source_norm'],np.array(fit['coefficients']),np.array(fit['prediction']))
            rows.append(dict(dataset=kind,case_id=identifier,method='ia_boundary_bias',applied=fit['applied'],
                localization_error_m=metric['localization_error_m'],q_relative_l2=metric['q_relative_l2'],
                displacement_m=fit['source_displacement_m'],reason=fit['correction'].get('reason','applied')))
            for method,item in [('original_ia',reference),('frozen_nls',old['methods']['joint_nls_trf'])]:
                rows.append(dict(dataset=kind,case_id=identifier,method=method,
                    localization_error_m=item['localization_error_m'],q_relative_l2=item['q_relative_l2']))
        else:
            truth=read(OUT/'inputs'/identifier/'truth.json')
            xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
            for method,item in [('ia_boundary_bias',fit),('original_ia',reference),
                               ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]:
                rows.append(dict(dataset=kind,case_id=identifier,method=method,
                    localization_error_m=float(np.linalg.norm(np.array(item['source_norm'])*100-xy)),
                    q_kg_h=3600*window_mean_release(item['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                    applied=item.get('applied'),displacement_m=item.get('source_displacement_m'),
                    reason=item.get('correction',{}).get('reason','applied') if method=='ia_boundary_bias' else '',
                    diagnostic_only=identifier==protocol['diagnostic_public']))
        print('BOUNDARY_DONE',identifier,rows[-3],flush=True)
        pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
        del objective
        if kind=='public':
            del response,model
        gc.collect()
        torch.cuda.empty_cache()
    save(DEST/'status.json',dict(pilot_complete=not bool(failures),analysis_failures=failures,
        nls_called=False,nonlinear_refits=False,production_promoted=False,full_goal_achieved=False))


if __name__=='__main__':
    run()
