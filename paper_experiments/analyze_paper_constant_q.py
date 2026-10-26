"""Audit the entire matched constant-Q cohort, including sampling failures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_paper_constant_q as study_code
from paper_probabilistic_demo import ConstantPuffPlan,rank_split_rhat


def analyze(study):
    study=Path(study).resolve()
    protocol=json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    for name,digest in protocol['input_sha256'].items():
        if hashlib.sha256((study_code.base.ROOT/name.replace('\\','/')).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen input changed: '+name)
    rows,statuses=[],[]
    ids=protocol['case_ids'][:1] if protocol['smoke'] else protocol['case_ids']
    for case in ids:
        path=study/case/'result.json'
        if not path.exists():
            statuses.append(dict(case_id=case,status='pending_or_interrupted'))
            continue
        value=json.loads(path.read_text(encoding='utf-8'))
        source=study_code.ROOT/'evaluation_v2_boundary_fix'/case
        data=np.load(source/'input.npz')
        metadata=json.loads((source/'input.json').read_text(encoding='utf-8'))
        samples=np.load(study/case/'posterior_samples.npz')
        theta=samples['theta'];q=samples['q'];qmean=samples['conditional_q_mean'];logp=samples['log_prob']
        if (theta.shape[0]!=4 or theta.shape[2:]!=(16,4) or q.shape!=theta.shape[:3]
                or qmean.shape!=q.shape or logp.shape!=q.shape
                or not all(np.isfinite(v).all() for v in (theta,q,qmean,logp)) or np.any(q<0)
                or np.any(np.abs(theta[:,:,:,:2])>.75)):
            raise ValueError('Invalid saved Bayesian samples')
        dq=.00100005+np.exp(theta[:,:,:,2]);dp=dq+np.exp(theta[:,:,:,3])
        independent=np.stack((theta[:,:,0,0],theta[:,:,0,1],dp[:,:,0],dq[:,:,0],qmean[:,:,0]),axis=-1)
        np.testing.assert_allclose(rank_split_rhat(independent),
            value['methods']['bayesian_constant']['diagnostic']['rhat'],rtol=1e-12,atol=1e-12)
        fine=ConstantPuffPlan(data['stations'],metadata['wind'],metadata['sigma'],samples=32,order=64)
        statuses.append(dict(case_id=case,status='completed',
            bayesian_diagnostics_passed=value['methods']['bayesian_constant']['diagnostic']['passed']))
        for name,method in value['methods'].items():
            if not np.isfinite(method['prediction']).all():
                raise ValueError('Nonfinite point prediction')
            parallel,perp=method['diffusion_norm']
            if not parallel>=perp>.00100005:
                raise ValueError('Invalid point diffusivity constraints')
            coordinate=np.r_[method['source_norm'],np.log(perp-.00100005),np.log(max(parallel-perp,np.finfo(float).tiny))]
            replay=fine.response(coordinate)*method['q']
            relative=float(np.linalg.norm(replay-method['prediction'])/max(np.linalg.norm(replay),np.finfo(float).tiny))
            row=dict(case_id=case,network=value['network'],wind=value['wind'],method=name,
                localization_error_m=method['localization_error_m'],q_relative_error=method['q_relative_error'],
                observed_mse=method['mse'],seconds=method['seconds'],convergence=method['convergence'],
                numerical_replay_relative_difference=relative,numerical_replay_passed=relative<=.01)
            if name=='bayesian_constant':
                diagnostic=method['diagnostic']
                intervals=np.asarray(method['coordinate_marginal_95_intervals'])
                truth=data['source']
                # Diagnose plug-in means without replacing the prescribed estimator.
                index=np.unravel_index(np.argmax(logp),logp.shape)
                response=ConstantPuffPlan(data['stations'],metadata['wind'],metadata['sigma'])
                representative=response.response(theta[index])*qmean[index]
                row.update(diagnostics_passed=diagnostic['passed'],rhat_max=max(diagnostic['rhat']),
                    ess_min=min(diagnostic['ess_approx']),
                    tau_max=float(np.max(diagnostic['tau_by_ensemble'])),
                    length_failed_ensemble_count=sum(not v for v in diagnostic['length_at_least_50tau_by_ensemble']),
                    diagnostics_failure_reason='; '.join(reason for reason,failed in (
                        ('rank/folded Rhat >1.01',max(diagnostic['rhat'])>1.01),
                        ('chain shorter than50 autocorrelation times',not all(diagnostic['length_at_least_50tau_by_ensemble'])),
                        ('approximate ESS <400',min(diagnostic['ess_approx'])<400)) if failed),
                    draws_per_walker=theta.shape[1],
                    highest_retained_collapsed_posterior_sample_mse=float(np.mean((representative-data['observed'])**2)),
                    x_marginal_interval_contains_truth=bool(intervals[0,0]<=truth[0]<=intervals[1,0]),
                    y_marginal_interval_contains_truth=bool(intervals[0,1]<=truth[1]<=intervals[1,1]))
            rows.append(row)
    output=study/'analysis'
    output.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'case_metrics.csv',index=False)
    pd.DataFrame(statuses).to_csv(output/'case_status.csv',index=False)
    completed=sum(v['status']=='completed' for v in statuses)
    final=completed==len(ids) and not protocol['smoke']
    study_code.base.save_json(output/'analysis_status.json',dict(final=final,planned_cases=len(ids),
        completed_cases=completed,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        warning='Cohort complete is NOT all samplers converged or all numerical checks passed; do not discard failures',
        bayesian_diagnostics_pass_count=sum(v.get('bayesian_diagnostics_passed',False) for v in statuses)))
    if final:
        frame.groupby(['network','method'],as_index=False).agg(n=('case_id','count'),
            localization_median_m=('localization_error_m','median'),
            q_relative_error_median=('q_relative_error','median'),
            reported_solver_failure_count=('convergence',lambda v:int(v.eq('reported_nonconverged').sum())),
            fixed_budget_convergence_unverified_count=('convergence',lambda v:int(v.eq('fixed budget unverified').sum())),
            sampling_diagnostic_failure_count=('diagnostics_passed',lambda v:int(v.eq(False).sum()) if v.notna().any() else np.nan),
            numerical_pass_count=('numerical_replay_passed','sum')).to_csv(output/'summary.csv',index=False)
    return frame


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args()
    frame=analyze(args.study)
    print('Analyzed method-case records:',len(frame))
