"""Post-fit truth evaluation; never called by the fitting objective."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_paper_controlled_release as physical


def replay_failed_precision(study):
    """Diagnose reference accuracy without changing the original fit or gate."""
    study=Path(study).resolve()
    torch.set_num_threads(1)
    checks=[]
    for path in sorted(study.glob('*/height_*/result.json')):
        result=json.loads(path.read_text(encoding='utf-8'))
        for name,method in result['methods'].items():
            if not name.startswith('transient_') or method['replay_gate_passed']:
                continue
            height=result['height_m']
            model,xyt,basis,response,_,_=physical.configure(height,
                device='cpu',dtype=torch.float64,time_order=64,release_order=64,
                beam_solver='analytic',data_dir=physical.ROOT/'controlled_release_preparation_v2'/path.parent.parent.name)
            best=method['best']
            with physical.base.ia.temporary_source_and_diffusion(model,best['source_x'],best['source_y'],
                    best['diffusion_parallel_norm'],best['diffusion_perp_norm']):
                matrix=response(model,0.,xyt,basis).numpy()
            if name=='transient_constant_q':
                matrix=matrix.sum(1,keepdims=True)
            prediction=matrix@np.asarray(best['coefficients'])
            previous=pd.read_csv(path.parent/(name+'_predictions.csv')).fine_replay_ppm.to_numpy()
            relative=float(np.linalg.norm(prediction-previous)/max(np.linalg.norm(prediction),np.finfo(float).tiny))
            checks.append(dict(event=path.parent.parent.name,height_m=height,method=name,
                original_16_vs_32=method['replay_relative_difference'],reference_32_vs_64=relative,
                reference_check_passed=relative<=.01,
                interpretation='Fixed source,D,Q coefficients only; does NOT repair or replace original16-order optimization/gate'))
    physical.base.save_json(study/'analysis/supplemental_numerical_checks.json',dict(checks=checks,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return checks


def analyze(study):
    study = Path(study).resolve()
    protocol = json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    root = physical.ROOT
    for name, expected in protocol['sha256'].items():
        if '/controlled_release_preparation_v2/' in name.replace('\\','/'):
            if hashlib.sha256((physical.base.ROOT/name.replace('\\','/')).read_bytes()).hexdigest() != expected:
                raise ValueError('Prepared input changed after protocol freeze: '+name)
    truth_path = root/'public_data/author_truth_values.json'
    truth = json.loads(truth_path.read_text(encoding='utf-8'))['values_as_stored']
    rows, files, pending = [], [truth_path], []
    for event in protocol['events']:
        key = event['source'].replace('_', '')+'_'+event['release']
        reference_path = root/'controlled_release_preparation_v2'/key/'cylinder_reference_v2.json'
        files.append(reference_path)
        reference = json.loads(reference_path.read_text(encoding='utf-8'))
        q_true = reference.get('ols_emission_reference_kg_h')
        source = np.asarray(truth[event['source']+'_location'][:2])
        for height in protocol['heights_m']:
            result_path = study/key/f'height_{height:.1f}'/'result.json'
            if not result_path.exists():
                pending.append(dict(event=key, height_m=height))
                continue
            files.append(result_path)
            result = json.loads(result_path.read_text(encoding='utf-8'))
            for name, method in result['methods'].items():
                stationary = name == 'stationary_constant_q'
                predicted_source = np.asarray(method['source_norm'])*100. if stationary else np.asarray(method['source_m'])
                q_fit = method['q_kg_s']*3600. if stationary else method['mean_q_kg_s']*3600.
                mse = method['observed_mse'] if stationary else method['best']['profile_score']
                passed = method['quadrature_gate_passed'] if stationary else method['replay_gate_passed']
                rows.append(dict(event=key, source_group=event['source'], height_m=height, method=name,
                    source_x_m=predicted_source[0], source_y_m=predicted_source[1],
                    truth_x_m=source[0], truth_y_m=source[1],
                    localization_error_m=float(np.linalg.norm(predicted_source-source)),
                    mean_q_kg_h=q_fit, reference_q_kg_h=q_true,
                    q_reference_status=reference['event_rate_reference_status'],
                    q_relative_error=None if q_true is None else abs(q_fit-q_true)/q_true,
                    observed_rmse_ppm=float(np.sqrt(mse)), quadrature_gate_passed=passed,
                    diffusion_parallel_m2_s=method['diffusion_m2_s'][0],
                    diffusion_perp_m2_s=method['diffusion_m2_s'][1],
                    optimization_status=('reported_converged' if method['converged'] else 'reported_nonconverged')
                        if stationary else 'fixed_budget_convergence_unverified',
                    method_seconds=method['seconds'] if stationary else method['refinement_and_replay_seconds'],
                    shared_coarse_seconds=result['shared_coarse_seconds']))
    output = study/'analysis'
    output.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'case_metrics.csv', index=False)
    if not frame.empty:
        summary = frame.groupby(['height_m','method'],as_index=False).agg(
            events=('event','nunique'),localization_median_m=('localization_error_m','median'),
            localization_max_m=('localization_error_m','max'),
            measured_q_events=('reference_q_kg_h','count'),
            q_relative_error_median=('q_relative_error','median'),
            rmse_median_ppm=('observed_rmse_ppm','median'),
            numerical_gate_pass_count=('quadrature_gate_passed','sum'))
        summary.to_csv(output/'summary.csv',index=False)
    physical.base.save_json(output/'analysis_status.json', dict(planned_events=len(protocol['events']),
        completed_height_jobs=len(rows)//3, planned_height_jobs=len(protocol['events'])*2,
        final=not pending and protocol['scope']=='controlled-release fixed-budget evaluation',
        pending=pending, scope=protocol['scope'], independent_source_positions=2,
        caution='Events share two physical source positions and one beam network; not independent geometries',
        truth_access='post-fit evaluation only; author aggregate emission rates not substituted for missing cylinder data',
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sha256={str(p.relative_to(physical.base.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--replay-failed-precision',action='store_true')
    args = parser.parse_args()
    print(analyze(args.study).to_string(index=False))
    if args.replay_failed_precision:
        print(replay_failed_precision(args.study))
