"""Replay, diagnose and summarize the complete 12-case development pilot."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import statistics
import time
import hashlib
import importlib.metadata
import sys
import zipfile

import numpy as np
import torch
from unittest.mock import patch

import run_paper_experiments as experiment
import json


def restore(model, record):
    experiment.ia.set_model_anisotropic_diffusion_norm(model, *record['diffusion_norm'])
    with torch.no_grad():
        model.xs.fill_(record['source_norm'][0])
        model.ys.fill_(record['source_norm'][1])


def layout_audit(model, xyt, observed):
    family = experiment.vp.build_release_layout_family(
        model.recurrent_integration_times, torch.linspace(0, experiment.HOURS, 7))
    measurements = []
    comparisons = None
    for repeat in range(3):
        started = time.perf_counter()
        shared = experiment.averaged_response(model, experiment.SIGMA, xyt, family.response_basis).cpu().numpy()
        shared_solutions = [experiment.vp.solve_nonnegative_q(
            shared @ layout.mapping.cpu().numpy(), observed) for layout in family.layouts]
        shared_seconds = time.perf_counter() - started
        started = time.perf_counter()
        separate = [experiment.averaged_response(model, experiment.SIGMA, xyt, layout.basis).cpu().numpy()
                    for layout in family.layouts]
        separate_solutions = [experiment.vp.solve_nonnegative_q(response, observed) for response in separate]
        separate_seconds = time.perf_counter() - started
        comparisons = []
        for layout, a, b, response in zip(family.layouts, shared_solutions, separate_solutions, separate):
            error = float(np.max(np.abs(shared @ layout.mapping.cpu().numpy() - response)))
            if not a.success or not b.success or error > 2e-5 or not np.allclose(
                    a.prediction, b.prediction, rtol=2e-4, atol=2e-5):
                raise RuntimeError("Shared layout equivalence gate failed")
            comparisons.append({"layout": layout.layout_id, "coefficients": layout.basis.shape[1],
                "shared_sse": a.weighted_sse, "separate_sse": b.weighted_sse,
                "max_response_difference": error})
        measurements.append({"repeat": repeat, "shared_seconds": shared_seconds,
                             "separate_seconds": separate_seconds})
    return {"geometry": "conditioned on fitted uniform IA position and diffusion",
        "layouts": comparisons, "measurements": measurements,
        "shared_basis_response_count": family.response_basis.shape[1],
        "separate_basis_response_count": sum(layout.basis.shape[1] for layout in family.layouts),
        "interpretation": "Fixed-geometry equivalence/cost only; not a full layout-search localization test"}


def analyze(pilot):
    records = []
    for index in range(12):
        case = pilot / experiment.case_definition(index)['case_id']
        if not (case / 'result.json').exists():
            raise ValueError(f"Incomplete pilot: {case.name}; no selective summary is produced")
        records.append(json.loads((case / 'result.json').read_text(encoding='utf-8')))
    analysis = pilot / 'analysis'
    analysis.mkdir(exist_ok=True)
    all_diagnostics, layout_diagnostics, rows = {}, {}, []
    torch.set_num_threads(1)
    with experiment.experiment_context():
        for index, record in enumerate(records):
            print(f"diagnostics {index+1}/12", flush=True)
            model, xyt, basis, a, observed, clean, _, _ = experiment.make_case(index)
            saved = np.load(pilot / record['case']['case_id'] / 'input.npz')
            if not np.array_equal(saved['observed'], observed):
                raise RuntimeError("Regenerated inputs differ from saved experiment inputs")
            fit = record['ia_vpdi']
            restore(model, fit)
            prediction = experiment.averaged_response(model, experiment.SIGMA, xyt, basis).cpu().numpy() @ fit['coefficients']
            exported = np.load(pilot / record['case']['case_id'] / 'predictions.npz')['ia_prediction']
            np.testing.assert_allclose(prediction, exported, rtol=2e-5, atol=1e-6)
            started = time.perf_counter()
            info = experiment.ia.compute_profile_information(model, experiment.SIGMA, xyt,
                observed, basis, fit['coefficients'], length_m=experiment.LENGTH_M,
                source_bounds=(experiment.BOUNDS, experiment.BOUNDS), boundary_margin_norm=0.01)
            diagnostic = asdict(info)
            diagnostic['seconds'] = time.perf_counter() - started
            diagnostic['location_error_m'] = fit['localization_error_m']
            matrices = {'naive': info.full_information[:2, :2],
                        'eliminate_q': info.profiled_theta_information[:2, :2],
                        'eliminate_q_and_d': info.profiled_source_information}
            diagnostic['information_eigenvalues'] = {name: np.linalg.eigvalsh(matrix).tolist()
                                                     for name, matrix in matrices.items()}
            diagnostic['interpretation'] = 'Uncalibrated local diagnostics on development data, not 95% coverage evidence'
            all_diagnostics[record['case']['case_id']] = diagnostic
            if index in (1, 5, 10):
                layout_diagnostics[record['case']['case_id']] = layout_audit(model, xyt, observed)
            for method in ('ia_vpdi', 'matched_joint_adam'):
                item = record[method]
                rows.append({'case': record['case']['case_id'], 'wind': record['case']['wind'],
                    'network': record['case']['network'], 'release': record['case']['release'],
                    'method': method, **{key: item[key] for key in ('localization_error_m',
                    'q_relative_l2', 'concentration_rmse_clean', 'objective_mse_observed',
                    'seconds_including_common_search', 'forward_count', 'at_search_boundary')}})
    experiment.save_json(analysis / 'local_reliability.json', all_diagnostics)
    experiment.save_json(analysis / 'shared_layout_equivalence.json', layout_diagnostics)
    with (analysis / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in ('ia_vpdi', 'matched_joint_adam'):
        selected = [row for row in rows if row['method'] == method]
        summary[method] = {key: statistics.median(row[key] for row in selected) for key in (
            'localization_error_m', 'q_relative_l2', 'concentration_rmse_clean',
            'objective_mse_observed', 'seconds_including_common_search')}
    summary['ia_lower_location_error_count'] = sum(
        r['ia_vpdi']['localization_error_m'] < r['matched_joint_adam']['localization_error_m'] for r in records)
    summary['case_count'] = len(records)
    summary['status'] = 'development pilot only; 4.1-4.3 NOT complete'
    experiment.save_json(analysis / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


def analyze_parcel_study(study):
    import run_paper_parcel_pilot as parcel
    output=study/'analysis'
    output.mkdir(exist_ok=True)
    records=[]
    diagnostics={}
    torch.set_num_threads(1)
    for index in (0,2,10):
        case=f'pilot_{index+1:02d}'
        record=json.loads((study/f'{case}_result.json').read_text(encoding='utf-8'))
        raw=json.loads((parcel.ROOT/'development_12'/case/'input.json').read_text(encoding='utf-8'))
        saved=np.load(parcel.ROOT/'baseline_demo'/f'{case}_input.npz')
        baseline=json.loads((parcel.ROOT/'baseline_demo_convergence'/f'{case}_result.json').read_text())
        if not baseline['converged']:
            raise ValueError('Do not silently treat a budget-limited baseline as converged')
        model,xyt,basis=parcel.configure(raw,saved['stations'])
        dense_times=np.linspace(0.,6.,193)
        true_q=np.interp(dense_times,np.linspace(0.,6.,5),saved['coefficients'])
        for method,fit in (('ia_vpdi',record['profile_best']),('matched_joint_adam',record['joint_best'])):
            source=([fit['source_x'],fit['source_y']] if method=='ia_vpdi' else fit['source'])
            diffusion=([fit['diffusion_parallel_norm'],fit['diffusion_perp_norm']]
                       if method=='ia_vpdi' else fit['diffusion'])
            experiment.ia.set_model_anisotropic_diffusion_norm(model,*diffusion)
            with torch.no_grad():
                model.xs.fill_(source[0]); model.ys.fill_(source[1])
            prediction=parcel.response_mean(model,experiment.SIGMA,xyt,basis).cpu().numpy()@fit['coefficients']
            np.testing.assert_allclose(prediction,fit['prediction'],rtol=2e-5,atol=1e-6)
            metric=next(r for r in record['metrics'] if r['method']==method)
            q=np.interp(dense_times,np.linspace(0.,6.,5),fit['coefficients'])
            records.append({'case':case,'method':method,'localization_error_m':metric['localization_error_m'],
                'q_relative_l2_common193':float(np.linalg.norm(q-true_q)/np.linalg.norm(true_q)),
                'concentration_rmse_clean':metric['concentration_rmse_clean'],
                'objective_mse_observed':metric['objective_mse_observed'],
                'seconds_diagnostic_only':metric['seconds_including_common_search']})
            if method=='ia_vpdi':
                with patch.object(experiment.ia,'build_response_matrix',parcel.response_mean):
                    info=experiment.ia.compute_profile_information(model,experiment.SIGMA,xyt,saved['observed'],
                        basis,fit['coefficients'],length_m=1000.,source_bounds=(experiment.BOUNDS,experiment.BOUNDS))
                diagnostics[case]=asdict(info)
        q=np.interp(dense_times,np.linspace(0.,6.,5),baseline['coefficients'])
        prediction=np.asarray(baseline['prediction'])
        records.append({'case':case,'method':'independent_puff_nnls_powell_demo',
            'localization_error_m':baseline['localization_error_m'],
            'q_relative_l2_common193':float(np.linalg.norm(q-true_q)/np.linalg.norm(true_q)),
            'concentration_rmse_clean':float(np.sqrt(np.mean((prediction-saved['clean'])**2))),
            'objective_mse_observed':baseline['mse'],'seconds_diagnostic_only':baseline['seconds']})
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    experiment.save_json(output/'local_diagnostics.json',diagnostics)
    experiment.save_json(output/'comparison.json',records)
    experiment.save_json(output/'interpretation.json',{
        'data':'same independent analytic-puff observations for all3 methods',
        'q_metric':'recomputed on identical193-point uniform clock, not each backend internal clock',
        'matching':'IA versus joint use exactly the same parcel forward; independent demo uses continuous analytic wind integrals',
        'baseline':'generic adapted Gaussian-puff/NNLS/Powell, not reproduction of named papers',
        'timing':'diagnostic; do not treat concurrent development runs as a controlled speed benchmark',
        'split':'three repeatedly inspected development cases, not independent evaluation',
        'uncertainty':'conditional local diagnostics, not calibrated coverage',
        'state_replay':'both saved fitted states passed response replay on all3 cases'})
    print(json.dumps([{k:r[k] for k in ('case','method','localization_error_m','q_relative_l2_common193')}
                      for r in records],indent=2))


def snapshot_current_code(output):
    if not output.resolve().is_relative_to((experiment.ROOT/'result').resolve()):
        raise ValueError('Snapshot must stay under result/')
    files=list((experiment.ROOT/'pinn_source').rglob('*.py'))
    files+=list((experiment.ROOT/'scripts').glob('*paper*.py'))
    files+=list((experiment.ROOT/'tests').glob('test_paper*.py'))
    files+=[experiment.ROOT/'scripts/inversion_validation_common.py']
    files=sorted(set(files))
    if not all(p.resolve().is_relative_to(experiment.ROOT.resolve()) for p in files):
        raise ValueError('Snapshot source escaped the project')
    record={'stage':'development experiments, formal4.1-4.3 incomplete',
        'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in
            ('numpy','scipy','torch','pandas','matplotlib','PyMuPDF')},
        'files':{p.relative_to(experiment.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,path.relative_to(experiment.ROOT).as_posix())
        archive.writestr('SNAPSHOT_MANIFEST.json',json.dumps(record,indent=2))
    print(f'Saved{len(files)} source files: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=experiment.ROOT / 'result/paper_experiments_20261017/development_12')
    parser.add_argument('--parcel-study',type=Path)
    parser.add_argument('--snapshot',type=Path)
    args=parser.parse_args()
    if args.snapshot:
        snapshot_current_code(args.snapshot)
    elif args.parcel_study:
        analyze_parcel_study(args.parcel_study)
    else:
        analyze(args.pilot)
