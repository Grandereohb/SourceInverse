"""Fixed-truth numerical checks before independent paper evaluation.

Compare the same physical source, wind and Q, without refitting or re-scaling
the predictions. Numerical tolerances are development gates, not statistical
confidence statements or external-model validation.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_paper_experiments as experiment

CASES = (0, 2, 10)  # constant, double peak, pulse; selected before calculation
SETTINGS = ((36, 8, 4), (36, 8, 8), (36, 16, 8), (36, 32, 16),
            (48, 16, 8), (64, 16, 8), (96, 16, 8), (96, 32, 16),
            (128, 32, 16), (128, 64, 32))
GATE_RELATIVE_L2 = 0.05


def fixed_truth_response(record, stations, grid, steps_per_hour, samples_per_hour,
                         solver='gaussian', parcel_subdivisions=8):
    if grid < 8 or steps_per_hour < 1 or samples_per_hour < 1:
        raise ValueError('Positive clock and sample rates and grid>=8 required')
    clock = np.linspace(0.0, experiment.HOURS, experiment.HOURS * steps_per_hour + 1)
    angle = (np.full_like(clock, 0.05) if record['wind'] == 'steady' else
             -0.35 + 0.9 * clock / experiment.HOURS)
    model = experiment.PINN(hidden=8, depth=1)
    experiment.field.configure_recurrent_context(model, -2.0, 4.0, -3.0, 3.0,
        clock, 1.2*np.cos(angle), 1.2*np.sin(angle), 0.0, 0.05,
        anisotropic_diffusion=True, d_parallel_min_norm=0.001,
        d_perp_min_norm=0.001, decay_norm=0.1, initial_release_dt=0.0,
        nx=grid, ny=grid, source_mode='gaussian', source_sigma_norm=experiment.SIGMA,
        diffusion_solver='gaussian' if solver == 'gaussian_parcels' else solver,
        source_decay_integrator='exponential_linear', transport_solver='grid')
    with torch.no_grad():
        model.xs.fill_(record['source_norm'][0])
        model.ys.fill_(record['source_norm'][1])
    experiment.ia.set_model_anisotropic_diffusion_norm(model, 0.08, 0.03)
    stations = torch.as_tensor(stations, dtype=torch.float32)
    times = torch.tensor((np.arange(experiment.HOURS*samples_per_hour)+0.5)/samples_per_hour,
                         dtype=torch.float32)
    xyt = torch.cat((stations.repeat(times.numel(), 1),
                     times.repeat_interleave(len(stations))[:, None]), dim=1)
    basis = experiment.vp.build_q_basis(model.recurrent_integration_times, 5)
    if solver == 'gaussian_parcels':
        from gaussian_parcels import gaussian_parcel_response
        with torch.no_grad():
            response = gaussian_parcel_response(model, experiment.SIGMA, xyt,
                model.recurrent_integration_times, basis, subdivisions=parcel_subdivisions,
                initial_release_fraction=0.0)
    else:
        response = experiment.vp.build_response_matrix(model, experiment.SIGMA, xyt, basis)
    mean = experiment.hourly_mean_response(response, samples=samples_per_hour)
    prediction = mean.cpu().numpy() @ np.asarray(record['coefficients_model_units'])
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite fixed-truth concentration')
    return prediction, {'substep_count': sum(model.recurrent_substeps_per_interval),
                        'response_basis_count': basis.shape[1]}


def screen_existing_solvers(pilot, output, reference_dir):
    if output.exists():
        raise FileExistsError('Use a fresh solver-screen directory')
    output.mkdir(parents=True)
    settings = [('tensor_semigroup', 36, 8, 4), ('tensor_semigroup', 36, 16, 8),
                ('tensor_semigroup', 64, 16, 8), ('gaussian_parcels', 36, 8, 4),
                ('gaussian_parcels', 36, 16, 8), ('gaussian_parcels', 36, 32, 16)]
    experiment.save_json(output/'manifest.json', {'cases': CASES, 'settings': settings,
        'reference_directory': str(reference_dir), 'subdivisions_for_parcels': 8,
        'no_refitting': True, 'no_q_rescaling': True, 'production_configuration_changed': False,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    rows = []
    torch.set_num_threads(1)
    with experiment.experiment_context():
        for index in CASES:
            case = experiment.case_definition(index)['case_id']
            record = json.loads((pilot/case/'input.json').read_text(encoding='utf-8'))
            saved = np.load(pilot/case/'input.npz')
            reference = np.load(reference_dir/f'{case}.npz')['reference']
            predictions = []
            for solver, grid, steps, samples in settings:
                started = time.perf_counter()
                prediction, _ = fixed_truth_response(record, saved['stations'], grid, steps, samples, solver)
                predictions.append(prediction)
                row = {'case': case, 'solver': solver, 'grid': grid, 'steps_per_hour': steps,
                    'samples_per_hour': samples, 'seconds': time.perf_counter()-started,
                    'relative_l2_to_fine_grid': float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))}
                rows.append(row)
                print(row, flush=True)
                experiment.save_json(output/'summary.json', rows)
            np.savez_compressed(output/f'{case}.npz', predictions=np.asarray(predictions), reference=reference)
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_reference_check(pilot, output):
    from paper_reference_transport import puff_response
    if output.exists():
        raise FileExistsError('Preserve prior independent-reference checks')
    output.mkdir(parents=True)
    settings = [('gaussian', 36, 8, 4), ('gaussian', 64, 16, 8),
                ('gaussian', 96, 32, 16), ('gaussian', 128, 64, 32),
                ('gaussian_parcels', 36, 8, 4), ('gaussian_parcels', 36, 16, 8),
                ('gaussian_parcels', 36, 32, 16)]
    experiment.save_json(output/'manifest.json', {'cases': CASES, 'settings': settings,
        'reference': 'Independent NumPy Gaussian puffs, exact analytic wind/tensor integration,64-point Gauss-Legendre per release interval,32 midpoint observations/hour',
        'fit_or_amplitude_rescale': False, 'reference_code_sha256': hashlib.sha256(
            (Path(__file__).parent/'paper_reference_transport.py').read_bytes()).hexdigest(),
        'reference_check_tolerance_relative_l2': 1e-6,
        'target_transport_relative_l2': 0.01})
    torch.set_num_threads(1)
    rows, reference_checks = [], []
    with experiment.experiment_context():
        for index in CASES:
            case = experiment.case_definition(index)['case_id']
            record = json.loads((pilot/case/'input.json').read_text(encoding='utf-8'))
            saved = np.load(pilot/case/'input.npz')
            times = (np.arange(experiment.HOURS*32)+0.5)/32
            reference_predictions = []
            for order in (32, 64):
                response = puff_response(saved['stations'], times, np.linspace(0., 6., 5),
                    record['source_norm'], 0.08, 0.03, wind=record['wind'], order=order)
                reference_predictions.append((response@np.asarray(record['coefficients_model_units'])).reshape(
                    experiment.HOURS, 32, 5).mean(1).ravel())
            reference = reference_predictions[-1]
            quadrature_change = float(np.linalg.norm(reference_predictions[0]-reference)/np.linalg.norm(reference))
            if quadrature_change > 1e-6:
                raise RuntimeError('Independent release quadrature failed convergence check')
            reference_checks.append({'case': case, 'order32_to64_relative_l2': quadrature_change})
            predictions = []
            for solver, grid, steps, samples in settings:
                prediction, _ = fixed_truth_response(record, saved['stations'], grid, steps, samples, solver)
                predictions.append(prediction)
                row = {'case': case, 'solver': solver, 'grid': grid, 'steps_per_hour': steps,
                    'samples_per_hour': samples, 'relative_l2_to_independent_reference': float(
                        np.linalg.norm(prediction-reference)/np.linalg.norm(reference))}
                rows.append(row)
                print(row, flush=True)
                experiment.save_json(output/'summary.json', rows)
            np.savez_compressed(output/f'{case}.npz', reference=reference,
                predictions=np.asarray(predictions), reference_order32=reference_predictions[0])
    experiment.save_json(output/'reference_convergence.json', reference_checks)
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(pilot, output):
    if output.exists():
        raise FileExistsError(f'Use a new output directory, preserving prior evidence: {output}')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = {'case_indices': CASES, 'settings_grid_steps_samples': SETTINGS,
        'gate_relative_l2': GATE_RELATIVE_L2,
        'reference': SETTINGS[-1], 'q_scale': 'original pilot coefficients, no renormalization',
        'interpretation': 'known-truth forward numerical checks; not an inversion or independent physical model',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pilot_manifest_sha256': hashlib.sha256((pilot/'manifest.json').read_bytes()).hexdigest()}
    experiment.save_json(output/'manifest.json', manifest)
    rows, conclusions = [], []
    try:
        with experiment.experiment_context():
            for index in CASES:
                case = experiment.case_definition(index)['case_id']
                record = json.loads((pilot/case/'input.json').read_text(encoding='utf-8'))
                saved = np.load(pilot/case/'input.npz')
                predictions = []
                case_rows = []
                for grid, steps, samples in SETTINGS:
                    started = time.perf_counter()
                    prediction, diagnostic = fixed_truth_response(record, saved['stations'], grid, steps, samples)
                    predictions.append(prediction)
                    case_rows.append({'case': case, 'grid': grid, 'steps_per_hour': steps,
                        'samples_per_hour': samples, 'seconds': time.perf_counter()-started, **diagnostic})
                    print(f'{case}: grid={grid}, dt=1/{steps}h, samples={samples}, '
                          f'elapsed={case_rows[-1]["seconds"]:.2f}s', flush=True)
                np.testing.assert_allclose(predictions[0], saved['clean'], rtol=2e-5, atol=1e-6)
                reference = predictions[-1]
                for row, prediction in zip(case_rows, predictions):
                    row['relative_l2_to_reference'] = float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))
                    row['maximum_absolute_error_to_reference'] = float(np.max(np.abs(prediction-reference)))
                np.savez_compressed(output/f'{case}.npz', settings=np.asarray(SETTINGS),
                    predictions=np.asarray(predictions), reference=reference, stations=saved['stations'],
                    original_clean=saved['clean'], coefficients=record['coefficients_model_units'])
                rows.extend(case_rows)
                conclusions.append({'case': case,
                    'pilot_relative_l2_to_reference': case_rows[0]['relative_l2_to_reference'],
                    'reference_refinement_relative_l2': case_rows[-2]['relative_l2_to_reference'],
                    'pilot_passes_5pct_gate': case_rows[0]['relative_l2_to_reference'] <= GATE_RELATIVE_L2,
                    'reference_passes_5pct_temporal_gate': case_rows[-2]['relative_l2_to_reference'] <= GATE_RELATIVE_L2})
                experiment.save_json(output/'summary.json', conclusions)
    except Exception as exc:
        experiment.save_json(output/'STOPPED.json', {'error_type': type(exc).__name__, 'message': str(exc)})
        raise
    with (output/'convergence.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = all(r['pilot_passes_5pct_gate'] and r['reference_passes_5pct_temporal_gate'] for r in conclusions)
    experiment.save_json(output/'gate.json', {'pass': passed,
        'action': 'Proceed to independently specified inversion checks' if passed else
        'Do not freeze the coarse pilot as a numerically adequate physical configuration; inspect errors first',
        'reference_is_exact': False})
    print(json.dumps(conclusions, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=experiment.ROOT/'result/paper_experiments_20261017/development_12')
    parser.add_argument('--output', type=Path, default=experiment.ROOT/'result/paper_experiments_20261017/numerical_checks')
    parser.add_argument('--solver-screen', action='store_true')
    parser.add_argument('--independent-reference', action='store_true')
    parser.add_argument('--reference', type=Path, default=experiment.ROOT/'result/paper_experiments_20261017/numerical_checks')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((experiment.ROOT/'result').resolve()):
        parser.error('Output must stay under result/')
    if args.independent_reference:
        independent_reference_check(args.pilot, args.output)
    elif args.solver_screen:
        screen_existing_solvers(args.pilot, args.output, args.reference)
    else:
        run(args.pilot, args.output)
