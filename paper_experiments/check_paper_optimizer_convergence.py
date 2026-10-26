"""Prespecified development-only iteration-budget checks from saved starts."""

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_paper_experiments as experiment

ROOT = experiment.ROOT / 'result/paper_experiments_20261017'
CASES = (0, 3, 10)
BUDGETS = (80, 200)


def candidate_from_record(record):
    values = dict(record)
    for key in ('coefficients', 'q_integration', 'prediction'):
        values[key] = np.asarray(values[key], dtype=float)
    return experiment.ia.IAVPDICandidate(**values)


def main():
    output = ROOT / 'optimizer_checks'
    if output.exists():
        raise FileExistsError('Preserve existing results; use a new study directory/version')
    output.mkdir()
    experiment.save_json(output/'manifest.json', {'cases': CASES, 'budgets': BUDGETS,
        'previous_budget': 20, 'learning_rate': 0.02,
        'selection': 'Prespecified development probes01,04,11; not independent evaluation',
        'stopping': 'Fixed budget, no truth-based stopping or result-based start selection',
        'same_saved_common_starts': True, 'generator_and_solver': 'original coarse grid pilot',
        'purpose': 'Optimizer convergence only; does not override failed physical-resolution gate',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    try:
        with experiment.experiment_context():
            for index in CASES:
                name = experiment.case_definition(index)['case_id']
                previous = json.loads((ROOT/'development_12'/name/'result.json').read_text(encoding='utf-8'))
                model, xyt, basis, coefficients, observed, clean, _, metadata = experiment.make_case(index)
                saved = np.load(ROOT/'development_12'/name/'input.npz')
                np.testing.assert_array_equal(observed, saved['observed'])
                starts = [candidate_from_record(r) for r in previous['common_starts']]
                true_q = basis.cpu().numpy() @ coefficients
                for method in ('ia_vpdi', 'matched_joint_adam'):
                    rows.append({'case': name, 'method': method, 'steps': 20,
                        'seconds_refinement': previous[method]['seconds_including_common_search']-previous['common_search_seconds'],
                        **{k: previous[method][k] for k in ('localization_error_m', 'q_relative_l2',
                            'objective_mse_observed', 'at_search_boundary')}})
                for budget in BUDGETS:
                    print(f'START {name}: {budget}steps/start', flush=True)
                    started = time.perf_counter()
                    profiles = [experiment.ia.refine_profiled_candidate(model, experiment.SIGMA,
                        xyt, observed, basis, start, experiment.BOUNDS, experiment.BOUNDS,
                        steps=budget, lr=0.02, patience=budget+1, min_delta=0.0) for start in starts]
                    profile_seconds = time.perf_counter()-started
                    if not all(run.success for run in profiles):
                        raise RuntimeError(f'Profile failed: {[r.message for r in profiles]}')
                    best = min((run.best for run in profiles), key=lambda r: r.profile_score)
                    joints = [experiment.joint_refine(model, xyt, basis, observed, start, budget) for start in starts]
                    joint = min(joints, key=lambda r: r['score'])
                    for method, source, fitted, prediction, seconds in (
                        ('ia_vpdi', [best.source_x, best.source_y], best.coefficients, best.prediction, profile_seconds),
                        ('matched_joint_adam', joint['source'], joint['coefficients'], joint['prediction'],
                         sum(run['seconds'] for run in joints))):
                        metric = experiment.metrics(source, fitted, prediction, metadata['source_norm'],
                            true_q, basis, clean, observed)
                        rows.append({'case': name, 'method': method, 'steps': budget,
                            'seconds_refinement': seconds, **{k: metric[k] for k in ('localization_error_m',
                            'q_relative_l2', 'objective_mse_observed', 'at_search_boundary')}})
                    experiment.save_json(output/f'{name}_{budget}.json', {
                        'profile': asdict(best), 'profile_trajectories': [r.trajectory for r in profiles],
                        'joint': joint, 'rows': rows[-2:]})
                    experiment.save_json(output/'summary.json', rows)
                    print(f'DONE {name}/{budget}: IA={rows[-2]["localization_error_m"]:.2f}m, '
                          f'joint={rows[-1]["localization_error_m"]:.2f}m', flush=True)
    except Exception as exc:
        experiment.save_json(output/'STOPPED.json', {'error_type': type(exc).__name__, 'message': str(exc)})
        raise
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
