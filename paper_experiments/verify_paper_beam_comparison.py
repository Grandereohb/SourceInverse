"""Higher-quadrature prediction replay at saved new-baseline parameters only.

No optimization, new cases, source selection or threshold tuning. Reuses the
existing1% relative prediction-change criterion for public beam evaluations.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_paper_controlled_release as physical

ROOT = Path(__file__).resolve().parents[1]/'result/paper_experiments_20261017'


def relative_prediction_change(prediction, reference):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError('Finite matching predictions required')
    return float(np.linalg.norm(prediction-reference)/max(np.linalg.norm(reference), np.finfo(float).tiny))


def verify(study, device='cpu'):
    study = Path(study)
    protocol = json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    if protocol['technical_smoke']:
        raise ValueError('Verify full-window fits, not a technical-smoke cohort')
    torch.set_num_threads(1)
    rows, completed = [], 0
    for event in protocol['events']:
        name = event['source'].replace('_', '')+'_'+event['release']
        for height in protocol['heights_m']:
            path = study/name/f'height_{height:.1f}'/'result.json'
            if not path.exists():
                continue
            result = json.loads(path.read_text(encoding='utf-8'))
            model, xyt, basis, response, observed, observations = physical.configure(height,
                device=device, dtype=torch.float64, q_count=1, time_order=32, release_order=32,
                beam_solver='analytic', data_dir=ROOT/'controlled_release_preparation_v2'/name)
            if len(observed) != result['observations']:
                raise ValueError('Numerical replay changed the observation window')
            for method, fits in result['methods'].items():
                for fit in fits.get('replicates', [fits]):
                    parameters = torch.tensor([*fit['source_norm'], fit['diffusion_norm'][1],
                        fit['diffusion_norm'][0]-fit['diffusion_norm'][1]], dtype=torch.float64, device=device)
                    with torch.no_grad():
                        matrix = response(model, 0., xyt, basis, physical_parameters=parameters)
                    prediction = matrix.cpu().numpy()@np.asarray(fit['coefficients'])
                    change = relative_prediction_change(fit['prediction'], prediction)
                    rows.append(dict(event=name, height_m=height, method=method, seed=fit.get('seed'),
                        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        fitting_time_order=protocol['numerical_orders']['time'],
                        fitting_release_order=protocol['numerical_orders']['release'],
                        replay_time_order=32, replay_release_order=32,
                        relative_prediction_change=change, gate_passed=change <= .01,
                        replay_observed_rmse_ppm=float(np.sqrt(np.mean((prediction-observed)**2)))))
            completed += 1
            print('VERIFIED_BEAM', name, height, flush=True)
    output = study/'analysis'
    output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'quadrature_replay.csv', index=False)
    report = dict(final=completed == len(protocol['events'])*len(protocol['heights_m']),
        completed_event_heights=completed, planned_event_heights=len(protocol['events'])*len(protocol['heights_m']),
        fitted_outputs_checked=len(rows), gate_relative_prediction_change=.01,
        passed=sum(row['gate_passed'] for row in rows), failed=sum(not row['gate_passed'] for row in rows),
        parameters_refitted=False, threshold_tuned=False,
        device=device,
        interpretation='finite higher-order reference, not exact physical truth or source-location validation')
    (output/'quadrature_replay_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT/'beam_comparison_cuda_v1')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    print(json.dumps(verify(args.study, args.device), indent=2))
