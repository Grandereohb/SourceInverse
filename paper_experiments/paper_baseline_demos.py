"""Independent conditional Gaussian-puff/NNLS/Powell development demo.

This implements a standard model/solver combination, NOT a strict reproduction
of Kumar2022, Xu2024 or Newman2025. It is not yet a formal matched comparison.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize, nnls

from paper_reference_transport import puff_response

ROOT = Path(__file__).resolve().parents[1]/'result/paper_experiments_20261017'


def mean_response(stations, source, d_parallel, d_perp, wind, samples=16, sigma=0.15,
                  q_basis_count=5):
    times = (np.arange(6*samples)+0.5)/samples
    count=5 if q_basis_count==1 else q_basis_count
    response = puff_response(stations, times, np.linspace(0., 6., count), source,
                             d_parallel, d_perp, wind=wind, order=32,sigma=sigma)
    response=response.reshape(6,samples,len(stations),count).mean(1).reshape(6*len(stations),count)
    return response.sum(1,keepdims=True) if q_basis_count==1 else response


def conditional_nnls(response, observed):
    response, observed = np.asarray(response, dtype=float), np.asarray(observed, dtype=float)
    if (response.ndim != 2 or observed.shape != (response.shape[0],)
            or not np.isfinite(response).all() or not np.isfinite(observed).all()):
        raise ValueError('Finite matching response and observations required')
    norms = np.linalg.norm(response, axis=0)
    if np.any(norms == 0):
        raise ValueError('Unobservable zero response column')
    scaled, _ = nnls(response/norms, observed, maxiter=1000)
    coefficients = scaled/norms
    prediction = response@coefficients
    return coefficients, prediction, float(np.mean((prediction-observed)**2))


def fit_demo(stations, observed, wind, max_evaluations=400, *, sigma=0.15,
             q_basis_count=5, search_step=0.5):
    calls, best = 0, None
    records = []

    def evaluate(theta):
        nonlocal calls, best
        source, perpendicular, delta = np.asarray(theta[:2]), float(theta[2]), float(theta[3])
        response = mean_response(stations, source, perpendicular+delta, perpendicular, wind,
                                  sigma=sigma,q_basis_count=q_basis_count)
        coefficients, prediction, score = conditional_nnls(response, observed)
        calls += 1
        record = {'source_norm': source.tolist(), 'diffusion_norm': [perpendicular+delta, perpendicular],
            'coefficients': coefficients.tolist(), 'prediction': prediction.tolist(), 'mse': score}
        if best is None or score < best['mse']:
            best = record
        return score

    started = time.perf_counter()
    for scale in (0.5, 1., 2.):
        for ratio in (1., 3.):
            perpendicular = 0.035*scale
            for x in np.arange(-0.75,0.75+search_step*0.1,search_step):
                for y in np.arange(-0.75,0.75+search_step*0.1,search_step):
                    theta = np.array([x, y, perpendicular, perpendicular*(ratio-1)])
                    score = evaluate(theta)
                    records.append((score, theta))
    records.sort(key=lambda r: r[0])
    starts = [records[0][1]]
    diverse = [theta for _, theta in records if np.linalg.norm(theta[:2]-starts[0][:2]) >= 0.4]
    starts.append(diverse[0] if diverse else records[1][1])
    optimizer_results = []
    for start in starts:
        result = minimize(evaluate, start, method='Powell',
            bounds=[(-0.75, 0.75), (-0.75, 0.75), (0.00100005, None), (0., None)],
            options={'maxfev': max_evaluations, 'maxiter': 30, 'xtol': 1e-4, 'ftol': 1e-8})
        optimizer_results.append({'success': bool(result.success), 'message': str(result.message),
                                   'nfev': int(result.nfev), 'last_score': float(result.fun)})
    return {**best, 'seconds': time.perf_counter()-started, 'response_matrix_calls': calls,
        'basis_response_count': calls*5, 'optimizer_results': optimizer_results,
        'coarse_calls': len(records), 'starts': [s.tolist() for s in starts],
        'selection': 'minimum observed SSE among all evaluations; truth not used',
        'converged': all(r['success'] for r in optimizer_results)}


def main(output, max_evaluations):
    if output.exists():
        raise FileExistsError('Preserve existing demo evidence; use a new version for changes')
    output.mkdir()
    manifest = {'method': 'conditional Gaussian-puff + NNLS + Powell demo', 'cases': [0, 2, 10],
        'production_code_changed': False,
        'not_strict_reproductions_of': ['Kumar2022', 'Xu2024', 'Newman2025'],
        'status': 'development feasibility only; IA has not yet been run on these NEW inputs',
        'data': 'independent analytic wind/tensor Gaussian-puff truth plus original saved pilot noise',
        'parameter_constraints': 'same source bounds and D_parallel>=D_perp>=0.00100005;5 nonnegative hats',
        'max_function_evaluations_per_start': max_evaluations, 'timing': 'diagnostic only, other experiment jobs may run concurrently',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256((Path(__file__).parent/'paper_reference_transport.py').read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary = []
    for index in manifest['cases']:
        case = f'pilot_{index+1:02d}'
        record = json.loads((ROOT/'development_12'/case/'input.json').read_text(encoding='utf-8'))
        old = np.load(ROOT/'development_12'/case/'input.npz')
        # New truth is kept separate from the old grid-generated observations.
        times = (np.arange(6*32)+0.5)/32
        matrix = puff_response(old['stations'], times, np.linspace(0., 6., 5),
            record['source_norm'], 0.08, 0.03, wind=record['wind'], order=64).reshape(6,32,5,5).mean(1).reshape(30,5)
        coefficients = np.asarray(record['coefficients_model_units'])
        clean = matrix@coefficients
        observed = clean+old['observed']-old['clean']
        np.savez_compressed(output/f'{case}_input.npz', stations=old['stations'], observed=observed,
                            clean=clean, coefficients=coefficients, source=record['source_norm'])
        print(f'START baseline demo {case}', flush=True)
        result = fit_demo(old['stations'], observed, record['wind'], max_evaluations)
        result['localization_error_m'] = float(1000*np.linalg.norm(np.asarray(result['source_norm'])-record['source_norm']))
        knots = np.linspace(0.,6.,5)
        q_times = np.linspace(0.,6.,193)
        true_q = np.interp(q_times, knots, coefficients)
        fitted_q = np.interp(q_times, knots, result['coefficients'])
        result['q_relative_l2'] = float(np.linalg.norm(fitted_q-true_q)/np.linalg.norm(true_q))
        result['case'] = case
        (output/f'{case}_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        summary.append({k: result[k] for k in ('case','localization_error_m','q_relative_l2','mse','converged','response_matrix_calls','seconds')})
        (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(summary[-1], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'baseline_demo')
    parser.add_argument('--max-evaluations', type=int, default=400)
    args = parser.parse_args()
    if args.max_evaluations < 1 or not args.output.resolve().is_relative_to((ROOT.parents[0]).resolve()):
        parser.error('Positive budget and output under result/ required')
    main(args.output, args.max_evaluations)
