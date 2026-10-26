"""Versioned Results 4.1 rerun with one response implementation for all solvers.

Reuses the frozen 120 inputs, not their fitted results. Production and previous
experiment drivers are unchanged. Powell and Adam budgets are different kinds
of budgets, not claims of equal work or a same-hardware speed benchmark.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
import torch

import run_paper_formal_experiments as formal
from paper_fast_parcels import CachedMeanResponse

STUDY = formal.ROOT / 'evaluation_v3_shared_forward'
METHOD = 'shared_response_nnls_powell'


class SharedCaseResponse(CachedMeanResponse):
    """Capture the existing driver's response context and selected starts."""
    def __init__(self, stations):
        super().__init__(stations=stations)
        self.starts = None

    def __call__(self, model, sigma, xyt, basis, track_gradients=False):
        self.context = model, sigma, xyt, basis
        return super().__call__(model, sigma, xyt, basis, track_gradients)

    def fit_powell(self, observations, protocol, common_seconds):
        model, sigma, xyt, basis = self.context
        best = None
        calls = 0

        def evaluate(theta):
            nonlocal best, calls
            x, y, perpendicular, contrast = map(float, theta)
            tolerance = 16*np.spacing(max(abs(perpendicular), 1.))
            if contrast < -tolerance or perpendicular < .00100005-tolerance:
                raise ValueError('Powell evaluated outside diffusion constraints')
            perpendicular = max(perpendicular, .00100005)
            parallel = perpendicular + max(contrast, 0.)
            candidate = formal.base.ia.score_all_data_candidate(
                model, sigma, xyt, observations, basis, x, y, parallel, perpendicular)
            calls += 1
            if best is None or candidate.profile_score < best.profile_score:
                best = candidate
            return candidate.profile_score

        # The same coarse candidates and both starts used by IA and joint Adam.
        started = time.perf_counter()
        optimizers = []
        for start in self.starts:
            theta = [start.source_x, start.source_y, start.diffusion_perp_norm,
                     start.diffusion_parallel_norm-start.diffusion_perp_norm]
            evaluate(theta)
            result = minimize(evaluate, theta, method='Powell',
                bounds=[formal.base.BOUNDS, formal.base.BOUNDS, (.00100005, None), (0., None)],
                options=dict(maxfev=protocol['puff_powell_max_evaluations_per_start'],
                             maxiter=30, xtol=1e-4, ftol=1e-8))
            optimizers.append(dict(success=bool(result.success), message=str(result.message),
                                   nfev=int(result.nfev), last_score=float(result.fun)))
        seconds = time.perf_counter()-started
        return dict(source_norm=[best.source_x, best.source_y],
            diffusion_norm=[best.diffusion_parallel_norm, best.diffusion_perp_norm],
            coefficients=best.coefficients, prediction=best.prediction,
            mse=best.profile_score, seconds=seconds,
            seconds_including_common_search=common_seconds+seconds,
            response_matrix_calls=calls, basis_response_count=calls*int(basis.shape[1]),
            optimizer_results=optimizers, converged=all(r['success'] for r in optimizers),
            selection='minimum observed SSE; identical response, NNLS, coarse search and starts',
            response_implementation='paper_fast_parcels.CachedMeanResponse')


def fit_case(case, protocol, device):
    metadata = json.loads((case/'input.json').read_text(encoding='utf-8'))
    with np.load(case/'input.npz') as data:
        response = SharedCaseResponse(len(data['stations']))
    select = formal.base.ia.select_refinement_starts

    def capture_starts(*args, **kwargs):
        response.starts = select(*args, **kwargs)
        return response.starts

    def fit_baseline(stations, observations, wind, *args, **kwargs):
        # Common search timing is recorded by the original driver; fill below.
        return response.fit_powell(observations, protocol, 0.)

    with patch.object(formal, 'CachedMeanResponse', lambda **kwargs: response), \
         patch.object(formal, 'fit_demo', fit_baseline), \
         patch.object(formal.base.ia, 'select_refinement_starts', capture_starts):
        result = formal.fit_case(case, protocol, device)
    baseline = result['methods'].pop('independent_puff_demo')
    baseline['seconds_including_common_search'] = result['common_search_seconds']+baseline['seconds']
    result['methods'][METHOD] = baseline
    result['comparison_version'] = 'shared_forward_v3'
    result['forward_consistency'] = verify_replays(response, result)
    formal.base.save_json(case/'result.json', result)
    return result


def verify_replays(response, result):
    model, sigma, xyt, basis = response.context
    records = {}
    for method, fit in result['methods'].items():
        with formal.base.ia.temporary_source_and_diffusion(
                model, *fit['source_norm'], *fit['diffusion_norm']):
            matrix = response(model, sigma, xyt, basis).cpu().numpy()
        prediction = matrix @ np.asarray(fit['coefficients'])
        np.testing.assert_allclose(prediction, fit['prediction'], rtol=2e-5, atol=2e-6)
        records[method] = float(np.max(np.abs(prediction-fit['prediction'])))
    return dict(shared_response=True, replay_max_absolute_difference=records)


def prepare(study=STUDY):
    previous = formal.ROOT/'evaluation_v2_boundary_fix'
    protocol = deepcopy(json.loads((previous/'protocol.json').read_text(encoding='utf-8')))
    protocol.update(previous_version=previous.name, reuse='frozen inputs only; ALL three methods rerun',
        correction='same CachedMeanResponse and same coarse starts for IA, joint Adam and NNLS-Powell',
        baselines='matched joint Adam; shared_response_nnls_powell, not independent physics',
        generator_independence='independent implementation, SAME Gaussian physics family',
        external_physics_validation='existing controlled releases; no new CFD cohort in this rerun',
        timing='all three methods use the same selected response device; unequal optimizer budgets disclosed')
    protocol.pop('reuse_provenance', None)
    paths = [Path(__file__), Path(formal.__file__), Path(formal.base.__file__),
             Path(formal.parcel.__file__), Path(__file__).with_name('paper_fast_parcels.py'),
             Path(__file__).with_name('paper_reference_transport.py')]
    paths += list((formal.base.ROOT/'pinn_source').rglob('*.py'))
    protocol['frozen_files'] = {p.relative_to(formal.base.ROOT).as_posix():
                              hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    study.mkdir(exist_ok=True)
    destination = study/'protocol.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Shared-forward protocol changed; use a new version')
    else:
        formal.base.save_json(destination, protocol)
    for spec in protocol['case_design']:
        old, new = previous/spec['case_id'], study/spec['case_id']
        new.mkdir(exist_ok=True)
        for name in ('input.json', 'input.npz'):
            if not (new/name).exists():
                shutil.copy2(old/name, new/name)
            if (old/name).read_bytes() != (new/name).read_bytes():
                raise ValueError(f'Input differs: {spec["case_id"]}/{name}')
    return protocol


def run(device, limit):
    protocol = prepare()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    formal.base.save_json(STUDY/'runtime.json', dict(device=device, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None))
    for spec in protocol['case_design'][:limit]:
        case = STUDY/spec['case_id']
        if (case/'result.json').exists():
            saved = json.loads((case/'result.json').read_text(encoding='utf-8'))
            if saved.get('comparison_version') == 'shared_forward_v3':
                continue
            raise ValueError('Interrupted intermediate result; inspect before resuming')
        print('START', spec['case_id'], spec['cohort'], spec['network'], flush=True)
        try:
            result = fit_case(case, protocol, device)
        except Exception as exc:
            formal.base.save_json(case/'STOPPED.json', dict(type=type(exc).__name__, message=str(exc)))
            raise
        print('DONE', spec['case_id'], {m: round(v['localization_error_m'], 2)
              for m, v in result['methods'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--limit', type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.limit <= 120:
        parser.error('limit must be between 1 and 120')
    run(args.device, args.limit)
