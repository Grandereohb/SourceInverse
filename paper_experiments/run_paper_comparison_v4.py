"""New baseline cohort on the existing 216 frozen inputs; no new scenarios.

All methods share the same finite box and common, observation-only warm starts.
Published old fits remain untouched. Timing is conditional on reused coarse
initialization; matrix counts are not equivalent FLOPs for different derivatives.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deployment_runs/paper_baseline_dependencies'))
import run_paper_formal_experiments as formal
from paper_fast_parcels import CachedMeanResponse
from paper_joint_baselines import JointBounds, JointResidual, fit_joint_nls, fit_standard_pso

STUDY = formal.ROOT


@contextmanager
def projected_diffusion_box(model, bounds, coefficient_scale=None):
    original = torch.optim.Adam
    def create(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        def project(*unused):
            dp, dq = formal.base.ia.model_anisotropic_diffusion_norm(model)
            perp, contrast = float(dq.detach()), float((dp-dq).detach())
            clipped = np.clip([perp, contrast], bounds.lower[2:4], bounds.upper[2:4])
            if not np.array_equal(clipped, [perp, contrast]):
                formal.base.ia.set_model_anisotropic_diffusion_norm(model,
                    float(clipped.sum()), float(clipped[0]))
            parameters = optimizer.param_groups[0]['params']
            if len(parameters) == 5:
                if coefficient_scale is None:
                    raise ValueError('Joint Adam needs its declared coefficient coordinate scale')
                with torch.no_grad():
                    parameters[-1].clamp_(0., float(bounds.upper[4])/coefficient_scale)
        optimizer.register_step_post_hook(project)
        return optimizer
    with patch.object(torch.optim, 'Adam', create):
        yield


def input_cases():
    return [(folder, case) for folder in ('evaluation_v3_shared_forward',
                                          'supplementary_scenarios_v2_48each')
            for case in json.loads((STUDY/folder/'protocol.json').read_text(encoding='utf-8'))['case_design']]


def prepare(output, device, smoke):
    cases = input_cases()
    if smoke:
        cases = cases[:2]
    protocol = dict(version='new_baselines_v4', technical_smoke=smoke, device=device,
        dtype='float64', cases=[dict(parent=f, case=s) for f, s in cases],
        objective='all valid observations, unweighted squared residuals, same5 hats',
        methods=['ia_vpdi', 'joint_nls_trf', 'standard_gbest_pso', 'matched_joint_adam'],
        initialization='reuse saved observation-only common coarse starts; refit Q at same positions',
        timing='conditional refinement timing; original common search NOT a new hardware timing',
        source_bounds=[-.75, .75], perp_bounds=[.00100005, 1.], contrast_bounds=[0., 1.],
        q_upper_rule='100 times largest coefficient among the TWO saved common coarse starts; no truth',
        cap_interpretation='broad numerical search limits, not verified physical priors',
        adam_steps=3 if smoke else 1000, nls_nfev=3 if smoke else 500,
        pso_particles=8 if smoke else 32, pso_iterations=2 if smoke else 64,
        pso_seeds=[1729] if smoke else [1729, 1730, 1731],
        pso_initialization='same two common starts plus random uniform population; no local refinement',
        budgets='IA/Adam2000 updates; NLS<=2000 residual/Jacobian response evaluations; '
                'PSO2048 particle evaluations per replicate; costs differ, wall time also reported',
        old_results='preserved; no winner filtering or extra optimization experiments')
    paths = [Path(__file__), Path(__file__).with_name('paper_joint_baselines.py'),
             Path(__file__).with_name('paper_fast_parcels.py'), Path(formal.__file__),
             Path(formal.base.__file__), Path(formal.parcel.__file__),
             *sorted((ROOT/'pinn_source').rglob('*.py'))]
    protocol['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol['input_sha256'] = {f'{f}/{s["case_id"]}/{name}':
        hashlib.sha256((STUDY/f/s['case_id']/name).read_bytes()).hexdigest()
        for f, s in cases for name in ('input.json', 'input.npz', 'result.json')}
    destination = output/'protocol.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Code/input/protocol changed; preserve output and choose a new version')
    else:
        output.mkdir(parents=True, exist_ok=False)
        formal.base.save_json(destination, protocol)
    return protocol


def fit_case(case, output, protocol):
    metadata = json.loads((case/'input.json').read_text(encoding='utf-8'))
    previous = json.loads((case/'result.json').read_text(encoding='utf-8'))
    with np.load(case/'input.npz') as archive:
        data = {name: archive[name].copy() for name in archive.files}
    torch.manual_seed(metadata['seed'])
    model, xyt, basis = formal.parcel.configure(metadata, data['stations'], protocol['device'],
        sigma=metadata['sigma'], wind_interval_mean=True)
    model.double()
    response = CachedMeanResponse(stations=len(data['stations']))
    amplitude_cap = 100*max(float(np.max(s['coefficients'])) for s in previous['common_starts'])
    bounds = JointBounds(np.r_[[-.75, -.75, .00100005, 0.], np.zeros(basis.shape[1])],
                         np.r_[[.75, .75, 1., 1.], np.full(basis.shape[1], amplitude_cap)])
    common_started = time.perf_counter()
    with patch.object(formal.base.ia, 'build_response_matrix', response):
        starts = [formal.base.ia.score_all_data_candidate(model, metadata['sigma'], xyt,
            data['observed'], basis, s['source_x'], s['source_y'],
            s['diffusion_parallel_norm'], s['diffusion_perp_norm'], q_upper_bound=amplitude_cap)
            for s in previous['common_starts']]
    parameters = [np.r_[s.source_x, s.source_y, s.diffusion_perp_norm,
        s.diffusion_parallel_norm-s.diffusion_perp_norm, s.coefficients] for s in starts]
    for initial in parameters:
        bounds.validate(initial)
    setup_seconds = time.perf_counter()-common_started
    results = {}
    before_calls = response.calls
    started = time.perf_counter()
    with patch.object(formal.base.ia, 'build_response_matrix', response), projected_diffusion_box(model, bounds):
        profiles = [formal.base.ia.refine_profiled_candidate(model, metadata['sigma'], xyt,
            data['observed'], basis, start, (-.75, .75), (-.75, .75),
            steps=protocol['adam_steps'], lr=.02, patience=protocol['adam_steps']+1,
            min_delta=0., q_upper_bound=amplitude_cap) for start in starts]
    if not all(run.success for run in profiles):
        raise RuntimeError('Profile refinement failed: '+str([run.message for run in profiles]))
    best = min((r.best for r in profiles), key=lambda candidate: candidate.profile_score)
    results['ia_vpdi'] = dict(source_norm=[best.source_x, best.source_y],
        diffusion_norm=[best.diffusion_parallel_norm, best.diffusion_perp_norm],
        coefficients=best.coefficients, prediction=best.prediction,
        seconds=time.perf_counter()-started, response_matrix_calls=response.calls-before_calls,
        trace=[[row['profile_score'] for row in r.trajectory] for r in profiles])
    objective = JointResidual(response, model, metadata['sigma'], xyt, basis, data['observed'], bounds)
    results['joint_nls_trf'] = fit_joint_nls(objective, parameters, protocol['nls_nfev'])
    pso = []
    for seed in protocol['pso_seeds']:
        objective = JointResidual(response, model, metadata['sigma'], xyt, basis, data['observed'], bounds)
        pso.append(fit_standard_pso(objective, seed=seed, particles=protocol['pso_particles'],
            iterations=protocol['pso_iterations'], starts=parameters))
    results['standard_gbest_pso'] = dict(replicates=pso,
        aggregation='all seeds retained; report per-case median metrics, not best-of-seeds')
    started = time.perf_counter()
    before_calls = response.calls
    joints = []
    with patch.object(formal.base, 'averaged_response', response), \
         patch.object(formal.base, 'SIGMA', metadata['sigma']):
        for start in starts:
            scale = max(float(np.max(start.coefficients)), 1e-6)
            with projected_diffusion_box(model, bounds, scale):
                joints.append(formal.base.joint_refine(model, xyt, basis, data['observed'],
                                                       start, protocol['adam_steps']))
    joint = min(joints, key=lambda run: run['score'])
    if np.max(joint['coefficients']) > amplitude_cap:
        raise ValueError('Joint Adam amplitude cap not enforced; do not save unmatched result')
    results['matched_joint_adam'] = dict(source_norm=joint['source'], diffusion_norm=joint['diffusion'],
        coefficients=joint['coefficients'], prediction=joint['prediction'],
        seconds=time.perf_counter()-started, response_matrix_calls=response.calls-before_calls,
        trace=[run['trajectory'] for run in joints])
    for name, result in results.items():
        for fit in result.get('replicates', [result]):
            fit.update(formal.metrics(metadata, data, fit['source_norm'], np.asarray(fit['coefficients']),
                                       np.asarray(fit['prediction'])))
            theta = np.r_[fit['source_norm'], fit['diffusion_norm'][1],
                fit['diffusion_norm'][0]-fit['diffusion_norm'][1], fit['coefficients']]
            # Model setters may differ by a few ULPs at a closed diffusion boundary.
            np.testing.assert_array_less(bounds.lower-1e-12, theta)
            np.testing.assert_array_less(theta, bounds.upper+1e-12)
            with formal.base.ia.temporary_source_and_diffusion(model, *fit['source_norm'], *fit['diffusion_norm']):
                replay = response(model, metadata['sigma'], xyt, basis).detach().cpu().numpy()@fit['coefficients']
            np.testing.assert_allclose(replay, fit['prediction'], rtol=2e-5, atol=2e-7)
            fit['replay_max_absolute_difference'] = float(np.max(np.abs(replay-fit['prediction'])))
    formal.base.ia.set_model_anisotropic_diffusion_norm(model, best.diffusion_parallel_norm, best.diffusion_perp_norm)
    with torch.no_grad():
        model.xs.fill_(best.source_x)
        model.ys.fill_(best.source_y)
    with patch.object(formal.base.ia, 'build_response_matrix', response):
        info = formal.base.ia.compute_profile_information(model, metadata['sigma'], xyt,
            data['observed'], basis, best.coefficients, length_m=metadata['length_m'],
            source_bounds=((-.75, .75), (-.75, .75)), q_upper_bound=amplitude_cap)
    record = dict(case_id=metadata['case_id'], metadata=metadata, bounds=asdict(bounds),
        common_starts=[asdict(s) for s in starts], setup_seconds=setup_seconds,
        methods=results, conditional_information=asdict(info), complete=True)
    formal.base.save_json(output/'result.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = prepare(output, args.device, args.smoke)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(output)
        for entry in protocol['cases'][:args.limit]:
            case_id = entry['case']['case_id']
            destination = output/case_id
            if (destination/'result.json').exists():
                if not json.loads((destination/'result.json').read_text(encoding='utf-8'))['complete']:
                    raise ValueError('Incomplete result requires inspection')
                continue
            destination.mkdir(exist_ok=True)
            print('START_V4', case_id, flush=True)
            record = fit_case(STUDY/entry['parent']/case_id, destination, protocol)
            print('DONE_V4', case_id, {k: v.get('localization_error_m') for k,v in record['methods'].items()}, flush=True)
    finally:
        os.chdir(previous_cwd)


if __name__ == '__main__':
    main()
