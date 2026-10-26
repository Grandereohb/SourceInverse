"""Matched constant-Q physical beam comparison on existing release windows.

IA, Joint NLS, public PSO and joint Adam share the transient3D observation
operator. Newman's stationary author implementation remains a separate track.
The old5-hat physical fits remain a capacity-separated supplementary result.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deployment_runs/paper_baseline_dependencies'))
import run_paper_controlled_release as physical
from run_paper_comparison_v4 import projected_diffusion_box
from paper_beam_joint_baselines import BeamJointResidual
from paper_joint_baselines import JointBounds, candidate_parameters, fit_joint_nls, fit_standard_pso


def protocol_for(device, smoke):
    previous = physical.ROOT/'controlled_release_fit_v2_precision'
    original = json.loads((previous/'protocol.json').read_text(encoding='utf-8'))
    events = original['events'][:1] if smoke else original['events']
    paths = [Path(__file__), Path(physical.__file__), Path(__file__).with_name('paper_beam_joint_baselines.py'),
        Path(__file__).with_name('paper_physical_beam_adapter.py'),
        Path(__file__).with_name('paper_joint_baselines.py'),
        Path(__file__).with_name('run_paper_comparison_v4.py'), Path(physical.base.ia.__file__)]
    paths = list(dict.fromkeys(paths+sorted((ROOT/'pinn_source').rglob('*.py'))))
    inputs = []
    for event in events:
        name = event['source'].replace('_', '')+'_'+event['release']
        inputs.extend(physical.ROOT/'controlled_release_preparation_v2'/name/item
                      for item in ('observations.csv', 'raw_axis_weather.csv', 'beams.csv', 'manifest.json'))
        inputs.extend(previous/name/f'height_{height:.1f}'/'transient_constant_q.json'
                      for height in (.3, 1.6))
    return dict(version='beam_shared_baselines_v1', events=events, heights_m=[.3, 1.6],
        technical_smoke=smoke, minutes=3 if smoke else None, device=device, dtype='float64',
        q_count=1, q_units='kg/s', diffusion_units='m2/s (normalized parameter times10000/60)',
        methods=['ia_vpdi', 'joint_nls_trf', 'standard_gbest_pso', 'matched_joint_adam'],
        interpretation='matched constant-Q capacity; old5-hat fits supplementary, not same-capacity ranking',
        observation='all prepared signed pre-window background residuals; minute/path means',
        initialization='old COMMON coarse positions/D, refit constant-Q against current operator',
        source_bounds=[0., 1.1], diffusion_perp_bounds=[.001001000001, 1.], contrast_bounds=[0., 1.],
        q_upper_rule='100 times maximum coefficient of TWO saved common coarse starts, no truth',
        adam_steps=2 if smoke else 500, nls_nfev=2 if smoke else 250,
        particles=4 if smoke else 32, pso_iterations=2 if smoke else 32,
        pso_seeds=[1729] if smoke else [1729, 1730, 1731],
        numerical_orders=dict(time=4 if smoke else 16, release=4 if smoke else 16,
                              beam_solver='analytic'),
        prior_newman='not substituted into this common-forward optimizer comparison',
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def fit_event(event, height, protocol):
    name = event['source'].replace('_', '')+'_'+event['release']
    data_dir = physical.ROOT/'controlled_release_preparation_v2'/name
    previous = json.loads((physical.ROOT/'controlled_release_fit_v2_precision'/name/
        f'height_{height:.1f}'/'transient_constant_q.json').read_text(encoding='utf-8'))
    orders = protocol['numerical_orders']
    model, xyt, basis, response, observed, observations = physical.configure(height,
        minutes=protocol['minutes'], device=protocol['device'], dtype=torch.float64, q_count=1,
        time_order=orders['time'], release_order=orders['release'], beam_solver='analytic', data_dir=data_dir)
    saved = [run['start'] for run in previous['runs']]
    cap = 100*max(float(np.max(start['coefficients'])) for start in saved)
    # Physical adapter has D_scale_norm=1, unlike the synthetic adapter's .05.
    # Match the actual softplus+epsilon floor rather than copying its old bound.
    minimum = float(model.recurrent_d_perp_min_norm)+float(model.recurrent_d_scale_norm)*1e-6+1e-12
    bounds = JointBounds(np.array([0., 0., minimum, 0., 0.]), np.array([1.1, 1.1, 1., 1., cap]))
    with patch.object(physical.base.ia, 'build_response_matrix', response):
        starts = [physical.base.ia.score_all_data_candidate(model, 0., xyt, observed, basis,
            s['source_x'], s['source_y'], s['diffusion_parallel_norm'], s['diffusion_perp_norm'],
            q_upper_bound=cap) for s in saved]
    initial = [candidate_parameters(start) for start in starts]
    for value in initial:
        bounds.validate(value)
    methods = {}
    started = time.perf_counter()
    before_calls = response.calls
    with patch.object(physical.base.ia, 'build_response_matrix', response), projected_diffusion_box(model, bounds):
        profiles = [physical.base.ia.refine_profiled_candidate(model, 0., xyt, observed, basis,
            start, (0., 1.1), (0., 1.1), steps=protocol['adam_steps'], lr=.02,
            patience=protocol['adam_steps']+1, min_delta=0., q_upper_bound=cap) for start in starts]
    if not all(run.success for run in profiles):
        raise RuntimeError('Physical profiled fit failed')
    best = min((run.best for run in profiles), key=lambda candidate: candidate.profile_score)
    methods['ia_vpdi'] = dict(source_norm=[best.source_x, best.source_y],
        diffusion_norm=[best.diffusion_parallel_norm, best.diffusion_perp_norm],
        coefficients=best.coefficients, prediction=best.prediction, seconds=time.perf_counter()-started,
        response_matrix_calls=response.calls-before_calls)
    def objective():
        return BeamJointResidual(response, model, 0., xyt, basis, observed, bounds)
    methods['joint_nls_trf'] = fit_joint_nls(objective(), initial, protocol['nls_nfev'])
    methods['standard_gbest_pso'] = dict(replicates=[fit_standard_pso(objective(), seed=seed,
        particles=protocol['particles'], iterations=protocol['pso_iterations'], starts=initial)
        for seed in protocol['pso_seeds']])
    started = time.perf_counter()
    before_calls = response.calls
    joints = []
    with patch.object(physical.base, 'averaged_response', response), \
         patch.object(physical.base, 'SIGMA', 0.), patch.object(physical.base, 'BOUNDS', (0., 1.1)):
        for start in starts:
            scale = max(float(np.max(start.coefficients)), 1e-6)
            with projected_diffusion_box(model, bounds, scale):
                joints.append(physical.base.joint_refine(model, xyt, basis, observed, start, protocol['adam_steps']))
    best_joint = min(joints, key=lambda run: run['score'])
    methods['matched_joint_adam'] = dict(source_norm=best_joint['source'], diffusion_norm=best_joint['diffusion'],
        coefficients=best_joint['coefficients'], prediction=best_joint['prediction'], seconds=time.perf_counter()-started,
        response_matrix_calls=response.calls-before_calls)
    # Source truth is not used in building common starts or solving any method.
    truth = json.loads((physical.ROOT/'public_data/author_truth_values.json').read_text(encoding='utf-8'))
    reference = np.asarray(truth['values_as_stored'][event['source']+'_location'][:2])
    for result in methods.values():
        for fit in result.get('replicates', [result]):
            theta = np.r_[fit['source_norm'], fit['diffusion_norm'][1],
                fit['diffusion_norm'][0]-fit['diffusion_norm'][1], fit['coefficients']]
            np.testing.assert_array_less(bounds.lower-1e-12, theta)
            np.testing.assert_array_less(theta, bounds.upper+1e-12)
            replay = objective().residual(np.clip(theta, bounds.lower, bounds.upper))+observed
            np.testing.assert_allclose(replay, fit['prediction'], rtol=2e-5, atol=1e-7)
            fit.update(source_m=(100*np.asarray(fit['source_norm'])).tolist(),
                mean_q_kg_h=float(fit['coefficients'][0]*3600.),
                localization_error_m=float(np.linalg.norm(100*np.asarray(fit['source_norm'])-reference)),
                observed_rmse_ppm=float(np.sqrt(np.mean((replay-observed)**2))),
                replay_max_difference=float(np.max(np.abs(replay-fit['prediction']))))
    return dict(event=name, height_m=height, observations=len(observed),
        minutes=observations.time.nunique(), q_count=1, methods=methods, bounds=asdict(bounds),
        common_starts=[asdict(start) for start in starts], complete=True,
        technical_smoke=protocol['technical_smoke'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = protocol_for(args.device, args.smoke)
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text(encoding='utf-8')) != protocol:
            raise ValueError('Beam comparison protocol changed; use a new output version')
    else:
        output.mkdir(parents=True, exist_ok=False)
        physical.base.save_json(output/'protocol.json', protocol)
    torch.set_num_threads(1)
    prior_cwd = Path.cwd()
    count = 0
    try:
        os.chdir(output)
        for event in protocol['events']:
            name = event['source'].replace('_', '')+'_'+event['release']
            for height in protocol['heights_m']:
                if args.limit is not None and count >= args.limit:
                    return
                count += 1
                destination = output/name/f'height_{height:.1f}'
                if (destination/'result.json').exists():
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                print('START_BEAM_COMPARISON', name, height, flush=True)
                result = fit_event(event, height, protocol)
                physical.base.save_json(destination/'result.json', result)
                print('DONE_BEAM_COMPARISON', name, height, flush=True)
    finally:
        os.chdir(prior_cwd)


if __name__ == '__main__':
    main()
