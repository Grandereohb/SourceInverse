"""Frozen-protocol real-release evaluation; never imported by production.

Both disputed source heights are evaluated. Truth is read only after fitting.
Use --smoke for a tiny plumbing test, not an accuracy result.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import run_paper_controlled_release as physical
from paper_steady_beam_baseline import fit_steady_baseline

base = physical.base
ROOT = physical.ROOT
DATA = ROOT/'controlled_release_preparation_v2'


class MappedResponse:
    """Constant Q is the sum of the same five hat responses, not new physics."""

    def __init__(self, response, basis, constant):
        self.response, self.basis, self.constant = response, basis, constant

    def __call__(self, model, sigma, xyt, basis, track_gradients=False):
        if not torch.equal(basis, self.basis):
            raise ValueError('Unexpected Q basis for fixed physical response mapping')
        matrix = self.response(model, sigma, xyt, self.response.basis,
                               track_gradients=track_gradients)
        return matrix.sum(1, keepdim=True) if self.constant else matrix


def mean_hat_q(coefficients, duration_minutes):
    coefficients = np.asarray(coefficients, dtype=float)
    if (coefficients.ndim != 1 or not len(coefficients) or not np.isfinite(coefficients).all()
            or not np.isfinite(duration_minutes) or duration_minutes <= 0):
        raise ValueError('Finite coefficients and a positive duration are required')
    if len(coefficients) == 1:
        return float(coefficients[0])
    knots = np.linspace(0., duration_minutes, len(coefficients))
    return float(np.trapezoid(coefficients, knots)/duration_minutes)


def prepare_protocol(output, smoke, device, pilot=False):
    catalog = json.loads((DATA/'catalog.json').read_text(encoding='utf-8'))
    events = [v['event'] for v in catalog['events'] if v['status'] == 'prepared']
    events.sort(key=lambda v: (v['date'], v['start']))
    if smoke or pilot:
        events = events[:1]
    protocol = dict(scope='plumbing only' if smoke else ('first chronological event pilot, not final evaluation' if pilot
                    else 'controlled-release fixed-budget evaluation'),
        events=events, heights_m=[.3, 1.6], device=device, dtype='float32',
        methods=['transient_constant_q', 'transient_hat5', 'stationary_constant_q'],
        source_bounds_m=[0., 110.], grid_points_per_axis=2 if smoke else 7,
        diffusion_perp_seeds_m2_s=[3.25, 6.5, 13.], anisotropy_seed_ratios=[1., 3.],
        starts=2, start_distance_m=20., adam_steps=2 if smoke else (100 if pilot else 500),
        learning_rate=.02, gradient_clip=10., early_stopping=False,
        analytic_flat_beams=True, time_order=16, release_order=16,
        replay_time_order=32, replay_release_order=32, replay_relative_tolerance=.01,
        q_units='kg/s', observations='all prepared signed background residuals; unweighted SSE',
        assumptions=['Dz=D_perp', 'reflecting ground', 'no decay or vertical mean wind',
            '2m-square source horizontal variance represented by Gaussian variance1/3m2',
            'minute mean weather; ideal-gas conversion using measured T/P; no humidity correction'],
        truth_policy='coordinates and cylinder rates evaluation-only, read after both transient fits and baseline',
        convergence='fixed budget, NOT an established converged optimum',
        minutes_limit=3 if smoke else None)
    sources = [Path(__file__), Path(physical.__file__),
        Path(__file__).with_name('paper_physical_beam_adapter.py'),
        Path(__file__).with_name('paper_steady_beam_baseline.py'),
        Path(base.__file__), *sorted((base.ROOT/'pinn_source').rglob('*.py'))]
    inputs = [DATA/'catalog.json', ROOT/'controlled_release_adapter_analytic_numerical_v1/gate_result.json',
              ROOT/'controlled_release_adapter_analytic_numerical_v1/float32_precision_gate.json']
    for gate in inputs[1:]:
        if not json.loads(gate.read_text(encoding='utf-8'))['passed']:
            raise ValueError('Physical numerical gates must pass before fitting')
    for event in events:
        folder = DATA/(event['source'].replace('_', '')+'_'+event['release'])
        inputs.extend(folder/name for name in ('observations.csv', 'raw_axis_weather.csv', 'beams.csv', 'manifest.json'))
    protocol['sha256'] = {str(p.relative_to(base.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources+inputs}
    if output.exists():
        previous = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
        if previous != protocol:
            raise ValueError('Protocol or source drift: use a new output version, not mixed results')
    else:
        output.mkdir(parents=True)
        for source in sources:
            target = output/'source_snapshot'/source.relative_to(base.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        base.save_json(output/'protocol.json', protocol)
    return protocol


def fit_event(data_dir, height, protocol, output):
    started = time.perf_counter()
    model, xyt, shared_basis, response, observed, observations = physical.configure(
        height, minutes=protocol['minutes_limit'], device=protocol['device'], dtype=torch.float32,
        time_order=16, release_order=16, beam_solver='analytic', data_dir=data_dir)
    methods = {}
    coarse = {False: [], True: []}
    grid = np.linspace(0., 1.1, protocol['grid_points_per_axis'])
    with torch.no_grad():
        for dq in protocol['diffusion_perp_seeds_m2_s']:
            for ratio in protocol['anisotropy_seed_ratios']:
                dp = dq*ratio + (1e-6 if ratio == 1. else 0.)
                for x in grid:
                    for y in grid:
                        with base.ia.temporary_source_and_diffusion(model, x, y, dp*.006, dq*.006):
                            matrix = response(model, 0., xyt, shared_basis)
                        for constant in (True, False):
                            basis = torch.ones_like(shared_basis[:, :1]) if constant else shared_basis
                            mapped = matrix.sum(1, keepdim=True) if constant else matrix
                            solution = base.vp.solve_nonnegative_q(mapped, observed)
                            coarse[constant].append(base.ia._candidate_from_solution(
                                solution, basis, x, y, dp*.006, dq*.006))
    coarse_seconds = time.perf_counter()-started
    for constant in (True, False):
        name = 'transient_constant_q' if constant else 'transient_hat5'
        basis = torch.ones_like(shared_basis[:, :1]) if constant else shared_basis
        callback = MappedResponse(response, basis, constant)
        candidates = sorted(coarse[constant], key=lambda c: c.profile_score)
        first = candidates[0]
        second = next(c for c in candidates if np.hypot(c.source_x-first.source_x,
                                                       c.source_y-first.source_y) >= .2)
        runs = []
        method_started = time.perf_counter()
        with patch.object(base.ia, 'build_response_matrix', callback):
            for index, candidate in enumerate((first, second)):
                runs.append(base.ia.refine_profiled_candidate(model, 0., xyt, observed, basis,
                    candidate, (0., 1.1), (0., 1.1), steps=protocol['adam_steps'], lr=.02,
                    patience=protocol['adam_steps']+1, min_delta=0., run_id=index))
        if not all(run.success for run in runs):
            raise RuntimeError('Transient refinement failed: '+str([run.message for run in runs]))
        best = min((run.best for run in runs), key=lambda c: c.profile_score)
        fine_model, fine_xyt, fine_basis, fine_response, _, _ = physical.configure(
            height, minutes=protocol['minutes_limit'], device='cpu', dtype=torch.float64,
            time_order=32, release_order=32, beam_solver='analytic', data_dir=data_dir)
        with base.ia.temporary_source_and_diffusion(fine_model, best.source_x, best.source_y,
                best.diffusion_parallel_norm, best.diffusion_perp_norm):
            matrix = fine_response(fine_model, 0., fine_xyt, fine_basis).numpy()
        if constant:
            matrix = matrix.sum(1, keepdims=True)
        replay = matrix@best.coefficients
        relative = float(np.linalg.norm(replay-best.prediction)/max(np.linalg.norm(replay), np.finfo(float).tiny))
        methods[name] = dict(best=asdict(best), runs=[asdict(run) for run in runs],
            refinement_and_replay_seconds=time.perf_counter()-method_started,
            source_m=[best.source_x*100., best.source_y*100.],
            diffusion_m2_s=[best.diffusion_parallel_norm/.006, best.diffusion_perp_norm/.006],
            mean_q_kg_s=mean_hat_q(best.coefficients, len(observations)//7),
            replay_relative_difference=relative, replay_gate_passed=relative <= .01,
            optimizer_convergence='unverified fixed budget')
        predictions = observations[['time', 'beam', 'residual_ppm']].copy()
        predictions['predicted_ppm'] = best.prediction
        predictions['fine_replay_ppm'] = replay
        predictions.to_csv(output/(name+'_predictions.csv'), index=False)
        pd.DataFrame(dict(minutes=model.recurrent_integration_times.detach().cpu().numpy(),
                          q_kg_s=best.q_integration, q_kg_h=best.q_integration*3600.)).to_csv(
                              output/(name+'_q.csv'), index=False)
        base.save_json(output/(name+'.json'), methods[name])
        print('TRANSIENT', data_dir.name, height, name, 'seconds',
              round(time.perf_counter()-method_started, 2), 'replay', relative, flush=True)
    _, weather, beams = physical.inputs(protocol['minutes_limit'], data_dir)
    methods['stationary_constant_q'] = fit_steady_baseline(
        np.column_stack((-weather['WindY_m/s'], weather['WindX_m/s'])),
        beams[['start_x_m', 'start_y_m', 'start_z_m']].to_numpy(),
        beams[['end_x_m', 'end_y_m', 'end_z_m']].to_numpy(), height,
        weather.Temp_K.to_numpy(), weather.Press_Pa.to_numpy(), observed,
        max_evaluations=10 if protocol['minutes_limit'] else 3000)
    result = dict(height_m=height, methods=methods, shared_coarse_seconds=coarse_seconds,
                  response_calls=response.calls, seconds=time.perf_counter()-started,
                  status='plumbing_only' if protocol['minutes_limit'] else 'fitted_pending_evaluation')
    base.save_json(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--pilot', action='store_true', help='First chronological full event,100 updates/start; not final')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261017)
    if args.smoke and args.pilot:
        parser.error('--smoke and --pilot are mutually exclusive')
    output = args.output or ROOT/('controlled_release_fit_smoke_v3' if args.smoke else
        ('controlled_release_fit_pilot_v1' if args.pilot else 'controlled_release_fit_v1'))
    protocol = prepare_protocol(output, args.smoke, args.device, args.pilot)
    for event in protocol['events']:
        key = event['source'].replace('_', '')+'_'+event['release']
        for height in protocol['heights_m']:
            destination = output/key/f'height_{height:.1f}'
            if (destination/'result.json').exists():
                continue
            destination.mkdir(parents=True, exist_ok=True)
            try:
                fit_event(DATA/key, height, protocol, destination)
            except Exception as error:
                base.save_json(destination/'STOPPED.json', dict(type=type(error).__name__, message=str(error),
                    consequence='No fabricated substitute result; remaining jobs not run'))
                raise
    base.save_json(output/'RUN_FINISHED.json', dict(status='fits completed; evaluation/validity audit still required',
        events=len(protocol['events']), heights=2, smoke=args.smoke))


if __name__ == '__main__':
    main()
