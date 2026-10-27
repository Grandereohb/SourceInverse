"""One frozen public event with unknown release history; no outcome-based retries."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_paper_controlled_release as physical
from paper_joint_baselines import JointBounds
from paper_beam_joint_baselines import BeamJointResidual
from paper_q_model_selection import fit_profiled_q_models

STUDY = Path(__file__).resolve().parents[1]/'result/paper_experiments_20261017'
OUT = STUDY/'public_unknown_release_pilot_20261027'
EVENT = 'source1_rel03'
HEIGHT = .3


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    physical.base.save_json(path, value)


def mean_release(coefficients):
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite source coefficients required')
    return float(values[0] if len(values) == 1 else np.trapezoid(values, dx=1/(len(values)-1)))


def make_bounds_and_starts(record):
    declared = record['bounds']
    cap = float(declared['upper'][4])
    bounds = JointBounds(np.r_[declared['lower'][:4], np.zeros(5)],
                         np.r_[declared['upper'][:4], np.full(5, cap)])
    # Flat starting curves are observation-only initial guesses, not a fixed release prior.
    starts = [np.r_[s['source_x'], s['source_y'], s['diffusion_perp_norm'],
        s['diffusion_parallel_norm']-s['diffusion_perp_norm'], np.full(5, s['coefficients'][0])]
        for s in record['common_starts']]
    return bounds, starts


def run(device):
    torch.set_num_threads(1)
    prep = STUDY/'controlled_release_preparation_v2'/EVENT
    legacy_path = STUDY/'paper_updated_20261026/public'/EVENT/'height_0.3/result.json'
    legacy = read(legacy_path)
    paths = [prep/name for name in ('observations.csv', 'raw_axis_weather.csv', 'beams.csv')]
    paths += [legacy_path, Path(__file__), Path(__file__).with_name('paper_q_model_selection.py'),
              Path(__file__).with_name('paper_reduced_solver.py'),
              Path(__file__).with_name('paper_beam_joint_baselines.py'),
              Path(__file__).with_name('paper_physical_beam_adapter.py')]
    protocol = dict(event=EVENT, height_m=HEIGHT,
        selection='first frozen public event, first declared height; no outcome selection',
        release_prior='unknown for both methods; true rate and location accessed only after fitting',
        methods=dict(ia_unknown='profiled nonnegative Q with BIC counts1/3/5',
                     joint_nls_unknown='joint constrained TRF with fixed5 piecewise-linear Q coefficients'),
        q_capacity_difference='complete-method comparison, NOT isolated variable-elimination evidence',
        q_count=5, max_nfev_per_start_per_model=250, common_starts=2,
        source_bounds='original finite domain, not tightened using truth',
        initialization='original observation-only geometry/D starts, repeated scalar Q; NLS Q initialized by bounded LS at each start only',
        all_valid_observations=True, no_crossfit=True, device=device, dtype='float64',
        fit_orders=dict(time=16, release=16), replay_orders=dict(time=32, release=32),
        replay_relative_threshold=.01, no_tuning_or_retries=True,
        budget_note='IA explores three capacities; NLS one fixed capacity, not equal total compute',
        bic_note='working Gaussian criterion; serially correlated observations prevent automatic Bayesian interpretation',
        source_sha256={str(p.relative_to(STUDY.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    frozen = OUT/'protocol.json'
    if frozen.exists() and read(frozen) != protocol:
        raise ValueError('Pilot protocol changed; do not silently resume')
    save(frozen, protocol)
    model, xyt, basis, response, observed, _ = physical.configure(HEIGHT, device=device,
        dtype=torch.float64, q_count=5, time_order=16, release_order=16,
        beam_solver='analytic', data_dir=prep)
    bounds, starts = make_bounds_and_starts(legacy)
    for name, solver, counts in [('ia_unknown', 'profiled', (1,3,5)),
                                 ('joint_nls_unknown', 'joint_nls', (5,))]:
        target = OUT/(name+'.json')
        if not target.exists():
            objective = BeamJointResidual(response, model, 0., xyt, basis, observed, bounds)
            fit = fit_profiled_q_models(objective, starts, counts=counts, solver=solver, max_nfev=250)
            save(target, fit)
        print('COMPLETED_UNKNOWN_RELEASE', name, flush=True)
    # Truth and independent mass records enter only this post-fit evaluation block.
    truth = read(STUDY/'public_data/author_truth_values.json')['values_as_stored']['source_1_location'][:2]
    reference = read(prep/'cylinder_reference_v2.json')['ols_emission_reference_kg_h']
    model32, xyt32, basis32, response32, observed32, _ = physical.configure(HEIGHT,
        device=device, dtype=torch.float64, q_count=5, time_order=32, release_order=32,
        beam_solver='analytic', data_dir=prep)
    np.testing.assert_array_equal(observed32, observed)
    rows = []
    for name in ('ia_unknown', 'joint_nls_unknown'):
        fit = read(OUT/(name+'.json'))
        theta = torch.tensor([*fit['source_norm'], fit['diffusion_norm'][1],
            fit['diffusion_norm'][0]-fit['diffusion_norm'][1]], dtype=torch.float64, device=device)
        with torch.no_grad():
            matrix = response32(model32, 0., xyt32, basis32, physical_parameters=theta).cpu().numpy()
        predicted32 = matrix@np.asarray(fit['coefficients'])
        change = np.linalg.norm(predicted32-fit['prediction'])/max(np.linalg.norm(predicted32), np.finfo(float).tiny)
        q_mean = 3600*mean_release(fit['coefficients'])
        rows.append(dict(method=name, q_count=fit['q_basis_count'],
            source_x_m=100*fit['source_norm'][0], source_y_m=100*fit['source_norm'][1],
            localization_error_m=float(np.linalg.norm(100*np.asarray(fit['source_norm'])-truth)),
            mean_q_kg_h=q_mean, measured_reference_q_kg_h=reference,
            q_mean_relative_error=abs(q_mean-reference)/reference,
            observed_rmse_ppm=float(np.sqrt(np.mean((np.asarray(fit['prediction'])-observed)**2))),
            seconds=fit['seconds'], converged=fit['converged'],
            replay_relative_change=float(change), replay_passed=bool(change <= .01)))
    for name in ('ia_vpdi', 'joint_nls_trf'):
        fit = legacy['methods'][name]
        rows.append(dict(method='known_constant_'+name, q_count=1,
            localization_error_m=fit['localization_error_m'], mean_q_kg_h=fit['mean_q_kg_h'],
            measured_reference_q_kg_h=reference,
            q_mean_relative_error=abs(fit['mean_q_kg_h']-reference)/reference,
            observed_rmse_ppm=fit['observed_rmse_ppm'], seconds=fit['seconds'],
            note='historical known-release-prior context, not ranked with unknown-release methods'))
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv', index=False)
    save(OUT/'status.json', dict(final=True, pilot_only=True, event_count=1, heights_count=1,
        new_fits=2, replay_passed=all(r['replay_passed'] for r in rows[:2]),
        true_release_history_not_used_in_fit=True, observations=len(observed),
        eval_reference='independent mass-derived mean, not a time-resolved Q truth'))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    run(parser.parse_args().device)
