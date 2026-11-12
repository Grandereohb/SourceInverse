"""IA-only two-event test of a wind-calibrated 3D incompressible closure."""

import gc
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT, OUT, STUDY, read, save, JointBounds
from ia_3d_affine_adapter import configure_3d_affine
from ia_3d_affine_transport import general_step
from ia_metec_point_adapter import xy_from_lon_lat, window_mean_release
from paper_beam_joint_baselines import BeamJointResidual
from paper_q_model_selection import fit_profiled_q_models

DEST = STUDY / 'ia_3d_affine_pilot_20261028'
WIND = STUDY / 'ia_horizontal_divergence_audit_20261028'


def flow_preflight(velocity, gradient, wind):
    largest = 0.
    for b, a, vector in zip(velocity, gradient, wind):
        norm = np.linalg.norm(vector)
        direction = vector / norm if norm else np.array([1., 0.])
        low = general_step(a, b, np.array([60.]), direction, order=16)
        high = general_step(a, b, np.array([60.]), direction, order=32)
        # Compare covariance Gramians rather than arbitrary QR signs.
        for index in [1, 2, 3]:
            x, y = low[index], high[index]
            if index > 1:
                x, y = x @ x.swapaxes(-1, -2), y @ y.swapaxes(-1, -2)
            error = float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))
            largest = max(largest, error)
    if largest > 1e-8:
        raise FloatingPointError('Local affine quadrature needs refinement before inversion')
    return dict(max_step_16_vs_32_relative_error=largest,
        scope='Local constant-minute quadrature only, not an end-to-end long-history replay')


def run():
    torch.set_num_threads(1)
    for name, digest in read(ROOT / 'paper_experiments/nls_baseline_frozen.json')['sha256_lf'].items():
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Frozen NLS changed: ' + name)
    cases = read(STUDY / 'ia_location_capacity_pilot_20261028/protocol.json')['public'][:2]
    parent = read(OUT / 'protocol.json')
    sources = [Path(__file__), *[ROOT / 'paper_experiments' / name for name in
        ['ia_3d_affine_adapter.py', 'ia_3d_affine_transport.py', 'ia_horizontal_divergence.py',
         'ia_metec_point_adapter.py', 'ia_physical_beam_adapter.py', 'paper_q_model_selection.py']],
         ROOT / 'pinn_source/reduced_trust_region.py']
    protocol = dict(cases=cases, orders=parent['numerical_orders'], flow_order=16, max_nfev=250,
        selection='Same fixed two development events; all observations, no source-based selection',
        physics='u=b+Ah*x; w=-trace(Ah)*z; full Gaussian covariance and advected source height, reflecting ground',
        wind='Saturated six-parameter fit to three measured wind vectors only; no concentration calibration',
        caveat='Vertical motion is a closure assumption, not observed; horizontal fit has zero residual degrees of freedom',
        unchanged='Raw input/q20/signed rows, original two starts, source/D/Q bounds and Q1/3/5 BIC',
        nls='Frozen stored results only; not executed or modified',
        interpretation='Full-method physical-model pilot, not a pure variable-projection gain',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        wind_sha256={case: hashlib.sha256((WIND / case / 'wind_fit.json').read_bytes()).hexdigest() for case in cases})
    if (DEST / 'protocol.json').exists() and read(DEST / 'protocol.json') != protocol:
        raise ValueError('Preserve previous 3D pilot protocol')
    save(DEST / 'protocol.json', protocol)
    for source in sources:
        target = DEST / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    rows = []
    for identifier in cases:
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        meta = read(OUT / 'inputs' / identifier / 'metadata.json')
        folder = DEST / identifier
        fit_path = folder / 'ia_fit.json'
        if fit_path.exists():
            fit = read(fit_path)
        else:
            record = read(WIND / identifier / 'wind_fit.json')
            velocity, gradient = np.asarray(record['intercept']), np.asarray(record['gradient'])
            save(folder / 'flow_preflight.json', flow_preflight(velocity, gradient, data['wind_xy']))
            print('THREE_D_PLAN_START', identifier, flush=True)
            begun = time.perf_counter()
            orders = protocol['orders']
            model, xyt, basis, response, observed = configure_3d_affine(data, meta['source_height_m'],
                velocity, gradient, device='cuda', time_order=orders['time'], release_order=orders['release'],
                history_minutes=orders['history_minutes'], flow_order=16)
            plan_seconds = time.perf_counter() - begun
            print('THREE_D_FIT_START', identifier, 'plan_seconds', plan_seconds, flush=True)
            initial = read(OUT / 'fits' / identifier / 'initialization.json')
            objective = BeamJointResidual(response, model, 0., xyt, basis, observed, JointBounds(**initial['bounds']))
            fit = fit_profiled_q_models(objective, [np.asarray(start) for start in initial['starts']], max_nfev=250)
            fit.update(plan_seconds=plan_seconds, nls_called=False,
                vertical_wind_measured=False, numerical_replay_pending=True)
            save(fit_path, fit)
            del model, response, objective
            gc.collect()
            torch.cuda.empty_cache()
        truth = read(OUT / 'inputs' / identifier / 'truth.json')
        true_xy = xy_from_lon_lat([truth['lon']], [truth['lat']], meta['origin'])[0]
        for method, item in [('ia_3d_affine', fit),
                            ('original_ia', read(OUT / 'fits' / identifier / 'ia_vpdi.json')),
                            ('frozen_nls', read(OUT / 'fits' / identifier / 'joint_nls_trf.json'))]:
            rows.append(dict(event_id=identifier, method=method,
                localization_error_m=float(np.linalg.norm(np.asarray(item['source_norm']) * 100 - true_xy)),
                q_kg_h=3600 * window_mean_release(item['coefficients'], len(data['concentration']) * 60., data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'], rmse_ppm=float(np.sqrt(item['mse'])),
                q_basis_count=item.get('q_basis_count', 5), converged=item.get('converged')))
        pd.DataFrame(rows).to_csv(DEST / 'comparison.csv', index=False, encoding='utf-8-sig')
        print('THREE_D_DONE', rows[-3], flush=True)
    save(DEST / 'status.json', dict(pilot_complete=True, production_promoted=False,
        numerical_replay_pending=True, nls_called=False, full_goal_achieved=False))


if __name__ == '__main__':
    run()
