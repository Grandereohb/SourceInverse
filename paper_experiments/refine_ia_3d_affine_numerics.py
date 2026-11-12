"""Refine only IA pilot cases failing the existing numerical replay gate."""

import gc
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT, OUT, STUDY, read, save, JointBounds
from run_ia_3d_affine_pilot import DEST as PILOT, WIND
from ia_3d_affine_adapter import configure_3d_affine
from ia_metec_point_adapter import xy_from_lon_lat, window_mean_release
from paper_beam_joint_baselines import BeamJointResidual
from paper_location_capacity import capacity_response
from reduced_trust_region import fit_reduced_trust_region

DEST = STUDY / 'ia_3d_affine_repair_20261028'


def run():
    torch.set_num_threads(1)
    parent = read(PILOT / 'protocol.json')
    replay = read(PILOT / 'replay_summary.json')
    for name, digest in parent['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pilot operator changed: ' + name)
    for name, digest in read(ROOT / 'paper_experiments/nls_baseline_frozen.json')['sha256_lf'].items():
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Frozen NLS changed: ' + name)
    failed = [row['event_id'] for row in replay['rows'] if not row['passed']]
    protocol = dict(cases=parent['cases'], refine_events=failed,
        trigger='Existing 1% fixed-parameter response replay gate; source truth not used',
        fit_orders=dict(time=32, release=32, history_minutes=40, flow=32),
        replay_orders=dict(time=64, release=64, history_minutes=60, flow=32),
        max_nfev=100, batch_minutes=8, relative_threshold=.01,
        fit='Same selected Q capacity, original bounds/SSE/wind/height, one previous IA physical optimum start',
        reuse='Keep already numerically passed pilot cases without refitting or source-truth selection',
        nls_called=False, pilot_protocol_sha256=hashlib.sha256((PILOT / 'protocol.json').read_bytes()).hexdigest(),
        replay_summary_sha256=hashlib.sha256((PILOT / 'replay_summary.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (DEST / 'protocol.json').exists() and read(DEST / 'protocol.json') != protocol:
        raise ValueError('Preserve previous 3D numerical repair')
    save(DEST / 'protocol.json', protocol)
    for identifier in failed:
        folder = DEST / identifier
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        meta = read(OUT / 'inputs' / identifier / 'metadata.json')
        old = read(PILOT / identifier / 'ia_fit.json')
        wind = read(WIND / identifier / 'wind_fit.json')
        velocity, gradient = np.asarray(wind['intercept']), np.asarray(wind['gradient'])
        path = folder / 'ia_fit.json'
        if path.exists():
            fit = read(path)
        else:
            save(folder / 'progress.json', dict(stage='building_high_order_fit_plan', complete=False))
            print('THREE_D_REFINE_PLAN', identifier, flush=True)
            model, xyt, basis, response, observed = configure_3d_affine(data, meta['source_height_m'],
                velocity, gradient, device='cuda', time_order=32, release_order=32,
                history_minutes=40, flow_order=32)
            print('THREE_D_REFINE_FIT', identifier, flush=True)
            save(folder / 'progress.json', dict(stage='fitting', complete=False))
            initial = read(OUT / 'fits' / identifier / 'initialization.json')
            objective = BeamJointResidual(response, model, 0., xyt, basis, observed, JointBounds(**initial['bounds']))
            function, mapping = capacity_response(objective, old['q_basis_count'])
            start = np.r_[old['source_norm'], old['diffusion_norm'][1],
                          old['diffusion_norm'][0] - old['diffusion_norm'][1]]
            fit = fit_reduced_trust_region(function, observed, [start],
                objective.bounds.lower[:4], objective.bounds.upper[:4],
                q_upper_bound=float(objective.bounds.upper[4]), max_nfev=100)
            theta = fit.pop('theta')
            native = np.asarray(fit['coefficients']).copy()
            fit.update(source_norm=theta[:2], diffusion_norm=[theta[2] + theta[3], theta[2]],
                native_coefficients=native, coefficients=mapping @ native,
                q_basis_count=old['q_basis_count'], mse=fit['sse'] / len(observed),
                nls_called=False, numerical_replay_pending=True)
            save(path, fit)
            del model, response, objective, function
            gc.collect()
            torch.cuda.empty_cache()
            print('THREE_D_REFINE_FIT_DONE', identifier, fit['sse'], flush=True)
        fit_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        completed = folder / 'replay.json'
        if not completed.exists():
            checkpoint = folder / 'replay_checkpoint.npz'
            first, prediction = 0, np.empty(0)
            if checkpoint.exists():
                with np.load(checkpoint) as saved:
                    if str(saved['fit_sha256']) != fit_hash:
                        raise ValueError('Checkpoint from another refined fit')
                    first, prediction = int(saved['minutes_done']), saved['prediction'].copy()
                if len(prediction) != np.isfinite(data['concentration'][:first]).sum():
                    raise ValueError('Replay checkpoint prefix does not match observations')
            for start in range(first, len(data['concentration']), 8):
                end = min(start + 8, len(data['concentration']))
                model, xyt, basis, response, observed = configure_3d_affine(data, meta['source_height_m'],
                    velocity, gradient, device='cuda', time_order=64, release_order=64,
                    history_minutes=60, flow_order=32, observation_minutes=np.arange(start, end))
                theta = model.xs.new_tensor([*fit['source_norm'], fit['diffusion_norm'][1],
                    fit['diffusion_norm'][0] - fit['diffusion_norm'][1]])
                with torch.no_grad():
                    value = (response(model, 0., xyt, basis, physical_parameters=theta) @
                        model.xs.new_tensor(fit['coefficients'])).cpu().numpy()
                if not np.isfinite(value).all() or len(value) != len(observed):
                    raise FloatingPointError('Invalid refined replay')
                prediction = np.r_[prediction, value]
                temporary = checkpoint.with_name('replay_checkpoint.tmp.npz')
                np.savez_compressed(temporary, minutes_done=end, prediction=prediction, fit_sha256=fit_hash)
                temporary.replace(checkpoint)
                save(folder / 'progress.json', dict(stage='high_order_replay', minutes_done=end,
                    total_minutes=len(data['concentration']), complete=False))
                print('THREE_D_REFINE_REPLAY', identifier, end, '/', len(data['concentration']), flush=True)
                del model, response
                gc.collect()
                torch.cuda.empty_cache()
            relative = float(np.linalg.norm(prediction - fit['prediction']) / max(np.linalg.norm(prediction), 1e-30))
            save(completed, dict(relative_l2=relative, passed=relative <= .01,
                prediction=prediction, fit_sha256=fit_hash))
            checkpoint.unlink(missing_ok=True)
        check = read(completed)
        save(folder / 'progress.json', dict(stage='complete', complete=True, numerically_passed=check['passed']))
        print('THREE_D_REFINE_DONE', identifier, check['relative_l2'], flush=True)
    rows, checks = [], []
    for identifier in protocol['cases']:
        refined = identifier in failed
        fit = read(DEST / identifier / 'ia_fit.json') if refined else read(PILOT / identifier / 'ia_fit.json')
        check = read(DEST / identifier / 'replay.json') if refined else next(r for r in replay['rows'] if r['event_id'] == identifier)
        checks.append(dict(event_id=identifier, refined=refined, passed=check['passed'], relative_l2=check['relative_l2']))
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        meta = read(OUT / 'inputs' / identifier / 'metadata.json')
        truth = read(OUT / 'inputs' / identifier / 'truth.json')
        xy = xy_from_lon_lat([truth['lon']], [truth['lat']], meta['origin'])[0]
        for method, item in [('ia_3d_refined', fit), ('ia_3d_previous', read(PILOT / identifier / 'ia_fit.json')),
                            ('original_ia', read(OUT / 'fits' / identifier / 'ia_vpdi.json')),
                            ('frozen_nls', read(OUT / 'fits' / identifier / 'joint_nls_trf.json'))]:
            rows.append(dict(event_id=identifier, method=method, refined=refined,
                localization_error_m=float(np.linalg.norm(np.asarray(item['source_norm']) * 100 - xy)),
                q_kg_h=3600 * window_mean_release(item['coefficients'], len(data['concentration']) * 60., data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'], converged=item.get('converged')))
    pd.DataFrame(rows).to_csv(DEST / 'comparison.csv', index=False, encoding='utf-8-sig')
    save(DEST / 'status.json', dict(complete=True, numerical_rows=checks,
        numerically_passed=all(row['passed'] for row in checks), production_promoted=False,
        nls_called=False, full_goal_achieved=False))


if __name__ == '__main__':
    run()
