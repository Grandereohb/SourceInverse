"""Frozen full public cohort; fitting requires both pilot numerical gates."""

import argparse
import gc
import hashlib
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT, OUT, STUDY, read, save, JointBounds
from run_ia_3d_affine_pilot import DEST as PILOT, WIND, flow_preflight
from refine_ia_3d_affine_numerics import DEST as REPAIR
from ia_3d_affine_adapter import configure_3d_affine
from ia_metec_point_adapter import xy_from_lon_lat, window_mean_release
from paper_beam_joint_baselines import BeamJointResidual
from paper_q_model_selection import fit_profiled_q_models
from analyze_affine_accuracy import intervals

DEST = STUDY / 'ia_3d_affine_full_20261028'


def run(fit_cases=False):
    torch.set_num_threads(1)
    parent = read(OUT / 'protocol.json')
    pilot = read(PILOT / 'protocol.json')
    repair = read(REPAIR / 'protocol.json')
    for name, digest in pilot['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pilot model changed before cohort evaluation: ' + name)
    for name, digest in read(ROOT / 'paper_experiments/nls_baseline_frozen.json')['sha256_lf'].items():
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Frozen NLS changed: ' + name)
    cases = parent['event_ids']
    sources = [Path(__file__), *[ROOT / name for name in pilot['source_sha256']]]
    input_paths = [OUT / 'inputs' / case / name for case in cases for name in ['input.npz', 'metadata.json']]
    input_paths += [OUT / 'fits' / case / 'initialization.json' for case in cases]
    input_paths += [WIND / case / 'wind_fit.json' for case in cases]
    protocol = dict(cases=cases, orders=parent['numerical_orders'], flow_order=16, max_nfev=250,
        selection='All original17 public releases, no exclusions based on outcomes or fit quality',
        observations='All original signed valid residuals/q20, original starts/bounds, Q1/3/5 BIC',
        model='Unchanged pilot 3D incompressible affine closure from independent wind-only calibration',
        pilot_reuse='Reuse both pilot estimates, refining only those failing the preset numerical gate',
        gate='No fitting until pilot numerical repair status is complete and numerically_passed',
        numerical_policy='All new cases need32/32-history40-flow32 replay before acceptance; repair failures, not thresholds',
        nls='7563376 stored baseline only, not executed or modified',
        inference_scope='Exploratory development cohort after mechanism screening; not independent confirmatory evidence',
        bootstrap=dict(seed=20261028, repetitions=10000, cluster='equipment_group'),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
        pilot_protocol_sha256=hashlib.sha256((PILOT / 'protocol.json').read_bytes()).hexdigest(),
        repair_protocol_sha256=hashlib.sha256((REPAIR / 'protocol.json').read_bytes()).hexdigest())
    if (DEST / 'protocol.json').exists() and read(DEST / 'protocol.json') != protocol:
        raise ValueError('Preserve previous complete-cohort protocol')
    save(DEST / 'protocol.json', protocol)
    for source in sources:
        target = DEST / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    if not fit_cases:
        print('THREE_D_FULL_PLAN_SAVED', len(cases), 'events; no fitting started', flush=True)
        return
    status_path = REPAIR / 'status.json'
    if not status_path.exists() or not read(status_path).get('numerically_passed', False):
        raise ValueError('Pilot numerical acceptance remains incomplete; do not expand')
    for index, identifier in enumerate(cases):
        folder = DEST / identifier
        path = folder / 'ia_fit.json'
        if path.exists():
            continue
        if identifier in pilot['cases']:
            origin = REPAIR if identifier in repair['refine_events'] else PILOT
            previous = origin / identifier / 'ia_fit.json'
            value = read(previous)
            value.update(reused_from=str(previous.relative_to(ROOT)),
                         reused_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(), numerical_replay_pending=False)
            save(path, value)
            continue
        print('THREE_D_FULL_START', index + 1, '/', len(cases), identifier, flush=True)
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        meta = read(OUT / 'inputs' / identifier / 'metadata.json')
        wind = read(WIND / identifier / 'wind_fit.json')
        velocity, gradient = np.asarray(wind['intercept']), np.asarray(wind['gradient'])
        save(folder / 'flow_preflight.json', flow_preflight(velocity, gradient, data['wind_xy']))
        orders = protocol['orders']
        model, xyt, basis, response, observed = configure_3d_affine(data, meta['source_height_m'],
            velocity, gradient, device='cuda', time_order=orders['time'], release_order=orders['release'],
            history_minutes=orders['history_minutes'], flow_order=16)
        initial = read(OUT / 'fits' / identifier / 'initialization.json')
        objective = BeamJointResidual(response, model, 0., xyt, basis, observed, JointBounds(**initial['bounds']))
        value = fit_profiled_q_models(objective, [np.asarray(start) for start in initial['starts']], max_nfev=250)
        value.update(nls_called=False, numerical_replay_pending=True, vertical_wind_measured=False)
        save(path, value)
        del response, model, objective
        gc.collect()
        torch.cuda.empty_cache()
        print('THREE_D_FULL_FIT_SAVED', identifier, flush=True)
    rows = []
    two_d_repair = STUDY / 'ia_affine_numerical_repair_20261028'
    two_d_refined = {r['event_id'] for r in read(two_d_repair / 'status.json')['numerical_rows'] if r['refined']}
    for identifier in cases:
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        meta = read(OUT / 'inputs' / identifier / 'metadata.json')
        truth = read(OUT / 'inputs' / identifier / 'truth.json')
        true_xy = xy_from_lon_lat([truth['lon']], [truth['lat']], meta['origin'])[0]
        two_d = (two_d_repair / identifier / 'ia_fit.json') if identifier in two_d_refined else (
            STUDY / 'ia_affine_wind_sqrt_full_20261028' / identifier / 'ia_affine_wind.json')
        fits = [('ia_3d_affine', read(DEST / identifier / 'ia_fit.json')),
                ('validated_2d_ia', read(two_d)),
                ('original_ia', read(OUT / 'fits' / identifier / 'ia_vpdi.json')),
                ('frozen_nls', read(OUT / 'fits' / identifier / 'joint_nls_trf.json'))]
        for method, value in fits:
            q = 3600 * window_mean_release(value['coefficients'], len(data['concentration']) * 60., data['release_window_s'])
            rows.append(dict(event_id=identifier, method=method,
                localization_error_m=float(np.linalg.norm(np.asarray(value['source_norm']) * 100 - true_xy)),
                q_kg_h=q, true_q_kg_h=truth['mean_q_kg_h'], q_relative_error=abs(q / truth['mean_q_kg_h'] - 1),
                rmse_ppm=float(np.sqrt(value['mse'])), converged=value.get('converged'),
                equipment_group=truth['equipment_group']))
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST / 'comparison.csv', index=False, encoding='utf-8-sig')
    frame.groupby('method').localization_error_m.agg(['mean', 'median', 'count']).to_csv(
        DEST / 'summary.csv', encoding='utf-8-sig')
    wide = frame.pivot(index='event_id', columns='method', values='localization_error_m')
    if len(wide) != len(cases) or wide.isna().any().any():
        raise ValueError('Do not summarize an incomplete favourable subset')
    groups = [read(OUT / 'inputs' / case / 'truth.json')['equipment_group'] for case in wide.index]
    paired = intervals(wide.ia_3d_affine - wide.frozen_nls, groups, np.random.default_rng(20261028))
    save(DEST / 'paired_accuracy.json', dict(comparison=paired,
        scope='Exploratory full-method physics comparison; no isolated solver claim or independent-site confirmation'))
    pending = [case for case in cases if case not in pilot['cases']]
    save(DEST / 'status.json', dict(complete_events=len(cases), numerical_replay_pending=pending,
        numerical_ready=False, production_promoted=False, nls_called=False, full_goal_achieved=False))
    print('THREE_D_FULL_SUMMARY', paired, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', action='store_true')
    run(parser.parse_args().fit)
