"""Audit DLQ author-style wind; optional IA-only fixed two-event pilot."""

import argparse
import gc
import hashlib
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT, OUT, AUDIT, STUDY, read, save, JointBounds
from ia_author_wind import author_wind_table
from ia_metec_point_adapter import configure, xy_from_lon_lat, window_mean_release
from paper_beam_joint_baselines import BeamJointResidual
from paper_q_model_selection import fit_profiled_q_models

DEST = STUDY / 'ia_author_wind_pilot_v2_20261028'


def run(pilot=False):
    torch.set_num_threads(1)
    for name, digest in read(ROOT / 'paper_experiments/nls_baseline_frozen.json')['sha256_lf'].items():
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('NLS baseline changed: ' + name)
    parent = read(OUT / 'protocol.json')
    cases = read(STUDY / 'ia_location_capacity_pilot_20261028/protocol.json')['public'][:2]
    sources = [Path(__file__), ROOT / 'paper_experiments/ia_author_wind.py',
               ROOT / 'paper_experiments/ia_metec_point_adapter.py',
               ROOT / 'paper_experiments/paper_q_model_selection.py',
               ROOT / 'pinn_source/reduced_trust_region.py',
               AUDIT / 'raw/code/MAIN_1_simulate.R']
    protocol = dict(cases=cases, audit_cases=parent['event_ids'],
        selection='Same two predeclared public development events, no truth-based selection',
        wind='Median speed plus component medians of unit coming-from direction; full-clock short gap fill <=5',
        duplicate_records='First record in original CSV order, matching MAIN_1 lines142-145',
        reference='DLQ d289f322 MAIN_1_simulate.R lines165-195; formula parity tested, R not executed',
        missing='No fallback for undefined aggregate; reject pilot if its event wind is nonfinite',
        fit='Original signed concentrations/q20, Q1/3/5 BIC, original two starts/bounds/cap, nfev250',
        nls='Original stored results and inputs untouched; not executed',
        interpretation='Meteorological preprocessing full-method comparison, not isolated elimination effect',
        orders=parent['numerical_orders'],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST / 'protocol.json').exists() and read(DEST / 'protocol.json') != protocol:
        raise ValueError('Preserve existing author-wind protocol')
    save(DEST / 'protocol.json', protocol)
    for source in sources:
        target = DEST / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    raw = pd.read_csv(AUDIT / 'raw/input_data/ADED_data_clean_SAMPLE.csv')
    wind = author_wind_table(raw)
    audit = []
    for identifier in parent['event_ids']:
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        new = wind.reindex(pd.to_datetime(data['time_utc'], utc=True)).to_numpy()
        valid = np.isfinite(new).all(axis=1)
        old = data['wind_xy']
        dot = np.sum(new * old, axis=1)
        norms = np.linalg.norm(new, axis=1) * np.linalg.norm(old, axis=1)
        directional = valid & (norms > 1e-12)
        angle = np.rad2deg(np.arccos(np.clip(dot[directional] / norms[directional], -1, 1)))
        audit.append(dict(event_id=identifier, undefined_minutes=int((~valid).sum()),
            rms_vector_difference_m_s=float(np.sqrt(np.mean((new[valid] - old[valid]) ** 2))),
            mean_direction_difference_deg=float(angle.mean()) if len(angle) else None,
            max_direction_difference_deg=float(angle.max()) if len(angle) else None,
            original_mean_speed_m_s=float(np.linalg.norm(old, axis=1).mean()),
            author_mean_speed_m_s=float(np.linalg.norm(new[valid], axis=1).mean())))
        save(DEST / identifier / 'wind_audit.json', dict(clock=data['time_utc'],
            author_wind_xy=new, original_wind_xy=old, **audit[-1]))
    pd.DataFrame(audit).to_csv(DEST / 'wind_audit.csv', index=False, encoding='utf-8-sig')
    print('AUTHOR_WIND_AUDIT', pd.DataFrame(audit).to_string(index=False), flush=True)
    if not pilot:
        return
    rows = []
    for identifier in cases:
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        meta = read(OUT / 'inputs' / identifier / 'metadata.json')
        changed = {key: value.copy() for key, value in data.items()}
        changed['wind_xy'] = wind.reindex(pd.to_datetime(data['time_utc'], utc=True)).to_numpy()
        if not np.isfinite(changed['wind_xy']).all():
            raise ValueError('Undefined author-style wind: ' + identifier)
        for key in data:
            if key != 'wind_xy':
                np.testing.assert_array_equal(changed[key], data[key])
        output = DEST / identifier / 'ia_fit.json'
        if output.exists():
            fit = read(output)
        else:
            print('AUTHOR_WIND_FIT_START', identifier, flush=True)
            orders = protocol['orders']
            model, xyt, basis, response, observed = configure(changed, meta['source_height_m'], device='cuda',
                time_order=orders['time'], release_order=orders['release'], history_minutes=orders['history_minutes'])
            initial = read(OUT / 'fits' / identifier / 'initialization.json')
            objective = BeamJointResidual(response, model, 0., xyt, basis, observed, JointBounds(**initial['bounds']))
            fit = fit_profiled_q_models(objective, [np.array(start) for start in initial['starts']], max_nfev=250)
            fit.update(wind_mode='author_component_medians', nls_called=False)
            save(output, fit)
            np.savez_compressed(DEST / identifier / 'ia_input.npz', **changed)
            del objective, response, model
            gc.collect()
            torch.cuda.empty_cache()
        # Source truth is first accessed after preprocessing and optimization.
        truth = read(OUT / 'inputs' / identifier / 'truth.json')
        xy = xy_from_lon_lat([truth['lon']], [truth['lat']], meta['origin'])[0]
        for method, item in [('ia_author_wind', fit),
                            ('original_ia', read(OUT / 'fits' / identifier / 'ia_vpdi.json')),
                            ('frozen_nls', read(OUT / 'fits' / identifier / 'joint_nls_trf.json'))]:
            rows.append(dict(event_id=identifier, method=method,
                localization_error_m=float(np.linalg.norm(np.array(item['source_norm']) * 100 - xy)),
                q_kg_h=3600 * window_mean_release(item['coefficients'], len(data['concentration']) * 60., data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'], rmse_ppm=float(np.sqrt(item['mse'])),
                converged=item.get('converged'), q_basis_count=item.get('q_basis_count', 5)))
        pd.DataFrame(rows).to_csv(DEST / 'comparison.csv', index=False, encoding='utf-8-sig')
        print('AUTHOR_WIND_FIT_DONE', identifier, rows[-3], flush=True)
    save(DEST / 'status.json', dict(pilot_complete=True, nls_called=False,
        numerical_replay_pending=True, production_promoted=False, full_goal_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    run(parser.parse_args().pilot)
