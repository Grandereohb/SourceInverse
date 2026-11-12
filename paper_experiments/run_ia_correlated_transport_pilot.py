"""Fixed two-event IA-only pilot of finite turbulent velocity correlation."""

import gc
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT, OUT, STUDY, read, save
from ia_metec_point_adapter import xy_from_lon_lat, window_mean_release
from ia_correlated_response import configure_variable_correlation
from ia_correlated_solver import fit_correlated

DEST = STUDY / 'ia_correlated_transport_pilot_20261028'


def run():
    torch.set_num_threads(1)
    for name, digest in read(ROOT / 'paper_experiments/nls_baseline_frozen.json')['sha256_lf'].items():
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Frozen NLS changed: ' + name)
    parent = read(OUT / 'protocol.json')
    cases = read(STUDY / 'ia_location_capacity_pilot_20261028/protocol.json')['public'][:2]
    sources = [Path(__file__), *[ROOT / 'paper_experiments' / name for name in
        ['ia_correlated_dispersion.py', 'ia_correlated_response.py', 'ia_correlated_solver.py',
         'ia_metec_point_adapter.py', 'ia_physical_beam_adapter.py']],
        ROOT / 'pinn_source/reduced_trust_region.py']
    protocol = dict(cases=cases, orders=parent['numerical_orders'], max_nfev_per_start=100,
        selection='Same two original development panel events; no outcome-based inclusion',
        transport='Integrated OU velocity, measured uniform mean wind, rotating diffusion axes; reflecting ground',
        amplitude='Keep each original selected Q capacity, same original bounds/cap, all signed observations and q20',
        tau='Seconds, nonnegative without an upper bound; tau=0 exactly restores original transport',
        initial='Original two physical starts with tau=1s plus previous observation-fitted IA optimum at tau=0',
        comparison='Full-method physics refinement with an extra warm start, not isolated elimination or a compute-matched solver claim',
        information='Local tau derivative after projecting source,D,freeQ; descriptive only, not a fit gate',
        truth='Read only after fitting, never used for bounds, initialization, capacity or source choice',
        nls='7563376 frozen baseline, stored results only; never executed',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST / 'protocol.json').exists() and read(DEST / 'protocol.json') != protocol:
        raise ValueError('Preserve previous correlated-transport pilot')
    save(DEST / 'protocol.json', protocol)
    for source in sources:
        target = DEST / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    rows = []
    for identifier in cases:
        folder = DEST / identifier
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        meta = read(OUT / 'inputs' / identifier / 'metadata.json')
        previous = read(OUT / 'fits' / identifier / 'ia_vpdi.json')
        path = folder / 'ia_fit.json'
        if path.exists():
            fit = read(path)
        else:
            print('CORRELATED_TRANSPORT_START', identifier, flush=True)
            begun = time.perf_counter()
            orders = protocol['orders']
            model, xyt, basis, response, observed = configure_variable_correlation(data,
                meta['source_height_m'], device='cuda', time_order=orders['time'],
                release_order=orders['release'], history_minutes=orders['history_minutes'])
            plan_seconds = time.perf_counter() - begun
            initial = read(OUT / 'fits' / identifier / 'initialization.json')
            lower = np.r_[initial['bounds']['lower'][:4], 0.]
            upper = np.r_[initial['bounds']['upper'][:4], np.inf]
            starts = [np.r_[start[:4], 1.] for start in initial['starts']]
            starts.append(np.r_[previous['source_norm'], previous['diffusion_norm'][1],
                previous['diffusion_norm'][0] - previous['diffusion_norm'][1], 0.])
            fit = fit_correlated(model, xyt, basis, response, observed, starts, lower, upper,
                initial['bounds']['upper'][4], count=previous['q_basis_count'], max_nfev=100)
            fit.update(plan_seconds=plan_seconds, original_input_not_modified=True,
                numerical_replay_pending=True, peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
            save(path, fit)
            del response, model
            gc.collect()
            torch.cuda.empty_cache()
        truth = read(OUT / 'inputs' / identifier / 'truth.json')
        true_xy = xy_from_lon_lat([truth['lon']], [truth['lat']], meta['origin'])[0]
        for method, item in [('ia_correlated_transport', fit), ('original_ia', previous),
                            ('frozen_nls', read(OUT / 'fits' / identifier / 'joint_nls_trf.json'))]:
            rows.append(dict(event_id=identifier, method=method,
                localization_error_m=float(np.linalg.norm(np.asarray(item['source_norm']) * 100 - true_xy)),
                q_kg_h=3600 * window_mean_release(item['coefficients'], len(data['concentration']) * 60., data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'], rmse_ppm=float(np.sqrt(item['mse'])),
                correlation_time_s=item.get('correlation_time_s', 0.), converged=item.get('converged'),
                q_basis_count=item.get('q_basis_count', 5)))
        pd.DataFrame(rows).to_csv(DEST / 'comparison.csv', index=False, encoding='utf-8-sig')
        print('CORRELATED_TRANSPORT_DONE', rows[-3], flush=True)
    save(DEST / 'status.json', dict(pilot_complete=True, nls_called=False,
        numerical_replay_pending=True, production_promoted=False, full_goal_achieved=False))


if __name__ == '__main__':
    run()
