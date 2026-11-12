"""Oracle-location diagnostic only; never selects or updates inversion sources."""

import gc
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT, OUT, AUDIT, STUDY, read, save, JointBounds
from ia_metec_point_adapter import configure, xy_from_lon_lat
from refine_ia_affine_numerics import make_history, PREVIOUS
from paper_beam_joint_baselines import BeamJointResidual
from paper_location_capacity import capacity_response
from reduced_trust_region import fit_reduced_trust_region, ReducedResidual

DEST = STUDY / 'ia_public_model_gap_20261028'


def run():
    torch.set_num_threads(1)
    for name, digest in read(ROOT / 'paper_experiments/nls_baseline_frozen.json')['sha256_lf'].items():
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Frozen NLS changed: ' + name)
    cases = read(STUDY / 'ia_location_capacity_pilot_20261028/protocol.json')['public'][:2]
    orders = read(OUT / 'protocol.json')['numerical_orders']
    save(DEST / 'protocol.json', dict(cases=cases, orders=orders,
        scope='Evaluation-only oracle diagnostic, not a usable inversion or fair accuracy competitor',
        fits='Only D and original selected Q capacity, fixed true location; two original D starts plus fitted D',
        no_selection='Do not replace estimated sources or select wind/parameters based on oracle results',
        uncertainty='Finite multi-start optimization does not certify a global optimum',
        nls_called=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    raw = pd.read_csv(AUDIT / 'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp'] = pd.to_datetime(raw.time, utc=True) - pd.Timedelta(minutes=1)
    sensors = pd.read_csv(AUDIT / 'raw/input_data/sensor_locations.csv').set_index('name')
    rows = []
    for identifier in cases:
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        meta = read(OUT / 'inputs' / identifier / 'metadata.json')
        initial = read(OUT / 'fits' / identifier / 'initialization.json')
        truth = read(OUT / 'inputs' / identifier / 'truth.json')
        true_position = xy_from_lon_lat([truth['lon']], [truth['lat']], meta['origin'])[0] / 100.
        for mode in ['original_uniform', 'measured_affine']:
            path = DEST / identifier / (mode + '.json')
            if path.exists():
                row = read(path)['summary']
                rows.append(row)
                continue
            previous = read((OUT / 'fits' / identifier / 'ia_vpdi.json') if mode == 'original_uniform'
                            else (PREVIOUS / identifier / 'ia_affine_wind.json'))
            history = None if mode == 'original_uniform' else make_history(data, meta, raw, sensors)[0]
            model, xyt, basis, response, observed = configure(data, meta['source_height_m'], device='cuda',
                time_order=orders['time'], release_order=orders['release'],
                history_minutes=orders['history_minutes'], transport_history=history)
            objective = BeamJointResidual(response, model, 0., xyt, basis, observed, JointBounds(**initial['bounds']))
            full, mapping = capacity_response(objective, previous['q_basis_count'])
            old_theta = np.r_[previous['source_norm'], previous['diffusion_norm'][1],
                              previous['diffusion_norm'][0] - previous['diffusion_norm'][1]]

            def oracle_matrix(diffusion, jacobian):
                value = full(np.r_[true_position, diffusion], jacobian)
                if jacobian:
                    matrix, derivative = value
                    return matrix, derivative[..., 2:]
                return value

            print('MODEL_GAP_START', identifier, mode, flush=True)
            starts = [np.asarray(s)[2:4] for s in initial['starts']] + [old_theta[2:]]
            oracle = fit_reduced_trust_region(oracle_matrix, observed, starts,
                objective.bounds.lower[2:4], objective.bounds.upper[2:4],
                q_upper_bound=float(objective.bounds.upper[4]), max_nfev=100,
                coordinate_scale=np.array([.03, .03]))
            # Reprofile Q at the stored estimated location for an equal-capacity comparison.
            matrix = full(old_theta, False)
            reduced = ReducedResidual(lambda unused, jac: matrix, observed,
                                      q_upper_bound=float(objective.bounds.upper[4]))
            residual = reduced.residual(np.zeros(1))
            estimated_sse = float(residual @ residual)
            row = dict(event_id=identifier, mode=mode, q_basis_count=previous['q_basis_count'],
                estimated_location_error_m=float(np.linalg.norm(old_theta[:2] - true_position) * 100),
                estimated_location_reprofiled_sse=estimated_sse, oracle_location_sse=oracle['sse'],
                oracle_minus_estimated_sse=oracle['sse'] - estimated_sse,
                oracle_converged=oracle.get('converged'), observations=len(observed),
                estimated_rmse_ppm=float(np.sqrt(estimated_sse / len(observed))),
                oracle_rmse_ppm=float(np.sqrt(oracle['sse'] / len(observed))))
            save(path, dict(summary=row, oracle_fit=oracle, recorded_true_position_norm=true_position,
                inverted_source_not_modified=True, nls_called=False))
            rows.append(row)
            pd.DataFrame(rows).to_csv(DEST / 'comparison.csv', index=False, encoding='utf-8-sig')
            print('MODEL_GAP_DONE', row, flush=True)
            del objective, response, model, full, oracle, reduced
            gc.collect()
            torch.cuda.empty_cache()
    save(DEST / 'status.json', dict(complete=True, diagnostic_rows=len(rows), nls_called=False,
        inversion_sources_modified=False, production_promoted=False,
        interpretation='Oracle fits diagnose model-objective alignment; not evidence of inversion accuracy gains'))


if __name__ == '__main__':
    run()
