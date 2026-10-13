"""Matched serial/batched kernel checks on historical inputs, without source search."""

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

from compare_historical_inversion_modes import CASES
from inversion_validation_common import capture_context, configured_variant, load_ia_vpdi_report
import pipeline
import torch
from ia_vpdi import _prediction_jacobian_raw
from variable_projection import build_q_basis, build_response_matrix, resolve_q_basis_count, solve_nonnegative_q


def measure(function, device, repeats):
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()
    result = function()
    sync()
    durations = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        result = function()
        sync()
        durations.append(time.perf_counter()-start)
    return result, statistics.median(durations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--smoke', action='store_true', help='Also replay short complete IA pipelines on CPU')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    os.environ['PINN_DEVICE'] = 'cpu'
    os.environ['PINN_TRANSPORT_SOLVER'] = 'grid'
    os.environ['PINN_IA_VP_TIMING'] = '0'
    rows, smoke = [], []
    with tempfile.TemporaryDirectory(prefix='inversion_runtime_') as temporary:
        root = Path(temporary)
        for case in CASES:
            folder = root/case
            folder.mkdir()
            context = capture_context(case, folder)
            for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
                prototype = copy.deepcopy(context['model']).to(device)
                model = configured_variant(prototype, pipeline.RECURRENT_GRID_NX, preserve_diffusion_mode=True)
                model.recurrent_transport_solver = 'grid'
                basis = build_q_basis(model.recurrent_integration_times.to(model.xs),
                    resolve_q_basis_count(model.recurrent_integration_times.numel(), context['q_basis_count']))
                query = context['observation_xyt'].to(model.xs)
                serial, before = measure(lambda: build_response_matrix(model, context['sigma_src'],
                    query, basis, batch_size=1), device, args.repeats)
                batched, after = measure(lambda: build_response_matrix(model, context['sigma_src'],
                    query, basis, batch_size=8), device, args.repeats)
                torch.testing.assert_close(batched, serial, rtol=2e-4, atol=1e-7)
                first = solve_nonnegative_q(serial, context['observations'])
                second = solve_nonnegative_q(batched, context['observations'])
                row = {'case': case, 'device': device, 'basis_count': basis.shape[1],
                    'observations': query.shape[0], 'serial_response_seconds': before,
                    'batched_response_seconds': after, 'response_speedup': before/after,
                    'response_max_delta': float((serial-batched).abs().max().cpu()),
                    'serial_sse': first.weighted_sse, 'batched_sse': second.weighted_sse}
                response = build_response_matrix(model, context['sigma_src'], query[:8], basis, track_gradients=True)
                prediction = response @ torch.as_tensor(first.coefficients, device=model.xs.device, dtype=model.xs.dtype)
                reference, before = measure(lambda: _prediction_jacobian_raw(model, prediction, 1), device, args.repeats)
                actual, after = measure(lambda: _prediction_jacobian_raw(model, prediction, 16), device, args.repeats)
                torch.testing.assert_close(actual, reference, rtol=2e-4, atol=1e-7)
                row.update(jacobian_rows=prediction.numel(), serial_jacobian_seconds=before,
                    batched_jacobian_seconds=after, jacobian_speedup=before/after)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if args.smoke:
                from unittest.mock import patch
                from inversion_validation_common import SOURCE
                for size in (1, 8):
                    env = dict(os.environ, PINN_DEVICE='cpu', PINN_INVERSION_MODE='ia_vpdi',
                        PINN_IA_VP_RESPONSE_BATCH_SIZE=str(size),
                        PINN_IA_VP_JACOBIAN_BATCH_SIZE='1' if size == 1 else '16',
                        PINN_IA_VP_SEARCH_STEP_M='100000', PINN_IA_VP_TOP_K='1',
                        PINN_IA_VP_OUTER_STEPS='3', PINN_IA_VP_Q_LAYOUT_STRATEGY='uniform',
                        PINN_TRANSPORT_SOLVER='grid', PINN_IA_VP_TIMING='1')
                    with patch.dict(os.environ, env), patch.multiple(pipeline,
                            IA_VP_DIFFUSION_SCALE_SEEDS=(1.,), IA_VP_DIFFUSION_RATIO_SEEDS=(1.,)):
                        start = time.perf_counter()
                        run = pipeline.run(*(SOURCE/case/name for name in ('sites.xlsx', 'concentration.xlsx', 'wind.xlsx')),
                            output_dir=folder/f'smoke_{size}', result_root_dir=folder/f'exports_{size}',
                            make_plots=False, random_seed=0)
                        report = load_ia_vpdi_report(run['output_dir'])
                        smoke.append({'case': case, 'batch_size': size, 'seconds': time.perf_counter()-start,
                            'source_x': run['xs'], 'source_y': run['ys'],
                            'score': report['best_profiled_score'], 'timing': report['timing']})
    for case in CASES if args.smoke else ():
        pair = [row for row in smoke if row['case'] == case]
        import numpy as np
        np.testing.assert_allclose(pair[0]['score'], pair[1]['score'], rtol=5e-5, atol=1e-7)
        np.testing.assert_allclose([pair[0]['source_x'], pair[0]['source_y']],
                                   [pair[1]['source_x'], pair[1]['source_y']], atol=1e-3, rtol=0.)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'scope': 'Matched kernel benchmark; 8-row Jacobian subset, not full diagnostic runtime.',
        'repeats': args.repeats, 'threads': 4, 'rows': rows, 'short_pipeline_smoke': smoke},
        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
