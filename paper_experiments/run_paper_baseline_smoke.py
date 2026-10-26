"""Short technical demos on two frozen inputs, not paper performance evidence.

Finite PSO diffusion/amplitude bounds are explicitly provisional in this demo.
They must be aligned with IA before the new comparative cohort is frozen.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ROOT/'deployment_runs/paper_baseline_dependencies'
if DEPENDENCIES.is_dir():
    sys.path.insert(0, str(DEPENDENCIES))

import run_paper_formal_experiments as formal
from paper_fast_parcels import CachedMeanResponse
from paper_joint_baselines import JointBounds, JointResidual, fit_joint_nls, fit_standard_pso


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--nls-evaluations', type=int, default=8)
    parser.add_argument('--particles', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    study = formal.ROOT/'evaluation_v3_shared_forward'
    protocol = dict(purpose='technical demo only; not a comparative accuracy experiment',
        cases=['eval_001', 'eval_002'], selection='first ideal5 and jjj6 in the frozen design',
        initialization='saved COMMON coarse starts; not any fitted solver solution',
        truth_policy='truth read for metrics only, never passed into solvers',
        nls='joint bounded TRF with exact autograd Jacobian; no NNLS iterations',
        pso='public GlobalBestPSO; fixed coefficients; random uniform unit-box initialization',
        finite_box_status='provisional engineering box, NOT a matched IA comparison',
        upper_bound_rule='D_perp and contrast each <=1 model unit; '
                         'each a_j<=100*maximum COMMON initializer coefficient',
        source_bounds=[-.75, .75], nls_evaluations=args.nls_evaluations,
        particles=args.particles, iterations=args.iterations, seeds=[1729, 1730, 1731],
        device=args.device, dtype='float64', torch=torch.__version__)
    protocol['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), Path(__file__).with_name('paper_joint_baselines.py'),
                  Path(__file__).with_name('paper_fast_parcels.py')]}
    formal.base.save_json(output/'protocol.json', protocol)
    # PySwarms' own default report.log belongs inside the demo, not repository root.
    previous_directory = Path.cwd()
    try:
        os.chdir(output)
        for case_id in protocol['cases']:
            case = study/case_id
            metadata = json.loads((case/'input.json').read_text(encoding='utf-8'))
            saved = json.loads((case/'result.json').read_text(encoding='utf-8'))
            with np.load(case/'input.npz') as archive:
                data = {name: archive[name].copy() for name in archive.files}
            model, xyt, basis = formal.parcel.configure(metadata, data['stations'], args.device,
                sigma=metadata['sigma'], wind_interval_mean=True)
            model.double()
            response = CachedMeanResponse(stations=len(data['stations']))
            initial = [np.r_[s['source_x'], s['source_y'], s['diffusion_perp_norm'],
                s['diffusion_parallel_norm']-s['diffusion_perp_norm'], s['coefficients']]
                for s in saved['common_starts']]
            amplitude_cap = 100*max(float(np.max(s[4:])) for s in initial)
            bounds = JointBounds(np.r_[[-.75, -.75, .00100005, 0.], np.zeros(basis.shape[1])],
                                 np.r_[[.75, .75, 1., 1.], np.full(basis.shape[1], amplitude_cap)])
            def objective():
                return JointResidual(response, model, metadata['sigma'], xyt, basis,
                                     data['observed'], bounds)
            results = [fit_joint_nls(objective(), initial, args.nls_evaluations)]
            print('NLS', case_id, results[0]['mse'], flush=True)
            for seed in protocol['seeds']:
                results.append(fit_standard_pso(objective(), seed=seed,
                    particles=args.particles, iterations=args.iterations))
                print('PSO', case_id, seed, results[-1]['mse'], flush=True)
            for result in results:
                result['metrics'] = formal.metrics(metadata, data, result['source_norm'],
                    np.asarray(result['coefficients']), np.asarray(result['prediction']))
                with formal.base.ia.temporary_source_and_diffusion(
                        model, *result['source_norm'], *result['diffusion_norm']):
                    replay = response(model, metadata['sigma'], xyt, basis).detach().cpu().numpy()@result['coefficients']
                np.testing.assert_allclose(replay, result['prediction'], rtol=1e-8, atol=1e-10)
                result['replay_max_absolute_difference'] = float(np.max(np.abs(replay-result['prediction'])))
            formal.base.save_json(output/(case_id+'.json'), dict(case_id=case_id,
                input_sha256=hashlib.sha256((case/'input.npz').read_bytes()).hexdigest(),
                bounds=asdict(bounds), methods=results))
        formal.base.save_json(output/'PASS.json', dict(technical_demo_passed=True,
            cases=2, fits=8, comparable_accuracy_claim=False))
    finally:
        os.chdir(previous_directory)


if __name__ == '__main__':
    main()
