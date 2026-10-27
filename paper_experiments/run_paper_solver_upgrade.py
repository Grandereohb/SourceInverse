"""Versioned validation of reduced TRF and scaled standard PSO on frozen inputs.

Old fits are read-only comparators. Truth is used only after solving for metrics.
Pilot cases are the first input in each nominal geometry plus each mismatch type.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deployment_runs/paper_baseline_dependencies'))
import run_paper_comparison_v4 as previous
from paper_joint_baselines import JointBounds, JointResidual, fit_standard_pso
from paper_reduced_solver import fit_profiled_trf, standard_pso_scales
from paper_q_model_selection import fit_profiled_q_models

STUDY = previous.STUDY


def planned_cases(protocol, scope):
    cases = protocol['cases']
    if scope == 'full':
        return cases
    selected, keys = [], set()
    for case in cases:
        spec = case['case']
        key = spec['network'] if spec['cohort'] == 'nominal' else spec['cohort']
        if key not in keys:
            selected.append(case)
            keys.add(key)
    if scope == 'diagnostic':
        # Known compensation failures: development only, not an independent test set.
        selected.extend(c for c in cases if c['case']['case_id'] in
                        ('eval_061', 'eval_119', 'supp_029') and c not in selected)
    return selected


def validate_resume(saved, current, audit, protocol_bytes):
    """Permit only individually audited source-only changes; retain frozen protocol."""
    if audit.get('frozen_protocol_sha256') != hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError('Resume audit does not match the frozen protocol')
    if saved['methods'] != ['pso'] or current['methods'] != ['pso']:
        raise ValueError('This compatibility audit is restricted to the PSO-only batch')
    changes = audit['approved_source_changes']
    normalized = dict(current, source_sha256=dict(current['source_sha256']))
    for path, change in changes.items():
        if (saved['source_sha256'].get(path) != change['before']
                or current['source_sha256'].get(path) != change['after']):
            raise ValueError('Resume source audit mismatch: '+path)
        normalized['source_sha256'][path] = change['before']
    if saved != normalized:
        raise ValueError('Unaudited protocol/input/solver change; choose a new batch')


def run(output, scope, device, methods, resume_audit=None):
    parent = STUDY/'new_comparison_v4'
    frozen = json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
    cases = planned_cases(frozen, scope)
    files = [Path(__file__), Path(__file__).with_name('paper_reduced_solver.py'),
             Path(__file__).with_name('paper_joint_baselines.py'),
             ROOT/'pinn_source/reduced_trust_region.py',
             ROOT/'pinn_source/q_capacity.py',
             Path(__file__).with_name('paper_q_model_selection.py'),
             Path(__file__).with_name('paper_fast_parcels.py')]
    protocol = dict(version='scaled_solver_validation_v5', scope=scope, device=device,
        methods=methods, cases=cases, source_sha256={str(p.relative_to(ROOT)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        input_sha256={case['case']['case_id']: hashlib.sha256(
            (parent/case['case']['case_id']/'result.json').read_bytes()).hexdigest() for case in cases},
        observation_sha256={case['case']['case_id']: hashlib.sha256(
            (STUDY/case['parent']/case['case']['case_id']/'input.npz').read_bytes()).hexdigest() for case in cases},
        initialization='same two common coarse starts, never fitted NLS or truth',
        model='same float64 forward, observations, bounds and SSE; optional nested subspaces of five hats',
        trf=dict(max_nfev_per_start=250, inner='column-equilibrated BVLS tol1e-12',
                 jacobian='full active-set residual sensitivity, not detached-Q approximation'),
        release_selection=dict(counts=[1, 3, 5], criterion='Gaussian unknown-variance BIC',
                               geometric_parameters=4, production_default_changed=False),
        pso=dict(particles=32, iterations=256, seeds=[1729, 1730, 1731],
                 local_fraction=.75, velocity_fraction=.2, fixed_options=True,
                 initial_spread=.1, zero_initial_velocity=True,
                 population_evaluation=True,
                 population_dtype='float32, double final replay; relative SSE gate 1e-4',
                 local_refinement=False, q_elimination=False),
        selection='first case by metadata stratum; no winner filtering',
        comparison='old NLS/Adam/PSO are frozen read-only references, not isolated new timings')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output/'protocol.json'
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding='utf-8'))
        if saved != protocol:
            if resume_audit is None:
                raise ValueError('Changed validation protocol: choose a new output version')
            validate_resume(saved, protocol, json.loads(Path(resume_audit).read_text(encoding='utf-8')),
                            protocol_path.read_bytes())
            print('AUDITED_RESUME: frozen protocol retained', flush=True)
    else:
        previous.formal.base.save_json(protocol_path, protocol)
    rows = []
    cwd = Path.cwd()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        os.chdir(output)
        for entry in cases:
            case_id = entry['case']['case_id']
            destination = output/case_id
            result_path = destination/'result.json'
            old = json.loads((parent/case_id/'result.json').read_text(encoding='utf-8'))
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
            else:
                print('START_UPGRADE', case_id, flush=True)
                case = STUDY/entry['parent']/case_id
                with np.load(case/'input.npz') as archive:
                    data = {name: archive[name].copy() for name in archive.files}
                metadata = old['metadata']
                model, xyt, basis = previous.formal.parcel.configure(metadata, data['stations'],
                    device, sigma=metadata['sigma'], wind_interval_mean=True)
                model.double()
                response = previous.CachedMeanResponse(stations=len(data['stations']))
                bounds = JointBounds(**old['bounds'])
                starts = [np.r_[s['source_x'], s['source_y'], s['diffusion_perp_norm'],
                    s['diffusion_parallel_norm']-s['diffusion_perp_norm'], s['coefficients']]
                    for s in old['common_starts']]
                def objective():
                    return JointResidual(response, model, metadata['sigma'], xyt,
                                         basis, data['observed'], bounds)
                fitted = {}
                if 'ia' in methods:
                    fitted['ia_vpdi_reduced_trf'] = fit_profiled_trf(objective(), starts, 250)
                    print('TRF_DONE', case_id, fitted['ia_vpdi_reduced_trf']['mse'], flush=True)
                if 'ia_release' in methods:
                    fit = fit_profiled_q_models(objective(), starts)
                    fitted['ia_vpdi_release_bic'] = fit
                    print('BIC_DONE', case_id, fit['q_basis_count'], fit['mse'], flush=True)
                if 'pso' in methods:
                    scale = standard_pso_scales(starts, bounds.lower, bounds.upper)
                    fitted['standard_pso_scaled'] = dict(replicates=[])
                    for seed in (1729, 1730, 1731):
                        particle_objective = objective()
                        particle_objective.population_dtype = torch.float32
                        particle_objective.population_batch_size = 16
                        particle_objective.population_kernel_elements = 8_000_000
                        fit = fit_standard_pso(particle_objective, seed=seed, particles=32,
                            iterations=256, starts=starts, coordinate_scales=scale,
                            local_fraction=.75, velocity_fraction=.2,
                            initial_spread=.1, zero_initial_velocity=True,
                            use_population_evaluation=True)
                        fit['population_precision_relative_sse_difference'] = abs(
                            fit['optimizer_sse']-fit['sse'])/max(fit['sse'], 1e-30)
                        if fit['population_precision_relative_sse_difference'] > 1e-4:
                            raise ValueError('Population precision gate failed; preserve completed cases')
                        fitted['standard_pso_scaled']['replicates'].append(fit)
                        print('PSO_DONE', case_id, seed, fit['mse'], flush=True)
                for value in fitted.values():
                    for fit in value.get('replicates', [value]):
                        theta = np.r_[fit['source_norm'], fit['diffusion_norm'][1],
                            fit['diffusion_norm'][0]-fit['diffusion_norm'][1], fit['coefficients']]
                        replay = objective().residual(theta)+data['observed']
                        np.testing.assert_allclose(replay, fit['prediction'], rtol=1e-8, atol=1e-10)
                        fit.update(previous.formal.metrics(metadata, data, fit['source_norm'],
                            np.asarray(fit['coefficients']), np.asarray(fit['prediction'])))
                result = dict(case_id=case_id, metadata=metadata, methods=fitted, complete=True)
                destination.mkdir(exist_ok=True)
                previous.formal.base.save_json(result_path, result)
                print('DONE_UPGRADE', case_id, flush=True)
            for method, value in {**old['methods'], **result['methods']}.items():
                fits = value.get('replicates', [value])
                rows.append(dict(case_id=case_id, network=old['metadata']['network'],
                    cohort=old['metadata']['cohort'], method=method,
                    localization_error_m=float(np.median([f['localization_error_m'] for f in fits])),
                    observed_mse=float(np.median([f['observed_mse'] for f in fits])),
                    q_relative_l2=float(np.median([f['q_relative_l2'] for f in fits])),
                    seconds=float(np.median([f['seconds'] for f in fits]))))
            pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=('pilot', 'diagnostic', 'full'), default='pilot')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--methods', nargs='+', choices=('ia', 'ia_release', 'pso'), default=['ia', 'pso'])
    parser.add_argument('--resume-audit', type=Path)
    args = parser.parse_args()
    run(args.output, args.scope, args.device, args.methods, args.resume_audit)
