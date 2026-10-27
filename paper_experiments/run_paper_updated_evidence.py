"""Sequential evidence refresh for the fixed paper cohorts; historical fits stay read-only."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import run_paper_comparison_v4 as prior
from paper_fast_parcels import CachedMeanResponse
from paper_joint_baselines import JointBounds, JointResidual, fit_standard_pso
from paper_reduced_solver import fit_profiled_trf, standard_pso_scales
from paper_q_model_selection import fit_profiled_q_models

STUDY = prior.STUDY
METHODS = ['ia_vpdi', 'joint_nls_trf', 'standard_gbest_pso', 'matched_joint_adam']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    prior.formal.base.save_json(Path(path), record)


def numerical_sources():
    paths = [Path(__file__).with_name(name) for name in (
        'paper_fast_parcels.py', 'paper_joint_baselines.py', 'paper_reduced_solver.py',
        'paper_q_model_selection.py', 'paper_physical_beam_adapter.py')]
    paths.extend(prior.ROOT/'pinn_source'/name for name in ('reduced_trust_region.py', 'q_capacity.py'))
    return {str(path.relative_to(prior.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def protocol(path, record):
    if path.exists():
        if read(path) != record:
            raise ValueError('Changed evidence protocol; choose a new output folder')
    else:
        save(path, record)


def synthetic_setup(entry, device):
    case_id = entry['case']['case_id']
    old = read(STUDY/'new_comparison_v4'/case_id/'result.json')
    with np.load(STUDY/entry['parent']/case_id/'input.npz') as archive:
        data = {k: archive[k].copy() for k in archive.files}
    meta = old['metadata']
    model, xyt, basis = prior.formal.parcel.configure(meta, data['stations'], device,
        sigma=meta['sigma'], wind_interval_mean=True)
    model.double()
    response = CachedMeanResponse(stations=len(data['stations']))
    bounds = JointBounds(**old['bounds'])
    objective = JointResidual(response, model, meta['sigma'], xyt, basis, data['observed'], bounds)
    starts = [np.r_[s['source_x'], s['source_y'], s['diffusion_perp_norm'],
        s['diffusion_parallel_norm']-s['diffusion_perp_norm'], s['coefficients']]
        for s in old['common_starts']]
    return old, data, objective, starts


def physical_information(obj, fit):
    """Existing information definition, with exact forward AD rather than row-wise VJP."""
    ia = prior.formal.base.ia
    count = fit['q_basis_count']
    basis = prior.formal.base.vp.build_q_basis(obj.model.recurrent_integration_times, count)
    coefficients = np.asarray(fit['native_coefficients'])
    dp, dq = fit['diffusion_norm']
    ia.set_model_anisotropic_diffusion_norm(obj.model, dp, dq)
    with torch.no_grad():
        obj.model.xs.fill_(fit['source_norm'][0])
        obj.model.ys.fill_(fit['source_norm'][1])
    def jacobian(model, unused_prediction, unused_rcond):
        values = obj.tensor([*fit['source_norm'], dp, dq])
        def prediction(theta):
            physical = torch.stack((theta[0], theta[1], theta[3], theta[2]-theta[3]))
            return obj.response(model, obj.sigma, obj.xyt, basis, track_gradients=True,
                physical_parameters=physical) @ obj.tensor(coefficients)
        return torch.autograd.functional.jacobian(prediction, values, vectorize=True,
            strategy='forward-mode').detach().cpu().numpy()
    with patch.object(ia, 'build_response_matrix', obj.response), \
         patch.object(ia, '_theta_jacobian_numpy', jacobian):
        info = ia.compute_profile_information(obj.model, obj.sigma, obj.xyt,
            obj.observations, basis, coefficients, q_upper_bound=float(obj.bounds.upper[4]))
    replay = obj.response(obj.model, obj.sigma, obj.xyt, basis).detach().cpu().numpy()@coefficients
    np.testing.assert_allclose(replay, fit['prediction'], rtol=1e-7, atol=1e-10)
    return asdict(info)


def prepare_synthetic(root, pso, device):
    source = STUDY/'new_comparison_v4'
    frozen = read(source/'protocol.json')
    out = root/'synthetic'
    changed = dict(frozen, version='paper_update_capacity_v1',
        updated_numerical_source_sha256=numerical_sources(),
        reused_fit_protocol_sha256={str(path.relative_to(STUDY)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                source/'protocol.json', STUDY/'ia_release_capacity_full_v1/protocol.json', pso/'protocol.json')},
        updated_algorithm='scaled reduced TRF with all-data BIC capacities1/3/5',
        pso_source=str(pso.relative_to(STUDY)), q_capacity_selection='bic',
        capacity_selection_uncertainty='local regions conditional on selected capacity',
        baseline_scope='fixed-five NLS and Adam retained; matched-capacity control is separate24-case study')
    protocol(out/'protocol.json', changed)
    for entry in frozen['cases']:
        case_id = entry['case']['case_id']
        path = out/case_id/'result.json'
        if path.exists():
            continue
        record, data, obj, _ = synthetic_setup(entry, device)
        updated = read(STUDY/'ia_release_capacity_full_v1'/case_id/'result.json')['methods']['ia_vpdi_release_bic']
        particle = read(pso/case_id/'result.json')['methods']['standard_pso_scaled']
        fit = dict(updated, coefficients=updated['native_coefficients'])
        record['methods']['ia_vpdi'] = fit
        record['methods']['standard_gbest_pso'] = particle
        record['conditional_information'] = physical_information(obj, updated)
        record['inference_scope'] = 'conditional selected release capacity; retrospective calibration'
        record['input_hashes'] = {str(p.relative_to(STUDY)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source/case_id/'result.json',
                      STUDY/'ia_release_capacity_full_v1'/case_id/'result.json', pso/case_id/'result.json')}
        save(path, record)
        print('UPDATED_SYNTHETIC', case_id, flush=True)
    from analyze_paper_comparison_v4 import analyze
    print(analyze(out), flush=True)


def control_cases(entries):
    """Six frozen-order cases per geometry, respecting the actual scenario design."""
    selected, counts = [], {}
    for entry in entries:
        spec = entry['case']
        if spec['cohort'] != 'nominal':
            continue
        network = spec['network']
        if counts.get(network, 0) < 6:
            selected.append(entry)
            counts[network] = counts.get(network, 0)+1
    if len(selected) != 24 or len(counts) != 4 or set(counts.values()) != {6}:
        raise ValueError('Expected six existing nominal cases in each of four geometries')
    return selected


def capacity_control(root, device):
    entries = read(STUDY/'new_comparison_v4/protocol.json')['cases']
    selected = control_cases(entries)
    out = root/'capacity_control'
    protocol(out/'protocol.json', dict(cases=selected,
        selection='first six nominal inputs per geometry in frozen protocol order, no truth',
        design='original two geometries: six wind/release strata; supplementary fixed scenarios: six repeats each',
        numerical_source_sha256=numerical_sources(),
        counts=[1, 3, 5], criterion='same BIC as updated IA', nls_nfev=250,
        differentiation='exact forward AD; joint TRF, no Q elimination during optimization',
        starts='same geometry and diffusion; bounded Q fit only at common initial points',
        interpretation='24-case capacity-matched mechanism control, not216 new independent cases'))
    rows = []
    for entry in selected:
        case_id = entry['case']['case_id']
        path = out/case_id/'result.json'
        if path.exists():
            record = read(path)
        else:
            old, data, obj, starts = synthetic_setup(entry, device)
            fit = fit_profiled_q_models(obj, starts, solver='joint_nls')
            fit.update(prior.formal.metrics(old['metadata'], data, fit['source_norm'],
                np.asarray(fit['coefficients']), np.asarray(fit['prediction'])))
            ia = read(STUDY/'ia_release_capacity_full_v1'/case_id/'result.json')['methods']['ia_vpdi_release_bic']
            record = dict(case_id=case_id, methods={'ia_vpdi_bic': ia, 'joint_nls_bic': fit}, complete=True)
            save(path, record)
        for name, fit in record['methods'].items():
            rows.append(dict(case_id=case_id, method=name, q_basis_count=fit['q_basis_count'],
                **{k: fit[k] for k in ('localization_error_m', 'q_relative_l2', 'observed_mse', 'seconds')}))
        print('CAPACITY_CONTROL', case_id, flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out/'comparison.csv', index=False)
    save(out/'status.json', dict(final=True, cases=24, rows=len(rows)))


def public_update(root, device):
    import run_paper_controlled_release as physical
    from paper_beam_joint_baselines import BeamJointResidual
    source = STUDY/'beam_comparison_cuda_v1'
    previous = read(source/'protocol.json')
    out = root/'public'
    protocol(out/'protocol.json', dict(previous, version='paper_update_public_trf_v1',
        updated_numerical_source_sha256=numerical_sources(),
        methods=['ia_vpdi', 'joint_nls_trf', 'matched_joint_adam'],
        ia_solver='reduced TRF', ia_nfev=250,
        q_capacity='constant known controlled release, same as all physical comparators',
        pso_scope='updated PSO evaluated on216 synthetic cases; old physical PSO archived, not rerun',
        copied_baselines='NLS and Adam read-only from original full-window results'))
    truth = read(STUDY/'public_data/author_truth_values.json')['values_as_stored']
    for event in previous['events']:
        name = event['source'].replace('_', '')+'_'+event['release']
        for height in previous['heights_m']:
            destination = out/name/f'height_{height:.1f}'/'result.json'
            if destination.exists():
                continue
            record = read(source/name/f'height_{height:.1f}'/'result.json')
            record['methods'].pop('standard_gbest_pso')
            model, xyt, basis, response, observed, _ = physical.configure(height, device=device,
                dtype=torch.float64, q_count=1, time_order=16, release_order=16,
                beam_solver='analytic', data_dir=STUDY/'controlled_release_preparation_v2'/name)
            bounds = JointBounds(**record['bounds'])
            starts = [np.r_[s['source_x'], s['source_y'], s['diffusion_perp_norm'],
                s['diffusion_parallel_norm']-s['diffusion_perp_norm'], s['coefficients']]
                for s in record['common_starts']]
            def objective():
                return BeamJointResidual(response, model, 0., xyt, basis, observed, bounds)
            record['methods']['ia_vpdi'] = fit_profiled_trf(objective(), starts)
            for method in ('ia_vpdi',):
                value = record['methods'][method]
                for fit in value.get('replicates', [value]):
                    fit.update(source_m=(100*np.asarray(fit['source_norm'])).tolist(),
                        mean_q_kg_h=float(fit['coefficients'][0]*3600),
                        localization_error_m=float(np.linalg.norm(100*np.asarray(fit['source_norm'])
                            -np.asarray(truth[event['source']+'_location'][:2]))),
                        observed_rmse_ppm=float(np.sqrt(fit['mse'])))
            save(destination, record)
            print('UPDATED_PUBLIC', name, height, flush=True)
    from analyze_paper_beam_comparison import analyze
    print(analyze(out, STUDY/'manuscript_results_v2/controlled_release_case_metrics.csv'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STUDY/'paper_updated_20261026')
    parser.add_argument('--pso', type=Path, default=STUDY/'paper_update_20261026_pso_fp32')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--stage', choices=('synthetic', 'control', 'public', 'all'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage in ('control', 'all'):
        capacity_control(args.output, args.device)
    if args.stage in ('synthetic', 'all'):
        prepare_synthetic(args.output, args.pso, args.device)
    if args.stage in ('public', 'all'):
        public_update(args.output, args.device)
