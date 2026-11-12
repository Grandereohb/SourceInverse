"""Read-only archive audit and paired ablation; no baseline fitting or mutation."""
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT, STUDY, OUT as PUBLIC, read, save
from ia_metec_point_adapter import configure, xy_from_lon_lat, window_mean_release
from paper_sqrt_affine_wind import SquareRootAffineWindHistory

DEST = ROOT / 'output/ablation_20261110'
AFFINE = STUDY / 'ia_affine_wind_sqrt_full_20261028'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metec():
    """Replay every archived cell through the same kernel, at the same orders."""
    parent = read(PUBLIC / 'protocol.json')
    for relative, expected in parent['input_sha256'].items():
        assert digest(ROOT / relative) == expected, relative
    for identifier in parent['event_ids']:
        target = DEST / 'audit' / (identifier + '.json')
        if target.exists():
            continue
        data = dict(np.load(PUBLIC / 'inputs' / identifier / 'input.npz'))
        meta = read(PUBLIC / 'inputs' / identifier / 'metadata.json')
        initial = read(PUBLIC / 'fits' / identifier / 'initialization.json')
        records = []
        for wind in ['mean', 'spatial']:
            history = None
            if wind == 'spatial':
                wind_path = AFFINE / identifier / 'wind_calibration.json'
                calibration = read(wind_path)
                history = SquareRootAffineWindHistory(np.arange(len(data['concentration'])+1)*60.,
                    calibration['velocity_intercepts_m_s'], calibration['gradient_per_s'], data['wind_xy'])
            begun = time.perf_counter()
            model, xyt, basis, response, observed = configure(data, meta['source_height_m'],
                device='cuda' if torch.cuda.is_available() else 'cpu', time_order=16,
                release_order=16, history_minutes=20, transport_history=history)
            for solver in ['vp_bic', 'nls5']:
                filename = ('ia_vpdi.json' if solver == 'vp_bic' else 'joint_nls_trf.json') if wind == 'mean' else (
                    'ia_affine_wind.json' if solver == 'vp_bic' else 'joint_nls_affine_wind.json')
                source = (PUBLIC / 'fits' if wind == 'mean' else AFFINE) / identifier / filename
                fit = read(source)
                dp, dq = fit['diffusion_norm']
                theta = model.xs.new_tensor([*fit['source_norm'], dq, dp-dq])
                with torch.no_grad():
                    pred = (response(model, 0., xyt, basis, physical_parameters=theta) @
                        model.xs.new_tensor(fit['coefficients'])).cpu().numpy()
                stored = np.asarray(fit['prediction'])
                relative = float(np.linalg.norm(pred-stored)/max(np.linalg.norm(stored), 1e-30))
                records.append(dict(event_id=identifier, wind=wind, solver=solver,
                    source=str(source.relative_to(ROOT)), source_sha256=digest(source),
                    initialization_sha256=digest(PUBLIC/'fits'/identifier/'initialization.json'),
                    relative_prediction_change=relative, reuse_verified=relative <= 1e-6,
                    n_observations=len(observed), replay_rmse=float(np.sqrt(np.mean((pred-observed)**2))),
                    plan_and_replay_seconds=time.perf_counter()-begun))
            del model, response
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        save(target, records)
        print('AUDITED', identifier, max(r['relative_prediction_change'] for r in records), flush=True)


def work(fit):
    runs = fit.get('optimizer_runs', [])
    return dict(nfev=sum(r.get('nfev', 0) for r in runs) if runs else None,
        response_calls=fit.get('response_matrix_calls', fit.get('counters', {}).get('response_matrix_calls')),
        derivatives=fit.get('derivative_calls', fit.get('counters', {}).get('jacobian_calls')),
        work_scope='selected_capacity_only' if 'release_models' in fit else 'single_capacity',
        converged=fit.get('converged'))


def assemble():
    rows = []
    mean_checks = pd.read_csv(PUBLIC/'analysis/replicate_metrics.csv', dtype={'event_id':str})
    spatial_checks = read(AFFINE/'replay_summary.json')['rows']
    for path in sorted((DEST/'audit').glob('*.json')):
        for audit in read(path):
            fit = read(ROOT/audit['source'])
            identifier = audit['event_id']
            data = dict(np.load(PUBLIC/'inputs'/identifier/'input.npz'))
            meta = read(PUBLIC/'inputs'/identifier/'metadata.json')
            truth = read(PUBLIC/'inputs'/identifier/'truth.json')
            true_xy = xy_from_lon_lat([truth['lon']], [truth['lat']], meta['origin'])[0]
            q = 3600*window_mean_release(fit['coefficients'], len(data['concentration'])*60., data['release_window_s'])
            if audit['wind']=='spatial':
                check = next(r for r in spatial_checks if r['event_id']==identifier and
                    r['method']==('ia_affine_wind' if audit['solver']=='vp_bic' else 'joint_nls_affine_wind'))
                gate, delta = check['passed'], check['relative_l2']
            else:
                check = mean_checks[(mean_checks.event_id==identifier)&(mean_checks.method==
                    ('ia_vpdi' if audit['solver']=='vp_bic' else 'joint_nls_trf'))].iloc[0]
                gate, delta = bool(check.replay_passed), float(check.replay_relative_l2)
            rows.append(dict(dataset='metec', case_id=identifier, group=audit['wind']+'_'+audit['solver'],
                location_m=float(np.linalg.norm(np.asarray(fit['source_norm'])*100-true_xy)),
                rmse=audit['replay_rmse'], q_error=abs(q-truth['mean_q_kg_h'])/truth['mean_q_kg_h'],
                seconds=fit['seconds'], provenance='reused_archive', comparable=audit['reuse_verified'],
                q_count=fit.get('q_basis_count', 5), max_nfev_per_start=250, starts=2,
                numerical_gate_passed=gate, high_order_relative_change=delta,
                **work(fit), source=audit['source']))
    protocol = read(STUDY/'ia_release_capacity_full_v1/protocol.json')
    old_protocol = read(STUDY/'new_comparison_v4/protocol.json')
    save(DEST/'configuration_audit.json', dict(metec_orders=dict(time=16, release=16, history_minutes=20),
        metec_starts='Original common two physical starts and identical bounds from initialization.json',
        metec_budget='250 function evaluations per start per capacity; VP BIC fits three capacities; NLS fixed5',
        original_791_not_used='3 refined events have different orders and one fitted start; use pre-refinement full17 instead',
        archived_spatial_nls='Historical matched wind control reused, not promoted to frozen production baseline',
        synthetic_initialization=protocol['initialization'], synthetic_trf=protocol['trf'],
        synthetic_nls_protocol=old_protocol,
        limitation='Archived fixed5 VP subrecords omit diffusion/prediction/counters; use stored SSE for observed RMSE; no missing counters invented',
        no_new_fits=True))
    for entry in protocol['cases']:
        if entry['case']['cohort'] != 'nominal':
            continue
        case = entry['case']['case_id']
        old_path = STUDY/'new_comparison_v4'/case/'result.json'
        assert digest(old_path) == protocol['input_sha256'][case], case
        old = read(old_path)
        path = STUDY/'ia_release_capacity_full_v1'/case/'result.json'
        selected = read(path)['methods']['ia_vpdi_release_bic']
        fixed = next(f for f in selected['release_models'] if f['q_basis_count'] == 5)
        with np.load(STUDY/entry['parent']/case/'input.npz') as data:
            clock = np.linspace(0, 6, 193)
            true_q = np.interp(clock, data['true_knots'], data['true_coefficients'])
            truth = data['source'].copy()
            n = data['observed'].size
        for group, fit, source in [('nls5', old['methods']['joint_nls_trf'], old_path),
                ('vp5', fixed, path), ('vp_bic', selected, path)]:
            coeff = np.asarray(fit.get('native_coefficients', fit.get('coefficients')))
            q = np.interp(clock, np.linspace(0, 6, len(coeff)), coeff)
            rows.append(dict(dataset='synthetic', case_id=case, group=group,
                location_m=float(np.linalg.norm(np.asarray(fit['source_norm'])-truth)*old['metadata']['length_m']),
                rmse=float(np.sqrt(fit['sse']/n)), q_error=float(np.linalg.norm(q-true_q)/np.linalg.norm(true_q)),
                seconds=fit['seconds'], q_count=len(coeff), provenance='reused_archive', comparable=True,
                max_nfev_per_start=500 if group=='nls5' else 250, starts=2,
                **work(fit), source=str(source.relative_to(ROOT))))
    frame = pd.DataFrame(rows)
    save(DEST/'source_manifest.json', {source:digest(ROOT/source) for source in sorted(set(frame.source))})
    frame.to_csv(DEST/'case_metrics.csv', index=False, encoding='utf-8-sig')
    frame.groupby(['dataset','group']).agg(n=('location_m','size'),
        comparable=('comparable','sum'), mean_m=('location_m','mean'), median_m=('location_m','median'),
        p90_m=('location_m',lambda s: s.quantile(.9)), rmse_mean=('rmse','mean'),
        q_error_mean=('q_error','mean'), seconds_mean=('seconds','mean')).to_csv(DEST/'summary.csv', encoding='utf-8-sig')
    pairs=[]
    for dataset, contrasts in [('metec', [('mean_vp_bic','mean_nls5'),('spatial_vp_bic','spatial_nls5'),
            ('spatial_vp_bic','mean_vp_bic'),('spatial_nls5','mean_nls5')]),
            ('synthetic',[('vp5','nls5'),('vp_bic','vp5')])]:
        for a,b in contrasts:
            aa=frame[(frame.dataset==dataset)&(frame.group==a)].set_index('case_id')
            bb=frame[(frame.dataset==dataset)&(frame.group==b)].set_index('case_id')
            for case in aa.index.intersection(bb.index):
                pairs.append(dict(dataset=dataset, contrast=a+' minus '+b, case_id=case,
                    delta_location_m=aa.loc[case,'location_m']-bb.loc[case,'location_m'],
                    delta_rmse=aa.loc[case,'rmse']-bb.loc[case,'rmse'],
                    delta_q_error=aa.loc[case,'q_error']-bb.loc[case,'q_error'],
                    delta_seconds=aa.loc[case,'seconds']-bb.loc[case,'seconds'],
                    comparable=bool(aa.loc[case,'comparable'] and bb.loc[case,'comparable'])))
    pd.DataFrame(pairs).to_csv(DEST/'paired_differences.csv', index=False, encoding='utf-8-sig')
    print(frame.groupby(['dataset','group']).location_m.agg(['count','mean','median']).to_string(), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    DEST.mkdir(parents=True, exist_ok=True)
    metec()
    assemble()
