"""Refresh all paper result tables for the accepted IA version, without fitting."""

import hashlib
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from deliver_best_ia import ROOT, STUDY, PUBLIC, RECOMMENDED, DEST as DELIVERY, read, save, check_baseline
from run_focused_chapter import classify, summarize_flags

OUT = ROOT / 'output/ia_vpdi_report_20261028'
DATA = OUT / 'data'
UPDATED = STUDY / 'paper_updated_20261026'
METHODS = dict(ia_vpdi='IA-VPDI', joint_nls_trf='Joint NLS',
               standard_gbest_pso='标准PSO', matched_joint_adam='联合Adam')
NETWORKS = dict(ideal5='理想五站', jjj6='实际六站几何',
    surround5_pulse='环绕五站', corridor6_double_peak='走廊六站')


def grouped(frame, groups):
    return frame.groupby(groups, sort=False).agg(n=('localization_error_m', 'size'),
        mean_location_m=('localization_error_m', 'mean'), median_location_m=('localization_error_m', 'median'),
        mean_q_error=('q_relative_l2', 'mean'), median_q_error=('q_relative_l2', 'median')).reset_index()


def prepare():
    check_baseline()
    for name, expected in read(DELIVERY / 'manifest.json')['result_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Accepted result changed: ' + name)
    DATA.mkdir(parents=True, exist_ok=True)
    frame = classify(pd.read_csv(DELIVERY / 'synthetic_cases.csv'))
    if len(frame) != 864 or frame.case_id.nunique() != 216:
        raise ValueError('Keep all216 synthetic cases and all four methods')
    frame.to_csv(DATA / 'synthetic_cases.csv', index=False, encoding='utf-8-sig')
    nominal = frame[frame.cohort == 'nominal']
    grouped(frame, ['cohort', 'method']).to_csv(DATA / 'synthetic_summary.csv', index=False, encoding='utf-8-sig')
    grouped(nominal, ['network', 'method']).to_csv(DATA / 'network_summary.csv', index=False, encoding='utf-8-sig')
    grouped(nominal, ['release', 'method']).to_csv(DATA / 'release_summary.csv', index=False, encoding='utf-8-sig')
    summarize_flags(nominal, ['method']).to_csv(DATA / 'wrong_location_summary.csv', index=False, encoding='utf-8-sig')
    sensitivity = []
    for threshold in [50., 100., 200.]:
        part = summarize_flags(classify(nominal, threshold), ['method'])
        part['location_threshold_m'] = threshold
        sensitivity.append(part)
    pd.concat(sensitivity).to_csv(DATA / 'threshold_sensitivity.csv', index=False, encoding='utf-8-sig')
    entries = {item['case']['case_id']: item for item in read(UPDATED / 'synthetic/protocol.json')['cases']}
    rows, sources = [], [DELIVERY / 'manifest.json', DELIVERY / 'synthetic_cases.csv', DELIVERY / 'public_cases.csv']
    for case in nominal.case_id.unique():
        path = STUDY / 'ia_release_capacity_full_v1' / case / 'result.json'
        record = read(path)
        fit = record['methods']['ia_vpdi_release_bic']
        with np.load(STUDY / entries[case]['parent'] / case / 'input.npz') as inputs:
            times = np.linspace(0., 6., 193)
            true = np.interp(times, inputs['true_knots'], inputs['true_coefficients'])
            origin = inputs['source'].copy()
        meta = record['metadata']
        for model in [*fit['release_models'], dict(q_basis_count=fit['q_basis_count'], source_norm=fit['source_norm'],
                        native_coefficients=fit['native_coefficients'], sse=fit['sse'], selected=True)]:
            a = np.asarray(model['native_coefficients'])
            q = np.interp(times, np.linspace(0., 6., len(a)), a)
            location = float(np.linalg.norm(np.asarray(model['source_norm']) - origin) * meta['length_m'])
            q_error = float(np.linalg.norm(q - true) / np.linalg.norm(true))
            label = 'adaptive_bic' if model.get('selected') else f'fixed_{model["q_basis_count"]}'
            rows.append(dict(case_id=case, release=meta['release'], mode=label,
                q_basis_count=model['q_basis_count'], localization_error_m=location, q_relative_l2=q_error))
            if model.get('selected'):
                existing = nominal[nominal.case_id.eq(case) & nominal.method.eq('ia_vpdi')].iloc[0]
                np.testing.assert_allclose([location, q_error], [existing.localization_error_m, existing.q_relative_l2],
                                           atol=1e-8, rtol=1e-8)
        sources.append(path)
    capacities = pd.DataFrame(rows)
    capacities.to_csv(DATA / 'capacity_cases.csv', index=False, encoding='utf-8-sig')
    grouped(capacities, ['mode']).to_csv(DATA / 'capacity_summary.csv', index=False, encoding='utf-8-sig')
    capacities[capacities['mode'] == 'adaptive_bic'].groupby(['release', 'q_basis_count']).size().rename('n').reset_index().to_csv(
        DATA / 'capacity_choices.csv', index=False, encoding='utf-8-sig')
    public = pd.read_csv(DELIVERY / 'public_cases.csv', dtype={'case_id': str})
    if len(public) != 68 or public.case_id.nunique() != 17:
        raise ValueError('Keep all17 METEC events')
    public.to_csv(DATA / 'metec_cases.csv', index=False, encoding='utf-8-sig')
    public.rename(columns={'q_relative_error': 'q_relative_l2'}).pipe(grouped, ['method']).to_csv(
        DATA / 'metec_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(read(RECOMMENDED / 'status.json')['numerical_rows']).to_csv(DATA / 'metec_ia_numerics.csv', index=False)
    baseline_qa = pd.read_csv(PUBLIC / 'analysis/replicate_metrics.csv', dtype={'event_id': str})
    baseline_qa[baseline_qa.method != 'ia_vpdi'].to_csv(DATA / 'metec_frozen_baseline_checks.csv', index=False)
    for relative, target in [('public/analysis/case_metrics.csv', 'chilbolton_cases.csv'),
                             ('public/analysis/quadrature_replay.csv', 'chilbolton_numerics.csv')]:
        source = UPDATED / relative
        shutil.copy2(source, DATA / target)
        sources.append(source)
    beam = pd.read_csv(DATA / 'chilbolton_cases.csv')
    if len(beam) != 48 or beam[['event', 'height_m']].drop_duplicates().shape[0] != 16:
        raise ValueError('Expected all8 Chilbolton windows at both heights and three methods')
    beam.groupby(['height_m', 'method']).agg(n=('localization_error_m', 'size'),
        mean_location_m=('localization_error_m', 'mean'), median_location_m=('localization_error_m', 'median'),
        mean_q_error=('q_relative_error', 'mean'), median_q_error=('q_relative_error', 'median')).reset_index().to_csv(
            DATA / 'chilbolton_summary.csv', index=False)
    newman_path = STUDY / 'newman_lhs_full_v1/analysis/case_metrics.csv'
    newman = pd.read_csv(newman_path)
    if len(newman) != 16:
        raise ValueError('Preserve all16 author-method runs, including stuck chains')
    newman.to_csv(DATA / 'newman_cases.csv', index=False, encoding='utf-8-sig')
    newman.groupby('height_m').agg(n=('localization_error_m', 'size'),
        mean_location_m=('localization_error_m', 'mean'), median_location_m=('localization_error_m', 'median')).reset_index().to_csv(
            DATA / 'newman_summary.csv', index=False)
    sources.append(newman_path)
    for name in ['local_diagnostics.csv', 'calibrated_ellipses.csv', 'coverage_summary.csv', 'calibration.json',
                 'risk_coverage.csv', 'sampled_alternatives.csv', 'alternatives_status.json']:
        source = UPDATED / 'synthetic/analysis' / name
        shutil.copy2(source, DATA / name)
        sources.append(source)
    from build_paper_comparison_report import check_inputs
    legacy = check_inputs(UPDATED / 'synthetic', UPDATED / 'public', STUDY / 'newman_lhs_full_v1',
                          UPDATED / 'figures', allow_numerical_warnings=True)
    if not legacy['ready']:
        raise ValueError('Existing chapter evidence incomplete: ' + str(legacy['missing']))
    save(OUT / 'legacy_input_check.json', dict(ready=legacy['ready'], missing=legacy['missing'],
        numerical_warnings_allowed=True, removed_matched_capacity_control_not_used_in_new_report=True))
    status = dict(date='2026-10-28', ready_for_report=True, fitting_performed=False,
        synthetic_cases=216, metec_events=17, metec_ia_numerically_passed=17,
        chilbolton_event_height_pairs=16, newman_runs=16,
        chilbolton_numerical_check=read(UPDATED / 'public/analysis/quadrature_replay_status.json'),
        newman_status=read(STUDY / 'newman_lhs_full_v1/analysis/analysis_status.json'),
        chapter_updates={'4.1': 'Method/version and comparison conditions refreshed',
            '4.2': 'All216 existing best-version fits checked; statistics regenerated, no refit',
            '4.3': 'Wrong-location flags and saved1/3/5-capacity ablation recomputed; removed control stays excluded',
            '4.4': 'All17 METEC IA rows replaced by validated spatial-wind results; baselines unchanged',
            'supplement': 'All Chilbolton/Newman and96-case reliability records retained unchanged with quality flags'},
        inference='Complete-cohort mean advantage; no statistical-significance or pure-solver claim',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    save(OUT / 'data_status.json', status)
    print('DATA_REFRESH_READY', {k: status[k] for k in ['synthetic_cases', 'metec_events', 'chilbolton_event_height_pairs', 'newman_runs']})


if __name__ == '__main__':
    prepare()
