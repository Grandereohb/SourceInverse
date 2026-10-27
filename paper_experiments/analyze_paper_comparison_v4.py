"""Read-only evaluation of the new baseline cohort, including partial status.

PSO replicates are reduced within each input case before case-level summaries.
No winner selection, truth-dependent source selection or additional optimization.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_paper_formal as original
from paper_uncertainty_calibration import calibrate

METRICS = original.METRICS


def ellipse_size(information, variance, length_m, available):
    if not available:
        return dict(minor_m=None, area_m2=None)
    eigenvalues = np.linalg.eigvalsh(np.asarray(information, dtype=float))
    axes = np.sqrt(5.991*variance/eigenvalues)*length_m
    return dict(minor_m=float(axes.min()), area_m2=float(np.pi*np.prod(axes)))


def risk_coverage(frame):
    records = []
    for (network, label), group in frame.groupby(['network', 'diagnostic']):
        ordered = group.assign(sort_risk=group.major_m.fillna(np.inf)).sort_values(
            ['sort_risk', 'case_id'], kind='stable')
        for retained in range(1, len(ordered)+1):
            sample = ordered.iloc[:retained]
            records.append(dict(network=network, diagnostic=label, retained=retained,
                n=len(ordered), coverage=retained/len(ordered),
                mean_error_m=float(sample.localization_error_m.mean()),
                median_error_m=float(sample.localization_error_m.median())))
    return pd.DataFrame(records)


def method_rows(record):
    meta = record['metadata']
    identity = {key: meta[key] for key in ('case_id', 'cohort', 'network', 'wind', 'release', 'repeat')}
    rows, replicates = [], []
    for method, value in record['methods'].items():
        fits = value.get('replicates', [value])
        for fit in fits:
            replicates.append(dict(identity, method=method, seed=fit.get('seed'),
                **{key: float(fit[key]) for key in METRICS}))
        row = dict(identity, method=method, replicates=len(fits),
            q_basis_count=fits[0].get('q_basis_count', len(fits[0].get('coefficients', []))),
            source_at_boundary=bool(any(fit['source_at_boundary'] for fit in fits)),
            optimizer_converged=value.get('converged'),
            seconds_all_replicates=float(sum(fit['seconds'] for fit in fits)))
        for key in METRICS:
            values = np.asarray([fit[key] for fit in fits], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite metric: {identity["case_id"]}/{method}/{key}')
            row[key] = float(np.median(values))
        rows.append(row)
    return rows, replicates


def analyze(study):
    study = Path(study)
    protocol = json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    rows, replicates, diagnostics, status = [], [], [], []
    for entry in protocol['cases']:
        spec = entry['case']
        result_path = study/spec['case_id']/'result.json'
        if not result_path.exists():
            status.append(dict(case_id=spec['case_id'], state='not_completed'))
            continue
        record = json.loads(result_path.read_text(encoding='utf-8'))
        if record['case_id'] != spec['case_id'] or not record.get('complete'):
            raise ValueError(f'Invalid result identity/status: {result_path}')
        method_values, runs = method_rows(record)
        if set(record['methods']) != set(protocol['methods']):
            raise ValueError(f'Missing/unexpected baseline: {result_path}')
        rows.extend(method_values)
        replicates.extend(runs)
        info = record['conditional_information']
        displacement = np.asarray(record['metadata']['source_norm'])-record['methods']['ia_vpdi']['source_norm']
        for label, information in (
                ('naive_fixed_Q_D', np.asarray(info['full_information'])[:2, :2]),
                ('eliminate_Q', np.asarray(info['profiled_theta_information'])[:2, :2]),
                ('eliminate_Q_D', info['profiled_source_information'])):
            diagnostic = original.ellipse_diagnostic(information, info['sigma2'], displacement,
                                                      record['metadata']['length_m'])
            diagnostics.append(dict(spec, diagnostic=label,
                localization_error_m=record['methods']['ia_vpdi']['localization_error_m'],
                **diagnostic, **ellipse_size(information, info['sigma2'],
                    record['metadata']['length_m'], diagnostic['available'])))
        status.append(dict(case_id=spec['case_id'], state='completed'))
    destination = study/'analysis'
    destination.mkdir(exist_ok=True)
    pd.DataFrame(status).to_csv(destination/'case_status.csv', index=False)
    final = len(rows) == len(protocol['cases'])*len(protocol['methods']) and not protocol['technical_smoke']
    reliability_final = False
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(destination/'case_metrics.csv', index=False)
        pd.DataFrame(replicates).to_csv(destination/'replicate_metrics.csv', index=False)
        original.summarize(frame).to_csv(destination/'group_summary.csv', index=False)
        local = pd.DataFrame(diagnostics)
        local.to_csv(destination/'local_diagnostics.csv', index=False)
        if final:
            original.paired_effects(frame).to_csv(destination/'paired_effects.csv', index=False)
        # This fixed96-case analysis is independent of later supplementary fits.
        expected_nominal = {entry['case']['case_id'] for entry in protocol['cases']
            if entry['case']['cohort'] == 'nominal' and entry['case']['network'] in ('ideal5', 'jjj6')}
        subset = local[(local.cohort == 'nominal') & local.network.isin(['ideal5', 'jjj6'])]
        reliability_final = (len(expected_nominal) == 96 and
            set(subset.case_id) == expected_nominal and not protocol['technical_smoke'])
        if reliability_final:
            # Preserve the original two-network48/48 calibration partition.
            risk_coverage(subset).to_csv(destination/'risk_coverage.csv', index=False)
            calibrated, summary, threshold = calibrate(subset)
            calibrated['calibrated_area_m2'] = calibrated.area_m2*(threshold/5.991)
            for index, row in summary.iterrows():
                sample = calibrated[calibrated.partition == row['partition']]
                if row['network'] != 'all':
                    sample = sample[sample.network == row['network']]
                summary.loc[index, 'raw_area_median_m2'] = sample.area_m2.median()
                summary.loc[index, 'calibrated_area_median_m2'] = sample.calibrated_area_m2.median()
            calibrated.to_csv(destination/'calibrated_ellipses.csv', index=False)
            summary.to_csv(destination/'coverage_summary.csv', index=False)
            (destination/'calibration.json').write_text(json.dumps(dict(
                threshold=threshold, axis_scale=float(np.sqrt(threshold/5.991)),
                retrospective=True, calibration_cases=48, validation_cases=48,
                unavailable_policy='positive-infinity calibration score; retained denominator; no ellipse counted as covered',
                defined_calibration_cases=int(calibrated.loc[calibrated.partition == 'calibration', 'available'].sum()),
                station_or_time_withholding=False), indent=2), encoding='utf-8')
    report = dict(final=final, planned_cases=len(protocol['cases']),
        reliability_final=reliability_final,
        completed_cases=sum(row['state'] == 'completed' for row in status),
        methods=protocol['methods'], raw_replicate_rows=len(replicates),
        case_method_rows=len(rows), technical_smoke=protocol['technical_smoke'],
        pso_summary='median across fixed seeds within each case, then summarize cases',
        comparison_scope='common finite domain and coarse warm starts; conditional refinement timing',
        interpretation='partial sequential cases do not establish overall method rankings')
    (destination/'analysis_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    print(json.dumps(analyze(parser.parse_args().study), indent=2))
