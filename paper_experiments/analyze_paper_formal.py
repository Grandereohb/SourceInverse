"""Case-level evaluation tables; partial runs are explicitly not final evidence.

Reads frozen inputs/results only. Does not tune, rerun or alter inversion.
"""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


METRICS = ('localization_error_m', 'q_relative_l2',
           'concentration_rmse_clean', 'observed_mse', 'seconds')


def source_information(full, nuisance):
    """Schur projection using the stored Gram matrix, not new forward solves."""
    full = np.asarray(full, dtype=float)
    answer = full[:2, :2].copy()
    if len(nuisance):
        cross = full[np.ix_([0, 1], nuisance)]
        answer -= cross @ np.linalg.pinv(full[np.ix_(nuisance, nuisance)],
                                         rcond=1e-10) @ cross.T
    return (answer + answer.T) * .5


def ellipse_diagnostic(information, variance, displacement, length_m):
    information = np.asarray(information, dtype=float)
    eigenvalues = np.linalg.eigvalsh(information)
    if (not np.isfinite(information).all() or not variance > 0
            or eigenvalues[0] <= 1e-10 * max(eigenvalues[-1], 1e-300)):
        return dict(available=False, major_m=None, contains_truth=None,
                    mahalanobis_squared=None)
    covariance = variance * np.linalg.inv(information)
    distance = float(np.asarray(displacement) @ information @ displacement / variance)
    return dict(available=True,
                major_m=float(np.sqrt(5.991 * np.linalg.eigvalsh(covariance)[-1]) * length_m),
                contains_truth=bool(distance <= 5.991), mahalanobis_squared=distance)


def summarize(frame):
    records = []
    for keys, group in frame.groupby(['cohort', 'network', 'method'], sort=True):
        row = dict(zip(('cohort', 'network', 'method'), keys))
        row.update(n=len(group), boundary_count=int(group.source_at_boundary.sum()),
                   optimizer_nonconvergence_count=int(group.optimizer_converged.eq(False).sum()),
                   optimizer_convergence_unverified_count=int(group.optimizer_converged.isna().sum()))
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row.update({f'{metric}_median': float(np.median(values)),
                        f'{metric}_q25': float(np.quantile(values, .25)),
                        f'{metric}_q75': float(np.quantile(values, .75))})
        records.append(row)
    return pd.DataFrame(records)


def sampled_alternatives(case, result):
    """Finite evaluated sets only, never a confidence-region construction."""
    archive = np.load(case/'optimization_traces.npz')
    surface = archive['candidate_surface']
    best = result['methods']['ia_vpdi']
    source = np.asarray(best['source_norm'])
    reference = float(best['observed_mse'])
    if np.min(surface[:, 4]) < reference-1e-8:
        raise ValueError(f'Selected IA score worse than coarse search: {case}')
    points = np.vstack((surface[:, :2], source))
    scores = np.r_[surface[:, 4], reference]
    records = []
    for tolerance in (.01, .05, .10):
        selected = points[scores <= reference*(1+tolerance)+1e-15]
        selected = np.unique(selected, axis=0)
        records.append(dict(case_id=result['case_id'], relative_score_tolerance=tolerance,
            sampled_locations=len(selected),
            farthest_alternative_m=float(np.linalg.norm(selected-source, axis=1).max()
                                         * result['metadata']['length_m']),
            sampled_boundary_reached=bool(np.any(np.isclose(np.abs(selected), .75, atol=1e-6))),
            interpretation='coarse D-grid plus selected optimum only; not global profile or confidence region'))
    return records


def paired_effects(frame, repeats=10000):
    """Case-paired effects; resample within wind/release design strata."""
    rng = np.random.default_rng(20261017)
    records = []
    for (cohort, network), group in frame.groupby(['cohort', 'network'], sort=True):
        metadata = group.drop_duplicates('case_id').set_index('case_id')[['wind', 'release']]
        for metric in ('localization_error_m', 'q_relative_l2'):
            wide = group.pivot(index='case_id', columns='method', values=metric).join(metadata)
            for comparator in sorted(set(wide.columns) - {'ia_vpdi', 'wind', 'release'}):
                strata = [v.ia_vpdi.to_numpy()-v[comparator].to_numpy()
                          for _, v in wide.groupby(['wind', 'release'], sort=True)]
                paired = np.concatenate(strata)
                row = dict(cohort=cohort, network=network, comparator=comparator, metric=metric,
                    cases=len(paired), IA_minus_comparator_median=float(np.median(paired)),
                    IA_minus_comparator_mean=float(np.mean(paired)),
                    IA_better_count=int(np.count_nonzero(paired < 0)),
                    interval_low=None, interval_high=None,
                    interval_method='not estimated: fewer than2 cases in a design stratum')
                if all(len(v) >= 2 for v in strata):
                    draws = np.concatenate([v[rng.integers(len(v), size=(repeats, len(v)))] for v in strata], axis=1)
                    lower, upper = np.quantile(np.median(draws, axis=1), [.025, .975])
                    row.update(interval_low=float(lower), interval_high=float(upper),
                        interval_method=f'95% percentile bootstrap of paired median, {repeats} draws within wind/release strata, seed20261017')
                records.append(row)
    return pd.DataFrame(records)


def analyze(study):
    protocol = json.loads((study / 'protocol.json').read_text(encoding='utf-8'))
    hashes = {f'{spec["case_id"]}/{name}': hashlib.sha256(
        (study/spec['case_id']/name).read_bytes()).hexdigest()
        for spec in protocol['case_design'] for name in ('input.json', 'input.npz')}
    manifest = study/'input_hash_manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))['sha256'] != hashes:
            raise ValueError('Evaluation inputs changed since their integrity manifest')
    else:
        manifest.write_text(json.dumps(dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            note='Integrity record created during the run, not a claim of pre-run external registration.',
            sha256=hashes), indent=2), encoding='utf-8')
    rows, diagnostic_rows, status, alternatives, characteristics = [], [], [], [], []
    for spec in protocol['case_design']:
        case = study / spec['case_id']
        path = case / 'result.json'
        state = 'completed' if path.exists() else ('stopped' if (case / 'STOPPED.json').exists() else 'pending')
        status.append(dict(**spec, state=state))
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding='utf-8'))
        metadata = result['metadata']
        if result['case_id'] != spec['case_id'] or metadata['seed'] != spec['seed']:
            raise ValueError(f'Case identity mismatch: {case}')
        with np.load(case/'input.npz') as data:
            clean = data['clean'].reshape(6, len(data['stations']))
            peaks = np.sort(clean.max(axis=0))
            ratio = float(peaks[-2]/peaks[-1])
            np.testing.assert_allclose(ratio, metadata['station_peak_ratio_second_to_first'],
                                       rtol=1e-10, atol=1e-12)
            characteristics.append(dict(**spec, station_count=len(peaks),
                peak_ratio_second_to_first=ratio, clean_peak=float(peaks[-1]),
                noise_std=metadata['noise_std'],
                stations_with_clean_peak_above_3sigma=int(np.sum(peaks>3*metadata['noise_std'])),
                interpretation='Post-fit case description only; no case exclusion or inversion weighting'))
        for method, value in result['methods'].items():
            row = dict(**spec, method=method,
                       source_at_boundary=bool(value['source_at_boundary']),
                       optimizer_converged=value.get('converged'),
                       termination_rule='optimizer_reported' if 'converged' in value else 'fixed1000_updates_per_start')
            for metric in METRICS:
                row[metric] = value.get('seconds_including_common_search', value.get('seconds')) if metric == 'seconds' else value[metric]
                if not np.isfinite(row[metric]):
                    raise ValueError(f'Nonfinite primary metric: {case}/{method}/{metric}')
            rows.append(row)
        ia = result['methods']['ia_vpdi']
        alternatives.extend(sampled_alternatives(case, result))
        info = result['conditional_information']
        full = np.asarray(info['full_information'])
        displacement = np.asarray(metadata['source_norm']) - ia['source_norm']
        for label, matrix in (
                ('naive_fixed_Q_D', full[:2, :2]),
                ('eliminate_Q', np.asarray(info['profiled_theta_information'])[:2, :2]),
                ('eliminate_Q_D', np.asarray(info['profiled_source_information']))):
            diagnostic_rows.append(dict(**spec, diagnostic=label,
                localization_error_m=ia['localization_error_m'],
                **ellipse_diagnostic(matrix, info['sigma2'], displacement, metadata['length_m'])))
    output = study / 'analysis'
    output.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    pd.DataFrame(status).to_csv(output / 'case_status.csv', index=False)
    pd.DataFrame(characteristics).to_csv(output/'case_characteristics.csv', index=False)
    if rows:
        frame.to_csv(output / 'case_metrics.csv', index=False)
        summarize(frame).to_csv(output / 'group_summary.csv', index=False)
        diagnostic = pd.DataFrame(diagnostic_rows)
        diagnostic.to_csv(output / 'local_diagnostics.csv', index=False)
        pd.DataFrame(alternatives).to_csv(output/'sampled_alternatives.csv', index=False)
        risks = []
        for (cohort, network, label), group in diagnostic.groupby(['cohort', 'network', 'diagnostic']):
            # Undefined/rank-deficient diagnostics are retained as highest risk.
            ordered = group.assign(sort_risk=group.major_m.fillna(np.inf)).sort_values(
                ['sort_risk', 'case_id'], kind='stable')
            for retained in range(1, len(ordered) + 1):
                selected = ordered.iloc[:retained]
                risks.append(dict(cohort=cohort, network=network, diagnostic=label,
                    retained=retained, available_cases=len(ordered), coverage=retained / len(ordered),
                    mean_error_m=float(selected.localization_error_m.mean()),
                    median_error_m=float(selected.localization_error_m.median()),
                    unavailable_count=int((~selected.available).sum())))
        pd.DataFrame(risks).to_csv(output / 'risk_coverage.csv', index=False)
        paired = frame.pivot(index='case_id', columns='method', values='localization_error_m')
        paired['IA_minus_joint_m'] = paired.ia_vpdi - paired.matched_joint_adam
        if 'shared_response_nnls_powell' in paired:
            paired['IA_minus_shared_powell_m'] = paired.ia_vpdi - paired.shared_response_nnls_powell
        else:
            paired['IA_minus_puff_m'] = paired.ia_vpdi - paired.independent_puff_demo
        paired.to_csv(output / 'paired_localization.csv')
    completed = sum(item['state'] == 'completed' for item in status)
    if completed == len(status):
        paired_effects(frame).to_csv(output/'paired_effects.csv', index=False)
    report = dict(planned_cases=len(status), completed_cases=completed,
        stopped_cases=[item['case_id'] for item in status if item['state'] == 'stopped'],
        final=completed == len(status), protocol_sha256=hashlib.sha256((study/'protocol.json').read_bytes()).hexdigest(),
        interpretation=[
            'Partial sequential results are not balanced final evaluation; do not infer overall superiority.',
            'The independent unit is a generated case, not an observation or optimization step.',
            'Local chi-square ellipses use fitted residual variance; empirical inclusion is not proven calibration.',
            'Risk curves are descriptive within prespecified cohort/network, not a newly tuned acceptance gate.',
            protocol.get('timing', 'Timing settings must be checked against the frozen protocol.'),
            'IA/joint terminate at a fixed budget; absence of an exception is not convergence proof.',
            'The legacy source_at_boundary flag means abs(coordinate)>=.74 for bounds+/- .75: a near-boundary flag, not exact equality.',
            protocol.get('baselines', 'Puff is an adapted independent demo, not a named-paper replication.') ])
    (output/'analysis_status.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return frame, diagnostic_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path(__file__).resolve().parents[1] /
                        'result/paper_experiments_20261017/evaluation_v1')
    args = parser.parse_args()
    analyze(args.study.resolve())
