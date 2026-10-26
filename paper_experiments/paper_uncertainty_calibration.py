"""One retrospective calibration attempt on frozen, disjoint CASE partitions.

This does not withhold any station or time sample from source inversion. The
nominal 95% target is fixed; validation truth never selects a scale or method.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]/'result/paper_experiments_20261017'


def calibration_threshold(scores, coverage=.95):
    values = np.asarray(scores, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values < 0) or not 0 < coverage < 1):
        raise ValueError('Finite nonnegative case scores and coverage in (0,1) required')
    rank = math.ceil((len(values)+1)*coverage)
    return float(np.sort(values)[rank-1]) if rank <= len(values) else float('inf')


def calibrate(frame):
    selected = frame[(frame.cohort == 'nominal') &
                     (frame.diagnostic == 'eliminate_Q_D')].copy()
    if selected.case_id.duplicated().any() or not selected.available.all():
        raise ValueError('Expected one defined Q/D-profiled ellipse per nominal case')
    calibration = selected[selected['repeat'].between(0, 3)]
    validation = selected[selected['repeat'].between(4, 7)]
    if len(calibration) != 48 or len(validation) != 48:
        raise ValueError('Expected the frozen balanced 48/48 case partition')
    threshold = calibration_threshold(calibration.mahalanobis_squared)
    selected['partition'] = np.where(selected['repeat'] < 4, 'calibration', 'validation')
    selected['raw_threshold'] = 5.991
    selected['calibrated_threshold'] = threshold
    selected['raw_contains_truth'] = selected.mahalanobis_squared <= 5.991
    selected['calibrated_contains_truth'] = selected.mahalanobis_squared <= threshold
    selected['calibrated_major_m'] = selected.major_m*np.sqrt(threshold/5.991)
    selected['axis_scale'] = np.sqrt(threshold/5.991)
    summaries = []
    for partition, group in selected.groupby('partition'):
        for network, sample in [('all', group), *group.groupby('network')]:
            summaries.append(dict(partition=partition, network=network, n=len(sample),
                raw_inclusions=int(sample.raw_contains_truth.sum()),
                calibrated_inclusions=int(sample.calibrated_contains_truth.sum()),
                raw_coverage=float(sample.raw_contains_truth.mean()),
                calibrated_coverage=float(sample.calibrated_contains_truth.mean()),
                raw_major_median_m=float(sample.major_m.median()),
                calibrated_major_median_m=float(sample.calibrated_major_m.median())))
    return selected, pd.DataFrame(summaries), threshold


def run(output):
    output.mkdir(exist_ok=False)
    protocol = dict(target_coverage=.95, attempts=1,
        input='evaluation_v3_shared_forward/analysis/local_diagnostics.csv',
        calibration_repeats=[0,1,2,3], validation_repeats=[4,5,6,7],
        selection='all nominal cases; no selection by errors or method rankings',
        diagnostic='eliminate_Q_D', scope='post-inversion uncertainty only',
        limitation='Retrospective split of existing simulations, not a fresh independent calibration study; finite-sample conformal guarantees require exchangeability.',
        stop_rule='Report validation coverage and ellipse size once, regardless of closeness to 95%; no test-truth tuning')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    frame = pd.read_csv(ROOT/protocol['input'])
    cases, summary, threshold = calibrate(frame)
    cases.to_csv(output/'case_ellipses.csv', index=False)
    summary.to_csv(output/'coverage_summary.csv', index=False)
    report = dict(final=True, attempts=1, threshold=threshold,
        axis_scale=float(np.sqrt(threshold/5.991)),
        calibration_cases=cases.loc[cases.partition=='calibration','case_id'].tolist(),
        validation_cases=cases.loc[cases.partition=='validation','case_id'].tolist(),
        source_estimates_changed=False, no_station_or_time_withholding=True)
    (output/'calibration_result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'uncertainty_calibration_v1')
    run(parser.parse_args().output)
