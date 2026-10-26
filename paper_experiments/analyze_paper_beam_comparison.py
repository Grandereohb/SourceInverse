"""Collect the fixed public-release comparison without selecting winning seeds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDY = Path(__file__).resolve().parents[1]/'result/paper_experiments_20261017'
METRICS = ('localization_error_m', 'observed_rmse_ppm', 'mean_q_kg_h', 'seconds')


def reference_table(path):
    frame = pd.read_csv(path)
    references = {}
    for event, group in frame.groupby('event'):
        finite = group.reference_q_kg_h.dropna().to_numpy()
        if finite.size and not np.allclose(finite, finite[0], rtol=1e-12, atol=0.):
            raise ValueError(f'Conflicting measured release references: {event}')
        statuses = group.q_reference_status.unique().tolist()
        if len(statuses) != 1:
            raise ValueError(f'Conflicting reference provenance: {event}')
        references[event] = dict(reference_q_kg_h=float(finite[0]) if finite.size else None,
                                  q_reference_status=statuses[0])
    return references


def method_rows(record, references):
    if not record['complete']:
        raise ValueError('Cannot summarize an incomplete event')
    reference = references[record['event']]
    shared = dict(event=record['event'], height_m=record['height_m'],
                  source_group=record['event'].split('_')[0], **reference)
    rows, raw = [], []
    for method, result in record['methods'].items():
        fits = result.get('replicates', [result])
        values = []
        for fit in fits:
            numeric = np.array([fit[key] for key in METRICS], dtype=float)
            if not np.isfinite(numeric).all():
                raise ValueError(f'Nonfinite public-release metrics: {shared}, {method}')
            row = dict(shared, method=method, seed=fit.get('seed'),
                       **{key: float(fit[key]) for key in METRICS})
            ref = reference['reference_q_kg_h']
            row['q_relative_error'] = abs(row['mean_q_kg_h']-ref)/ref if ref else None
            raw.append(row)
            values.append(row)
        aggregate = dict(shared, method=method, replicates=len(values),
                         seconds_all_replicates=sum(row['seconds'] for row in values))
        for key in (*METRICS, 'q_relative_error'):
            samples = [row[key] for row in values]
            aggregate[key] = None if samples[0] is None else float(np.median(samples))
        rows.append(aggregate)
    return rows, raw


def analyze(study, references_path):
    study = Path(study)
    protocol = json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    references = reference_table(references_path)
    rows, raw, status = [], [], []
    for event in protocol['events']:
        name = event['source'].replace('_', '')+'_'+event['release']
        for height in protocol['heights_m']:
            path = study/name/f'height_{height:.1f}'/'result.json'
            state = dict(event=name, height_m=height, state='pending')
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                if record['technical_smoke'] != protocol['technical_smoke']:
                    raise ValueError('Mixed smoke and formal public-release results')
                if set(record['methods']) != set(protocol['methods']):
                    raise ValueError('Public-release method set does not match protocol')
                case_rows, case_raw = method_rows(record, references)
                rows.extend(case_rows)
                raw.extend(case_raw)
                state['state'] = 'completed'
            status.append(state)
    destination = study/'analysis'
    destination.mkdir(exist_ok=True)
    pd.DataFrame(status).to_csv(destination/'case_status.csv', index=False)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(destination/'case_metrics.csv', index=False)
        pd.DataFrame(raw).to_csv(destination/'replicate_metrics.csv', index=False)
        summaries = []
        for keys, group in frame.groupby(['height_m', 'source_group', 'method']):
            summary = dict(zip(('height_m', 'source_group', 'method'), keys))
            summary['event_count'] = len(group)
            for metric in (*METRICS, 'q_relative_error'):
                values = group[metric].dropna().to_numpy(dtype=float)
                summary[metric+'_count'] = len(values)
                for label, quantile in [('q25', .25), ('median', .5), ('q75', .75)]:
                    summary[metric+'_'+label] = float(np.quantile(values, quantile)) if len(values) else None
            summaries.append(summary)
        pd.DataFrame(summaries).to_csv(destination/'group_summary.csv', index=False)
    completed = sum(item['state'] == 'completed' for item in status)
    report = dict(final=completed == len(status) and not protocol['technical_smoke'],
                  completed_event_heights=completed, planned_event_heights=len(status),
                  case_method_rows=len(rows), raw_replicate_rows=len(raw),
                  pso_summary='within-event-height median of all fixed seeds, never best seed',
                  grouping='eight windows at two source locations; heights are sensitivity analyses, not replicates',
                  q_reference='existing independent per-window measured references only; absent references stay missing',
                  newman_scope='separate author-forward comparison; not mixed into common-forward optimizer ranking')
    (destination/'analysis_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--references', type=Path,
                        default=STUDY/'manuscript_results_v2/controlled_release_case_metrics.csv')
    args = parser.parse_args()
    print(json.dumps(analyze(args.study, args.references), indent=2))
