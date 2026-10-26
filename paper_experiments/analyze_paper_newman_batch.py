"""Summarize fixed-budget author chains separately from optimizer comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_paper_beam_comparison import reference_table

ROOT = Path(__file__).resolve().parents[1]/'result/paper_experiments_20261017'


def chain_summary(record, audit):
    position = np.concatenate([np.asarray(record['effective_sample_size'][key]).ravel()
                               for key in ('source_x', 'source_y')])
    source_strength = np.asarray(record['effective_sample_size']['s']).ravel()
    position_ess = float(position.min()) if np.isfinite(position).all() else None
    q_ess = float(source_strength.min()) if np.isfinite(source_strength).all() else None
    transition = audit['chains']['mh_block']['retained_transition_fraction']
    status = ('stuck_retained_mh_chain' if transition == 0. else
              'undefined_ess' if position_ess is None or q_ess is None else
              'fixed_budget_single_chain')
    return dict(position_min_ess=position_ess, q_ess=q_ess, sampling_status=status,
                mh_retained_transition_fraction=transition,
                retained_unique_mh_states=audit['chains']['source_x']['distinct_retained_states'],
                posterior_convergence_certified=False)


def analyze(folder, references):
    folder = Path(folder)
    protocol = json.loads((folder/'batch_protocol.json').read_text(encoding='utf-8'))
    measured = reference_table(references)
    rows, status = [], []
    for case in protocol['events']:
        directory = folder/case['directory']
        path, audit_path = directory/'result.json', directory/'chain_audit.json'
        state = dict(case, state='pending')
        if path.exists() and audit_path.exists():
            record = json.loads(path.read_text(encoding='utf-8'))
            audit = json.loads(audit_path.read_text(encoding='utf-8'))
            if (record['event'] != case['event'] or record['height_m'] != case['height_m'] or
                    not record['finite_required_chains'] or record['iterations'] != protocol['iterations'] or
                    audit['result_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError('Author case identity, budget, chain or audit mismatch: '+case['directory'])
            ref = measured[case['event']]
            estimate = record['posterior_mean_q_kg_h']
            row = dict(event=case['event'], source_group=case['event'].split('_')[0],
                height_m=case['height_m'], method='newman_author_mmala',
                localization_error_m=record['localization_error_m'], mean_q_kg_h=estimate,
                **ref, q_relative_error=(abs(estimate-ref['reference_q_kg_h'])/ref['reference_q_kg_h']
                    if ref['reference_q_kg_h'] else None), seconds=record['seconds'],
                parameter_mean_forward_rmse_ppm=audit['parameter_mean_forward_rmse_ppm'],
                **chain_summary(record, audit))
            numeric = [row[key] for key in ('localization_error_m', 'mean_q_kg_h',
                'seconds', 'parameter_mean_forward_rmse_ppm', 'mh_retained_transition_fraction')]
            if not np.isfinite(numeric).all():
                raise ValueError('Nonfinite reported author metrics: '+case['directory'])
            rows.append(row)
            state['state'] = 'completed_with_chain_audit'
        status.append(state)
    output = folder/'analysis'
    output.mkdir(exist_ok=True)
    pd.DataFrame(status).to_csv(output/'case_status.csv', index=False)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(output/'case_metrics.csv', index=False)
        summaries = []
        for height, group in frame.groupby('height_m'):
            summary = dict(height_m=height, events=len(group), source_locations=group.source_group.nunique())
            for metric in ('localization_error_m', 'q_relative_error', 'parameter_mean_forward_rmse_ppm',
                           'seconds', 'position_min_ess', 'mh_retained_transition_fraction'):
                values = group[metric].dropna().to_numpy(dtype=float)
                summary[metric+'_n'] = len(values)
                for label, quantile in [('q25', .25), ('median', .5), ('q75', .75)]:
                    summary[metric+'_'+label] = float(np.quantile(values, quantile)) if len(values) else None
            summaries.append(summary)
        pd.DataFrame(summaries).to_csv(output/'group_summary.csv', index=False)
    report = dict(final=len(rows) == len(protocol['events']), completed_event_heights=len(rows),
        planned_event_heights=len(protocol['events']),
        undefined_position_ess=sum(row['position_min_ess'] is None for row in rows),
        stuck_retained_chains=sum(row['sampling_status'] == 'stuck_retained_mh_chain' for row in rows),
        interpretation='batch completion is NOT posterior convergence certification',
        point_estimator='single fixed-budget chain mean after half-budget burn-in',
        q_reference='independent measured reference only; absent values retained as missing',
        method_comparison='separate author plume/prior/background track, not same-forward optimizer ranking')
    (output/'analysis_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT/'newman_lhs_full_v1')
    parser.add_argument('--references', type=Path,
                        default=ROOT/'manuscript_results_v2/controlled_release_case_metrics.csv')
    args = parser.parse_args()
    print(json.dumps(analyze(args.study, args.references), indent=2))
