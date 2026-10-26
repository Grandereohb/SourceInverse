"""Matched constant-Q supplemental figure, retaining all sampling failures.

Contract: compare location and amplitude errors at identical Q capacity across
two geometries. Four paired cases per geometry are displayed individually;
no calibrated posterior coverage or named-paper reproduction is claimed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ('ia_constant', 'joint_constant', 'puff_mle_constant', 'bayesian_constant')
LABELS = ('IA', 'Joint', 'Puff MLE', 'Bayes mean')
COLORS = ('#197A9A', '#55585B', '#BC8625', '#875B85')


def plot(analysis):
    analysis = Path(analysis)
    status = json.loads((analysis/'analysis_status.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(analysis/'case_metrics.csv', dtype={
        'diagnostics_passed': 'boolean', 'numerical_replay_passed': 'boolean'})
    if not status['final'] or status['completed_cases'] != 8 or len(frame) != 32:
        raise ValueError('Only the complete eight-case cohort may be plotted')
    if frame.duplicated(['case_id', 'method']).any():
        raise ValueError('Duplicate method-case records')
    for _, case in frame.groupby('case_id'):
        if set(case.method) != set(METHODS):
            raise ValueError('All four matched methods are required')
    if not np.isfinite(frame[['localization_error_m', 'q_relative_error']]).all().all():
        raise ValueError('Nonfinite errors cannot be silently omitted')
    output = analysis/'figures'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    fig.subplots_adjust(left=.12, right=.97, bottom=.13, top=.87, wspace=.34, hspace=.49)
    failed = []
    for col, network in enumerate(('ideal5', 'jjj6')):
        selected = frame[frame.network == network]
        if selected.case_id.nunique() != 4:
            raise ValueError('Expected four paired cases in each geometry')
        for row, metric in enumerate(('localization_error_m', 'q_relative_error')):
            ax = axes[row, col]
            for case_id, rows in selected.groupby('case_id', sort=True):
                rows = rows.set_index('method').loc[list(METHODS)]
                values = rows[metric].to_numpy()
                ax.plot(np.arange(4), values, color='#C1C3C4', lw=.75, zorder=1)
                for index, (method, color) in enumerate(zip(METHODS, COLORS)):
                    failed_sampling = method == 'bayesian_constant' and not bool(rows.loc[method, 'diagnostics_passed'])
                    failed_numerics = not bool(rows.loc[method, 'numerical_replay_passed'])
                    failed_optimizer = rows.loc[method, 'convergence'] == 'reported_nonconverged'
                    marker = 'x' if failed_sampling or failed_numerics or failed_optimizer else 'o'
                    ax.scatter(index, values[index], marker=marker, color=color, s=26, linewidths=1., zorder=2)
                    if row == 0 and (failed_sampling or failed_numerics or failed_optimizer):
                        failed.append(dict(case_id=case_id, method=method,
                            sampling_failed=failed_sampling, numerical_failed=failed_numerics,
                            optimizer_failed=failed_optimizer))
            ax.set_xticks(np.arange(4), LABELS)
            ax.set_xlim(-.35, 3.35)
            ax.set_yscale('symlog', linthresh=1. if row == 0 else .01)
            ax.set_ylim(0, max(float(selected[metric].max())*1.6, 1. if row == 0 else .01))
            ax.set_ylabel('Source-location error (m)' if row == 0 else 'Mean Q error (relative)')
            ax.text(-.17, 1.08, 'abcd'[row*2+col], transform=ax.transAxes, fontsize=10, fontweight='bold')
            if row == 0:
                ax.set_title(('Ideal five-site network' if col == 0 else 'Actual six-site geometry')+' (n=4)', fontsize=9, pad=11)
    fig.text(.54, .97, 'Constant-Q capacity in every method', ha='center', fontsize=10)
    fig.text(.54, .925, 'Paired cases connected; crosses flag solver/sampling failure or failed replay', ha='center', fontsize=8)
    qa = Path.home()/'.codex/skills/nature-figure/scripts'
    if not (qa/'audit_panel_alignment.py').exists():
        raise RuntimeError('Required figure alignment helper unavailable')
    sys.path.insert(0, str(qa))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    stem = output/'constant_q_matched'
    require_matplotlib_panel_alignment(fig, json_out=str(stem.with_suffix('.alignment.json')),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'), dpi=600)
    plt.close(fig)
    frame.to_csv(output/'source_data.csv', index=False)
    notes = dict(figure='constant_q_matched', archetype='quantitative grid',
        conclusion='Matched-capacity comparisons are reported by geometry without a prespecified winner',
        unit='Four previously evaluated generated cases per geometry, two winds and repeats0/4; not fresh blind data',
        uncertainty='Individual paired cases only; no aggregate uncertainty bars or significance testing',
        inclusion='All32 method-case results, including nonconverged sampling and failed numerical replay',
        optimizer_status='Reported Powell failures are crossed; IA/joint fixed budgets are unverified, not proved converged',
        marker_failures=failed, differences='Bayes has proper Q/D priors; other methods minimize SSE',
        posterior='Point is posterior mean parameters with plug-in prediction, not posterior predictive mean',
        scope='Generic task-adapted emcee Gaussian-puff demo, not Newman M-MALA reproduction',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((analysis/'case_metrics.csv').read_bytes()).hexdigest(),
        transforms='Nonnegative symlog axes; full errors shown without clipping',
        backend='Python/matplotlib; PNG600dpi, editable PDF/SVG; alignment is checked before export')
    (output/'figure_notes.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    plot(parser.parse_args().analysis.resolve())
