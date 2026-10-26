"""Complete-cohort figures for4.1 and compensation diagnostics for4.2.

Figure contract: characterize paired methods across both network geometries,
then test whether nuisance-aware diagnostics convey empirical location risk.
No claim of superiority or calibrated confidence is hard-coded. Partial runs
are rejected. Independent case is the unit; no observation-level pseudoreplication.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ('ia_vpdi', 'matched_joint_adam', 'independent_puff_demo')
LABELS = ('IA-VPDI', 'Matched joint Adam', 'Independent puff demo')
COLORS = ('#197A9A', '#55585B', '#BC8625')
NETWORKS = ('ideal5', 'jjj6')


def export(fig, stem):
    qa = Path.home()/'.codex/skills/nature-figure/scripts'
    if not (qa/'audit_panel_alignment.py').exists():
        raise RuntimeError('Figure alignment helper unavailable')
    sys.path.insert(0, str(qa))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(stem.with_suffix('.alignment.json')),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=600)
    plt.close(fig)


def ecdf(ax, values, label, color, linestyle='-'):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Nonempty finite nonnegative errors required')
    ax.step(np.r_[0., values], np.arange(len(values)+1)/len(values), where='post',
            label=label, color=color, lw=1.5,linestyle=linestyle)
    ax.set_ylim(0., 1.03)


def performance(frame, output):
    notes = []
    methods, labels = METHODS, LABELS
    if 'shared_response_nnls_powell' in set(frame.method):
        methods = ('ia_vpdi', 'matched_joint_adam', 'shared_response_nnls_powell')
        labels = ('IA-VPDI', 'Matched joint Adam', 'Shared NNLS-Powell')
    for cohort in ('nominal', 'wind_mismatch', 'q_mismatch'):
        selected = frame[frame.cohort == cohort]
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
        fig.subplots_adjust(left=.11, right=.97, top=.85, bottom=.10, wspace=.31, hspace=.48)
        for col, network in enumerate(NETWORKS):
            rows = selected[selected.network == network]
            n = rows.case_id.nunique()
            for row, metric in enumerate(('localization_error_m', 'q_relative_l2')):
                ax = axes[row, col]
                maximum=0.
                for method, label, color,style in zip(methods, labels, COLORS,('-','--',':')):
                    values = rows[rows.method == method][metric]
                    if len(values) != n:
                        raise ValueError('Incomplete method pairing in a figure')
                    ecdf(ax, values, label, color,style)
                    maximum=max(maximum,float(values.max()))
                threshold=1. if row==0 else .01
                ax.set_xscale('symlog', linthresh=threshold)
                upper=max(maximum*1.08,threshold)
                ax.set_xlim(0.,upper)
                powers=np.arange(int(np.log10(threshold)),int(np.floor(np.log10(upper)))+1)
                stride=max(1,int(np.ceil(len(powers)/5)))
                powers=np.unique(np.r_[powers[::stride],powers[-1]])
                if upper/threshold > 1e6 and len(powers)>1:
                    powers=powers[1:]
                ax.set_xticks([0.,*(10.**powers)])
                ax.set_xticklabels(['0',*(('1' if p==0 else rf'$10^{{{p}}}$') for p in powers)])
                ax.set_xlabel('Source-location error (m)' if row == 0 else 'Release error (relative L2)')
                ax.set_ylabel('Cumulative fraction of cases')
                ax.text(-.14, 1.07, 'abcd'[row*2+col], transform=ax.transAxes,
                        fontsize=10, fontweight='bold', ha='left')
                if row == 0:
                    ax.set_title(('Prescribed five-site geometry' if col == 0 else 'Six-site geometry (synthetic)')+f' (n={n})',
                                 fontsize=9, pad=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53, .99),
                   ncol=3, frameon=False, handlelength=2., columnspacing=1.4)
        export(fig, output/f'performance_{cohort}')
        notes.append(dict(figure=f'performance_{cohort}', cohort=cohort,
            case_count=int(selected.case_id.nunique()), method_case_records=len(selected),
            uncertainty='individual-case ECDFs; no aggregate error bars or significance annotations',
            axes='nonnegative symmetric-log error axes with a linear neighborhood of zero; full ranges retained, no errors clipped',
            inclusion='all methods and cases in the complete prespecified cohort, including optimizer nonconvergence'))
    return notes


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    radius = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return p, center-radius, center+radius


def reliability(diagnostic, risks, output):
    selected = diagnostic[diagnostic.cohort == 'nominal']
    labels = ('Fixed Q and D', 'Eliminate Q', 'Eliminate Q and D')
    codes = ('naive_fixed_Q_D', 'eliminate_Q', 'eliminate_Q_D')
    colors = ('#777777', '#75B5C5', '#197A9A')
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    fig.subplots_adjust(left=.15, right=.98, bottom=.24, top=.78, wspace=.55)
    summary = []
    for index, (code, label, color) in enumerate(zip(codes, labels, colors)):
        rows = selected[selected.diagnostic == code]
        valid = rows[rows.available]
        successes = int((valid.contains_truth == True).sum())
        values = wilson(successes, len(valid))
        summary.append(dict(diagnostic=code, planned_cases=len(rows), defined_ellipses=len(valid),
                            truth_inclusions=successes, wilson=values))
        if values is not None:
            p, low, high = values
            axes[0].errorbar(p, index, xerr=[[p-low], [high-p]], fmt='o', color=color, capsize=3)
        axes[0].text(1.04, index, f'{len(valid)}/{len(rows)}', va='center', fontsize=7)
    axes[0].set_yticks(range(3), labels)
    axes[0].set_ylim(2.5, -.5)
    axes[0].set_xlim(0, 1)
    axes[0].axvline(.95, color='#AAAAAA', ls='--', lw=.8)
    axes[0].set_xlabel('Truth inclusion (defined ellipses)')
    axes[0].set_title('Local-ellipse validation', fontsize=9, pad=10)
    for col, network in enumerate(NETWORKS, 1):
        for code, label, color in zip(codes, labels, colors):
            values = risks[(risks.cohort == 'nominal') & (risks.network == network) &
                           (risks.diagnostic == code)].sort_values('coverage')
            axes[col].plot(values.coverage, values.mean_error_m, label=label, color=color, lw=1.5)
        axes[col].set_xlabel('Retained fraction of cases')
        axes[col].set_ylabel('Mean source error (m)')
        axes[col].set_xlim(0, 1)
        axes[col].set_ylim(bottom=0)
        axes[col].set_title('Ideal five-site network' if col == 1 else 'Actual six-site geometry',
                            fontsize=9, pad=10)
    handles, legend = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend, loc='upper center', ncol=3, bbox_to_anchor=(.55, .99), frameon=False)
    for index, ax in enumerate(axes):
        ax.text(-.20, 1.10, 'abc'[index], transform=ax.transAxes, fontsize=10, fontweight='bold')
    export(fig, output/'compensation_diagnostics')
    (output/'ellipse_inclusion_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return dict(figure='compensation_diagnostics', cohort='nominal',
        unit='independent generated case; diagnostic variants are paired',
        panel_a='Observed inclusion of nominal chi-square ellipses, Wilson95% binomial intervals among defined ellipses; valid/total counts shown',
        panels_bc='Descriptive mean localization error after ranking cases by major-axis radius; undefined ellipses ranked last',
        limitation='No claimed95% calibration, no truth-tuned acceptance threshold; curves contain no uncertainty bands')


def plot(analysis):
    status = json.loads((analysis/'analysis_status.json').read_text(encoding='utf-8'))
    if not status['final'] or status['completed_cases'] != status['planned_cases']:
        raise ValueError('Refusing manuscript figures from a partial evaluation')
    frame = pd.read_csv(analysis/'case_metrics.csv')
    if len(frame) != 3*status['planned_cases']:
        raise ValueError('Expected exactly three paired methods per case')
    output = analysis/'figures'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8, 'pdf.fonttype':42, 'svg.fonttype':'none',
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.8})
    notes = performance(frame, output)
    notes.append(reliability(pd.read_csv(analysis/'local_diagnostics.csv'),
                              pd.read_csv(analysis/'risk_coverage.csv'), output))
    (output/'figure_notes.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    plot(parser.parse_args().analysis.resolve())
