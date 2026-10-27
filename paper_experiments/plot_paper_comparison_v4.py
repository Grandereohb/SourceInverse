"""Figures from the new fixed-cohort comparison; no inversion or sample selection.

Formal exports require complete analyses. --preview is explicitly nonfinal and
is intended only for internal layout checks with currently completed cases.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


METHODS = ('ia_vpdi', 'joint_nls_trf', 'standard_gbest_pso', 'matched_joint_adam')
LABELS = ('IA-VPDI', 'Joint NLS', 'Standard PSO', 'Joint Adam')
COLORS = ('#2876A8', '#C27A3F', '#49866D', '#707070')
STYLES = ('-', '--', '-.', ':')
ROOT = Path(__file__).resolve().parents[1]/'result/paper_experiments_20261017'


def complete_analysis(folder, preview):
    status = json.loads((folder/'analysis/analysis_status.json').read_text(encoding='utf-8'))
    if not status['final'] and not preview:
        raise ValueError(f'Complete batch required before formal figures: {folder.name}')
    return status


def ecdf(axis, values, **kwargs):
    values = np.sort(np.asarray(values, dtype=float))
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Finite nonnegative plotted metrics required')
    if len(values):
        axis.step(np.r_[0., values], np.r_[0., np.arange(1, len(values)+1)/len(values)],
                  where='post', **kwargs)


def legend(fig, active=METHODS):
    handles = [Line2D([], [], color=c, ls=s, lw=1.5, label=n)
               for code, n, c, s in zip(METHODS, LABELS, COLORS, STYLES) if code in active]
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(.53, .995),
               columnspacing=1.4, handlelength=2.3, frameon=False)


def save(fig, destination, tools, note):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(destination)+'.alignment.json',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(str(destination)+'.pdf')
    fig.savefig(str(destination)+'.svg')
    fig.savefig(str(destination)+'.png', dpi=300)
    (destination.with_suffix('.caption.txt')).write_text(note, encoding='utf-8')
    plt.close(fig)


def synthetic_figures(folder, output, tools, preview=False):
    complete_analysis(folder, preview)
    frame = pd.read_csv(folder/'analysis/case_metrics.csv')
    groups = [('nominal', 'ideal5', 'Ideal five-station'),
              ('nominal', 'jjj6', 'Actual six-station geometry'),
              ('nominal', 'surround5_pulse', 'Surrounding five-station'),
              ('nominal', 'corridor6_double_peak', 'Corridor six-station'),
              ('wind_mismatch', None, 'Wind mismatch'),
              ('q_mismatch', None, 'Release-shape mismatch')]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), squeeze=False)
    fig.subplots_adjust(left=.08, right=.985, bottom=.105, top=.855, wspace=.28, hspace=.6)
    maximum = max(float(frame.localization_error_m.max()), 10.)*1.04
    for index, (axis, (cohort, network, title)) in enumerate(zip(axes.flat, groups)):
        sample = frame[frame.cohort == cohort]
        if network is not None:
            sample = sample[sample.network == network]
        count = sample.case_id.nunique()
        for method, color, style in zip(METHODS, COLORS, STYLES):
            ecdf(axis, sample.loc[sample.method == method, 'localization_error_m'],
                 color=color, ls=style, lw=1.45)
        axis.set(title=f'{chr(97+index)}  {title}  n={count}', xlim=(0., maximum), ylim=(0., 1.02),
                 xlabel='Localization error (m)', ylabel='Fraction of cases')
        axis.set_xscale('symlog', linthresh=10)
        axis.grid(axis='y', color='.9', lw=.5)
    legend(fig)
    if preview:
        fig.text(.5, .905, 'INCOMPLETE BATCH - INTERNAL LAYOUT PREVIEW', ha='center', fontsize=8)
    save(fig, output/'synthetic_localization', tools,
        'All 216 prespecified inputs in six disjoint groups (192 nominal and24 mismatch). '
        'Curves are case-level empirical CDFs, not confidence bands. PSO is the within-case '
        'median of three fixed seeds, never the best seed. Source data: case_metrics.csv. '
        f'Current input has {frame.case_id.nunique()} cases; preview={preview}.')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(left=.09, right=.985, bottom=.23, top=.75, wspace=.38)
    for method, color, style in zip(METHODS, COLORS, STYLES):
        ecdf(axes[0], frame.loc[frame.method == method, 'q_relative_l2'], color=color, ls=style, lw=1.45)
    axes[0].set(xlabel='Relative source-strength L2 error', ylabel='Fraction of cases',
                xlim=(0., max(float(frame.q_relative_l2.max())*1.05, .1)),
                ylim=(0., 1.02), title='a  Release reconstruction')
    axes[0].set_xscale('symlog', linthresh=.1)
    wide = frame.pivot(index='case_id', columns='method', values='seconds')
    for index, method in enumerate(METHODS[1:]):
        values = (wide[method]/wide.ia_vpdi).to_numpy()
        low, center, high = np.quantile(values, [.25, .5, .75])
        axes[1].errorbar(index, center, yerr=[[center-low], [high-center]], fmt='o',
                         color=COLORS[index+1], markersize=4, capsize=3, lw=1)
    axes[1].axhline(1., color=COLORS[0], lw=.8, ls='--')
    axes[1].set(xticks=range(3), xticklabels=LABELS[1:], ylabel='Solve time / IA-VPDI time',
                yscale='log', title='b  Paired computational cost', xlim=(-.5, 2.5))
    legend(fig)
    if preview:
        fig.text(.5, .865, 'INCOMPLETE BATCH - INTERNAL LAYOUT PREVIEW', ha='center', fontsize=8)
    save(fig, output/'synthetic_release_cost', tools,
        'Panel a: all case-level source-strength relative L2 errors. Panel b: paired runtime ratios '
        'against IA-VPDI, median and interquartile range across cases; PSO per-seed median '
        'time, not sum of all three repeats. Conditional solver time excludes reused common '
        'coarse initialization; update counts are not equal FLOPs. No timing-isolation claim. '
        f'Cases={len(wide)}; preview={preview}.')


def reliability_figure(folder, output, tools):
    status = json.loads((folder/'analysis/analysis_status.json').read_text(encoding='utf-8'))
    if not status.get('reliability_final'):
        raise ValueError('Complete frozen96-case reliability cohort required')
    local = pd.read_csv(folder/'analysis/local_diagnostics.csv')
    local = local[(local.cohort == 'nominal') & local.network.isin(['ideal5', 'jjj6']) &
                  local['repeat'].between(4, 7)]
    calibrated = pd.read_csv(folder/'analysis/calibrated_ellipses.csv')
    calibrated = calibrated[calibrated.partition == 'validation']
    definitions = [('naive_fixed_Q_D', 'Fixed Q and D'), ('eliminate_Q', 'Eliminate Q'),
                   ('eliminate_Q_D', 'Eliminate Q and D')]
    samples = [(local[local.diagnostic == key], label, 'area_m2', 'contains_truth')
               for key, label in definitions]
    samples.append((calibrated, 'Calibrated', 'calibrated_area_m2', 'calibrated_contains_truth'))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    axes = axes.ravel()
    fig.subplots_adjust(left=.09, right=.98, bottom=.13, top=.88, wspace=.38, hspace=.8)
    labels, available_counts = [], []
    for index, (sample, label, area_key, contains_key) in enumerate(samples):
        if len(sample) != 48:
            raise ValueError('Reliability figure requires the same48 validation cases')
        labels.append(label.replace(' ', '\n', 1))
        covered = sample[contains_key].eq(True).sum()/48
        axes[0].bar(index, covered, color=COLORS[min(index, 3)], width=.58)
        axes[0].text(index, covered+.025, f'{covered:.1%}', ha='center', fontsize=7)
        area = sample[area_key].dropna().to_numpy()/1e6
        available_counts.append(dict(label=label, before=len(sample), after=len(area),
            excluded=len(sample)-len(area), reason='undefined local ellipse has no finite area'))
        low, center, high = np.quantile(area, [.25, .5, .75])
        axes[1].errorbar(index, center, yerr=[[center-low], [high-center]], fmt='o',
                         color=COLORS[min(index, 3)], capsize=3)
    for axis in axes[:2]:
        axis.set_xticks(range(4), labels)
    axes[0].axhline(.95, color='.4', ls='--', lw=.7)
    axes[0].set(ylim=(0., 1.14), ylabel='Empirical source coverage', title='a  Validation-case coverage')
    axes[1].set(yscale='log', title='b  Region size')
    axes[1].set_ylabel(r'Ellipse area (km$^2$)', fontsize=8)
    risks = pd.read_csv(folder/'analysis/risk_coverage.csv')
    for axis, network, label in zip(axes[2:], ('ideal5', 'jjj6'),
            ('c  Ideal five-station (n=48)', 'd  Actual six-station geometry (n=48)')):
        for index, (code, _) in enumerate(definitions):
            group = risks[(risks.network == network) & (risks.diagnostic == code)].sort_values('coverage')
            axis.plot(group.coverage, group.mean_error_m, color=COLORS[index], ls=STYLES[index], lw=1.3)
        axis.set(xlabel='Retained fraction of cases', ylabel='Mean localization error (m)',
                 title=label, xlim=(0., 1.), ylim=(0., None))
    handles = [Line2D([], [], color=COLORS[i], ls=STYLES[i], lw=1.3, label=label)
               for i, (_, label) in enumerate(definitions)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .995), ncol=3, frameon=False)
    save(fig, output/'localization_reliability', tools,
        'Same48 retrospectively designated validation cases in every column. Coverage denominator '
        'includes unavailable ellipses as noncoverage; region-size medians/IQR use finite defined '
        'ellipses. Calibration uses the other48 cases only and rescales area by squared axis factor. '
        'No station/time withholding from inversion, no source coordinates changed. '
        'Bottom panels: original48 nominal cases per network ranked by estimated major axis '
        '(not truth); descriptive mean location error versus retained fraction, no confidence bands '
        'or inversion rejection. Source data: local_diagnostics.csv, calibrated_ellipses.csv, risk_coverage.csv. '
        'Defined-area counts: '+json.dumps(available_counts)+'.')


def public_figure(folder, output, tools, preview=False):
    complete_analysis(folder, preview)
    frame = pd.read_csv(folder/'analysis/case_metrics.csv')
    raw = pd.read_csv(folder/'analysis/replicate_metrics.csv')
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    events = [item['source'].replace('_', '')+'_'+item['release'] for item in protocol['events']]
    active = tuple(code for code in METHODS if code in protocol['methods'])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7))
    fig.subplots_adjust(left=.09, right=.985, bottom=.16, top=.88, hspace=.55)
    for row, (axis, height) in enumerate(zip(axes, protocol['heights_m'])):
        for index, (method, color) in enumerate(zip(METHODS, COLORS)):
            if method not in active:
                continue
            series = frame[(frame.height_m == height) & (frame.method == method)].set_index('event')
            y = series.reindex(events).localization_error_m.to_numpy()
            x = np.arange(len(events))+(active.index(method)-(len(active)-1)/2)*.16
            axis.scatter(x, y, s=20, c=color, marker=['o', 's', '^', 'D'][index])
            if method == 'standard_gbest_pso':
                replicates = raw[(raw.height_m == height) & (raw.method == method)]
                for position, event in zip(x, events):
                    values = replicates[replicates.event == event].localization_error_m
                    axis.scatter(np.full(len(values), position), values, s=7, c=color, alpha=.4)
        axis.set(xticks=np.arange(len(events)), xticklabels=[name.replace('_', '\n') for name in events],
                 ylabel='Localization error (m)', title=f'{chr(97+row)}  Prescribed source height {height:g} m',
                 ylim=(0., max(frame.localization_error_m.max()*1.08, 1.)))
        axis.grid(axis='y', color='.9', lw=.5)
    legend(fig, active)
    if preview:
        fig.text(.5, .925, 'INCOMPLETE BATCH - INTERNAL LAYOUT PREVIEW', ha='center', fontsize=8)
    save(fig, output/'public_release_localization', tools,
        'Eight windows at two physical sources, all seven light paths and shared constant-Q '
        'capacity. Both prescribed heights shown, no best-height selection. '
        + ('PSO main symbol is seed-median error; small points retain all three seeds. '
           if 'standard_gbest_pso' in active else '')+
        'Heights and shared physical '
        'sources are not independent experimental replicates. Newman uses its own author '
        'plume/prior/background model and is reported separately. '
        f'Preview={preview}. Source data: public case_metrics.csv and replicate_metrics.csv.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', type=Path, default=ROOT/'new_comparison_v4')
    parser.add_argument('--public', type=Path, default=ROOT/'beam_comparison_cuda_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--section', choices=('all', 'synthetic', 'reliability', 'public'), default='all')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--figure-tools', type=Path,
                        default=Path.home()/'.codex/skills/nature-figure/scripts')
    args = parser.parse_args()
    sys.path.insert(0, str(args.figure_tools))
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.titlesize': 8,
        'axes.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False})
    args.output.mkdir(parents=True, exist_ok=True)
    if args.section in ('all', 'synthetic'):
        synthetic_figures(args.synthetic, args.output, args.figure_tools, args.preview)
    if args.section in ('all', 'reliability'):
        reliability_figure(args.synthetic, args.output, args.figure_tools)
    if args.section in ('all', 'public'):
        public_figure(args.public, args.output, args.figure_tools, args.preview)
    manifests = []
    for folder in (args.synthetic, args.public):
        path = folder/'analysis/case_metrics.csv'
        if path.exists():
            manifests.append(dict(study=folder.name, file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.output/'figure_manifest.json').write_text(json.dumps(dict(preview=args.preview,
        backend='python/matplotlib', source_data=manifests,
        qa='Alignment measured before export; PDF text/collision and visual audits remain required'), indent=2),
        encoding='utf-8')


if __name__ == '__main__':
    main()
