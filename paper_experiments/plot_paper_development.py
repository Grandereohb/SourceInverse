"""Plot all predeclared development cases, not a formal validation figure."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(analysis):
    rows = list(csv.DictReader((analysis / 'metrics.csv').open(encoding='utf-8-sig')))
    cases = sorted({r['case'] for r in rows})
    if len(cases) != 12 or len(rows) != 24:
        raise ValueError('Expected all12 paired cases; refusing selective plotting')
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.right': False, 'axes.spines.top': False})
    audit_root = Path.home() / '.codex/skills/nature-figure/scripts'
    if not (audit_root / 'audit_panel_alignment.py').exists():
        raise RuntimeError('Figure alignment QA helper unavailable')
    sys.path.insert(0, str(audit_root))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    x = np.arange(1, 13)
    for metric, ylabel, filename in (
            ('localization_error_m', 'Source-location error (m)', 'pilot_localization'),
            ('q_relative_l2', 'Source-strength error\n(relative L2)', 'pilot_release')):
        values = {method: np.array([float(next(r[metric] for r in rows if r['case'] == case
                            and r['method'] == method)) for case in cases])
                  for method in ('ia_vpdi', 'matched_joint_adam')}
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError('Nonfinite plot data; report failures instead of dropping them')
        fig, ax = plt.subplots(figsize=(7.2, 3.25))
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.82)
        ax.vlines(x, values['ia_vpdi'], values['matched_joint_adam'], colors='#C6C9CC', lw=1)
        ax.scatter(x, values['ia_vpdi'], color='#197A9A', s=28, marker='o', label='IA-VPDI', zorder=3)
        ax.scatter(x, values['matched_joint_adam'], facecolor='white', edgecolor='#55585B',
                   s=28, marker='s', label='Matched joint Adam', zorder=3)
        ax.set_xticks(x)
        ax.set_xlabel('Predeclared development case')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(bottom=0)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
        fig.text(0.12, 0.96, 'Development pilot: same-model synthetic data, not independent validation',
                 ha='left', va='top', fontsize=9)
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(analysis / f'{filename}.alignment.json'),
                                          tolerance_pt=1.5, strict=True)
        fig.savefig(analysis / f'{filename}.png', dpi=600)
        fig.savefig(analysis / f'{filename}.pdf')
        fig.savefig(analysis / f'{filename}.svg')
        plt.close(fig)
    caption = {'source_data': 'metrics.csv; all24 method-case records included',
        'n': '12 predeclared development scenarios; one noise realization/scenario, not repeated independent trials',
        'marks': 'Paired individual-case estimates, no aggregate uncertainty intervals or significance claims',
        'case_order': cases, 'status': 'Diagnostic development figures, not submission-ready Results4.1',
        'limitations': '20-step/start pilot; same-model generator, idealized network, no external baseline'}
    (analysis / 'figure_notes.json').write_text(json.dumps(caption, indent=2), encoding='utf-8')


def plot_transport_check(study):
    rows = list(csv.DictReader((study/'comparison.csv').open(encoding='utf-8-sig')))
    cases = ('pilot_01', 'pilot_03', 'pilot_11')
    if len(rows) != 21 or {r['case'] for r in rows} != set(cases):
        raise ValueError('Expected all21 prespecified independent-reference comparisons')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.right': False, 'axes.spines.top': False})
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.79)
    for case, color, marker in zip(cases, ('#197A9A', '#666666', '#A86A18'), ('o', 's', '^')):
        values = np.array([float(r['relative_l2_to_independent_reference']) for r in rows if r['case']==case])
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Log plot requires finite positive measured errors')
        ax.plot(np.arange(4), values[:4], marker=marker, color=color, lw=1, ms=4,
                label=f'Case {case[-2:]}')
        ax.plot(np.arange(4,7), values[4:], marker=marker, color=color, lw=1, ms=4)
    ax.axhline(0.01, color='#999999', ls=':', lw=1, label='1% target')
    ax.axvline(3.5, color='#C0C0C0', ls='--', lw=0.8)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(7), ['Grid36\n1/8 h', 'Grid64\n1/16 h', 'Grid96\n1/32 h',
                               'Grid128\n1/64 h', 'Puff\n1/8 h', 'Puff\n1/16 h', 'Puff\n1/32 h'])
    ax.set_xlabel('Transport solver and time step')
    ax.set_ylabel('Concentration error\n(relative L2)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,1.02), ncol=4, frameon=False)
    fig.text(0.13,0.96,'Fixed source and Q: comparison with an independent continuous-space reference',
             ha='left',va='top',fontsize=9)
    sys.path.insert(0, str(Path.home()/'.codex/skills/nature-figure/scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(study/'transport_check.alignment.json'), strict=True)
    fig.savefig(study/'transport_check.png', dpi=600)
    fig.savefig(study/'transport_check.pdf')
    fig.savefig(study/'transport_check.svg')
    plt.close(fig)
    (study/'figure_notes.json').write_text(json.dumps({
        'source_data': 'comparison.csv; all21 rows included, no fitted or amplitude-rescaled predictions',
        'reference': 'Independent NumPy implementation; exact prescribed wind/tensor integrals, release quadrature64',
        'observation_average': '4,8,16,32 midpoint samples/hour for respective settings; see CSV',
        'case01': 'constant release, steady wind, single-dominant network geometry',
        'case03': 'double peak, steady wind, single-dominant network geometry',
        'case11': 'pulse, veering wind, distributed network geometry',
        'uncertainty': 'deterministic numerical comparisons; no replication error bars required',
        'interpretation': 'Forward numerical accuracy only; not localization-performance evidence'}, indent=2),encoding='utf-8')


def plot_paired_study(study):
    analysis=study/'analysis'
    rows=list(csv.DictReader((analysis/'comparison.csv').open(encoding='utf-8-sig')))
    cases=('pilot_01','pilot_03','pilot_11')
    methods=('ia_vpdi','matched_joint_adam','independent_puff_nnls_powell_demo')
    if len(rows)!=9 or {(r['case'],r['method']) for r in rows}!={(c,m) for c in cases for m in methods}:
        raise ValueError('Expected all9 paired development results')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.right':False,'axes.spines.top':False})
    fig,ax=plt.subplots(figsize=(7.2,3.3))
    fig.subplots_adjust(left=0.12,right=0.98,bottom=0.19,top=0.8)
    for method,label,color,marker,offset in zip(methods,
        ('IA-VPDI','Matched joint Adam','Puff + NNLS/Powell demo'),
        ('#197A9A','#666666','#A86A18'),('o','s','^'),(-0.12,0.,0.12)):
        values=np.array([float(next(r['localization_error_m'] for r in rows if
            r['case']==case and r['method']==method)) for case in cases])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite paired comparison values')
        ax.scatter(np.arange(3)+offset,values,color=color,marker=marker,s=28,label=label)
    ax.set_xticks(np.arange(3),['01: Constant','03: Double peak','11: Pulse'])
    ax.set_xlim(-0.5,2.5);ax.set_ylim(bottom=0)
    ax.set_ylabel('Source-location error (m)')
    ax.set_xlabel('Predeclared development case')
    ax.legend(loc='lower center',bbox_to_anchor=(0.5,1.02),ncol=3,frameon=False,fontsize=8)
    fig.text(0.12,0.96,'Independent-generator inputs: three-case development feasibility, not final evaluation',
             ha='left',va='top',fontsize=9)
    sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(analysis/'paired_localization.alignment.json'),strict=True)
    fig.savefig(analysis/'paired_localization.png',dpi=600)
    fig.savefig(analysis/'paired_localization.pdf')
    fig.savefig(analysis/'paired_localization.svg')
    plt.close(fig)
    (analysis/'figure_notes.json').write_text(json.dumps({
        'source_data':'comparison.csv; all9 method-case rows included',
        'sample_size':'3 development scenarios with one saved noise realization each; not an independent test set',
        'forward_match':'IA/joint identical; independent puff demo uses analytic continuous wind integrals',
        'baseline_label':'generic adapted demo, not a reproduction of a named published algorithm',
        'uncertainty':'individual deterministic estimates, no significance or coverage claims'},indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=Path(__file__).resolve().parents[1] /
                        'result/paper_experiments_20261017/development_12/analysis')
    parser.add_argument('--transport-study', type=Path)
    parser.add_argument('--paired-study',type=Path)
    args = parser.parse_args()
    if args.paired_study:
        plot_paired_study(args.paired_study)
    elif args.transport_study:
        plot_transport_check(args.transport_study)
    else:
        plot(args.analysis)
