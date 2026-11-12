"""Plot refreshed complete-cohort tables; never run an inversion."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/ia_vpdi_report_20261028'
DATA, FIG = OUT / 'data', OUT / 'figures'
METHODS = ['ia_vpdi', 'joint_nls_trf', 'matched_joint_adam', 'standard_gbest_pso']
LABELS = ['IA-VPDI', 'Joint NLS', '联合 Adam', '标准 PSO']
COLORS = ['#2876A8', '#C27A3F', '#707070', '#49866D']


def finish(fig, ax, name):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / f'{name}.png', dpi=320, facecolor='white')
    fig.savefig(FIG / f'{name}.pdf', facecolor='white')
    fig.savefig(FIG / f'{name}.svg', facecolor='white')
    plt.close(fig)


def build():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'Arial'], 'font.size': 10,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    frame = pd.read_csv(DATA / 'network_summary.csv')
    networks = ['ideal5', 'jjj6', 'surround5_pulse', 'corridor6_double_peak']
    fig, ax = plt.subplots(figsize=(8, 3.1))
    x = np.arange(4)
    for i, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        values = frame[frame.method.eq(method)].set_index('network').loc[networks, 'mean_location_m']
        ax.bar(x + (i-1.5)*.19, values, width=.18, color=color, label=label)
    ax.set_xticks(x, ['理想五站', '实际六站几何', '环绕五站脉冲', '走廊六站双峰'])
    ax.set_ylabel('平均定位误差（米）')
    ax.legend(ncol=4, loc='upper center', frameon=False, fontsize=9)
    ax.set_ylim(0, 750)
    finish(fig, ax, 'synthetic_networks')

    frame = pd.read_csv(DATA / 'synthetic_cases.csv')
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for method, label, color in zip(METHODS[:2], LABELS[:2], COLORS[:2]):
        rows = frame[frame.cohort.eq('nominal') & frame.method.eq(method)]
        ax.scatter(rows.fit_rmse, rows.localization_error_m, s=20, alpha=.65,
                   color=color, label=label, edgecolors='none')
    ax.axhline(100, color='#555555', ls='--', lw=.8)
    ax.axvline(.02, color='#555555', ls=':', lw=.8)
    ax.set_yscale('symlog', linthresh=10)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('观测浓度拟合 RMSE（归一化浓度）')
    ax.set_ylabel('定位误差（米）')
    ax.legend(frameon=False, loc='upper right')
    finish(fig, ax, 'fit_vs_location')

    frame = pd.read_csv(DATA / 'metec_cases.csv', dtype={'case_id': str})
    ids = sorted(frame.case_id.unique())
    fig, ax = plt.subplots(figsize=(8, 3.3))
    for method, label, color in zip(METHODS, LABELS, COLORS):
        values = frame[frame.method.eq(method)].set_index('case_id').loc[ids, 'localization_error_m']
        ax.plot(np.arange(1, 18), values, marker='o', ms=3.5, lw=1.2, color=color, label=label)
    ax.set_xticks(np.arange(1, 18))
    ax.set_xlabel('公开事件编号（按原始事件编号排序，完整 17 例）')
    ax.set_ylabel('定位误差（米）')
    ax.legend(ncol=4, frameon=False, fontsize=9, loc='lower center', bbox_to_anchor=(.5, 1.01))
    ax.set_ylim(bottom=0)
    finish(fig, ax, 'metec_all_events')
    (FIG / 'figure_sources.json').write_text(json.dumps({
        'synthetic_networks': 'data/network_summary.csv',
        'fit_vs_location': 'data/synthetic_cases.csv: nominal IA and NLS all192',
        'metec_all_events': 'data/metec_cases.csv: all17 x4, fixed chronological order',
        'fitting_performed': False}, indent=2), encoding='utf-8')
    print(FIG)


if __name__ == '__main__':
    build()
