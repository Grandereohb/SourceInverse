"""Small matched-Q control and readable chapter update; original results stay read-only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'result/paper_experiments_20261017'
UPDATED = STUDY/'paper_updated_20261026'
OUT = STUDY/'paper_focused_20261027'
METHODS = dict(ia_vpdi='IA-VPDI', joint_nls_trf='Joint NLS',
               standard_gbest_pso='标准PSO', matched_joint_adam='联合Adam')
NETWORKS = dict(ideal5='理想五站', jjj6='实际六站几何',
                surround5_pulse='环绕五站', corridor6_double_peak='走廊六站')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if hasattr(x, 'tolist') else x), encoding='utf-8')


def freeze():
    source = UPDATED/'synthetic/protocol.json'
    protocol = dict(version='focused_chapter_v1', date='2026-10-27',
        source_protocol_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        fit_threshold_noise_multiplier=2., primary_location_threshold_m=100.,
        location_sensitivity_m=[50., 100., 200.],
        threshold_scope='Descriptive thresholds, not an instrument LOD or established business requirement',
        primary_cohort='all192 nominal cases;24 mismatches reported separately',
        pso='case-wise median of all three prespecified seeds; not best seed',
        control='same24 cases previously frozen; fixed5 Q coefficients; two common starts;250 nfev per start',
        control_methods=['profiled', 'joint_nls'], no_model_selection_in_control=True,
        representatives='eval_007 fixed nominal illustration; first nominal case ID with Joint NLS good fit and location error>100m, regardless of IA outcome',
        no_crossfit=True, no_new_scenarios=True, no_initialization_sensitivity=True)
    path = OUT/'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Frozen analysis changed; choose a different output directory')
    save(path, protocol)
    return protocol


def run_control(device):
    import run_paper_updated_evidence as previous
    from paper_q_model_selection import fit_profiled_q_models
    freeze()
    entries = read(UPDATED/'capacity_control/protocol.json')['cases']
    rows = []
    for entry in entries:
        case = entry['case']['case_id']
        target = OUT/'control'/case/'result.json'
        if target.exists():
            result = read(target)
        else:
            old, data, obj, starts = previous.synthetic_setup(entry, device)
            result = dict(case_id=case, methods={}, metadata=old['metadata'])
            for method in ('profiled', 'joint_nls'):
                fit = fit_profiled_q_models(obj, starts, counts=(5,), max_nfev=250, solver=method)
                fit.update(previous.prior.formal.metrics(old['metadata'], data, fit['source_norm'],
                    np.asarray(fit['coefficients']), np.asarray(fit['prediction'])))
                result['methods'][method] = fit
            save(target, result)
        for method, fit in result['methods'].items():
            rows.append(dict(case_id=case, network=entry['case']['network'], method=method,
                **{key: fit[key] for key in ('localization_error_m', 'q_relative_l2',
                    'observed_mse', 'seconds', 'converged')}))
        print('MATCHED_FIXED_Q', case, flush=True)
    frame = pd.DataFrame(rows)
    if frame.case_id.nunique() != 24 or len(frame) != 48:
        raise ValueError('24 complete paired controls required')
    frame.to_csv(OUT/'control.csv', index=False)
    save(OUT/'control_status.json', dict(final=True, cases=24, fits=48))


def classify(frame, location_m=100., fit_multiplier=2.):
    out = frame.copy()
    out['fit_rmse'] = np.sqrt(out.observed_mse)
    out['fit_threshold'] = fit_multiplier*out.noise_std
    out['good_fit'] = out.fit_rmse <= out.fit_threshold
    out['wrong_location'] = out.localization_error_m > location_m
    out['good_fit_wrong_location'] = out.good_fit & out.wrong_location
    return out


def summarize_flags(frame, group_columns):
    rows = []
    for key, g in frame.groupby(group_columns, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        count = int(g.good_fit.sum())
        wrong = int(g.good_fit_wrong_location.sum())
        rows.append(dict(zip(group_columns, key), n=len(g), good_fit=count,
            wrong_location=int(g.wrong_location.sum()), good_fit_wrong=wrong,
            joint_rate=wrong/len(g), conditional_rate=wrong/count if count else None,
            median_location_m=float(g.localization_error_m.median()),
            mean_location_m=float(g.localization_error_m.mean()),
            p90_location_m=float(g.localization_error_m.quantile(.9)),
            median_q_error=float(g.q_relative_l2.median()),
            mean_q_error=float(g.q_relative_l2.mean())))
    return pd.DataFrame(rows)


def representative_ids(frame):
    candidates = frame[(frame.cohort == 'nominal') & (frame.method == 'joint_nls_trf')
                       & frame.good_fit_wrong_location].sort_values('case_id')
    return list(dict.fromkeys(['eval_007', *candidates.case_id.head(1).tolist()]))


def analyze():
    protocol = freeze()
    frame = pd.read_csv(UPDATED/'synthetic/analysis/case_metrics.csv')
    metadata = {case: read(UPDATED/'synthetic'/case/'result.json')['metadata']
                for case in frame.case_id.unique()}
    frame['noise_std'] = frame.case_id.map(lambda key: metadata[key]['noise_std'])
    frame = classify(frame)
    if len(frame) != 864 or frame.case_id.nunique() != 216:
        raise ValueError('Expected all216 cases x4 methods')
    analysis = OUT/'analysis'
    analysis.mkdir(exist_ok=True)
    frame.to_csv(analysis/'case_flags.csv', index=False)
    nominal = frame[frame.cohort == 'nominal']
    summarize_flags(nominal, ['method']).to_csv(analysis/'nominal_summary.csv', index=False)
    summarize_flags(nominal, ['network', 'method']).to_csv(analysis/'network_summary.csv', index=False)
    summarize_flags(nominal, ['release', 'method']).to_csv(analysis/'release_summary.csv', index=False)
    summarize_flags(frame[frame.cohort != 'nominal'], ['cohort', 'method']).to_csv(
        analysis/'mismatch_summary.csv', index=False)
    sensitivity = []
    for threshold in protocol['location_sensitivity_m']:
        part = summarize_flags(classify(nominal, threshold), ['method'])
        part['location_threshold_m'] = threshold
        sensitivity.append(part)
    pd.concat(sensitivity).to_csv(analysis/'threshold_sensitivity.csv', index=False)
    control = pd.read_csv(OUT/'control.csv')
    control['noise_std'] = control.case_id.map(lambda key: metadata[key]['noise_std'])
    summarize_flags(classify(control), ['method']).to_csv(analysis/'control_summary.csv', index=False)
    paired = control.pivot(index='case_id', columns='method', values='localization_error_m')
    differences = paired.profiled-paired.joint_nls
    ids = representative_ids(frame)
    save(analysis/'summary.json', dict(final=True, nominal_cases=192, mismatch_cases=24,
        control_cases=24, control_comparison=dict(improved_over1m=int((differences < -1).sum()),
            worsened_over1m=int((differences > 1).sum()), similar_within1m=int((differences.abs() <= 1).sum()),
            median_paired_difference_m=float(differences.median())),
        representative_ids=ids, no_source_estimates_changed=True))
    inputs = [UPDATED/'synthetic/analysis/case_metrics.csv', UPDATED/'synthetic/protocol.json',
              UPDATED/'public/analysis/case_metrics.csv', OUT/'control.csv', OUT/'protocol.json']
    inputs += [UPDATED/'synthetic'/key/'result.json' for key in metadata]
    inputs += list((OUT/'control').glob('*/result.json'))
    inputs += [UPDATED/'public/analysis/quadrature_replay_status.json',
               STUDY/'newman_lhs_full_v1/analysis/case_metrics.csv',
               STUDY/'newman_lhs_full_v1/analysis/analysis_status.json']
    inputs += [UPDATED/'public/source1_rel03/height_0.3/result.json',
               STUDY/'controlled_release_preparation_v2/source1_rel03/observations.csv',
               STUDY/'controlled_release_preparation_v2/source1_rel03/beams.csv',
               STUDY/'public_data/author_truth_values.json']
    save(analysis/'sources.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in inputs})
    print(pd.read_csv(analysis/'nominal_summary.csv').to_string(index=False))
    print(pd.read_csv(analysis/'control_summary.csv').to_string(index=False))
    print(read(analysis/'summary.json'))


def plots():
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0, str(Path.home()/'.codex/skills/nature-figure/scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 9,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.unicode_minus': False})
    colors = ['#2876A8', '#C27A3F', '#49866D', '#707070']
    figures = OUT/'figures'
    figures.mkdir(exist_ok=True)
    def finish(fig, name):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(figures/(name+'.alignment.json')),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        for suffix in ('pdf', 'png', 'svg'):
            fig.savefig(figures/(name+'.'+suffix), dpi=240)
        plt.close(fig)
    frame = pd.read_csv(OUT/'analysis/case_flags.csv')
    nominal = frame[frame.cohort == 'nominal']
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    fig.subplots_adjust(left=.10, right=.97, top=.90, bottom=.12, wspace=.28, hspace=.60)
    for ax, (method, label), color in zip(axes.flat, METHODS.items(), colors):
        g = nominal[nominal.method == method]
        ax.scatter(g.fit_rmse, g.localization_error_m, s=13, color=color, alpha=.65, edgecolors='none')
        ax.axvline(.02, color='#555555', ls='--', lw=.8)
        ax.axhline(100, color='#555555', ls='--', lw=.8)
        ax.set(xscale='log', yscale='symlog', ylim=(0, 5500), xlim=(.0001, .3),
            title=label+'  n=192', xlabel='浓度拟合误差', ylabel='定位误差（米）')
    finish(fig, 'fit_and_location')
    entries = {item['case']['case_id']: item for item in read(UPDATED/'synthetic/protocol.json')['cases']}
    ids = read(OUT/'analysis/summary.json')['representative_ids']
    for case in ids:
        result = read(UPDATED/'synthetic'/case/'result.json')
        meta = result['metadata']
        with np.load(STUDY/entries[case]['parent']/case/'input.npz') as data:
            stations = data['stations'].copy()
            observed = data['observed'].reshape(6, -1).copy()
        fig, axes = plt.subplots(1, 3, figsize=(9., 3.0))
        fig.subplots_adjust(left=.075, right=.985, top=.75, bottom=.20, wspace=.48)
        source = np.asarray(meta['source_norm'])*meta['length_m']
        axes[0].scatter(*(stations*meta['length_m']).T, marker='o', color='#999999', s=18)
        axes[0].scatter(*source, marker='*', color='black', s=85)
        times = np.linspace(0, 6, 121)
        axes[1].plot(times, np.interp(times, meta['true_knots'], meta['true_coefficients']), color='black', lw=1.5)
        station = int(np.argmax(observed.max(0)))
        axes[2].plot(np.arange(6)+.5, observed[:, station], 'ko-', ms=3)
        for method, color in zip(list(METHODS)[:2], colors):
            fit = result['methods'][method]
            axes[0].scatter(*(np.asarray(fit['source_norm'])*meta['length_m']), marker='x', s=50, color=color)
            coefficients = fit['coefficients']
            axes[1].plot(times, np.interp(times, np.linspace(0, 6, len(coefficients)), coefficients), color=color)
            axes[2].plot(np.arange(6)+.5, np.asarray(fit['prediction']).reshape(6, -1)[:, station], '--', color=color)
        axes[0].set(xlabel='东西距离（米）', ylabel='南北距离（米）', title='源点与监测站')
        axes[0].set_aspect('equal', adjustable='datalim')
        axes[1].set(xlabel='时间（小时）', ylabel='排放量（模型单位）', title='排放过程')
        axes[2].set(xlabel='时间（小时）', ylabel='归一化浓度', title='峰值站浓度')
        fig.legend(handles=[Line2D([], [], color=c, label=l) for c,l in
            [('black', '真值或观测'), (colors[0], 'IA-VPDI'), (colors[1], 'Joint NLS')]],
            loc='upper center', ncol=3, frameon=False)
        finish(fig, 'case_'+case)
    public = pd.read_csv(UPDATED/'public/analysis/case_metrics.csv')
    events = list(dict.fromkeys(public.event))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.8))
    fig.subplots_adjust(left=.10, right=.98, top=.82, bottom=.18, hspace=.70)
    for ax, height in zip(axes, [.3, 1.6]):
        for offset, method, color, marker in zip([-.10, 0., .10],
                ['ia_vpdi', 'joint_nls_trf', 'matched_joint_adam'],
                [colors[0], colors[1], colors[3]], ['o', 'x', 's']):
            g = public[(public.method == method) & (public.height_m == height)].set_index('event')
            ax.scatter(np.arange(8)+offset, g.loc[events, 'localization_error_m'],
                color=color, marker=marker, s=20)
        ax.set(xticks=np.arange(8), xticklabels=[e.replace('source', '源').replace('_rel', '事件') for e in events],
            ylabel='定位误差（米）', title=f'源高{height:g}米', ylim=(0, 7.8))
        ax.tick_params(axis='x', labelsize=7)
    fig.legend(handles=[Line2D([], [], color=c, marker=m, ls='', label=l) for c,m,l in
        [(colors[0], 'o', 'IA-VPDI'), (colors[1], 'x', 'Joint NLS'), (colors[3], 's', '联合Adam')]],
        loc='upper center', ncol=3, frameon=False)
    finish(fig, 'public_locations')
    # Fixed first event and first reported height, not an outcome-selected illustration.
    public_case = read(UPDATED/'public/source1_rel03/height_0.3/result.json')
    prep = STUDY/'controlled_release_preparation_v2/source1_rel03'
    observations = pd.read_csv(prep/'observations.csv').sort_values(['time', 'beam'])
    beams = pd.read_csv(prep/'beams.csv').sort_values('beam')
    values = observations.residual_ppm.to_numpy().reshape(-1, len(beams))
    beam_index = int(values.max(0).argmax())
    truth = read(STUDY/'public_data/author_truth_values.json')['values_as_stored']['source_1_location'][:2]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    fig.subplots_adjust(left=.09, right=.98, top=.76, bottom=.22, wspace=.38)
    for beam in beams.itertuples():
        axes[0].plot([beam.start_x_m, beam.end_x_m], [beam.start_y_m, beam.end_y_m], color='#cccccc', lw=.7)
    axes[0].scatter(*truth, marker='*', color='black', s=60)
    axes[1].plot(np.arange(len(values))+.5, values[:, beam_index], color='black', lw=.9)
    for method, color, marker in zip(['ia_vpdi','joint_nls_trf','matched_joint_adam'],
            [colors[0],colors[1],colors[3]], ['o','x','s']):
        fit = public_case['methods'][method]
        axes[0].scatter(*fit['source_m'], marker=marker, color=color, s=24, facecolors='none' if marker != 'x' else color)
        axes[1].plot(np.arange(len(values))+.5,
            np.asarray(fit['prediction']).reshape(-1,len(beams))[:,beam_index], color=color, ls='--', lw=.9)
    axes[0].set(xlabel='东西距离（米）', ylabel='南北距离（米）', title='真实源点与估计位置')
    axes[0].set_aspect('equal', adjustable='datalim')
    axes[1].set(xlabel='时间（分钟）', ylabel='扣除背景后的浓度 ppm', title='峰值最高光路的观测与预测')
    fig.legend(handles=[Line2D([],[],color=c,label=l) for c,l in [('black','真值或观测'),
        (colors[0],'IA-VPDI'),(colors[1],'Joint NLS'),(colors[3],'联合Adam')]],
        loc='upper center', ncol=4, frameon=False, fontsize=8)
    finish(fig, 'public_case')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['control', 'analysis', 'plots'], required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    {'control': lambda: run_control(args.device), 'analysis': analyze, 'plots': plots}[args.stage]()
