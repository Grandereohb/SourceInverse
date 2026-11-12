"""Read-only delivery of the recommended IA results; optional short replay, no fitting."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pinn_source'))
STUDY = ROOT / 'result/paper_experiments_20261017'
PUBLIC = STUDY / 'metec_comparison_20261027_endtime_q16'
PREVIOUS = STUDY / 'ia_affine_wind_sqrt_full_20261028'
RECOMMENDED = STUDY / 'ia_affine_numerical_repair_20261028'
DEST = ROOT / 'output/ia_vpdi_best_20261028'
METHODS = ('ia_vpdi', 'joint_nls_trf', 'matched_joint_adam', 'standard_gbest_pso')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def check_baseline():
    baseline = read(ROOT / 'paper_experiments/nls_baseline_frozen.json')
    for name, expected in baseline['sha256_lf'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            raise ValueError('Frozen NLS changed: ' + name)
    return baseline['revision']


def validate_cohort(frame, count):
    if set(frame.method) != set(METHODS):
        raise ValueError('Keep all four original comparison methods')
    if frame.duplicated(['case_id', 'method']).any():
        raise ValueError('Duplicate case-method rows')
    wide = frame.pivot(index='case_id', columns='method', values='localization_error_m')
    if len(wide) != count or not np.isfinite(wide.to_numpy()).all() or (wide < 0).any().any():
        raise ValueError('Incomplete or invalid fixed cohort')
    return wide


def selected_public(case):
    status = read(RECOMMENDED / 'status.json')
    refined = case in status['refined_events']
    folder = RECOMMENDED if refined else PREVIOUS
    fit_path = folder / case / ('ia_fit.json' if refined else 'ia_affine_wind.json')
    return fit_path, folder / case / 'wind_calibration.json', refined


def replay(case, minutes, device):
    import torch
    from ia_metec_point_adapter import configure
    from paper_sqrt_affine_wind import SquareRootAffineWindHistory
    check_baseline()
    if case not in read(PUBLIC / 'protocol.json')['event_ids']:
        raise ValueError('Unknown validated public event')
    data = dict(np.load(PUBLIC / 'inputs' / case / 'input.npz'))
    meta = read(PUBLIC / 'inputs' / case / 'metadata.json')
    fit_path, wind_path, refined = selected_public(case)
    fit, wind = read(fit_path), read(wind_path)
    query = np.asarray(minutes, int)
    if (query.ndim != 1 or not len(query) or np.any(np.diff(query) <= 0) or
            query.min() < 0 or query.max() >= len(data['concentration'])):
        raise ValueError('Provide increasing unique observation minute indices within the event')
    history = SquareRootAffineWindHistory(np.arange(len(data['concentration']) + 1) * 60.,
        wind['velocity_intercepts_m_s'], wind['gradient_per_s'], data['wind_xy'])
    order, age = (32, 40) if refined else (16, 20)
    torch.set_num_threads(1)
    m, x, b, response, _ = configure(data, meta['source_height_m'], device=device,
        time_order=order, release_order=order, history_minutes=age,
        observation_minutes=query, transport_history=history)
    theta = m.xs.new_tensor([*fit['source_norm'], fit['diffusion_norm'][1],
                            fit['diffusion_norm'][0] - fit['diffusion_norm'][1]])
    with torch.no_grad():
        actual = (response(m, 0., x, b, physical_parameters=theta) @
                  m.xs.new_tensor(fit['coefficients'])).cpu().numpy()
    valid = np.flatnonzero(np.isfinite(data['concentration'].ravel()))
    rows = np.arange(data['concentration'].size).reshape(data['concentration'].shape)[query].ravel()
    rows = rows[np.isfinite(data['concentration'].ravel()[rows])]
    expected = np.asarray(fit['prediction'])[np.searchsorted(valid, rows)]
    relative = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))
    if relative > 1e-7 or not np.isfinite(actual).all():
        raise ValueError('Recommended operator no longer reproduces saved parameters')
    record = dict(event_id=case, observation_minutes=query.tolist(), relative_l2=relative,
        observations=len(actual), prediction_norm=float(np.linalg.norm(actual)),
        passed=True, fitting_performed=False, source_truth_accessed=False, device=device)
    save(DEST / ('replay_' + case + '.json'), record)
    print(record)


def build():
    revision = check_baseline()
    public_protocol = read(PUBLIC / 'protocol.json')
    cases = public_protocol['event_ids']
    status = read(RECOMMENDED / 'status.json')
    checks = status['numerical_rows']
    if (len(cases) != 17 or len(checks) != 17 or set(r['event_id'] for r in checks) != set(cases) or
            not status['numerically_passed'] or not all(r['passed'] and r['relative_l2'] <= .01 for r in checks)):
        raise ValueError('Recommended public cohort has not passed all numerical gates')
    for name, expected in public_protocol['input_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Original public inputs changed: ' + name)
    synthetic_path = STUDY / 'paper_focused_20261027/analysis/case_flags.csv'
    audit = read(STUDY / 'synthetic_paired_accuracy_audit_20261028/summary.json')
    if digest(synthetic_path) != audit['source_sha256']:
        raise ValueError('Original synthetic cohort changed')
    synthetic = pd.read_csv(synthetic_path)
    groups = {'nominal': 192, 'q_mismatch': 12, 'wind_mismatch': 12}
    if set(synthetic.cohort) != set(groups):
        raise ValueError('Do not omit a synthetic scenario group')
    datasets = {}
    for group, count in groups.items():
        frame = synthetic[synthetic.cohort == group].copy()
        validate_cohort(frame, count)
        datasets['synthetic_' + group] = frame
    combined = synthetic.copy()
    combined['case_id'] = combined.cohort + ':' + combined.case_id
    validate_cohort(combined, 216)
    datasets['synthetic_all'] = combined
    public_path = PUBLIC / 'analysis/case_metrics.csv'
    frame = pd.read_csv(public_path, dtype={'event_id': str}).rename(columns={'event_id': 'case_id'})
    validate_cohort(frame, 17)
    parameters, result_hashes = [], {}
    from ia_metec_point_adapter import xy_from_lon_lat, window_mean_release
    for case in cases:
        fit_path, wind_path, refined = selected_public(case)
        fit = read(fit_path)
        data = dict(np.load(PUBLIC / 'inputs' / case / 'input.npz'))
        meta = read(PUBLIC / 'inputs' / case / 'metadata.json')
        truth = read(PUBLIC / 'inputs' / case / 'truth.json')
        true_xy = xy_from_lon_lat([truth['lon']], [truth['lat']], meta['origin'])[0]
        xy = np.asarray(fit['source_norm']) * 100.
        error = float(np.linalg.norm(xy - true_xy))
        q = 3600 * window_mean_release(fit['coefficients'], len(data['concentration']) * 60., data['release_window_s'])
        mask = frame.case_id.eq(case) & frame.method.eq('ia_vpdi')
        frame.loc[mask, ['localization_error_m', 'q_relative_error', 'rmse_ppm']] = [
            error, abs(q / truth['mean_q_kg_h'] - 1), float(np.sqrt(fit['mse']))]
        frame.loc[mask, 'replay_relative_l2'] = next(row['relative_l2'] for row in checks if row['event_id'] == case)
        # The old timing column describes a different operator; do not relabel it.
        frame.loc[mask, 'seconds'] = np.nan
        frozen = read(PUBLIC / 'fits' / case / 'joint_nls_trf.json')
        nls_error = np.linalg.norm(np.asarray(frozen['source_norm']) * 100 - true_xy)
        np.testing.assert_allclose(frame.loc[frame.case_id.eq(case) & frame.method.eq('joint_nls_trf'),
            'localization_error_m'].to_numpy(), [nls_error], atol=1e-10, rtol=1e-10)
        lon = meta['origin'][0] + np.rad2deg(xy[0] / (6371000 * np.cos(np.deg2rad(meta['origin'][1]))))
        lat = meta['origin'][1] + np.rad2deg(xy[1] / 6371000)
        parameters.append(dict(event_id=case, source_lon_lat=[float(lon), float(lat)],
            source_norm=fit['source_norm'], source_height_m=meta['source_height_m'],
            diffusion_norm=fit['diffusion_norm'], diffusion_m2_s=[float(d * 10000 / 60) for d in fit['diffusion_norm']],
            base_hat_coefficients_kg_s=fit['coefficients'], q_basis_count=fit['q_basis_count'],
            duration_s=len(data['concentration']) * 60., release_window_s=data['release_window_s'].tolist(),
            selected_by='Numerical threshold trigger only; never minimum source error across variants',
            refined=refined, fit_path=str(fit_path.relative_to(ROOT)), wind_path=str(wind_path.relative_to(ROOT))))
        for path in [fit_path, wind_path, PUBLIC / 'fits' / case / 'joint_nls_trf.json']:
            result_hashes[str(path.relative_to(ROOT))] = digest(path)
    validate_cohort(frame, 17)
    datasets['public_metec'] = frame
    summaries = []
    for dataset, data in datasets.items():
        for method, part in data.groupby('method'):
            summaries.append(dict(dataset=dataset, method=method, cases=len(part),
                mean_localization_m=float(part.localization_error_m.mean()),
                median_localization_m=float(part.localization_error_m.median())))
    summary = pd.DataFrame(summaries)
    DEST.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DEST / 'summary.csv', index=False, encoding='utf-8-sig')
    synthetic.to_csv(DEST / 'synthetic_cases.csv', index=False, encoding='utf-8-sig')
    frame.to_csv(DEST / 'public_cases.csv', index=False, encoding='utf-8-sig')
    save(DEST / 'public_parameters.json', dict(base_q_definition='Five hats over0..duration_s, native1/3/5 mapped to5',
        wind_model='b(t)+A(t)x, trace(A)=0, no mean vertical velocity', events=parameters))
    names = ['deliver_best_ia.py', 'ia_metec_point_adapter.py', 'ia_physical_beam_adapter.py',
        'paper_affine_wind.py', 'paper_sqrt_affine_wind.py', 'paper_sqrt_gaussian.py',
        'paper_q_model_selection.py', 'paper_reduced_solver.py', 'paper_fast_parcels.py']
    sources = [ROOT / 'paper_experiments' / name for name in names]
    sources += [ROOT / 'pinn_source' / name for name in ['reduced_trust_region.py', 'q_capacity.py']]
    for source in sources:
        target = DEST / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    save(DEST / 'manifest.json', dict(version='ia_best_20261028',
        acceptance='User-confirmed clear advantage in complete-cohort mean, not statistical significance',
        nls_revision=revision, nls_fitted_or_modified=False, fitting_performed=False,
        synthetic_cases=216, synthetic_primary_cases=192, public_cases=17,
        public_numerically_passed=17, empirical_loss_added=False,
        recommendation='Reduced TRF + nonnegative Q elimination/BIC; public data use validated traceless spatial wind',
        production_config_changed=False, public_requires_three_spatial_wind_stations=True,
        scope='Development-cohort results; full-method, not isolated variable-projection superiority',
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources}, result_sha256=result_hashes,
        synthetic_source_sha256=digest(synthetic_path), public_baseline_table_sha256=digest(public_path)))
    def value(dataset, method):
        return summary[(summary.dataset == dataset) & (summary.method == method)].iloc[0]
    headline = []
    for label, dataset in [('常规合成', 'synthetic_nominal'), ('全部合成（含失配，附加汇总）', 'synthetic_all'), ('METEC公开事件', 'public_metec')]:
        ia, nls = value(dataset, 'ia_vpdi'), value(dataset, 'joint_nls_trf')
        headline.append(f'| {label} | {ia.cases} | {ia.mean_localization_m:.2f} | {nls.mean_localization_m:.2f} | {(1-ia.mean_localization_m/nls.mean_localization_m)*100:.1f}% |')
    text = '# IA 最佳版本交付说明\n\n'
    text += '验收口径：完整样本平均定位误差有明显优势；不要求统计显著，不删样、不修改冻结NLS。\n\n'
    text += '| 数据 | 案例数 | IA平均误差/m | 冻结NLS平均误差/m | 平均降幅 |\n|---|---:|---:|---:|---:|\n' + '\n'.join(headline)
    text += '\n\n## 保留版本\n\n保留约化信赖域TRF、非负源强消元和1/3/5源强容量选择。公开数据保留已验证的二维空间风场，不采用三维垂直闭合、共同背景或相关速度等后续试验分支。公开17例全部通过自适应数值门槛，其中3例经过精化；不是全部使用同一高阶重算。\n'
    text += '\n## 完整比较\n\n| 方法 | 常规192例平均/m | 全部216例平均/m | 公开17例平均/m |\n|---|---:|---:|---:|\n'
    labels = dict(ia_vpdi='推荐IA', joint_nls_trf='冻结Joint NLS', matched_joint_adam='联合Adam', standard_gbest_pso='标准PSO')
    for method in METHODS:
        numbers = [value(dataset, method).mean_localization_m for dataset in ['synthetic_nominal', 'synthetic_all', 'public_metec']]
        text += '| ' + labels[method] + ' | ' + ' | '.join(f'{number:.2f}' for number in numbers) + ' |\n'
    text += '\n全部逐例数据、中位数及失配分组均保留在CSV。释放形状失配12例中，IA平均误差仍高于NLS；总体领先不代表每类场景都领先。216例附加汇总按案例等权，不替代原192例主分析。\n'
    text += '\n公开IA使用多风站空间输运及数值精化，冻结NLS使用原平均风正演，因此这是完整方法对比，不能归因于单独源强消元。公开数据的统计置信区间仍跨零，不写成已证明普遍统计显著领先。\n'
    text += '\n## 使用与复现\n\n在仓库根目录、已有Python环境运行：\n\n```powershell\npython paper_experiments/deliver_best_ia.py\npython paper_experiments/deliver_best_ia.py --replay-case 20220510001 --minutes 40 80 --device cpu\n```\n\n默认只核验和整理表格；第二条只做指定分钟的前向重建，不训练、不运行NLS。源点、源强和扩散参数见public_parameters.json。source_snapshot封存核心代码，原始输入和完整实验记录仍在原result目录；这不是独立部署包。\n'
    text += '\n生产入口仍为pinn_source/config.py中的INVERSION_MODE，本次不改用户当前配置。实验的通用求解组合为INVERSION_MODE="ia_vpdi"、IA_VP_OUTER_SOLVER="trf"、IA_VP_Q_CAPACITY_SELECTION="bic"、IA_VP_Q_CAPACITY_COUNTS=(1,3,5)。当前用户配置中的fixed源强容量没有被自动覆盖。公开数据的空间风场需要三座独立风站及其位置，仍通过专用适配器计算；不能承诺仅切换配置就让单风站案例获得同样20%的改善，也不声称已完成该空间风场的生产接口接入。\n'
    (DEST / 'README.md').write_text(text, encoding='utf-8')
    print(summary.to_string(index=False))
    print('DELIVERY', DEST)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-case')
    parser.add_argument('--minutes', type=int, nargs='+')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    arguments = parser.parse_args()
    if arguments.replay_case:
        if arguments.minutes is None:
            parser.error('--replay-case requires explicit --minutes to avoid an accidental long replay')
        replay(arguments.replay_case, arguments.minutes, arguments.device)
    else:
        build()
