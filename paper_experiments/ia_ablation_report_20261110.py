"""Summarize paired controls without selecting winning cases."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/ablation_20261110'


def main():
    frame = pd.read_csv(OUT/'case_metrics.csv')
    pairs = pd.read_csv(OUT/'paired_differences.csv')
    assert len(frame[frame.dataset.eq('metec')]) == 68
    assert len(frame[frame.dataset.eq('synthetic')]) == 576
    if not frame.comparable.all():
        raise ValueError('Archive equivalence failed: do not report unmatched cells')
    summary = pd.read_csv(OUT/'summary.csv')
    paired = pairs.groupby(['dataset','contrast']).agg(n=('delta_location_m','size'),
        mean_delta_m=('delta_location_m','mean'), median_delta_m=('delta_location_m','median'),
        p90_delta_m=('delta_location_m',lambda s:s.quantile(.9)),
        mean_delta_rmse=('delta_rmse','mean'), mean_delta_q=('delta_q_error','mean'),
        mean_delta_seconds=('delta_seconds','mean')).reset_index()
    paired.to_csv(OUT/'paired_summary.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.unicode_minus':False})
    fig, axes = plt.subplots(1,2,figsize=(10.2,4.0))
    labels = [['平均风\nVP+BIC－NLS','空间风\nVP+BIC－NLS','VP+BIC\n空间风－平均风','NLS\n空间风－平均风'],
              ['固定5系数\nVP－NLS','变投影\nBIC－固定5']]
    for ax,dataset,names in zip(axes,['metec','synthetic'],labels):
        groups=list(dict.fromkeys(pairs[pairs.dataset.eq(dataset)].contrast))
        for i,group in enumerate(groups):
            values=pairs[(pairs.dataset==dataset)&(pairs.contrast==group)].delta_location_m.to_numpy()
            ax.scatter(i+np.linspace(-.12,.12,len(values)),values,s=12,color='#7E929D',alpha=.6,edgecolors='none')
            ax.scatter(i,values.mean(),marker='D',s=42,color='#2876A8',zorder=5)
        ax.axhline(0,color='#555555',lw=.8)
        ax.set_yscale('symlog',linthresh=1)
        ax.set_xticks(range(len(names)),names,fontsize=9)
        ax.set_ylabel('逐例定位误差差值（米，负值表示前者更好）')
        ax.set_title(('a  METEC 17例' if dataset=='metec' else 'b  合成192例'),loc='left',fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    fig.tight_layout()
    sys.path.insert(0,'C:/Users/98567/.codex/skills/nature-figure/scripts')
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig, tolerance_pt=1.5, strict=True,
        json_out=OUT/'paired_ablation.alignment.json')
    fig.savefig(OUT/'paired_ablation.png',dpi=400,facecolor='white')
    fig.savefig(OUT/'paired_ablation.pdf',facecolor='white')
    fig.savefig(OUT/'paired_ablation.svg',facecolor='white')
    plt.close(fig)
    text=['# 污染源定位论文消融实验','日期：2026-11-10。全部结果复用归档；本轮无优化器运行。',
        '## 结论',
        '统一配置下，METEC平均风场的VP+BIC/NLS均值为9.968/9.920米，空间风场为7.887/7.530米。同风场下VP+BIC未优于NLS；空间风场使VP+BIC降低2.081米、使NLS降低2.391米。因此此前现场总体改善主要由风场设定解释，不能归为变投影求解流程独有优势。',
        '合成192例：固定5系数NLS为104.043米，固定5系数VP为105.302米，VP+BIC为72.107米。固定自由度后VP平均高1.260米，逐例差值中位数约为零；同一VP流程启用BIC后平均降低33.196米（31.5%）。当前证据支持自适应排放容量选择的定位收益，不支持单纯源强消元必然改善定位。',
        '## 配置及复用核验',
        'METEC使用全部17事件，统一16阶观测积分、16阶释放积分、20分钟历史，同一IA点式烟团输运实现复核四格的全部有效观测预测。事件输入、背景、释放支持、原公共两起点、参数边界和搜索域不变。四格历史结果在统一响应下相对差≤1e-6才复用，逐格核验见audit。',
        '空间风NLS采用既有历史对照，不替换7563376冻结NLS基线，不修改或重新执行NLS。7.91米版本中的3例额外精化使用32阶积分及已拟合位置起点，故不与原起点预算混拼；本表采用精化前全部17例共同协议版本。',
        '归档预测复现仅验证同一离散目标，不等于所有结果通过1%高阶数值门槛。逐例numerical_gate_passed保留原高阶回放状态；本表用于统一16阶计算配置的机制对照，不冒充全部通过数值精化后的最终精度。',
        'METEC NLS固定5系数，VP+BIC拟合1/3/5后选择，每容量每起点最多250次评估，两起点；完整流程预算不同本就是此次处理因素。',
        '合成192例位置及扩散初值相同，原NLS上限500次/起点，VP每容量250次/起点，因此是原预定预算比较，不是等预算最优化理论比较。实际计数见逐例CSV；BIC档案只保留选中容量的计数，固定5的容量子记录未保留计数，不把缺失值填0，也不把选中容量计数冒充三容量总工作。运行时间沿用历史记录，受当时硬件负载影响，不作严格速度基准。',
        '## 汇总表',
        '|数据|组别|n|定位均值m|中位数m|90分位m|浓度RMSE均值|源强误差均值%|历史求解时间均值s|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        text.append(f'|{r.dataset}|{r.group}|{r.n}|{r.mean_m:.3f}|{r.median_m:.3f}|{r.p90_m:.3f}|{r.rmse_mean:.6f}|{100*r.q_error_mean:.2f}|{r.seconds_mean:.2f}|')
    text += ['合成源强误差为整段时序相对L2误差；METEC为事件平均释放率相对误差。RMSE均为全部有效观测上的未加权拟合误差，两个数据集的浓度单位不同，不跨数据集平均。',
        '## 逐例配对差异','差值为组名前者减后者，负值表示前者定位更好。',
        '|比较|n|定位差均值m|定位差中位数m|','|---|---:|---:|---:|']
    for r in paired.itertuples():
        text.append(f'|{r.dataset}: {r.contrast}|{r.n}|{r.mean_delta_m:.4f}|{r.median_delta_m:.4f}|')
    text += ['## 解释范围','用同风场的VP+BIC减NLS描述完整求解流程作用，用同流程的空间风减平均风描述风场作用。两种作用可能交互，不能把一个总差值拆成唯一百分比。固定5的VP减NLS检验固定释放自由度下的收益；BIC减固定5检验容量选择。',
        '图中每点是一个案例，蓝色菱形是配对均值，纵轴采用对称对数尺度；未按差值过滤案例。',
        '原始失败/未收敛结果不删除，converged字段保留。缺失的原始计算量明确留空。原始JSON不改动，逐例source可追溯。若需严格共同高精度数值门槛下的2×2，需要另定对所有格一致的数值精化协议，不能仅替换IA三例。']
    gate=frame[frame.dataset.eq('metec')].groupby('group').numerical_gate_passed.sum()
    text += ['## 数值及未收敛记录',
        '原1%高阶回放通过数：'+ '；'.join(f'{k} {int(v)}/17' for k,v in gate.items())+'。未通过结果全部保留，因此这里完成的是统一离散模型的对照归因，不是四格均通过高阶物理数值验收的宣称。',
        '平均风VP+BIC有1例选中容量优化器未收敛；合成NLS有1例未收敛，均保留在汇总中。固定5 VP子记录的运行时间有记录，但响应/导数调用数未记录；完整BIC运行时间为三容量总耗时，其计数仅能追溯选中容量，不能用于虚构全流程工作量。']
    (OUT/'消融实验报告.md').write_text('\n'.join(text),encoding='utf-8')
    (OUT/'completion.json').write_text(json.dumps(dict(complete=True,metec_rows=68,synthetic_rows=576,
        new_optimizer_runs=0,reused_all=True,production_baseline_unchanged=True),indent=2),encoding='utf-8')
    print(paired.to_string(index=False))


if __name__=='__main__':
    main()
