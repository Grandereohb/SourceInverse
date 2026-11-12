"""Create an editable concise chapter report from verified refreshed CSVs."""
import csv
import json
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from build_paper_results_summary import font, paragraph, heading, table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/ia_vpdi_report_20261028'
DATA = OUT / 'data'
METHODS = ['ia_vpdi', 'joint_nls_trf', 'matched_joint_adam', 'standard_gbest_pso']
NAMES = dict(zip(METHODS, ['IA-VPDI', 'Joint NLS', '联合 Adam', '标准 PSO']))


def read(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def num(value):
    return f'{float(value):.2f}'


def pair(row, field='location_m', percent=False):
    factor = 100 if percent else 1
    return f'{float(row["mean_"+field])*factor:.2f} / {float(row["median_"+field])*factor:.2f}'


def methods(rows):
    return sorted(rows, key=lambda r: METHODS.index(r['method']))


def image(doc, stem, caption):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(OUT / 'figures' / (stem + '.png')), width=Cm(17.1))
    paragraph(doc, caption, size=9)


def build():
    status = json.loads((OUT / 'data_status.json').read_text(encoding='utf-8'))
    assert status['ready_for_report'] and status['synthetic_cases'] == 216 and status['metec_events'] == 17
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(1.6), Cm(1.6)
    sec.left_margin, sec.right_margin = Cm(1.8), Cm(1.8)
    normal = doc.styles['Normal']
    normal.font.name, normal.font.size = 'Arial', Pt(10)
    normal._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    for style in doc.styles:
        for border in list(style._element.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(footer.add_run('IA-VPDI 论文实验报告  2026年10月28日    '), 8)
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    p = doc.add_paragraph(style='Title')
    font(p.add_run('IA-VPDI 论文实验结果更新'), 21, True)
    paragraph(doc, '2026年10月28日  完整样本统计  平均值优先', size=10)
    paragraph(doc, '本版汇总当前已验收的 IA 方法。常规合成场景的平均定位误差为72.11米，METEC公开事件为7.91米，相比冻结的 Joint NLS 分别降低30.7%和20.2%。全部案例与不利结果均保留，不按算法胜负删样。', bold=True)
    heading(doc, '4.1 实验设置与评价方法')
    paragraph(doc, '目的：验证方法能否在稀疏监测下更准确地确定源位置，并重建释放过程。定位误差为估计源与真源的水平距离；合成真源由独立模拟输入给定，公开数据真源来自释放记录。源强误差与浓度拟合误差单独评价。')
    table(doc, ['实验部分', '数据与规模', '本轮处理'], [
        ['4.2 定位与源强', '192例常规＋24例失配', '核对最佳版本，重算全部统计'],
        ['4.3 释放表达与误定位', '192例及既有容量结果', '重算消融与误定位统计'],
        ['4.4 公开数据', 'METEC 17例', '更新全部IA结果，基线不变'],
        ['补充公开与可靠性', 'Chilbolton 8窗×2源高；96例诊断', '保留完整数据和原有质量标记']], [3.7, 6.6, 7.1])
    paragraph(doc, '方法：IA采用非负源强消元、尺度化约化信赖域求解，并从1、3、5个源强参数中按BIC选择表达复杂度。对照为 Joint NLS、联合 Adam 和标准 PSO；PSO每例使用3个固定随机种子的中位结果，不取最优种子。')
    heading(doc, '主要结果概览')
    syn = read('synthetic_cases.csv')
    public = {r['method']: r for r in read('metec_summary.csv')}
    nominal = {r['method']: r for r in read('synthetic_summary.csv') if r['cohort'] == 'nominal'}
    overview = []
    for method in METHODS:
        subset = [r for r in syn if r['method'] == method]
        mean_all = sum(float(r['localization_error_m']) for r in subset) / len(subset)
        overview.append([NAMES[method], num(nominal[method]['mean_location_m']), num(mean_all), num(public[method]['mean_location_m'])])
    table(doc, ['方法', '常规192例 均值m', '全部216例 均值m', 'METEC17例 均值m'], overview, [3.6, 4.6, 4.6, 4.6])
    paragraph(doc, '统计口径：后续表格中的“均值 / 中位数”顺序统一。合成源强误差为整段时序相对L2误差；公开数据为估计平均释放率的相对误差。192例为主分析，216例混合汇总仅作补充。公开IA利用三风站估计空间变化风场，冻结NLS使用原平均风模型，因此不是只更换优化器的对照。', size=9.5)

    doc.add_page_break()
    heading(doc, '4.2 合成场景定位与源强重建')
    paragraph(doc, '内容：四类站网各48例，覆盖不同风况及恒定、脉冲、双峰释放。实际六站仅复用真实站网几何，浓度和真源仍为合成数据。观测为6个小时均值；所有方法使用相同观测，合成噪声标准差为0.01。')
    table(doc, ['方法', '定位误差 均值 / 中位数 m', '源强误差 均值 / 中位数 %'],
        [[NAMES[m], pair(nominal[m]), pair(nominal[m], 'q_error', True)] for m in METHODS], [3.6, 7.0, 6.8])
    image(doc, 'synthetic_networks', '图1  四类站网的平均定位误差，每类48例，均包含全部方法与样本。')
    networks = read('network_summary.csv')
    labels = dict(ideal5='理想五站', jjj6='实际六站几何', surround5_pulse='环绕五站脉冲', corridor6_double_peak='走廊六站双峰')
    table(doc, ['网络 均值m', 'IA', 'NLS', 'Adam', 'PSO'],
        [[label] + [num(next(r['mean_location_m'] for r in networks if r['network'] == key and r['method'] == m)) for m in METHODS] for key, label in labels.items()], [5.0, 3.1, 3.1, 3.1, 3.1])
    table(doc, ['失配场景 均值m', 'IA', 'NLS', 'Adam', 'PSO'],
        [[label] + [num(next(r['mean_location_m'] for r in read('synthetic_summary.csv') if r['cohort'] == key and r['method'] == m)) for m in METHODS]
         for key, label in [('q_mismatch', '释放形状失配 12例'), ('wind_mismatch', '风场失配 12例')]], [5.0, 3.1, 3.1, 3.1, 3.1])
    paragraph(doc, '结果：IA的总体均值改善主要来自实际六站等困难案例；走廊六站中IA与NLS接近。释放形状失配中IA仍未优于NLS，说明当前低维释放表达不能覆盖所有真实释放过程。', size=9.5)

    doc.add_page_break()
    heading(doc, '4.3 释放表达与错误定位的关系')
    paragraph(doc, '目的：检查浓度已经拟合较好时，是否仍会出现位置偏差，以及释放表达是否影响定位。这里评价完整方法的作用，不把所有差异都归因于源强消元。按此前要求，已删除主报告中24例相同排放容量的对照实验。')
    paragraph(doc, '误定位统计：以拟合RMSE≤0.02且定位误差>100米定义“拟合好但定位偏差大”。0.02等于合成噪声标准差的两倍；100米是描述性阈值，50米和200米的完整结果另附CSV。')
    wrong = methods(read('wrong_location_summary.csv'))
    table(doc, ['方法', '浓度拟合达标', '达标但误定位', '定位误差90分位 m'],
        [[NAMES[r['method']], r['good_fit']+'/192', r['good_fit_wrong']+'/192', num(r['p90_location_m'])] for r in wrong], [4.0, 4.2, 4.5, 4.7])
    paragraph(doc, 'IA相对NLS将该类案例由39例降至34例，并将定位误差90分位从207.26米降至154.00米；仍存在34例，不能写成已经消除了位置与源强的补偿问题。', bold=True, size=10)
    table(doc, ['源强表达', '定位 均值 / 中位数 m', '源强误差中位数 %'],
        [[{'fixed_1': '固定1参数', 'fixed_3': '固定3参数', 'fixed_5': '固定5参数', 'adaptive_bic': '1/3/5自动选择'}[r['mode']], pair(r), num(100*float(r['median_q_error']))] for r in read('capacity_summary.csv')], [5.2, 7.3, 4.9])
    paragraph(doc, '容量消融使用同一批192例已保存的拟合，比较释放表达，而非新旧代码版本。自动选择保留复杂释放的表达能力，同时避免恒定释放被过度参数化；不需要新增经验损失项。')
    image(doc, 'fit_vs_location', '图2  全部192例中IA与NLS的拟合误差和定位误差。水平虚线为100米，竖向点线为拟合达标阈值；纵轴为对称对数刻度。')

    doc.add_page_break()
    heading(doc, '4.4 METEC公开受控释放验证')
    paragraph(doc, '数据：17个不重叠的单源释放事件，8个浓度站、3个风站，涉及5个设备组和11个记录源坐标。定位真值和源强来自公开释放记录，不是17个独立场地。保留全部有限观测与带符号的背景扣除残差。')
    paragraph(doc, '本版IA：根据风站位置和观测估计水平空间风场，使用精确高斯输运、地面反射及非负源强消元，并按数值误差检查决定是否精化。相同的物理分支用于全部17例，不按真值为每例挑选模型。')
    table(doc, ['方法', '定位 均值 / 中位数 m', '源强相对误差 均值 / 中位数 %'],
        [[NAMES[m], pair(public[m]), pair(public[m], 'q_error', True)] for m in METHODS], [3.6, 6.2, 7.6])
    paragraph(doc, '主要结果：IA平均定位误差7.91米，较NLS的9.92米降低2.01米，约20.2%，在四种方法中均值最低。中位数不是四种方法中最低；源强误差仍较大，因此定位改善不能写成源强也已精确恢复。', bold=True)
    image(doc, 'metec_all_events', '图3  全部17个公开事件的定位误差，编号对应下一页逐例表。未删除IA不占优的事件。')
    paragraph(doc, '对比口径：NLS、Adam和PSO结果沿用原冻结版本，没有为配合IA更新其代码或重新调参。由于IA新增了空间风场处理，本表支持完整方法的平均定位改善，不能单独证明消元求解优于联合求解。')
    paragraph(doc, '数值核验：17例IA均通过自适应精度门槛；其中3例进行了更细积分，再用更高精度回放检查。该检查验证保存参数的计算稳定性，不替代释放真值定位评估。三风站适配器仍属实验入口，尚未接入单风站生产接口。', size=9.5)

    doc.add_page_break()
    heading(doc, '4.4 全部公开事件逐例结果')
    rows = read('metec_cases.csv')
    ids = sorted({r['case_id'] for r in rows})
    table(doc, ['编号与原始事件ID', 'IA m', 'NLS m', 'Adam m', 'PSO m'],
        [[f'{i:02d}  {case}'] + [num(next(r['localization_error_m'] for r in rows if r['case_id'] == case and r['method'] == m)) for m in METHODS] for i, case in enumerate(ids, 1)], [5.8, 2.9, 2.9, 2.9, 2.9])
    paragraph(doc, '均值 / 中位数：IA 7.91 / 8.06米；NLS 9.92 / 8.16米；Adam 10.48 / 7.93米；PSO 10.32 / 7.68米。全部原始精度数值及源强误差见 metec_cases.csv。', bold=True)
    heading(doc, '这组公开数据支持什么结论')
    paragraph(doc, '在完整17例开发队列中，当前IA版本的平均定位误差相对冻结NLS降低20.2%。该结论可作为论文中的公开数据验证结果；不表述为每个事件都更优，也不使用“统计显著领先”。这些事件曾用于方法开发，仍需在论文中说明，不能称为完全未见的独立外部测试。')
    paragraph(doc, '主线建议：常规合成实验说明总体定位和困难案例的改善；释放表达实验说明避免不合适的源强复杂度；METEC说明空间风场与源强消元相结合的实测效果。三个层次互相补充，不用浓度拟合好替代定位准确。')

    doc.add_page_break()
    heading(doc, '补充公开数据与资料索引')
    paragraph(doc, 'Chilbolton：8个释放窗口、2个实际源位置、7条光路；0.3米和1.6米两种源高分别保留，不按误差择优。这里沿用恒定源强设定，缺少METEC式三风站空间输入，因此不强行套用同一个风场适配器。')
    table(doc, ['源高m', '方法', '定位 均值 / 中位数 m', 'Q误差 均值 / 中位数 %'],
        [[r['height_m'], NAMES[r['method']], pair(r), pair(r, 'q_error', True)] for r in sorted(read('chilbolton_summary.csv'), key=lambda r: (float(r['height_m']), METHODS.index(r['method'])))], [2.0, 3.7, 5.6, 6.1])
    paragraph(doc, '定位覆盖每个源高的8个窗口；源强误差仅使用有独立称重记录的6个窗口。IA与NLS定位结果仍基本一致，作为补充验证保留，不把METEC的改善推广至这个数据集。')
    table(doc, ['作者方法单列参照', '窗口数', '定位 均值 / 中位数 m'],
        [[f'Newman  源高{r["height_m"]}m', r['n'], pair(r)] for r in read('newman_summary.csv')], [7.0, 3.0, 7.4])
    paragraph(doc, 'Newman采用作者Gaussian plume和M-MALA-within-Gibbs实现，物理模型与先验不同，不并入同正演求解器排名。16次运行中有3次位置有效样本数未定义、3次保留链停滞；因此这里只保留结果记录，不能据此宣称优于充分收敛的贝叶斯方法。', size=9.5)
    paragraph(doc, '质量说明：Chilbolton原48组数值回放有43组通过1%门槛，5组略超门槛，最大约1.039%，完整保留标记。96例定位区域诊断另存补充数据；校准只描述区域覆盖，不改变源位置拟合，不恢复已弃用的时空交叉拟合。', size=9.5)
    heading(doc, '本轮更新与保存内容')
    paragraph(doc, '已完成：核对全部216例合成结果并重算统计；更新17例METEC的最佳IA结果；保留全部冻结基线、Chilbolton和作者方法记录；重绘本报告3幅图。没有重新运行大规模反演，也没有修改生产配置或用户数据。')
    paragraph(doc, '数据文件：data目录包含逐例、分组、源强容量、误定位阈值、数值检查和可靠性CSV；figures含PNG、PDF、SVG；data_status.json记录输入路径与校验信息。此前报告和实验原件均保留。论文可据此开始写作，源强偏差和释放形状失配作为明确的适用范围说明。', size=9.5)
    path = OUT / 'IA-VPDI论文实验结果更新报告_20261028.docx'
    doc.save(path)
    print(path)


if __name__ == '__main__':
    build()
