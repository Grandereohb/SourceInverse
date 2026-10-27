# 论文写作与实验专用脚本

此目录独立保存论文研究代码，不属于日常溯源和服务器部署入口。2026年10月21日从原scripts目录迁入42个专用脚本，相关21个测试文件迁入tests子目录。

## 目录职责

- `run_paper_*.py`、`extend_paper_*.py`：论文实验及补充场景运行。
- `paper_*.py`：论文专用输运适配、基线、缓存、源强求解和校准组件。
- `prepare_paper_*.py`、`audit_paper_*.py`：实验输入准备与来源检查。
- `analyze_paper_*.py`、`verify_paper_*.py`、`check_paper_*.py`：统计、数值检查与诊断。
- `plot_paper_*.py`：论文绘图。
- `build_paper_writing_data.py`、`build_paper_results_summary.py`：写作数据表和Word汇总。
- `tests/`：仅针对这些研究组件的单元测试。

原来的数据提取、Excel筛选、批量溯源、运行评估等脚本仍在项目根目录的`scripts/`。`pinn_source/`和`deployment/`的生产代码不在本次迁移范围。论文准备脚本可调用原有提取器，但原有脚本不反向依赖本目录。

## 使用方式

在项目根目录执行。原日常运行命令不变：

```powershell
.venv_clean/Scripts/python.exe scripts/run_recent_leak_source_inversions.py
```

论文脚本的新位置示例：

```powershell
.venv_clean/Scripts/python.exe paper_experiments/run_paper_shared_forward.py --help
.venv_clean/Scripts/python.exe -m unittest discover -s paper_experiments/tests -p "test_*.py" -v
```

Word汇总在安装了python-docx的Python环境中生成：

```powershell
python paper_experiments/build_paper_results_summary.py --packet result/paper_experiments_20261017/manuscript_results_v2
```

结果仍保存在`result/paper_experiments_20261017/`，本次没有移动或重算实验数据、图表和文档。

## 冻结实验的历史路径

既有protocol、源代码快照、结果和历史日志中的`scripts/...paper...`路径是当时的真实记录，迁移时未批量改写。严格复现旧实验应使用对应的冻结快照；不要为了绕过哈希校验修改旧协议。使用当前代码开展新版本实验时，源代码记录会采用`paper_experiments/`的新路径，仍应另建版本化结果目录。

此前的48例扩样后台任务已经完成。其系统任务入口路径同步到本目录并停用，不会在目录迁移后自动重跑旧实验。

## 新基线接入（2026-10-26，固定批次评估已完成）

- `paper_joint_baselines.py`：共同残差接口、带显式非负约束的Joint NLS/TRF、公开PySwarms固定系数GlobalBestPSO。两种求解器均不在优化迭代中消去Q。
- `run_paper_baseline_smoke.py`：读取原冻结输入与共同粗搜索初值，执行两例短流程；默认只有8次NLS函数评估预算与3代PSO，不能作为论文精度结果。临时有限边界尚未与IA正式比较协议统一。
- `paper_newman_adapter.py`：下载并校验作者Source 2反演notebook和sourceinversion 0.1.5发布包；提取作者原始后验定义，调用作者Manifold MALA within Gibbs。`run_paper_newman_event.py`使用既有事件窗口、七条光路及作者原始风统计函数接入真实数据；短流程已运行，完整事件后验尚需检查，不以接口单测代替实测通过。
- `requirements-baselines.txt`：PSO轻量研究依赖；当前安装于被忽略的`deployment_runs/paper_baseline_dependencies`，未改生产依赖。Newman旧版JAX栈不在此文件中混装。

正式实验必须另建协议/结果版本，先统一参数可行域、初始化与实际计算预算；历史协议和结果不改。新共享响应的外部物理参数接口保留原默认调用路径。

`run_paper_comparison_v4.py`已统一四方法有限参数域，读取既有216例输入，重估共同粗初值的Q；IA/联合Adam作对应的边界投影，不改变生产核心。三次PSO随机重复全部保存，`analyze_paper_comparison_v4.py`先在案例内取种子中位数再汇总案例，不取最好种子。计算预算是公开记录的更新/响应次数，不能声称等价FLOPs；时间不含复用的旧粗搜索。未完成批次的analysis_status明确为final=false。

Newman隔离运行环境位于`deployment_runs/newman_env`，Python3.10.12及作者固定直接依赖均已安装。旧JAXlib通过官方Google归档获取，见`requirements-newman.txt`。与作者全文实验不同之处（单窗口、共同背景扣除基准、初始化及先验差异）记录在每次运行报告中，不称原文表格的严格复现；短链不是收敛证据。

`run_paper_beam_comparison.py`在既有八个公开释放窗口及两种源高上运行相同常源强容量的四方法比较；正演仍为同一个物理光路适配器。`analyze_paper_beam_comparison.py`保留PSO全部重复、按事件及源高取种子中位数，缺少独立称重参考的窗口不计算源强相对误差。两种源高是敏感性设定，不算独立事件；Newman使用不同作者正演，不混入共同正演优化器排名。

`audit_paper_newman_result.py`只检查已保存链并回放物理参数均值处的预测，不重采样或改写链；该预测不等于后验预测均值。首个完整事件发现位置ESS约3、保留5000步仅14个不同MH状态，不能将其作为收敛后验比较。`paper_newman_initialization.py`补入作者似然筛选LHS初值，保留作者后验与采样器；LHS使用SciPy，范围来自场地及作者先验而非真值，差异和选中初值均单独记录。`--initialization generic`保留旧入口，`--initialization author-lhs`开启初始化适配，仍需检查完整链而不是以短流程有限性代替收敛。

## 新版图表与报告

`analyze_paper_comparison_v4.py`现在同时导出局部椭圆短半轴和面积，校准时按轴缩放因子的平方换算面积；只对已保存信息矩阵做后处理，不改变反演。`plot_paper_comparison_v4.py`生成六条件定位分布、源强与配对时间、可靠性覆盖与区域面积、公开释放逐事件定位四张图。默认要求相应批次完整；`--preview`只用于内部排版，显式标记未完成，不作为正式图。

`build_paper_comparison_report.py --check-inputs`核对实际案例、方法、Newman结果及链审计、校准表与正式图件，不生成Word。完整后才构建新报告，沿用原报告的实验目的、内容、方法、结果与图片组织；不会覆盖旧报告。构建须使用带python-docx的文档运行时，之后仍需Word/PDF逐页检查。Newman每个事件及源高的结果路径约定为`事件名_h03`或`事件名_h16`；链收敛问题必须随数据报告，不伪称作者原文严格复现或自动收敛。

`run_paper_newman_batch.py`顺序运行既有八窗口和两种源高，并调用链审计，不增加新条件或调参。只在确认先导完整链结束并检查混合情况后启动；已有兼容完整结果复用，发现不完整目录则停止，不能据文件标记自行重启仍存活任务。`analyze_paper_newman_batch.py`汇总该批次，输出ESS、MH状态变化及实测源强误差；`final`只表示固定预算批次齐全，不代表后验收敛认证。作者两个源的后验、LHS似然和先验已经AST对照一致，差异审计保存在研究依赖目录。

公开比较主批次已迁移至`beam_comparison_cuda_v1`，新图表和报告入口默认读此目录；`beam_comparison_v1`两个已完成CPU参考结果保留，不混入CUDA队列。首个完整案例的CPU/CUDA四方法对照及受控迁移记录保存在新批次中，求解预算、观测、边界和正演未改。该单例的硬件耗时差不作通用算法加速结论。

可靠性分析只依赖已冻结的96个名义案例，独立用`reliability_final`标记完成；不必等待后续合成扩样结束。Newman的未定义ESS或停滞链保留在统计中，ESS为空并附状态，不用0填补、不删案例，也不把批次完成等同于后验收敛。

`evaluate_paper_alternatives_v4.py`补齐原可靠性框架的近等价位置统计：仅重评原有96例各294个候选，在新共同边界内重解Q，不新增位置或更改已选源点。`sampled_alternatives.csv`及`alternatives_status.json`与局部椭圆、校准和风险曲线一并供新报告使用；有限候选池不是连续置信域。

`verify_paper_beam_comparison.py`在公开拟合结果处做32阶时间/释放求积回放，与16阶拟合预测作原1%阈值核验，不重新优化或按真值改精度。完整批次已统一复核；报告默认要求该核查齐全且通过，显式报告例外的入口见下文，不会把旧方法的48项通过记录移用于新基线。

## 本轮评估状态与使用边界

- 固定合成批次216例、公开共同正演16个事件/源高组合、作者Newman固定预算16组及原96例可靠性分析均已执行。研究测试106项，105通过、1项跳过。批次完成不代表作者采样链收敛，也不证明IA优于其他方法。
- 公开96项加密求积回放中92项通过既定1%阈值，4项未通过。报告默认严格检查不放宽阈值；显式`--allow-numerical-warnings`仅允许生成列出全部失败条目的完整报告，不能作为数值核查全部通过的证明。
- 事后诊断发现原合成PSO全部648次运行的历史最优目标未改善。这组数据保留用于诊断，不能据此推断PSO算法能力上限。后续尺度化PSO及约化信赖域验证使用独立结果版本，不覆盖该批次；未实现DE替代。
- `result/`中的真实/合成输入、冻结协议、逐案例结果和Word报告，以及`deployment_runs/`中的运行环境和日志均为本地研究资料，不随此次代码提交上传。运行正式入口需要准备对应输入和协议；结果目录不可混用或覆盖。
- 本目录不改变生产算法默认值，不参与日常批量溯源入口。不要将本地案例参数变动或敏感原始监测资料与研究代码一起批量提交。

## 尺度化求解器验证

`run_paper_solver_upgrade.py`只读旧比较批次的输入、共同初值和边界，在新目录保存结果。`--scope pilot`按元数据选四种几何及两种失配各首例，不按算法胜负筛选；`--scope full`使用原216例。`--methods ia`或`pso`可分开串行执行，PSO保留三随机种子，汇总用种子中位数。

`paper_reduced_solver.py`调用`pinn_source/reduced_trust_region.py`：每次试探位置/扩散时用列均衡BVLS求有界非负Q，以完整活跃集残差Jacobian进行物理坐标尺度化TRF；Jacobian含非零残差项，不将固定Q的偏导误作约化残差导数。生产入口另由`IA_VP_OUTER_SOLVER="trf"`选择，`"adam"`保留旧外层；不改变联合Adam分支或用户选择的算法模式。有限目标值和优化器收敛状态分别记录，活跃集及秩变化处仍是分段光滑问题，不保证全局最优。

PSO仍用公开PySwarms固定系数GlobalBestPSO。新增可选物理尺度速度限幅、共同初值附近的粒子初始化和零初速度；剩余粒子覆盖共同全域。Q仍与位置/扩散联合搜索，不使用NNLS、局部NLS或真值边界。新版预算较旧版增加，必须记录实际评价次数和耗时，不称公平等预算加速；先导调参案例不能另称独立验证集。目标下降不保证真实定位误差下降。

## IA释放复杂度选择

`--methods ia_release`在同一五帽函数输运空间中拟合常量/三帽/五帽三个非负源强子空间，用Gaussian工作似然BIC选择复杂度。模型内仍使用全观测、原SSE、有界Q消元和约化TRF；不按真实释放类型选容量。`--scope diagnostic`仅用于开发，除原六个代表例还包含已知失败例eval_061、eval_119、supp_029，不称独立验证。`ia_release_capacity_full_v1`已完成原216例、不删案例；其结果代表新增容量选择后的方法，不是单独归因于变量投影，也不是与固定五参数NLS的等容量消融。BIC在源位置不可辨识、约束活跃时只有工作模型意义，不能视为已校准的后验概率。

生产入口已增加默认关闭的`IA_VP_Q_CAPACITY_SELECTION="fixed"`，改为`"bic"`开启，容量由`IA_VP_Q_CAPACITY_COUNTS=(1,3,5)`控制，建议配合`IA_VP_OUTER_SOLVER="trf"`。也可使用环境变量`PINN_IA_VP_Q_CAPACITY_SELECTION=bic`。与自定义knots及local_templates布局互斥；积分时间节点必须足够，观测数需大于4+最大容量。粗搜索共享最大容量的输运响应，各容量分别保留起点再精修；定位诊断条件于所选容量，不把跨容量的MSE集合当作置信区间。研究216例采用原共同初值，生产入口包含每容量的粗搜索，二者不是逐位相同的执行流程；已用独立小流程验证生产接入。

## METEC公开固定站样本

`audit_metec_public_sample.py`核验固定作者提交的一周8站样本与释放记录，只有17次释放有完整时间覆盖，不把596条长期释放记录当作596个可运行案例。原数据保留于本地审查目录，数据再发布权限需另核实。

`run_metec_comparison.py --stage prepare`冻结全17例规则并生成输入；默认运行完整四方法（IA、Joint NLS、标准PSO三种子、联合Adam），只支持串行。需要现有PyTorch、SciPy、pandas和已准备的PySwarms；不安装新依赖。`--limit 1`仅为计算检查，不能当作完整结果。

`metec_point_adapter.py`复用三维物理输运核，采用点传感器高度及地面反射。源高作为共同已知条件，源位置/计量流率仅用于评价。全有效观测拟合，不做时空留出；IA使用1/3/5容量选择，其余固定5，不能声称纯消元等容量比较。背景、风缺测、分钟对齐及固定ppm转换均写入protocol.json。

完整拟合后运行`analyze_metec_comparison.py`，它要求17例全部完成并核验输入哈希，在保存参数处用更高阶积分及更长历史做回放，保留未通过1%门槛的条目。`plot_metec_comparison.py`与`build_metec_report.py`生成全样本图和简明Word；Word使用文档工具指定运行环境。图件须先完成字体/碰撞/对齐检查。

`build_focused_chapter_report.py --metec --output ...`可将新4.4接入完整实验简报；按用户要求移除正文24例相同排放表达对照，但不删除原实验数据或旧版报告。任何公开验证完成都不自动证明IA定位领先，应以全部事件的实际差异为准。
