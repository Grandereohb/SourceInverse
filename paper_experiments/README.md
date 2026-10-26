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
- 事后诊断发现当前合成PSO全部648次运行的历史最优目标未改善，初始化尺度和速度配置仍需验证。这组数据保留用于诊断，不能据此推断PSO算法能力上限。尚未实现后续讨论中的PSO调整、DE替代或约化信赖域求解。
- `result/`中的真实/合成输入、冻结协议、逐案例结果和Word报告，以及`deployment_runs/`中的运行环境和日志均为本地研究资料，不随此次代码提交上传。运行正式入口需要准备对应输入和协议；结果目录不可混用或覆盖。
- 本目录不改变生产算法默认值，不参与日常批量溯源入口。不要将本地案例参数变动或敏感原始监测资料与研究代码一起批量提交。
