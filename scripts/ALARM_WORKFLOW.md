# 振华报警合并与批量溯源

先运行：

```powershell
python scripts/merge_station_alarm_events_zhenhua.py
```

脚本顶部手动设置输入、输出及允许的最大间隔。仅按污染物及时间分组，不按站点分组；相邻报警间隔不超过2小时就归为同一事件，即允许中间缺1小时。保留事件起止时间中点最近的实际报警；距离相同选较晚时刻，同一时刻多站报警则选浓度最高一条。没有报警的时刻不造新记录。

`MIN_CONCENTRATION = None`表示不过滤；例如改为`200.0`，则合并后代表记录浓度低于200的事件删除，等于200的保留。单位与报警表浓度一致。此过滤发生在事件合并之后，不先删原始小时记录，不改变事件起止与代表时刻。

结果在`data/zhenhua1112/merged_alarm_events_q1_by_pollutant.xlsx`。仅包含station（站点名）、pollutant（污染物）、time（污染时间）、event_start（开始时间）、event_end（结束时间）、concentration（浓度）六列。旧的按站点分组文件不再作为批量入口输入。

预览批量任务，不训练：

```powershell
python scripts/run_alarm_source_inversions_zhenhua.py --dry-run --count 10
```

实际执行：

```powershell
python scripts/run_alarm_source_inversions_zhenhua.py
```

在第二个脚本顶部设置合并表、对应季度历史监测表、站点坐标、运行数量、方向及污染物筛选。默认最近10个事件，按原事件起止各外扩6小时提取浓度和风。每个合并事件单独运行，代表记录的站点仅用于选择风站，不用于再次划分事件。保留现有模型配置，不自动选择IA或Adam。

注意：
- 当前历史表用ppb，报警记录用μg/m³。报警浓度仅作事件信息，拟合使用历史表值，不混用两套浓度。不提供未经条件确认的单位转换。
- 二甲苯显式按邻-二甲苯＋间-二甲苯及对-二甲苯计算；任一分量缺测时保留缺测，不补0。`POLLUTANT_COMPONENTS`可配置其他已确认的组分关系。
- VOCS-36缺少对应历史列时记录失败并继续，不能自行用非甲烷总烃或其他VOC总量替代。
- 坐标表支持station/lon/lat列及lon/lat两行格式，使用模型既有坐标解析器；缩写站名按脚本顶部明确映射。原坐标表不改写。
- 各站只有实际有效数据才参与反演；单站会警告，不能据此认为源位置已可靠确定。
- 输出沿用原`result/时间_污染物`，增加`alarm_run_summary_*.xlsx`。失败日志及已提取输入保存至该汇总表对应的`*_failed`目录。配置在退出时恢复；执行过程中不要同时手动编辑config或并行启动另一批任务。
- 原监测提取脚本和原批量入口未修改。本次仅验证合并、预览及真实输入提取，没有自动运行整批反演。
