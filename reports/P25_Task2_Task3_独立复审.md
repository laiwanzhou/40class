# P25 Task2＋Task3 独立复审（Independent Review）

日期：2026-10-28。分支：`experiment/teammate-single-teacher-task1`。

结论：修复后的Task2＋Task3为 **GO**，无未解决的阻断问题。GO仅覆盖准备、公共缓存导入与几何源一致性，不代表Task4–14或正式训练已放行。

复审者为独立子智能体（Independent Subagent）`p25_task23_geometry_review`，以独立上下文只读审查。范围是Task2提交`fa9fe7f..b699fb7`及Task3工作区变更；依据唯一[P01计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)的Task2/3和有界校验约束，以及当前规格。审查者没有修改工作树或派生其他审查者。

## 发现与关闭（Finding and Resolution）

**P2，已修复：共享元数据缓存键被循环变量覆盖。**

`src/experiments/artifact_record.py`的`read()`先构造`(protocol, ref, path)`元组缓存键，随后`for key in (...)`把它覆盖为`adaptation_users`。SHA可缓存，但已解析的记录（Parsed Records）缓存永远无法命中，祖先JSON重复读取，违反Task2的读取成本约束；不改变实验数值。

独立探针对同一正式manifest/同一VerificationContext连续调用3次：修复前JSON读取3次、返回不同对象、缓存键为字符串；修复后JSON读取1次、返回同一对象、缓存键为元组。父智能体最小修改循环变量为`field`，不改协议或旧记录身份。

新增回归`test_shared_context_parses_ancestor_record_json_once`直接统计`Path.read_text`，先观察到5轮read/verify重复读取10次，再修复转绿。此前只统计SHA调用的测试未覆盖JSON解析次数，本轮已补足。修复后51项针对性集成测试通过，62.91秒；独立复审再次用同一小探针确认修复。

## 实际核对（Evidence）

- 六个真实导入根（P29/features/pixels，各refit14与final4）以及12个唯一祖先记录均属于新协议（Protocol）和新run，payload路径均位于新run，没有fit/select/adaptation人口或学习祖先（Learned Ancestors）。仅核对元数据，没有全量读取缓存。
- P28/P29精确父关系、共享refit14公共子集、ID/用户/阶段（Phase）检查与实现一致。Task3保留原时间戳恢复逻辑。
- 两轮姿态、ROI质量、缺失回退以及人物/工作区裁剪有针对性夹具覆盖，未发现方法偏差。
- 正式小证明记录22帧、14个ROI返回字段精确一致；重建pixels的裁剪、质量、掩码、帧和时间轴一致；5个直接源文件与最终提交包字节相同。真实CLI耗时8.614秒，没有detector、全量几何重生成或raw SHA。
- 复审者读取了15项初步测试通过日志及51项修复后集成通过日志，并独立运行小型只读探针。没有读取final4标签、旧专家预测或扫描原始数据。

## 未裁决及后续门槛（Declined to Judge and Remaining Gates）

未消费缓存的全量内容完整性、全量检测器输出复算、Task4–14、训练数值及最终准确率不在本轮GO范围。这里采用有界验证（Bounded Verification），继续依赖历史不可变缓存及来源凭据，不把单样本验收外推为全量重验证。

P12原配置/公开资产、Thermal额外公共清单及P89/A18生产证据仍未闭合，`formal_generation_ready=false`继续阻止正式生成。没有通过修改身份或默认值消除门槛。本轮停止于Task3，当前唯一交接为[P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。
