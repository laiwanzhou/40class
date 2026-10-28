# P28 Task4 产物独立验收（Independent Artifact Acceptance）

日期：2026-10-28。审查者为独立子智能体`p25_task23_geometry_review`。本报告持续记录同一Task4产物验收，不是新交接或计划。

## 头部产物：GO（Visual Heads: GO）

独立审查者实际核验68组targets/raw scores及102个唯一元数据记录，无未解决的P1/P2。未重跑72格拟合，特征大表重读次数0。

- 四区样本ID、40类列、阶段（Phase）、valid掩码和缺IR先验一致。
- A1与固定early_late输出逐值一致；有效行满足raw/T_head；十项机制保持T1、logits与raw一致；祖先无A2或适配模型。
- 72格选参表每族12格，六族refit均继承对应select胜出超参。
- 只用允许的fit标签和保存的raw scores重新调用源温度函数，select/refit温度分别为0.1206300185060063和0.1397821300164829，与保存值一致。
- 成绩复算为train12=1985/2039、development2=268/388、refit14=2355/2427；final4不读取标签，correct=null。
- 独立复验重复全特征扫描P2修复；最终受影响针对性测试49项通过。阶段就绪只放行Task4视觉范围，旧协议身份及全局false未改。

此前实施前审查的最后汇总曾遇额度限制，随后恢复并完成本次真实头部产物核对；没有将额度中断当作通过。实施前范围与修复详情见[P27](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P27_Task4_实施前独立复审.md)。

## 完整全窗产物：GO（Full-window: GO）

完整refit14/final4编码均正常退出并登记，独立审查者完成最后有限验收，结论为 **整个Task4 GO**，无未解决的P1/P2；既有头部GO未重复全查。

| 分区（Partition） | 样本数／有效数 | 特征／Kinetics logits |
|---|---:|---|
| refit14 | 2427／2342 | float16 `[2427,3,1024]`／`[2427,3,400]` |
| final4 | 609／591 | float16 `[609,3,1024]`／`[609,3,400]` |

完整ID、固定类别列、valid与公共IR availability精确一致。仅完整读取sample_ids/class_ids/valid小成员；每区从聚合features/logits读取首行合计8544个载荷字节，与相应trial逐值一致，未加载完整特征大表。另抽查refit位置0/19/1213/2426、final位置0/3/304/608，共8个trial，有限性、ID、uniform16索引、时间戳与缺失零填充均通过。

真实train12验收样本与refit正式对应trial的features/logits逐值一致。两区共享同一公开初始化父ArtifactRef；48个attention偏置最大验证差0的证明位于refit14，按同一进程模型复用关系共同使用。final4目录没有重复写初始化证明不构成缺陷。源producer摘要、full0–1窗口、1.15/1.40裁剪和父关系均一致。

未执行大payload SHA、GPU重跑或读取final4标签；协议身份和全局`formal_generation_ready=false`不变。这里只放行Task4已实现范围，不放行完整教师bank或后续融合。

未裁决（Declined to Judge）：未抽查payload的全量完整性／有限性、Task5及后续阶段、最终准确率。后续新任务仍需满足自身依赖与验收，不能把Task4 GO外推为整条实验已完成。
