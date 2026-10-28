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

## 完整全窗产物：待完成（Full-window: Pending）

真实单样本GPU验收已通过；完整refit14/final4全窗编码正在运行。当前头部GO不涵盖尚未生成的全窗最终产物；完成后仍需独立核对ID、形状、掩码与小切片，不重扫大数组。

未裁决（Declined to Judge）：后续学生训练、P12/教师库、未消费缓存完整性和最终准确率。不能据本次局部GO进入Task5。
