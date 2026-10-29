# P36 Task8 IR监督修复独立复审（Independent Review of IR-Supervision Restoration）

日期：2026-10-29，Asia/Shanghai。用户批准恢复IR-only并重新训练Task8后，对修复代码和实际新产物进行独立复审。唯一指导为[P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为[P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)，重训交付为[P35](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P35_Task8_IR监督恢复与重训.md)。本文是复审记录，不另建plan或handoff。

## 结论（Verdict）

**新版 `ir_supervision_v2` PASS／GO，P34-M1 CLOSED；未发现新增P1/P2。** 适用修复代码281e770及 `A5/ir_supervision_v2` 的两模型和四区输出。旧A5保留其原实施定义和P34历史CONDITIONAL结论，不能将本次通过追认给旧模型。Task1–7可复用，Task9未启动。

三名独立审查者沿用 `/root/p34_engineering_review`、`/root/p34_methodology_review`、`/root/p34_artifact_review`，分别完成代码工程／版本兼容、方法修复和实际产物核验。工程在训练未齐时只给代码PASS，方法和产物在两阶段真实完成后分别直接核对并给最终PASS；没有把“等待产物”的条件状态提前算作通过。

## 工程与版本兼容（Engineering and Version Compatibility）

新训练强制独立revision，模型／resume、capability和ready分开保存；旧readiness、select/refit模型角色及旧路径路由仍能验证。新训练不能沿用旧resume或将旧预测登记为新产物；旧epoch恢复引擎没有改变，沿用P34独立连续24轮／第7轮中断恢复等价验证，不重复训练或GPU探针。

训练selected与class weights的 `y[valid]`共用一份mask和顺序；没有人口错位。独立工程读取真实批次证明，确认128样本含恢复的IR-only1条、源Dataset／损失／梯度／参数更新差0。工程范围未发现新P1/P2；验证日志[工程小证据](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/.superpowers/sdd/2026-09-23-teammate-single-teacher-fixed-split/P36-engineering-review/result.log)。P34-E1的可变raw输入缓存复用限制仍非阻断，本次没有原始输入漂移证据，也没有扩大为全目录校验。

## 方法修复（Methodology Resolution）

用户已明确选择恢复IR-only方案；规格第8节同步明确 `sk|imu|同phase A1.valid`，只排除全部无有效输入／监督行。独立方法审查从公共IDs、A1.valid和运动mask重新计算：

| 阶段（Phase） | 每轮训练行 | 恢复IR-only | 排除无证据行 |
|---|---:|---:|---:|
| select／train12 | 1957 | 1 | 82 |
| refit／refit14 | 2342 | 1 | 85 |

实际fit摘要、类别权重人口摘要、恢复行摘要均与独立重算一致。新mask只选择训练人口，Dataset及源loss调用没有增添CE、视觉KD或特征损失屏蔽，恢复行确实进入原监督项；重建、对比与RF监督仍按源mask。两阶段都完整24轮末态、随机重新初始化，首末LR4e-4／2e-5。由此实现和训练元数据共同关闭P34-M1，未新增方法差异。

## 正式产物（Formal Artifacts）

独立产物审查实际CPU加载两个新checkpoint各一次，核对width96、双40类头、24轮、同phase祖先及新revision。四区12份标准目标完整ID／40类列／文件摘要／逐行prior／combined available-mean诊断通过；S/I实际输入有效数仍为1956/1914、385/369、2341/2283、590/580，没有因为恢复训练监督而伪装IR-only具备运动输入。

允许的开发集独立复算：Skeleton143/388＝36.86%，IMU112/388＝28.87%，combined额外诊断151/388＝38.92%。fixed endpoint结果没有用于选择旧／新版或追加epoch。combined仍不进入主bank或16候选。

Task6–8原15项小证据绑定，以及旧A5 checkpoint／history／model和12份旧targets文件摘要均保持；Task1–7登记链可复用，本轮没有全扫其历史载荷。全局formal_generation_ready=false与P12／Thermal／P89／A18材料门槛不改，仅A5局部scope通过。

root自验小证据 `outputs/task5_audit/P35-task8-final-acceptance.json` 已记录64条公共final输入的真实模型推理，教师／标签载荷打开0、S/I logits对导出差0；独立审查只读该小证据，不重复GPU。没有读取final私有标签／类别分布／准确率，未重训、扫描整个motion／visual数值载荷或递归祖先payload；此结论不外推未抽payload全部完整性或后续新用户准确率。

## 当前入口（Current Entry）

新版完成入口为 `protocol/task8_ir_supervision_v2_complete.json`，绑定新版ready、两模型、四区输出、P35/P36及针对性验证；`protocol/task8_current.json`指向通过复审的revision。后续任务必须读取该当前入口，不能继续使用旧 `task8_artifacts_ready.json`默认为已修复。原Task8 complete／P34 review标记保留历史事实；本次是新产物的P34-M1闭合，不覆写旧身份。

源配方、固定用户划分、原A1／RF目标、Task1–7缓存和既有ROI质量精度残余保持原状。Task9及后续训练仍需用户授权和各自依赖验收，本次GO不等于完整全局链已生成。
