# P02 当前实验交接（Current Experiment Handoff）

更新：2026-10-28，Asia/Shanghai。工作树`D:/work/2026.7.14_kaggle/_single_visual_processing_replication`；分支`experiment/teammate-single-teacher-task1`，origin为`laiwanzhou/40class`。

本文件是本分支唯一交接入口（Canonical Handoff），后续原位更新。唯一指导为[P01计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)。历史文档在docs/archive/P04–P23，审计输入快照仅供历史证据，不作为新计划。

## 当前完成状态（Current Status）

v3.1 Task1–4已完成。Task1、Task2＋3及Task4均经实际独立复审GO。Task4完成全部六族select/refit头、两个先验、两phase固定Ridge机制头、四区68组targets/raw scores与两个完整全窗特征。Task5已获授权并进入正式固定16轮select/refit，尚未完成；无定时自动化。当前训练进程见下方Task5状态，不再按Task4旧PID续跑。

Task4代码已推送`202d57c`；最终完成文档的提交号见本任务ledger和Git最新日志。ledger仍位于`.superpowers/sdd/2026-09-23-teammate-single-teacher-fixed-split/progress.md`，保留旧目录名以连续记录；首行指导路径已更新P01。

当前run：`outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops`。协议SHA：`2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`。权威整体完成入口为`run/protocol/task4_complete.json`，绑定头部摘要和两个完整fullwindow ArtifactRef。旧`task4_heads_complete.json`的fullwindow_completed=false仅保留头部完成当时事实，不重写旧身份。

## Task4结果与验收（Results and Acceptance）

主头固定early_late，选定alpha300／class_weight_power0.75；select/refit温度分别为0.1206300185060063／0.1397821300164829，均仅由本phase fit人口估计。train12为1985/2039，development2为268/388（69.07%），refit14为2355/2427。refit包含val，不能称val成绩；final4不评分、correct=null。

六族每族12格，共72格按源tie-break选择；主头不在六族间再选择。十机制源固定Ridge(alpha3000/power0.75)及同phase L2均值，保持raw T1，未使用MC3或旧序列。训练目标明确in_sample/non-OOF。

全窗为真正0–1 uniform16、scene/person/workspace原裁剪与公开冻结VideoMAE编码，不是早晚特征均值。refit14特征float16[2427,3,1024]、logits[2427,3,400]、valid2342；final4为[609,3,1024]/[609,3,400]、valid591。两个根在`run/fullwindow_features/{refit14,final4}/full/artifact.json`。真实编码2034.8＋319.0秒，合计39.2分钟；初始化48偏置张量maxdiff0，模型在两区共享。

最终49项受影响测试通过；头部独立核验68组targets/raw与102记录，ID/phase/先验/raw÷T/配置继承/无A2祖先均通过。全窗独立核对完整ID/数组头/valid、每区首行和8个trial小切片、验收样本对正式样本逐值一致、共享公开初始化父及源参数。大feature全表重读0、大cache SHA0、GPU重跑0、final标签读取0；不外推未抽查payload全部完整或有限。

交付与证据：

- [P26 Task4交付](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P26_Task4_视觉教师目标恢复.md)
- [P27实施前复审](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P27_Task4_实施前独立复审.md)
- [P28最终产物独立GO](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P28_Task4_产物独立验收.md)
- `outputs/task5_audit/P26-task4-final-tests.log`、`P26-task4-heads.log`、`P26-task4-fullwindow.log`
- `outputs/task5_audit/P26-task4-fullwindow-acceptance.json`：自验header/ID/mask/首行/小trial，验收对正式特征差0。

审查额度曾在实施前汇总时中断，随后恢复并真实完成两轮产物验收；从未把额度受限当作通过。

## 门槛与下一任务（Readiness and Next Task）

旧全局门槛把P12/P89缺项阻断全部学习，与规格Task4/5不等待P12冲突。本轮独立认可的修订是`visual_readiness`描述符绑定source_ops、Task3证明与5个消费源pin，只放行A1/六头/class_prior/固定机制；没有改变YAML、旧协议身份或全局`formal_generation_ready=false`，不放行P12/专家bank/融合。

当前任务是P01 Task5：源16轮MC3、新targets消费和同phase native sequence/anchor；用户已授权，独立代码审查及真实批次验收通过后正式启动。以新增student_readiness单独绑定Task4 complete和本phase目标等公开依赖，不把旧v2学生权重当作v3已完成。P12 S+D最终配置、Thermal最终配置/公开资产/额外公共清单，以及P89 safe-base/A18完整producer证据仍未闭合，完整teacher bank正式生成及最终冻结仍受阻断；不得填默认值、删必要专家或直接改true。

## 数据与复用边界（Data and Reuse Boundaries）

固定train12=2039、val user6/7=388、refit14=2427、封存test user4/17/23/24=609。训练/校准/门控仅fit本phase；不做OOF，不读旧专家预测或用final分数调参。此前用户只授权独立子智能体返回final联合40类覆盖“是”，不允许再次读取标签。

Task2只复制四公共清单和前三分区允许标签，未读/复制final标签；P28/P29、公开冻结features、pixels和公开initializer按同卷硬链接新登记，新祖先全部v3，旧ref仅provenance。独立共享VerificationContext修复了for key覆写缓存键导致重复JSON解析的问题。Task3一条22帧14ROI字段/原crop/quality/time逐值一致；原timestamp恢复函数与两轮操作不改。详情[P03](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P03_Task2_准备与缓存复用.md)、[P24](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P24_Task3_姿态与ROI一致性.md)、[P25](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P25_Task2_Task3_独立复审.md)。

原始数据、队友快照和两份提交包只读。main历史文档等待完整实验结束再更新。旧v2 69.07%教师/48.71%学生保持历史；两个resume和四旧sequence数组已清理，不能把退休sequence作为新祖先。仍保留公共几何/特征/pixels/权重、正式checkpoint/targets/来源记录。两份原有未跟踪历史报告不顺手提交。

## 校验成本（Verification Cost Bound）

不重新SHA原始目录/全images/features/sequence，不递归读祖先payload，不每Task重跑旧全仓suite。新直接输入只查一次、祖先元数据去重、数组只检查header/ID/mask和消费切片。v3辅助头返回Prediction，主A1用显式bounded TeacherTargets，旧默认full检查保持；InitVar不纳入旧dataclass fields/协议身份。不要为“证明不全扫”实际先做一轮全扫描。


## Task5进行中（Task5 In Progress）

用户要求推进Task5并在完成后独立复审；最近一次追问进度时尚未启动正式32轮，随后fixed-flow夹具修复通过24.73秒，已启动PID37608/session18655。正式日志outputs/task5_audit/P29-task5-training.log，命令scripts/run_no_vote_visual_student.py --phase source-pipeline --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml --device cuda。接续先检查进程／日志，禁止重复启动。预计select+refit约2小时，加四区预测／sequence导出；每epoch完整resume可恢复。

新visual_student_v3.py直接源class_weights/relation_loss/optimizer/schedule/augmentation，真实四train12 GPU批次12.03秒：loss16.706321716与源完全相同、梯度max0、native[4,2,3,16,512]及anchor差0，尾组optimizer1步。41受影响tests GREEN42.59秒，额外v3fixed16完整恢复夹具GREEN24.73秒；最新扩展包含推理禁读teacher/label的整链测试P29-task5-final-tests.log仍待读结果。

单独student_readiness只放行A2、固定epochs16，公共描述符不混phase学习祖先，实际A2直接ownphase v3 early_late教师目标。不会把global false改true或解锁P12/bank。inference_only protocol保留receipt/pins/身份，允许未消费teacher public assets移除；训练默认仍要求文件。旧v2默认分支不改选择行为。

详情[P29交付记录](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P29_Task5_固定16轮视觉学生.md)与[P30独立复审](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P30_Task5_独立复审.md)。P30当前仅run前GO；正式模型／四区targets/native sequence齐后用户要求的实际独立产物复审仍必须完成，不提前标完成、不进入Task6。
