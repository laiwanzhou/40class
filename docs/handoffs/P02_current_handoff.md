# P02 当前实验交接（Current Experiment Handoff）

更新：2026-10-29，Asia/Shanghai。工作树`D:/work/2026.7.14_kaggle/_single_visual_processing_replication`；分支`experiment/teammate-single-teacher-task1`，origin为`laiwanzhou/40class`。

本文件是本分支唯一交接入口（Canonical Handoff），后续原位更新。唯一指导为[P01计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)。历史文档在docs/archive/P04–P23，审计输入快照仅供历史证据，不作为新计划。

## 当前完成状态（Current Status）

v3.1 Task1–5已完成。Task1、Task2＋3、Task4和Task5均经实际独立复审GO。Task4完成全部六族select/refit头、两个先验、两phase固定Ridge机制头、四区68组targets/raw scores与两个完整全窗特征。Task5两阶段固定16轮与四区预测／原生序列已完成；没有训练进程或定时自动化，Task6未启动。

Task4代码已推送`202d57c`；最终完成文档的提交号见本任务ledger和Git最新日志。ledger仍位于`.superpowers/sdd/2026-09-23-teammate-single-teacher-fixed-split/progress.md`，保留旧目录名以连续记录；首行指导路径已更新P01。

当前run：`outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops`。协议SHA：`2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`。最新权威完成入口为`run/protocol/task5_complete.json`，绑定学生模型／四区产物与独立验收；`task4_complete.json`仍保留此前视觉教师头部和全窗特征的完成依据。旧`task4_heads_complete.json`的fullwindow_completed=false仅保留头部完成当时事实，不重写旧身份。

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

Task5已按P01完成源16轮MC3、新targets消费和同phase native sequence/anchor，实际正式产物已独立验收GO。下一任务为Task6运动缓存，当前未进入。以新增student_readiness单独绑定Task4 complete和本phase目标等公开依赖，不把旧v2学生权重当作v3已完成。P12 S+D最终配置、Thermal最终配置/公开资产/额外公共清单，以及P89 safe-base/A18完整producer证据仍未闭合，完整teacher bank正式生成及最终冻结仍受阻断；不得填默认值、删必要专家或直接改true。

## 数据与复用边界（Data and Reuse Boundaries）

固定train12=2039、val user6/7=388、refit14=2427、封存test user4/17/23/24=609。训练/校准/门控仅fit本phase；不做OOF，不读旧专家预测或用final分数调参。此前用户只授权独立子智能体返回final联合40类覆盖“是”，不允许再次读取标签。

Task2只复制四公共清单和前三分区允许标签，未读/复制final标签；P28/P29、公开冻结features、pixels和公开initializer按同卷硬链接新登记，新祖先全部v3，旧ref仅provenance。独立共享VerificationContext修复了for key覆写缓存键导致重复JSON解析的问题。Task3一条22帧14ROI字段/原crop/quality/time逐值一致；原timestamp恢复函数与两轮操作不改。详情[P03](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P03_Task2_准备与缓存复用.md)、[P24](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P24_Task3_姿态与ROI一致性.md)、[P25](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P25_Task2_Task3_独立复审.md)。

原始数据、队友快照和两份提交包只读。main历史文档等待完整实验结束再更新。旧v2 69.07%教师/48.71%学生保持历史；两个resume和四旧sequence数组已清理，不能把退休sequence作为新祖先。仍保留公共几何/特征/pixels/权重、正式checkpoint/targets/来源记录。两份原有未跟踪历史报告不顺手提交。

## 校验成本（Verification Cost Bound）

不重新SHA原始目录/全images/features/sequence，不递归读祖先payload，不每Task重跑旧全仓suite。新直接输入只查一次、祖先元数据去重、数组只检查header/ID/mask和消费切片。v3辅助头返回Prediction，主A1用显式bounded TeacherTargets，旧默认full检查保持；InitVar不纳入旧dataclass fields/协议身份。不要为“证明不全扫”实际先做一轮全扫描。


## Task5完成（Task5 Complete，2026-10-29）

用户要求Task5完成后独立复审，已全部完成：select与refit各16轮末态，refit公开重新初始化，不按val挑epoch。四区标准targets和原生sequence/anchor已齐，整体完成入口`run/protocol/task5_complete.json`绑定两模型、四区产物、ready摘要及最终独立报告。旧ready的review=pending保留生成时事实，不修改旧身份；以新的complete入口为准。Task6未开始。

学生train12=1975/2039（96.86%，有效IR1945/1957）；固定user6/7开发=182/388（46.91%，macro-F1 .3521767785）；refit14=2343/2427（96.54%，有效IR2313/2342），含val用户不能称val；final4=609行、591有效／18prior，不读取标签、不评分。旧v2学生最佳189/388=48.71%，当前低7条（-1.80个百分点），未观察到提升；同时改变了预算／选择规则／目标校准等，不能据此断言单一原因或追加调参。

实际训练select3661.98秒＋refit4368.81秒=133.85分钟；四区预测／sequence联合导出399.41秒=6.66分钟，总计约2小时20分钟，不含前置实现测试。PID37608/session18655已结束，不可重复启动旧任务。正式日志`outputs/task5_audit/P29-task5-training.log`以source_student_pipeline_complete结束。

源代码已提交推送0630ce7；完成文档提交见最新Git日志／ledger。53项针对性测试通过81.10秒；真实train12批次loss／参与梯度／anchor与源参考差0。完整恢复夹具证明16轮末态／只末轮一次val／refit重新初始化，中断恢复不重复完整epoch。单独student_readiness只放A2且绑定Task4 complete、源pins、pixels、MC3公开init和本phase校准early_late目标；共享描述符无学习祖先，实际模型直接ownphase教师父。

最终独立审查者实际核验两模型／16轮history／源配方、ownphase祖先、四区targets完整IDs／40类／prob／valid／prior、四份native `[N,2,3,16,512]` float16 header与只读mmap、全部small anchor和prediction逐值一致，以及16条sequence小切片；开发指标独立复算一致。真实正式refit模型禁读teacher／label／public initializer载荷推理通过（模拟不可用、未删除真实文件），打开尝试0，native和直接anchor差0。未读final标签、未扫描大sequence／pixel／feature或raw。整个Task5 GO，无未解决P1/P2；未外推未抽payload完整性、独立重训或后续Task6+。

详情：[P29交付](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P29_Task5_固定16轮视觉学生.md)、[P30最终独立GO](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P30_Task5_独立复审.md)。证据`outputs/task5_audit/P29-task5-final-tests.log`、`P29-task5-real-batch.log`、`P29-task5-final-acceptance.json/.log`。inference_only protocol保留同receipt/pins/identity，只不要求未消费public teacher/init文件存在；实际checkpoint/pixels/prior照常校验。

协议身份仍2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a，全局formal_generation_ready=false不变，P12/Thermal/P89/A18材料门槛仍约束后续teacher bank／融合。下一步只有收到授权才进入P01 Task6，不按本轮低分改配方。保持唯一P01计划／P02交接；原数据、快照及两提交包只读，main历史文档仍等完整实验分支结束。
