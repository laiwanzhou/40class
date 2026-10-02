# Task4审计与执行交接（Audit and Execution Handoff）

客户端日期：2026-10-02，Asia/Shanghai。工作树`D:/work/2026.7.14_kaggle/_single_visual_processing_replication`，分支`experiment/teammate-single-teacher-task1`。当前用户要求接续保留的独立子智能体历史复审；原先授权审计GO后推进Task4仍有效，不进入Task5。

## 权威文件（Authoritative Documents）

- 完整计划：[2026-09-23计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md)。
- 规格：[固定划分规格](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md)。
- [审计报告](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-02-task23-task4-independent-audit.md)。
- [Task4接口及命令](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/task4_visual_teacher.md)。

## 审计已完成（Audit Completed）

复用audit_20261002_engineering、audit_20261002_methodology、v2_review_methodology三名独立审计者。工程末轮曾因额度中断，用户要求接续后同一审计者恢复并最终GO；其独立teacher/source19测试通过。方法审计对P46/P85纯函数及96帧crop逐元素一致；数据审计认可raw refit14公开冻结features按ID供train/dev共享。三方无剩余Important代码阻塞。

修复：numeric IR与timestamp Skeleton匹配、相同numeric/timestamp别名合并、真实acquisition_ids与无IR空schema；inspect.unwrap来源校验保留inference_mode；抽取/续跑/head/predict重新核验P29 raw文件集合；prior显式artifact引用。全部问题有RED→GREEN回归。22条真实numeric IR中21条骨骼可用，全部完整对齐且保存真实采集标识，另1条保持缺失。

非阻塞P2：Depth–Skeleton语义数组复制IR侧，当前P29/Task4及上游P31/P86不消费。Task6前核对消费者及必要遮罩；不要随意改当前ROI producer源码导致整批记录SHA失效。

## 实现与验证（Implementation and Verification）

Task4模块visual_teacher.py、CLI run_no_vote_visual_teacher.py及10个测试已实现。A1固定head T1，六clip公开骨干冻结；仅一个Ridge选定头，无历史OOF温度或教师投票。select只fit valid train12，dev全388分母选择；refit重建模型fit valid refit14。缺IR保留相应人口prior；final4标签只在Task13冻结全部候选后揭示。

最新针对性72passed；全仓513passed+同名7个既有旧缓存失败，无新增失败。最新代码真实单样本GPU编码完成，峰值1.449GiB：

- features：`visual_features/train12/acceptance`，artifact SHA `13c183310accff711dc1f18a6f7e60038a398c3c02d799265bc7a5f110c601a8`。
- 原producer验收仅在`acceptance_before_engineering_fixes_20261002`归档；不能用于正式拟合。
- 相关日志`outputs/task4_audit/targeted-current.log`、`full-suite-current.log`、`features-current-acceptance.log`、`source-math-parity.log`。

## 全量运行状态（Full Run Status）

run根：`outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2`；协议身份仍为`2640dfa234af5f18b6d438f6834223d6ceeb8b445afb35c1d91de5f3c7bddd5c`。

完整P28/P29已生成，refit14各2427条，final4各609条，summary.complete=true/partial=false。

- refit14 P29 record SHA：`baf910f1d8c4e60e347356ae8bf41882d5500bcbaeecf8cd376347743f429b4f`。
- final4 P29 record SHA：`d79aa1017144530782efec6265fa6ff8c07385cc8957c8e4b866cf3bbe4e3a03`。
- 完整registry/schema逐项验证当前运行中，日志`outputs/task4_audit/full-roi-verify.log`，结果应写`full-roi-verified.json`；原exec session65308，Python PID36288，读取原始数据哈希耗时，进度无逐条输出。先核查进程，不重复启动验证。
- refit14正式feature抽取已启动，日志`outputs/task4_audit/features-refit14.log`，原exec session90679，Python PID27312。入口先递归验证完整父产物和raw inventory，可能较久无输出；真正编码后每20条打印进度，逐条cache可恢复。
- final4正式features、72grid选择、refit与四分区targets尚未运行。不能声明正式Task4训练完成或准确率。

接续时先读日志/进程和缓存计数；若进程还在，继续等待，不重复抽取。若中断，仅按相同已冻结producer/clip_batch/device命令续跑；校验漂移必须诊断，不能覆盖identity强行复用。完整features验收后按Task4说明执行select/refit/predict，final标签一直封存。不会自动进入Task5。

原始提交包、队友快照、原始数据均未修改。历史7个全仓错误依赖已清理旧路线缓存，不恢复废弃数据来掩盖它们。
