# v3.1 源操作一致性独立再审（Independent Source-Parity Reaudit）

结论：**GO，仅限 Task1–3 实施（Implementation）**。复核 HEAD `8d1f3943b05b76e81d71e5163b5ec82537219301` 之上的当前未提交 v3.1 正文修订后，原独立复审的 **P1 已在文档关闭**；本次没有发现新增、有直接源证据的 P1/P2。原 RF、bank fallback／route、全窗 KNN 三项空缺已转为具体操作要求及失败门槛。此结论不表示未来生产器（Producer）全部闭合，不放行正式训练／生成／冻结，也不表示 v3 数值或梯度测试已通过。

此次仅读取修订正文、文档 diff 和必要的源代码文本；没有训练、运行 suite、读取私有 final 标签、历史预测、退休 sequence 缓存或计算 final 分数。元数据（Metadata）、导入（Import）及共享校验上下文（Verification Context）的工程正确性由另一独立审查覆盖，本报告不替代其结论。

## 原问题关闭核对（Closure Check）

| 原问题 | 修订正文位置 | 独立判断与源依据 |
|---|---|---|
| P1：mechanism误归MC3 | [计划107行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md:107)、[119行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md:119)；[规格51行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md:51)、95行 | 已关闭。Task4独立固定Ridge头使用VideoMAE六clip逐clip L2、alpha3000、power0.75、原scaler/Ridge、fit归一化特征均值、十项原干预、aligned_scores_40，输出raw logits/T1、不套A1温度、无A2祖先；Task5明确不生产这些专家。符合源 `audit_p86_teacher_mechanisms.py:25–26,101–121,174,208–225` 与目标侧 `p89_build_multiexpert_test_logits.py:162–173`。 |
| RF缺类值／阶段RNG（Missing-Class Fill / Phase RNG） | [计划139行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md:139)、140–141行；规格71–73行 | 已明确。fit_teacher_target为-1e6，terminal_export为log(1e-12)，实际classes均log clip；两phase直接terminal无fold RNG seed20260723，不捏造fold或偏移。分别对应 `run_imu_stat_baseline.py:178,193–215` 的consumer填充值和 `train_final_imu_rf.py:83–106,117–120` 的无fold拟合与目标导出。固定划分的RNG映射被显式声明，属于获准的无OOF适配；A4导出不校准和Task11 RF T3消费不再混淆。 |
| 全窗KNN隐藏全局融合依赖（Global-Mixture Dependency） | [计划106行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md:106)、[181行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md:181)、182–183行；规格97行 | 已明确。Task4只提取标签无关全窗特征；Task11重建P12八列按源顺序，加六视觉头，log_softmax→fit_simplex(softmax权重，L-BFGS-B,maxiter500,ftol1e-12)→global scores，再按各无标签分区3邻居/self排除/weight0.40平滑；不替换为A1/A7。符合 `build_p85_multiexpert_submission_v1.py:29–47,83–101` 与 `build_p85_fullwindow_knn_teacher_v3.py:23,26–27,50–74,102–120`。P12来源、S+D0.6/0.4、thermal校准／21点权重网格、router C0.05/maxiter2000/threshold0.5均给出源函数约束；必要producer未锁定时拒绝正式生成。 |
| safe-base、缺专家回退及路由（Fallback / Route） | [计划185行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md:185)、186行；规格103行 | 已明确。safe=normalise(0.95*source_base_probability+0.05*softmax(RF_terminal_logits/3))；非detail／available=false专家行回退bank第0列并保留不可用状态；detail_ids精确成员资格决定visual_available；P255/P309 group route带visual_available，P270 sequence route没有额外AND，P310仅new!=old。符合 `p165_deployable_group_teacher.py:229–247`、`p173_vjepa_augmented_group_teacher.py:83–87`、`p255_repeat_augmented_physical_group.py:18`、`p309_union_repeat_group_test.py:22–23`、`p270_fixed_emission065_transition045.py:17` 与 `p310_union_repeat_precedence_teacher.py:56–68`。正文已区分合法逐行fallback和复制假专家，明确source_base不能任取A7。 |

当前规格与计划中原“同学生mechanism／同一MC3十项”要求已消失，没有保留相反的Task5生产约束。学习KNN移到Task11后，不再反向阻塞Task4/5；固定头不新增大视觉根节点。原主预算和最终适配配方没有在本次修订中被改写。

## 放行范围与实施门槛（Scope and Implementation Gates）

可以进入 Task1–3，将修订要求落为 v3 专属协议、源操作清单（Source Ops Manifest）、教师依赖清单（Teacher Roster）、公开资产绑定及精确缓存复用。该阶段必须继续输出具体 `source path/symbol/line`、actual override/default 和保留／排除原因；[计划70–72行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md:70) 与181行已给出这项门槛。

P12独立小模型的全部训练配置／初始化、P89 source_base 的完整实际producer图、被删大根节点的派生专家排除、各分区detail_ids及真实availability，是 Task1及后续producer应提交的证据。正文要求已经明确，**尚未生成这些证据不等于它们已闭合**；不得把此次GO扩展为整个Task4–12或完整bank已验证。缺任一必要独立producer、合法源输入或对应资产，按现文档停止正式生成，不用A5/A7、先验或重复列补齐。

后续按已有针对性fixture逐算子验收：mechanism train/test raw logits与fit均值、RF缺类两consumer与固定设备删除、global-mixture/KNN列序及数值、正常／缺IR有motion／单专家缺失的fallback和route、old/new/sequence precedence与锐化。正式A8仍受[计划191行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md:191) 的全部父记录和数值闭合门槛约束。

固定分区、fit-only校准／阈值、同分区无标签联合邻居、已声明全缺失prior策略沿用既定授权；本报告没有要求恢复内部OOF或读取final私有标签。
