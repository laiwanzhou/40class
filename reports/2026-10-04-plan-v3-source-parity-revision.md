# v3计划的源码一致性修订（Source-Parity Plan Revision）

日期：2026-10-04，Asia/Shanghai。用户目标：继续研究一个大视觉教师在队友处理链中的作用；保留当前固定划分，恢复其他生效操作，同时避免重复大面积校验。本轮修改规格、计划和交接，不修改生产代码、旧配置、原始数据、提交包或模型，不启动训练。

## 对照依据（Reference Evidence）

以用户指定的`CUHK-X_Small_Model_Submission(1)`为准：最终checkpoint的deployment_model_config、docs/TRAINING_PIPELINE.md实际40轮命令、生效train_p86_visual_pixel_oof、train_p86_mobind_pretrain、train_p86_mobind_fusion_proxy、P255/P270/P307/P309/P310与adapt_p87s_test_student路径。关键六个视觉源文件与教师分支快照逐字节相同，最终视觉结构与当前A2参数键/形状匹配。

用户已补充：队友纯视觉超过0.8来自多个大视觉教师共同作用。本轮不要求单教师达到同分数，不把最终0.91542当纯视觉基线；此前40轮适配的visual head LR引用错了较早默认值，本轮按最终包实际命令修正。

## 差异与处理（Differences and Resolution）

| 操作（Operation） | 原v2 | 最终源码事实及v3处理 | Task |
|---|---|---|---|
| MC3结构/预训练/16帧160/六clip | 相同 | 保留原模型类与公开初始化；不加未启用候选模块 | 5 |
| 几何、裁剪与subject_robust | 相同 | 复用已验收raw缓存；保留时间戳修复 | 2/3/5 |
| 主教师feature family | val在六族间选window_mean | 原Dataset固定early_late；主头恢复early_late，同大骨干六族头保留给后续源算子 | 4 |
| 分类头选择决胜 | accuracy/F1/worst-user/dimension/大alpha | 恢复原accuracy/balanced accuracy/F1/小power/小alpha；固定val替代原OOF评价 | 4 |
| 教师head温度 | 固定1 | 原fit_temperature NLL/log-T约0.1–10；恢复函数，仅fit阶段训练预测/标签 | 4 |
| 教师训练软目标 | in-sample、非OOF | 用户明确继续固定划分，因此仍in-sample；作为允许差异记录，不引入内部OOF，不复用旧OOF | 4/7 |
| MC3预算 | 18轮val选第17轮 | 原终端固定16；主链恢复select/refit16，val末轮诊断 | 5 |
| 生效视觉loss/增强/优化器 | 基本一致 | 保留CE+KD+relation和直接feature/stage=0，源class_weights与调度逐函数验收 | 5 |
| motion跨样本mean/std | 新增全局fit z-score | 原Dataset直接读取缓存；删除二次z-score，保留源物理转换与模型LayerNorm，外部转换恒等 | 6 |
| RF seed | 20260811 | 源run_imu_stat_baseline默认20260723；由源操作清单记录来源，恢复源RF/default | 7 |
| RF特征/dropout/结构 | 基本一致 | 原240+10维、400树及存在设备dropout；fixed-split训练目标显式非OOF，不另加RF温度 | 7 |
| MoBind预算 | 1–24轮val选择 | 原终端固定24；select/refit均24 | 8 |
| 部分缺模态loss | 自行将分支CE/KD置零 | 原CE、visual KD等不这样屏蔽；恢复原losses各项mask及batch归约 | 8/10 |
| A6温度/平均 | 纳入阶段比较 | 属额外诊断，不是最终主链，不用于改变监督目标、预算或后处理输入 | 9 |
| A7主fusion/model/corruption/budget | 大部一致但额外loss masks | 保留separate/additive、4+20、原corruption；full恢复源loss/mode，control额外mask不回流full | 10 |
| 冻结motion train/eval模式 | 易把冻结解释为全部eval | 按原train_stage保留Dropout模式，只执行源冻结参数行为 | 10 |
| 教师bank处理 | 全部删除bank与派生专家 | 仅删多大视觉根节点及必然依赖；保留同骨干派生头、同MC3机制输出及独立非大/非视觉专家，逐源producer重建 | 1/4/5/11 |
| 会话/重复 | 自拟two-peer+cosine+margin+混合网格 | 恢复p134 CONFIG、重复会话对齐、p137分组特征/逻辑回归与源threshold | 11 |
| sequence发射/转移/gate | alpha0.25、transition0.25–0.35、posterior0.80门槛 | P270为emission0.65/base0.35、transition0.45、alpha1/trigram1/beam50；恢复p139五项gate及源网格 | 11 |
| 分组优先覆盖 | 未复刻P310 old/new group route | 原新group与旧group不同时覆盖sequence，否则保留sequence；恢复并记录route | 11 |
| 无标签pseudo目标 | 普通A8概率直接使用 | 原P310 argmax0.9805，其余0.0005；恢复锐化/confidence0.9805，18缺失prior不进loss | 11/12 |
| 最终40轮visual LR | 1e-5 | 最终(1)命令为5e-5；fusion/motion1e-4/min5e-6/heads_motion_encoder/seed20260826 | 12 |
| 12轮scope/主端点 | 两端点heads_motion_encoder，主12 | 原基础12默认heads；40是最终主配方，12只作独立额外诊断 | 12/13 |
| 重复校验 | 已修复direct-metadata | 保留：raw SHA0、祖先payload0、大数组写时digest/header与小切片、阶段验证上下文、针对性回归 | 全部 |

这张表恢复的是已核对的生效算法与参数。训练软目标来源、原OOF选参人口和源cohort门控拟合人口必须随固定划分改变，明确记录，不能称为原OOF数值等价。源逻辑回归分组输入的专家数改变也会改变输出；保留其算法，不用伪专家维持30列。

## 教师清单闭合（Teacher-roster Closure）

原30专家不是30个互相独立的大视觉骨干：包含同VideoMAE六族头、同MC3的输入机制预测、其他视觉骨干的token头、骨骼/热/物理等派生专家。因此仅保留一个大视觉骨干不能被实现成“所有其他源操作都删掉”。

v3 Task1先按p88 CANDIDATE_SOURCES→p165→p173→P255/P307/P309展开source_roster，逐个记录根节点、算法、公开资产、依赖、保留/排除原因。Task11重新构建保留源专家，禁止拿新MoBind分支任意替换原专家；缺对应producer或模态即停止，不能以简单平均/先验/重复列伪装完成。这会增加实际训练工作，预检重新估计，不能沿用先前单A1/MC3/RF的预算。

## 固定划分与独立覆盖检查（Fixed Split and Coverage Check）

维持12名train、user6/user7 val、14名refit、user4/user17/user23/user24 test。已允许标签统计显示train12/val/refit联合均40类，val的两名用户单独分别34/33类。

用户明确授权一次独立子智能体覆盖核实。子智能体`/root/final_class_coverage_only`只在自身内存中核对final4人口与类别集合，返回：**对应final4联合覆盖全部40类=是**。未向主智能体返回样本标签、频率、缺类或分用户明细，没有模型预测/accuracy/F1，没有修改划分。主智能体没有打开该私有标签文件。该检查属于查看test覆盖元信息，必须如实记录；不能宣称所有test信息从未被查看，也不能据此更换用户或调参数。其他final标签封存规则不变。

## 实施状态与资源（Implementation State and Resources）

v2 Task4/5已完成，69.07%/48.71%保留历史；不是v3失败或v3已完成。旧plan/spec按原bytes归档，旧运行配置与监督产物不变。v3命名fixed-split-single-teacher-v3-source-ops，尚无正式配置/代码实现/模型。

编号仍14个Task。先实现Task1–3的新协议/源操作清单/精确raw-cache导入，再重做Task4教师目标和Task5源16轮学生；不得拿v2未经校准目标直接进入Task6。冻结公开features、pixels与几何可复用；Ridge/校准/targets、学生和模型sequence重新生成。后续Task6–14按修订计划，正式最终评估仍在所有候选冻结之后。

本轮自审检查规格覆盖、源参数来源、接口一致、16候选名单、禁止OOF条款、不可变v2和轻量校验约束。仅文档修改，不运行训练或新自动化，不更新main项目历史文档；计划通过不等于数值对照测试或正式模型验收通过。

完整规格：[v3规格](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md)。完整计划：[v3实施计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md)。
