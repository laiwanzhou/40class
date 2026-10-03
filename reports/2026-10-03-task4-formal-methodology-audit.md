# Task4 正式产物方法学独立审计（Formal Methodology Audit）

日期：2026-10-03，Asia/Shanghai。审计者：独立子智能体 `/root/audit_20261002_methodology`；未编写本轮产物或校验性能修订。审计对象为正式生成的实际产物，不沿用此前代码审查的 GO。

## 结论与范围（Verdict and Scope）

**GO：Task4 正式 A1 产物与开发选择结果通过本次有界方法学审计（bounded methodology audit），没有发现模型或预测需要重生成的 P1/P2 问题。** Task5 的整体启动仍须等待主审正在进行的校验性能修订及其独立工程审查；本报告不提前批准尚未审查的性能代码。

先阅读 ae3e154 的[校验策略修订](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md:25)。本次只读取代码、公共清单、22个祖先记录的元数据（metadata）、两个小型 head.joblib、四份 targets、先验 JSON 和 development2 标签；核对记录引用及直接小型模型/targets 的内容摘要（digest）。没有调用 ArtifactRegistry.verify，没有打开公开权重、原始图像/骨骼/IMU或全量 features 数组，没有遍历原始数据，没有执行全量缓存 SHA，没有读取 final4 私有标签，没有启动训练。

本轮只写本报告。验证状态是上述有限范围已验证（verified within stated scope），不是全量原始内容复验，也不是最终揭盲评估。

## 正式完成与版本证据（Completion and Producer Evidence）

- [continuation.log](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/task4_audit/continuation.log:14)记录北京时间 `2026-10-03T19:20:46.2045984+08:00 DONE predict-final4`；第15行包含 `TASK4_CLI_STAGES_COMPLETE`。
- 四分区均存在唯一的 A1/predictions/predict 正式记录，complete=true、fixture=false；不是仅根据日志认定完成。
- 所查产物协议身份统一为 `2640dfa234af5f18b6d438f6834223d6ceeb8b445afb35c1d91de5f3c7bddd5c`。
- select/refit 模型的生产者（producer）源码摘要均为 `580d2aea6bbc2a291355b5dddcf21cd3b8380393de616ec81ae0b82adc458781`；从 `git show ae3e154:src/experiments/visual_teacher.py` 取得历史源码并重算一致。性能修订后应继续保留此历史身份，不把旧产物改登记为新实现。

## 72候选与开发结果（Candidate Selection and Development Results）

[grid.json](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/A1/select/grid.json)恰有72个唯一 candidate_id；候选组合精确覆盖六特征族 × 三种类别权重指数 × 四个 alpha。独立按 accuracy、fixed-40 macro-F1、最差用户 accuracy、较小维度、较大 alpha、candidate_id 重算决胜顺序，得到与[selection.json](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/A1/select/selection.json)相同的唯一配方：

| 项目（Item） | 正式值（Value） |
|---|---:|
| feature family | window_mean |
| class_weight_power | 0.5 |
| Ridge alpha | 1000 |
| 特征维度（feature dimension） | 3072 |
| A1 head temperature | 1 |
| valid train12 fit rows | 1957 |

用正式 development2 targets 与独立开发标签按 opaque ID 重新连接计分，覆盖全部388条，包括3条缺IR先验回退；结果与 grid winner 和 selection.metric 一致：

| 开发人口（Development Population） | 正确数/全分母 | accuracy |
|---|---:|---:|
| user6 | 132/203 | 65.0246% |
| user7 | 136/185 | 73.5135% |
| 合计（total） | 268/388 | 69.0722% |

fixed-40 macro-F1 = 0.6289501959796908；最差用户 accuracy = 0.6502463054187192。开发 ID 摘要与 selection.development_ids_sha256 一致。

这些是72候选选择后在同一开发集上的描述性结果（descriptive development results），不能称独立测试成绩，不报告 final4 accuracy。

## 拟合人口与先验边界（Fit Populations and Prior Boundaries）

实际加载两个小型模型，均只有一个持久化 head.joblib，Ridge classes_ 精确为0–39，alpha=1000、solver=lsqr；head配置声明 T1。模型不是72个头的投票（voting）。

| 项目 | select | refit |
|---|---:|---:|
| 规范拟合人口行数 | 2039 | 2427 |
| StandardScaler.n_samples_seen_ | 1957 | 2342 |
| scaler 特征维度 | 3072 | 3072 |
| 先验计数总量 | 2039 | 2427 |
| 监督标签父产物 | labels_train12 / select | labels_refit14 / refit |
| select_users | user6,user7 | 空 |

select 模型 fit_users 为固定train12；refit为固定refit14。两个模型直接父节点均只有同一份标签无关 visual_features/raw_cache 与本 phase 的监督标签记录，没有 select 模型作为 refit 父节点。

共享原始特征引用摘要为 `0568814aa824809d937c88cc0b9f0155c12a4aefe88df3f53cc24b8715e5be19`。其无 fit_users，角色为 raw；共享refit14公开冻结特征不等于共享refit监督模型。

历史生产代码的[select拟合](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_teacher.py:164)在取train12子集后，用 valid 掩码同时筛特征、标签及 _weights 输入；Pipeline的 StandardScaler 因而只见有效train12。dev特征只用于预测，全部388条参与决胜。记录与模型中的1957样本数一致。[refit](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_teacher.py:196)重新构建模型，并以有效refit14重新拟合。

先验（class prior）使用各phase全部规范拟合标签计数，而非仅IR有效样本；prior文件中的 counts、probabilities、fit_ids_sha256与该规则一致。此次未为审计重新读取train/refit标签或重新拟合模型；类别权重只使用valid标签的依据是绑定历史生产代码及配方记录，不声称独立重训复现。

## 四份正式目标（Four Registered Teacher Targets）

| 分区 | 模型/先验phase | 规范行数 | valid IR | 先验回退行数 |
|---|---|---:|---:|---:|
| train12 | select | 2039 | 1957 | 82 |
| development2 | select | 388 | 385 | 3 |
| refit14 | refit | 2427 | 2342 | 85 |
| final4 | refit | 609 | 591 | 18 |

每份NPZ键精确为 sample_ids/class_ids/logits/probabilities/valid；无labels或旧多头/OOF字段。ID唯一，顺序及集合与本分区公共manifest和记录一致；class_ids=0–39；logits/probabilities形状为[N,40]，valid为bool[N]。valid与公共IR可用性逐条一致，所有概率有限、非负、归一化，最大行和误差不超过1.192093e-7。

全部invalid行概率与同phase先验一致。final4保留609条，18条缺IR/全纳入模态缺失行均恢复refit14先验；未因缺失删除分母。有效行概率与 softmax(logits) 在1e-7绝对容差内一致，T1写入预测配置；不复用旧OOF温度，不混同后续KD温度或Task9校准。

直接目标文件摘要（target SHA256）：

| 分区 | SHA256 |
|---|---|
| train12 | 022ddb67472ac88c50e56d119596048b971291617aa7cf810796d0aae6d4cf7a |
| development2 | de726269090d06d72e13d3382d0c3e51466903096137e16e61b987e33ab505b1 |
| refit14 | b7d90ffb24fa9c538d744920ccf01725f814177b339168f27758fcd7e9ad8002 |
| final4 | 930ff4c2f90930a3e41c52324affc810c82a999fcf40c019a476bc3f34c9176e |

对应预测记录路径（record paths）：

- [train12 record](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/artifacts/830f1b0177910e9cbe7ccd23e0df074b26b7f04e8d6735d82aa4cc18e053724f.json)
- [development2 record](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/artifacts/90eaf0f20791a15ec514a641b5afc971ce3ded4521ac063bdf0805e9fa0a4d4c.json)
- [refit14 record](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/artifacts/78a1a604e72001eaa2af6873e13d371ffae3d999162b829dc5922477db400760.json)
- [final4 record](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/artifacts/982908962a4aa7abaea66855e125926eb27f0c42c623700544376c83c30d577a.json)

## 祖先来源与封存边界（Ancestry and Sealed-label Boundary）

按四份targets的父引用去重检查22个记录元数据及引用哈希。监督/统计产物只有A1和class_prior；其标签父产物仅labels_train12与labels_refit14。其他角色是公共manifest、P28/P29、冻结visual_features，以及pose/visual公开初始化。没有历史教师预测、expert bank、P310、旧OOF目标或adapted_model父节点。

train/dev targets直接引用select模型及select prior；refit/final targets直接引用refit模型及refit prior。全闭包complete=true且formal，协议/配置一致；原始缓存没有被误登记为监督refit模型。

本次没有打开任何final私有标签。所查生成代码、配置、预测schema及祖先记录中未见其作为输入；这支持当前正式产物的标签封存声明，不等于对所有历史文件读取进行完整取证。Task13尚未生成16候选总冻结清单（generation manifest），不提前揭盲，也不能把Task4完成等同整个A1–A9实验完成。

## 剩余问题与启动边界（Remaining Issues and Next Gate）

**P2，状态同步（status synchronization），非模型效度阻塞：** 当前[计划第3行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md:3)、Task4状态段仍写refit登记待完成/final待接续；heartbeat-state.json还含task4_complete=false及旧阶段名。触发场景是后续接手或自动监测依赖旧状态，而非完成日志及正式记录，可能重复启动已完成阶段或误报仍在运行。最小修订是主审在审计汇总后同步计划、交接和监测状态，区分“Task4生成完成”“正式产物审计GO”“性能代码审计待通过”“Task5尚未开始”，不改模型或旧产物身份。本审计只写报告，不自行修改这些文件。

run_state.json当前仍为prepared，尚无frozen/revealed；本报告仅核对其现状，不声称完整Task13/14单向状态流程已验收。后续编排/冻结须按既定计划落实，不由本次A1审计替代。

校验性能修订（verification-performance revision）是用户已要求的工程门槛，正在由主审处理并另行独立工程审查。本方法学报告不触发全量SHA复验，也不要求重训已通过的A1模型。该工程门槛通过并完成状态同步后，本报告支持继续Task5。
