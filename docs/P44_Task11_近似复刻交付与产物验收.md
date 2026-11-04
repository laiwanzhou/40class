# P44 Task11 近似复刻交付与产物验收（Reconstruction Delivery and Artifact Acceptance）

正式产物生成于2026-10-30；收尾记录于2026-11-04，Asia/Shanghai。唯一指导仍为[P01计划](superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为[P02](handoffs/P02_current_handoff.md)。本报告不是新计划。

## 交付范围（Delivery Scope）

Task11按用户批准的源码约束近似复刻（Source-constrained Reconstruction）完成保留专家、全局混合（Global Mixture）、近邻平滑（KNN Smoothing）、中间学生适配（Intermediate Adaptation）、安全基座（Safe Base）、教师表（Teacher Bank）、分组／序列／门控（Group / Sequence / Gate）及P310优先覆盖（Precedence），生成四区正式A8锐化目标（Sharpened Targets）。不启动Task12，不揭示final4标签或计算其准确率（Accuracy）。

生成代码为`5a5f0e8`；A8人口／引用绑定与不可变复用修复为`ba11c7e`。历史产物的源码哈希（Source Hashes）保留生成当时版本，未追认为当前源码重新生成。

活动修订为`task11_reconstruction_v2`，基础协议身份（Protocol Identity）仍为`2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`。固定配方文件为`configs/experiments/task11_reconstruction_v2.json`，SHA256为`39f5c5a5a67a6d128ae089dfc619ccfdb537aea92b595aaf9d2d4bd7c67ba89c`。Task1–10产物继续复用。

## 完成的源操作（Completed Source Operations）

- P12骨骼／深度／热成像（Skeleton / Depth / Thermal）各自select与refit固定15轮末态，保留FP32主权重和新生成的FP16副本；没有按开发或最终成绩选轮次。骨骼／深度fit行数为1939／2324，热成像1840／2214。
- P12八分量按原固定列序，S+D权重0.6／0.4；RF消费terminal logits、原IMU校准和热路由。加六个视觉头后，原L-BFGS-B全局混合只用本phase训练标签拟合；同预测分区三近邻排除自身，0.40混合。
- 原中间结构化目标只为train12／refit14生成；各自A7 full初始化、固定12轮原无标签蒸馏（Label-free Distillation），拟合1957／2342条有效训练输入。开发区只用select模型，未在开发区或final4做本阶段适配。
- Safe probability为归一化后的`0.95 * adapted_raw + 0.05 * softmax(RF_terminal / 3)`；先对raw学生做原序列解码，再按源GLOBAL_REPEAT执行joint decision，保留原evidence 0.25／scale 1。
- 实际20列bank为safe、P85全局KNN教师、六个同根视觉头、十个同根机制头、P12 Skeleton和Thermal。每列绑定新run真实producer；缺失专家回退第0列同时保持availability=false，没有复制列凑教师数。
- 分组沿用P134／P137的sqrt/full/full特征与C0.03、peer2；序列沿用P270的emission0.65、transition0.45、alpha0.25、trigram1、beam50；门控沿用P139五评分／401阈值网格及决胜。未知时间不连边；分组的visual_available来自实际detail IDs。
- P310可用行目标为0.9805加其余39列各0.0005；全缺失行严格回退同phase训练先验（Training Prior），target_mask=false。保留锐化前概率、最终预测与route，供Task12精确配对。

## 四区正式产物（Four-partition Artifacts）

| 分区（Partition） | 完整行数 | 有效行 | 先验回退 | 分组改判 | 序列改判 | P310额外覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| train12 | 2039 | 1957 | 82 | 0 | 0 | 0 |
| development2 | 388 | 385 | 3 | 0 | 0 | 0 |
| refit14 | 2427 | 2342 | 85 | 6 | 0 | 0 |
| final4 | 609 | 591 | 18 | 154 | 0 | 0 |

这些数字只表示覆盖与改判计数，**154次改判不等于154次纠错**。本阶段没有新计算开发或最终准确率。

P309相较P255新增的P306依赖被排除大模型根；在当前合法20列和相同配方下，old／new group输入与输出等价，因此P310额外覆盖为0。来源证明为`group_bank_equivalence.json`；不人为制造不同结果，也不能由此推断队友原完整版P310无效。select分组阈值为2，refit为-0.1；两phase序列门控阈值均为2，合法退化为恒等输出。

## 验收与来源（Acceptance and Provenance）

正式目录为`outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops`，其`protocol/task11_reconstruction_v2/`保存三个FP16 component、P12/global/KNN component、source-base decoder／targets／adapt12及source_postprocessing component。后者绑定safe模型、四区safe输出、四区bank、两phase后处理模型与四区A8引用。

针对性回归（Targeted Regression）：后处理与旧角色16项通过（75.92秒）；A8错人口／错safe引用修复后3项通过（8.90秒）。原有RED和失败日志保留，不覆盖为成功。

最终有界验收脚本为`scripts/check_no_vote_task11.py`；2026-10-30记录18.18秒通过，验证237份去重祖先元数据、四区ID／40列／掩码／先验／锐化公式、训练预算与重复应用A8不改写产物。无标签复用探针记录监督标签打开次数0。目录改名后的新验收见[P46](P46_项目目录迁移与历史身份兼容.md)。独立复审见[P45](../reports/P45_Task11_独立复审结论.md)。

`source_postprocessing_component.json`中的`pending / task11_complete=false`是生成瞬间历史状态，原字节保留。最终收尾凭据另存`protocol/task11_complete.json`及`protocol/task11_independent_review_P45.json`，不得改写旧component来伪造当时已完成审计。

## 差异与后续边界（Differences and Next Boundary）

本次不是队友历史训练的精确重现（Exact Reproduction）。已授权差异包括单大视觉根、固定划分代替三折OOF、缺失记录下预先固定的15轮末态和decoder／repeat配方、中间适配仅本phase fit人口，以及随根剪枝形成的20列等价bank。固定划分的温度、阈值和gate在本phase训练人口内拟合，属于in-sample而非OOF；refit可以使用已纳入训练的user6／7，但开发预测只能来自select。

Windows workers0／torch threads4、公共Thermal路径无法定位时的unknown保留、Task5既有ROI质量FP16舍入残余继续披露，不宣称全链仅两项差异。原始数据、队友快照、两份提交包未修改。

Task12仍未实施：将从同一A7 refit独立进行最终40轮主适配和12轮诊断，消费本Task的A8；两端点不得串接，也不得读取最终标签或按最终分数选端点。须由后续任务明确推进。
