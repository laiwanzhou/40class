# P02 当前实验交接（Current Experiment Handoff）

更新：2026-11-04，Asia/Shanghai。本文件是本分支唯一交接入口（Canonical Handoff），后续原位更新；唯一指导为[P01计划](../superpowers/plans/P01_teammate_single_teacher_fixed_split.md)。旧归档和审计输入快照只作历史，不作为第二份计划或交接。

## 当前状态（Current Status）

**Task11已完成且独立GO；用户已授权推进Task12，目前代码／真实批次／独立生成前复审通过，即将正式训练。** 正式生成于2026-10-30，本次完成中断后的审计落档、目录迁移与收尾。交付[P44](../P44_Task11_近似复刻交付与产物验收.md)，三项历史独立审计[P45](../../reports/P45_Task11_独立复审结论.md)，本日目录迁移及独立复审[P46](../P46_项目目录迁移与历史身份兼容.md)。

分支`experiment/teammate-single-teacher-task1`，origin为`https://github.com/laiwanzhou/40class.git`。用户只改父目录名：真实项目目录`D:/work/26.7.14_kaggle`；工作树为其下`_single_visual_processing_replication`。历史`D:/work/2026.7.14_kaggle`现为同文件目录联接（Directory Junction），保持读取历史字符串路径。用户明确要求保留新真实目录名。

显式迁移配置`configs/experiments/project_root_relocation.json`绑定七项小文件锚点；物理路径用于实际访问，逻辑路径用于原身份。历史JSON/SHA、训练配置、权重和预测未改写。迁移已通过42项针对性回归（38.59秒）、正式消费验收（15.62秒）及独立最终复审，无未解决P1/P2。旧ROI提取生产器的raw_input_hashes字串比较未认证迁移，不能宣称所有旧提取/训练断点均可直接恢复。

## 权威入口（Authoritative Records）

Run为`outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops`。

- 最终完成：`run/protocol/task11_complete.json`，含四区A8/bank、组件、配方、验收和审计引用及SHA。
- 最终独立审计：`run/protocol/task11_independent_review_P45.json`，verdict=GO。
- 活动状态：`run/protocol/task11_reconstruction_v2/progress.json`，task11_complete/a8_generated=true，task12_started=false。
- 原`source_postprocessing_component.json`中的pending/false为2026-10-30生成瞬间历史，原字节保留；不能覆盖它伪造当时已审计。
- 原暂停／恢复和早期preflight记录只供历史。当前没有训练worker或待接续supervisor，不启动旧continuation脚本。

基础协议身份（Protocol Identity）仍`2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`；活动revision为`task11_reconstruction_v2`，配方SHA`39f5c5a5a67a6d128ae089dfc619ccfdb537aea92b595aaf9d2d4bd7c67ba89c`。不得改identity/旧描述符/配方来绕过校验。

## 产物与处理（Artifacts and Operations）

Task1–10既有产物继续复用，Task9/10已独立PASS（P39）。Task11新增三个P12教师各select/refit固定15轮、FP16副本、P12八分量/global混合/本分区KNN，中间source-base各12轮适配，safe base、20列真实bank、group/gate/sequence/P310与四区A8。

| 分区（Partition） | 完整行数 | 有效A8 | 先验回退 |
|---|---:|---:|---:|
| train12 | 2039 | 1957 | 82 |
| development2 | 388 | 385 | 3 |
| refit14 | 2427 | 2342 | 85 |
| final4 | 609 | 591 | 18 |

分组改判分别0/0/6/154，序列与P310额外覆盖均0；没有用真值判断这些改判是否正确。P306被单根约束排除后，old/new合法20列与配方等价，是自然退化，不复制专家造差异。四区40列表示类别轴，不需要重开最终标签检查类别支持。

## 近似与数据边界（Approximation and Data Boundaries）

这不是精确历史重现：单大视觉根、固定划分替代三折OOF、缺记录下批准的15轮末态及decoder/repeat配方、中间仅本phase fit池适配、剪枝20列等价bank均已披露。组分类／温度／门控在本phase训练人口内拟合，为in-sample而非OOF。既有Task5 ROI质量FP16舍入残余保留，不把Task1–10追认为严格数值全等。

train12为12用户；开发为user6/7；refit14为两者并集；final4为user4/17/23/24。开发只用select，refit可用已纳入训练的user6/7；不得把refit给开发报分。只允许前三分区训练所需标签，禁止读取final4私有标签、分布或准确率，禁止按最终结果调参。

原始数据、队友源码快照、两份提交包只读。禁止用历史教师预测作新祖先。main历史回顾文档等整个复刻实验完成才更新。两个既有未跟踪报告`reports/2026-10-02-history-cache-reuse-inventory.md`、`reports/2026-10-02-old-visual-worktree-retirement.md`不顺手提交或删除。

## 验证与后续（Verification and Next Step）

最新日志`outputs/task5_audit/P46-final-regression.log`、`P46-migrated-acceptance-final.log`；结果JSON`P44-final-acceptance.json`本次更新，原2026-10-30结果保留在旧P44日志。237份祖先元数据、四区小输出、15/12轮预算与先验验收通过；cached A8引用及payload stamp不变，监督标签打开0，准确率未计算。

遵守有界校验（Bounded Verification）：不重新全SHA raw/images/features/sequence、不递归加载祖先大payload、不每步重跑全仓旧suite。只核实当前消费输入／小预测表／去重元数据；新问题才加针对回归。

当前按唯一P01的Task12执行最终40轮主端点和独立12轮诊断，各自从相同A7 refit初始化，消费本Task A8。两端点不得串接；不得在开发user6/7另做伪标签适配，不用一致率或最终分数选端点。用户已明确推进Task12；本轮完成其生成／验收／独立复审后停在Task13之前。Task13负责完整冻结及一次最终揭示，Task14负责编排收尾。

没有定时自动化或新服务。本机正式Python为`D:/Anaconda/envs/PyTorch2.7/python.exe -B`。早期训练中断、暂停、恢复及源配方细节见P41–P43和现有台账，无需重做已完成训练。

## Task12最新实施（Latest Target Adaptation）

新增target_adaptation.py及CLI，40轮与独立12轮从同A7 refit初始化；原训练函数＋完整轮恢复、强制609/591/18、三监督标签payload拒绝探针、无增强raw概率及三分母一致率。30 targeted GREEN38.20秒；真实train12四条含IR-only，原最终Dataset除既有quality FP16残余2.44140625e-4外tensor精确一致，原40scope真实更新全state差0，29.60秒。生成前独立p47_task12_review GO；缓存预测requested-model/endpoint/target绑定P2经RED→GREEN并复审CLOSED。无正式A9产物已生成；正式日志将为outputs/task5_audit/P47-target-formal.log，运行状态run/protocol/task12_status.json；不要重跑旧Task11 continuation。训练中不改target_adaptation.py，以免恢复身份漂移。
