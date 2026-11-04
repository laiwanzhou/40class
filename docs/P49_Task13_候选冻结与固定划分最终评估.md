# P49 Task13 候选冻结与固定划分最终评估（Candidate Freeze and Fixed-split Final Evaluation）

日期：2026-11-04，Asia/Shanghai。唯一指导为[P01](superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为[P02](handoffs/P02_current_handoff.md)。本报告记录一次性最终评估，不构成新计划。

## 评估对象与授权（Population and Authorization）

用户要求推进Task13，并明确可使用固定划分标签。本次对象为user4／17／23／24的609条封存样本；user6／7仍是此前开发区，不混称最终测试。无需Kaggle官方隐藏测试标签。

旧临时目录中的`final_labels.csv`已不存在。原准备命令记录于`docs/task23_input_pose_roi.md`，标签来源为本分支`metadata/manifest.csv`。冻结前仅确认该文件存在，未打开。评估器在冻结且独立复审通过后，先持久化`revealed`，再以`canonical_manifest`模式读取原清单；用原`NAMESPACE + original_sample_id`的SHA256精确匹配公开opaque ID，只提取冻结609行，额外核对用户归属，不依据类别或动作目录猜配对。

这项后续授权覆盖P01旧“现阶段不正式揭示”限制，不改变16候选、预算、主比较或禁止最终调参的边界。

## 冻结清单与证据（Frozen Candidates and Evidence）

固定候选按原顺序为A1、A2、A5-S、A5-I、A6-VS、A6-VI、A6-VSI、A7、A7-mask、A7-shuffle-S、A7-shuffle-I、A7-zero-S、A7-zero-I、A8、A9-12、A9-40。主端点始终为A9-40，不根据最终分数挑选。

正式冻结文件为`run/protocol/task13_frozen_generation.json`，SHA256：`0a120e444d6142ed72213f1d02d5af4587dbb57a361605ddf890bf732cecc4ee`。基础协议身份仍为`2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`。

冻结用时12.19秒：16个候选完整609行／40列，A8固定591条有效／18条先验回退；验证176份去重祖先元数据、308份实际消费小文件绑定。禁止打开原带标签清单与三个监督标签payload的探针计数0。未扫描原始图像、骨干特征／序列或模型大载荷。

冻结绑定每个预测的ArtifactRef、实际小NPZ SHA、样本／用户／类轴、A8目标mask、训练先验、公开模态可用性，以及源操作、roster、完成／复审凭据和评估器源码。揭示前重新检查绑定；正式揭示还必须持有批准同一generation SHA／protocol的独立复审GO。

## 验证与性能（Verification and Performance）

72项针对性回归通过（46.39秒），包括精确ID、标签重排、原清单ID映射、固定40类macro-F1、相同准确率但预测不同、配对rescue／harm、不可重复揭示、输入漂移拒绝、冻结后禁止注册／训练，以及原协议和目录迁移契约。

独立复审发现小型描述文件未按记录SHA验证的P2；已补充冻结时和揭示前两时点检查，先RED再GREEN并复审关闭。正式评估先写`revealed`再开标签；标签读取或校验失败也不恢复到frozen重复评估。

一次预检profile测得119.5秒，根因是误枚举原始缓存中的大量JSON及重复路径解析，随后被大载荷限制拒绝；该失败未冻结或读取真值。修订后排除祖先raw cache／公开权重／监督标签payload，仅验证已消费小元数据；缓存文件路径列表但每次仍检查stat／SHA。不可变协议的canonical identity只计算一次，未改变哈希字段或设置override；新配置实例仍产生新身份。优化预检12.70秒，正式冻结12.19秒。

## 解释边界（Interpretation Limits）

A9使用过final4公共输入及A8伪标签，属于无标签目标域适配（Transductive Target Adaptation）；它不等于对另一批从未适配样本的纯归纳泛化。真实标签在冻结后才进入独立评估器，不进入训练或选择。

本次继承Task11批准的近似配方、单大视觉根、固定划分替代OOF、20列合法剪枝与Task5／10质量FP16残余；不能宣称只有教师数量一项差异。所有最终差值只描述当前固定人口，不据此重训、改阈值或更换主端点。

Task14尚未实施，main历史回顾文档继续等整个复刻实验完成再更新。

## 一次最终结果（One-time Final Results）

正式评估已按授权完成，run_state已为`revealed`。只打开一次原固定划分标签用于本次评估，未重复执行评估器；结果为`run/evaluation/task13_final4_once/metrics.json`。实际最终标签联合覆盖40类；这不表示每名用户分别都有40类。

主端点A9-40与A8均为**467/609＝76.68%**；单视觉教师A1为**445/609＝73.07%**。主比较提升22条、3.61个百分点。18条全缺失先验行正确数0；排除这些行后的固定591分母上，A9-40与A8均为467/591＝79.02%。主指标仍保留完整609分母。

| 候选（Candidate） | 作用（Role） | 正确/609 | 准确率（Accuracy） | 40类宏平均F1（Macro-F1） |
|---|---|---:|---:|---:|
| A1 | 单大视觉教师（Visual Teacher） | 445/609 | 73.07% | 68.16% |
| A2 | 基础视觉学生（Visual Student） | 394/609 | 64.70% | 58.86% |
| A5-S | 骨骼学生（Skeleton Student） | 299/609 | 49.10% | 40.47% |
| A5-I | IMU学生（IMU Student） | 247/609 | 40.56% | 32.08% |
| A6-VS | 视觉＋骨骼平均诊断（Average Diagnostic） | 389/609 | 63.88% | 57.70% |
| A6-VI | 视觉＋IMU平均诊断 | 373/609 | 61.25% | 54.36% |
| A6-VSI | 三模态平均诊断 | 374/609 | 61.41% | 54.02% |
| A7 | 源加性融合（Source Additive Fusion） | 398/609 | 65.35% | 59.21% |
| A7-mask | 独立缺运动控制（Mask Control） | 391/609 | 64.20% | 57.48% |
| A7-shuffle-S | 独立骨骼打乱控制（Shuffle Control） | 395/609 | 64.86% | 59.15% |
| A7-shuffle-I | 独立IMU打乱控制 | 397/609 | 65.19% | 58.74% |
| A7-zero-S | 同A7推理骨骼置零（Zero at Inference） | 359/609 | 58.95% | 52.16% |
| A7-zero-I | 同A7推理IMU置零 | 404/609 | 66.34% | 59.65% |
| A8 | 完整源教师表与后处理（Teacher Bank and Postprocessing） | 467/609 | 76.68% | 71.36% |
| A9-12 | 独立12轮诊断适配（Diagnostic Adaptation） | 460/609 | 75.53% | 70.17% |
| A9-40 | 固定40轮主适配（Primary Adaptation） | 467/609 | 76.68% | 71.32% |

## 同人口配对比较（Paired Same-population Contrasts）

| 比较（After − Before） | 正确数增量 | Accuracy百分点 | 救回（Rescue） | 新增错误（Harm） | 预测不同 |
|---|---:|---:|---:|---:|---:|
| A9-40_minus_A1 | +22 | +3.61 | 32 | 10 | 61 |
| A8_minus_A1 | +22 | +3.61 | 32 | 10 | 61 |
| A9-40_minus_A8 | +0 | +0.00 | 0 | 0 | 2 |
| A9-40_minus_A7 | +69 | +11.33 | 92 | 23 | 170 |

40轮适配对A8没有准确率净损失：两者都467条正确。仍有2条预测不同，rescue／harm均为0，即两处都答错但错类不同；固定40类macro-F1从A8的71.36%变为学生71.32%，不能称所有指标或逐样本完全无损。12轮诊断为460/609＝75.53%，不据此重选端点。

链路上主要可观察现象：A1→A2下降8.37个百分点；A2→A7增加0.66个百分点；A7→A8增加11.33个百分点；最后A8→A9-40准确率持平。A7→A8包含整套已冻结教师表／中间适配／后处理，不是某个单独机制的纯因果增益。当前全流程相对A1总体增加3.61个百分点，不能把它外推为队友官方测试0.91的直接可比差距分解。

A7-zero-I比A7多6条正确，A7-zero-S少39条；这是固定模型推理置零诊断，提示当前骨骼信息更有帮助、IMU可能产生部分干扰，不足以据此在揭示后删除分支或重新选择训练方案。

## 主端点分用户结果（Primary Endpoint by User）

| 用户（User） | 正确/样本数 | 准确率（Accuracy） |
|---|---:|---:|
| user17 | 133/166 | 80.12% |
| user23 | 90/141 | 63.83% |
| user24 | 131/159 | 82.39% |
| user4 | 113/143 | 79.02% |

完整JSON另外保留每候选的40×40混淆矩阵、类别支持、用户／模态可用性分层、A8.mask分层，以及四项比较在609／591／18上的完整rescue／harm计数。后续验收仅用这些聚合结果和冻结预测，不重复打开真值。

## 收尾状态（Completion Status）

独立精确冻结审计、登记CLI复审和一次评估后的聚合验收均PASS，无未解决问题；详见[P50](../reports/P50_Task13_独立复审与冻结代次验收.md)。Task13完成凭据为`run/protocol/task13_complete.json`。冻结代次与预测从未改写，revealed后不训练或选参；Task14尚未启动。
