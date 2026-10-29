# P37 Task9 额外平均融合诊断（Additional Mean-fusion Controls）

日期：2026-10-29，Asia/Shanghai。分支 `experiment/teammate-single-teacher-task1`。

Task9 已完成实现、正式生成及有限自验（Bounded Self-acceptance）。本轮没有独立子智能体复审（Independent Subagent Review），不将自验称为独立 GO。Task10 尚未启动。按用户要求，完整实验结束后再统一评估效果，本阶段只记录结果与方法。

唯一指导仍为 [P01 计划（Plan）](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接仍为 [P02（Handoff）](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。本文件是交付记录，不是第二份计划或交接。

## 方法与来源（Method and Provenance）

只消费 Task5 的 A2 视觉学生（Visual Student）预测，以及 `protocol/task8_current.json` 指向的修正版 `ir_supervision_v2` 的 A5-S／A5-I 分支预测。核对 Task5 完成标记、当前 Task8 指针、新完成标记、P36 复审标记及其正式 ready 共六份小文件绑定；旧 A5 和 combined 诊断不作为输入。没有训练新网络、读取历史教师输出或修改 Task1–8 产物。

新增 `simple_fusion.py`、运行及验收 CLI，输出 A6-VS／A6-VI／A6-VSI。其角色明确为 `diagnostic_not_source_main`：这是额外校准平均对照，**不是队友最终主链操作**。注册器拒绝 A6 祖先进入 A7／A8 等主链及教师库（Teacher Bank）；本阶段不能改变主链配方、预算或监督目标。

保留原诊断契约：每个分支只在同阶段有效 fit 行上最小化负对数似然（Negative Log-likelihood，NLL），温度范围 0.25–4；缺失分支不参与温度拟合。只比较全体恒等温度（Identity）和全体拟合温度（Fitted）两种策略，在 development2 的 A6-VSI 上按 accuracy／macro-F1／最差用户准确率（Worst-user Accuracy）决胜，完全平局取 identity。同一策略用于三种融合，不分别按验证结果选 VS／VI 参数。

select 温度只 fit train12；refit 固定策略后只 fit refit14 重新估温，不复用 select 数值或选择新策略。目标属于固定划分的样本内预测（In-sample Predictions），不是折外预测（Out-of-fold，OOF）。refit 的选择来源通过小型 Selection 文件摘要记录，不把 select 学习模型作为 refit 的监督祖先。

按样本 ID 显式对齐，拒绝少 ID、错集合、用户映射错误及非 40 列顺序。可用分支等权平均概率（Available-branch Probability Mean）；全缺失行使用本阶段完整 fit 人口的类别先验（Class Prior），不删除样本或让缺失分支的先验稀释其他有效分支。

## 正式产物（Formal Artifacts）

运行根目录为 `outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops`，协议身份保持 `2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`。

- `A6/selection.json`：两候选结果、唯一选中策略、训练／开发 ID 摘要。
- `A6/{select,refit}/model.json`：两套温度参数与先验；分别为三个 A6 候选登记同 phase 模型引用，不是六个神经网络。
- `A6/{select,refit}/*_targets.npz`：四分区 × 三候选，共 12 份标准输出及登记引用。NPZ 包含样本 ID、类别列、log 概率、概率和 valid，不包含真值标签。
- `protocol/task9_artifacts_ready.json`：实际输入／模型／输出引用及六份前置绑定。
- `protocol/task9_complete.json`：有限自验完成入口，绑定本记录、测试与验收证据；自验范围不外推为全分支独立审计。

选中 fitted。温度（Temperature）如下：

| 阶段（Phase） | 视觉 V | 骨骼 S | IMU I | 温度拟合有效行 V／S／I |
|---|---:|---:|---:|---:|
| select／train12 | 1.1170526993 | 0.6134863948 | 0.25 | 1957／1956／1914 |
| refit／refit14 | 1.0037292862 | 0.5858160189 | 0.3182503180 | 2342／2341／2283 |

这里 S 的有效预测行比 Task8 训练人口少一条，是因为 IR-only 训练行仍没有骨骼推理输入；这不撤销 Task8 已恢复的监督训练行。

## 记录的阶段结果（Recorded Stage Results）

| 分区（Partition） | A6-VS | A6-VI | A6-VSI | 有效／先验行 |
|---|---:|---:|---:|---:|
| train12，2039 行 | 1981／2039，97.16% | 1982／2039，97.20% | 1983／2039，97.25% | 1957／82 |
| development2，388 行 | 184／388，47.42% | 183／388，47.16% | 196／388，50.52% | 385／3 |
| refit14，2427 行 | 2342／2427，96.50% | 2351／2427，96.87% | 2350／2427，96.83% | 2342／85 |
| final4，609 行 | 不评分 | 不评分 | 不评分 | 591／18 |

development2 用于既定两策略选择，不能把选后结果当作另一个独立测试集。refit14 包含 user6/7，不称验证成绩。final4 三份结果全部保留 609 行、40 类列，没有读取 final 私有标签或计算其准确率。此处不据分数修改主链或作整体效果结论。

## 验证与成本（Verification and Cost）

先写 8 项核心测试并观察全部 RED（模块缺失），实现后通过；期间发现测试夹具错误地共用 fit／dev 用户，修正为不同用户／ID，而没有放松人口隔离。补充当前版本绑定、旧版／条件性复审／错输出引用拒绝和全局门槛关闭时的假 A6 拒绝测试。

最终针对性回归（Targeted Regression）：`test_no_vote_simple_fusion.py`、`test_no_vote_v3_roles.py`、`test_artifact_record.py`，**33 passed，22.20 秒**。没有重跑旧全仓测试或退休缓存相关失败路线。

正式生成（Formal Generation）18.58 秒；有限自验 22.42 秒，核验两阶段人口／温度／fit NLL、完整 12 份小预测的 ID／40 列／valid／先验／来源，以及独立手写均值算术。FP64 参考与原 FP32 计算最大概率差 `2.6052382207e-7`，在预设 FP32 精度范围 `1e-6` 内，所有 argmax 一致。初次验收使用过严 `1e-7` 阈值失败，定位为分支 softmax 精度差异后仅修订验收的 FP64 对照容差，未修改生产预测、模型或选参。相同生产算法的无标签推理复算仍使用 `1e-7` 阈值。

final4 全 609 行均值推理期间，禁止打开标签／网络权重载荷的探针尝试数为 0；这证明均值推理只需已载入的预测和温度，不是对全流程任意文件访问的普遍声明。前置六文件绑定保持，12 份目标总计 9,599,018 字节，约 9.15 MiB。大缓存扫描、原始目录 SHA、网络重训均为 0。

证据在 `outputs/task5_audit/`：`P37-task9-red.log`、`P37-task9-core-tests.log`、`P37-task9-regression.log`、`P37-task9-formal.log`、`P37-task9-acceptance.json/.log`。

全局 `formal_generation_ready=false` 保留；A6 只获诊断局部就绪（Scoped Diagnostic Readiness），P12／Thermal／P89／A18 等完整教师库门槛、已记录的 ROI 质量精度残余仍独立存在。下一任务为 P01 Task10；本轮按用户授权范围完成 Task9，未启动 Task10。
