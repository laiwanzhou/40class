# Task5 正式产物独立工程验收（Formal Engineering Acceptance）

日期：2026-10-04。工作树（Worktree）：`_single_visual_processing_replication`；提交（Commit）：`faf8980`；运行（Run）：`fixed-split-single-teacher-v2`。

结论：**GO**。正式 A2 select/refit 模型、四区预测（Targets）和四区原生序列（Native Sequence）通过本次约定的轻量实际产物验收（Lightweight Artifact Acceptance），没有发现重要问题（Important Findings）。本结论不推进 Task6，不计算或报告 final4 accuracy。

## 模型与预算闭合（Model and Budget Closure）

| 阶段（Phase） | 完成 epoch | 登记预算（Budget） | 有效拟合行（Valid Fit Rows） | 每 epoch 实际训练样本（Training Samples） |
|---|---:|---:|---:|---:|
| select | 18 | 17 | 1957 | 1956 |
| refit | 17 | 17 | 2342 | 2342 |

select 的 history 覆盖 epoch1–18，按既定 accuracy、macro-F1、worst-user、较早 epoch 的排序，最优为 epoch17。selection、model 配置与 completion 中的预算一致；refit history 覆盖 epoch1–17，使用同一已选预算。select 每 epoch 少 1 条来自既有 singleton 尾批丢弃规则（Drop Singleton Batch），未将 1956 与有效拟合人口 1957 混称。

两个 completion 的 identity 与本阶段 identity 文件一致，引用本阶段登记 artifact；select completion 保存相同 Selection，refit 的 selection 字段为 null。model 文件与登记 config 精确一致，recipe 与正式协议一致；拟合人口（Fit Population）、用户集合、有效拟合 ID 摘要均与实际 public rows/masks 对齐。历史训练损失有限（Finite），直接 feature loss 与 stage KD 记录为零。

直接 checkpoint 文件各校验一次，实际摘要分别为：

- select：`657126306c122bb710b9f1e856eb3522b35084c9f63cf687afceee0e64bcb1a7`。
- refit：`0c1cc272e0ab063fadcee0696a9bee496d720aa66c57938c23f03cfd8cd1fc43`。

两 checkpoint 及模型 ArtifactRef 不同；refit 直接父节点不含 select 模型，监督祖先阶段由登记元数据（Metadata）校验。

## 四区预测与序列（Predictions and Sequences）

| 分区（Partition） | 模型阶段 | 规范人口（Rows） | 有效（Valid） | 缺失（Unavailable） | sequence shape |
|---|---|---:|---:|---:|---|
| train12 | select | 2039 | 1957 | 82 | `[2039,2,3,16,512]` |
| development2 | select | 388 | 385 | 3 | `[388,2,3,16,512]` |
| refit14 | refit | 2427 | 2342 | 85 | `[2427,2,3,16,512]` |
| final4 | refit | 609 | 591 | 18 | `[609,2,3,16,512]` |

四份 targets 的登记引用、内容摘要（Content Digest）、sample_id 精确集合与顺序、用户绑定、40 类列顺序（Class Order）均通过。schema 仅包含 sample_ids/class_ids/logits/probabilities/valid；logits 与 probabilities 为有限值，概率范围和逐行归一化（Row Normalization）满足标准 Prediction 契约。

所有无效行的 probabilities 与同 phase 先验（Class Prior）逐元素差为 0；fallback logits 与 `log(prior)` 一致。**final4 的全部 18 条缺失样本使用 refit 先验**，没有借用 select 先验、排除规范评估行或读取最终标签。

四份 sequence 登记记录均为正式 complete，phase/partition、模型摘要、pixels 摘要及直接 model/pixels/prior 父引用一致。rows.csv 与规范分区的 ID/用户/类别轴精确一致。全部序列 header 为 `float16[N,2,3,16,512]`；小型 completed 掩码（Completion Mask）全真，anchor_valid 与 A2 targets valid 及实际 pixels view_valid 对齐。

四区 **anchor_logits 与同 phase A2 targets logits 的最大绝对差（Maximum Absolute Difference）均为 0**。因此本次真实产物中的冻结锚点（Frozen Anchor）与对应 A2 预测完全一致；select/refit 没有混用模型身份。

## 少量真实时间片段（Sampled Temporal Fragments）

每区按有效行顺序选首条、中间条、末条，共 12 条真实有效 sequence；只读取这些行，未遍历 sequence 内容。所有抽样片段为有限值，第一与最后时间位置的表示不同：

| 分区 | 抽查行号（0-based） | 首末时间最大绝对差 |
|---|---|---|
| train12 | 0、1015、2038 | 3.283203、3.399414、4.091370 |
| development2 | 0、192、387 | 5.673828、4.265991、5.497192 |
| refit14 | 0、1208、2426 | 3.119141、3.569336、5.619873 |
| final4 | 0、306、608 | 4.878052、4.658203、2.753418 |

这些实际片段不存在池化向量沿时间复制（Repeated Pooled Vector）的情形，与已 GO 的原生 MC3 时间序列接口相符。每区另抽取 1 条无效行，sequence 为全零；该检查仅证明抽样行，不宣称全量 sequence 内容逐元素复验。

## 验证证据与限制（Evidence and Limits）

独立探针输出：`outputs/task5_audit/formal-independent-engineering-probe.json`。完整轻量探针退出码 0，计时 `7.958 s`；执行过程中对 `images.npy` 或 `sequence.npy` 的整文件 SHA 调用设为失败条件。实际校验直接 checkpoint 为 2 次、四份 targets 为 4 次；images/sequence 全内容 SHA 为 0。

本次仅读取直接模型、小型 targets、登记/完成/预算/历史元数据、public rows、掩码与数组 header，以及上述 16 条 sequence 抽样行。祖先只核对记录元数据（Metadata Ancestry），不递归打开祖先 payload；没有遍历 raw 目录、扫描 images 或全量 sequence 内容，没有训练、修改代码或重做已通过的代码审计。

final4 私有标签（Private Labels）未读取，final accuracy 未计算。本验收证明人口、来源、预算、概率、掩码与抽样原生时间特征契约；不声称原始文件或全部缓存内容完整性审计（Full Content Integrity Audit）通过。
