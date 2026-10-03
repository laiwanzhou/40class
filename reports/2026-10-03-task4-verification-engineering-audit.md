# Task4 校验性能修订工程复审（Engineering Audit）

日期：2026-10-03，Asia/Shanghai。审计对象为 `ae3e15466e0dc6e0a80c07cc220658255906f0c2` 基础上的当前工作区修订，以及已生成的 Task4 正式产物。审计员未参与实现；本次只写本报告，不修改实现，不读取最终私有标签，不训练模型。

## 结论（Decision）

**GO。当前没有未解决的 Critical / Important 工程阻塞，可以按新校验策略进入 Task5。** 这是本次代码、回归与正式产物证据复核的结论，不沿用先前实现审计的 GO。

本报告采用实施计划第25–40行的校验策略修订（Verification Policy Revision）：取消后续缓存消费中的原始全目录哈希、完整缓存额外扫描和祖先载荷递归复验；保留直接输入、记录元数据、ID、类别列、phase、partial 与先验边界。未要求恢复旧大面积检查。

## 本轮发现与修复（Findings and Fixes）

本轮曾发现两项 Important，已在最终受审补丁中关闭：

1. **实际消费 ROI 缓存未校验（Consumed ROI integrity）。** 注册表改为元数据检查后，`_prepare_clips` 仍会直接打开选中 ROI NPZ。现于 `visual_teacher.py:318` 在真正消费该单条文件之前调用 `verify_file`；不扫描其余 ROI 或原始目录。回归测试 `test_extraction_checks_only_the_consumed_roi_payload` 独立复测通过。
2. **重新登记初始化权重可能掩盖内容漂移（Initializer receipt binding）。** 仅比较文件位置并检查新记录自己的摘要，不能防止同尺寸变化的权重被重新登记为公开初始化。现于 `visual_teacher.py:291–293` 将实际 initializer 记录的路径与摘要精确对照已固定权重回执；`visual_teacher.py:327` 仅在首次实际初始化模型时检查将加载的三个文件。回归测试 `test_reregistering_modified_initializer_cannot_replace_pinned_weight_receipt` 独立复测通过。缓存拟合与预测不因此读取初始大权重。

最终受审 `visual_teacher.py` SHA256：`445c4cc89595f4c6d7b03fe10c907d0c88ea69d12ef452669804ac8a40f556e3`。

## 保留的工程边界（Retained Contracts）

- `artifact_record.py:119–178`：记录哈希、协议/配置、fixture/formal、用户角色、精确 ID、类别列、partial 祖先、监督 phase、A9 同一 A7/A8 来源及环检测仍保留。`checked` 按记录摘要去重，同一次 `verify` 遍历不会重复校验共享祖先记录。该去重范围是单次调用，不是跨进程的永久缓存。
- `artifact_record.py:135–136,180–187`：请求的模型、先验、预测、公共清单等直接文件继续检查；raw cache / public weights 的具体消费文件由 `verify_file` 显式检查。未消费的祖先文件内容不再打开。
- `visual_teacher.py:94–115`：读取特征时保留 P29/public ID 连接、formal 祖先要求及 complete 检查；只检查所选聚合 `features.npz`，随后验证数组键、ID/order、40类列、shape、有限值及 boolean valid。已删除后续 head/predict 的 `snapshot_raw_files` 调用。
- `no_vote_protocol.py:122–129`：加载协议检查已登记源码/权重回执，绑定其内容摘要，不重读全部1167个源码文件或全部初始化权重。独立 acquisition 的全内容校验默认仍保留。
- `teammate_source.py:70–125`：实际请求模块及已导入快照依赖检查来源和内容；装饰函数通过 `inspect.unwrap` 校验原定义，返回完整 wrapper，保留推理模式（Inference mode）。
- 生成原始视觉特征的入口仍保留 producer 阶段的 inventory 检查（`visual_teacher.py:282–286`）；这是原始生产路径的明确范围例外。它不进入使用已登记特征的拟合、预测或本次轻量审计，也不能据此声称未来所有原始抽取入口已取消扫描。

## 历史产物与正式证据（Historical Compatibility and Evidence）

旧记录没有补写新的字段、identity 或父引用。缺少字段的旧记录解析为历史 `verification_policy="full-v1"`，新登记记录写 `direct-metadata-v2`；当前校验器对旧记录也采用新的消费范围，因此旧字段表示生产时策略，不表示本次重跑了全内容校验。

`run/protocol/task4_v1_sources/manifest.json` 保存五份旧实现。独立读取这五个小型源码归档并逐项计算 SHA256，**五项全部与清单相符**：artifact_record、visual_teacher、teammate_source、no_vote_protocol、no_vote_weights。未对原始数据、权重或完整缓存执行全量 SHA 审计。

正式状态与 `outputs/task4_audit/verification-performance-real.json` 记录四份目标产物已闭合：train12 2039、development2 388、refit14 2427、final4 609。该真实验收记录为4.256秒、原始文件哈希调用0、元数据及直接文件哈希调用63；一次特征加载和8条 train12 复算的最大 logits 差为0。此为已记录的实测证据，本审计没有再次重算全人口。最终标签读取标志为 false。

正式方法学、开发结果和四份预测的数值契约由另一个独立审计覆盖；本报告不重复其全人口计算。

## 独立验证与检查范围（Validation and Scope）

- 独立运行 `tests/test_no_vote_verification_performance.py`：**4 passed**，验证缓存模型和登记不重读原始祖先、协议不扫描未消费资产、符号加载不哈希未使用源码、直接文件变化仍拒绝。
- 独立运行本轮两项直接消费修复的针对性回归：**2 passed / 10 deselected**，4.05秒。
- 角色、ID、phase、partial 和标签边界相关逻辑未因本次优化移除；原始祖先载荷改变而尚未实际消费，不再作为自动拒绝条件，符合用户修订。

本次没有运行旧全量校验、遍历原始目录、加载初始大权重、再次训练教师、读取最终私有标签或要求重训已完成 Task4。后续 Task5 应继续使用已登记 A1 目标与特征，直接消费时检查必要文件和数组契约；不能恢复每个 batch/epoch 的祖先内容扫描，也不能把本次 GO 解释为 Task5 已实现或训练通过。
