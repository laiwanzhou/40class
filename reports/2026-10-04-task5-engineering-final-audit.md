# Task5 最终独立工程审计（Final Independent Engineering Audit）

日期：2026-10-04。工作树：`_single_visual_processing_replication`；分支（Branch）：`experiment/teammate-single-teacher-task1`。

结论：**GO**。在本次约定工程审查范围内，原审计 E1–E4 及上一轮复审 R1–R4 已修订闭合，没有剩余重要问题（Important Findings）。可以进入已授权的 Task5 正式学生训练（Formal Student Training）；本结论不表示完整训练已完成，也不推进 Task6。

## 最新修订核对（Patch Verification）

| 上轮问题 | 当前实现与独立证据 | 判断 |
|---|---|---|
| R1 失效锁竞争（Stale Lock Race） | `phase_lock` 使用 `filelock.FileLock`，整个阶段持有 Windows 操作系统文件锁（OS File Lock）、`timeout=0`，不再依赖 PID/JSON/删除失效路径。独立探针确认锁类型为 `WindowsFileLock`；持锁线程存在时第二调用被拒绝，释放后留下旧文件名仍可重新获得锁 | 闭合 |
| R2 部分完成发布（Partial Completion Publication） | 内部实现先完成 selection；外层先原子写 `completion.json`，再发布 `artifact.json`。独立探针在 completion 已写、artifact 尚未发布时模拟中断；恢复返回相同 ArtifactRef 与预算，内部实现总调用仅 1 次，checkpoint 字节保持不变 | 闭合 |
| R3 writer receipt 文件集（Receipt File Set） | 恢复精确要求六个 `PIXEL_KEYS.npy` 加 `rows.csv` 共七个当前 output 路径；摘要格式和文件存在性由登记器（ArtifactRegistry）检查。独立探针删除 receipt 的 `images.npy` 项，确认发布 artifact 前拒绝 | 闭合 |
| R4 重复直接输入校验（Duplicate Direct-input Validation） | CLI 创建仅服务本阶段的 `StageVerificationContext`，`load_teacher` 与 `guard_teacher` 共享按 ArtifactRef 缓存的实际特征表（Feature Table）；refit 外层验证过的 select 记录传入内部复用。独立探针中 teacher 加载及 guard 的 `_load_features` 调用总数为 1，替换为同形状零特征仍被明确拒绝 | 闭合 |

## 原始接口与恢复边界（Interfaces and Resume Boundaries）

- 教师特征（Teacher Features）绑定 prediction 的唯一直接 `visual_features` 父引用，按登记 ID/用户/类别轴重排后与提供值精确比较；没有因加载去重取消数值绑定。
- refit 预算（Epoch Budget）与已登记 select `budget_epochs` 一致，并检查 select 模型角色、协议及 recipe。refit 从公开 MC3 权重重新初始化，不复用 select 训练权重。
- 完成同 identity 阶段幂等返回（Idempotent Return）；completion 恢复使用既有登记记录，不重新序列化 checkpoint。未完成训练按 epoch 边界恢复 model、optimizer、scaler、best/history 与 Torch/Python/NumPy/CUDA 随机数状态（RNG State）。
- 无完整写入摘要凭据（Writer Digest Receipt）的部分 pixels 明确拒绝自动恢复并要求新输出目录；完整 receipt 支持补登记。这是明确的有限恢复范围（Limited Resume Scope），无需为了恢复补全量缓存 SHA。
- 正式旧 pixels 保持已有 producer 身份并直接加载，没有重建或改写历史摘要。直接消费数组继续检查 schema、ID、完成标记及小型 mask/time 元数据。
- 已有推理及序列（Inference and Sequence）接口保持无教师/标签依赖；同 phase 模型、pixels 与冻结 anchor 的身份绑定仍在。本轮未改损失、原生时间序列或既有 singleton 尾批规则（Drop Singleton Batch）。

## 验证证据与范围（Evidence and Scope）

主代理报告最新定向测试（Targeted Tests）为 `30 passed`，用时 `24.63 s`，包括 epoch 中断恢复、完成后幂等返回、错误预算/特征拒绝，以及删除 teacher/labels 后推理与 sequence 的 fixture。本次已阅读相应代码与测试；没有重复运行整套已通过测试。

本次独立执行四项临时隔离探针（Isolated Probes），合并命令退出码 0，总用时约 `4.05 s`：

1. `shared_feature_validation_context`：`loads=1`，`tampered_features_rejected=true`。
2. `os_lock_active_and_stale_filename`：active duplicate 被拒绝，释放后旧文件名重用成功，实际类为 `WindowsFileLock`。
3. `completion_before_artifact_recovery`：completion 先写成功；恢复同 artifact；`impl_calls=1`；`checkpoint_unchanged=true`；selection 预算为 1。
4. `pixel_receipt_missing_images`：`rejected_before_publication=true`。

探针只使用系统临时目录（Temporary Directory）、小型合成输入及 mock；没有执行模型训练、读取 final4 私有标签（Private Labels）、扫描 raw 目录或读取正式像素缓存的全部内容。只写本报告，没有修改实现代码。此前真实 GPU 4 样本验收及正式 pixels 完成结果沿用已有证据，没有重新触发大面积校验。

GO 的含义是本轮指定接口、来源绑定、生命周期（Lifecycle）及校验性能（Verification Performance）修订可以进入正式 Task5 执行；完整训练耗时、最终模型质量及全流程完成状态仍需由后续实际执行记录证明。本报告不是原始文件或全部缓存内容完整性审计（Full Content Integrity Audit）。
