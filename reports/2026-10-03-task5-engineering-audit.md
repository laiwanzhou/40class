# Task5 独立工程审计（Independent Engineering Audit）

日期：2026-10-03。审查基线（Baseline）：`25b8ec0`；分支（Branch）：`experiment/teammate-single-teacher-task1`。本报告针对本轮工作树差异，不沿用此前未完成审计者的结论。

结论：**NO-GO，暂不启动正式 A2 学生训练（Formal Student Training）**。下面四项重要问题（Important Findings）应修复并定向复审。既有正式像素缓存完成和真实 GPU 批次验收不消除这些接口及续跑问题。

## 范围与边界（Scope and Boundaries）

已审查计划中的 Task5、2026-10-03 校验策略修订（Verification Policy Revision）、设计 S7，以及 `pixel_cache.py`、`no_vote_datasets.py`、`visual_student.py`、三个入口脚本（CLI Scripts）、三个新增测试文件（Tests）和 `artifact_record.py` 的写入摘要（Writer Digest）改动。

未读取 final4 私有标签（Private Labels），未读取旧安全或越界审计，未训练、未修改实现代码；未运行全仓历史测试，未扫描原始文件目录、公开权重全集或正式像素缓存的全部内容。没有 `.codegraph/` 索引，代码导航采用定向文本搜索（Targeted Text Search）。

使用完成前验证（verification-before-completion）原则，以直接代码证据和三项临时隔离样例（Isolated Fixture Probes）支撑问题；未重复运行已经报告通过且未变化的测试。

## 重要发现与最小修订（Important Findings and Minimal Fixes）

### E1／P1：教师特征未绑定登记来源（Teacher Feature Provenance Binding）

位置：`src/experiments/visual_student.py:100`–`122`，尤其 `guard_teacher` 对特征的检查。

`guard_teacher` 验证 A1 预测产物与传入的 logits、probabilities、valid，但对 `teacher.features` 只验证形状。`load_teacher` 从固定路径取 `visual_features/refit14/full`，也没有证明该特征引用（ArtifactRef）就是当前 A1 prediction 的直接特征父产物（Direct Feature Parent）。因此标准训练接口可接受同 ID、同 logits、同 valid 的任意有限同形状特征，关系蒸馏（Relation Distillation）随之改变，而新 A2 记录仍只声明原 A1 产物。

独立样例：在真实登记器（ArtifactRegistry）建立 40 行 select A1 prediction，父节点包含正确的 `visual_features`。传入全 1 特征与随机特征，两者均通过 `guard_teacher`；同一学生表示上的关系损失分别为 `0.8369415402` 与 `0.0023132691`。该样例仅使用临时合成文件。

最小修订：从 prediction 的唯一直接 `visual_features` 父引用解析特征来源，按已登记 ID、用户、类别轴重排；在阶段边界一次性绑定实际消费特征与 `TeacherTargets`，或直接由已验证加载器构造目标。拒绝其他特征引用及被替换的传入特征。保持祖先只查元数据（Metadata），不递归读取祖先负载（Payload），不为证明绑定新增正式缓存全量扫描。

### E2／P1：像素恢复会为摘要重读全部已完成行（Pixel Resume Digest Rescan）

位置：`src/experiments/pixel_cache.py:130`–`137`。

无最终 `artifact.json` 时，即使 `completed` 已全部为真，恢复循环仍为每行执行 `np.ascontiguousarray(array[i]).tobytes()`，包括整个 `images` 数组。已完成行没有图像生成或训练消费，该读取仅用于重新计算摘要（Digest）。这正是用户禁止的续跑额外全量缓存扫描；正常产物完成路径不会触发，但生成中断、登记中断会触发。

独立样例：生成 1 条无 IR 的合成像素行后，仅移除临时最终登记标记；重进 `build_pixels`，没有任何需要计算的 trial，仍读取 1 条完整图像行，即 `2,457,600` 字节。正式 refit14 的 2427 行可扩大为约 5.56 GiB 的无计算读取。

最小修订：生成时保存可恢复的逐行或分片摘要凭据（Digest Receipt），恢复只读取小型凭据和未完成行；使用可恢复的登记摘要/分片清单方案完成产物闭合（Artifact Closure），不得为了补最终标记重读全部完成图像。新增读取计数回归（Read-count Regression）覆盖全 completed、缺 artifact 的恢复场景。已有完整正式像素产物保持历史 producer 身份，不为代码修订重建或改写旧摘要。

### E3／P1：阶段重复或并发启动覆写登记模型（Repeated or Concurrent Stage Overwrite）

位置：`src/experiments/visual_student.py:233`–`260` 与 `scripts/run_no_vote_visual_student.py` 的训练入口。

训练入口没有阶段直接输入身份（Stage Input Identity）、完成产物检查或排他锁（Exclusive Lock）。同 phase 再次启动会重新公开初始化并从 epoch1 训练，持续替换固定路径的 `checkpoint.pt`、`history.json` 和 `model.json`。这些路径可能已经被旧 A2 ArtifactRef、预测及 sequence 引用；第一次新 checkpoint 写入即使旧模型记录的文件摘要失效。两个进程还共享 `checkpoint.tmp.pt`，可互相替换或触发文件缺失。当前只有权重 checkpoint，缺 optimizer、scaler、随机数状态（RNG State）和已完成 epoch，不能当作可靠断点恢复（Resume）。

这是直接写入路径及控制流的静态证据（Static Evidence），未为复现而重训正式模型。

最小修订：GPU 初始化前检查阶段输入 identity 与完成标记；完整同 identity 阶段幂等返回（Idempotent Return），不一致拒绝；用阶段锁拒绝并发。在未完成阶段选择明确且安全的策略：保存完整训练状态进行真正 resume，或明确拒绝并要求使用独立的新阶段目录重启，不能静默覆写已登记产物。补充 tiny model 测试，证明重复调用不再次训练、不改已登记 checkpoint，且第二并发调用被拒绝。

### E4／P2：refit 预算未绑定 select 记录（Selection Budget and Metadata Binding）

位置：`src/experiments/visual_student.py:199`–`204`、`219`–`220`。

refit 仅调用 `registry.read(selection.fit_artifact)`，读取操作只保证记录路径与记录文件摘要，未执行协议身份、配置摘要、完整性及角色验证。随后没有把 `selection.budget['epochs']` 与 select A2 记录里的 `config['budget_epochs']` 比较；只要调用者的 `selection.config` 等于当前 recipe、开发 ID 摘要匹配，范围内被改写的预算就可进入训练。因此不能证明 refit 固定使用选中的 epoch。

独立样例：登记 select A2 的 `budget_epochs=3`，构造同 recipe、同开发 ID 摘要但 `epochs=9` 的 Selection；该对象通过 refit 的 selection 检查并到达 `_labels` 入口。样例用入口探针（Entry Probe）停止后续处理，没有训练。

最小修订：验证 select A2 记录完整元数据（Complete Metadata）、本协议身份、模型 recipe、kind/phase/人口及配置摘要；要求预算是合法整数，并严格等于已登记 `budget_epochs`。小型 selection 文件应形成可验证闭合关系。该验证只处理记录与小型配置，不为 refit 初始化读取 select 权重或递归验证祖先 payload。新增改预算及错协议负例（Negative Cases）。

## 已确认的实现与实际证据（Confirmed Implementation and Evidence）

- 登记器对直接 A2 模型和预测文件执行校验（Direct File Validation），祖先只查记录；`load_student` 没有缺少 checkpoint 校验，因为 `registry.verify` 已负责该直接文件。
- 公开 MC3 初始化（Public Initialization）使用本 run 权重凭据里的实际文件摘要，并对 stem/layer1–4 严格加载（Strict Loading）；推理加载 A2 checkpoint 时关闭公开权重初始化。
- 推理 Dataset（Inference Dataset）无 label/teacher 键；预测及序列提取（Sequence Extraction）通过元数据保留祖先边界，没有打开教师或监督标签祖先。
- 训练标签 ID、40 类顺序（Class Order）和合法类别范围有检查；fit/development 的 StageInputs 由现有加载器检查监督标签摘要。没有发现 final4 标签读取入口。
- hybrid 生效项为 CE、温度 2 的 KD、六 clip 关系损失；直接 feature loss 与 stage KD 为零。损失、学习率（Learning Rate）及同步增强（Synchronized Augmentation）移植与已读取的队友对应源码一致。
- select/refit 模型在预测入口由规范完整分区 ID 限制；sequence 的配置显式绑定模型摘要、像素摘要和 anchor phase，缓存重用会比较这些身份。原生时间序列（Native Temporal Sequence）调用源码 `encode_backbone_sequence`，没有复制池化向量。
- `save_array_with_digest` 在写出 sequence 时计算摘要，`register(file_digests=...)` 路径避免 raw cache 登记时重新扫描大数组。E2 是中断恢复的剩余例外。
- `outputs/task5_audit/pixel-continuation.log` 显示 refit14 于 22:36:31、final4 于 22:39:12 完成，随后写出 `PIXEL_STAGES_COMPLETE`。本次仅读取日志，没有对正式图像重新全量校验。
- `outputs/task5_audit/real-batch.json` 记录真实 GPU 4 样本 forward/backward/native sequence 用时 `2.481 s`、峰值显存（Peak CUDA Memory）`2.072508 GiB`，sequence shape 为 `[4,2,3,16,512]`；冻结前缀无梯度、head 有梯度。这是前序真实验收证据，本次没有重复 GPU 运行。它不能证明完整 epoch 时间、长期训练稳定性或资源预算（Resource Budget）。

## 复审门槛（Re-audit Gate）

修复 E1–E4 后，只运行对应新负例、直接受影响接口回归和必要 tiny model 生命周期测试；记录额外 cache 读取次数为零，验证重复调用不改已登记模型。随后独立复审这些差异；通过后才启动本轮授权的 Task5 正式训练。本报告不授权或推进 Task6。

本次三个独立 probe 均退出码 0，成功复现缺口；其数据均在系统临时目录（Temporary Directory）中，运行后删除。正式学生训练尚未启动。本结论是边界内工程审查，不能描述为原始文件或全部缓存内容完整性审计通过（Full Content Integrity Audit）。
