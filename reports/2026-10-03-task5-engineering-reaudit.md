# Task5 独立工程复审（Independent Engineering Re-audit）

复审日期：2026-10-04（沿用约定报告文件日期）。工作树：`_single_visual_processing_replication`；分支：`experiment/teammate-single-teacher-task1`。

结论：**NO-GO**。原 E1、E2、E4 的主体缺口已修订；E3 的常规幂等与 epoch 恢复已实现，但完成发布及失效锁回收仍存在下述重要问题（Important Findings）。正式学生训练（Formal Student Training）应等待这些修订及定向复审。

## 原发现处理情况（Original Findings）

| 原项 | 已审查的修订 | 本轮判断 |
|---|---|---|
| E1 教师特征来源绑定（Teacher Feature Binding） | `guard_teacher` 解析唯一直接 `visual_features` 父引用，加载登记特征、按 ID 重排，并与传入特征精确比较 | 主体修复；直接缓存加载还有下述重复校验问题 |
| E2 像素恢复重扫（Pixel Resume Rescan） | 完整 `write-digests.json` 用于补登记；有 completed 行且无 writer receipt 时明确拒绝并要求新目录 | 重扫缺口修复；有限恢复范围（Limited Resume Scope）可接受，不要求恢复全量 SHA；receipt 文件集检查仍不足 |
| E3 重复/并发覆写与中断恢复（Lifecycle and Resume） | 同 identity 完成阶段幂等返回；PID/create_time 锁；原子 `resume.pt` 保存 model、optimizer、scaler、best/history 及 RNG | 常规路径改善；失效锁竞争与完成发布窗口仍待修订 |
| E4 select/refit 预算绑定（Selection Budget Binding） | 验证 select A2 记录、kind/phase、model_recipe、开发 ID 与已登记 budget_epochs | 原预算错配缺口修复 |

检查范围包括入口脚本（CLI）、本轮新增恢复代码、对应测试及训练状态序列化。没有读取 final4 私有标签（Private Labels）、扫描 raw 数据或正式像素缓存内容，没有训练或修改实现代码。前序 `30 passed` 加完整模拟中断恢复 fixture `1 passed` 是主代理提供的验证证据；本次没有为重复通过结论重跑它们。

## 剩余重要问题（Remaining Important Findings）

### R1／P1：失效锁回收不是原子操作（Non-atomic Stale Lock Reclamation）

位置：`src/experiments/visual_student.py` 的 `phase_lock`。

两个竞争进程可以同时读取同一失效锁（Stale Lock）。A 删除旧锁后创建自己的新锁；B 随后的无条件 `path.unlink()` 会删除 A 的新锁，再创建 B 的锁。`open('x')` 仅保证创建瞬间独占，不能保证失效判定与删除组成原子操作。因此正式训练仍可被同时启动，重新出现 checkpoint、history、临时文件相互覆写风险。

独立探针（Independent Probe）：临时目录中建立失效 PID 锁，以两线程和受控失效进程查询安排上述顺序；两个 contender 在任何释放前均进入锁上下文，输出 `both_entered_before_release=true`、`results` 同时包含 `A`、`B`。清理阶段还可发生 Windows 文件访问竞争。探针没有调用训练。

另一个恢复边界：锁文件创建与 JSON 写完之间存在空文件窗口；其他调用或创建者在该窗口中断会使 `json.loads` 抛错，当前逻辑不能处理该残留锁。不能把 JSON 解码失败当作自动可删锁，否则会扩大竞争问题。

最小修订：使用操作系统级文件锁（OS File Lock）或互斥量（Mutex），持有句柄覆盖整个阶段；让失效判断与回收受同一原子排他机制保护，避免按路径删除新 owner 的锁。添加两个竞争者回收同失效锁的定向测试，不需要真实训练。

### R2／P2：select 部分发布后无法恢复（Partial Completion Publication）

位置：`_train_visual_student` 的最终登记写入及 `train_visual_student` 的完成分支。

内部实现先写 `artifact.json`，随后写 `selection.json`；返回外层后才写 `completion.json`。若在 artifact 发布之后、selection 写入之前中断，重进外层会因 artifact 已存在走完成分支，直接读取不存在的 selection，抛 `FileNotFoundError`。已有 `resume.pt` 不会进入恢复路径。这一情况不同于已经完整完成后单独删除 artifact 的现有测试。

独立入口探针（Entry Probe）：用临时合成 StageInputs 和 mock 在 artifact 写出后停止，第二次调用同 identity；结果是 `FileNotFoundError(selection.json)`，且 selection/completion 均不存在。没有执行模型训练。

最小修订：先形成 selection 与完整 completion 凭据，再原子发布 artifact 完成标记；或将任何不完整发布识别为可恢复状态并利用已保存 resume 完成登记。完成分支须验证 selection 的模型引用与登记预算，不能仅因 artifact 文件存在宣告阶段完整。新增在发布窗口中断的 tiny fixture。

### R3／P2：writer receipt 未检查固定文件集（Writer Receipt File-set Closure）

位置：`src/experiments/pixel_cache.py` 的 `digest_receipt.exists()` 恢复分支。

恢复仅检查 identity，将 receipt 中任意 `files` 映射直接传入登记器。登记器只能保证当前传入路径与摘要映射一致，不能知道 pixels schema 的全部必需文件。因此缺 `images.npy` 等条目的 receipt 仍会被发布为产物。

独立小样例（Small Fixture）：生成 1 条合成像素行，删最终 artifact 标记并从临时 receipt 移除 `images.npy`；恢复成功登记，随后 `load_pixels(..., complete=False)` 抛 `KeyError('images.npy')`。该例未重新读取正式图像或运行全量哈希。

最小修订：恢复时精确校验当前 output 下 `PIXEL_KEYS` 的六个 `.npy` 加 `rows.csv` 七个路径，以及摘要格式、完成 population/identity。可读必要数组头（Array Headers）与小型完成元数据，不扫描图像负载。receipt 缺项、多项或路径漂移应在发布前明确拒绝。

### R4／P2：同一阶段重复验证直接缓存及模型（Duplicate Direct-input Validation）

位置：`load_teacher`、`guard_teacher`、外层 `train_visual_student` 与内部 `_train_visual_student`。

正式 CLI 先调用 `load_teacher`，其中 `_load_features` 对整份直接 features 文件 SHA 校验并加载；进入训练后 `guard_teacher` 再次 `_load_features` 同一文件。refit 的 select 模型也被外层与内部 `registry.verify` 各验证一次，后者会打开直接 checkpoint 文件。虽然没有沿祖先读取 raw payload，这仍不符合计划的“同阶段调用去重、直接输入边界检查一次”。E1 的新绑定不应恢复额外的缓存内容读取。

最小修订：阶段内共享已验证直接输入/加载上下文（Validation and Load Context），明确记录实际 feature 父引用及绑定凭据；保留任意替换 `teacher.features` 必须拒绝的负例，同时避免再次从磁盘 SHA/加载同一完整 feature 缓存。select 记录验证同一次阶段调用只执行一次。用小型读取探针验证调用数，不用真实大缓存计时或全量扫描来证明去重。

## 恢复设计已确认部分（Confirmed Resume Design）

- `resume.pt` 通过临时文件替换（Atomic Replace）提交，保存当前 model、optimizer、scaler、best_state、epoch/history，以及 Torch/Python/NumPy/CUDA 随机数状态（RNG State）。
- 恢复使用 `.cpu()` 还原 CPU RNG 与 NumPy 状态，CUDA RNG 显式转回 CPU 后调用恢复接口，未发现 map_location 导致 RNG 设备错配。
- epoch 边界恢复保留最佳 select 权重，refit 不用 select 权重初始化；余下 epoch 使用原预算和学习率日程（Learning-rate Schedule）。中断的未提交 epoch 会重做，这是合理的边界恢复策略。
- 原 singleton 尾批规则（Drop Singleton Batch）已复用；不把这一既有行为作为新增方法学变更。
- 完成旧正式 pixels 仍由 `load_pixels` 直接消费，历史 producer SHA 已归档，未为修订重建数据。无完整 writer receipt 的部分像素拒绝自动恢复是明确的功能限制（Explicit Limitation），不要求用全量扫描消除该限制。

## 本轮证据与后续门槛（Evidence and Gate）

三个新独立 probe 均只使用临时目录与合成数据，退出码 0；分别复现锁竞争、部分完成发布及 receipt 缺文件。上述 R4 是直接调用链（Call-chain）证据，没有用正式缓存进行额外扫描。

修复 R1–R4 后，仅运行这些边界的定向回归（Targeted Regression）并再次独立复审。原完整训练模拟已通过且未受影响的部分不要求重新广泛测试。本轮仍只推进 Task5，不进入 Task6；当前 NO-GO 不等于否定已有真实 GPU 批次及正式像素生成结果。
