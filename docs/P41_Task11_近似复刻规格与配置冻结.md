# P41 Task11 近似复刻规格与配置冻结（Approximate Reconstruction Contract）

日期：2026-10-29，Asia/Shanghai。用户已明确同意“可以改为近似复刻”，并要求继续工作。

当前路线改为**源码约束下的近似复刻（Source-constrained Approximate Reconstruction）**。历史未保存的所选配置和中间 decoder 记录，不再要求找回后才能实现；改为预先固定、显式标注的重建配置。可核实的源算子、现有 Task1–10、固定用户边界和单大视觉根约束继续保留。不得把重建配置称为队友原始选中配置，不按 final 分数调整。

本文件是规格与决策记录，唯一执行计划仍为 [P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接仍为 [P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。旧 [P39](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P39_Task9_Task10_独立复审.md)／[P40](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P40_Task11_来源闭合与会话适配.md) 保持审计当时结论；本次用户授权解除的是“必须恢复历史记录”这一条件，不代表新生产链已经完成。

## 当前配置（Active Recipe）

机器可读配置为 `configs/experiments/task11_reconstruction_v2.json`；父协议身份仍为 `2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`，旧 YAML／roster／模型身份不改。配置另有 SHA256，新增缓存、教师、目标均应绑定这个重建版本及各自 phase 的真实父记录。v1 准备尝试因无公共路径行未完成，只是退役草稿，不是活动配置或训练来源。

| 项目（Item） | 固定重建选择（Frozen Choice） | 依据与差异（Basis／Difference） |
|---|---|---|
| Skeleton | 原 `configs/skeleton_only.json`、原 `AlignedMultimodalModel` 的 Skeleton 分支 | 配置及模型确实存在；无法证明其就是 p11_fp16 当时所选版本 |
| Depth | 原 `configs/depth_layer3_spatial.json`，jet RGB、layer3 spatial、随机初始化 | `residual_logit_fusion.py` 的默认 depth checkpoint 指向该同名路线；不擅自选 ImageNet 变体 |
| Thermal | 原 `thermal_baseline/configs/resnet18_tsm_imagenet_12f.json`、原 ResNet18-TSM、ImageNet1K V1 | runner 明确引用该配置，但与旧 `_fp16` 所选 checkpoint 的绑定缺失，接受为重建 |
| 共同训练预算 | 12帧、144×192、batch32、15轮固定末轮、seed20260720、AdamW LR0.0007／wd0.0002、smoothing0.1／dropout0.3、balanced sampling、原 cosine／AMP／clip5 | 来自上述现存源配置；从 best-validation checkpoint 改为固定第15轮，避免用开发或封存标签补历史选择 |
| FP16 导出 | 保留 FP32 master，另存新 FP16 副本 | 重建 source 消费格式；不声称找回旧转换命令或逐位相同权重 |
| 中间 decoder | gap30、transition0.30、trigram backoff1、beam50、posterior T1，原 entropy backoff | 使用源码已有固定参数，替代无法恢复的三外折摘要中位数；不重做内部OOF／不按final选参 |
| 中间 source-base | P85 global mixture→KNN→原中间结构目标→独立12轮适配→raw学生概率 | 保留这一步，不以原 A7 或最终 A9-12冒充；使用原12轮 heads 配方 |
| 中间适配人口 | select 仅 train12，refit 仅 refit14；development2／final4 不参与本中间阶段拟合 | 本次固定划分保守适配。**与原 source-base 在旧无标签输入池适配不同**，单独声明；Task12 的 final4 适配仍按原计划执行 |
| safe probability | `normalise(.95×重建中间adapt12 raw + .05×softmax(RF terminal／3))` | 保留已知表达式，来源是新中间模型，不是旧预测或裸 A7 |
| safe decision／joint grouping | 原 joint decode，evidence0.25、transition scale1；固定 P89 `GLOBAL_REPEAT` 配置 | rank3、gap300、similarity0.84、overlap0.2、length ratio0.8、consensus0.75、alignment gap0.2、group size3，替代缺失的 H1 所选摘要 |
| A18／P128 | 排除 | 已核实实际依赖被删 VideoMAEv2／InternVideo2 等根，不能作为小教师保留或用 A5替代 |

已确认的 P12 八分量顺序、global simplex／三邻居0.4、分组／sequence gate／P310 priority 和最终0.9805／0.0005锐化继续按源顺序实现。当前重建视图保留20个专家列，其减少是根节点排除的自然结果，不复制列恢复原数量。完整 bank 的允许列仍需与真实 producer 逐一匹配。

## Thermal 未知覆盖（Unknown Coverage）

只能由现有公共模态路径找到同一 trial 的 Thermal 同级目录，不反解 opaque ID、不读私有标签／旧 ID映射来寻找全路径缺失行。路径不一致或用户不符仍拒绝。

准备脚本分别输出：标准可消费 sidecar（只有可用输入记1）和 observation sidecar（保留 `unknown`）。无法定位行的标准可用 mask 为0表示“本实验没有可消费输入”，**不是证明其物理模态不存在**；完整行数不变，预测按既定缺失策略保留。

| 分区（Partition） | 完整行数 | 已找到文件名 | 已知无文件 | 无法定位／未知 |
|---|---:|---:|---:|---:|
| train12 | 2039 | 1840 | 117 | 82 |
| development2 | 388 | 374 | 11 | 3 |
| refit14 | 2427 | 2214 | 128 | 85 |
| final4 | 609 | 574 | 17 | 18 |

这些只是公共路径／文件名发现结果，尚不证明图像全部可解码或可用于训练，不是类别覆盖或准确率。最终18行全缺失策略维持当前分母；发现新的可用输入也不得根据标签选择性改分母。

## 验证与状态（Verification and Status）

公共输入适配新增两个测试先 RED，基本实现后 GREEN；真实准备发现所有模态路径为空的行，新增未知覆盖用例先 RED 后修正。新版配置和侧表准备已成功，10.27秒；没有解码原图、整目录哈希或读取 final 标签。receipt 的时间仅打印到日志，不混入不可变内容，避免相同配置重复运行因耗时变化冲突。

准备凭据在 `run/protocol/task11_reconstruction_v2/preparation.json`，sidecars 在其 `thermal_inputs/`。旧 Task11 严格预检继续按历史契约阻断，不能修改其结论；新的重建流程须验证自己的配置、公开 initializer receipt、真实缓存与同 phase 模型再放行，不能只翻旧 `formal_generation_ready`。

当前已完成配置冻结、公共输入准备、Skeleton 两阶段15轮及四分区FP16输出；Depth／Thermal正式训练已启动。Task11未完成，Task12未启动。Task1–10及独立复审继续复用。整体效果留到完整实验结束后统一评估，不用单个开发分数追加候选或改变预算。

## 教师实现与进行中训练（Teacher Implementation and Running Training）

Skeleton 采用原 IR／Depth／Skeleton 共同帧与原首人姿态处理（Common-frame Alignment / First-person Pose）。select有效1939条，refit有效2324条；这与Task8的运动缓存人口不同，不能互相冒充。原平衡采样（Balanced Sampling）和 `drop_last=True` 保留，每轮实际采样1920／2304条。select／refit各15轮，训练累计52.28／44.77秒；完整人口输出仍为2039／388／2427／609，有效1939／385／2324／590，其余100／3／103／19条用同phase先验。未删行，也未读取final标签。

FP32主模型（Master Checkpoint）及旧输出保持不变，另存FP16副本；所有浮点张量恰为主模型half结果，整数计数保留，执行时恢复FP32参数并按源AMP。入口 `run/protocol/task11_reconstruction_v2/p12_skeleton_current.json` 指向 `skeleton_fp16_component.json`。有限验收核对两历史、四份小预测及先验，不重复扫描原始数据或大缓存。主producer SHA与Git `69eed6d` 的源码一致；后续适配层扩展不追改历史来源身份。

Depth／Thermal输入适配调用原采样、增强、归一化和模型类（Source Dataset / Model）。预测接口仅用AST移除原输出字典中的一个 `sample.class_id` 读取，不构造伪标签；Depth按源缓存的PIL双线性resize，Thermal保留原仿射与翻转顺序。Thermal公开ResNet18 V1已登记完整SHA与官方来源，按跨盘复制到独立run，未改C盘缓存；初始化与原 `imagenet_pretrained=True` 模型逐参数相同。

两个真实32条train12批次，输入源差0，更新前梯度全部有限、参数实际改变、AMP跳步0，峰值约2819／2810 MiB。最初验收误把源函数更新后清空梯度解释成失败，现已改为优化器更新前hook核验；旧失败日志保留，未据此更改训练配方。24项针对性测试30.66秒通过，含连续15轮与第4轮中断恢复的模型／AdamW／scheduler／scaler／全部RNG／sampler顺序及历史等价；小夹具保留30条sklearn预测类别warning。真实批次验收10.82秒，无开发或封存标签。

有效工程覆盖（Engineering Overrides）：Windows使用workers0，CPU torch threads4；CUDA保留源high matmul precision和cuDNN benchmark。无训练轮数、批量32、增强或初始化的额外选择。每完整epoch保存恢复点，崩溃恢复会重跑未完成轮，已完成轮不增加预算。

正式训练启动于2026-10-29 20:35（Asia/Shanghai），PID47268，producer `89c1930`。先Depth select15／refit15及四区FP16推理，再Thermal相同步骤。日志 `outputs/task5_audit/P41-p12-images-formal.log`，stderr同名 `-error.log`，PID文件 `P41-p12-images-formal.pid`。已实际完成Depth select第1轮，347.69秒；尚不能据第一轮精确预测整个流程耗时。不得在进程尚活着时重复启动，不能把第一轮完成当教师或Task11完成。

P12校准／融合纯算子（Calibration / Fusion Operators）已补入 `p12_fusion.py`：S+D的0.6／0.4、IMU温度及0–0.5十一格、设备数／5权重、Thermal 81格温度／21格权重及accuracy／macro-F1／较低权重决胜、89维路由、global simplex、原self-excluded三邻居0.4。4项源对照测试24.34秒通过；尚无正式拟合参数或全量组件表，不把纯算子测试称为完整producer闭合。固定划分若导致路由敏感行只有一个目标类别，复用原 `route_cross_fitted` 的常数回退（Constant Fallback），不加伪样本；这区别于源最终全量router遇此情况直接拒绝的分支，必须在实际触发时记录，仍不新增OOF。

后续仍需正式P12八分量、global／KNN、中间独立适配、safe decision、完整bank／group／gate／sequence／A8。当前状态见 `run/protocol/task11_reconstruction_v2/progress.json`，它明确 `task11_complete=false`，无Task12启动或独立复审GO。

## 用户暂停与恢复点（User Pause and Resume Point）

**用户要求暂停（Paused by User，2026-10-29 20:47）：** 已停止核验身份后的Depth／Thermal训练进程PID47268，当前无此后台训练；不要自动恢复。Depth select已完成2／15轮，`run/task11_reconstruction_v2/p12_depth/select/resume.pt`已实际用CPU装载核验：identity相符，history连续1、2，模型／AdamW／scheduler／scaler／sampler及全部RNG齐，scheduler.last_epoch=2。恢复从第3轮开始；当前未完整轮的进度舍弃，不增加预算。Skeleton两阶段及FP16四区产物已完成并保留；Depth refit与Thermal尚未启动，Task11未完成、Task12未启动。暂停凭据 `run/protocol/task11_reconstruction_v2/pause.json`，总progress已标paused_by_user。恢复时先看进程，再使用原producer89c1930对应的同脚本与配置；日志改用新的resume日志，保留原formal日志。当前分支后续提交2f94ebe只增加纯融合算子与文档，未改运行中的image producer，因此不影响本恢复点。等待用户明确继续。

## 用户恢复（User Resume，2026-10-29）

用户现已明确继续，下方／上方暂停只是历史；当前worker5632从第3轮恢复并完成，接续supervisor42704等待其完成后生成P12/global/KNN与fit-pool结构化目标。两phase转移模型已fit，不等于中间学生12轮适配。P41冻结配置不变，详细实施与有限验收见[P42](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P42_Task11_断点恢复与目标接续.md)，最新状态以唯一P02与revision progress为准。Task11未完成，Task12未启动。
