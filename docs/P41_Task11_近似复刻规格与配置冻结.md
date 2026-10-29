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

当前已完成配置冻结和公共输入准备；P12新教师、global／KNN、中间适配、safe decision、完整bank／group／gate／A8仍需逐步实现。Task11未完成，Task12未启动。Task1–10及独立复审继续复用。整体效果留到完整实验结束后统一评估，不用单个开发分数追加候选或改变预算。
