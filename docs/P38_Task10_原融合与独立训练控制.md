# P38 Task10 原融合与独立训练控制（Source Fusion and Independent Controls）

日期：2026-10-29，Asia/Shanghai。分支 `experiment/teammate-single-teacher-task1`。

当前状态：**Task10 已完成八组固定预算训练、四分区 24 份预测及有限自验（Bounded Self-acceptance）**。本轮未执行独立子智能体复审（Independent Subagent Review），不称独立 GO；Task11 未启动。唯一 [P01 计划（Plan）](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md) 与 [P02 交接（Handoff）](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md) 继续原位维护；本文件是交付记录。

## 方法（Method）

full 主模型采用队友原 `P86SeparateMotionEncoder`、additive residual 和 clip 融合，直接调用原 `train_stage`。select 从当前 select A2／修正版 `ir_supervision_v2` A5 初始化；refit 从自己的 refit A2／A5 初始化，不继承 select 融合模型。阶段 A 固定 4 轮、阶段 B 固定 20 轮，不按 user6/7 成绩选择 epoch。

四种独立训练为 full、mask_motion、shuffle_s、shuffle_i，对应 A7、A7-mask、A7-shuffle-S、A7-shuffle-I。每阶段全部使用相同初始化、原批次顺序和预算；实际记录初始化头部摘要与每轮 ID 顺序摘要并核对。mask-motion 将运动输入及其 mask 置零，保留原 motion_aux 等 loss，不把门控改写回主模型。shuffle 只在本阶段有效 fit 人口内固定打乱相应模态全部字段及 mask，标签／视觉／其他模态不动；验证与 final 不打乱。另在 full 的同一 checkpoint 上导出 zero-S／zero-I 推理敏感性（Inference Sensitivity），不再训练两个模型。

训练人口在控制干预前由有效 IR／Skeleton／IMU 的并集确定，保留 IR-only 监督；各控制共享人口和类别权重（Class Weights），仅原始全无有效输入行不 fit。完整预测人口不删除，原始全缺失行使用同 phase 完整 fit 类别先验（Class Prior）。预测不加载标签／教师载荷。

源 `train_stage` 负责参数组、冻结、训练模式、AdamW、cosine、AMP、梯度裁剪和全部损失。运动编码器参数冻结但保持源训练时 Dropout；视觉 backbone 与 BatchNorm 按源冻结；同 phase A2 anchor 固定。CE／KD／motion_aux／reliability 不增加通用 availability 门控。

## 明确的适配（Explicit Adaptations）

1. 固定用户划分（Fixed User Split）和单大教师（Single Large Teacher）仍按 P01；不加载旧 OOF 或官方测试预测。
2. 当前质量值及 anchor 缓存为 FP32，原融合消费者的磁盘缓存为 FP16。Task10 在消费时显式 FP16 舍入后转 FP32，匹配源缓存语义；原 Task5 文件、模型和完成记录不改。Task5 已存在的质量精度残余仍属于前置历史，不能据本步声称前置已重训修复。
3. PyTorch 2.7 对源 residual 的 `safe_mask[empty,0]=True` 三维布尔索引报错。独立内存子类只把这一句改为首列与 empty 做 OR；队友文件和原模块类不改。对照显式展平行索引的独立参考，前向和真实训练梯度／更新一致。这是索引兼容修复（Indexing Compatibility Fix），不是新增模型或损失。
4. 阶段级恢复（Stage-level Recovery）：阶段 A 完成保存模型与 Python／NumPy／Torch／CUDA RNG；中断于 B 时从已完成 A 末态重跑 B，不把部分 B 当作已完成，不声称逐轮恢复。模型目录 identity 防止配置混用。

首次 full/select 训练算完两阶段后，在保存最终 checkpoint 时因嵌套冻结配置 `mappingproxy` 不能 pickle 而退出，未形成正式模型登记。新增序列化回归先 RED 复现；仅修复共享 `_save_torch` 的元数据解冻（Metadata Thawing），保持 tensor 叶子、state_dict 的 OrderedDict 类型及 `_metadata`，不改变原冻结配置或既有 checkpoint 文件。修复后 42 项针对回归通过，43.74 秒。主融合 producer 文件及训练 identity 未改；保留第一次错误日志和历史副本，从已保存 A4/RNG 重新运行 B20，续跑日志为 `P38-task10-formal-resume.log`，不能把首次丢失的最终参数当作可复用正式模型。

## 训练前证据（Pre-training Evidence）

五项核心测试先因 Task10 模块缺失全部 RED；布尔索引兼容测试另 RED；控制祖先不准回流主链测试先 RED 后修复。最终 41 项受影响测试通过，43.21 秒，保留三个源 Transformer norm_first 库警告。真实输入桥接夹具修正了源构造器原本应做的教师 FP32 转换，没有改生产目标。

真实 train12 64 条含 IR-only 1 条：所有消费 Dataset 字段、时间增强、源损失、参与梯度及参数更新差为 0；A 阶段可训练参数 3,709,515，B 阶段 10,186,872；冻结 backbone／BN、运动 Dropout 行为一致。源 B 阶段新 GradScaler 的首步出现源参考同样的初始 AMP overflow；验证按原 scaler 自动恢复，B 同一真实批次重复四步后参与梯度全部有限、更新一致，未更改正式 scaler 或损失。mask-motion 全零输入仍有有限 loss 且 motion_aux 大于 0。

真实对照耗时 17.40 秒，峰值显存 596.44 MiB。该参考夹具不消费的 early／late／delta 额外教师字段只填占位以执行源 Dataset，未将其当成真实监督或祖先；本配方消费的全部字段均对照实际新 run 数据。缓存只读取 header／ID／mask／小元数据及训练所需切片，不扫原始目录、images 或完整大 feature／sequence 的 SHA。

运行根：`outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops`，协议身份仍 `2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`。日志及证据在 `outputs/task5_audit/P38-task10-*.log/json`。全局 formal_generation_ready=false 不改，A7 获本任务自己的局部就绪（Scoped Readiness），不放行 P12／完整专家 bank。

## 正式结果与最终验收（Formal Results and Final Acceptance）

八个模型都完成 A4＋B20，共保留 192 轮，select 每轮 1957 条、refit 每轮 2342 条。每 phase 三个控制与 full 的实际初始化摘要、全部 24 轮 ID 顺序摘要一致，类别权重人口一致。refit 重新装载本 phase A2／新版 A5；四组之间也独立初始化，不以 full 的训练后权重初始化控制。主链参数未根据控制成绩变更。

| 候选（Candidate） | development2 正确数／388 | 准确率（Accuracy） |
|---|---:|---:|
| A7 full | 210 | 54.12% |
| A7-mask | 181 | 46.65% |
| A7-shuffle-S | 195 | 50.26% |
| A7-shuffle-I | 198 | 51.03% |
| A7-zero-S | 187 | 48.20% |
| A7-zero-I | 205 | 52.84% |

full 的 train12 为 1980／2039＝97.11%，refit14 为 2353／2427＝96.95%。refit14 包含 user6/7，不能称验证成绩。表中开发分数只作固定端点记录，不挑模型、追加预算或作最终效果结论。所有任务结束后再统一评估。

六候选都生成完整四分区输出：train12 2039／有效1957／prior82；development2 388／385／3；refit14 2427／2342／85；final4 609／591／18。共 24 份标准 NPZ，总计 9,287,337 字节（约 8.86 MiB），没有标签成员。final4 不读取真值、不评分。

保留历史中的八组训练时间合计 1400.40 秒（23.34 分钟）；续跑训练＋导出流程 1589.0 秒（26.48 分钟）。首次保存失败造成额外 B20 重算成本，不计入已保留模型的 192 轮预算，也未把旧 B20 参数连续接到新 B20。旧／新 full/select 的 24 轮历史除耗时外逐项一致，证明本次阶段恢复没有改变其训练轨迹。

最终自验 57.48 秒：核对八模型 metadata／两阶段固定学习率与预算／同 phase 祖先／人口与 CW／配对初始化与顺序／fit-only shuffle；24 份小预测的完整 ID、40 列、valid、prior、来源与允许的开发结果。实际正式 refit/full checkpoint 在 64 条 final4 公共输入上，对 full／zero-S／zero-I 三项 logits 的复算差均为 0。教师目标／标签打开尝试 0，A2／A5／public initializer 参数装载尝试 0。

此推理探针仍运行在完整 provenance 协议中，来源校验会核对 A2／A5 初始化 checkpoint 的摘要；不能外推为把所有祖先文件物理移走仍可运行。实际前向只装载自己的 A7 权重和公共缓存，不读取教师目标或标签。没有原始目录／images／大 feature／sequence 整表哈希、额外重训或 final 标签读取。验证器检查当前实际消费的小输出与真实批次，不声称未抽查缓存载荷全部完整。

最终针对回归 42 passed，43.74 秒；另旧 Task8 固定24轮／中断恢复引擎专项 1 passed，9.36 秒，确认共享保存修复没有破坏该使用路径。源数值对照证据保留训练前记录（其损失／优化逻辑未变），未为了文档收尾重复真实训练验证。

当前入口 `run/protocol/task10_complete.json`，绑定 `task10_artifacts_ready.json`、八模型／24 输出及本交付、测试、真实批次、最终验收和恢复证据。各模型／目标路径为 `run/{A7,A7-mask,A7-shuffle-S,A7-shuffle-I,A7-zero-S,A7-zero-I}/{select,refit}`，zero-S／I 目录只有推理产物，引用同一个 full checkpoint。阶段 A resume 作为恢复中间产物保留。PID45888／session18327 已结束，最终验收 session59416 已结束，无后台训练或自动化。

实现提交 `1e9648f`、序列化修复及验收脚本提交 `8dbd621` 已推送；最终收尾提交号见 Git 日志／台账。原数据、队友快照、旧 Task1–9 产物和两份未跟踪历史报告保持原样。全局 P12／Thermal／P89／A18 门槛与 Task5 已记录的精度残余仍在；Task11 尚未推进。
