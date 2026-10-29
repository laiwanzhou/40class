# P38 Task10 原融合与独立训练控制（Source Fusion and Independent Controls）

日期：2026-10-29，Asia/Shanghai。分支 `experiment/teammate-single-teacher-task1`。

当前状态：实现及训练前验证通过，正式训练进行中，**尚未完成 Task10**。唯一 [P01 计划（Plan）](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md) 与 [P02 交接（Handoff）](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md) 继续原位维护；本文件是交付记录。

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

正式八组训练、24 份预测、最终自验及完成标记待生成；未执行独立子智能体复审，Task11 未启动。用户要求所有任务完成后统一评估，本步不据开发分数改主链。
