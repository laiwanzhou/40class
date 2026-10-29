# P39 Task9／Task10 独立复审（Independent Re-review）

日期：2026-10-29，Asia/Shanghai。工作树 `D:/work/2026.7.14_kaggle/_single_visual_processing_replication`，分支 `experiment/teammate-single-teacher-task1`。

**结论：Task9 PASS，Task10 PASS，未发现新增可复现的 P1／P2 问题。可以推进 Task11 代码实现与前置证据闭合；完整教师库／A8 正式生成仍受 Task11 自身材料门槛阻断。** 用户已纠正后续为 Task11，不是 Task12，本次不推进 Task12。

复审范围 `git diff 2f43593...4f21daf`，包含 Task9 `4e148cb`、Task10 实现 `1e9648f`、序列化修复 `8dbd621`、收尾 `4f21daf`。唯一指导为 [P01（Plan）](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，实施记录为 [P37](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P37_Task9_额外平均融合诊断.md)／[P38](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P38_Task10_原融合与独立训练控制.md)。三名独立子智能体均已完成审查，未把待审状态当作通过。

## 工程轴（Engineering／Standards）：PASS

审查者 `p39_engineering_review` 独立运行两个新增测试文件：20 passed，24.35 秒，三个源 Transformer 警告；另运行旧 Task8 固定24轮恢复专项：1 passed，6.67 秒。没有跑旧全仓测试或 GPU。

共享 `_save_torch` 只递归解冻配置，保留 Tensor、OrderedDict、`_metadata` 及原冻结输入。源 AST 索引兼容修复与独立展平参考一致；装配／RNG、损失、冻结参数但保持运动 Dropout、时间对齐通过。阶段 A 保存模型和四类 RNG，恢复从 A4 末态重跑 B20；不是 B 阶段逐轮恢复。

独立核验八模型 metadata、四组每 phase 相同初始化摘要及 24 轮 ID 顺序摘要、1957／2342 fit 人口、同 phase A2／新版 A5 祖先、原生 sequence／pixels／时间／anchor 绑定。shuffle 只置换 fit 行的相应模态与 mask，推理恢复原始输入；mask-motion 与 zero 控制角色明确，诊断祖先不得回流主链。

非阻塞代码味判断：`FusionDataset` 构造器传递多组数组与索引，存在轻度数据泥团（Data Clumps），`dataset_for` 已集中桥接。目前不影响正确性，不为此重构或重训。

## 规格轴（Specification／Methodology）：PASS

审查者 `p39_methodology_review` 独立对照生产实现、必要源函数与现存验收证据。

- Task9 温度仅在各 phase 有效 fit 行拟合，范围 0.25–4；identity／fitted 两种整体策略仅按 development2 A6-VSI 决胜，平局取 identity。refit 固定策略后重新估温，不继承 select 温度值，不称 OOF。
- Task10 直接调用源 `train_stage`／`losses`，源最终命令预算确为 4＋20；同 phase A2／`ir_supervision_v2` A5 初始化，四控制独立、预算固定。
- quality 与 anchor 消费为 FP16→FP32；这不意味着 Task5 旧精度残余已重训消除。主链 CE／KD／motion_aux／anchor／reliability 保留源语义，未增加通用缺模态门控。冻结 motion 参数仍保持源训练 Dropout。
- 源三维布尔索引改写具有独立展平参考的前向／真实训练梯度证据，旧快照不改。源批次含恢复的 IR-only 行；A／B 损失、梯度、更新差 0，正式模型实际推理证据闭合。

以上结论限于 Task9／10 及明确适配，不扩大成完整历史流水线或所有环境逐位等价。

## 正式产物轴（Formal Artifacts）：PASS

审查者 `p39_artifact_review` 独立核验八模型记录及 24 份 A7＋12 份 A6 小预测，共 18,886,355 字节：完整 2039／388／2427／609 ID 与 user、40 列顺序、valid／prior、同 phase 祖先、摘要和无标签成员通过。A4＋B20 共 192 轮；fit 与 CW 人口摘要、fit-only shuffle 摘要、两阶段相同初始化／顺序一致。首次与恢复 full/select 的 24 轮历史删除 seconds 后严格相等。

独立用 SciPy FP64 NLL 优化复算六个温度，记录值 NLL 距独立最优不超过 `2e-8`。select 有效 V／S／I＝1957／1956／1914，refit＝2342／2341／2283。独立 ID 对齐均值复算两策略：identity 191／388，fitted 196／388；accuracy／macro-F1／worst-user 均严格匹配。Task10 六开发正确数为 210／181／195／198／187／205，与记录一致。

Task9 原 complete 中 registry／roles 的证据摘要与**历史提交 `4e148cb` 的原始 Git blob**严格相等；其后 Task10 正当修改了共享校验器，当前文件不同不代表 Task9 模型／预测失效。保留原 complete 的历史身份，不改 SHA 冒充当前文件；新的 P39 审查标记显式记录历史提交及当前兼容性。

本轮 checkpoint 载荷读取 0、GPU／训练 0、大 sequence／feature／pixel／raw 扫描或哈希 0、final 私有标签／分布／类别覆盖读取 0。真实 GPU 对照属于 P38 已存证据，本轮只核验，不重复执行。

## Task11 前置问题（Prerequisite Findings）：正式 A8 仍阻断

这些属于 Task11 来源闭合问题，**不是 Task9／10 新回归**，不据此作废前十步或要求重训。

1. **P39-M1，P1：safe-base 不是当前 A7 概率，缺中间 producer。** `p165_deployable_group_teacher.py:223` 读取 P89 的 `base_probability`，其实际来源 `p89_build_final_test_submissions.py:302` 是 P87S 最终学生 `student_logits.npy`。`run_p87s_final_pipeline.py:285` 表明该学生先完成 12 轮无标签适配，旧目标是 `build_p87s_test_structured_targets.py` 由 P85 KNN emission、三 outer-fold decoder 摘要中位配置、fit transition、beam posterior 和 entropy backoff 生成。它不是当前原始 A7，也不是最终 P310 A8。必须先闭合这条 phase 通用中间 producer；不能以 A7 替代，也不能用后续 Task12 A9-12 倒填制造循环。
2. **P39-M2，P2：A18 的旧 roster 分类不准确。** `a18_full_teacher_data.py:18` 导入并消费 `p100a_global_teacher_data` 中的 VMAE／IV2，后者实际路径为 VideoMAEv2 distilled 与 InternVideo2。A18 不是独立非视觉小专家；其 JSON 配置已经打包，但不解除单大教师边界。Task11 应新增版本化谱系闭合证据（Versioned Lineage Closure），显式排除 A18，保留 Task1–10 原 roster／descriptor／protocol identity；不能改旧摘要追认，也不能以 A5 凑列。
3. **P39-M3，P1：P12 所选配置／初始化 receipt／Thermal 输入未闭合。** Skeleton／Depth 候选配置存在，但没有证据确定哪组对应 `p11_fp16_oof`。Thermal runner 绑定 `resnet18_tsm_imagenet_12f.json` 到 `p11_thermal_imagenet`，尚未闭合到 `_fp16` 所选 checkpoint／转换链。源初始化 enum 可知为 `ResNet18_Weights.IMAGENET1K_V1`，缺的是实际选用配置与版本化 receipt 绑定。当前已有 Depth 公共路径，Thermal 仍缺显式公共 sidecar，不能将缺表视为全不可用。

允许先做显式会话适配（Session Adapter）、纯 source 算子与小 fixture 对照、锐化（Sharpening）及缺 producer 拒绝等工作。生成正式 A8 前，上述生产链与所有保留专家必须闭合。不得凭 Task9／10 PASS 直接翻转全局 `formal_generation_ready`。
