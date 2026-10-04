# 视觉结构与蒸馏目标对照（Visual Architecture and Distillation Target Comparison）

日期：2026-10-04，Asia/Shanghai。当前分支：`experiment/teammate-single-teacher-task1`。本次为只读诊断（Read-only Diagnosis），不修改训练源码、冻结协议、A1/A2模型或提交包，不重训，不读取final4私有标签。

当前结果是A1教师开发准确率69.07%、A2学生48.71%，差20.36个百分点。有效训练样本推理准确率分别98.47%、99.69%。学生已能拟合训练样本；仅凭训练/开发差距不能确定单一原因，不能保证换一个组件就恢复这些百分点。

## 1. 最终视觉学生结构一致（Final Visual Student Architecture Parity）

当前`initialize_student`直接加载队友`p86_mc3_visual_model.P86MC3VisualStudent`，没有重写MC3网络。以队友最终视觉训练记录（Final Visual Training Record）为比较对象，以下设置一致：

| 项目（Item） | 队友最终视觉学生 | 当前Task5 |
|---|---|---|
| 骨干（Backbone） | MC3-18，Kinetics400公开初始化 | 同骨干与公开权重 |
| 输入（Input） | IR，早/晚两窗，场景/人物/工作区三视野 | 相同 |
| 采样（Sampling） | 每窗16帧，160分辨率 | 相同 |
| 时间建模（Temporal Modeling） | 时间位置参数、单层8头Transformer、时间统计融合 | 直接使用同一个源模型类 |
| 视野融合（View Fusion） | 六clip Transformer、质量门控、早晚窗口融合 | 相同 |
| 冻结（Freezing） | 冻结到layer2，BatchNorm统计冻结 | 相同 |
| 优化（Optimization） | AdamW，head LR2e-4、backbone LR1e-5、wd0.08、batch4/accum4 | 相同 |
| 增强（Augmentation） | subject_robust，同步水平翻转/平移/时间扰动/亮度变换 | 直接复用同一增强函数 |
| 损失（Loss） | hybrid CE/KD/relation；直接feature、stage KD为0 | 相同生效损失类型与权重 |
| 参数量（Parameter Count） | 17,967,597 | 17,967,597 |

本次实际CPU结构探针使用`kinetics_pretrained=False`避免下载，再与当前select检查点逐键比较：全部state keys和tensor shapes匹配，参数量与队友记录一致。该探针只检查结构，不声称重新验证公开权重字节；公开初始化已在此前Task5训练/验收中核对。证据：`outputs/task5_audit/visual-architecture-parity.json`。

当前exact_time_modeling、cross_view_time_modeling、spatial_region_modeling、region_temporal_modeling、structured_region_modeling全为false；队友最终训练记录中的对应开关同样关闭。因此不能把这些候选模块称为本轮漏掉的最终模型组件。

源码参考：[队友MC3模型](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/p86_mc3_visual_model.py:20)、[当前初始化](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student.py:33)、[队友最终训练记录](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission_20260911/docs/provenance/visual_student_training_record.json)。

## 2. 教师软目标有实质差异（Material Teacher-target Differences）

| 项目（Item） | 队友P86/P87S视觉训练路径 | 当前A1→A2 |
|---|---|---|
| 类别目标特征族（Target Feature Family） | Dataset固定读取`early_late_logits`，保留早晚拼接 | 72候选选择`window_mean`头，先平均早晚窗口 |
| 目标来源（Target Origin） | 三折OOF预测，按用户排除对应held fold拟合 | 训练目标来自在同一train12人口拟合的单头预测 |
| 教师校准（Teacher Calibration） | `fit_temperature(raw_oof, labels)`，保存`raw_oof / temperature` | head_temperature固定1，保存原始Ridge decision scores |
| 学生KD温度（KD Temperature） | 对已校准logits再除2 | 对未校准Ridge scores再除2 |

“KD温度都是2”并不意味着目标相同。队友实际传入学生的是`softmax(raw_oof / (teacher_temperature * 2))`；当前是`softmax(raw_scores / 2)`。原`fit_temperature`搜索范围为约0.1–10；现有包没有对应原head的实际温度与OOF结果文件，因此不能推断其具体值或造成的收益。

本次用当前A1小型targets直接计算（不拟合任何新参数）：

| 当前目标统计（Target Statistic） | train12有效1957条 | development2有效385条 |
|---|---:|---:|
| head温度1时平均最高类别概率 | 9.66% | 7.00% |
| 再经KD温度2后平均最高类别概率 | 4.93% | 4.15% |
| KD目标平均熵（Entropy，nats） | 3.67736 | 3.68035 |
| 40类均匀分布熵 | 3.68888 | 3.68888 |

训练KD目标与均匀分布的平均KL仅0.01152 nats。当前Ridge头的argmax准确率很高，同时它的分数作为概率蒸馏目标十分平坦；准确率不能反映软目标标度。**未校准标度是优先级较高的待检验假设（Hypothesis），已确认数值差异，尚未确认20.36个百分点的因果归属。** 接近均匀的KD仍可产生平滑正则作用，不能称为损失失效或无梯度。

已生成的72候选记录还显示：当前最佳early_late候选准确率同为268/388=69.07%，与window_mean选中头相同；因此没有证据把教师的验证差距归因于这一次特征族选择。两头的目标分数及其校准行为仍可能不同，不能只用argmax相同推断蒸馏等价。

证据：`outputs/task5_audit/visual-design-diagnostic.json`，`A1/select/head.json`和`grid.json`。源码参考：[原教师校准](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/train_p85_videomae_full40_head.py:207)、[原Dataset目标读取](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/p86_visual_pixel_data.py:43)、[当前KD](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student.py:72)。

## 3. 用户划分与历史候选（User Splits and Historical Candidates）

当前select拟合12名用户、使用user6/user7全部388条选择；refit拟合14名用户。队友终端视觉阶段拟合18名用户2914条、固定16轮、没有验证loader；历史P86普通入口则在三折外层之内，用四折StratifiedGroupKFold生成inner_train/inner_dev，并另行报告outer_held结果。日志`val_accuracy`可能是inner开发成绩，不能未核对人口就当作固定user6/user7成绩；源码实际按用户分组，并非随机样本验证。

当前有效train12中，类别25仅有user1的3条样本，其余类别至少来自3名训练用户。这是固定人口的支持限制（Support Limitation），不能绕过边界加入保留用户标签。也不能在没有对应分区结果的情况下认为它解释了全部差距。

队友代码另有VideoMAE-Small学生、源帧时间去重（Temporal Source Deduplication）、layer3空间金字塔残差（Spatial Pyramid Residual）、2×2区域时间建模等候选；本轮复刻的是最终P87S记录的MC3 temporal基线。源码中存在候选不证明它已经被采用，亦不证明它取得更高准确率。参考：[P86-v2候选](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/p86v2_visual_model.py:11)、[原分组划分](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/train_p86_visual_student_oof.py:78)。

用户补充记忆：队友纯视觉val_acc超过0.8，其余模态只增加不到约10个百分点；最终全部源码位于`CUHK-X_Small_Model_Submission(1)`。这作为待追溯的原实验描述保留，不能因当前复刻低分就否定该记忆。

现有包缺少相关`runs/.../summary.json`、`candidate_oof_logits.npz`和历史训练日志；目前无法确认该高val的具体分区及处于初始视觉训练、融合后的视觉头还是其他候选阶段。此前找到的64.44%仅为监测脚本中写死的缓存筛选参考，不能替代对应正式结果。最终提交约0.91也不能替代纯视觉阶段成绩。

## 4. 直接检查最终提交包（Direct Final-bundle Inspection）

本次已按用户指定目录读取`CUHK-X_Small_Model_Submission(1)/checkpoints/model.pth`，使用`torch.load(..., weights_only=True, map_location='cpu')`。包内Student仅包含stage、deployment_model_config、model_state；没有历史validation metrics或labels。本次不运行该模型评分。

实际stage为`P87S_label_free_test_adaptation`，参数前缀为visual与motion_residual。deployment_model_config.visual明确为mc3_18_temporal、40类、width512、dropout0.18、gated、frames16、resolution160、freeze-through layer2、projection关闭；额外exact_time/cross_view/spatial_region/region_temporal/structured_region全关闭。

最终视觉model_state去掉visual前缀后，与当前A2 select检查点的全部键及张量形状精确匹配，没有额外或缺少的结构参数。最终包legacy模型源码与training模型源码相同。还逐字节比较最终包与教师分支快照中的六个关键文件，全部一致：p86_mc3_visual_model.py、train_p86_visual_pixel_oof.py、run_p87s_final_pipeline.py、p86_visual_pixel_data.py、train_p85_videomae_full40_head.py、build_p86_visual_pixel_cache.py。

因此最终包可以直接证明当前复刻没有选错最终视觉网络结构；不需要历史日志才能回答这一点。探针证据：`outputs/task5_audit/final-bundle-visual-metadata.json`。

最终权重经过融合与适配，不能恢复初始纯视觉训练时的独立val_acc；它的训练范围已包括user6/user7，在当前这两名用户上直接评分也不能称为跨用户验证。可提取最终visual分支研究结构和参数，但不能把它作为本run的无泄露初始化或把关闭motion后的分数等同于原A2基线。

包README第6–8行明确不包含历史预测、特征、实验和validation logs；Student三字段也确实无历史成绩。此处的证据缺失不代表队友的0.8描述不真实，而是最终导出无法单独恢复该阶段结果。参考：[最终包说明](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission(1)/README.md:6)、[最终部署构造](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission(1)/code/legacy/p87s_deploy_model.py:47)。

## 5. 解释与后续诊断边界（Interpretation and Diagnostic Boundary）

当前准确率差距不能简单解释为“视觉骨干做弱了”。与最终记录比较，模型结构及主要训练配方一致；教师目标来源、特征族、温度标度和人口不同，当前复刻应准确称为固定划分变体（Fixed-split Variant），不是整个原历史实验的数值等价复现（Numerically Equivalent Reproduction）。

此前代码与正式产物审计GO证明已实现冻结规格及其数据边界，不能证明规格中的head温度1与原校准目标等价，也不保证性能。head温度1在规格中明确写成复刻变体；本次发现该选择确实显著改变软目标分布。

若继续诊断，优先在独立实验产物中比较教师目标标度，固定学生结构、增强、优化器和训练人口；校准仅用train12内部独立预测/标签，不读取最终标签，不借用队友全人口OOF。必须先明确内部校准协议及与refit人口的对应规则，再训练对照；不覆盖当前已完成A1/A2、修改其identity或把本次诊断当作已证明的修复。本次未启动该实验或Task6。
