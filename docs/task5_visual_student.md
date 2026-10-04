# Task5视觉学生与序列复刻（Visual Student and Sequence Replication）

工作树：`D:/work/2026.7.14_kaggle/_single_visual_processing_replication`；分支：`experiment/teammate-single-teacher-task1`。本任务止于Task5，不进入Task6；final4标签继续封存。

## 实现与验收（Implementation and Acceptance）

像素采用IR早晚两窗、每窗16帧、scene/person/workspace三视野，160分辨率及队友裁剪/质量规则。refit14与final4原始像素各构建一次，覆盖2427/609条；学习产物按select/refit分离。实际时间戳恢复，未知值为NaN，不伪造10Hz。

学生使用公开MC3-18 Kinetics权重，冻结到layer2，保留队友全部BatchNorm评估模式、subject_robust同步增强和hybrid CE/KD/relation。直接feature/stage蒸馏项为0。训练batch4、累积4，select最多18轮，仅train12拟合、development2选择；refit从公开权重重新初始化，预算严格绑定已登记selection。

推理与sequence不需要教师目标或标签。原生sequence为`[N,2,3,16,512]`，anchor绑定同phase模型与pixels，不能用重复池化向量替代时间特征。

针对性回归30项通过。真实4条训练样本GPU前向/反向/native sequence耗时2.481秒、峰值2.073GiB；冻结层无梯度、分类头有非零梯度。此前工程/方法学审查发现的问题已修复并独立复审GO，见：

- [工程最终审计](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-04-task5-engineering-final-audit.md)。
- [方法学最终审计](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-04-task5-methodology-final-audit.md)。

## 校验与恢复（Validation and Recovery）

取消原始全目录SHA与祖先文件内容递归复验。实际消费ROI、模型与小型targets直接检查；大数组写入时计算摘要，登记不额外重扫。阶段内验证上下文（Stage Verification Context）复用教师特征加载，同时仍将传入特征逐元素绑定登记来源。

阶段使用OS文件锁（File Lock）拒绝并发。协议/输入/教师/代码/标签摘要形成identity；同identity完成产物幂等返回。模型、optimizer、scaler、最佳状态、历史与所有RNG按完整epoch原子保存，续跑不重新训练已完成epoch；先持久化selection/completion，再发布artifact。

像素缓存完成但登记中断时，从小型write-digests凭据恢复；必须精确包含六份NPY和rows.csv。若像素部分完成且没有最终凭据，明确拒绝在该目录恢复，需使用独立新输出目录；不为恢复而重新读取全部图像摘要。既有完整像素不重建，旧producer源码按SHA归档保留。

## 运行状态与命令（Runtime Status and Commands）

正式生成于2026-10-04 03:14:02（北京时间）完成。00:53:35启动select，01:57:05完成18轮选参并开始refit；03:01:42完成17轮refit，随后完成四分区预测与四分区sequence。正式计算接续共约2小时20分钟，错误日志为空，进程已退出。完成标记为`TASK5_CLI_STAGES_COMPLETE`；原PID文件只作为历史记录，不表示仍有运行中的作业。

开发集（Development Set）采用user6/user7全部388条：A2选中第17轮，准确率（Accuracy）189/388=48.71%，宏平均F1（Macro F1）36.48%，最差用户准确率（Worst-User Accuracy）46.80%。同划分A1视觉教师为268/388=69.07%；学生低20.36个百分点，少正确79条。这表明当前固定配方的视觉蒸馏没有保住教师准确率；不能据此断言后续多模态与后处理无效，也没有达到0.91的目标。

select每轮实际训练1956条，按队友规则丢弃最后单条尾批；类别权重仍来自1957条完整有效训练表。refit每轮训练2342条，从公开MC3初始化重新训练17轮，没有沿用select模型参数。开发结果未触发额外调参或更改预算。

| 分区（Partition） | 模型阶段（Model Phase） | 总样本 | IR有效 | 缺失回退 |
|---|---|---:|---:|---:|
| train12 | select | 2039 | 1957 | 82 |
| development2 | select | 388 | 385 | 3 |
| refit14 | refit | 2427 | 2342 | 85 |
| final4 | refit | 609 | 591 | 18 |

四份标准预测（Targets）位于运行目录`A2/select`或`A2/refit`，四份时间序列（Sequence）位于`sequence/<phase>/<partition>`。轻量验收确认规范ID、40类顺序、有限归一化概率、同phase先验、完整completed标记；四区anchor logits与A2预测最大差均为0。sequence为原生float16 `[N,2,3,16,512]`，真实片段存在时间变化，不是池化向量复制。未做原始全目录扫描或完整大缓存内容扫描。

正式产物独立工程与方法学验收（Independent Formal Artifact Acceptance）均为GO，无未解决重要问题；报告见[工程验收](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-04-task5-formal-engineering-acceptance.md)与[方法学验收](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-04-task5-formal-methodology-acceptance.md)。方法学审计独立复算user6为95/203=46.80%、user7为94/185=50.81%，包括全部388条开发样本。Task5已完成，final4私有标签保持封存，没有计算final准确率；Task6尚未启动，整个实验状态仍为generating，而非最终冻结或揭示。

Python为`D:/Anaconda/envs/PyTorch2.7/python.exe`，以下以python代指，均从本工作树执行：

```powershell
python -B scripts/build_no_vote_pixels.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition refit14
python -B scripts/build_no_vote_pixels.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition final4
python -B scripts/run_no_vote_visual_student.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --phase select
python -B scripts/run_no_vote_visual_student.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --phase refit
```

预测显式传`--phase predict --partition <分区> --model-phase select|refit`；sequence使用`build_no_vote_sequence.py`及相同分区/模型阶段。train12/development2使用select，refit14/final4使用refit。原10分钟自动化已不存在，本次后台PowerShell负责计算接续，不依赖该自动化；异常立即停止并保留日志。
