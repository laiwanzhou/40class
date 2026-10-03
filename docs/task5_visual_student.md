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

2026-10-04 00:53启动正式select，后续由隐藏接续脚本顺序运行refit、四分区预测与四分区sequence。正式运行尚未完成，不能将接口测试或审计GO当作训练结果。最新状态以`outputs/task5_audit/student-continuation.log`、`student-continuation-error.log`及各阶段日志为准；进程PID见`student-continuation.pid`，不要重复启动。

Python为`D:/Anaconda/envs/PyTorch2.7/python.exe`，以下以python代指，均从本工作树执行：

```powershell
python -B scripts/build_no_vote_pixels.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition refit14
python -B scripts/build_no_vote_pixels.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition final4
python -B scripts/run_no_vote_visual_student.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --phase select
python -B scripts/run_no_vote_visual_student.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --phase refit
```

预测显式传`--phase predict --partition <分区> --model-phase select|refit`；sequence使用`build_no_vote_sequence.py`及相同分区/模型阶段。train12/development2使用select，refit14/final4使用refit。原10分钟自动化已不存在，本次后台PowerShell负责计算接续，不依赖该自动化；异常立即停止并保留日志。
