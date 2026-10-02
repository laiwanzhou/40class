# CUHK-X Task2/3完成交接（Task2/3 Handoff）

日期：2026-10-02，Asia/Shanghai。继续分支experiment/teammate-single-teacher-task1，当前新增Task2/3，Task4尚未开始。工作树D:/work/2026.7.14_kaggle/_single_visual_processing_replication。

## 当前状态（Status）

- Task1保留已完成；Task2已实现可信准备、无标签清单与来源注册表，生成正式2039/388/2427/609行四分区清单。
- Task3已移植P28/P29算子及实际时间戳恢复，真实train12一条user5样本22帧，22帧Skeleton对齐；CPU YOLO推理与续跑验证通过。
- Task3正式全量ROI还未生成，当前raw_cache完整标志为false/partial=true，不可供完整教师训练消费。
- 相关测试57通过；全仓498通过、7个已有旧缓存缺失失败，未新增失败。
- 全程没有Teacher/Student训练，不读取final标签作模型输入或选择。

## 权威文件（Authoritative Files）

- docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md：Task1–3已勾选，Task4–14未实现。
- docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md。
- docs/task23_input_pose_roi.md：运行命令、模态覆盖和实施边界。
- reports/2026-10-02-task23-verification.json：正式清单、22帧结果、测试和已有失败名称。
- 本地run根outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2；产物/权重均Git忽略。
- 私有评估资料C:/Users/LaiWanzhou/AppData/Local/Temp/cuhkx_no_vote_labels/fixed-split-single-teacher-v2；final标签和旧ID映射不在模型配置或Git中。

## 实施审计与修复（Review and Fixes）

独立审计发现两项Important，均回归复现后修复：predictions必须存在匹配模型祖先，A8目标须完整final ID且complete；ROI完整或中断续跑均重新比较输入文件集合、内容、producer/config/weight和cache SHA。另区分development2预测与refit14内含开发用户的预测，避免错误拒绝合法refit目标。

IMU按原读取器CSV文件规则检测为final580，而早期估计584；样本分母609、全缺失18及591适配池不变。CSV存在不代表后续有效设备/数值支持，Task6还需测量。原始目录及队友源码未修改。初次可用性诊断缓存只在outputs中归档。

build_pose_roi增加必传protocol关键字，来源/权重/registry显式传入；计划签名已同步。

## 下一步（Next Steps）

用户继续授权后推进Task4。需先由既有Task3构建器生成相应分区完整ROI（不带max-trials），不能将smoke记录视为全量缓存。复用原始缓存或冻结公开特征，学习统计与select/refit模型继续分开。

不用旧实验缓存恢复7个已有失败，不重新引入OOF、旧教师bank或已丢弃的接口匹配。最终揭示仍只在后续Task13全部候选冻结之后。

## 建议技能（Suggested Skills）

executing-plans、test-driven-development、systematic-debugging、verification-before-completion；实验结果解释使用academic-research-suite。遵循当前明确任务范围，不自动启动长训练。

计划SHA256：872ce3d661eb04e0acc3253a543d9599eb6a570f44a022227c33de5b34b8b108

规格SHA256：99754cb055c3ed62f91495a2cf40d9b239f4d87665fcc903f5dd6174a6cc35d6
