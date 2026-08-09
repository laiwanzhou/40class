# ROI640 第二阶段：时序与左右身份审计

## 1. 结论

- 第二阶段验收：**passed**。
- 固定使用现有 YOLO `imgsz=640` 姿态缓存；没有测试 1280。
- 未读取 competition test，未训练模型，未重导 Depth。
- 有效局部ROI相邻转移：224,281；高风险尺度归一化跳变上界：473。
- 左右交换可审计中心帧：63,019；腕轨迹候选：34；ROI交叉证据同时支持：28。
- 跳变和交换门槛按最坏情况计算：即把全部高置信候选都视为真实错误，因此无需用主观目视结果降低计数才能通过。

## 2. 输入完整性与JET源色

- 有效标记但纯黑且仍未被content mask排除：0。
- 扫描原始Depth_Color：84,906帧；不可读：0；非黑非JET像素：0。

## 3. 三路IR局部冗余

- 全局三路均失效：3.62%。
- 小动作24类三路均失效：2.14%。
- 最差小动作：`Take_a_selfie`，9.66%。
- `P(relation valid | right invalid)`：72.23%。
- `P(relation valid | left invalid)`：17.57%。
- `P(relation valid | both invalid)`：0.03%。

## 4. 尺度归一化跳变

高风险候选定义为：相邻帧ROI IoU不高于0.10，并且中心位移/人物框对角线或人物相对中心变化不低于0.50。
候选率上界：0.2109%；最差类别上界：2.3613%。
该指标同时输出相邻IoU、人物尺度归一化位移、人物相对中心变化和面积倍率；不再使用单一的全图对角线20%作为最终判据。

## 5. 左右身份交换

高置信候选要求三帧腕轨迹交叉匹配显著优于直接匹配，并由左右ROI的交叉IoU同时支持。
候选率上界：0.0444%；最差类别上界：0.4420%。
这些是保守的自动候选，不是对每一帧人体解剖身份的人工真值标注；验收采用全部候选均为真错误的最坏上界。

## 6. 连续失效和短时序材料

- 长度不低于8帧的局部失效区间：541。
- 生成审查contact sheet：161，位于 `D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_stage2_temporal_review`。
- swap/jump材料展示事件前后 `t-3..t+3`；长失效材料展示开始、中段和恢复附近，并带完整mask时间轴。

## 7. 验收门

- PASS: `missing_files_zero`
- PASS: `unreadable_files_zero`
- PASS: `wrong_dimensions_zero`
- PASS: `known_p0_still_effective_zero`
- PASS: `source_depth_unreadable_zero`
- PASS: `unexplained_non_jet_pixels_zero`
- PASS: `global_all_local_invalid_le_5pct`
- PASS: `small24_all_local_invalid_le_3pct`
- PASS: `every_small_action_all_local_invalid_le_10pct`
- PASS: `swap_global_upper_bound_le_1pct`
- PASS: `swap_every_class_upper_bound_le_5pct`
- PASS: `jump_global_upper_bound_le_1pct`
- PASS: `jump_every_class_upper_bound_le_5pct`

## 8. 产物

- `D:\work\2026.7.14_kaggle\40class-ir-primary-interaction-wt\reports\roi640_stage2_temporal_identity\normalized_temporal_transitions.csv`
- `D:\work\2026.7.14_kaggle\40class-ir-primary-interaction-wt\reports\roi640_stage2_temporal_identity\normalized_jump_candidates.csv`
- `D:\work\2026.7.14_kaggle\40class-ir-primary-interaction-wt\reports\roi640_stage2_temporal_identity\left_right_swap_candidates.csv`
- `D:\work\2026.7.14_kaggle\40class-ir-primary-interaction-wt\reports\roi640_stage2_temporal_identity\long_invalid_runs.csv`
- `D:\work\2026.7.14_kaggle\40class-ir-primary-interaction-wt\reports\roi640_stage2_temporal_identity\stage2_review_artifacts.csv`
- `D:\work\2026.7.14_kaggle\40class-ir-primary-interaction-wt\reports\roi640_stage2_temporal_identity\stage2_acceptance_gate.json`

第二阶段到此停止；没有执行第三阶段的逆JET导出器实现。
