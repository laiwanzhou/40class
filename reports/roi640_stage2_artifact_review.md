# ROI640 第二阶段实际产物独立复审

## 1. 复审范围与结论

本次仅复审当前固定环境中已经生成的第二阶段产物，不评价脚本的跨环境兼容性、路径可移植性或防御式鲁棒性。复审期间未读取 competition test，未运行训练，也未修改数据和图像。

**结论：第二阶段按 handoff 中预先定义的验收门槛通过是合理的。** 所有硬门槛均可由 CSV/JSON 实际产物重新计算并支持；未发现会把结论改为 failed 的 P0 问题。

需要保留一个重要限定：`passed` 表示完整性、内容掩码、全局/逐类局部可用率、左右交换与归一化跳变的既定门槛通过，不表示每条 trial 的三路局部 ROI 都充分可用。实际有 93/2,910 条 trial 在至少 50% 帧中三路局部 IR 全无效，其中小动作 24 类为 26/1,711（1.52%）。这些是 P1 人工复审项，但 context 始终有效，且没有突破既定总体或逐类门槛。

## 2. 产物完整性与相互一致性

| 产物 | 实际行数 | 复核结果 |
|---|---:|---|
| `effective_view_audit.csv` | 509,436 | 84,906 帧 x 6 路；2,910 个 sample；主键无重复 |
| `normalized_temporal_transitions.csv` | 224,281 | 高风险标记合计 473，与候选表完全一致 |
| `normalized_jump_candidates.csv` | 473 | 与逐类表 events 合计一致 |
| `normalized_jump_by_action.csv` | 40 | eligible 合计 224,281，events 合计 473 |
| `left_right_swap_candidates.csv` | 34 | 其中 `high_confidence_swap_candidate=true` 恰好 28 |
| `left_right_swap_eligibility.csv` | 2,910 | eligible center 合计 63,019 |
| `left_right_swap_by_action.csv` | 40 | eligible 合计 63,019，events 合计 28 |
| `long_invalid_runs.csv` | 541 | 全部为长度至少 8 帧的连续局部失效段 |
| `source_depth_jet_integrity.csv` | 84,906 | 路径唯一；全部可读；异常非 JET 像素为 0 |
| `local_availability_by_action.csv` | 40 | 与逐帧三路 effective mask 重算一致 |
| `stage2_review_artifacts.csv` | 161 | 28 swap + 60 jump + 73 long-invalid；161 个文件均存在 |

审计报告、逐类 CSV 和 `stage2_acceptance_gate.json` 的关键数值一致。JSON 中 `swap_candidates=28` 实际指“34 个腕轨迹候选中获得 ROI 交叉证据支持的 28 个”，字段名略简化，但报告正文已经正确区分 34 与 28，不影响门槛结论。

## 3. 验收门槛复算

| 验收项 | 实际值 | 门槛 | 结论 |
|---|---:|---:|---|
| 缺失文件 | 0 | = 0 | PASS |
| 不可读视图 | 0 | = 0 | PASS |
| 非 256x256 视图 | 0 | = 0 | PASS |
| 已知 P0 仍为 effective valid | 0 | = 0 | PASS |
| Depth 源不可读 | 0/84,906 | = 0 | PASS |
| Depth 非黑且非 JET 像素 | 0 | = 0 | PASS |
| 全局三路局部全无效 | 3,076/84,906 = 3.6228% | <= 5% | PASS |
| 小动作 24 类三路局部全无效 | 1,089/50,955 = 2.1372% | <= 3% | PASS |
| 最差小动作 `Take_a_selfie` | 124/1,284 = 9.6573% | 每类 <= 10% | PASS，距离门槛较近 |
| swap 全局保守上界 | 28/63,019 = 0.0444% | <= 1% | PASS |
| swap 最差类 `Take_off_clothes` | 4/905 = 0.4420% | 每类 <= 5% | PASS |
| jump 全局保守上界 | 473/224,281 = 0.2109% | <= 1% | PASS |
| jump 最差类 `Put_on_clothes` | 200/8,470 = 2.3613% | 每类 <= 5% | PASS |

本轮门槛把全部 28 个 ROI 交叉支持 swap 和全部 473 个 jump 都按真实错误计算，没有依赖人工主观判断来降低计数，因此视觉抽查不会使验收率变差。

## 4. 三路局部冗余

每帧有效局部视图数量分布为：0 路 3,076 帧，1 路 2 帧，2 路 8,656 帧，3 路 73,172 帧，总数为 84,906。

- `P(relation valid | right invalid) = 8,002/11,078 = 72.2333%`
- `P(relation valid | left invalid) = 656/3,733 = 17.5730%`
- `P(relation valid | left and right invalid) = 1/3,077 = 0.0325%`

因此 relation 对右路失效具有明显恢复作用，对左路失效的恢复较弱；当左右同时失效时，relation 几乎不会独立恢复。这与 relation 由同一套姿态关键点构造的机制一致。六路设计仍有冗余价值，但不能把 relation 理解为独立于姿态失败的兜底检测器。

## 5. contact sheet 视觉复审

### 5.1 归一化 jump

抽查最高严重度和代表性样本后，候选主要由三类现象组成：

1. **真实入画/离画**：最高严重度 `Walk` 样本中，人物从画面边缘进入，局部 ROI 从无效或极小局部突然切换到完整人体局部。这是合理的真实场景变化，不是持续性身份错误。
2. **快速动作、自遮挡与尺度变化**：穿脱衣、伸展、跳跃等会使腕部短暂遮挡或靠近人体边缘，产生真实的大幅 ROI 变化。
3. **姿态跳点或局部身份不稳定**：`Put_on_clothes`、`Comb_hair`、`Brush_teeth`、`Take_medicine` 的部分候选中，context 连续稳定，但局部框在手、头部、躯干或桌面间突跳，属于应人工确认的真实 ROI 质量问题。

必须人工复审的高严重度 jump：

- `jump/train__c05__user7__1-1-2__ir_right__f0003.png`
- `jump/train__c05__user8__7-1-1__ir_relation__f0023.png`
- `jump/train__c02__user7__1-1-3__ir_left__f0014.png`
- `jump/train__c01__user6__1-1-2__ir_left__f0035.png`
- `jump/train__c31__user9__1-1-1__ir_relation__f0025.png`
- `jump/train__c37__user9__7-1-3__ir_right__f0010.png`

可作为“真实入画而非系统错误”对照的文件：

- `jump/train__c36__user6__7-3-1__ir_relation__f0014.png`
- `jump/train__c36__user17__5-2-1__ir_left__f0003.png`

### 5.2 左右身份交换

34 个腕轨迹候选中 28 个同时获得 ROI 交叉 IoU 支持，计数和 contact sheet 完整。抽查显示多数事件发生在双臂交叉、穿脱衣、跳跃、躺卧或手臂快速经过身体中线时，属于最容易触发左右关键点交换的真实动作阶段；未看到大范围、长时间持续把同一侧 ROI 固定写入另一侧的系统性错误。

contact sheet 本身没有叠加左右腕关键点编号，因此只能确认两路裁剪是否发生突变，不能单凭灰度裁剪给每个事件建立人工左右真值。下列文件动作交叉或遮挡最明显，应人工复审关键点身份：

- `swap/train__c03__user18__6-1-2__f0006.png`
- `swap/train__c03__user18__6-1-3__f0013.png`
- `swap/train__c03__user18__7-1-1__f0001.png`
- `swap/train__c03__user18__7-1-2__f0001.png`
- `swap/train__c16__user22__3-2-2__f0020.png`
- `swap/train__c30__user1__1-1-2__f0022.png`
- `swap/train__c33__user6__5-3-1__f0003.png`
- `swap/train__c34__user6__5-3-1__f0016.png`

即使将这 28 个全部视为真实交换，0.0444% 的全局上界和 0.4420% 的最差类上界仍远低于门槛。

### 5.3 连续失效

541 个连续失效段中，右路 312、左路 120、relation 109；中位长度 12 帧，90 分位 24 帧，最大 84 帧。抽查结论如下：

- `Walk` 的长失效常由人物尚未入画造成，属于合理 invalid。
- `Lie_down` 的 84 帧三路失效来自人物躺卧后腕部严重遮挡和姿态检测失效，符合已知困难场景。
- `Peel_fruits`、`Eat_food`、`Pour_drinks`、`Take_a_selfie` 的部分单路长失效由另一路和 relation 持续补偿，冗余实际有效。
- 某些远景、低照或人物较小的小动作 trial 中三路会同时长时间失效；mask 正确阻止黑色占位进入有效特征，但这些 trial 的局部物品信息事实上不可用。

必须人工复审的长失效文件：

- `long_invalid/train__c17__user23__5-3-2__ir_left__f0000-0022.png`：`Tap_the_keyboard`，23/23 帧三路全无效。
- `long_invalid/train__c15__user6__3-3-1__ir_relation__f0004-0072.png`：`Wipe_windows_and_tables`，70/73 帧三路全无效。
- `long_invalid/train__c22__user1__6-2-3__ir_relation__f0003-0036.png`：`Turn_pages`，35/37 帧三路全无效。
- `long_invalid/train__c11__user21__5-2-3__ir_relation__f0003-0032.png`：`Peel_fruits`，31/33 帧三路全无效。
- `long_invalid/train__c18__user2__4-3-3__ir_relation__f0006-0026.png`：`Write`，23/27 帧三路全无效。
- `long_invalid/train__c21__user1__6-2-2__ir_relation__f0000-0008.png`：`Read_documents`，9/11 帧三路全无效。
- `long_invalid/train__c20__user1__5-1-3__ir_relation__f0007-0021.png`：`Check_the_time`，17/25 帧三路全无效。
- `long_invalid/train__c24__user2__7-3-1__ir_relation__f0007-0018.png`：`Use_a_mobile_phone`，12/21 帧三路全无效。
- `long_invalid/train__c33__user16__5-3-1__ir_relation__f0036-0119.png`：`Lie_down` 遮挡对照。
- `long_invalid/train__c36__user2__3-2-2__ir_relation__f0000-0041.png`：`Walk` 入画对照。

小动作 24 类中共有 26/1,711 条 trial 的三路局部全无效率达到 50% 或以上。这是当前产物中最值得在进入正式训练前知悉的 P1 风险。它不代表文件损坏；这些帧的 context 仍正常，且局部无效状态已被 mask 正确标记。

## 6. Depth JET 与 2-of-3 content mask

### 6.1 Depth 源扫描

`source_depth_jet_integrity.csv` 覆盖 2,910 个 sample 的 84,906 个唯一 Depth_Color 源文件。全部文件可读，所有非黑像素均属于 OpenCV `COLORMAP_JET` 的 256 色离散 LUT，未发现任何异常颜色。该 LUT 的 256 个颜色值彼此唯一，因此实际产物支持后续对非黑像素做确定性 ordinal 逆映射；黑色仍必须由显式 pixel mask 单独处理。

### 6.2 content mask

2-of-3 低信息规则实际标出 18 个原先 `valid_flag=1` 的低信息视图，分布为：

- `ir_right`：13
- `ir_left`：2
- `ir_relation`：1
- `depth_relation`：2

18 个视图来自 9 个 sample，全部已变为 `effective_valid=false`。已知两张纯黑 P0（`Write` 的 `ir_right` 和 `Walk` 的 `depth_relation`）均被排除；不存在 `exact_black=true` 且 `effective_valid=true` 的记录。三路局部全无效的 3,076 帧均来自原始 pose valid flag 无效，而不是 content mask 意外扩大，因此 2-of-3 规则没有造成局部可用率的结构性下降。

## 7. 问题分级与最终判断

### P0

无。没有发现文件缺失、损坏、尺寸错误、已知黑图仍有效、CSV/JSON 计数矛盾、JET 源异常，或会令既定验收门槛失败的问题。

### P1

1. **trial 级局部失效存在聚集**：93/2,910 条 trial 至少半数帧三路局部全无效；小动作 24 类为 26/1,711。总体率和逐类率仍通过，但这些 trial 训练时只能主要依赖 context。
2. **少数高严重度 jump 明显像姿态跳点**：尤其穿衣、梳头、刷牙、吃药样本，需按上列文件人工确认。
3. **28 个 swap 事件缺少人工左右真值**：自动证据足以构成保守上界且不影响 pass；人工审查的意义是了解错误类型，不是为了让门槛通过。

### 最终判断

维持 `Stage2 status = passed`。该结论有实际产物支持，且门槛余量总体充足。后续使用时必须保留 `view_valid_mask` 和 availability/reliability 信息，并把本报告列出的高局部失效 trial 视为“context 可用、局部物品信息不可用”的真实输入情况，而不是普通有效局部样本。
