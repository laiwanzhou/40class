# Skeleton 帧级清洗索引

## 决策

采用**索引式清洗**，不复制、不移动、不改写原始 JSON。两份互补索引记录规范 JSON 路径、候选人数、选中的 `candidate_index`、视觉 margin、保留状态和排除原因。

训练只读取 `skeleton_retained_frame_index.csv`；`skeleton_ambiguous_frame_index.csv` 仅用于诊断，不用于该帧训练。

## 规则

1. 单候选帧：保留，`candidate_index=0`。
2. 多候选帧且视觉匹配 `margin≥20%`：保留视觉选中的 candidate。
3. 其他多候选帧：标记 `ambiguous`，`use_for_frame_training=false`。
4. 同一 frame_id 有重复命名 JSON 时只索引一份，优先时间戳文件。

## 统计

| 项目 | 帧数 | 占全部唯一帧 |
|---|---:|---:|
| 全部唯一 Skeleton 时间步 | 85879 | 100.0000% |
| 保留：单候选 | 82356 | 95.8977% |
| 保留：多候选且 margin≥20% | 2441 | 2.8424% |
| 总保留 | 84797 | 98.7401% |
| ambiguous / 不用于该帧训练 | 1082 | 1.2599% |

ambiguous 中，65 帧是已有视觉比较但 margin 不足，1017 帧是没有满足当前诊断条件的视觉决策。共有 6 个 trial 没有保留帧，全部位于 validation；train 中没有整条 trial 被清空。

注意：多候选视觉诊断限定在 fold-0 train 用户与完整跨模态配对帧。因此 validation 多候选帧以及 train 中没有可靠视觉决策的多候选帧均保守标为 ambiguous，不用 held-out validation 调阈值或补做选择。

## 使用契约

- `skeleton_retained_frame_index.csv` 与 `skeleton_ambiguous_frame_index.csv` 是全集的互补分区。
- `skeleton_json_path` 相对于 competition-train 数据根，便于迁移而不绑定绝对盘符。
- `candidate_index` 是原 JSON 顶层 person list 的零基索引；ambiguous 行为空值。
- 下游应按 `sample_id, frame_id` 排序，且只能读取 `use_for_frame_training=true` 的行。
- `retained_segment_index` 在 ambiguous 或原始缺帧位置断开；速度、窗口和插值不得跨 segment 直接连接。
- 原始 JSON 仍是唯一事实来源；该索引只表达当前版本的清洗决策。
