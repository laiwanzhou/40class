# 第四阶段独立复审：Depth ordinal 导出器与组合 Manifest

## 结论

**通过。未发现阻止 Stage 5 的 P0/P1 问题。**

本次只复审第四阶段已经形成的实现、测试、报告和 handoff 状态，重点核对实际数据契约及全量 train/val 元数据连接结果。不检查跨环境可移植性、路径配置风格或与固定运行环境无关的鲁棒性。

本次未执行 Stage 5 真实数据导出，未读取 competition test，未训练模型。

## 复审范围

- `scripts/export_depth_ordinal_assets.py`
- `src/data/ordinal_depth.py` 中本阶段使用的 crop、mask-aware resize 与 letterbox 路径
- `tests/test_depth_ordinal_exporter.py`
- `tests/test_ordinal_depth.py`
- `reports/depth_ordinal_stage4_implementation.md`
- `C:\Users\LaiWanzhou\AppData\Local\Temp\codex-handoff-ir-ordinal-fullseq-2026-08-09.md` 中的 Stage 4 状态
- 现有 ROI640 train/val 导出的 84,906 帧元数据及 Stage 2 审计表

## 1. 实际写入边界

代码中的图像写入只发生在两路 Depth 循环内：

- `depth_context_ordinal`
- `depth_context_pixel_valid`
- `depth_relation_ordinal`
- `depth_relation_pixel_valid`

四路 IR：

- `ir_context`
- `ir_left`
- `ir_right`
- `ir_relation`

均通过绝对路径引用现有 `roi640_full_inputs_256` 文件，没有 IR 复制或重新编码路径。除四类 Depth PNG 外，输出仅包含组合 manifest、Depth 视图审计表、元数据 JSON 和 `_SUCCESS` 标志。

专项集成测试也确认 `ir_images_duplicated=false`，且 IR manifest 路径解析到原有 baseline 根目录。

## 2. Depth 处理顺序与 mask 语义

处理顺序符合已冻结设计：

1. 从原生 `Depth_Color` 文件读取 BGR。
2. 在任何 crop/resize 之前执行精确 OpenCV JET 逆变换。
3. 纯黑像素形成独立 `pixel_valid=false`；非黑、非 JET 颜色按严格默认阈值拒绝。
4. 使用原生 ordinal 值和原生 pixel mask 按已保存 ROI 坐标裁剪。
5. 对 `depth * valid` 与 `valid` 使用同一插值计算加权值；下采样使用 area，其他情况使用 bilinear。
6. 最终二值 mask 使用 nearest-neighbour。
7. 等比例缩放后居中 letterbox，padding 保持无效。
8. resize/letterbox 完成后再次以最终 mask 将无效 ordinal 像素清零。

因此无效零值不会在插值中稀释相邻有效 ordinal 值，pixel mask 也没有被当作普通图像通道来替代显式约束。

## 3. 组合 Manifest 契约

逐帧记录包含：

- split、class、action、sample、user、`source_frame_index`；
- 原始 IR 和原始 Depth 路径；
- timestamp、frame id、相邻帧时间差；
- 四路现有 IR 文件引用；
- 两路 ordinal Depth 及两路独立 pixel-mask 引用；
- 六路 pose-valid、content-valid、effective-valid；
- 两路 Depth pixel coverage；
- 六路 deterministic reliability；
- `temporal_valid`。

离线导出不采样、不 padding、不进行逐帧或逐 trial 归一化。所有真实帧在 manifest 中仍以原始顺序保留。

固定 reliability 实现为：

```text
effective_valid * clip((pose_score - 0.25) / 0.75, 0, 1) * pixel_coverage
```

IR 的 pixel coverage 固定为 1；Depth 使用实际 mask 均值。现有 frame manifest 中 context confidence 全部为 1，左右方向与 relation confidence 已由冻结的 ROI 构建流程按肘腕/单手/双手规则生成。

## 4. Content validity

ordinal Depth 使用已批准的三个指标：

- `dynamic_range <= 8`
- `std <= 2`
- `entropy_32 <= 1`

至少两个成立时写入 `ordinal_content_invalid_2of3=true`。最终 Depth content invalid 为：

```text
ordinal_content_invalid_2of3 OR Stage 2 retained content-invalid evidence
```

Stage 2 中两条已知 Depth content-invalid 证据均可被当前连接表定位，导出器不会因新 scalar crop 指标通过而清除它们。IR 的 16 条既有 content-invalid 证据直接沿用 Stage 2 effective audit。

## 5. 独立测试结果

独立执行：

```text
python -m pytest tests/test_ordinal_depth.py tests/test_depth_ordinal_exporter.py -q
```

结果：`11 passed`。

覆盖内容包括：

- 256 个 JET index 精确逆变换；
- 黑色 invalid 与合法 index 0 的区分；
- 未知颜色拒绝及显式 invalid；
- mask-aware 上采样/下采样；
- 全 invalid 清零；
- crop/letterbox 比例与 padding mask；
- 2-of-3 content 规则；
- deterministic reliability；
- Depth-only synthetic export；
- IR 只引用、不复制；
- 输出 ordinal/mask 对齐及 invalid 再次清零。

`py_compile` 与 `git diff --check` 也通过。

## 6. 全量 train/val 只读核验

独立加载当前真实元数据并调用同一严格配对函数，未生成任何真实 ordinal 资产：

| 检查项 | 结果 |
|---|---:|
| 原始 frame manifest 行数 | 84,906 |
| trial 数 | 2,910 |
| train 帧 | 67,216 |
| val 帧 | 17,690 |
| ROI audit 行数 | 509,436 |
| effective audit 行数 | 509,436 |
| source Depth 行数 | 84,906 |
| 严格配对后的 Depth/IR 行数 | 84,906 |
| frame 与配对表 outer join 非闭合行 | 0 |
| frame key 重复 | 0 |
| 配对 key 重复 | 0 |
| 每帧不是恰好 6 路 ROI 的记录 | 0 |
| 每帧不是恰好 6 路 effective 记录 | 0 |
| trial 内 `source_frame_index` 非连续 | 0 |
| timestamp 缺失 | 0 |
| frame id 缺失 | 0 |
| 负 inter-frame delta | 0 |
| trial 首帧 delta 非 0 | 0 |
| 四路 IR manifest/ROI-audit 路径不一致 | 0 |
| 四路 IR Stage 2 缺失记录 | 0 |
| 四路 IR Stage 2 不可读记录 | 0 |

严格配对过程遍历全部 2,910 个 trial 的原始 Depth/IR 目录，要求两种模态的 `(timestamp, frame_id)` 键集合完全相同，然后按该键排序。全部 trial 均通过。

## 7. 报告与 handoff 一致性

`reports/depth_ordinal_stage4_implementation.md` 中的 84,906 帧、2,910 trial、509,436 视图行、0 个缺失 IR 引用和 11 项测试均由本次独立检查复现。

handoff 已将 Stage 4 标记为完成，并明确 Stage 5 尚未开始。默认真实输出目录当前不存在，与“尚未进行真实导出”的边界一致。

## 8. P0/P1 与后续边界

- P0：无。
- P1：无。
- Stage 5 可以按既定计划进行有限真实数据 smoke export，并检查 native ordinal、裁剪后 scalar、pixel mask 以及相邻帧的视觉/数值对齐。
- 本复审不等于 Stage 5 的真实输出验收；真实 crop 内容统计、最终新增 ordinal content-invalid 数和成品 PNG 对齐仍应在 Stage 5 基于有限导出验证。

**最终判定：第四阶段成果通过独立复审，可以进入 Stage 5；本次复审在 Stage 4 边界停止。**
