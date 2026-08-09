# Stage 3 独立复审：inverse-JET ordinal Depth

## 结论

**维持通过。** 本次复审未发现会阻止进入下一阶段的 P0/P1 问题。

复审只检查第三阶段的实际正确性和既有产物，不评价跨环境可移植性、路径鲁棒性或无关代码风格；未修改数据、未读取 competition test、未导出 Depth 资产、未进入 Stage 4，也未运行训练。

## 复审范围

- `src/data/ordinal_depth.py`
- `tests/test_ordinal_depth.py`
- `scripts/verify_depth_ordinal_stage3.py`
- `reports/depth_ordinal_stage3_verification.json`
- `reports/depth_ordinal_stage3_verification.md`
- `C:\Users\LaiWanzhou\AppData\Local\Temp\codex-handoff-ir-ordinal-fullseq-2026-08-09.md` 中的 Stage 3 状态

## 1. OpenCV JET 逆映射

- 使用当前环境的 `cv2.COLORMAP_JET` 独立生成了 256 项 BGR LUT。
- 256 个 BGR 值全部唯一，LUT 中不存在纯黑 `[0, 0, 0]`。
- 独立构建 `BGR -> ordinal index` 映射后，索引 `0..255` 全部能够精确恢复。
- 模块使用 BGR 三通道打包、排序和 `searchsorted` 查找，返回的索引与独立字典映射一致。
- ordinal 值 `0` 与无效黑像素虽然都在 `values` 中保存为 `0`，但由 `pixel_valid` 明确区分：JET 索引 0 为有效，纯黑为无效。

结论：逆 JET 映射是一一、精确且无黑色冲突的。

## 2. 未知颜色与掩码语义

- `matched` 仅表示精确属于 JET LUT 的像素。
- `black` 仅表示三通道均为 0 的纯黑像素。
- `unexpected_mask = non_black AND NOT matched`，不会把纯黑错误计为未知颜色。
- 默认 `max_unexpected_pixels=0`，任何未知非黑颜色都会抛出 `UnexpectedJetColorError`。
- 在显式容忍阈值内，未知颜色仍保持 `pixel_valid=false`、`unexpected_mask=true`、`values=0`，不会混入有效 Depth。

结论：`pixel_valid` 与 `unexpected_mask` 的职责清楚，未知颜色不会被静默当作有效深度。

## 3. Mask-aware resize

实现符合 handoff 规定：

```text
numerator = resize(depth * valid)
coverage  = resize(valid)
depth_out = numerator / max(coverage, eps)
```

- 同时缩小时对值和 coverage 使用 `INTER_AREA`，其他尺寸变化使用 `INTER_LINEAR`。
- 最终二值 mask 使用 `INTER_NEAREST`。
- 输出尺寸参数按 `(height, width)` 接收，并按 OpenCV 的 `(width, height)` 顺序传入。
- 结果四舍五入、截断到 `uint8 0..255`，随后在最终 mask 无效处再次强制清零。
- 单个有效像素周围不会因无效占位零而稀释；全无效输入仍保持全零且全无效。

结论：数值插值、coverage 归一化、最终 mask 和无效清零均符合当前阶段约定。

## 4. 测试与真实数据独立抽查

### 单元测试

```text
8 passed in 4.63s
```

覆盖 256 索引恢复、纯黑与索引 0 区分、未知颜色拒绝/显式无效、上采样、下采样、全无效输入和形状检查。

### 独立真实帧抽查

根据 `metadata/splits/fold_0.json` 的 14/4 用户划分，独立选择：

- 12 条训练用户 trial；
- 8 条留出验证用户 trial；
- 每条 trial 检查首帧、中间帧和末帧，共 60 帧；
- 累计检查 15,127,652 个有效 JET 像素和 3,304,348 个纯黑像素。

逐帧独立重算结果：

- 未出现未知非黑颜色；
- 模块 ordinal 输出与独立 LUT 查找逐像素一致；
- `pixel_valid` 与独立 black/JET 分类逐像素一致；
- 有效像素重新着色后与源 BGR 精确一致；
- 独立重算的 `256x256` mask-aware resize 与模块输出逐像素一致；
- 最终无效 mask 后的非零像素为 0。

源清单中的 `sample_id` 均以 `train__` 开头，因为它们来自赛事提供的训练目录；其中验证集是 fold 定义的 4 个留出用户，不是 competition test。抽查实际包含 12 条训练用户 trial 和 8 条留出用户 trial。

## 5. 正式验证报告复现

使用临时输出路径完整复跑 `scripts/verify_depth_ordinal_stage3.py`，核心结果与正式 JSON/Markdown 完全一致：

| 指标 | 正式报告 | 独立复跑 |
| --- | ---: | ---: |
| trial 数 | 2,910 | 2,910 |
| 有效源像素 | 723,756,503 | 723,756,503 |
| 黑色无效源像素 | 170,195,497 | 170,195,497 |
| JET 往返不一致像素 | 0 | 0 |
| resize 后有效像素 | 154,227,320 | 154,227,320 |
| 无效 mask 后非零像素 | 0 | 0 |
| 输出尺寸 | 256x256 | 256x256 |

复跑耗时为 110.235 秒，正式报告为 114.350 秒；耗时差异不影响结果。JSON 与 Markdown 的核心数字一致。

## 6. 阶段边界与 handoff 状态

- competition test read：`false`；验证输入清单只指向赛事训练目录中的 Depth_Color 文件。
- Depth assets exported：`false`；未发现新建的 ordinal Depth 资产目录。
- training run：`false`；第三阶段脚本只执行编码与 resize 验证。
- handoff 已将 Stage 1、Stage 2、Stage 3 标为完成，将 Stage 4 明确标为未开始。

## Findings

- **P0：无。**
- **P1：无。**

第三阶段的范围是 codec 与 mask-aware resize 的实现和代表帧验证。全量 Depth-only 裁剪/导出、联合 manifest、导出后内容有效性重算属于 Stage 4，不应被当作 Stage 3 缺失。
