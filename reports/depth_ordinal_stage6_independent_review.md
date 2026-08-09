# 第六阶段全量产物独立复审

复审日期：2026-08-09

## 1. 复审范围与结论

本次复审只检查第六阶段已经生成的实际产物，不评估脚本在其他机器或路径下的鲁棒性，也不依赖第五阶段复审结论。未修改数据，未启动第七阶段，未运行训练。

**独立结论：PASS。未发现 P0 或 P1 问题。**

全量资产目录：

`D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256`

## 2. 产物总量与闭合性

| 项目 | 独立复核结果 |
|---|---:|
| trial | 2,910 |
| 原始配对帧 | 84,906 |
| 类别 | 40（class 0-39 均存在） |
| train / val 帧 | 67,216 / 17,690 |
| Depth ordinal 图 | 169,812 |
| Depth pixel-valid mask | 169,812 |
| Depth 数据 PNG 总数 | 339,624 |
| `depth_ordinal_view_audit.csv` 行数 | 169,812 |
| 连续帧预览图 | 12 |

`_SUCCESS` 内容为 `complete`。`export_metadata.json`、`combined_frame_manifest.csv` 和 `depth_ordinal_view_audit.csv` 均存在且可完整读取。metadata 中的 2,910 trial、84,906 帧、两类各 169,812 张图与目录实物一致；`full_dataset_export=true`、`temporal_sampling_applied=false`。

## 3. 帧键、时序与配对独立核验

独立将全量 manifest 与现有 ROI640 基线帧表进行键集合比较，并逐 trial 重建顺序和时间差：

- manifest 重复 `(sample_id, source_frame_index)`：0；
- Depth audit 重复视图键：0；
- manifest 与基线帧键集合差异：0；
- audit 与 84,906 帧 x 2 路 Depth 的期望键集合差异：0；
- 非零起始帧、帧号不连续、timestamp/delta 不一致：均为 0；
- Depth 与 IR 文件名中的 timestamp/frame_id 配对不一致：0；
- source Depth/IR 缺失：0；
- 四路 IR 引用缺失：0；
- 四路 IR 路径落入新导出目录：0；
- 新导出目录内 IR 视图目录：0。

因此，原始完整时序得到保留，Depth/IR 严格配对，四路 IR 仅引用旧资产而没有复制。

## 4. 全量逐像素重算

复审独立读取全部 84,906 张原始 Depth_Color，重新建立 OpenCV JET 反查表；随后根据原 ROI 坐标，为每帧的 `depth_context` 和 `depth_relation` 独立完成 mask-aware crop、resize、letterbox，并与导出的 ordinal 和 pixel-valid mask 逐像素比较。本节不是抽样验证。

| 检查项 | 结果 |
|---|---:|
| 原始图非黑且非 JET 像素 | 0 |
| ordinal 值不一致像素 | 0 |
| pixel-valid mask 不一致像素 | 0 |
| mask 无效区非零 ordinal 像素 | 0 |
| 非 256x256 图 | 0 |
| 非 uint8 图 | 0 |
| 非二值 mask | 0 |
| 低信息 2-of-3 标志重算差异 | 0 |

这证明 339,624 张数据 PNG 的实物内容与原始 Depth_Color、ROI 坐标和既定解码/裁剪规则一致。

## 5. 低信息视图与最终排除

独立重算和 audit 表共同得到：

- ordinal 低信息标志：3,078 路；
- pose-invalid 的全零占位：3,076 路；
- pose-valid 的真实低信息视图：2 路；
- 最终 `content_invalid`：2 路；
- 两条最终无效视图仍被标为 `effective_valid`：0；
- 最终无效键与 Stage 2 两条证据的集合差异：0。

因此 3,076 路只是预期的姿态无效占位；Stage 2 发现的两条 pose-valid 低信息证据均被保留并从有效输入中排除，没有被全量导出流程重新激活。

## 6. 连续帧预览检查

已读取 `_stage5_review` 下全部 12 张 900x540 连续帧 contact sheet。图像均可正常解码，面板按连续三帧展示 source JET、两路 ordinal 和各自 mask。代表性检查显示：

- ordinal 的空间结构、轮廓和前后关系与 source JET 连续一致；
- mask 与无效黑区对齐，未观察到无效区残留数值；
- crop/letterbox 在相邻帧间保持连续；
- 个别 relation 连续为空是 pose-invalid 的显式占位，ordinal 与 mask 同时为零，和 manifest/audit 有效性标志一致，不属于导出损坏。

预览目录：

`D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256\_stage5_review`

## 7. 正式报告一致性

独立结果与以下正式产物逐项一致：

- `reports/depth_ordinal_stage6_audit.json`
- `reports/depth_ordinal_stage6_audit.md`

正式报告中的 trial、帧、split、PNG、低信息拆分、最终无效数量及所有零差异结论均可由实际文件重新得到，未发现报告与 CSV/JSON/目录实物矛盾。

## 8. 边界确认

- manifest 仅包含 `train` 和 `val`；所有关键路径中 competition test 路径计数为 0；
- metadata 和 Stage 6 audit 均记录 `competition_test_read=false`；
- 未发现或启动训练进程，metadata 记录 `training_run=false`；
- handoff 仍明确标记 `Stage 7 not started`；本次复审未执行 Stage 7 代码。

## 9. 最终判定

**PASS，可以将第六阶段视为完整通过。**

- P0：无；
- P1：无；
- 非阻塞观察：部分 relation 视图因姿态无效而为显式全零占位，这是已知且被 mask/effective-valid 正确表达的数据可用性事实，并非产物错误。
