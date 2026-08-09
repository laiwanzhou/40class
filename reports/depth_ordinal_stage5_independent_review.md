# Stage 5 独立产物复审

## 结论

**PASS。未发现阻止 Stage 6 全量导出的 P0 或 P1 问题。**

本次仅复审固定运行环境中的实际冒烟产物，不评价脚本的跨环境可移植性或防御性。复审没有启动 Stage 6、没有训练模型，也没有读取 competition test。

## 独立核验范围

- 冒烟根目录：`D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256_smoke_stage5`
- 样本选择表：`reports/depth_ordinal_stage5_selection.csv`
- 原始 Depth_Color、现有 ROI640 IR 引用、ROI 坐标、ordinal 值图及 pixel-valid mask
- 全部 `3,562` 帧、两路 Depth view，共 `14,248` 张正式 PNG
- 12 张连续三帧 contact sheet

## 选择与覆盖

- 选择表共有 `82` 行，且为 `82` 个不同 trial。
- 40 类均包含一条 train 和一条 val，形成完整的 `80` 个 class/split 组合。
- train 为 `42` 条 trial、`1,877` 帧；val 为 `40` 条 trial、`1,685` 帧；合计 `3,562` 帧。
- 额外两条 train trial 均为 Stage 2 已知 Depth 低信息证据：
  - `train__c36__user1__2-1-2`，frame 0，`depth_relation`
  - `train__c36__user8__2-2-1`，frame 0，`depth_relation`
- 两条证据在导出审计中均为 `content_invalid=1`、`effective_valid=0`，没有被后续处理重新激活。

## 全量像素复算

复审没有直接采信 Stage 5 审计 JSON，而是独立构造 OpenCV JET 颜色到 ordinal 值的精确映射，并根据原 ROI 坐标重新执行 mask-aware crop、resize 和 letterbox，对全部导出值图及 mask 逐像素比较。

| 检查项 | 结果 |
|---|---:|
| 正式 PNG 数 | 14,248 / 14,248 |
| 缺失或不可读文件 | 0 |
| shape 非 256x256 | 0 |
| dtype 非 uint8 | 0 |
| 非二值 mask | 0 |
| ordinal 不一致像素 | 0 |
| pixel-valid mask 不一致像素 | 0 |
| mask 无效区非零像素 | 0 |
| 源图非 JET、非黑异常像素 | 0 |

这说明实际保存的 ordinal 值、无效像素约束和 ROI 对齐均与源 Depth_Color 及 ROI 证据一致。

## 时间与模态引用

- 所有 trial 的 `source_frame_index` 均从 0 连续递增，异常为 0。
- 从 Depth 文件名独立解析的 timestamp 和 frame ID 与 manifest 完全一致。
- 相邻 timestamp 重算的帧间隔与 manifest 完全一致。
- 四路 IR 共 `4 x 3,562` 个引用全部存在，每路均覆盖 `3,562` 个不同帧文件。
- IR 路径全部指向原 `roi640_full_inputs_256`，没有任何 IR 文件被复制进冒烟导出目录。

## 低信息口径

`ordinal_content_invalid_2of3` 共标记 `121` 路，拆分结果严格闭合：

- `119` 路为 pose-invalid 后生成的全零占位；
- `2` 路为 pose-valid、但内容低信息的已知 Depth relation 证据；
- `119 + 2 = 121`；
- 最终 `content_invalid` 为 `2` 路，因为 pose-invalid 视图由 pose/effective mask 管理，不应重复计作内容异常。

该口径与实际图片、mask 和审计字段一致，不是漏标或计数矛盾。

## 连续帧人工复核

实际查看了 Walk 已知低信息样本以及 Write、Stir_drinks 等代表性连续三帧 contact sheet。观察结果：

- source JET、context ordinal、relation ordinal 与各自 mask 空间对齐；
- 相邻帧中的人物轮廓、场景结构和局部交互区域连续，无明显乱序；
- 黑色 relation 占位与无效 mask 同步出现，没有以有效视觉内容进入后续模型的迹象；
- ordinal 灰度变化保留了 JET 所表达的相对深度次序。

## 边界确认

- manifest 所有源路径及导出路径中均无 `test` 数据段。
- `competition_test_read=false`，且实际路径检查为 0 条 test 引用。
- 完整输出目录 `roi640_depth_ordinal_256` 尚不存在，确认未执行全量导出。
- 当前没有本实验训练进程；本次复审也未训练模型。

## 分级结论

- P0：无。
- P1：无。
- Stage 6 是否可启动：**是**。

Stage 6 仍应在独立输出目录完成后执行全量计数、逐像素一致性、mask 清零、时间顺序及 IR 引用审计，但 Stage 5 当前产物没有显示需要先返工的证据。
