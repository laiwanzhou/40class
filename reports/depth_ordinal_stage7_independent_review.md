# Stage 7 独立复审报告

## 结论

**PASS。未发现 P0/P1 问题，Stage 7 可以标记完成。**

本次复审仅检查 Stage 7 的代码、测试和 train/validation `combined_frame_manifest.csv`。未读取 competition test，未启动训练，未开始 Stage 8，也未读取或接入 Skeleton、IMU、Radar。

## 严重度结论

### P0

无。

### P1

无。

### P2

无未解决项。

初次复审提出的两项 P2 已在增量修订中解决：实现报告的测试口径已更新；`ExpertBatchResult.validate()` 已覆盖全部固定输出、全部可选输出的首维，并检查 `quality_mask` 与 `quality` 形状完全一致。

## 增量修订复审

### Spatial activation checkpoint 与 BatchNorm

- 非重入 checkpoint 的重算上下文会保存并恢复空间 encoder 中所有 BatchNorm 的 `running_mean`、`running_var` 和 `num_batches_tracked`，因此重算不会造成第二次统计更新。
- 新增回归测试的一次 forward/backward 中，IR 与 Depth 首层 BatchNorm 的 `num_batches_tracked` 均为 `1`，两条空间分支的首层卷积均获得梯度。
- 独立交叉测试使用相同权重、相同输入分别执行 checkpoint on/off：全部 `43` 个 BatchNorm 层的 tracked count 完全一致，running mean/variance 最大差值均为 `0`；IR 与 Depth 首层梯度均存在且最大差值为 `0`。

### ExpertBatchResult 完整形状校验

- 固定输出 `main_logits`、`embedding`、`quality`、`quality_mask`、`availability` 均检查首维与 `sample_ids` 数量一致。
- 可选输出 `small_gate_logits`、`sequence_features`、`temporal_mask`、`timestamps` 在存在时执行相同首维检查。
- `quality_mask.shape == quality.shape` 被显式强制。
- 独立负向核验分别破坏上述十项约束，全部按预期抛出 `ValueError`。

## 契约核验

| Stage 7 要求 | 独立核验结果 |
|---|---|
| 完整可变长 trial | PASS。dataset 每个 item 保留一个完整 trial，真实长度范围 train `1..236`、val `3..135`。 |
| 无 24/96 抽样、无 Stage A/B 缓存路径 | PASS。活动 dataset/model/trainer/config 中没有固定帧采样或旧缓存阶段；trainer 为单阶段端到端入口。 |
| frame-budget sampler | PASS。train `391` 批、val `106` 批；覆盖全部 `2320/590` 样本，无重复、无遗漏、无批次超过 `256` padded-frame budget。 |
| 仅 batch 内 padding + temporal mask | PASS。真实最短与最长 train trial 合并后长度为 `[1,236]`，mask 和原长度一致；短 trial 尾部 IR、Depth、view-valid 均为零，frame index 为 `-1`。 |
| padding-safe TCN | PASS。每个残差块重新应用 mask，使用逐时间点 channel LayerNorm；独立测试证明追加 padding 不改变有效序列 logits。 |
| 固定 Depth 三通道接口 | PASS。三种表示均为 `[raw, relative, pixel_valid]`；`raw` 的 relative 通道为零，`relative` 的 raw 通道为零，组合版对应通道逐元素一致。 |
| 显式 pixel/view mask | PASS。实数据抽验中无效像素后非零输入为 `0`；mask 通道等于 pixel-valid 与 effective-view mask 的交集；Depth encoder 仍显式使用独立 pixel mask 做池化。 |
| IR/Depth 有效视图与 reliability | PASS。仅有效视图进入空间编码；六路确定性 reliability 进入空间权重并与融合特征拼接后投影。真实 manifest 中 IR context、Depth context 均全有效；局部无效视图得到硬屏蔽。 |
| 独立空间编码与浅层门控 | PASS。IR MobileNetV3-Small 和轻量 Depth encoder 独立编码，Depth 作为带 availability 的门控残差进入 IR 主特征。 |
| 全时序多尺度 TCN | PASS。短分支 dilation `[1,2,4]`，长分支 `[1,2,4,8,16,32]`，完整 temporal mask 贯穿处理和池化。 |
| 40 类主头 + 视觉 small gate | PASS。主输出为 40 类，small gate 为独立二分类辅助头；不存在条件细分类头。 |
| ExpertOutput / ExpertBatchResult | PASS。模型张量和 sample metadata 已分离，`sample_ids` 不进入 forward。 |
| sample_id join / class hash | PASS。显式检查 duplicate、missing/extra、set equality 和 class-map hash，并按 ID 生成重排索引，不依赖 loader 顺序。 |
| `alpha=0` 概率融合恒等 | PASS。实现是真正 probability mixture；以 `atol=0, rtol=0` 验证严格等于视觉 softmax。 |
| 无额外模态或越界执行 | PASS。没有 sensor 输入或融合损失；未读 test、未训练、未运行 Stage 8。 |

## 真实 Manifest 只读验收

输入：`D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256\combined_frame_manifest.csv`

| 项目 | 结果 |
|---|---:|
| 总帧数 | 84,906 |
| train / val 帧数 | 67,216 / 17,690 |
| train / val trial | 2,320 / 590 |
| train / val 用户 | 14 / 4 |
| 用户重叠 | 0 |
| 类别数 | 40 |
| train 长度范围 | 1..236 |
| val 长度范围 | 3..135 |
| class-map hash 一致 | 是 |
| class-map hash | `5dbf6af1a1df88314484ba45f9fccc02ab47a1af2adba951a71e42886ec4c5e5` |
| sampler 遗漏 / 重复 / 超预算批次 | 0 / 0 / 0 |

实数据还确认：IR context 无效帧 `0`，Depth context 无效帧 `0`；IR left/right/relation 无效帧分别为 `3,733 / 11,078 / 3,077`，Depth relation 无效帧 `3,078`。这些局部缺失由 view mask 和 reliability 保留为局部无效，不会使整个时间点失效。

## 独立验证命令

- Stage 3-7 相关测试：`25 passed in 4.03s`。
- 增量相关测试：`8 passed in 3.83s`。
- 全仓测试：`42 passed in 4.91s`。
- `py_compile`：通过。
- `git diff --check`：通过。
- 真实 manifest 的 trial、用户、类别、长度、sampler 和 padding 只读核验：通过。
- 三种 Depth 表示的实数据通道语义与 mask 核验：通过。

## 最终判定

Stage 7 已满足计划中的实现和验收边界，可更新为完成状态。初次复审的两项 P2 已解决，增量修订未引入新的 P0/P1/P2，可进入 Stage 8 的三种 Depth 表示 smoke test。
