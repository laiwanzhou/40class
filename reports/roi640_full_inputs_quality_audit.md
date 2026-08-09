# ROI640 全量训练输入质量独立审计

## 1. 审计范围与结论

- 数据目录：`D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_full_inputs_256`
- 姿态缓存：YOLO11n-pose `imgsz=640`，人物检测阈值 `0.25`；13条补救 trial 使用 `0.01`；关键点有效阈值 `0.25`
- 审计对象：2,910条 train/val 样本、84,906个原始配对帧、509,436张导出PNG
- 六路视图：IR context/left/right/relation，Depth context/relation
- 未读取 competition test，未训练模型，未修改任何导出图片

**独立结论：当前导出在文件完整性上通过，在模型输入层面“有条件通过”。** 人物上下文视图稳定，IR联合交互ROI对多数小物品动作确实包含手、物品或交互对象；但单侧腕部ROI存在明显缺失和少量严重时序异常，尤其右侧ROI。正式训练必须使用有效掩码，并应先人工处理2条P0错误、抽查627条P1高风险项。640方案目前没有证据表明必须整体废弃，但也不能只依据全局有效率宣称物品语义已经充分保留。

## 2. 文件与清单完整性

| 检查项 | 结果 |
|---|---:|
| `all_frame_inputs.csv`帧行数 | 84,906 |
| `roi_frame_audit.csv`视图行数 | 509,436 |
| 每帧恰好六路视图 | 84,906 / 84,906 |
| 帧键重复 | 0 |
| 帧-视图键重复 | 0 |
| 缺失文件 | 0 |
| 无法解码PNG | 0 |
| 非256x256图片 | 0 |
| invalid但不是黑色占位图 | 0 |
| valid但完全黑图 | **2** |

IR为单通道PNG，Depth_Color为三通道PNG，数量均与清单一致。机械完整性很好，两个有效标记错误是全量扫描中唯一的P0问题。

## 3. 六路视图总体质量

| 视图 | Valid率 | 边界触碰率 | valid低信息图数 |
|---|---:|---:|---:|
| IR人物上下文 | 100.00% | 68.00% | 0 |
| Depth人物上下文 | 100.00% | 68.00% | 0 |
| IR左方向交互 | 95.61% | 13.44% | 19 |
| IR右方向交互 | **86.97%** | 10.18% | 38 |
| IR自适应联合 | 96.38% | 17.82% | 2 |
| Depth自适应联合 | 96.38% | 17.82% | 13 |

人物上下文的边界触碰主要来自扩展框达到原图边缘，不能单独解释为人物被裁掉。局部ROI触边更值得检查，因为手中物品可能位于腕点外侧。全部六路中共72张valid近常量/低信息图，占比约0.014%，总体很少，但其中2张是完全黑图。

左右ROI重复抑制工作正常：73,186帧左右均有效时，没有一帧的左右框IoU达到0.80；另有2,368帧出现高重叠，但至少一路已经被置为invalid。也就是说，模型不会把高度重复的左右局部图同时当作两条有效证据。

## 4. 关键类别风险

### 4.1 局部视图有效率最低的动作

- `Lie_down`：左71.57%，右70.49%，联合71.66%。问题是腕部长期整体失效，不只是单腕不可见。
- `Walk`：左74.06%，右69.80%，联合79.62%。该类贡献了最多的连续失效与跳变异常。
- `Wipe_hands`：右71.22%。
- `Wash_face`：右72.73%。
- `Take_a_selfie`：左90.11%，右72.98%，联合90.34%。
- `Comb_hair`：右76.86%。
- `Wipe_windows_and_tables`：右77.80%。
- `Write`：左92.64%，右81.27%，联合92.64%。

### 4.2 小物品动作

| 动作 | IR左Valid | IR右Valid | IR联合Valid | 联合ROI触边 |
|---|---:|---:|---:|---:|
| Check_the_time | 97.86% | 93.73% | 97.86% | 3.79% |
| Turn_pages | 94.74% | 86.72% | 94.83% | 5.17% |
| Take_medicine | 99.78% | 87.91% | 99.78% | 15.62% |
| Use_a_mobile_phone | 98.57% | 96.09% | 98.70% | 3.84% |
| Play_games | 99.57% | 97.98% | 99.57% | 1.16% |
| Drink_water | 98.57% | 86.40% | 98.63% | 13.66% |
| Take_and_use_tableware | 98.86% | 84.66% | 98.86% | 27.46% |
| Write | 92.64% | 81.27% | 92.64% | 7.44% |
| Tap_the_keyboard | 94.42% | 91.36% | 94.42% | 5.58% |
| Stir_drinks | 99.60% | 93.96% | 99.60% | **33.60%** |
| Pour_drinks | 99.00% | 87.42% | 99.00% | 21.54% |

实际像素抽查显示：IR联合ROI中，手表、手机、饮水容器、书页和餐具通常能进入画面；`Take_medicine`中的药片仍太小，`Play_games`的设备轮廓对比度较弱。对应Depth裁剪能够提供身体、手臂与桌面的几何关系，但小物品身份明显弱于IR。这与“IR高容量外观分支、Depth轻量几何残差分支”的设计假设一致。

## 5. 时序质量

- 278条样本至少有一路局部视图连续invalid达到8帧，其中train 242条、val 36条。
- 共有576个样本-视图组合最长连续失效不少于8帧；relation在IR/Depth中共享坐标，因此这里不应解释为576个独立姿态故障。
- 在相邻两帧均valid的前提下，发现454个ROI中心跳变达到原图对角线20%以上的事件。
- 高风险样本主要集中在`Walk`（72条样本）、`Wipe_hands`（16）、`Drink_water`（15）、`Eat_food`（13）、`Pour_drinks`（13）、`Peel_fruits`（11）、`Lie_down`（10）、`Wipe_windows_and_tables`（10）、`Take_and_use_tableware`（10）。

最严重的连续失效包括：

| 动作 | sample_id | 视图 | 最长失效/总帧 |
|---|---|---|---:|
| Lie_down | `train__c33__user16__5-3-1` | 四路局部 | 84 / 122 |
| Wipe_windows_and_tables | `train__c15__user6__3-3-1` | 四路局部 | 69 / 73 |
| Walk | `train__c36__user2__3-2-2` | right | 46 / 65 |
| Take_a_selfie | `train__c27__user20__7-2-1` | right | 45 / 45 |
| Peel_fruits | `train__c11__user6__5-2-2` | right | 41 / 41 |
| Eat_food | `train__c07__user20__3-1-2` | right | 40 / 54 |
| Pour_drinks | `train__c09__user2__6-1-1` | right | 40 / 50 |

这些片段仍有100%有效的人物上下文可用，但不能假定每个时间点都存在可靠的局部交互证据。TCN输入必须携带逐帧、逐视图mask；不能把黑色占位图仅做归一化后直接输入。

## 6. Train/Val分布提示

部分动作的局部有效率具有明显用户划分差异。例如右ROI：`Take_a_selfie`为train 64.24% / val 93.92%，`Write`为70.92% / 100%，`Wipe_windows_and_tables`为73.23% / 94.56%；反向差异中，`Do_lunges`为train 95.52% / val 77.97%。这可能让模型利用“局部视图是否缺失”作为用户或类别捷径，必须在评估中记录按用户的mask统计。

数据支持数也高度不均衡：`Watch_TV`仅train 6 / val 6条，`Fold_clothes`为12 / 9，`Do_lunges`为21 / 5。它们的泛化风险不能归因于ROI质量本身。

## 7. 必须人工审查的清单

完整清单：`reports\roi640_quality_audit\roi640_manual_review_candidates.csv`。其中含动作、split、sample_id、user_id、源帧索引、视图、原因、相对路径和绝对路径。

| 优先级 | 图片行数 | 唯一样本 | 动作数 | 处理要求 |
|---|---:|---:|---:|---|
| P0 | 2 | 2 | 2 | 训练前必须确认并修正valid标记或生成逻辑 |
| P1 | 627 | 237 | 40 | 高风险定量异常，训练前至少逐项快速目视 |
| P2 | 630 | 42 | 21 | 小物品动作按train/val分层语义抽检 |

P1由268张局部触边图、160张长连续失效代表帧、160张最大跳变帧和39张低信息图组成。P2不是统计错误，而是像素指标无法回答“药片、遥控器、手机等是否语义可辨”，因此每个小物品动作分别抽取一个train和一个val样本的首/中/末帧及关键视图。

两条P0为：

1. `Walk`，`train__c36__user8__2-2-1`，源帧0，`depth_relation`：valid=1但完全黑图。
2. `Write`，`train__c18__user3__3-2-1`，源帧28，`ir_right`：valid=1但完全黑图。

## 8. 是否可以正式训练

建议状态为**条件放行，不立即将当前版本认定为最终输入基线**：

1. 先处理2条P0，并人工审查P1；P0至少应在加载时被像素空图检测再次置为invalid。
2. 确认训练加载器在归一化后按`*_valid`把无效张量清零，同时将mask传入空间融合与TCN。
3. 对278条长连续失效样本优先做640/1280姿态对照，而不是仅比较全局人物检出率。
4. 若1280能显著降低这些片段的连续失效和跳变，才值得重建全量缓存；若只提高人物检出率而局部物品覆盖不变，640已经足够。
5. 正式实验需要按动作和用户同时记录局部mask率，防止模型学习缺失模式捷径。

当前640方案最可靠的是人物上下文和联合ROI冗余；最薄弱的是右方向ROI及少数姿态长期失效片段。对小物品识别而言，IR确实比Depth保留更多外观信息，但“进入裁剪框”不等于“物品足够清晰”，P2语义抽检仍不可省略。

## 9. 审计产物

- `reports\roi640_quality_audit\roi640_quality_audit_metrics.json`
- `reports\roi640_quality_audit\roi640_quality_by_action_view.csv`
- `reports\roi640_quality_audit\roi640_temporal_quality_by_sample_view.csv`
- `reports\roi640_quality_audit\roi640_left_right_overlap.csv`
- `reports\roi640_quality_audit\roi640_manual_review_candidates.csv`
- `reports\roi640_quality_audit\roi640_pixel_metrics.csv`

审计脚本：`scripts\audit_exported_roi_dataset.py`。脚本只读取导出目录，并支持复用像素统计缓存；未修改数据集。
