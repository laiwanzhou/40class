# Skeleton 原始数据全量审计

## 审计边界

- 数据根：`D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\Skeleton`。
- 只读取 competition train 数据根下的 Skeleton；未读取 competition test。
- 清单覆盖全部 3036 个 competition-train trial，并按现有 fold 标注 train/validation，仅用于分组统计，不改变划分。
- 扫描脚本：`scripts/audit_skeleton_raw_dataset.py`。

## 1. 原始结构

|   manifest_rows |   manifest_without_skeleton_path |   declared_path_missing_directory |   zero_JSON_trials |   JSON_files_seen |   valid_17x3_file_records |   unique_time_steps |   duplicate_frame_id_files |   exact_duplicate_frame_id_files |   missing_frame_id_slots |   invalid_frames |   multi_person_frames |   frames_with_nonfinite_points |   frames_with_nonpositive_scores |   frames_with_whole_zero_joint |
|----------------:|---------------------------------:|----------------------------------:|-------------------:|------------------:|--------------------------:|--------------------:|---------------------------:|---------------------------------:|-------------------------:|-----------------:|----------------------:|-------------------------------:|---------------------------------:|-------------------------------:|
|            3036 |                              105 |                                 0 |                105 |             86050 |                     86050 |               85879 |                        171 |                              171 |                      979 |                0 |                  3525 |                              0 |                                0 |                              0 |

| category              | value                     |   frames |
|:----------------------|:--------------------------|---------:|
| top_level_type        | list                      |    86050 |
| selected_person_keys  | keypoint_scores|keypoints |    86050 |
| keypoints_shape       | (17, 3)                   |    86050 |
| keypoint_scores_shape | (17,)                     |    86050 |

每个有效帧文件的直接可证实结构是：顶层 `list`，其中人物记录包含且仅包含 `keypoints` 与 `keypoint_scores`。全部 90458 个人物记录均为 `keypoints[17][3]` 和 `keypoint_scores[17]`，不合格人物记录 0。顶层 list 是人物维，不能把它当时间维；时间维由一个 trial 的多个 JSON 文件组成。

全部人物的 `keypoint_scores` 唯一值为 `[1.0]`。因此它在这份导出中不是有信息量的逐关节置信度或可见性信号；不能据此识别遮挡或 missing joint。

人物数分布：1 人 82525 帧，2 人 2667 帧，3 人 833 帧，4 人 25 帧。共有 3525 个多人物帧。JSON 不含 track ID，且所有 score 相同；旧 loader 的“最高平均 score”选择在多人物帧上实际退化为取列表第一个人，不能保证跨帧身份连续。

### 多候选帧的时间聚集性

按唯一 `frame_id` 去重后，多候选时间步为 3523，分布在 299 个 trial，组成 428 个连续段。连续段定义为相邻 `frame_id` 差 1。

| run_length   |   runs |   frames |   frame_rate |
|:-------------|-------:|---------:|-------------:|
| 1            |    210 |      210 |       0.0596 |
| 2-4          |     83 |      213 |       0.0605 |
| 5-9          |     35 |      235 |       0.0667 |
| 10-19        |     38 |      556 |       0.1578 |
| 20-39        |     42 |     1179 |       0.3347 |
| 40+          |     20 |     1130 |       0.3207 |

- 孤立单帧共有 210 段、210 帧，占全部多候选唯一帧的 5.9608%。
- 长度至少 5 帧的连续段承载 3100 帧，占 87.9932%；长度至少 20 帧的连续段承载 2309 帧，占 65.5407%。
- 220 / 299 个受影响 trial 只有一个连续段；84 个 trial 从首帧到末帧全部为多候选，它们承载 2335 帧，占全部多候选唯一帧的 66.2787%。
- 最长连续段为 76 帧。逐段边界见 `multi_person_temporal_runs.csv`，逐 trial 的段数、最长段和覆盖率见 `trial_inventory.csv`。

## 2. 关节数量、顺序与字段含义

实际关节数是 **17**。数据文件没有 joint-name 元数据。根据 17 点骨链排列、左右肢体连续性以及坐标锚点，顺序与 H36M-17 一致：

| Index | 推断语义 | Index | 推断语义 |
|---:|---|---:|---|
| 0 | pelvis | 9 | neck |
| 1 | right_hip | 10 | head |
| 2 | right_knee | 11 | left_shoulder |
| 3 | right_ankle | 12 | left_elbow |
| 4 | left_hip | 13 | left_wrist |
| 5 | left_knee | 14 | right_shoulder |
| 6 | left_ankle | 15 | right_elbow |
| 7 | spine | 16 | right_wrist |
| 8 | thorax |  |  |

该命名是由拓扑和数值签名建立的强推断，不是 organizer JSON 中明示的字段。逐关节坐标范围、零值率和 score 统计见 `joint_coordinate_statistics.csv`，骨链长度统计见 `inferred_h36m_bone_statistics.csv`。

每个 joint 的三个数值字段只能可靠解释为导出的 `(x, y, z)` 三维姿态坐标；它们不带单位、相机内参、外参、原点或轴定义元数据。

文件 schema、H36M-17 关节顺序和“最低关节高度归零”行为与 MMPose 官方 `human3d`/MotionBERT inferencer 文档高度一致。官方文档也说明预测输出按人物给出 `keypoints`、`keypoint_scores`，并提供关闭高度 rebasing 的选项。这是生成来源的强证据，但数据集没有保存模型配置、版本和执行命令，因此不能把具体生成器视为已被文件本身完全证明。来源：[MMPose inference 文档](https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md)。

## 3. 坐标系诊断

| invariant                        |   rate |
|:---------------------------------|-------:|
| joint 0 x=y exactly zero         | 1.0000 |
| per-frame minimum z exactly zero | 1.0000 |

- 全局绝对范围：x=`[-1.265358, 1.189595]`，y=`[-0.498945, 0.536668]`，z=`[0.000000, 2.177424]`；全轴最大绝对值 `[1.2653584480285645, 0.5366678237915039, 2.177424430847168]`。
- joint 0 的 x/y 被逐帧强制锚定到 0；每帧至少一个 joint 的 z 被强制锚定到 0。最低 z 关节分布见 `minimum_z_joint_distribution.csv`。
- 从身体拓扑看，z 随 pelvis→spine→thorax→neck/head 上升，并在 ankle 附近取零，因此这里的 z 更像**竖直高度轴**，不是常规相机坐标中“离相机的深度 z”。x/y 是以 pelvis 为原点的另外两个轴。
- 结论：这是经过逐帧平移规范化的、单位与尺度未知的 body/world-like 3D pose 表示。它不是图像像素坐标，不是保留人体全局平移的 camera-global XYZ，也不能可靠解释为米或毫米。没有生成器元数据时，x/y 的精确朝向和单位不能再从文件本身唯一恢复。

时间戳相邻间隔统计（ms）：median=100.000，p01=100.000，p99=100.000，min=100.000，max=14900.000。

## 4. Sequence 长度

| scope                   |   trials |    min |    p01 |    p05 |   median |    mean |     p95 |      p99 |      max |
|:------------------------|---------:|-------:|-------:|-------:|---------:|--------:|--------:|---------:|---------:|
| Skeleton-present trials |     2931 | 1.0000 | 4.0000 | 7.0000 |  24.0000 | 29.3002 | 70.0000 | 107.7000 | 236.0000 |

| length_bucket   |   trials |   rate |
|:----------------|---------:|-------:|
| 1               |        2 | 0.0007 |
| 2-3             |        9 | 0.0031 |
| 4-7             |      179 | 0.0611 |
| 8-15            |      632 | 0.2156 |
| 16-31           |     1101 | 0.3756 |
| 32-63           |      799 | 0.2726 |
| 64-95           |      158 | 0.0539 |
| 96-127          |       37 | 0.0126 |
| 128+            |       14 | 0.0048 |

- 这里按 trial 内唯一 `frame_id` 计数。磁盘共有 86050 个 JSON，但 4 个 trial 同时保存纯帧号和时间戳命名的相同数据，形成 171 个逐数组完全一致的重复文件；实际唯一时间步为 85879。
- 完整逐 trial、动作、用户和 fold 分布：`trial_inventory.csv`、`sequence_length_summary.csv` 和 `sequence_length_buckets.csv`。
- 原始 sequence 没有固定 64 帧；64 是旧 loader 的在线线性重采样目标。

## 5. Missing joint 与 invalid frame

- 原始 JSON 没有 joint-valid mask、occlusion 字段或逐关节有效性字段。
- `keypoint_scores` 全量恒定，不能当 missingness mask。
- 非有限坐标、非正 score、整 joint 三轴全零和 malformed frame 的全量统计见上表；逐异常文件见 `frame_anomalies.csv`。
- 清单没有 Skeleton 路径的 trial 与“有目录但 frame 异常”是两种不同缺失，已分别统计。
- 105 个 trial 在 manifest 中没有 Skeleton 路径；已声明的 2931 个 Skeleton 目录全部存在。按 fold、动作和用户的覆盖率见 `skeleton_availability_summary.csv`。唯一 frame-id 序列内部还缺少 979 个编号位置，这些是时间序列缺帧而不是 missing joint。
- 旧 loader 遇到空人物列表或没有 `keypoints` 的帧会制造 `17×3` 全零姿态，并继续参与中心化、缩放和插值；它没有把该帧标成 temporal invalid。这是潜在的数据语义问题。

## 6. 旧 `64×102` 的精确组成

旧实现位于 `src/data/skeleton_dataset.py`，处理顺序如下：

```text
raw poses                       [T, 17, 3]
root = mean(joint 11, joint 12) [T, 1, 3]
centered = poses - root          [T, 17, 3]
scale = RMS(centered over 17×3)  [T, 1, 1]
normalized_pose = centered/scale [T, 17, 3]
velocity[t] = pose[t]-pose[t-1]  [T, 17, 3], velocity[0]=0
concat per joint [x,y,z,vx,vy,vz] [T,17,6]
joint-major flatten             [T,102]
per-channel linear interpolation [64,102]
train-set featurewise z-score    [64,102]
temporal_mask                    [64], 全 True
```

102 的准确来源是 `17 × (3 normalized coordinates + 3 first-order velocities)`。它不包含 bone vector、joint angle、score、valid mask、timestamp 或原始 sequence length。

关键纠正：在 H36M-17 顺序下 joint 11=`left_shoulder`，joint 12=`left_elbow`。所以旧代码的 `(11+12)/2` **不是 hip midpoint**，而是左上肢中点。旧特征实际上围绕左肩/左肘中心化，并且每帧独立 RMS 缩放。随后对 train split 的 102 个通道计算均值/标准差，validation 使用同一组统计。

## 7. 审计结论

1. 原始模态是逐帧 JSON 的单/多人物容器，主 schema 为一个人物、17 个 H36M 风格关节、每关节 3 坐标加一个恒定 score。
2. 坐标已经逐帧去除 x/y pelvis 平移并把最低 z 移到 0；尺度和轴定义未提供，不能作为度量 camera-space Skeleton。
3. 原始长度是可变长，旧 `64` 来自强制插值，不是采集长度。
4. 原始格式没有可信 missing-joint 指示；必须从 schema、有限性和显式规则建立 mask，不能使用恒定 score。多人物帧还需要显式身份跟踪。
5. 4 个 trial 的重复命名文件会被旧 loader 当成额外时间步读入，制造整段重复与连接处伪速度。
6. 旧 `64×102` 的最大语义错误是按 COCO 索引用 11/12 做 hip root。未来 Skeleton 专家应先修正拓扑和 root，再决定是否重采样；不能把旧 baseline 的预处理直接视为正确的 H36M 特征管线。
