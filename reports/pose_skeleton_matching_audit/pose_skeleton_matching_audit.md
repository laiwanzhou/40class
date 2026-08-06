# 图像姿态—三维 Skeleton 匹配可行性诊断

## 范围与防泄漏约束

- 只读取 `fold_0` 的 14 个 train 用户；未读取 validation 用户或 competition test。
- 清单 train trial：2427；具备既有 Depth/IR 严格配对与 YOLO 缓存：2320；最终纳入：2152；排除：168；纳入帧：66268。
- 官方 Skeleton 的 17 点拓扑由连续骨链确认为 H36M 风格顺序，而 YOLO 为 COCO-17。实验只映射两者可明确对应的 12 个肩、肘、腕、髋、膝、踝关节；YOLO 面部点以及 Skeleton 的 hip-center/spine/thorax/neck/head 不强行配对。
- YOLO 置信度阈值为 0.25；低置信度关节保持缺失，未插值、未人工补全。
- 每个被评估用户使用其余 13 个 train 用户拟合一套固定线性弱透视映射；映射不按帧拟合。每个正/负匹配只在 ±2 帧内搜索时移。
- 2.5D 的 z 来自 `Depth_Color` Jet 伪彩图在 YOLO 点周围的相对色序，**不是毫米深度或相机坐标**。因此该分支只能回答现有缓存条件下伪彩相对深度是否增加对应证据。

## 总体匹配指标

PCK 使用躯干归一化距离阈值 0.2；coverage 的分母包含所有帧与全部 12 个公共关节。

| mode   |   trials |   frames |   coverage |    pck |   rmse |   angle_correlation |   velocity_correlation |   matching_score |
|:-------|---------:|---------:|-----------:|-------:|-------:|--------------------:|-----------------------:|-----------------:|
| 25d    |     2152 |  30.7937 |     0.8037 | 0.4355 | 0.4903 |              0.5071 |                 0.2279 |           0.5585 |
| 2d     |     2152 |  30.7937 |     0.8288 | 0.7537 | 0.1959 |              0.6512 |                 0.3047 |           0.7568 |

## 同 trial 检索

具备至少 48 个有效“重采样帧×关节”观测的 IR/YOLO query，在同一用户的全部官方 Skeleton trial 中检索；低于门槛的 query 不进入检索分母。这样不会借用用户身份完成检索；gallery 同时包含同动作其他 trial 和不同动作 trial。

| mode   |   queries |   mean_gallery_size |   top1 |   top5 |    mrr |   positive_vs_all_controls_auc |
|:-------|----------:|--------------------:|-------:|-------:|-------:|-------------------------------:|
| 25d    |      2085 |            157.4379 | 0.4168 | 0.6897 | 0.5451 |                         0.6216 |
| 2d     |      2117 |            157.4549 | 0.5933 | 0.8318 | 0.6994 |                         0.8415 |

## 正匹配与负对照

分数越高越相似。`same_action_other_trial`、`different_action`、`time_shuffle`、`left_right_swap` 均使用与正样本相同的映射和时移搜索。

| mode   | control                 |    n |   mean |    std |    p05 |   median |    p95 |
|:-------|:------------------------|-----:|-------:|-------:|-------:|---------:|-------:|
| 25d    | different_action        | 2085 | 0.7069 | 0.0885 | 0.5461 |   0.7178 | 0.8372 |
| 25d    | left_right_swap         | 2085 | 0.7257 | 0.0879 | 0.5635 |   0.7357 | 0.8450 |
| 25d    | positive_same_trial     | 2085 | 0.7713 | 0.0974 | 0.5864 |   0.7896 | 0.8930 |
| 25d    | same_action_other_trial | 2072 | 0.7431 | 0.0949 | 0.5686 |   0.7538 | 0.8732 |
| 25d    | time_shuffle            | 2085 | 0.7609 | 0.0961 | 0.5806 |   0.7775 | 0.8858 |
| 2d     | different_action        | 2117 | 0.7151 | 0.0836 | 0.5706 |   0.7183 | 0.8464 |
| 2d     | left_right_swap         | 2117 | 0.7571 | 0.0733 | 0.6359 |   0.7584 | 0.8726 |
| 2d     | positive_same_trial     | 2117 | 0.8772 | 0.0477 | 0.8003 |   0.8863 | 0.9301 |
| 2d     | same_action_other_trial | 2104 | 0.8025 | 0.0906 | 0.6158 |   0.8264 | 0.9069 |
| 2d     | time_shuffle            | 2117 | 0.8442 | 0.0616 | 0.7443 |   0.8538 | 0.9165 |

## 薄弱用户与动作

### 用户

| mode   | user_id   |   trials |   coverage |    pck |   matching_score |   retrieval_top1 |   reciprocal_rank |
|:-------|:----------|---------:|-----------:|-------:|-----------------:|-----------------:|------------------:|
| 25d    | user1     |      143 |     0.7134 | 0.4212 |           0.5148 |           0.3817 |            0.4841 |
| 25d    | user3     |      134 |     0.7982 | 0.3865 |           0.5171 |           0.3721 |            0.4866 |
| 25d    | user2     |      153 |     0.7762 | 0.3976 |           0.5289 |           0.5000 |            0.6115 |
| 2d     | user1     |      143 |     0.7188 | 0.7464 |           0.7011 |           0.5344 |            0.6319 |
| 2d     | user5     |       99 |     0.8005 | 0.6928 |           0.7270 |           0.5657 |            0.6699 |
| 2d     | user8     |      153 |     0.7572 | 0.7108 |           0.7282 |           0.7450 |            0.8176 |

### 动作

| mode   |   class_id | action_name      |   trials |   coverage |    pck |   matching_score |   retrieval_top1 |   reciprocal_rank |
|:-------|-----------:|:-----------------|---------:|-----------:|-------:|-----------------:|-----------------:|------------------:|
| 25d    |         16 | Fold_clothes     |       12 |     0.9565 | 0.2556 |           0.4635 |           0.8333 |            0.8783 |
| 25d    |         17 | Tap_the_keyboard |       68 |     0.8527 | 0.3357 |           0.4677 |           0.0462 |            0.1705 |
| 25d    |         18 | Write            |       25 |     0.7820 | 0.3124 |           0.4703 |           0.2500 |            0.3985 |
| 25d    |         33 | Lie_down         |       25 |     0.7501 | 0.3337 |           0.4800 |           0.4400 |            0.5913 |
| 25d    |          0 | Wash_face        |       27 |     0.5613 | 0.4152 |           0.4881 |           0.7619 |            0.8532 |
| 2d     |         18 | Write            |       25 |     0.8042 | 0.5638 |           0.6372 |           0.3750 |            0.4672 |
| 2d     |         22 | Turn_pages       |       48 |     0.8533 | 0.6347 |           0.6502 |           0.2609 |            0.4345 |
| 2d     |         33 | Lie_down         |       25 |     0.7537 | 0.6136 |           0.6550 |           0.8400 |            0.9080 |
| 2d     |         17 | Tap_the_keyboard |       68 |     0.8567 | 0.6389 |           0.6607 |           0.1231 |            0.2448 |
| 2d     |         21 | Read_documents   |       51 |     0.8658 | 0.6801 |           0.6686 |           0.4286 |            0.5705 |

完整的 40 类和 14 用户结果分别见 `per_action_metrics.csv` 与 `per_user_metrics.csv`；逐关节结果见 `per_joint_metrics.csv`。

## 代表失败案例

| mode   | sample_id                 | action_name      | user_id   |   coverage |   pck |   matching_score |   rank | failure_reason     |
|:-------|:--------------------------|:-----------------|:----------|-----------:|------:|-----------------:|-------:|:-------------------|
| 25d    | train__c17__user1__3-2-1  | Tap_the_keyboard | user1     |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 25d    | train__c17__user1__3-2-2  | Tap_the_keyboard | user1     |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 25d    | train__c20__user1__3-2-1  | Check_the_time   | user1     |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 25d    | train__c20__user1__3-2-2  | Check_the_time   | user1     |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 25d    | train__c21__user1__6-2-2  | Read_documents   | user1     |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 25d    | train__c21__user1__6-2-3  | Read_documents   | user1     |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 25d    | train__c22__user1__6-2-2  | Turn_pages       | user1     |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 25d    | train__c26__user19__6-2-3 | Play_games       | user19    |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 25d    | train__c27__user19__6-2-3 | Take_a_selfie    | user19    |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 25d    | train__c32__user2__4-3-3  | Stand_up         | user2     |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 2d     | train__c17__user1__3-2-1  | Tap_the_keyboard | user1     |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |
| 2d     | train__c17__user1__3-2-2  | Tap_the_keyboard | user1     |     0.0000 |   nan |           0.1750 |    nan | low_joint_coverage |

成功/失败叠图在 `outputs/pose_skeleton_matching_audit/visualizations/`，青色为 IR YOLO，红色为固定映射后的 Skeleton。图片作为本地诊断产物未纳入 Git；索引见 `representative_cases.csv`。

## 最终判断

**适合序列级融合，不支持帧级关节直接融合：trial 身份可检索，但逐帧几何/动态一致性未同时达标。**

判定规则预先固定：帧级融合要求 PCK≥0.60、角度相关≥0.50、速度相关≥0.40、Top-1≥0.50 且正负 AUC≥0.80；序列级融合要求 Top-1≥0.30、MRR≥0.50 且 AUC≥0.75。若只有 2.5D 达标，仍因伪彩深度非度量而降一级解释。

本实验不训练动作分类或融合模型，也没有修改 B2、数据划分和已有缓存。结论只针对现有 IR YOLO 缓存、官方 Skeleton 表示及当前可用的伪彩 Depth。
