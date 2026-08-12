# 六模态共享 OOF Fold 可用性审计

## 结论

- `train14_oof_3fold.json` 的 outer validation 用户恰好分割 train14，每个用户只出现一次；每个 inner fit/validation 也严格分割对应 outer train。
- 全部 18 个用户是否在每个模态至少有一个 manifest-present trial：**True**。
- 六模态、三折、四个 scope 中，缺少整个 assigned user 的 manifest-present 情况：**0 行**。
- 六模态、三折、四个 scope 中，类别少于 40 的情况：**24 行**。这主要反映固定用户组本身的类别支持，不等价于模态特有缺失。
- 其中相对 canonical user scope 额外丢失类别的模态行：**5 行**；应与固定用户组本身缺类分开解释。

因此可以复用 IR 的三套 user-level fold。共享的是 user ownership；每个模态仍按自身 present/usable 状态形成稀疏样本集，不替换缺失用户。

冻结 fold 文件 SHA256：`2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`。

## 全局用户覆盖

| modality   |   present_trials |   minimum_trials_for_one_user |   maximum_trials_for_one_user |
|:-----------|-----------------:|------------------------------:|------------------------------:|
| Depth      |             2931 |                           100 |                           201 |
| IMU        |             2903 |                            91 |                           201 |
| IR         |             2933 |                           100 |                           201 |
| Radar      |             2914 |                           100 |                           201 |
| Skeleton   |             2931 |                           100 |                           201 |
| Thermal    |             2891 |                            95 |                           195 |

每个模态的 `minimum_trials_for_one_user` 都大于 0。因此这里没有“IR 有某用户、Skeleton 或其他模态完全没有该用户”的情况。

## Skeleton Master Clean v1 在共享折中的覆盖

|   fold | scope            |   assigned_user_count |   users_with_data_count |   trial_count |   class_count | missing_class_ids   |
|-------:|:-----------------|----------------------:|------------------------:|--------------:|--------------:|:--------------------|
|      0 | outer_train      |                     9 |                       9 |          1529 |            40 |                     |
|      0 | outer_validation |                     5 |                       5 |           812 |            39 | 25                  |
|      0 | inner_fit        |                     6 |                       6 |          1052 |            40 |                     |
|      0 | inner_validation |                     3 |                       3 |           477 |            39 | 25                  |
|      1 | outer_train      |                    10 |                      10 |          1674 |            40 |                     |
|      1 | outer_validation |                     4 |                       4 |           667 |            40 |                     |
|      1 | inner_fit        |                     7 |                       7 |          1224 |            40 |                     |
|      1 | inner_validation |                     3 |                       3 |           450 |            37 | 25|26|33            |
|      2 | outer_train      |                     9 |                       9 |          1479 |            40 |                     |
|      2 | outer_validation |                     5 |                       5 |           862 |            40 |                     |
|      2 | inner_fit        |                     6 |                       6 |           953 |            40 |                     |
|      2 | inner_validation |                     3 |                       3 |           526 |            39 | 25                  |

train14 的每个 outer/inner scope 都保留全部 assigned users。此前 6 个 clean 后空 trial 属于 heldout4 的 `user17`，不在这套 train14 三折中。

唯一模态特有的类别缺口出现在 fold1 inner-validation：canonical 用户组包含 class 33，但 Depth、IMU、IR、Radar、Skeleton 对该类均无 path，只有 Thermal 有 user5 的 6 个 class-33 trial。固定用户组自身还缺 class 25/26。因此这个 inner-validation 对五个模态覆盖 37 类，对 Thermal 覆盖 38 类。

## 口径

- `manifest_present`：对应 path 字段非空，只证明原始 trial 被登记；不证明 expert preprocessing 成功。
- `master_clean_usable_non_strict_oof`：Skeleton Master Clean v1 至少保留一帧。它只用于数据开发统计，不能替代 fold-specific strict-OOF identity curation。
- 本审计未读取 competition test，未训练模型。

## 文件

- `user_modality_availability.csv`：18 users × 六模态 present trial 数，以及 Skeleton master-clean usable trial 数。
- `fold_scope_modality_availability.csv`：fold × outer/inner scope × modality 的用户、trial、类别覆盖及缺口。
