# B2-256 same-action cross-user supervised contrastive experiment

## Protocol

- Baseline: B2-256 epoch 25.
- Contrastive checkpoint: best_macro_f1.pt, epoch 18.
- Training stopped after epoch 21 due to validation degradation.
- Loss: CE + 0.1 x cross-user SupCon; temperature 0.1; same-user different-action negative weight 2.0.
- Split: fixed 14 train users / 4 unseen validation users. Competition test read: no.

## Overall comparison

| system | checkpoint_epoch | train_accuracy | val_accuracy | accuracy_gap | train_macro_f1 | val_macro_f1 | macro_f1_gap | val_loss | val_weighted_f1 | top3_accuracy | top5_accuracy | predicted_class_count | zero_f1_class_count | zero_recall_class_count | high_confidence_error_count | high_confidence_error_rate | train_user_probe_accuracy | val_user_probe_accuracy | action_probe_val_accuracy | action_probe_val_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B2 | 25 | 0.976293 | 0.462712 | 0.513581 | 0.975735 | 0.392827 | 0.582908 | 3.409093 | 0.463045 | 0.688136 | 0.784746 | 39 | 2 | 2 | 207 | 0.350847 | 0.292672 | 0.620339 | 0.450847 | 0.381072 |
| CrossUserSupCon | 18 | 0.934914 | 0.423729 | 0.511185 | 0.893338 | 0.341988 | 0.551350 | 3.189900 | 0.417938 | 0.637288 | 0.730508 | 39 | 3 | 3 | 165 | 0.279661 | 0.301293 | 0.552542 | 0.405085 | 0.333061 |

- Accuracy delta: -0.038983.
- Macro-F1 delta: -0.050839.
- Accuracy generalization-gap delta: -0.002396.
- Train-user probe delta: +0.008621.
- Rescued / harmed / net rescue: 56 / 79 / -23.
- All stated success criteria met: no.

## Largest per-class gains

| action_name | f1_b2 | f1_supcon | delta_f1_supcon_minus_b2 | net_rescue |
| --- | --- | --- | --- | --- |
| Wipe_windows_and_tables | 0.000000 | 0.285714 | 0.285714 | 2 |
| Lie_down | 0.666667 | 0.909091 | 0.242424 | 2 |
| Stir_drinks | 0.200000 | 0.370370 | 0.170370 | 2 |
| Read_documents | 0.200000 | 0.315789 | 0.115789 | 4 |
| Do_stretching_exercises | 0.375000 | 0.461538 | 0.086538 | 3 |
| Check_the_time | 0.230769 | 0.315789 | 0.085020 | 3 |
| Comb_hair | 0.190476 | 0.272727 | 0.082251 | 1 |
| Play_games | 0.190476 | 0.250000 | 0.059524 | 1 |
| Sweep_the_floor | 0.444444 | 0.500000 | 0.055556 | 2 |
| Do_jumping_jacks | 0.521739 | 0.571429 | 0.049689 | 0 |

## Largest per-class losses

| action_name | f1_b2 | f1_supcon | delta_f1_supcon_minus_b2 | net_rescue |
| --- | --- | --- | --- | --- |
| Wash_face | 0.956522 | 0.470588 | -0.485934 | -7 |
| Brush_teeth | 0.608696 | 0.235294 | -0.373402 | -5 |
| Make_a_phone_call | 0.352941 | 0.000000 | -0.352941 | -3 |
| Watch_TV | 0.333333 | 0.000000 | -0.333333 | -2 |
| Wipe_hands | 0.620690 | 0.296296 | -0.324393 | -5 |
| Jog_in_place | 0.857143 | 0.666667 | -0.190476 | -2 |
| Take_off_clothes | 0.500000 | 0.320000 | -0.180000 | -2 |
| Pour_drinks | 0.592593 | 0.416667 | -0.175926 | -3 |
| Drink_water | 0.592593 | 0.428571 | -0.164021 | -2 |
| Turn_pages | 0.303030 | 0.142857 | -0.160173 | -4 |

## Interpretation boundary

The user-ID probes are diagnostic linear separability tests, not identity classifiers used by training. The validation-user probe is cross-validated within the four held-out users; the action probe is fitted only on train-user embeddings and evaluated on held-out users.
