# IMU Random Forest action performance report

## Scope and interpretation

This report describes the finalized compact Random Forest on the held-out fold-0 validation split. It uses the exact-match seed `20260725` reproduction run: 573 validation samples from 40 actions, with no overlap with the 2,184 training samples. The full-data production model has no held-out validation set, so its resubstitution metrics are intentionally not used here.

Overall validation performance was:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.424084 (243 / 573) |
| Macro precision | 0.386825 |
| Macro recall | 0.362099 |
| Macro F1 | 0.344430 |
| Weighted F1 | 0.403319 |

All actions are ranked by F1 descending. Ties are resolved by recall descending, precision descending, and label index ascending. `Predicted` is the total number of validation samples assigned to that action, not its true support. `Primary confusion` lists the most frequent wrong prediction for samples whose true class is the action on that row; tied destinations are all shown.

Small-support classes need care: for example, rank 1 `Lie_down` and rank 4 `Do_jumping_jacks` each have only six validation samples. `Walk`, with 67 samples and F1 `0.8905`, is the strongest result supported by a comparatively large validation set.

## Complete action ranking

| Rank | Label | Action | Support | Correct | Predicted | Precision | Recall | F1 | Primary confusion |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 33 | Lie_down | 6 | 5 | 5 | 1.0000 | 0.8333 | 0.9091 | Sit_down (1) |
| 2 | 36 | Walk | 67 | 61 | 70 | 0.8714 | 0.9104 | 0.8905 | Sit_down (3) |
| 3 | 0 | Wash_face | 12 | 10 | 11 | 0.9091 | 0.8333 | 0.8696 | Brush_teeth (1); Wipe_hands (1) |
| 4 | 30 | Do_jumping_jacks | 6 | 6 | 8 | 0.7500 | 1.0000 | 0.8571 | — |
| 5 | 1 | Brush_teeth | 10 | 8 | 12 | 0.6667 | 0.8000 | 0.7273 | Wipe_hands (1); Watch_TV (1) |
| 6 | 34 | Sit_down | 30 | 22 | 34 | 0.6471 | 0.7333 | 0.6875 | Walk (3) |
| 7 | 32 | Stand_up | 23 | 14 | 18 | 0.7778 | 0.6087 | 0.6829 | Do_squats (3); Sit_down (3) |
| 8 | 28 | Jog_in_place | 11 | 8 | 13 | 0.6154 | 0.7273 | 0.6667 | Mop_the_floor (1); Do_squats (1); Do_jumping_jacks (1) |
| 9 | 5 | Put_on_clothes | 15 | 11 | 20 | 0.5500 | 0.7333 | 0.6286 | Wipe_hands (2) |
| 10 | 29 | Do_squats | 15 | 11 | 21 | 0.5238 | 0.7333 | 0.6111 | Jog_in_place (3) |
| 11 | 31 | Do_stretching_exercises | 18 | 11 | 20 | 0.5500 | 0.6111 | 0.5789 | Put_on_clothes (3) |
| 12 | 23 | Listen_to_music_with_headphones | 17 | 9 | 15 | 0.6000 | 0.5294 | 0.5625 | Eat_food (3) |
| 13 | 3 | Take_off_clothes | 11 | 5 | 8 | 0.6250 | 0.4545 | 0.5263 | Put_on_clothes (2); Do_stretching_exercises (2) |
| 14 | 12 | Sweep_the_floor | 9 | 6 | 14 | 0.4286 | 0.6667 | 0.5217 | Mop_the_floor (1); Watch_TV (1); Walk (1) |
| 15 | 20 | Check_the_time | 12 | 7 | 18 | 0.3889 | 0.5833 | 0.4667 | Drink_water (2) |
| 16 | 17 | Tap_the_keyboard | 16 | 9 | 29 | 0.3103 | 0.5625 | 0.4000 | Stir_drinks (4) |
| 17 | 4 | Wipe_hands | 15 | 9 | 33 | 0.2727 | 0.6000 | 0.3750 | Watch_TV (3) |
| 18 | 16 | Fold_clothes | 9 | 2 | 2 | 1.0000 | 0.2222 | 0.3636 | Take_off_clothes (2); Wipe_hands (2) |
| 19 | 10 | Stir_drinks | 16 | 7 | 24 | 0.2917 | 0.4375 | 0.3500 | Pour_drinks (3) |
| 20 | 15 | Wipe_windows_and_tables | 6 | 2 | 8 | 0.2500 | 0.3333 | 0.2857 | Sweep_the_floor (2) |
| 21 | 14 | Wipe_bowls | 6 | 1 | 3 | 0.3333 | 0.1667 | 0.2222 | Watch_TV (2) |
| 22 | 19 | Make_a_phone_call | 8 | 1 | 1 | 1.0000 | 0.1250 | 0.2222 | Eat_food (3) |
| 23 | 2 | Comb_hair | 8 | 1 | 2 | 0.5000 | 0.1250 | 0.2000 | Wipe_hands (2); Check_the_time (2) |
| 24 | 22 | Turn_pages | 9 | 2 | 12 | 0.1667 | 0.2222 | 0.1905 | Peel_fruits (2); Check_the_time (2) |
| 25 | 9 | Pour_drinks | 16 | 3 | 17 | 0.1765 | 0.1875 | 0.1818 | Drink_water (2); Stir_drinks (2); Tap_the_keyboard (2); Check_the_time (2); Watch_TV (2) |
| 26 | 38 | Massage_oneself | 12 | 2 | 12 | 0.1667 | 0.1667 | 0.1667 | Wipe_hands (5) |
| 27 | 11 | Peel_fruits | 18 | 3 | 22 | 0.1364 | 0.1667 | 0.1500 | Watch_TV (4) |
| 28 | 6 | Drink_water | 13 | 2 | 20 | 0.1000 | 0.1538 | 0.1212 | Eat_food (3) |
| 29 | 39 | Take_body_temperature | 15 | 1 | 3 | 0.3333 | 0.0667 | 0.1111 | Pour_drinks (2); Watch_TV (2); Massage_oneself (2) |
| 30 | 37 | Take_medicine | 17 | 1 | 3 | 0.3333 | 0.0588 | 0.1000 | Wipe_hands (4); Watch_TV (4) |
| 31 | 21 | Read_documents | 15 | 1 | 7 | 0.1429 | 0.0667 | 0.0909 | Turn_pages (6) |
| 32 | 7 | Eat_food | 31 | 2 | 36 | 0.0556 | 0.0645 | 0.0597 | Watch_TV (5) |
| 33 | 8 | Take_and_use_tableware | 13 | 0 | 10 | 0.0000 | 0.0000 | 0.0000 | Wipe_windows_and_tables (3) |
| 34 | 13 | Mop_the_floor | 6 | 0 | 7 | 0.0000 | 0.0000 | 0.0000 | Sweep_the_floor (3) |
| 35 | 18 | Write | 11 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | Stir_drinks (3); Tap_the_keyboard (3) |
| 36 | 24 | Use_a_mobile_phone | 16 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | Eat_food (6) |
| 37 | 25 | Watch_TV | 6 | 0 | 29 | 0.0000 | 0.0000 | 0.0000 | Eat_food (3) |
| 38 | 26 | Play_games | 11 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | Eat_food (8) |
| 39 | 27 | Take_a_selfie | 14 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | Eat_food (5) |
| 40 | 35 | Do_lunges | 4 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | Mop_the_floor (1); Do_squats (1); Sit_down (1); Walk (1) |

## What the ranking shows

The clearest strengths are large whole-body or posture-changing actions. `Walk`, `Sit_down`, `Stand_up`, `Jog_in_place`, `Do_squats`, and `Do_jumping_jacks` all rank in the top ten. Several strongly instrumented personal-care actions also work well, particularly `Wash_face` and `Brush_teeth`.

The weakest region is dominated by visually or contextually defined actions whose wrist/body motion can resemble many alternatives: `Watch_TV`, `Play_games`, `Take_a_selfie`, `Use_a_mobile_phone`, `Write`, and `Take_and_use_tableware`. Floor-exercise discrimination is not uniformly strong either: `Do_squats` performs well, while `Do_lunges` has no correct validation prediction, though its support is only four.

Several classes behave as prediction sinks:

- `Eat_food` is predicted 36 times but is correct only twice; it is also the primary wrong destination for `Play_games`, `Use_a_mobile_phone`, `Take_a_selfie`, `Make_a_phone_call`, and `Drink_water`.
- `Watch_TV` is predicted 29 times with zero correct predictions. It absorbs errors from multiple hand-centric actions, so its zero precision is driven by both complete miss rate and heavy false positives.
- `Wipe_hands` is predicted 33 times for 15 true samples, producing recall `0.6000` but precision only `0.2727`.
- `Tap_the_keyboard` is predicted 29 times for 16 true samples, again showing over-prediction despite moderate recall.

These results support using the IMU classifier as one fusion input rather than treating it as equally informative for all 40 actions. The production model was trained on all 2,757 labeled samples, but this ranking remains the appropriate generalization evidence because it comes from the held-out fold-0 run.

## Provenance

Source directory:

```text
D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU_stage2_training\fold_0\experiments\random_forest_finalization_v1\reproducibility_seed20260725
```

| Source file | SHA-256 |
| --- | --- |
| `per_class_metrics.csv` | `b1e0f4a8401e53d272c8eef737d48a1f2d0901dd29c05ad04b7679b5170f3961` |
| `confusion_matrix.csv` | `374ad7241033c13ece992f5d34a2063c2c2800303e7ec3841d7e4df606cf3ae5` |
| `validation_outputs.npz` | `f4a15995c2c21c183fb885e3ae8480ca320214563e6df135b16e021d1843b9fa` |
| `training_summary.json` | `b6b3de01e0e82738847ad57baec68bc6a6080a537471a92fec231d53f9ff0c53` |

The formal reproducibility comparison established zero prediction mismatches and identical validation metrics between the source and fresh runs. This report performs no retraining and modifies no external artifact.
