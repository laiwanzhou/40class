# INVALID: label-tuned IR + Depth VideoMAE P3-R1 result

This report is retained only as audit evidence. Each held-out user fold was
also used for epoch selection, so its pooled score is optimistic and must not
be used for model selection. See the fixed-epoch P3-R1 result instead.

- Selected by train-user grouped CV: `margin_routes`
- Validation Accuracy: `0.711688`
- Decision: `reject`
- Validation users entered training/CV selection: `False`

## Route controls

| Route | Accuracy | Macro-F1 | Worst-user |
|---|---:|---:|---:|
| full_hard2 | 0.722078 | 0.637747 | 0.701493 |
| ir_hard2 | 0.703896 | 0.632187 | 0.686567 |
| full_hard3 | 0.714286 | 0.628680 | 0.686567 |
| full_hard4 | 0.709091 | 0.622720 | 0.686567 |
| full_soft | 0.709091 | 0.622720 | 0.686567 |
| full_group_context10 | 0.714286 | 0.628562 | 0.691542 |
| full_group_context25 | 0.711688 | 0.626107 | 0.686567 |
| full_group_context50 | 0.701299 | 0.612020 | 0.671642 |
| ir_group_context25 | 0.703896 | 0.635725 | 0.686567 |
| full_context_only | 0.641558 | 0.542110 | 0.631841 |
| full_wrists_only | 0.722078 | 0.637747 | 0.701493 |

## Old uniform fusion versus current hard Top-2

- Uniform Accuracy: `0.714286`
- Current Accuracy: `0.722078`
- Rescue/Harm: `24/21`

| Class | Support | Uniform recall | Current recall | Delta |
|---|---:|---:|---:|---:|
| Wash_face | 3 | 1.000000 | 1.000000 | +0.000000 |
| Brush_teeth | 6 | 0.333333 | 0.333333 | +0.000000 |
| Comb_hair | 5 | 0.800000 | 0.800000 | +0.000000 |
| Take_off_clothes | 6 | 0.833333 | 0.833333 | +0.000000 |
| Wipe_hands | 9 | 0.666667 | 0.555556 | -0.111111 |
| Put_on_clothes | 6 | 1.000000 | 1.000000 | +0.000000 |
| Drink_water | 15 | 0.800000 | 0.800000 | +0.000000 |
| Eat_food | 25 | 0.640000 | 0.600000 | -0.040000 |
| Take_and_use_tableware | 27 | 0.259259 | 0.518519 | +0.259259 |
| Pour_drinks | 18 | 0.888889 | 0.777778 | -0.111111 |
| Stir_drinks | 18 | 0.777778 | 0.555556 | -0.222222 |
| Peel_fruits | 12 | 1.000000 | 1.000000 | +0.000000 |
| Sweep_the_floor | 9 | 0.444444 | 0.666667 | +0.222222 |
| Mop_the_floor | 6 | 1.000000 | 1.000000 | +0.000000 |
| Wipe_bowls | 6 | 0.833333 | 1.000000 | +0.166667 |
| Wipe_windows_and_tables | 10 | 0.800000 | 1.000000 | +0.200000 |
| Fold_clothes | 3 | 1.000000 | 1.000000 | +0.000000 |
| Tap_the_keyboard | 7 | 1.000000 | 1.000000 | +0.000000 |
| Write | 3 | 0.000000 | 0.000000 | +0.000000 |
| Make_a_phone_call | 3 | 1.000000 | 0.666667 | -0.333333 |
| Check_the_time | 15 | 0.600000 | 0.666667 | +0.066667 |
| Read_documents | 9 | 0.333333 | 0.111111 | -0.222222 |
| Turn_pages | 6 | 0.833333 | 1.000000 | +0.166667 |
| Listen_to_music_with_headphones | 9 | 0.555556 | 0.555556 | +0.000000 |
| Use_a_mobile_phone | 9 | 0.000000 | 0.000000 | +0.000000 |
| Watch_TV | 3 | 0.000000 | 0.000000 | +0.000000 |
| Play_games | 2 | 0.000000 | 0.000000 | +0.000000 |
| Take_a_selfie | 3 | 1.000000 | 1.000000 | +0.000000 |
| Jog_in_place | 6 | 1.000000 | 1.000000 | +0.000000 |
| Do_squats | 6 | 0.666667 | 0.833333 | +0.166667 |
| Do_jumping_jacks | 6 | 1.000000 | 1.000000 | +0.000000 |
| Do_stretching_exercises | 6 | 0.333333 | 0.333333 | +0.000000 |
| Stand_up | 6 | 0.500000 | 0.000000 | -0.500000 |
| Lie_down | 9 | 1.000000 | 1.000000 | +0.000000 |
| Sit_down | 30 | 0.933333 | 0.866667 | -0.066667 |
| Do_lunges | 3 | 0.666667 | 1.000000 | +0.333333 |
| Walk | 43 | 0.976744 | 0.953488 | -0.023256 |
| Take_medicine | 6 | 0.500000 | 0.500000 | +0.000000 |
| Massage_oneself | 3 | 0.666667 | 1.000000 | +0.333333 |
| Take_body_temperature | 8 | 0.500000 | 0.875000 | +0.375000 |

## Final rerankers

| Candidate | OOF Acc | Val Acc | Rescue/Harm |
|---|---:|---:|---:|
| logit_only | 0.937984 | 0.722078 | 5/5 |
| routes_no_margin | 0.940052 | 0.724675 | 5/4 |
| margin_routes | 0.941085 | 0.711688 | 3/7 |
