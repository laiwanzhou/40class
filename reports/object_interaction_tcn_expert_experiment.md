# Object interaction residual TCN expert experiment

## Integrity

- Epoch 8 predictions reproduced exactly: True.
- Train/validation samples: 2320/590.
- Target train/validation samples: 880/222.
- Competition test data read: no.
- Artificial keypoint completion: no.

## Base versus final

| model | accuracy | macro_f1 | weighted_f1 | target16_macro_f1 | hand_head7_macro_f1 | table7_macro_f1 | screen2_macro_f1 | control10_macro_f1 | non_target_macro_f1 | stable_easy_macro_f1 | zero_f1_count | target16_zero_f1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_epoch8 | 0.444068 | 0.339342 | 0.427365 | 0.188270 | 0.209546 | 0.183518 | 0.130435 | 0.419095 | 0.440057 | 0.762169 | 5 | 4 |
| final_expert | 0.462712 | 0.349791 | 0.442325 | 0.225633 | 0.214335 | 0.265683 | 0.125000 | 0.405276 | 0.432563 | 0.758475 | 6 | 5 |

- Best target16 epoch: 11.
- Rescued/harmed/net rescue: 22/11/11.
- Mean absolute expert residual: 2.779093.
- Peak allocated/reserved VRAM: 3408.24/4302.00 MB.
- Training seconds: 10197.75.
- Total inference parameter bytes: 18606788.

## Target-class F1 changes

| class_id | action_name | base_f1 | final_f1 | delta_f1 | net_rescue |
| --- | --- | --- | --- | --- | --- |
| 8 | Take_and_use_tableware | 0.000000 | 0.342857 | 0.342857 | 6 |
| 10 | Stir_drinks | 0.242424 | 0.413793 | 0.171369 | 2 |
| 24 | Use_a_mobile_phone | 0.083333 | 0.250000 | 0.166667 | 2 |
| 21 | Read_documents | 0.333333 | 0.451613 | 0.118280 | 4 |
| 11 | Peel_fruits | 0.226415 | 0.318182 | 0.091767 | 1 |
| 6 | Drink_water | 0.411765 | 0.444444 | 0.032680 | -1 |
| 22 | Turn_pages | 0.315789 | 0.333333 | 0.017544 | 0 |
| 27 | Take_a_selfie | 0.294118 | 0.307692 | 0.013575 | -1 |
| 25 | Watch_TV | 0.000000 | 0.000000 | 0.000000 | 0 |
| 19 | Make_a_phone_call | 0.000000 | 0.000000 | 0.000000 | 0 |
| 18 | Write | 0.000000 | 0.000000 | 0.000000 | 0 |
| 26 | Play_games | 0.260870 | 0.250000 | -0.010870 | -1 |
| 7 | Eat_food | 0.415094 | 0.387097 | -0.027998 | 1 |
| 37 | Take_medicine | 0.068966 | 0.000000 | -0.068966 | -1 |
| 39 | Take_body_temperature | 0.193548 | 0.111111 | -0.082437 | -2 |
| 14 | Wipe_bowls | 0.166667 | 0.000000 | -0.166667 | -1 |

## Largest harmed classes

| class_id | action_name | base_f1 | final_f1 | delta_f1 | harmed_count |
| --- | --- | --- | --- | --- | --- |
| 14 | Wipe_bowls | 0.166667 | 0.000000 | -0.166667 | 1 |
| 2 | Comb_hair | 0.300000 | 0.157895 | -0.142105 | 0 |
| 38 | Massage_oneself | 0.260870 | 0.142857 | -0.118012 | 1 |
| 39 | Take_body_temperature | 0.193548 | 0.111111 | -0.082437 | 2 |
| 37 | Take_medicine | 0.068966 | 0.000000 | -0.068966 | 1 |
| 9 | Pour_drinks | 0.470588 | 0.428571 | -0.042017 | 0 |
| 7 | Eat_food | 0.415094 | 0.387097 | -0.027998 | 0 |
| 28 | Jog_in_place | 0.642857 | 0.620690 | -0.022167 | 0 |
| 31 | Do_stretching_exercises | 0.428571 | 0.409091 | -0.019481 | 0 |
| 26 | Play_games | 0.260870 | 0.250000 | -0.010870 | 2 |

## Decision

- Recommend Router training next: yes.
- Overall Macro-F1 delta: +0.010449; stable-easy mean F1 delta: -0.003695.
