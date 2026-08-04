# Full 40-class Depth_Color / IR pairing audit

Pairing uses parsed `(absolute timestamp, frame ID)` keys. Sorted array positions are never used.

- Manifest samples: 3036.
- Fold samples before modality filtering: train 2427, validation 609.
- Samples with both Depth_Color and IR paths: 2931.
- Completely paired and readable samples: 2910.
- Exceptional samples: 21.
- Final usable samples: train 2320, validation 590.
- Strictly paired frames: 84906.

## Class support

| class_id | action_name | train | validation |
| ---: | --- | ---: | ---: |
| 0 | Wash_face | 32 | 12 |
| 1 | Brush_teeth | 36 | 12 |
| 2 | Comb_hair | 45 | 9 |
| 3 | Take_off_clothes | 30 | 11 |
| 4 | Wipe_hands | 55 | 15 |
| 5 | Put_on_clothes | 39 | 15 |
| 6 | Drink_water | 107 | 13 |
| 7 | Eat_food | 122 | 32 |
| 8 | Take_and_use_tableware | 82 | 14 |
| 9 | Pour_drinks | 114 | 16 |
| 10 | Stir_drinks | 100 | 16 |
| 11 | Peel_fruits | 88 | 18 |
| 12 | Sweep_the_floor | 50 | 9 |
| 13 | Mop_the_floor | 49 | 6 |
| 14 | Wipe_bowls | 29 | 6 |
| 15 | Wipe_windows_and_tables | 43 | 9 |
| 16 | Fold_clothes | 12 | 9 |
| 17 | Tap_the_keyboard | 69 | 17 |
| 18 | Write | 27 | 11 |
| 19 | Make_a_phone_call | 35 | 8 |
| 20 | Check_the_time | 82 | 12 |
| 21 | Read_documents | 56 | 15 |
| 22 | Turn_pages | 50 | 9 |
| 23 | Listen_to_music_with_headphones | 58 | 17 |
| 24 | Use_a_mobile_phone | 32 | 16 |
| 25 | Watch_TV | 6 | 6 |
| 26 | Play_games | 28 | 12 |
| 27 | Take_a_selfie | 27 | 14 |
| 28 | Jog_in_place | 26 | 11 |
| 29 | Do_squats | 62 | 15 |
| 30 | Do_jumping_jacks | 38 | 6 |
| 31 | Do_stretching_exercises | 67 | 18 |
| 32 | Stand_up | 60 | 24 |
| 33 | Lie_down | 27 | 6 |
| 34 | Sit_down | 115 | 31 |
| 35 | Do_lunges | 21 | 5 |
| 36 | Walk | 260 | 71 |
| 37 | Take_medicine | 51 | 17 |
| 38 | Massage_oneself | 50 | 12 |
| 39 | Take_body_temperature | 40 | 15 |

## Exceptions

| sample_id | split | class_id | action_name | reason |
| --- | --- | ---: | --- | --- |
| train__c05__user3__1-1-1 | train | 5 | Put_on_clothes | depth_unparsed=73; ir_unparsed=73 |
| train__c05__user3__1-1-2 | train | 5 | Put_on_clothes | depth_unparsed=44; ir_unparsed=44 |
| train__c05__user3__1-1-3 | train | 5 | Put_on_clothes | depth_unparsed=34; ir_unparsed=34 |
| train__c08__user1__2-1-1 | train | 8 | Take_and_use_tableware | depth_unparsed=50; ir_unparsed=50 |
| train__c09__user1__2-1-1 | train | 9 | Pour_drinks | depth_unparsed=67; ir_unparsed=67 |
| train__c10__user1__2-1-1 | train | 10 | Stir_drinks | depth_unparsed=37; ir_unparsed=37 |
| train__c12__user2__3-1-1 | train | 12 | Sweep_the_floor | depth_unparsed=56; ir_unparsed=56 |
| train__c12__user3__6-2-1 | train | 12 | Sweep_the_floor | depth_unparsed=59; ir_unparsed=59 |
| train__c12__user3__6-2-2 | train | 12 | Sweep_the_floor | depth_unparsed=67; ir_unparsed=67 |
| train__c12__user3__6-2-3 | train | 12 | Sweep_the_floor | depth_unparsed=40; ir_unparsed=40 |
| train__c13__user2__3-1-1 | train | 13 | Mop_the_floor | depth_unparsed=29; ir_unparsed=29 |
| train__c16__user2__1-1-1 | train | 16 | Fold_clothes | depth_unparsed=140; ir_unparsed=140 |
| train__c16__user2__1-1-2 | train | 16 | Fold_clothes | depth_unparsed=95; ir_unparsed=95 |
| train__c16__user2__1-1-3 | train | 16 | Fold_clothes | depth_unparsed=42; ir_unparsed=42 |
| train__c21__user3__6-2-2 | train | 21 | Read_documents | depth_unparsed=31; ir_unparsed=31 |
| train__c21__user3__6-2-3 | train | 21 | Read_documents | depth_unparsed=24; ir_unparsed=24 |
| train__c34__user3__6-2-2 | train | 34 | Sit_down | depth_unparsed=6; ir_unparsed=6 |
| train__c36__user1__2-1-1 | train | 36 | Walk | depth_unparsed=17; ir_unparsed=17 |
| train__c36__user3__6-2-1 | train | 36 | Walk | depth_unparsed=25; ir_unparsed=25 |
| train__c36__user3__6-2-2 | train | 36 | Walk | depth_unparsed=20; ir_unparsed=20 |
| train__c36__user3__6-2-3 | train | 36 | Walk | depth_unparsed=17; ir_unparsed=17 |
