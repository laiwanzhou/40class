# Pose-guided ROI hard action visual expert

## Matched experiment

E0 and E1 use the same 11 hard classes, fold_0 14/4 users, continuous 24-frame windows, ImageNet-pretrained MobileNetV3-Small, frame projection, single-layer GRU, optimizer, seed, image size, batch size, epoch limit, and early stopping. E1 alone receives the upper-body, left-hand, and right-hand Depth_Color crops localized from aligned IR pose coordinates.

| Experiment | Train/val | Best epoch | Completed | Accuracy | Macro-F1 | Weighted F1 | Val loss | Zero-F1 | GPU MB alloc/reserved | Mean epoch | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 global-only | 596/148 | 16 | 22 | 0.304054 | 0.262462 | 0.294938 | 4.050372 | 2 | 675.97/1014.00 | 24.88s | 0h 09m 07.4s |
| E1 pose-ROI | 596/148 | 8 | 14 | 0.371622 | 0.311394 | 0.365204 | 2.479343 | 2 | 2590.34/3908.00 | 42.90s | 0h 10m 00.6s |

## Primary comparison

- E1 - E0 Accuracy: +0.067568.
- E1 - E0 Macro-F1: +0.048932.
- E1 - E0 weighted F1: +0.070266.
- E1 - E0 zero-F1 class count: +0.
- Improved classes (6): Drink_water, Eat_food, Stir_drinks, Turn_pages, Take_a_selfie, Take_body_temperature.
- Declined classes (3): Wipe_bowls, Make_a_phone_call, Take_medicine.

## Per-class metrics

| Action | E0 F1 | E1 F1 | Delta | E1 precision | E1 recall | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Drink_water | 0.266667 | 0.514286 | +0.247619 | 0.409091 | 0.692308 | 13 |
| Eat_food | 0.363636 | 0.530612 | +0.166976 | 0.764706 | 0.406250 | 32 |
| Stir_drinks | 0.450000 | 0.545455 | +0.095455 | 0.529412 | 0.562500 | 16 |
| Wipe_bowls | 0.461538 | 0.181818 | -0.279720 | 0.200000 | 0.166667 | 6 |
| Make_a_phone_call | 0.100000 | 0.062500 | -0.037500 | 0.041667 | 0.125000 | 8 |
| Turn_pages | 0.166667 | 0.600000 | +0.433333 | 0.545455 | 0.666667 | 9 |
| Watch_TV | 0.000000 | 0.000000 | +0.000000 | 0.000000 | 0.000000 | 6 |
| Play_games | 0.000000 | 0.000000 | +0.000000 | 0.000000 | 0.000000 | 12 |
| Take_a_selfie | 0.400000 | 0.428571 | +0.028571 | 0.428571 | 0.428571 | 14 |
| Take_medicine | 0.250000 | 0.117647 | -0.132353 | 0.117647 | 0.117647 | 17 |
| Take_body_temperature | 0.428571 | 0.444444 | +0.015873 | 0.380952 | 0.533333 | 15 |

## Target confusion directions

| Direction | E0 | E1 | Rate delta |
| --- | ---: | ---: | ---: |
| Take_medicine -> Eat_food | 0/17 (0.00%) | 0/17 (0.00%) | +0.00% |
| Eat_food -> Take_medicine | 2/32 (6.25%) | 5/32 (15.62%) | +9.38% |
| Take_medicine -> Drink_water | 0/17 (0.00%) | 0/17 (0.00%) | +0.00% |
| Drink_water -> Take_medicine | 1/13 (7.69%) | 2/13 (15.38%) | +7.69% |
| Watch_TV -> Play_games | 0/6 (0.00%) | 0/6 (0.00%) | +0.00% |
| Play_games -> Watch_TV | 0/12 (0.00%) | 0/12 (0.00%) | +0.00% |
| Make_a_phone_call -> Take_a_selfie | 0/8 (0.00%) | 1/8 (12.50%) | +12.50% |
| Take_a_selfie -> Make_a_phone_call | 0/14 (0.00%) | 0/14 (0.00%) | +0.00% |
| Take_body_temperature -> Make_a_phone_call | 0/15 (0.00%) | 6/15 (40.00%) | +40.00% |
| Make_a_phone_call -> Take_body_temperature | 3/8 (37.50%) | 2/8 (25.00%) | -12.50% |
| Stir_drinks -> Wipe_bowls | 3/16 (18.75%) | 2/16 (12.50%) | -6.25% |
| Wipe_bowls -> Stir_drinks | 2/6 (33.33%) | 3/6 (50.00%) | +16.67% |

## Required-group interpretation

- Medicine/eating/drinking: `Eat_food` improves +0.166976 and `Drink_water` +0.247619, but `Take_medicine` declines -0.132353. E1 medicine predictions are: Take_a_selfie 7; Make_a_phone_call 4; Take_body_temperature 3; Take_medicine 2; Turn_pages 1. The key medicine ambiguity is not resolved.
- TV/gaming: both remain F1=0. E1 `Watch_TV` predictions are: Make_a_phone_call 2; Drink_water 1; Turn_pages 1; Take_medicine 1; Take_body_temperature 1; E1 `Play_games` predictions are: Make_a_phone_call 7; Turn_pages 3; Drink_water 2.
- Phone/selfie: `Make_a_phone_call` changes -0.037500 and `Take_a_selfie` +0.028571; phone remains weak and one phone sample is newly confused with selfie.
- Temperature/phone: `Take_body_temperature` changes +0.015873, but temperature-to-phone errors rise to 40.00%.
- Stir/wipe: `Stir_drinks` improves +0.095455, whereas `Wipe_bowls` declines -0.279720; this pair is not jointly improved.
- Page turning: `Turn_pages` improves +0.433333, from F1 0.166667 to 0.600000.

## ROI quality and gating

- Full hard-subset IR cache: person 98.13%, left wrist 98.05%, right wrist 97.28%, at least one wrist 98.13%.
- Validation upper-body fallback rate: 0.00%; combined hand fallback rate: 0.06%.
- Mean ROI area ratios: upper 11.40%, left hand 1.80%, right hand 1.82%.
- Validation ROI gate means: upper 0.394305, left hand 0.309661, right hand 0.296034; entropy 0.809031 versus uniform ln(3)=1.098612.
- Attention does not collapse to one ROI. It has a moderate upper-body preference while both hand regions retain substantial weight; the entropy is below uniform, so the gate is selective rather than permanently fixed.

## Inference weights

- YOLO11n-pose locator: 6255593 bytes (5.97 MiB).
- E1 classifier checkpoint, including MobileNet and GRU: 5191176 bytes (4.95 MiB).
- Total required inference weights: 11446769 bytes (10.92 MiB), below 100 MB.

## Decision

The route improves aggregate Accuracy and Macro-F1, but it does not meet all predefined value criteria: zero-F1 classes do not decrease, TV/gaming remain unresolved, and medicine/wipe-bowls regress. Do not integrate it into the existing fusion system without a new, separately justified experiment.
