# Depth_Color + IR dual-stem pose-ROI hard-action expert

## Pairing and comparison validity

The audit parsed the absolute timestamp and frame ID from every filename; it never paired sorted array positions. Of 744 hard-subset samples, 743 are exactly aligned. `train__c10__user1__2-1-1` has 37 legacy frame names without absolute timestamps and was excluded, leaving 595 train / 148 validation samples. Therefore the old 596/148 E1 run is context only; the retrained 595/148 Depth-only model is the strict baseline.

## Architecture and probes

Depth and native one-channel IR are cropped with the same cached pose ROI coordinates before resizing. The pretrained RGB stem is unchanged; the IR stem is initialized by the exact RGB-kernel channel mean. A zero-initialized 1x1 gate starts at 0.5, followed by one shared MobileNetV3-Small body, the inherited ROI attention, frame projection, and GRU.

- Initial gate mean: 0.500000.
- Real batch shapes: Depth [4, 24, 4, 3, 192, 192]; IR [4, 24, 4, 1, 192, 192].
- All required gradient groups nonzero: True.
- Real batch=4 peak allocated GPU memory: 6228.43 MiB; no batch reduction was needed.
- Fourteen representative samples (11 action representatives plus far, near, and off-center geometry cases) were rendered at three temporal positions each and inspected.

## Main results

| Experiment | Train/val | Best/completed | Accuracy | Macro-F1 | Weighted F1 | Val loss | Zero-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Depth-only common subset | 595/148 | 6/12 | 0.378378 | 0.259677 | 0.330001 | 2.134853 | 5 |
| Depth+IR paired | 595/148 | 10/16 | 0.405405 | 0.363335 | 0.386433 | 2.980224 | 3 |
| Old E1 (context only) | 596/148 | 8/14 | 0.371622 | 0.311394 | 0.365204 | 2.479343 | 2 |

Strict V2 - Depth-only delta: Accuracy +0.027027, Macro-F1 +0.103658, weighted F1 +0.056432, validation loss +0.845370. Zero-F1 classes decrease from 5 to 3.
Training time: Depth-only 714.3s total / 59.5s mean epoch; Depth+IR 1330.3s total / 83.1s mean epoch.

## IR contribution diagnostics

| Validation input | Accuracy | Macro-F1 | Weighted F1 | Changed predictions vs paired |
| --- | ---: | ---: | ---: | ---: |
| Normal paired IR | 0.405405 | 0.363335 | 0.386433 | 0/148 (0.00%) |
| IR normalized-zero masked | 0.277027 | 0.178100 | 0.224639 | 104/148 (70.27%) |
| IR sample-shuffled, no self-pair | 0.250000 | 0.243088 | 0.244068 | 78/148 (52.70%) |

Both destructive diagnostics substantially reduce performance, so V2 uses aligned IR content rather than benefiting only from added parameters.

## Per-class strict comparison

| Action | Support | Depth F1 | Depth+IR F1 | Delta | Masked F1 | Shuffled F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Drink_water | 13 | 0.500000 | 0.357143 | -0.142857 | 0.300000 | 0.105263 |
| Eat_food | 32 | 0.500000 | 0.489796 | -0.010204 | 0.392523 | 0.196078 |
| Stir_drinks | 16 | 0.470588 | 0.642857 | +0.172269 | 0.363636 | 0.461538 |
| Wipe_bowls | 6 | 0.000000 | 0.625000 | +0.625000 | 0.352941 | 0.285714 |
| Make_a_phone_call | 8 | 0.000000 | 0.173913 | +0.173913 | 0.000000 | 0.000000 |
| Turn_pages | 9 | 0.230769 | 0.592593 | +0.361823 | 0.000000 | 0.333333 |
| Watch_TV | 6 | 0.000000 | 0.000000 | +0.000000 | 0.000000 | 0.250000 |
| Play_games | 12 | 0.000000 | 0.000000 | +0.000000 | 0.000000 | 0.125000 |
| Take_a_selfie | 14 | 0.592593 | 0.615385 | +0.022792 | 0.000000 | 0.486486 |
| Take_medicine | 17 | 0.000000 | 0.000000 | +0.000000 | 0.300000 | 0.055556 |
| Take_body_temperature | 15 | 0.562500 | 0.500000 | -0.062500 | 0.250000 | 0.375000 |

## Target confusion directions

| Direction | Depth-only | Depth+IR |
| --- | ---: | ---: |
| Take_medicine -> Eat_food | 1/17 (5.88%) | 0/17 (0.00%) |
| Eat_food -> Take_medicine | 0/32 (0.00%) | 0/32 (0.00%) |
| Watch_TV -> Play_games | 0/6 (0.00%) | 0/6 (0.00%) |
| Play_games -> Watch_TV | 0/12 (0.00%) | 0/12 (0.00%) |
| Make_a_phone_call -> Take_a_selfie | 1/8 (12.50%) | 0/8 (0.00%) |
| Take_a_selfie -> Make_a_phone_call | 0/14 (0.00%) | 1/14 (7.14%) |
| Take_body_temperature -> Make_a_phone_call | 2/15 (13.33%) | 6/15 (40.00%) |
| Make_a_phone_call -> Take_body_temperature | 2/8 (25.00%) | 3/8 (37.50%) |
| Stir_drinks -> Wipe_bowls | 0/16 (0.00%) | 3/16 (18.75%) |
| Wipe_bowls -> Stir_drinks | 6/6 (100.00%) | 0/6 (0.00%) |

## Difficult-class prediction distributions

- Watch_TV: Make_a_phone_call 6.
- Play_games: Turn_pages 5; Make_a_phone_call 4; Eat_food 3.
- Take_medicine: Take_body_temperature 5; Make_a_phone_call 4; Take_a_selfie 4; Drink_water 2; Stir_drinks 1; Turn_pages 1.
- Wipe_bowls: Wipe_bowls 5; Drink_water 1.

## Learned gates and weight budget

Mean Depth gate by view (IR weight is `1-g`): global 0.519527, upper 0.519801, left hand 0.519478, right hand 0.519172. Mean binary gate entropy is 0.692384 versus maximum ln(2)=0.693147. The scalar summaries stay near balanced fusion rather than collapsing to one modality; destructive IR tests show that this balance carries useful IR information.
ROI attention means: upper 0.197003, left hand 0.328822, right hand 0.474175.

### Per-class modality gates

| Action | Global Depth | Upper Depth | Left-hand Depth | Right-hand Depth |
| --- | ---: | ---: | ---: | ---: |
| Drink_water | 0.519692 | 0.519912 | 0.519894 | 0.518764 |
| Eat_food | 0.519659 | 0.519776 | 0.519264 | 0.519119 |
| Stir_drinks | 0.519857 | 0.520260 | 0.519744 | 0.519666 |
| Wipe_bowls | 0.520128 | 0.520772 | 0.520376 | 0.520376 |
| Make_a_phone_call | 0.519346 | 0.519772 | 0.520023 | 0.519136 |
| Turn_pages | 0.519070 | 0.519050 | 0.518577 | 0.518584 |
| Watch_TV | 0.519280 | 0.519067 | 0.518127 | 0.518101 |
| Play_games | 0.518921 | 0.519111 | 0.518220 | 0.518202 |
| Take_a_selfie | 0.519533 | 0.519831 | 0.519194 | 0.519061 |
| Take_medicine | 0.519599 | 0.520210 | 0.520414 | 0.519904 |
| Take_body_temperature | 0.519238 | 0.519522 | 0.519798 | 0.519485 |

Watch_TV and Play_games do not show a meaningful shift toward global IR, and both remain zero-F1. Drink_water and Eat_food also keep nearly balanced hand gates; the learned scalar means are not strongly class-specific.
YOLO pose weights plus V2 classifier checkpoint total 11449529 bytes (10.92 MiB), below 100 MB.

## Decision

V2 is a positive exploratory result on the exact common subset: aggregate metrics improve, zero-F1 classes decrease, and both IR masking and sample shuffling cause large degradations. Six of the eight continuation criteria are met. TV/gaming remain unresolved and Drink_water declines, so extending this exact expert unchanged to all 40 classes is not yet recommended as a replacement mainline. It is reasonable to retain it as an auxiliary visual branch for a later 40-class fusion experiment.

The next targeted experiment should prioritize a long-duration screen-activity branch for Watch_TV / Play_games. Hand-head or joint-two-hand ROIs are secondary candidates for medicine/eating/drinking and stirring/wiping, but the present IR result already improves Wipe_bowls and phone-related local interactions, while screen activities remain the clear gap.

The experiment remains isolated on its branch; no existing task03 fusion or training framework was replaced.
