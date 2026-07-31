# Depth_Color global + six fixed overlapping patches (fold_0 14/4)

## Result

- Formal split: 2341 train / 590 validation samples; no test data read.
- Completed 21 of 30 requested epochs (patience 6); best Accuracy checkpoint: epoch 15.
- Accuracy: 0.225424; Macro-F1: 0.135058; weighted F1: 0.190494; validation loss: 8.577983.
- Best-epoch history row: Accuracy 0.225424, Macro-F1 0.135058, loss 8.577983.
- Hard-10 Macro-F1: 0.054925; zero-F1 classes: 6/10.

## Runtime

- Actual batch size: 4 (no CUDA OOM; no gradient accumulation used).
- Real optimizer-step probe peak: 2118.96 MB allocated / 2978.00 MB reserved.
- Formal run peak: 2260.54 MB allocated / 3436.00 MB reserved.
- Mean epoch time: 140.51s (range 127.39-157.39s); epoch-loop total: 0h 49m 10.8s.

### Per-epoch timing

| Epoch | Train s | Validation s | Total s |
| ---: | ---: | ---: | ---: |
| 1 | 111.37 | 46.01 | 157.39 |
| 2 | 118.32 | 30.75 | 149.07 |
| 3 | 110.00 | 26.45 | 136.45 |
| 4 | 105.59 | 26.70 | 132.29 |
| 5 | 107.84 | 28.26 | 136.10 |
| 6 | 105.44 | 28.04 | 133.48 |
| 7 | 102.13 | 25.26 | 127.39 |
| 8 | 108.47 | 27.69 | 136.16 |
| 9 | 110.91 | 29.26 | 140.17 |
| 10 | 110.85 | 28.04 | 138.89 |
| 11 | 113.59 | 28.99 | 142.58 |
| 12 | 110.02 | 28.40 | 138.43 |
| 13 | 110.09 | 25.43 | 135.53 |
| 14 | 103.33 | 29.18 | 132.51 |
| 15 | 116.86 | 29.18 | 146.05 |
| 16 | 111.63 | 27.73 | 139.36 |
| 17 | 104.01 | 29.04 | 133.05 |
| 18 | 113.50 | 30.80 | 144.29 |
| 19 | 122.22 | 31.03 | 153.24 |
| 20 | 122.07 | 30.76 | 152.83 |
| 21 | 117.32 | 28.21 | 145.53 |

## Patch attention

- Validation mean weights: P1 0.104176 / P2 0.218574 / P3 0.143596 / P4 0.115635 / P5 0.183672 / P6 0.234347.
- Mean per-frame entropy: 0.998763; uniform maximum ln(6): 1.791759; ratio: 55.74%; mean winning-patch weight: 0.622273.
- Per-frame argmax rates: P1 8.08% / P2 23.73% / P3 12.82% / P4 10.89% / P5 19.15% / P6 25.32%.
- Patch 6 is +0.067681 above uniform 1/6, indicating a persistent spatial preference that should be inspected for background bias.
- Fixed layout: P1/P2/P3 are top left/center/right; P4/P5/P6 are bottom left/center/right.
- The distribution does not collapse to one fixed patch, but P6/P2 are selected most often and the entropy is well below uniform. Because these fixed regions mix person and background, this is a spatial-bias warning rather than proof of useful body-part selection.

## Hard-10

| Class | F1 | Support |
| --- | ---: | ---: |
| Drink_water | 0.055556 | 13 |
| Eat_food | 0.000000 | 32 |
| Stir_drinks | 0.068966 | 16 |
| Wipe_bowls | 0.166667 | 6 |
| Make_a_phone_call | 0.000000 | 8 |
| Turn_pages | 0.258065 | 9 |
| Watch_TV | 0.000000 | 6 |
| Take_a_selfie | 0.000000 | 14 |
| Take_medicine | 0.000000 | 17 |
| Take_body_temperature | 0.000000 | 15 |

## Baseline comparison

No ordinary Depth_Color run on the same 14/4 fold was found. The available baseline used 1931/1000 train/validation samples, versus 2341/590 here; its old 12/6 values are not used as evidence or deltas.
Therefore Accuracy, Macro-F1, weighted-F1, validation-loss, per-class improvement/decline, and zero-F1 deltas versus a same-fold baseline are reported as N/A rather than mixing folds.

## Decision

No. The fixed-patch result should first be compared against a newly trained ordinary Depth_Color baseline on the identical 14/4 fold; adding motion Top-2 now would confound the conclusion.
