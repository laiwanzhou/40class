# X3D-S Train12/Val2 Partial1 Matched Result

> Development-only matched capacity ablation. This is not unbiased OOF and does not replace canonical Phase 4/5 evidence.

Decision: `human_review_regression`

| Metric | Partial1 | Partial2 | Delta |
|---|---:|---:|---:|
| accuracy | 0.503086 | 0.552469 | -0.049383 |
| macro_f1 | 0.399940 | 0.418651 | -0.018711 |
| worst_user_accuracy | 0.451128 | 0.458647 | -0.007519 |

Selected epoch: `13`. Train Accuracy: `0.922846`. Train-minus-validation gap: `0.419759`.

## Per User

| User | Accuracy | Delta vs partial2 |
|---|---:|---:|
| user21 | 0.451128 | -0.007519 |
| user22 | 0.539267 | -0.078534 |

## Duration

| Frames | N | Accuracy | Delta vs partial2 |
|---|---:|---:|---:|
| <=13 | 118 | 0.474576 | -0.042373 |
| 14-32 | 145 | 0.531034 | -0.041379 |
| 33-64 | 47 | 0.510638 | -0.042553 |
| >64 | 14 | 0.428571 | -0.214286 |

## Matched Prediction Diagnosis

The models disagree on 94/324 trials. Partial1-only correct: 14; partial2-only correct: 30; both wrong: 131.

Partial1 NLL is 1.954141; partial2 NLL is 1.655127.

The greater-than-two-point Accuracy regression rule is triggered. Preserve all artifacts and require human review; do not automatically launch another experiment.
