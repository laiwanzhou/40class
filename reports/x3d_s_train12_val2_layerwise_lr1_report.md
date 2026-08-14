# X3D-S Train12/Val2 Layer-wise LR1 Matched Result

> Development-only matched optimization ablation. This is not unbiased OOF and does not replace canonical Phase 4/5 evidence.

Decision: `human_review_regression`

| Metric | Layer-wise LR1 | Partial2 | Delta |
|---|---:|---:|---:|
| accuracy | 0.521605 | 0.552469 | -0.030864 |
| macro_f1 | 0.409074 | 0.418651 | -0.009578 |
| worst_user_accuracy | 0.451128 | 0.458647 | -0.007519 |

Selected epoch: `17`. Train Accuracy: `0.937876`. Train-minus-validation gap: `0.416271`.

Block learning rates: block4 `3e-6`, block5 `1e-5`, custom head `3e-4`.

## Per User

| User | Accuracy | Delta vs partial2 |
|---|---:|---:|
| user21 | 0.451128 | -0.007519 |
| user22 | 0.570681 | -0.047120 |

## Duration

| Frames | N | Accuracy | Delta vs partial2 |
|---|---:|---:|---:|
| <=13 | 118 | 0.449153 | -0.067797 |
| 14-32 | 145 | 0.544828 | -0.027586 |
| 33-64 | 47 | 0.617021 | +0.063830 |
| >64 | 14 | 0.571429 | -0.071429 |

## Matched Prediction Diagnosis

The models disagree on 89/324 trials. Layer-wise-only correct: 15; partial2-only correct: 25; both wrong: 130.

Layer-wise NLL is 1.802232; partial2 NLL is 1.655127.

## Parameter Drift

| Scope | Layer-wise LR1 | Partial2 | Ratio |
|---|---:|---:|---:|
| block4 | 0.000618 | 0.003632 | 0.170 |
| block5 | 0.004070 | 0.008501 | 0.479 |
| embedding_head | 0.522514 | 0.471703 | 1.108 |
| classifier | 1.166021 | 1.027802 | 1.134 |

Lower block learning rates substantially reduce backbone drift, but custom-head drift increases and the train-validation gap remains severe. Backbone drift alone is not the dominant overfitting mechanism.

The greater-than-two-point Accuracy regression rule is triggered. Preserve all artifacts and require human review; do not automatically launch L2-SP or another experiment.
