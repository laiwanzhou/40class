# X3D-S Train12/Val2 Partial-Backbone Result

> Development-only evidence on user21/user22. This is not unbiased OOF, does not replace Phase 4/5 evidence, and is not matched to the former fold0 experiments.

Decision: `freeze_result_no_matched_baseline_claim`

## Result

| Metric | Value |
|---|---:|
| Accuracy | 0.552469 |
| Macro-F1 (fixed 40 classes) | 0.418651 |
| Worst-user Accuracy | 0.458647 |
| Selected epoch | 10 |
| Train Accuracy at selected epoch | 0.889780 |
| Train minus validation Accuracy | 0.337310 |

Validation contains 324 usable-IR trials across 36 observed classes. Missing class IDs are `[25, 26, 33, 35]` and contribute zero to fixed-40 Macro-F1.

## Per User

| User | Accuracy |
|---|---:|
| user21 | 0.458647 |
| user22 | 0.617801 |

## Duration

| Frames | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| <=13 | 118 | 0.516949 | 0.260129 |
| 14-32 | 145 | 0.572414 | 0.401081 |
| 33-64 | 47 | 0.553191 | 0.298095 |
| >64 | 14 | 0.642857 | 0.070000 |

## Interpretation

Standalone partial-backbone result on the frozen shared development split. No matched full-backbone run exists on this split, so the effect of partial unfreezing is not causally identified.
The selected checkpoint still has a substantial train-to-validation gap, so this intervention does not by itself resolve cross-user overfitting.
