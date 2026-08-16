# X3D-S User6/User7 Partial2 Matched Reference

> Development-only evidence on user6, user7. This is not unbiased OOF, does not replace Phase 4/5 evidence, and is not matched to the former fold0 experiments.

Decision: `freeze_matched_reference_for_direct_head`

## Result

| Metric | Value |
|---|---:|
| Accuracy | 0.532468 |
| Macro-F1 (fixed 40 classes) | 0.421575 |
| Worst-user Accuracy | 0.532338 |
| Selected epoch | 14 |
| Train Accuracy at selected epoch | 0.952455 |
| Train minus validation Accuracy | 0.419987 |

Validation contains 385 usable-IR trials across 40 observed classes. Missing class IDs are `[]` and contribute zero to fixed-40 Macro-F1.

## Per User

| User | Accuracy |
|---|---:|
| user6 | 0.532338 |
| user7 | 0.532609 |

## Duration

| Frames | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| <=13 | 85 | 0.541176 | 0.206886 |
| 14-32 | 168 | 0.494048 | 0.362899 |
| 33-64 | 105 | 0.590476 | 0.437871 |
| >64 | 27 | 0.518519 | 0.160000 |

## Interpretation

Unchanged partial2 result on the frozen user6/user7 development split; this is the sole matched reference for Direct-Head Generation D.
The selected checkpoint still has a substantial train-to-validation gap, so this intervention does not by itself resolve cross-user overfitting.

## Direct-Head Boundary

This report is the sole matched reference for Generation D. The frozen human-review Accuracy floor is `0.512467532468`.
