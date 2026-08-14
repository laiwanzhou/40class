# X3D-S Fold0 Generalization Tuning

> Development-only fold0 evidence. This is not unbiased OOF and does not replace Phase 4/5 evidence.

Decision: `human_review_regression`

| Metric | Candidate | Target | Canonical fold0 | Delta |
|---|---:|---:|---:|---:|
| accuracy | 0.458750 | 0.630000 | 0.571250 | -0.112500 |
| macro_f1 | 0.352348 | 0.520000 | 0.486918 | -0.134570 |
| worst_user_accuracy | 0.421053 | 0.533800 | 0.533835 | -0.112782 |

## Per User

| User | Accuracy |
|---|---:|
| user18 | 0.483146 |
| user20 | 0.421384 |
| user21 | 0.421053 |
| user3 | 0.422819 |
| user9 | 0.524862 |

## Duration

| Frames | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| <=13 | 225 | 0.453333 | 0.294840 |
| 14-32 | 353 | 0.478754 | 0.313355 |
| 33-64 | 184 | 0.429348 | 0.336104 |
| >64 | 38 | 0.447368 | 0.101774 |
