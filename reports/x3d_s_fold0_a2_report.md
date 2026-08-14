# X3D-S Fold0 Generalization Tuning

> Development-only fold0 evidence. This is not unbiased OOF and does not replace Phase 4/5 evidence.

Decision: `continue_stage_a`

| Metric | Candidate | Target | Canonical fold0 | Delta |
|---|---:|---:|---:|---:|
| accuracy | 0.607500 | 0.630000 | 0.571250 | +0.036250 |
| macro_f1 | 0.526830 | 0.520000 | 0.486918 | +0.039912 |
| worst_user_accuracy | 0.539326 | 0.533800 | 0.533835 | +0.005491 |

## Per User

| User | Accuracy |
|---|---:|
| user18 | 0.539326 |
| user20 | 0.622642 |
| user21 | 0.593985 |
| user3 | 0.617450 |
| user9 | 0.662983 |

## Duration

| Frames | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| <=13 | 225 | 0.613333 | 0.414906 |
| 14-32 | 353 | 0.631728 | 0.468734 |
| 33-64 | 184 | 0.570652 | 0.456438 |
| >64 | 38 | 0.526316 | 0.166032 |
