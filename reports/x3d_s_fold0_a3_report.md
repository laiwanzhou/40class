# X3D-S Fold0 Generalization Tuning

> Development-only fold0 evidence. This is not unbiased OOF and does not replace Phase 4/5 evidence.

Decision: `continue_stage_a`

| Metric | Candidate | Target | Canonical fold0 | Delta |
|---|---:|---:|---:|---:|
| accuracy | 0.610000 | 0.630000 | 0.571250 | +0.038750 |
| macro_f1 | 0.525564 | 0.520000 | 0.486918 | +0.038646 |
| worst_user_accuracy | 0.528090 | 0.533800 | 0.533835 | -0.005745 |

## Per User

| User | Accuracy |
|---|---:|
| user18 | 0.528090 |
| user20 | 0.647799 |
| user21 | 0.578947 |
| user3 | 0.604027 |
| user9 | 0.685083 |

## Duration

| Frames | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| <=13 | 225 | 0.613333 | 0.387960 |
| 14-32 | 353 | 0.637394 | 0.463851 |
| 33-64 | 184 | 0.570652 | 0.461024 |
| >64 | 38 | 0.526316 | 0.170437 |
