# X3D-S Fold0 Generalization Tuning

> Development-only fold0 evidence. This is not unbiased OOF and does not replace Phase 4/5 evidence.

Decision: `continue_stage_a`

## A2 Parent Comparison

| Metric | A4-T | A2 | Delta |
|---|---:|---:|---:|
| accuracy | 0.615000 | 0.607500 | +0.007500 |
| macro_f1 | 0.525929 | 0.526830 | -0.000901 |
| worst_user_accuracy | 0.563910 | 0.539326 | +0.024584 |

A4-T becomes the preferred fold0 development candidate. It passes the
Macro-F1 and worst-user guards but remains 0.015 below the Accuracy target.
The parent regression guard did not trigger, and A2 remains frozen.

| Metric | Candidate | Target | Canonical fold0 | Delta |
|---|---:|---:|---:|---:|
| accuracy | 0.615000 | 0.630000 | 0.571250 | +0.043750 |
| macro_f1 | 0.525929 | 0.520000 | 0.486918 | +0.039011 |
| worst_user_accuracy | 0.563910 | 0.533800 | 0.533835 | +0.030075 |

## Per User

| User | Accuracy |
|---|---:|
| user18 | 0.567416 |
| user20 | 0.641509 |
| user21 | 0.563910 |
| user3 | 0.604027 |
| user9 | 0.685083 |

## Duration

| Frames | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| <=13 | 225 | 0.635556 | 0.398211 |
| 14-32 | 353 | 0.651558 | 0.487470 |
| 33-64 | 184 | 0.548913 | 0.432260 |
| >64 | 38 | 0.473684 | 0.162117 |

Relative to A2, Accuracy changes by +0.022222 for `<=13`, +0.019830 for
`14-32`, -0.021739 for `33-64`, and -0.052632 for `>64`. Training-only clip
dropout improved aggregate cross-user performance while worsening the longer
duration buckets, consistent with a remaining training/inference temporal
coverage mismatch.
