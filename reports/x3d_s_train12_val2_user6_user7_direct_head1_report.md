# X3D-S User6/User7 Direct-Head1 Matched Result

> Development-only composite-head replacement. This is not unbiased OOF and does not replace canonical Phase 4/5 evidence.

Decision: `non_winning_ablation`

| Metric | Direct-Head1 | Partial2 | Delta |
|---|---:|---:|---:|
| accuracy | 0.516883 | 0.532468 | -0.015584 |
| macro_f1 | 0.423792 | 0.421575 | +0.002217 |
| worst_user_accuracy | 0.472637 | 0.532338 | -0.059701 |

The intervention replaces the learned 2048-to-256 projector plus classifier with dropout and a direct 2048-to-40 classifier. All other frozen recipe fields remain matched.

Accuracy train-to-validation gap: `0.273298`. Macro-F1 gap: `0.364146`.

Direct classifier relative L2 drift from its seeded initialization: `1.915404`.

No follow-up IR experiment is authorized automatically by this report.
