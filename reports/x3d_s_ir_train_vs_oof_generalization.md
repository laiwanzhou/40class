# X3D-S IR Train-vs-OOF Generalization Diagnostic

This is a pure-inference diagnostic using only canonical seed `20260715`. It does not change training, checkpoint selection, canonical evidence, retention, or finalization.

The three formal checkpoint SHA-256 values are `2962631e...b36d7a`,
`7340ffad...7897f7`, and `97507219...636803`. Repeating the complete diagnostic
twice produced byte-identical per-fold prediction archives with SHA-256 values
`29cfa997...9721e3`, `91b8e502...e5784`, and `a1ed1a10...063fae`.

| Fold | Epoch | training-log train Acc/F1 | deterministic train_eval Acc/F1 | formal OOF val Acc/F1 | Acc gap | F1 gap | train/val worst-user Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 0.900658 / 0.894710 | 0.952632 / 0.949166 | 0.571250 / 0.486918 | 0.381382 | 0.462247 | 0.865672 / 0.533835 |
| 1 | 29 | 0.980746 / 0.982058 | 0.980144 / 0.981578 | 0.565350 / 0.491366 | 0.414795 | 0.490212 | 0.905473 / 0.420690 |
| 2 | 11 | 0.930727 / 0.927969 | 0.962963 / 0.961544 | 0.560325 / 0.450485 | 0.402638 | 0.511059 | 0.921348 / 0.482587 |

## Combined view

Across `4640` outer-train evaluation rows and `2320` unique formal OOF rows, concatenated deterministic train_eval Accuracy/Macro-F1 are `0.965733` / `0.964861`; formal OOF Accuracy/Macro-F1 are `0.565517` / `0.480786`. The concatenated-population gaps are `0.400216` Accuracy and `0.484075` Macro-F1.

The 4,640 outer-train rows are not 4,640 unique trials: each of the 2,320 canonical train-14 trials is evaluated by the two fold models whose training population included that trial. Formal OOF remains the 2,320-row exactly-once unseen-user population.

`training-log train_acc` was measured during the final training epoch with training-mode stochastic augmentation/dropout. `deterministic train_eval_acc` is a new eval-mode deterministic pass over outer-train. `formal OOF val_acc` is the saved untouched-user prediction. Only the latter two form the reported generalization gap.

## Freeze decision

Diagnostic only. The frozen combined frame manifest was opened as the path-index
source, then immediately projected to the 14 train users. No heldout4 row entered
a dataset, model forward, metric, or output; no heldout evidence/prediction or
competition-test path was accessed. `ir_x3d_s_k400_pure`, the Phase 4
competition-retention decision, Phase 5 registration, and the final checkpoint
remain unchanged. The IR/X3D route is frozen; no new IR single-modality
training, tuning, matched baseline, or ablation may start without explicit
approval.
