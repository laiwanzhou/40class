# D1-v1 Lightweight ST-GCN Strict OOF Report

## Frozen Contract

- Model commit: `ed3ac7535aaf2fd0b42958032005fed4743e8a85`.
- Fixed H36M-17 graph; input `xyz + velocity`; C1 per-frame bone scale.
- Channels `32, 48, 64`; temporal kernel 5; dilations `1, 2, 4`; receptive field 29.
- Segment-aware temporal convolution; 128D embedding; dropout 0.2; seed 20260812.
- Parameter count: 60,920. No adaptive adjacency, attention, extra stream, or augmentation.
- All three formal outer folds were run consecutively after contract freeze. No outer fold was used as a gate.

## Fold Results

| model       |   fold |   selected_epoch |   outer_validation_trials |   outer_accuracy |   outer_macro_f1_40class |   outer_weighted_f1 |
|:------------|-------:|-----------------:|--------------------------:|-----------------:|-------------------------:|--------------------:|
| C1-TCN      |      0 |               13 |                       812 |         0.360837 |                 0.237641 |            0.316879 |
| D1-v1-STGCN |      0 |               32 |                       812 |         0.256158 |                 0.133225 |            0.205715 |
| C1-TCN      |      1 |               34 |                       667 |         0.464768 |                 0.374526 |            0.458399 |
| D1-v1-STGCN |      1 |               13 |                       667 |         0.181409 |                 0.059915 |            0.129929 |
| C1-TCN      |      2 |               23 |                       862 |         0.411833 |                 0.294329 |            0.384482 |
| D1-v1-STGCN |      2 |               32 |                       862 |         0.258701 |                 0.150782 |            0.223350 |

## Combined OOF

| model       |   samples |   accuracy |   macro_f1_40class |   weighted_f1 |
|:------------|----------:|-----------:|-------------------:|--------------:|
| C1-TCN      |      2341 |   0.409227 |           0.300835 |      0.383085 |
| D1-v1-STGCN |      2341 |   0.235797 |           0.125394 |      0.192324 |

- D1-v1 - C1 accuracy: **-0.173430**.
- D1-v1 - C1 macro-F1: **-0.175441**.
- Paired discordance: C1-only correct=514, D1-only correct=108; exact McNemar p=3.02219e-64.
- Paired bootstrap 95% CI for D1-v1 - C1 accuracy: `[-0.193080, -0.154208]` (10000 replicates, seed 20260812).
- D1-v1 improves accuracy for 0/14 users and 2/40 classes.

## Training Diagnosis

|     fold |   selected_epoch |   inner_final_train_accuracy |   inner_max_train_accuracy |   inner_best_validation_accuracy |   formal_final_train_accuracy |   formal_max_train_accuracy |   outer_accuracy |
|---------:|-----------------:|-----------------------------:|---------------------------:|---------------------------------:|------------------------------:|----------------------------:|-----------------:|
| 0.000000 |        32.000000 |                     0.366920 |                   0.369772 |                         0.211740 |                      0.340746 |                    0.354480 |         0.256158 |
| 1.000000 |        13.000000 |                     0.327614 |                   0.327614 |                         0.184444 |                      0.225209 |                    0.238351 |         0.181409 |
| 2.000000 |        32.000000 |                     0.344176 |                   0.347324 |                         0.250951 |                      0.334009 |                    0.350237 |         0.258701 |

Train accuracy remains low in inner selection and formal refit, while D1-v1 loses substantially on every formal outer fold. This is consistent with a strong underfitting/representation bottleneck in D1-v1, not an isolated fold failure. It does not support retaining this graph model over the C1-TCN expert.

## Decision

**D1-v1 fails the topology-retention test.** C1-TCN remains the Skeleton expert baseline. The paired interval is wholly below zero and the loss occurs across all three folds. Do not use D1-v1 for fusion or replace C1-TCN with it. Any D1-v2, joint/bone stream, or joint-identity experiment must be treated as a new pre-frozen experiment and cannot retroactively change this OOF result.

## Provenance

- Frozen split: `metadata/splits/train14_oof_3fold.json`.
- OOF assignment SHA256: `2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- Raw histories and predictions remain under `outputs/skeleton_lightweight_stgcn_strict_oof`; no checkpoint or model weight was saved.
