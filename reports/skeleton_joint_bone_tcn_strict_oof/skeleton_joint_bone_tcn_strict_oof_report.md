# D2-v1 Joint/Bone TCN Strict Cross-User OOF Development Evidence

## Frozen Contract

- Model commit: `90c9b30518f60b3900cbca1691858d424e4f981d`.
- C1 per-frame scale, 64-step gap-aware resampling, and strict nested user folds are unchanged.
- Input order: joint `17 x [xyz, velocity]` followed by H36M edge-order bone `16 x [xyz, velocity]`.
- Bone position is child minus parent after C1 scale normalization; bone velocity is child joint velocity minus parent joint velocity before resampling.
- Input has 198 independently normalized channels; original TemporalClassifier `[64,128]`, 128D embedding, 40 logits.
- Parameter count: 209,640; seed 20260812. All three formal outer folds ran consecutively without an outer-fold gate.

## Fold Results

| model               |   fold |   selected_epoch |   outer_validation_trials |   outer_accuracy |   outer_macro_f1_40class |   outer_weighted_f1 |
|:--------------------|-------:|-----------------:|--------------------------:|-----------------:|-------------------------:|--------------------:|
| C1-TCN              |      0 |               13 |                       812 |         0.360837 |                 0.237641 |            0.316879 |
| D2-v1-JointBone-TCN |      0 |               28 |                       812 |         0.440887 |                 0.349113 |            0.427062 |
| C1-TCN              |      1 |               34 |                       667 |         0.464768 |                 0.374526 |            0.458399 |
| D2-v1-JointBone-TCN |      1 |               11 |                       667 |         0.431784 |                 0.294516 |            0.400172 |
| C1-TCN              |      2 |               23 |                       862 |         0.411833 |                 0.294329 |            0.384482 |
| D2-v1-JointBone-TCN |      2 |                9 |                       862 |         0.368910 |                 0.214771 |            0.317931 |

## Combined OOF

| model               |   samples |   accuracy |   macro_f1_40class |   weighted_f1 |
|:--------------------|----------:|-----------:|-------------------:|--------------:|
| C1-TCN              |      2341 |   0.409227 |           0.300835 |      0.383085 |
| D2-v1-JointBone-TCN |      2341 |   0.411790 |           0.292657 |      0.381801 |

- D2-v1 - C1 accuracy: **+0.002563**.
- D2-v1 - C1 macro-F1: **-0.008178**.
- Paired discordance: C1-only correct=202, D2-only correct=208; exact McNemar p=0.804997.
- Paired bootstrap 95% CI for D2-v1 - C1 accuracy: `[-0.014097, +0.019223]` (10000 replicates, seed 20260812).
- D2 improves accuracy for 7/14 users and 14/40 classes.

## Training Curves

|     fold |   selected_epoch |   inner_max_train_accuracy |   inner_best_validation_accuracy |   formal_max_train_accuracy |   outer_accuracy |
|---------:|-----------------:|---------------------------:|---------------------------------:|----------------------------:|-----------------:|
| 0.000000 |        28.000000 |                   0.823194 |                         0.404612 |                    0.761282 |         0.440887 |
| 1.000000 |        11.000000 |                   0.739379 |                         0.366667 |                    0.546595 |         0.431784 |
| 2.000000 |         9.000000 |                   0.711438 |                         0.425856 |                    0.492224 |         0.368910 |

## Pre-Registered Decision

**D2-v1 does not pass the success criterion.** Combined accuracy is only marginally higher, its paired confidence interval includes zero, and Macro-F1 decreases, violating the protection metric. C1-TCN remains the Skeleton expert. The D2-only outcomes are recorded as complementarity evidence only; no OOF ensemble weight is fitted in this experiment. Under the frozen route, topology experiments pause here rather than advancing to D3.

## Provenance

- Evidence type: strict cross-user OOF development evidence, not a one-time untouched final test.
- Frozen split SHA256: `2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- Raw histories and predictions remain under `outputs/skeleton_joint_bone_tcn_strict_oof`; no checkpoint or model weight was saved.
