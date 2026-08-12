# Skeleton C0 vs C1 Strict OOF Report

## Protocol

- Frozen shared assignment: `metadata/splits/train14_oof_3fold.json`.
- Six independently fitted clean views: each `inner_selection` projection uses inner-fit users only; each `formal_outer` projection uses outer-train users only.
- Visual candidate margin is fixed at 20%; corresponding validation users never fit projection or feature normalization.
- C0 and C1 use the same H36M-17 median bone-length estimator, 64-step gap-aware timeline, segment-local velocity, residual TCN, optimizer, seed schedule, and epoch-selection rule.
- C0 uses one median trial scale; C1 uses the per-frame value of the same estimator.
- Formal models are freshly initialized and refit for the epoch selected on inner validation. Outer-validation labels are evaluated exactly once.

## Clean Views

|   fold | scope           |   calibration_frames |   retained_frames |   ambiguous_frames |   confident_multi_person_retained |   empty_trials |
|-------:|:----------------|---------------------:|------------------:|-------------------:|----------------------------------:|---------------:|
|      0 | inner_selection |                27419 |             46337 |                475 |                              1761 |              0 |
|      0 | formal_outer    |                41793 |             67583 |                606 |                              2441 |              0 |
|      1 | inner_selection |                28977 |             45434 |                277 |                              1776 |              0 |
|      1 | formal_outer    |                41938 |             67581 |                608 |                              2439 |              0 |
|      2 | inner_selection |                26510 |             43393 |                462 |                              1343 |              0 |
|      2 | formal_outer    |                38783 |             67583 |                606 |                              2441 |              0 |

All six scopes have disjoint projection-fit and validation users and zero empty trials.

## Fold Results

| representation   |   fold |   selected_epoch |   outer_validation_trials |   outer_accuracy |   outer_macro_f1_40class |   outer_worst_user_accuracy |
|:-----------------|-------:|-----------------:|--------------------------:|-----------------:|-------------------------:|----------------------------:|
| C0               |      0 |               17 |                       812 |         0.389163 |                 0.268434 |                    0.283019 |
| C0               |      1 |               11 |                       667 |         0.379310 |                 0.209620 |                    0.266667 |
| C0               |      2 |               15 |                       862 |         0.390951 |                 0.249514 |                    0.310000 |
| C1               |      0 |               13 |                       812 |         0.360837 |                 0.237641 |                    0.240602 |
| C1               |      1 |               34 |                       667 |         0.464768 |                 0.374526 |                    0.357576 |
| C1               |      2 |               23 |                       862 |         0.411833 |                 0.294329 |                    0.290000 |

## Combined OOF

| representation   |   samples |   accuracy |   macro_f1_40class |   weighted_f1 |
|:-----------------|----------:|-----------:|-------------------:|--------------:|
| C0               |      2341 |   0.387014 |           0.250153 |      0.346919 |
| C1               |      2341 |   0.409227 |           0.300835 |      0.383085 |

- C1 - C0 accuracy: **+0.022213**.
- C1 - C0 macro-F1: **+0.050682**.
- Paired discordance: C0-only correct=116, C1-only correct=168; exact McNemar p=0.00241637.
- Paired bootstrap 95% CI for C1 - C0 accuracy: `[+0.008116, +0.036309]` (10000 replicates, seed 20260812).

## Decision

C1 wins combined OOF accuracy and macro-F1, and the paired accuracy interval excludes zero. However, fold0 favors C0 while folds1/2 favor C1, and selected epochs differ materially. The evidence supports **C1 per-frame bone scale as the current preprocessing winner**, with fold heterogeneity retained as a documented residual risk. No graph/ST-GCN experiment was run.

## Artifacts

- `fold_results.csv`: fold-level metrics and provenance hashes.
- `combined_oof_summary.csv`: primary combined metrics.
- `paired_oof_outcomes.csv`: one row per canonical OOF trial.
- `paired_oof_per_user.csv` and `paired_oof_per_class.csv`: heterogeneity diagnostics.
- Raw training histories and normalization statistics remain under `outputs/skeleton_c0_c1_strict_oof`; no checkpoint or model weight was saved.
