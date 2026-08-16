# E1-v1 Three-Seed C1-TCN Ensemble Strict Cross-User OOF Development Evidence

## Frozen Contract

- Frozen implementation commit: `8cf30b4f7c7af6be5dbe14f6f7206091d8592992`.
- Three pre-registered member seeds: `20260812`, `20260912`, `20261012`; fold index is added to each training seed.
- Every member independently performs legal inner-user epoch selection and a fresh formal outer refit.
- C1 per-frame scale, joint `xyz + velocity` 102D, T=64, clean views, architecture, optimizer, scheduler, and all training settings are unchanged.
- Ensemble rule is the fixed equal mean of member softmax probabilities. No member selection, outer-fold weighting, or gate is fitted.
- Each member has 172,776 parameters; total inference parameters are 518,328.

## Member Results

| model            |   samples |   accuracy |   macro_f1_40class |   weighted_f1 |
|:-----------------|----------:|-----------:|-------------------:|--------------:|
| C1-seed-20260812 |      2341 |   0.409227 |           0.300835 |      0.383085 |
| C1-seed-20260912 |      2341 |   0.407091 |           0.306685 |      0.387563 |
| C1-seed-20261012 |      2341 |   0.399402 |           0.280869 |      0.369012 |

## Fold Results

|     fold |    samples |   c1_accuracy |   e1_accuracy |   e1_minus_c1 |   bootstrap_ci_low |   bootstrap_ci_high |   c1_only |   e1_only |
|---------:|-----------:|--------------:|--------------:|--------------:|-------------------:|--------------------:|----------:|----------:|
| 0.000000 | 812.000000 |      0.360837 |      0.390394 |      0.029557 |           0.012315 |            0.048030 | 15.000000 | 39.000000 |
| 1.000000 | 667.000000 |      0.464768 |      0.451274 |     -0.013493 |          -0.037481 |            0.010495 | 37.000000 | 28.000000 |
| 2.000000 | 862.000000 |      0.411833 |      0.433875 |      0.022042 |           0.004640 |            0.039443 | 22.000000 | 41.000000 |

## Combined OOF

| model                |   samples |   accuracy |   macro_f1_40class |   weighted_f1 |
|:---------------------|----------:|-----------:|-------------------:|--------------:|
| C1-TCN-single-seed   |      2341 |   0.409227 |           0.300835 |      0.383085 |
| E1-C1-3seed-ensemble |      2341 |   0.423751 |           0.315606 |      0.397567 |

- E1 - C1 accuracy: **+0.014524**.
- E1 - C1 Macro-F1: **+0.014771**.
- Paired discordance: C1-only correct=74, E1-only correct=108; exact McNemar p=0.0142157.
- Paired bootstrap 95% CI for E1 - C1 accuracy: `[+0.003417, +0.025630]` (10000 replicates, seed 20260812).
- E1 improves accuracy for 2/3 folds, 9/14 users, and 17/40 classes.

## Member Diversity

| group               |   samples |   sample_fraction |   c1_accuracy |   e1_accuracy |   e1_minus_c1 |
|:--------------------|----------:|------------------:|--------------:|--------------:|--------------:|
| unanimous           |      1176 |          0.502349 |      0.601190 |      0.601190 |      0.000000 |
| member_disagreement |      1165 |          0.497651 |      0.215451 |      0.244635 |      0.029185 |

This is a diagnostic decomposition. It shows where fixed probability averaging changes outcomes; it was not used to select members or fit weights.

## Pre-Registered Decision

**E1 passes the replacement criterion and becomes the preferred Skeleton expert.** The decision requires positive combined Accuracy, non-decreasing Macro-F1, a paired bootstrap interval above zero, and improvement on at least two folds. Per-user and per-class results remain stability diagnostics rather than tuning inputs.

## Provenance

- Evidence type: strict cross-user OOF development evidence, not a one-time untouched final test.
- Frozen split SHA256: `2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- All nine member summaries state that outer labels were not used for selection and preprocessing excluded scope-validation users.
- Raw histories and member predictions remain under `outputs/skeleton_c1_seed_ensemble_strict_oof`; no checkpoint or model weight was saved.
