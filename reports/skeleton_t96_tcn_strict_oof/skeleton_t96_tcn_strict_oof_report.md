# T2-v1 T=96 C1-TCN Strict Cross-User OOF Development Evidence

## Frozen Contract

- Model commit: `4e731b438fe3b50e0f799771c0324ce0b3694d25`.
- C1 per-frame preprocessing, joint `xyz + velocity` 102D input, segment-local velocity, and strict nested user folds are unchanged.
- Gap-aware sequence length is the only experimental variable: T=64 becomes T=96.
- The original `TemporalClassifier`, channels `[64,128]`, kernels `5/3`, dilation schedule, residual, BatchNorm, GELU, dropout, optimizer, scheduler, seed, and pooling remain unchanged.
- Parameter count remains 172,776. All formal outer folds ran consecutively after freeze; no outer fold was used as a gate.

## Fold Results

| model      |   fold |   selected_epoch |   outer_validation_trials |   outer_accuracy |   outer_macro_f1_40class |   outer_weighted_f1 |
|:-----------|-------:|-----------------:|--------------------------:|-----------------:|-------------------------:|--------------------:|
| C1-TCN-T64 |      0 |               13 |                       812 |         0.360837 |                 0.237641 |            0.316879 |
| T2-v1-T96  |      0 |               26 |                       812 |         0.408867 |                 0.307795 |            0.387582 |
| C1-TCN-T64 |      1 |               34 |                       667 |         0.464768 |                 0.374526 |            0.458399 |
| T2-v1-T96  |      1 |               24 |                       667 |         0.440780 |                 0.340161 |            0.427095 |
| C1-TCN-T64 |      2 |               23 |                       862 |         0.411833 |                 0.294329 |            0.384482 |
| T2-v1-T96  |      2 |               22 |                       862 |         0.421114 |                 0.311894 |            0.394128 |

## Combined OOF

| model      |   samples |   accuracy |   macro_f1_40class |   weighted_f1 |
|:-----------|----------:|-----------:|-------------------:|--------------:|
| C1-TCN-T64 |      2341 |   0.409227 |           0.300835 |      0.383085 |
| T2-v1-T96  |      2341 |   0.422469 |           0.321171 |      0.401780 |

- T2-v1 - C1 accuracy: **+0.013242**.
- T2-v1 - C1 macro-F1: **+0.020336**.
- Paired discordance: C1-only correct=90, T2-only correct=121; exact McNemar p=0.0386433.
- Paired bootstrap 95% CI for T2-v1 - C1 accuracy: `[+0.000854, +0.025203]` (10000 replicates, seed 20260812).
- T2 improves accuracy for 5/14 users and 19/40 classes.
- Per-fold T2 - C1 accuracy deltas: fold0=+0.048030, fold1=-0.023988, fold2=+0.009281.
- Per-fold net additional correct trials: fold0=+39, fold1=-16, fold2=+8.

## Retained-Length Diagnosis

| length_bucket   |   samples |   c1_accuracy |   t2_accuracy |   t2_minus_c1 |   bootstrap_ci_low |   bootstrap_ci_high |   c1_only |   t2_only |
|:----------------|----------:|--------------:|--------------:|--------------:|-------------------:|--------------------:|----------:|----------:|
| <=15            |       702 |      0.480057 |      0.511396 |      0.031339 |           0.011396 |            0.051282 |        16 |        38 |
| 16-31           |       835 |      0.372455 |      0.382036 |      0.009581 |          -0.010778 |            0.029940 |        33 |        41 |
| 32-63           |       640 |      0.407813 |      0.403125 |     -0.004688 |          -0.029688 |            0.020313 |        36 |        33 |
| >=64            |       164 |      0.298780 |      0.323171 |      0.024390 |          -0.018293 |            0.067073 |         5 |         9 |

This retained original sequence-length stratification is post-hoc diagnosis only, not a separate success criterion. Length is the number of retained, frame-training-usable rows in each fold-specific formal outer clean view before resampling. A credible resolution effect would be expected to concentrate in longer trials; random positive and negative bucket fluctuations do not establish that denser interpolation adds information.

## Pre-Registered Decision

**T2-v1 does not pass the replacement criterion and does not replace C1-TCN-T64. C1-TCN remains the frozen Skeleton expert, and the planned Skeleton single-modality architecture search ends here.** The headline metrics are positive: Accuracy is +1.32pp, Macro-F1 is +2.03pp, the paired bootstrap interval narrowly excludes zero, and exact McNemar p=0.0386. However, the required stability condition fails. Fold0 contributes +39 net correct trials while the combined net gain is only +31; fold1 loses 16, fold2 gains 8, and only 5/14 users improve. The retained-length pattern also does not support the proposed mechanism: the clearest gain is in the shortest bucket, while `32-63` is slightly negative and `>=64` is inconclusive. This is positive development evidence, but it is too fold- and user-concentrated to replace C1 under the pre-registered rule.

## Provenance

- Evidence type: strict cross-user OOF development evidence, not a one-time untouched final test.
- Frozen split SHA256: `2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- T2 fold summaries report `sequence_length=96`, `model_class=TemporalClassifier`, and validation-excluding preprocessing provenance.
- Raw histories and predictions remain under `outputs/skeleton_t96_tcn_strict_oof`; no checkpoint or model weight was saved.
