# T1-v1 Segment-Aware C1-TCN Strict Cross-User OOF Development Evidence

## Frozen Contract

- Model commit: `77b4455a81ff874a1bbd28245f600e32aae2a204`.
- C1 per-frame preprocessing, joint `xyz + velocity` 102D input, T=64, and strict nested user folds are unchanged.
- Original channels `[64,128]`, kernels `5/3`, dilation schedule, residual, BatchNorm, GELU, dropout, optimizer, and seed are unchanged.
- Only temporal edges change: source and destination must both be valid and share the same retained segment ID.
- Parameter count remains 172,776. Single-segment fully valid inference is numerically equivalent after copying C1 weights and BatchNorm state.
- All formal outer folds ran consecutively after freeze; no outer fold was used as a gate.

## Fold Results

| model                  |   fold |   selected_epoch |   outer_validation_trials |   outer_accuracy |   outer_macro_f1_40class |   outer_weighted_f1 |
|:-----------------------|-------:|-----------------:|--------------------------:|-----------------:|-------------------------:|--------------------:|
| C1-TCN                 |      0 |               13 |                       812 |         0.360837 |                 0.237641 |            0.316879 |
| T1-v1-SegmentAware-TCN |      0 |               30 |                       812 |         0.400246 |                 0.306223 |            0.381791 |
| C1-TCN                 |      1 |               34 |                       667 |         0.464768 |                 0.374526 |            0.458399 |
| T1-v1-SegmentAware-TCN |      1 |               34 |                       667 |         0.454273 |                 0.372259 |            0.451833 |
| C1-TCN                 |      2 |               23 |                       862 |         0.411833 |                 0.294329 |            0.384482 |
| T1-v1-SegmentAware-TCN |      2 |               23 |                       862 |         0.410673 |                 0.300932 |            0.386467 |

## Combined OOF

| model                  |   samples |   accuracy |   macro_f1_40class |   weighted_f1 |
|:-----------------------|----------:|-----------:|-------------------:|--------------:|
| C1-TCN                 |      2341 |   0.409227 |           0.300835 |      0.383085 |
| T1-v1-SegmentAware-TCN |      2341 |   0.419479 |           0.324430 |      0.404614 |

- T1-v1 - C1 accuracy: **+0.010252**.
- T1-v1 - C1 macro-F1: **+0.023595**.
- Paired discordance: C1-only correct=78, T1-only correct=102; exact McNemar p=0.0861873.
- Paired bootstrap 95% CI for T1-v1 - C1 accuracy: `[-0.000854, +0.021358]` (10000 replicates, seed 20260812).
- T1 improves accuracy for 5/14 users and 19/40 classes.

## Segment-Stratified Evidence

| segment_group     |   samples |   c1_accuracy |   t1_accuracy |   t1_minus_c1 |   bootstrap_ci_low |   bootstrap_ci_high |   c1_only |   t1_only |
|:------------------|----------:|--------------:|--------------:|--------------:|-------------------:|--------------------:|----------:|----------:|
| multiple_segments |        88 |      0.397727 |      0.363636 |     -0.034091 |          -0.113636 |            0.045455 |         8 |         5 |
| single_segment    |      2253 |      0.409676 |      0.421660 |      0.011984 |           0.000444 |            0.023080 |        70 |        97 |

This stratification is diagnostic, not a separately pre-registered success test. Only 88 trials contain multiple retained segments, and their point estimate favors C1. The aggregate gain comes from single-segment trials, where T1 and C1 have the same temporal-edge semantics and are numerically equivalent at inference after state copy. Their independently trained outcomes can still diverge through floating-point operation order and epoch-selection trajectories. Therefore the aggregate point estimate cannot be attributed to blocking cross-segment edges.

## Pre-Registered Decision

**T1-v1 does not pass the replacement criterion.** Accuracy and Macro-F1 point estimates improve, but the paired accuracy interval includes zero, McNemar is not significant at 0.05, only fold0 improves, and only 5/14 users improve. More importantly, the multiple-segment subgroup does not improve. The evidence does not support cross-gap temporal mixing as a major C1 bottleneck. C1-TCN remains the Skeleton expert; T1 is retained as development evidence only. Under the frozen route, the next temporal question may evaluate resolution, but it must be registered as a new experiment rather than tuned from these outer-fold outcomes.

## Provenance

- Evidence type: strict cross-user OOF development evidence, not a one-time untouched final test.
- Frozen split SHA256: `2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- Raw histories and predictions remain under `outputs/skeleton_segment_aware_tcn_strict_oof`; no checkpoint or model weight was saved.
