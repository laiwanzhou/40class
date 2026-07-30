# Compact IMU Random Forest Results

## Outcome

The fixed `imu-rf-summary-v1` features and fold-0 split support a substantially
smaller forest without a validation penalty. The recommended candidate is
`trees_150_leaf4`: 150 trees, `min_samples_leaf=4`, otherwise the accepted
balanced-subsample policy. Across three seeds it is 5.90–5.94 MiB, compared
with 103.31–103.51 MiB for the accepted uncompressed 300-tree model.

| model | size MiB mean | accuracy mean ± std | macro F1 mean ± std | weighted F1 mean | zero-F1 mean |
|---|---:|---:|---:|---:|---:|
| accepted Balanced RF | 103.405 | 0.410704 | 0.322327 | 0.378831 | 9.000 |
| `trees_150` fallback | 7.634 | 0.411286 ± 0.009488 | 0.325285 ± 0.007707 | 0.380320 | 9.000 |
| `trees_150_leaf4` primary | 5.919 | 0.418848 ± 0.005138 | 0.335155 ± 0.007871 | 0.395609 | 7.333 |

The primary candidate improves the three-seed mean over the accepted Balanced
RF by +0.008144 accuracy, +0.012828 macro F1, and +0.016778 weighted F1 while
reducing mean file size by about 94.3%. It is the winner for every declared
8/16/32/50 MiB budget. The 2.190 MiB aggressive candidate is also Pareto
nondominated at seed 20260725 (accuracy 0.406632, macro F1 0.318487), but it
does not beat the primary candidate under any declared budget.

For context, the formal TCN reference is accuracy 0.287958 / macro F1 0.223470,
and the CE+SupCon experimental reference is mean accuracy 0.30657 / macro F1
0.24758. These are contextual references, not inputs to candidate selection.

## Lossless serialization screen

All 18 combinations (three accepted models × six predeclared methods) retained
identical predictions, metadata, estimator parameters, class order, and tree
fields. Probability differences were limited to parallel floating-reduction
roundoff (maximum observed absolute difference `1.1102230246251565e-16`).

| method | mean MiB | max MiB | mean size ratio | mean load s |
|---|---:|---:|---:|---:|
| lzma level 3 | 15.404 | 15.433 | 0.14897 | 0.6851 |
| bz2 level 3 | 16.042 | 16.093 | 0.15514 | 0.8673 |
| zlib level 6 | 17.635 | 17.675 | 0.17054 | 0.2655 |
| zlib level 3 | 18.651 | 18.690 | 0.18037 | 0.2564 |
| gzip level 3 | 18.653 | 18.702 | 0.18039 | 0.2517 |
| zlib level 1 | 19.038 | 19.067 | 0.18411 | 0.2566 |

Thus serialization alone meets the 16 MiB goal with lzma level 3. Warm-cache
one-pass measurements for the uncompressed baselines were load 0.130–0.146 s,
one-sample inference 0.033–0.037 s, and 573-sample inference 0.071–0.077 s.

## Frozen structural screen (seed 20260725)

| candidate | MiB | accuracy | macro F1 | weighted F1 | zero F1 | nodes |
|---|---:|---:|---:|---:|---:|---:|
| `trees_150` | 7.594 | 0.424084 | 0.335994 | 0.394744 | 8 | 140,832 |
| `trees_150_depth20` | 7.014 | 0.403141 | 0.312221 | 0.374790 | 10 | 125,196 |
| `trees_150_leaf4` | 5.917 | 0.424084 | 0.344430 | 0.403319 | 8 | 79,674 |
| `trees_150_sample0p7` | 5.997 | 0.394415 | 0.301084 | 0.364661 | 11 | 113,904 |
| `aggressive_80_depth16_leaf4` | 2.190 | 0.406632 | 0.318487 | 0.381380 | 9 | 28,588 |
| `performance_180_depth24` | 8.104 | 0.396161 | 0.299878 | 0.359166 | 10 | 149,594 |

The predeclared gates selected `trees_150_leaf4` as primary and `trees_150` as
the distinct fallback. No validation-driven candidates were added.

## Paired and class behavior

Across three seeds, `trees_150_leaf4` had 100 compact-only-correct and 86
baseline-only-correct sample outcomes; `trees_150` had 50 and 49 respectively.
The full 3,438-row paired table remains outside Git in
`paired_sample_compact_summary.csv` and is keyed by candidate, seed, sample ID,
and user ID.

For `trees_150_leaf4`, zero-F1 classes common to all three seeds were
18, 24, 25, 26, and 35. Relative to each same-seed accepted forest, it recovered
classes `[8,14,19]`, `[2]`, and `[6,19,27]`; class 13 became newly zero-F1 in
the first two seeds. The external per-class and per-user summaries preserve the
full breakdown.

## Reproducibility and safety

Execution command:

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/run_imu_rf_compact_screen.py `
  --config configs/imu_rf_compact_fold0.json `
  --feature-root D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU_stage2_training\fold_0\derived\imu_rf_summary_v1 `
  --baseline-root D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU_stage2_training\fold_0\experiments\random_forest_screen_v1 `
  --output-root D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU_stage2_training\fold_0\experiments\random_forest_compact_v1
```

The feature input canonical SHA-256 remained
`15d1257fd06d520efe1dbabb071f0c721ece1e3d21cc8178aa91fb353c427cdf`;
the accepted RF root remained
`b976f7120ded332615dbdb722960c547c56435d6b4be07a61934c380d529e984`.
The completed external experiment contains 151 files / 418,746,984 bytes and
has canonical SHA-256
`df77f9e7053e76717a5029a846bb442ab91eb6bab8befb6e9589e71c37611a70`.
All 12 structural/confirmation ten-file runs and all 18 lossless roundtrips were
reopened successfully. No `.joblib`, feature matrix, probability array, or
experiment directory is tracked by Git.
