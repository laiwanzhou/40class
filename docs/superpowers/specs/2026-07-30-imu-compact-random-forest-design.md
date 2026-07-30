# Compact IMU Random Forest Design

## Scope

This experiment compresses the accepted 300-tree balanced random forest while
holding the `imu-rf-summary-v1` 2,310-feature representation, fold-0 split,
train-only imputer, class order, and prediction rule fixed. It runs only on
`IMU_rf`; it does not alter Stage 1, Stage 2, TCN experiments, or accepted RF
artifacts.

## Fresh-only phases

The experiment root must not exist before execution. Phase A losslessly
re-serializes all three accepted balanced forests using the finite compression
table in `configs/imu_rf_compact_fold0.json`. Predictions, probabilities,
metadata, estimator parameters, class order, and every tree field must remain
identical.

Phase B trains the six frozen structural candidates only at seed 20260725.
The table covers tree-count reduction, depth limiting, larger leaves, row
subsampling, an aggressive small candidate, and a performance-first fallback.
No candidate may be added after viewing validation results.

At most two candidates enter Phase C: one passing the declared 16 MiB primary
gate and one distinct candidate passing the declared 32 MiB fallback gate.
Each is confirmed at seeds 20260724, 20260725, and 20260726. If a gate has no
qualifying candidate, its selection is null and the search is not expanded.

## Measurement and publication

Sizes are exact serialized bytes and MiB means 1,048,576 bytes. Every run also
records tree/node/depth statistics, a reproducible in-memory array-byte
estimate, load time, one-sample inference time, and full-validation inference
time. Structural runs retain the accepted ten-file RF run contract. The root
reports include all candidates, the size/F1 Pareto frontier, 8/16/32/50 MiB
budget winners, multi-seed class and user summaries, and paired changes against
the accepted balanced RF.

Large `.joblib` files, matrices, probabilities, and experiment directories stay
outside Git. Git contains only code, frozen configuration, tests, this design,
the execution plan, and small result summaries.
