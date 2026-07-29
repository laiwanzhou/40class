# IMU Random Forest Screen Implementation Plan

1. Verify the fixed branch refs, clean linked worktree, absent `IMU_rf`, and
   absent formal derived/output targets; create `IMU_rf` from formal `IMU`.
2. Snapshot the accepted Stage 2, training-index, normalization, and formal TCN
   run inputs using deterministic tree hashes.
3. Write RED tests for deterministic masked summary features, train-only
   imputation, leakage exclusion, strict feature artifacts, frozen RF config,
   deterministic training, probability and reload contracts, ten-file atomic
   publication, CLIs, and comparison summaries.
4. Implement `src/features/imu_rf_features.py`, the feature builder, the RF-only
   trainer, frozen config, CLIs, and experiment summarizer without changing
   Stage 1, Stage 2, TCN, or existing training code.
5. Run focused RF tests, all Stage 2 tests, Stage 1 regression, full repository
   tests, tracked Python compilation, CLI help, `git diff --check`, and strict
   `git fsck`.
6. Run read-only formal-data preflight for all six configurations. Confirm
   2184/573 samples, 40 classes, omitted 0/0, finite non-leaking features, valid
   `[573,40]` probabilities, and no formal run directory creation.
7. Fresh-publish `imu_rf_summary_v1`, train the six fixed runs, validate every
   ten-file set, and publish the four comparison reports without overwrite.
8. Re-snapshot all immutable inputs, audit Git scope and ignored experiment
   products, create local `IMU_rf` commits, and stop without push or integration.

