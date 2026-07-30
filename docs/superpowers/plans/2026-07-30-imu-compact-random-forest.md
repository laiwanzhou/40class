# Compact IMU Random Forest Execution Plan

1. Verify `IMU_rf`, fixed branch SHAs, a clean index, accepted feature and RF
   roots, six ten-file baseline runs, and a nonexistent compact output root.
2. Add RED tests for lossless compression, strict frozen configuration,
   structure metrics, deterministic candidate training, ten-file publication,
   gate selection, Pareto budgets, and external-CWD CLI startup.
3. Implement `src/training/imu_rf_compact.py`, minimally extend the existing RF
   writer to accept joblib compression and extra summary fields, and add the
   fresh-only orchestration CLI.
4. Run Phase A over the three accepted balanced models and all six declared
   serialization methods without retraining.
5. Run all six Phase B candidates at seed 20260725, validate each ten-file run,
   and choose no more than two finalists using only the frozen gates.
6. Run the chosen finalists at all three fixed seeds and validate each run.
7. Build comparison, Pareto, per-class, per-user, and paired-sample summaries;
   capture input hashes before and after and verify exact equality.
8. Run focused RF, all Stage 2, Stage 1, full-repository, compilation, diff,
   fsck, and artifact validation checks.
9. Commit lightweight code/config/tests/docs/results, verify no model or data is
   tracked, and normally push only `IMU_rf` to `origin/IMU_rf`.
