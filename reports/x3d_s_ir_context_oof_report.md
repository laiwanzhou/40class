# X3D-S IR-Context Phase 4 Report

## Decision

**Competition-retained; full matched primary rule not evaluated.** The approved compute amendment replaced the complete matched three-fold experiment with a fixed fold-0, 10-epoch sanity check.

## Three-Seed Stability

|     seed |   accuracy |   macro_f1 |   worst_user_accuracy | worst_user_id   | protocol                         |
|---------:|-----------:|-----------:|----------------------:|:----------------|:---------------------------------|
| 20260715 |   0.565517 |   0.480786 |              0.420690 | user1           | strict                           |
| 20260716 |   0.569397 |   0.489552 |              0.448276 | user1           | strict                           |
| 20260717 |   0.550000 |   0.477880 |              0.400000 | user1           | recovery_deviation_fold1_epoch10 |

Mean Accuracy `0.561638` (sample SD `0.010264`); mean Macro-F1 `0.482739` (sample SD `0.006077`). Seed 20260717 is stability-supporting evidence with the documented fold-1 recovery deviation; seed 20260715 remains canonical.

## Checkpoint Reverification

All `9` formal fold checkpoints regenerated their saved arrays and metrics within `1e-06`. Observed maximum delta: `0.0`.

## Fixed-Budget Matched Sanity

On the exact 800-trial outer fold 0, X3D Accuracy/Macro-F1/worst-user were `0.571250` / `0.486918` / `0.533835`. MobileNet/TCN at the frozen 10-epoch budget achieved `0.283750` / `0.173536` / `0.237569`. X3D deltas were `0.287500` / `0.313382` / `0.296266`.

The baseline is below the preregistered `0.53` anomaly threshold. X3D uniquely corrected `257` trials; the baseline uniquely corrected `27`; oracle-pair Accuracy was `0.605000` and prediction disagreement was `0.638750`. These are fold-0 diagnostics, not three-fold confidence intervals.

## Duration Diagnostics (Canonical Seed)

| bucket   |   sample_count |   accuracy |   macro_f1 |   mean_num_clips |
|:---------|---------------:|-----------:|-----------:|-----------------:|
| <=13     |            555 |   0.540541 |   0.366255 |         1.000000 |
| 14-32    |           1013 |   0.587364 |   0.481378 |         1.000000 |
| 33-64    |            596 |   0.575503 |   0.460482 |         2.000000 |
| >64      |            156 |   0.474359 |   0.304065 |         3.416667 |

Representative submission-path latency remains the Phase 3 measurement: `<=13` 0.8110 s, `14-32` 0.2162 s, `33-64` 0.4824 s, and `>64` 11.3689 s per trial including YOLO/ROI. The trained X3D+head checkpoint is 14,388,607 bytes; with YOLO the provisional IR route is 20,644,200 bytes.

## Claim Boundary

The teammate VideoMAE `0.641` result and historical MobileNet/Skeleton results remain descriptive because their split and metric contracts are not fully matched. No held-out-4 labels or predictions were read. The original 10,000-replicate paired bootstrap was not performed after the user-approved compute amendment, so this report does not claim the original matched three-fold primary criterion passed.
