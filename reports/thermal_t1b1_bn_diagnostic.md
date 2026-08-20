# Thermal T1-B.1 zero-training BN diagnostic

- Status: **root_cause_not_confirmed_training_still_stopped**
- Epoch-16 checkpoint: `ca9c11c0f4d50f67c89da52284f726085dfeb3d1578621438872b9255a05d827`
- Boundary: train12 plus user6/user7 only; no sealed, competition-test, or quarantined evidence access.
- Execution: inference-only feature collection and offline statistic replay; no optimizer, backward, checkpoint mutation, or training epoch.

## Controlled mode results

| Mode | Logit std | |logit| max | Mean NLL | Accuracy | Macro-F1 | Worst-user acc |
|---|---:|---:|---:|---:|---:|---:|
| checkpoint_eval | 85.0581 | 3824.0080 | 9.3435 | 0.2944 | 0.1958 | 0.2103 |
| checkpoint_analytic_replay | 85.0581 | 3824.0080 | 9.3435 | 0.2944 | 0.1958 | 0.2103 |
| checkpoint_per_frame_consensus | 85.0581 | 3824.0080 | 9.3435 | 0.2944 | 0.1958 | 0.2103 |
| pretrained_frozen_running_stats | 40.2620 | 1838.4550 | 5.1532 | 0.2679 | 0.1733 | 0.1897 |
| train12_trial_population_stats | 14.2559 | 623.3612 | 4.4980 | 0.2493 | 0.1299 | 0.2051 |
| train12_frame_population_stats | 6.6953 | 262.8721 | 3.7840 | 0.2122 | 0.1086 | 0.1641 |
| train12_robust_trial_population_stats | 77.3738 | 3444.0861 | 6.8849 | 0.2891 | 0.1854 | 0.2205 |
| identity_bn_control | 66.9080 | 3005.9919 | 10.9899 | 0.0610 | 0.0512 | 0.0604 |

`checkpoint_analytic_replay` is a numerical control. `checkpoint_per_frame_consensus` is also expected to match checkpoint eval because eval-mode BN plus Linear is affine. The population-stat and identity modes are diagnostic interventions, not deployable candidates.

## Root-cause gate

- Verdict: **not_confirmed**
- BN-free short fresh run authorized: **false**
- `analytic_control`: True (analytic_max_error=1.364e-12, consensus_max_error=2.501e-12)
- `checkpoint_logits_pathological`: True (abs_max=3824.008, mean_nll=9.344)
- `checkpoint_stats_mismatch_train_population`: False (robust normalized_mean_error_rms=0.182, median_variance_ratio=0.834)
- `population_stats_reduce_scale_and_nll`: False (relative_nll_improvement=0.263, relative_logit_std_reduction=0.090)
- `batch4_order_sensitivity`: True (robust replay vs_batch64 mean_dispersion_ratio=6.113, variance_dispersion_ratio=6.393)

## User distribution

- user6/user7 centroid L2 distance: `162.992796`
- centroid cosine similarity: `0.437006`
- standardized mean-difference RMS: `0.140994`
- Per-user accuracy and fixed-label Macro-F1 are reported for every controlled mode; missing single-user classes are not interpreted as zero capability.

## Embedding spikes

- Robust flags (not exclusions): train12 `77`, validation `23`.
- Largest validation trial: `train__c36__user7__1-2-2`, bfloat16 norm `22947.436`, FP32 norm `32075.539`.
- Validation flagged class counts: `{'4': 2, '6': 2, '7': 1, '8': 1, '11': 2, '14': 1, '24': 1, '28': 1, '36': 12}`.
- Robust-inlier user6/user7 centroid cosine: `0.998129`; standardized mean-difference RMS: `0.381724`.
- The spike persists in FP32, so CUDA bfloat16 autocast is not its cause. A layerwise upstream activation trace is required before any head redesign is authorized.

## Limitations

- Final epoch-16 embeddings differ from embeddings seen throughout optimization, so EMA replay measures sensitivity and does not reconstruct exact historical BN statistics.
- Train population moments use deterministic center crops to remove augmentation randomness; they are diagnostic statistics and are not written into a deployable checkpoint.
- Controlled replacement modes reuse the learned BN affine parameters and Linear weights, so they isolate inference normalization but cannot prove how a fresh head would optimize.
- User6 and user7 have different class support; per-user Macro-F1 uses the fixed 0..39 label set and absent classes are not interpreted as model incapability.
