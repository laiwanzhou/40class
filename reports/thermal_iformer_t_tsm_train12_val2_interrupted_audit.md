# Thermal iFormer-T + TSM T1-B Interrupted Audit

## Status and boundary

This is an interrupted development snapshot, not a completed T1-B result. Human authorization started the frozen 30-epoch train12/user6-user7 run and a later human instruction stopped it after 17 completed epochs because validation performance remained weak. No automatic resume is authorized.

- Branch: `experiment/thermal-iformer-t-t1b`
- Parent corrective T1-A commit: `d97f9c940cc16da470ddb5a05ed70b06897a6e39`
- Scientific baseline: `c42bb43091c79903e5fde5655c2846c87305895a`
- Fitting population: 1,922 Thermal-usable train12 trials
- Selection population: 377 Thermal-usable user6+user7 trials
- Input: full-frame Thermal, 16 uniform normalized-time segments
- Heldout labels, competition test, quarantined evidence: not accessed
- Frozen IR/X3D and ExpertEvidence: not modified

## Best observed checkpoint

The frozen ranking rule selected epoch 16 by combined fixed-label `0..39` Macro-F1, then Accuracy, then worst-user Accuracy, then lower epoch.

| Metric | Train | Validation |
|---|---:|---:|
| Loss | 2.13564 | 9.96920 |
| Accuracy | 0.50676 | 0.29443 |
| Macro-F1 | 0.41815 | 0.19523 |

- Worst-user Accuracy: `0.21026`
- user6: 195 trials, Accuracy `0.21026`, fixed-label Macro-F1 `0.12418`, present classes `34/40`
- user7: 182 trials, Accuracy `0.38462`, fixed-label Macro-F1 `0.22682`, present classes `33/40`
- Zero-recall classes: `1, 2, 8, 11, 13, 14, 16, 18, 19, 20, 21, 22, 24, 25, 26, 35, 37, 38`

Single-user absent classes are support gaps and are not interpreted as zero model capability. This development snapshot does not authorize architecture retention; shared train-14 OOF remains the final gate.

## Epoch trajectory

| Epoch | Train loss | Train Acc | Train Macro-F1 | Val loss | Val Acc | Val Macro-F1 | Worst-user Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.69469 | 0.04370 | 0.03443 | 3.82553 | 0.04775 | 0.02871 | 0.04615 |
| 2 | 3.51382 | 0.12383 | 0.06219 | 3.93295 | 0.11406 | 0.06289 | 0.09231 |
| 3 | 3.26670 | 0.18730 | 0.08745 | 4.24799 | 0.11671 | 0.05304 | 0.09231 |
| 4 | 3.09667 | 0.21280 | 0.10507 | 6.48961 | 0.17772 | 0.10119 | 0.16923 |
| 5 | 2.96760 | 0.25858 | 0.15137 | 8.77336 | 0.15119 | 0.08514 | 0.14286 |
| 6 | 2.84434 | 0.28512 | 0.16103 | 3.07531 | 0.21485 | 0.09764 | 0.21429 |
| 7 | 2.71911 | 0.32934 | 0.21455 | 40.71414 | 0.18302 | 0.10996 | 0.16410 |
| 8 | 2.62239 | 0.34860 | 0.24172 | 23.06989 | 0.21485 | 0.12662 | 0.19487 |
| 9 | 2.52886 | 0.38241 | 0.26530 | 6.98415 | 0.22546 | 0.15181 | 0.21026 |
| 10 | 2.46671 | 0.40375 | 0.29730 | 32.59058 | 0.19894 | 0.10897 | 0.15385 |
| 11 | 2.39560 | 0.42560 | 0.32003 | 10.85640 | 0.24403 | 0.16035 | 0.21026 |
| 12 | 2.33268 | 0.43913 | 0.34238 | 9.47536 | 0.28382 | 0.17742 | 0.20513 |
| 13 | 2.27254 | 0.45890 | 0.37210 | 24.63351 | 0.24403 | 0.14810 | 0.18974 |
| 14 | 2.21676 | 0.49532 | 0.41905 | 18.70378 | 0.25464 | 0.15953 | 0.18462 |
| 15 | 2.18007 | 0.50260 | 0.43034 | 48.80758 | 0.28382 | 0.18913 | 0.21538 |
| 16 | 2.13564 | 0.50676 | 0.41815 | 9.96920 | 0.29443 | 0.19523 | 0.21026 |
| 17 | 2.11477 | 0.52810 | 0.45956 | 39.47821 | 0.27056 | 0.18614 | 0.20513 |

## Artifact inventory

Runtime artifacts remain local because the repository intentionally ignores `outputs/`. Their identities are recorded so an auditor can verify the exact files without committing learned weights or predictions.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `best_macro_f1.pt` | 10,862,703 | `ca9c11c0f4d50f67c89da52284f726085dfeb3d1578621438872b9255a05d827` |
| `best_validation_predictions.npz` | 34,167 | `2281512e004a3817adac975005a420fddea7ebbc61140b8af99ce63ef9907879` |
| `history.json` | 972,987 | `2495272fa678ec6e555b1792f0a900e3440eb97ad5809a84e70f3451d45cd3dd` |
| `cuda_smoke.json` | 230 | `b2f825d485e9c0eb5cfd3d742ee36cc395c55964ef2b4c71908e9c8124377497` |

Local directory: `outputs/thermal_iformer_t_tsm_train12_val2_seed20260715/`.

## Diagnostic evidence and hypotheses

The best validation archive is finite, but its logits are badly scaled: absolute maximum `3824`, standard deviation `85.03`, mean NLL `9.34`, p95 NLL `5.87`, and maximum NLL `1760`. Validation loss also oscillates from `3.08` to `48.81` while validation Accuracy stays between `0.15` and `0.29` after epoch 5.

The leading hypothesis is classifier BatchNorm instability, not yet a confirmed root cause. The official mobile iFormer head is `BatchNorm1d(256) -> Linear`. The TSM wrapper averages 16 frame embeddings into one trial embedding before applying this head, so the classifier BatchNorm trains on only four trial rows per physical batch. Gradient accumulation does not combine BatchNorm statistics. At epoch 16 its recorded running mean absolute maximum is `14.55`, with running variance range `0.101..1.489`.

Ranked follow-up probes, all requiring a new human approval before training:

1. Re-evaluate the saved checkpoint with controlled classifier-BN modes and feature/logit instrumentation; do not fit weights.
2. Compare the current `mean(frame_features) -> BN -> Linear` head against a BN-free trial head initialized from the audited classifier linear layer.
3. Measure train/validation embedding norms and per-user feature shift before considering learning-rate or augmentation changes.
4. Only after those probes, decide whether a short fresh run is justified. Do not resume epoch 18 from this checkpoint as if the 30-epoch recipe remained validated.

## Stop decision

T1-B is paused after 17 epochs. This snapshot is suitable for code and result audit, but it is not evidence that iFormer-T is qualified or rejected. MobileNet control, iFormer-S, OOF evidence, heldout work, and competition inference remain unstarted.

## Verification at snapshot

- T1-A/T1-B focused tests: `36 passed`
- T0 audit contract: `11 passed`
- T0.5 audit contract: `13 passed`
- Full suite: `276 passed, 1 failed`
- `compileall`: passed
- `git diff --check`: passed

The only full-suite failure is the pre-existing independent-worktree dependency `outputs/x3d_s_ir_context_fold0/x3d_s_ir_context_adaptive_smoke/best_accuracy.pt`, which is an ignored frozen X3D smoke checkpoint and is not generated or modified by this Thermal branch.
