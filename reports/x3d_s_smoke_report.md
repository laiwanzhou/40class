# X3D-S Phase 3 End-to-End Verification

Date: 2026-08-11 (Asia/Shanghai)

## Scope

This is implementation and deployment-path evidence for the pure X3D-S IR expert. It is not scientific validation evidence and does not access competition-test data or held-out labels.

## Real-Manifest Contract

- Audited 84,906 frame rows, 2,910 trials, and all 40 classes.
- Train/validation users and sample IDs are disjoint; frame indices are contiguous and ordered.
- Adaptive windows exactly cover every trial. The 1-frame and 236-frame boundaries emit shapes `[1,1,3,13,182,182]` and `[8,1,3,13,182,182]`.
- Repeated validation access is deterministic and no selected path belongs to competition-test data.

## CUDA and Overfit Smoke

- Three-epoch CUDA smoke used two train and two validation trials. Epoch 3 unfroze the backbone with nonzero learning rates; backbone and head gradients were finite and nonzero.
- Both checkpoints reloaded and both validation archives were unique, finite, complete, and shaped `[2,40]`.
- Peak allocated CUDA memory was 594,302,464 bytes.
- The 16-trial, 8-class implementation overfit run used one- and multi-clip trials with augmentation disabled. Loss fell from 3.685888 to 0.004131 and final train Accuracy was 1.0. This result is not scientific evidence.

## Deployment Audit

- Trained X3D plus embedded custom head checkpoint: 14,388,607 bytes.
- YOLO11n-pose checkpoint: 6,255,593 bytes.
- Provisional IR-route serialized subtotal: 20,644,200 / 95,000,000 bytes.
- The embedded head was not double-counted. Archive completeness, class-map hash, shuffled sample alignment, and `alpha=0` exact X3D probability recovery passed.

## Online/Offline Input Parity

The deterministic training-only sample set covered frame lengths 6, 19, 54, 97, 231, and 236, producing clip counts `[1,1,2,4,8,8]`. It included high/low pose reliability, the historical low-confidence recovery path, all four latency buckets, and both available eight-clip train trials.

Exact gates passed for frame order, clip count, window bounds, sampled source indices, historical recovery routing, shared X3D normalization, and prohibition of silent full-frame fallback.

| Gate | Frozen limit | Observed worst case | Result |
|---|---:|---:|---|
| ROI-box absolute drift | <= 1 px | 0.920441 px | Pass |
| Trial crop MAE | <= 1/255 | 0.575740/255 | Pass |
| Crop P99 absolute error | <= 8/255 | 7/255 | Pass |
| Crop PSNR | >= 40 dB | 43.0948 dB | Pass |
| Worst-frame crop MAE | <= 2/255 | 1.668167/255 | Pass |

The maximum individual pixel difference was 168/255 on the 231-frame trial. It was not a gate: the trial MAE was 0.031608/255, P99 was 1/255, and PSNR was 52.5802 dB, consistent with edge-localized resampling from subpixel detector drift.

## Fixed-Checkpoint Sensitivity

Checkpoint SHA-256: `77ed27ddb9a09edf2c675cdcfbe4e052bacb11a0e6b9f1ab1582108609404039`.

The four shorter trials were model-output identical. The two non-identical long trials produced:

| Frames | Embedding cosine | Probability L1 | JS divergence | Max probability delta | Top-1 |
|---:|---:|---:|---:|---:|---|
| 231 | 0.999978 | 0.001953 | 0.000000676 | 0.000203 | Agree |
| 236 | 0.998655 | 0.014518 | 0.000041122 | 0.001600 | Agree |

These are sensitivity diagnostics only. No post-hoc threshold was selected, and top-1 disagreement alone would not override passing preprocessing parity.

## Measured Submission-Path Latency

Latency is one warm-state representative measurement per bucket except `>64`, which averages the 97-, 231-, and 236-frame trials.

| Frame bucket | Trials | YOLO/ROI mean (s) | X3D per clip mean (s) | Complete trial mean (s) |
|---|---:|---:|---:|---:|
| `<=13` | 1 | 0.6931 | 0.1180 | 0.8110 |
| `14-32` | 1 | 0.1973 | 0.0189 | 0.2162 |
| `33-64` | 1 | 0.4542 | 0.0141 | 0.4824 |
| `>64` | 3 | 11.3056 | 0.0090 | 11.3689 |

## Verification

- Focused Phase 3 suite: 65 passed.
- Full repository suite: 107 passed.
- `python -m compileall -q src scripts tests`: passed.
- Audit CLI help and `git diff --check`: passed.

The first full-suite run identified an Ultralytics import-time global OpenCV I/O replacement. The final implementation lazy-loads YOLO, restores `cv2.imread/imwrite/imshow`, and defensively normalizes singleton-channel grayscale arrays. The targeted regression passed before the final full-suite run.

## Phase Gate

Steps 1-7 pass. Step 8 is ready for the evidence commit; no Phase 4 scientific results are claimed here.
