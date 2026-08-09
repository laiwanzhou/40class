# Stage 7 variable-length visual pipeline implementation

## Result

Stage 7 implementation is complete pending independent review. The active visual path now consumes each complete original train/validation trial, pads only within a runtime batch, and passes an explicit temporal mask through a padding-safe multi-scale TCN. No training or Stage 8 Depth-representation pilot was run.

## Implemented contract

- `IRPrimaryFullSequenceDataset` loads all frames of one trial without 24/96-frame sampling.
- `FrameBudgetBatchSampler` uses length buckets and bounds `max_sequence_length * batch_size`.
- Depth keeps one fixed three-channel encoder interface: `[raw, relative, pixel_valid]`; inactive representation channels are zeroed for strict pilot comparability.
- Relative Depth statistics use valid context pixels over the complete trial, one median/scale for both views, with the preregistered fallback and minimum IQR scale.
- The spatial path encodes only valid views, applies explicit Depth pixel masks, and injects deterministic six-view reliability before the temporal encoder.
- Temporal masks are reapplied inside every TCN residual block; normalization is channel-only per time step so padded positions cannot alter valid-frame statistics.
- The model exposes a 40-class main output, a visual-only small-action gate, fixed-schema quality tensors, a clip embedding, and availability metadata.
- `sample_id` stays in `ExpertBatchResult`, outside model `forward()`. Expert alignment checks class-map hashes, duplicates, missing IDs, and performs an explicit ID join.
- The reserved calibrated probability mixture exactly reproduces visual probabilities at `alpha=0`. No sensor expert is connected.
- The trainer is one end-to-end variable-length stage. The obsolete Stage A/B cache and fixed-frame path is absent.

## Real-manifest read-only acceptance

Input: `D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256\combined_frame_manifest.csv`

| Check | Result |
|---|---:|
| Manifest frames | 84,906 |
| Train trials | 2,320 |
| Validation trials | 590 |
| Train users | 14 |
| Held-out validation users | 4 |
| User overlap | 0 |
| Classes | 40 |
| Minimum / maximum complete sequence | 1 / 236 frames |
| Class-map hash | `5dbf6af1a1df88314484ba45f9fccc02ab47a1af2adba951a71e42886ec4c5e5` |
| Sampler omissions / duplicates | 0 / 0 |
| Batches exceeding the 256 padded-frame budget | 0 |

The shortest and longest real train trials were loaded from disk and collated together. Their original frame indices and temporal masks were preserved, and only the shorter sample's in-memory tail was padded.

## Verification

- Full repository test suite after independent-review follow-up: `42 passed`.
- Focused Stage 3-7 pipeline suite after review follow-up: `27 passed`.
- Activation-checkpoint backward verification: both spatial encoders receive gradients, while BatchNorm running statistics update exactly once per forward/backward step.
- `py_compile`: passed for the dataset, expert contract, TCN, visual model, and trainer.
- `git diff --check`: passed.
- Competition test read: **false**.
- Skeleton/IMU/Radar read or connected: **false**.
- Training run: **false**.
- Stage 8 started: **false**.

## Independent review

The independent review and its follow-up both passed with no P0/P1 findings. The reviewer separately verified the real manifest, all three Depth input modes, padding behavior, expert alignment, and checkpoint-on/off equality for all 43 BatchNorm states and both spatial-encoder gradients. Stage 7 is complete; Stage 8 has not started.
