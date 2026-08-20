# X3D Single13 Fixed Trial Context Experiment

## Status

- Approved as formal training 1 of 2 on 2026-08-20.
- Development-only experiment on the frozen `train12 / user6-user7 val2 / heldout4` partition.
- Parent result: `x3d_s_ir_context_train12_val2_user6_user7_single13_global_seed20260715`.
- Canonical Phase 4/5 IR evidence remains immutable.

## Question

Does replacing the existing per-frame moving person-context ROI with one fixed
person-context box per trial improve cross-subject IR classification while
retaining the efficient one-clip, 13-frame temporal route?

## Sole Intervention

Keep the complete Single13 recipe fixed and change only spatial crop generation:

```text
reference: precomputed moving person-context ROI for each frame
candidate: one fixed trial-level person-context ROI reused by every frame
```

The fixed box is generated from eight endpoint-uniform temporal probes. Probe
person boxes come from the frozen pose-track cache. Valid probe boxes are united,
expanded to a square side of `1.4 * max(union_width, union_height)`, constrained
to at least `0.35 * max(frame_width, frame_height)`, and clipped to image bounds.
No valid probe is a hard error; full-frame fallback is forbidden. The resulting
floating-point box is used for every sampled frame and resized to `256x256` with
PIL Lanczos before the unchanged X3D spatial transform.

## Frozen Contract

- Seed: `20260715` only.
- Train users: user1, user2, user3, user5, user8, user9, user16, user18,
  user19, user20, user21, user22.
- Validation users: user6, user7.
- Heldout users user4, user17, user23, user24 remain inaccessible.
- Train/validation trials: 1935/385 usable IR trials; both contain all 40 classes.
- Temporal input: one global `[0,T)` window, 13 equal temporal bins, one frame
  per bin, random within-bin sampling for training and deterministic midpoints
  for validation.
- Model, partial two-block unfreezing, optimizer, augmentation, loss, BatchNorm,
  scheduler, early stopping, and checkpoint selection are identical to Single13.
- No Depth input, additional seed, motion peak sampling, extra view, TTA, or
  canonical evidence mutation is allowed.

## Pre-Training Audit

The full train/validation manifests were indexed before training. Every one of
the 2320 trials produced a valid fixed box without fallback.

| Split | Trials | Classes | Mean area fraction | P50 | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1935 | 40 | 0.474576 | 0.481520 | 0.785506 | 0.103646 | 1.000000 |
| val | 385 | 40 | 0.511483 | 0.451552 | 0.777559 | 0.163333 | 1.000000 |

An area fraction of 1.0 is admissible only when the valid probe-box union and
frozen margin reach the image bounds; it is not a fallback path.

## Decision Rule

Matched Single13 reference:

- Accuracy: `0.5220779221`
- Macro-F1: `0.4114452159`
- Worst-user Accuracy: `0.5024875622`

The fixed-context candidate is preferred when Accuracy strictly improves,
Macro-F1 is at least `0.4014452159`, and worst-user Accuracy is at least
`0.4824875622`. If Accuracy is below `0.5020779221`, preserve all artifacts,
stop, and request human review. No result is allowed to overwrite the parent or
canonical evidence.

## Execution

1. Run dataset/config/runner contract tests.
2. Run a protected CUDA smoke test with one train and one validation batch.
3. Freeze and push the pre-result implementation and preregistration.
4. Launch one formal background run under a new run ID.
5. Report metrics and resource use; do not automatically launch the approved
   four-channel IR+Depth experiment.
