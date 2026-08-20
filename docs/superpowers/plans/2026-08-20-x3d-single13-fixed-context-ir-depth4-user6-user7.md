# X3D Single13 Fixed-Context IR+Depth4 Experiment

## Status

- Approved as formal training 2 of 2 on 2026-08-20.
- Development-only, single-seed experiment on the frozen
  `train12 / user6-user7 val2 / heldout4` partition.
- No three-fold or multi-seed stability experiment is authorized here.
- A user-authorized operational amendment replaces `num_workers=0` with a
  memory-gated `num_workers=4` loader after the initial run completed one slow
  epoch. This changes input throughput only, not sample ownership, sampling,
  augmentation, model, optimization, metrics, or the scientific intervention.
- Before the replacement formal run, execute a CUDA smoke test with four
  non-persistent spawn workers, prefetch factor two, and one Torch CPU thread
  per worker. Non-persistent workers are required because epoch-derived dataset
  augmentation must observe each main-process `set_epoch` update. Require at
  least 4 GiB system memory to remain available throughout.
  If and only if that smoke fails the memory gate or raises a worker-memory
  error, create an equivalent two-worker config and repeat the smoke. A passing
  smoke authorizes the replacement formal run with that worker count.

The final four-worker non-persistent CUDA smoke passed: root plus descendant
peak working set was 5.683 GiB, minimum remaining system memory was 13.342 GiB,
peak GPU memory was 1258 MiB, and the process exited successfully. The earlier
persistent-worker capacity probe was superseded because it could not preserve
epoch-derived augmentation. The two-worker fallback is not required.
- Parent result: `x3d_s_ir_context_train12_val2_user6_user7_single13_fixed_context_seed20260715`.

## Question

Does aligned four-channel early fusion improve the retained Single13 fixed-context
X3D route when the only new evidence is `Depth_Color RGB` alongside `IR gray`?

## Input Contract

```text
trial-level fixed person-context box
             |
             +-- Depth_Color -> R,G,B --+
             +-- IR          -> gray ----+--> [Depth R,G,B,IR] x 13 -> X3D-S
```

- One fixed box is generated from eight endpoint-uniform cached pose probes and
  reused for every Depth and IR frame in the trial.
- Depth and IR use the same floating crop coordinates, Lanczos resize to
  `256x256`, random resized crop, and horizontal flip.
- Brightness, contrast, and gamma augmentation apply only to IR. Depth_Color is
  not photometrically altered.
- Channel order is `[depth_r, depth_g, depth_b, ir_gray]`.
- Mean/std are X3D `(0.45, 0.225)` per channel; the fourth-channel values equal
  the mean of the three original channel constants.
- The K400 X3D stem changes from `(24,3,1,3,3)` to `(24,4,1,3,3)`. Original RGB
  kernels are preserved and the IR kernel is initialized from their channel mean.
- The expanded stem stays inside the frozen early backbone under the matched
  partial2 policy. Only X3D blocks 4/5 unfreeze after the two-epoch warmup.

## Frozen Training Contract

- Seed: `20260715` only.
- Train users: user1, user2, user3, user5, user8, user9, user16, user18,
  user19, user20, user21, user22.
- Validation users: user6, user7.
- Heldout users user4, user17, user23, user24 remain inaccessible.
- Population: 1935 train and 385 validation trials; both contain all 40 classes
  and have usable IR and Depth.
- Temporal input: exactly one globally stratified 13-frame clip per trial.
- Spatial input: the retained fixed trial person-context box.
- Architecture outside the four-channel stem, projected head, optimizer, loss,
  LR schedule, BatchNorm policy, two-block partial unfreezing, augmentation
  ranges, early stopping, and checkpoint selection match the parent experiment.
- Checkpoint objective remains Accuracy, then Macro-F1, then earlier epoch.

## Decision Contract

Matched IR-only fixed-context reference:

- Accuracy: `0.5376623377`
- Macro-F1: `0.4275141329`
- Worst-user Accuracy: `0.5174129353`

Rules:

1. Accuracy below `0.5176623377` is a human-review regression. Preserve all
   artifacts and stop.
2. Accuracy above the matched reference with Macro-F1 delta at least `-0.01`
   and worst-user delta at least `-0.02` is an improved fusion ablation.
3. Only Accuracy at least `0.63` is eligible for a separately approved
   stability review.
4. Reaching `0.63` does not automatically authorize another seed, three-fold
   run, heldout access, or new tuning. Report and wait for explicit approval.

## Execution

1. Verify dataset channel order, synchronized geometry, IR-only photometric
   augmentation, four-channel stem initialization, config, and runner contracts.
2. Audit all 1935/385 trial ownership and modality usability.
3. Run one protected CUDA smoke test.
4. Freeze and push preregistration before formal metrics are revealed.
5. Launch exactly one formal background run on user6/user7.
6. Produce a result report and stop. Do not launch stability work.
