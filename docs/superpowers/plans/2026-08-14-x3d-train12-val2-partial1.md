# X3D Train12/Val2 Partial1 Execution Plan

**Goal:** Run one matched capacity-control experiment that trains only X3D block5 after warmup while preserving complete temporal coverage.

**Design authority:** `docs/superpowers/specs/2026-08-14-x3d-train12-val2-partial1-design.md`

## Task 1: Freeze the single-variable candidate

- [x] Add a contract test requiring the partial1/partial2 configs to differ only at `training.unfrozen_backbone_blocks`.
- [x] Confirm RED before the partial1 config exists.
- [x] Create the partial1 YAML and preregistration record.
- [x] Run focused tests and structured config comparison.
- [ ] Commit and push before any full-run result exists.

## Task 2: Verify runtime behavior

- [ ] Run protected CUDA smoke through the train12/val2 runner.
- [ ] Require full temporal coverage, exactly 968,544 trainable backbone parameters after warmup, finite block5/head gradients, and unchanged canonical hashes.

## Task 3: Execute and report

- [ ] Run the frozen 20-epoch candidate with seed 20260715.
- [ ] Preserve checkpoints and stop for human review if the frozen Accuracy regression rule is crossed.
- [ ] Independently recompute fixed-40 metrics, per-user, duration, confidence, and train-validation gap.
- [ ] Compare only against matched partial2 and apply the frozen decision contract.
- [ ] Run full verification, commit, push, and update the single handoff.
