# X3D Train12/Val2 Layer-wise LR1 Execution Plan

**Goal:** Test lower block-specific adaptation rates while retaining block4/block5 capacity and every temporal window.

**Design authority:** `docs/superpowers/specs/2026-08-14-x3d-train12-val2-layerwise-lr1-design.md`

## Task 1: Auditable block-specific optimizer groups

- [x] Add failing tests for block-specific parameter LR assignment, scope labels, and config validation.
- [x] Implement optional `optimizer.backbone_block_lrs` without changing existing configs.
- [x] Record scheduled LR by scope in development and finalization histories/summaries.
- [x] Run focused regression tests.

## Task 2: Freeze candidate before results

- [x] Create the layerwise_lr1 config and preregistration record.
- [x] Prove all non-LR behavior equals partial2.
- [x] Record the partial1 local-output deletion while preserving versioned evidence.
- [x] Commit and push before smoke/full results.

## Task 3: Execute and report

- [x] CUDA smoke verifies block4 `3e-6`, block5 `1e-5`, head `3e-4`, finite gradients, and unchanged canonical hashes.
- [x] Run the frozen candidate to the existing checkpoint/early-stop rule.
- [x] Independently compare trial, user, duration, confidence, and train-gap metrics against partial2.
- [x] Apply the frozen decision rule; preserve artifacts on human-review regression.
- [x] Run full verification.
- [x] Commit, push, and update the single handoff.
