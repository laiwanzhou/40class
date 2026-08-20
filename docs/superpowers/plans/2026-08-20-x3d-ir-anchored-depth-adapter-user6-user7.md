# X3D IR-Anchored Depth Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether aligned Depth adds cross-user information when the proven IR input path is preserved exactly instead of being replaced by a four-channel X3D stem.

**Architecture:** Read the same `[Depth R, Depth G, Depth B, IR gray]` Single13 fixed-context tensor as the failed early-fusion run. Before the untouched three-channel K400 X3D-S backbone, form `[IR, IR, IR] + Conv3d_1x1x1(Depth RGB)`, with a bias-free zero-initialized Depth projection. The IR anchor is parameter-free and immutable; only the nine Depth residual weights, the existing projected head, and the existing last two X3D blocks are trainable under the matched schedule.

**Tech Stack:** PyTorch 2.7, PyTorchVideo X3D-S K400 weights, BF16 CUDA AMP, pytest, YAML/JSON experiment contracts.

**Spec:** `docs/superpowers/plans/2026-08-20-x3d-single13-fixed-context-ir-depth4-user6-user7.md` plus the user-approved IR-anchored adapter intervention in this task.

## Global Constraints

- Development population remains 12 train users and validation users `user6,user7`; heldout users and competition test remain unavailable to model selection.
- Use seed `20260715` only. Do not run folds or stability seeds automatically.
- Keep one globally stratified 13-frame clip and one fixed trial person-context box.
- Keep aligned Depth RGB plus IR gray reading, synchronized geometry, IR-only photometric augmentation, workers=4 non-persistent spawn loading, batch limits, loss, optimizer schedule, BN policy, last-two-block unfreeze, and Accuracy checkpoint selection unchanged.
- The new adapter must be exactly IR-equivalent at initialization: changing Depth while holding IR fixed cannot change its output before training.
- Adapter definition is `repeat(IR,3) + zero_initialized_bias_free_Conv3d(DepthRGB, 3->3, kernel=1)`; parameter count is exactly 9.
- Adapter learning rate is `3e-4`, matching the projected head; AdamW weight decay remains `0.05` on the zero-initialized residual weights.
- Primary comparison is the fixed-context IR reference (`0.537662 Accuracy / 0.427514 Macro-F1 / 0.517413 worst-user Accuracy`). The failed expanded-stem IR+Depth4 result (`0.457143 / 0.355353 / 0.427861`) is descriptive context only.
- Accuracy `>=0.63` only makes the candidate eligible for a separately approved stability review. It does not authorize additional training.
- Accuracy below `0.517662` triggers human review; preserve all artifacts and stop.

---

### Task 1: IR-Anchored Adapter Model Contract

**Files:**
- Modify: `src/models/x3d_s_visual_expert.py`
- Test: `tests/test_x3d_s_visual_expert.py`

**Interfaces:**
- Produces: `IRAnchoredDepthAdapter.forward(x: Tensor[B,4,T,H,W]) -> Tensor[B,3,T,H,W]`
- Extends: `X3DSVisualExpert(..., input_adapter_mode="ir_anchored_depth_residual")`

- [x] Write failing tests proving exact `[IR,IR,IR]` initialization, zero initial Depth sensitivity, nine trainable adapter parameters, four-channel model input, and three-channel backbone input.
- [x] Run the focused tests and confirm failure because the adapter API does not exist.
- [x] Implement the minimal adapter and apply it immediately before the backbone.
- [x] Run the focused tests and confirm pass.

### Task 2: Optimizer, Gradient, and Config Contract

**Files:**
- Modify: `src/train_x3d_s_visual_expert.py`
- Create: `configs/experiments/x3d_s_ir_anchored_depth_adapter_train12_val2_user6_user7_single13_fixed_context_workers4.yaml`
- Test: `tests/test_x3d_s_trainer_contract.py`
- Test: `tests/test_x3d_s_train12_val2_dev.py`

**Interfaces:**
- Consumes: `input_adapter_mode="ir_anchored_depth_residual"`
- Produces: optimizer scope `input_adapter` at `3e-4` and gradient scope `input_adapter`

- [x] Write failing tests for strategy validation, standard K400 three-channel backbone construction, the separate adapter LR group, finite adapter gradients, and loader/config equality with the failed four-channel reference apart from the registered model intervention.
- [x] Run focused tests and confirm the expected failures.
- [x] Implement trainer model construction, validation, optimizer grouping, LR reporting, and gradient-scope reporting.
- [x] Run focused tests and confirm pass.

### Task 3: Preregistration and Identity-Safe Reporting

**Files:**
- Create: `reports/x3d_s_train12_val2_user6_user7_single13_fixed_context_ir_anchored_depth_adapter_preregistration.json`
- Create: `scripts/report_x3d_s_ir_anchored_depth_adapter.py`
- Create: `tests/test_report_x3d_s_ir_anchored_depth_adapter.py`
- Modify: `tests/test_x3d_s_train12_val2_dev.py`

**Interfaces:**
- Produces: strict report that verifies run ID, resolved config SHA, input channels, fusion strategy, adapter initialization, split hash, and canonical-artifact non-mutation before applying the decision contract.

- [x] Write failing tests for artifact bindings, the `0.63` stability gate, human-review floor, and rejection of an IR-only or expanded-stem run directory.
- [x] Run focused tests and confirm failure because the registration and reporter do not exist.
- [x] Add the preregistration and identity-safe reporter.
- [x] Recompute and verify every bound SHA256 after implementation freezes.
- [x] Run focused tests and confirm pass.

### Task 4: Verification, CUDA Smoke, and Formal Run

**Files:**
- Create after smoke: `reports/x3d_s_train12_val2_user6_user7_single13_fixed_context_ir_anchored_depth_adapter_smoke.json`

**Interfaces:**
- Formal run ID: `x3d_s_ir_anchored_depth_adapter_train12_val2_user6_user7_single13_fixed_context_workers4_seed20260715`

- [x] Run all tests and `git diff --check`.
- [x] Run a two-epoch/one-batch CUDA smoke and require adapter gradient, last-two-block gradient after unfreeze, head gradient, finite losses, no worker crash, and at least 4 GiB remaining system memory.
- [ ] Commit and push the frozen implementation and experiment contract to `experiment/x3d-ir-anchored-adapter-user6-user7`.
- [ ] Launch exactly one background formal run with seed `20260715`.
- [ ] On completion, generate the identity-safe report, preserve all checkpoints, push the result report, and stop without stability training.
