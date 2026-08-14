# X3D Train12/Val2 Partial Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the shared 12-user/2-user development split and test full-temporal-coverage X3D-S with only the final two backbone blocks trainable after warmup.

**Architecture:** A dedicated runner resolves the frozen split into the effective config before hashing and saving provenance. The experiment inherits A2's complete adaptive-window input, augmentation, optimizer, loss, and 20-epoch schedule; the sole model intervention is `unfrozen_backbone_blocks: 2`.

**Tech Stack:** Python, PyTorch, YAML/JSON, pytest, existing X3D-S trainer.

## Global Constraints

- Development train users are exactly `user1,user2,user3,user5,user6,user7,user8,user9,user16,user18,user19,user20`.
- Development validation users are exactly `user21,user22`.
- Validation has 324 usable-IR trials and 36 observed classes; Macro-F1 uses labels `0..39`.
- Heldout4 and competition test remain inaccessible.
- Full adaptive windows are used in both training and validation; no clip dropout or motion-peak selection.
- Keep A2 brightness/contrast/gamma `[0.9,1.1]`, two warmup epochs, backbone LR `3e-5`, head LR `3e-4`, no label smoothing/noise/blur, and frozen BN running statistics.
- After warmup, train only the final two X3D backbone blocks.
- Preserve all artifacts and stop for review on abnormal failure; do not compare the resulting metric numerically with fold0 A2/A4-T as if populations matched.

---

### Task 1: Protected Train12/Val2 Runner

**Files:**
- Create: `scripts/run_x3d_s_train12_val2_dev.py`
- Create: `tests/test_x3d_s_train12_val2_dev.py`

**Interfaces:**
- Consumes: `metadata/splits/train12_val2_development.json` and an experiment YAML.
- Produces: effective resolved config with actual user ownership, provenance, checkpoints, and predictions under `outputs/x3d_s_ir_context_train12_val2_dev`.

- [ ] Write failing tests for exact user ownership, split disjointness, resolved-config replacement, and protected output root.
- [ ] Confirm RED due to the missing runner.
- [ ] Implement the runner and rerun focused tests.

### Task 2: Partial-Backbone Experiment Freeze

**Files:**
- Create: `configs/experiments/x3d_s_ir_context_train12_val2_partial2.yaml`
- Create: `reports/x3d_s_train12_val2_partial2_preregistration.json`

**Interfaces:**
- Consumes: the protected runner.
- Produces: one frozen candidate with full temporal coverage and `unfrozen_backbone_blocks: 2`.

- [ ] Verify the config has no clip-dropout field and preserves the A2 recipe except split/output ownership and unfreeze depth.
- [ ] Commit and push the preregistration before GPU results exist.

### Task 3: Execute and Report

**Files:**
- Create at runtime: `outputs/x3d_s_ir_context_train12_val2_dev/x3d_s_ir_context_train12_val2_partial2_seed20260715/`
- Create: `reports/x3d_s_train12_val2_partial2_report.json`
- Create: `reports/x3d_s_train12_val2_partial2_report.md`

**Interfaces:**
- Consumes: frozen candidate artifacts.
- Produces: trial Accuracy, fixed-40-class Macro-F1, per-user/worst-user and duration metrics, training gap, hashes, and a frozen decision.

- [ ] Run focused/full tests and CUDA smoke.
- [ ] Run the 20-epoch candidate and retain all checkpoints.
- [ ] Independently recompute metrics and verify actual split ownership.
- [ ] Verify, commit, push, and update the handoff.
