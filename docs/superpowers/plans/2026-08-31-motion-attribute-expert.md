# Motion Attribute Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one native H36M-17 multi-task Motion Attribute Expert and conditionally train a bounded frozen-visual residual gate only if every expert qualification gate passes.

**Architecture:** A cached T=96 gap-aware `xyz+velocity` sequence feeds a compact multi-scale residual TCN. The 256D embedding jointly predicts six motion families, sixteen deterministic geometric attributes, and auxiliary 40-class logits. A later residual gate is strictly conditional and cannot update the visual anchor.

**Tech Stack:** Python 3.12, PyTorch 2.7, NumPy, pandas, pytest, CUDA AMP, JSON/NPZ evidence.

**Spec:** `docs/superpowers/specs/2026-08-31-motion-attribute-expert-design.md`

## Global Constraints

- Fixed train12/user6-user7 split; no grouped CV or multiple seed.
- Canonical population `2039/388`; Skeleton-supported `1956/385`; class order `0..39`.
- Input `[96,17,6]`; 16 frozen attributes; 6 frozen family targets.
- Loss weights family/attribute/action = `1.00/0.50/0.25`.
- Fixed 15 epochs, batch 32, AdamW `3e-4`, weight decay `1e-4`, seed `20260715`.
- Expert parameter count below 3,000,000.
- Validation is evaluated once after epoch 15.
- Residual gate is forbidden unless every expert gate passes.
- Preserve unrelated untracked IR/Depth and hierarchical final reports.

---

### Task 1: Freeze experiment config and family mapping

**Files:**
- Create: `configs/experiments/motion_attribute_expert.yaml`
- Create: `src/experiments/motion_attribute_config.py`
- Create: `tests/test_motion_attribute_contract.py`

**Interfaces:**
- Produces: `load_motion_attribute_config(path: Path) -> dict[str, Any]`
- Produces: `motion_family_targets(class_ids: Tensor) -> Tensor [B,6]`

- [ ] Write RED tests for populations, input, family mapping, attributes, loss,
  epochs, optimizer, gates, and forbidden modalities/protocols.
- [ ] Run `pytest -q tests/test_motion_attribute_contract.py` and verify failure.
- [ ] Implement the exact Spec contract and deterministic family lookup table.
- [ ] Verify tests and commit `experiment: freeze motion attribute contract`.

---

### Task 2: Build and cache T=96 attributes

**Files:**
- Create: `src/data/motion_attribute_dataset.py`
- Create: `scripts/cache_motion_attribute_inputs.py`
- Create: `tests/test_motion_attribute_dataset.py`

**Interfaces:**
- Produces: `resample_motion_trial(rows, data_root, frames=96) -> MotionTrial`
- Produces: `compute_motion_attributes(features, mask) -> Tensor [16]`
- Produces: canonical cache with features, masks, segment IDs, raw attributes,
  normalized attributes, families, labels, users, sample IDs, availability.

- [ ] Write RED synthetic tests proving no cross-gap interpolation, segment-local
  velocity, exact 16 attributes, finite one-frame behavior, and family shape.
- [ ] Write RED real-population tests for `2039/388`, `1956/385`, 40 classes,
  projection ownership, and train-only attribute normalization.
- [ ] Implement cache generation with atomic NPZ and hashes.
- [ ] Run the formal cache once and record generation time/bytes.
- [ ] Verify and commit `feat: cache motion attribute inputs`.

---

### Task 3: Implement multi-task TCN and loss

**Files:**
- Create: `src/models/motion_attribute_expert.py`
- Create: `src/training/motion_attribute_loss.py`
- Create: `tests/test_motion_attribute_expert.py`

**Interfaces:**
- Produces: `MotionAttributeExpert.forward(features, mask) -> family_logits,
  attribute_predictions, action_logits, embedding`
- Produces: `motion_attribute_loss(output, targets) -> dict[str, Tensor]`

- [ ] Write RED tests for shapes, masks, unsupported rows, parameter ceiling,
  finite gradients for encoder/all heads, and exact weighted loss.
- [ ] Implement three multi-scale residual temporal blocks and masked pooling.
- [ ] Verify tests and commit `feat: add motion attribute expert`.

---

### Task 4: Real CUDA smoke

**Files:**
- Create: `src/train_motion_attribute_expert.py`
- Create: `scripts/run_motion_attribute_expert.py`
- Create: `tests/test_run_motion_attribute_expert.py`
- Create: `reports/motion_attribute_expert_smoke.json`

**Interfaces:**
- Produces: `run_motion_attribute_smoke(config_path, output_root) -> dict`

- [ ] Write RED smoke test for real train-only backward, validation forward,
  all parameter groups changed, fallback, finite values, reload parity, memory.
- [ ] Implement output overwrite protection and atomic evidence.
- [ ] Run formal CUDA smoke once; stop and report on failure.
- [ ] Verify and commit `experiment: qualify motion attribute runtime`.

---

### Task 5: Fixed 15-epoch expert training

**Files:**
- Modify: `src/train_motion_attribute_expert.py`
- Modify: `scripts/run_motion_attribute_expert.py`
- Create: `tests/test_train_motion_attribute_expert.py`

**Interfaces:**
- Produces: `run_motion_attribute_training(config_path) -> dict`
- Produces: checkpoint, train predictions, validation predictions, gate decision.

- [ ] Write RED tests for fixed epochs, validation-once, train-only sampling,
  checkpoint/RNG resume, canonical prediction ownership, and gate recomputation.
- [ ] Implement training and metric collection.
- [ ] Run formal 15 epochs once.
- [ ] Atomically write expert gate before any residual action.
- [ ] Verify and commit `report: evaluate motion attribute expert`.

---

### Task 6: Conditional residual gate

**Files:**
- Create only after expert pass: `src/models/motion_attribute_residual.py`
- Create only after expert pass: `scripts/run_motion_attribute_residual.py`
- Create only after expert pass: `tests/test_motion_attribute_residual.py`

- [ ] If any expert gate failed, record `residual_executed=false` and skip all
  remaining Task 6 steps.
- [ ] If all passed, write RED tests for frozen visual evidence, exact class
  budgets, zero-init parity, user protection, Eat_food harm ceiling, and gate.
- [ ] Train only the small residual MLP on cached train evidence.
- [ ] Evaluate user6/user7 once and freeze the residual decision.

---

### Task 7: Report and unified audit

**Files:**
- Create: `scripts/report_motion_attribute_expert.py`
- Create: `tests/test_report_motion_attribute_expert.py`
- Create: `reports/motion_attribute_expert_result.json`
- Create: `reports/motion_attribute_expert_result.md`

- [ ] Recompute every metric/gate/hash from archives.
- [ ] Run full pytest, compileall, and `git diff --check`.
- [ ] Perform Standards/Spec self-audit; resolve scientific-integrity findings.
- [ ] Commit final evidence and update the temporary handoff document.

## Plan Self-Review

- Contract/family mapping: Task 1.
- Gap-aware T=96 and 16 attributes: Task 2.
- Multi-task TCN/loss: Task 3.
- CUDA/resource/reload gate: Task 4.
- Fixed training/expert qualification: Task 5.
- Strict conditional residual: Task 6.
- Reproducible report/audit: Task 7.
- Candidate, mappings, attributes, shapes, losses, epochs, optimizer, gates,
  paths, and stop conditions contain no unresolved placeholders.
