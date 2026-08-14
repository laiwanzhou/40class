# X3D Fold0 A4-T Temporal Clip Dropout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether reducing simultaneous local-window exposure during training improves A2 cross-user generalization.

**Architecture:** Keep the complete A2 recipe unchanged except for a training-only temporal keep fraction of `0.5`. A trial with `K` adaptive windows retains a deterministic-per-epoch random chronological subset of `ceil(K/2)` windows; validation continues to use all `K` windows and mean-probability aggregation.

**Tech Stack:** Python, PyTorch, pytest, YAML, existing X3D-S fold0 development runner.

## Global Constraints

- Use frozen fold 0 and canonical seed `20260715`.
- Do not read heldout4 or competition test data.
- Do not mutate canonical Phase 4/5 evidence or A2 artifacts.
- Keep A2 backbone, learning rates, warmup, scheduler, loss, ROI, 13-frame clips, photometric augmentation, and validation inference unchanged.
- Stop for human review on an Accuracy regression greater than `0.02`; never delete trained artifacts automatically.

---

### Task 1: Training-Only Window Selection Contract

**Files:**
- Modify: `src/data/x3d_clip_dataset.py`
- Test: `tests/test_x3d_clip_dataset.py`

**Interfaces:**
- Consumes: adaptive windows from `partition_trial_windows()`.
- Produces: `select_training_window_indices(num_windows, keep_fraction, generator) -> list[int]` and dataset argument `train_clip_keep_fraction: float = 1.0`.

- [ ] Add failing tests proving `K=1` remains one, `K=2` retains one, `K=8` retains four, selections change across epochs, selected windows stay ordered, and validation retains all windows.
- [ ] Run the focused tests and confirm they fail because the new API does not exist.
- [ ] Implement deterministic selection and apply it only to training datasets.
- [ ] Run focused and existing dataset/trainer contract tests.

### Task 2: A4-T Experiment Freeze

**Files:**
- Create: `configs/experiments/x3d_s_ir_context_fold0_dev_a4_t.yaml`
- Create: `reports/x3d_s_fold0_a4_t_preregistration.json`
- Modify: `scripts/run_x3d_s_fold0_dev.py`
- Test: `tests/test_x3d_s_fold0_dev.py`

**Interfaces:**
- Consumes: `temporal.train_clip_keep_fraction` from the resolved config.
- Produces: a protected development run whose provenance records the A4-T temporal policy.

- [ ] Add a failing runner contract test proving only the training dataset receives `0.5` while validation receives `1.0`.
- [ ] Thread the frozen temporal field into dataset construction and provenance.
- [ ] Verify a structured A2/A4-T config comparison has exactly one behavioral difference.
- [ ] Run fold0 development and trainer contract tests.

### Task 3: Execute and Freeze A4-T

**Files:**
- Create at runtime: `outputs/x3d_s_ir_context_fold0_dev/x3d_s_ir_context_fold0_dev_a4_t_seed20260715/`
- Create: `reports/x3d_s_fold0_a4_t_report.json`
- Create: `reports/x3d_s_fold0_a4_t_report.md`

**Interfaces:**
- Consumes: frozen A4-T config and fold assignment.
- Produces: retained checkpoint, predictions, independent metrics, comparison with A2, and artifact hashes.

- [ ] Run the CUDA smoke test and verify gradients plus canonical/A2 artifact hashes.
- [ ] Run the complete 20-epoch fold0 experiment.
- [ ] Recompute all frozen metrics independently and compare with A2.
- [ ] Apply target and regression rules, retain every artifact, verify, commit, and push.
