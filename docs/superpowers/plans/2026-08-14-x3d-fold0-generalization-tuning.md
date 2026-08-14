# X3D Fold 0 Generalization Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch an isolated fold0 development experiment targeting Accuracy >= 0.6300, Macro-F1 >= 0.5200, and worst-user Accuracy >= 0.5338 without modifying canonical Phase 4/5 evidence.

**Architecture:** Extend the existing X3D trainer with configuration-driven partial-backbone fine-tuning, trial-level label smoothing, and deterministic clip-consistent IR photometric augmentation. A dedicated fold0 development runner consumes the frozen Phase 4 assignment but writes only to a protected development output root and records canonical artifact hashes before training.

**Tech Stack:** Python 3.11, PyTorch 2.7, PyTorchVideo X3D-S, torchvision transforms, pytest, YAML, PowerShell/CUDA.

## Global Constraints

- Branch is `test/x3d-fold0-generalization`.
- Fold 0 is development-only after tuning; its results are not unbiased OOF evidence.
- Do not overwrite or replace canonical Phase 4 OOF or Phase 5 `ir_x3d_s_k400_pure` evidence.
- Do not access heldout-4 or competition-test data.
- Keep fold ownership, ROI assets, 13-frame clips, deterministic validation sampling, mean-probability trial aggregation, equal-trial accumulation, and frozen BatchNorm running statistics unchanged.
- Every experiment has a fresh run ID and output directory; never automatically delete checkpoints.
- Stop successfully only when one checkpoint satisfies Accuracy >= 0.6300, Macro-F1 >= 0.5200, and worst-user Accuracy >= 0.5338 simultaneously.
- An Accuracy regression greater than 0.02 from the parent candidate requires retained artifacts and human review.

---

### Task 1: Protect Canonical Artifacts and Add the Fold0 Development Entry Point

**Files:**
- Create: `scripts/run_x3d_s_fold0_dev.py`
- Create: `tests/test_x3d_s_fold0_dev.py`
- Create: `configs/experiments/x3d_s_ir_context_fold0_dev_a1.yaml`

**Interfaces:**
- Consumes: `trainer.validate_oof_assignment()`, `trainer.prepare_partition_manifest()`, `trainer.train_partition()`, and `X3DClipDataset`.
- Produces: `collect_canonical_artifact_hashes(paths: Sequence[Path]) -> dict[str, str]`, `select_fold0(assignment, allowed_users) -> UserFold`, and a CLI that writes `development_provenance.json` before training.

- [ ] **Step 1: Write failing runner-contract tests**

Add tests that require fold index 0 and seed 20260715, reject run IDs containing `strict_v3`, reject any output root other than `outputs/x3d_s_ir_context_fold0_dev`, select the frozen fold0 users, and hash canonical files without modifying them.

```python
def test_fold0_dev_rejects_canonical_run_id() -> None:
    args = parser().parse_args(required_args + ["--run-id", "strict_v3_seed20260715"])
    with pytest.raises(ValueError, match="canonical"):
        validate_dev_args(args)


def test_collect_canonical_hashes_is_read_only(tmp_path: Path) -> None:
    artifact = tmp_path / "formal_outer_refit.pt"
    artifact.write_bytes(b"frozen")
    before = artifact.stat()
    hashes = collect_canonical_artifact_hashes([artifact])
    assert hashes[str(artifact.resolve())] == hashlib.sha256(b"frozen").hexdigest()
    assert artifact.stat() == before
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_x3d_s_fold0_dev.py -q`

Expected: FAIL because `scripts.run_x3d_s_fold0_dev` does not exist.

- [ ] **Step 3: Implement the protected runner and A1 config**

The runner must:

```python
DEV_OUTPUT_ROOT = Path("outputs/x3d_s_ir_context_fold0_dev")
FOLD_INDEX = 0
CANONICAL_SEED = 20260715
FORBIDDEN_RUN_ID_PARTS = ("strict_v3", "phase4", "phase5")
```

It loads the immutable three-fold assignment, selects only fold0, writes the resolved config and canonical SHA256 map, builds outer-train and outer-validation datasets, then calls `train_partition()` directly. Provenance must set `role="fold0_development_tuning"`, `unbiased_oof=False`, and `canonical_evidence_mutation_permitted=False`.

Create the A1 config by copying the canonical data/loader/ROI/AMP contracts and changing only:

```yaml
output_root: outputs/x3d_s_ir_context_fold0_dev
dropout: 0.25
optimizer:
  backbone_lr: 0.00001
  head_lr: 0.0003
  weight_decay: 0.05
  gradient_accumulation: 4
  gradient_clip: 1.0
training:
  epochs: 20
  scheduler_horizon_epochs: 20
  warmup_epochs: 5
  unfrozen_backbone_blocks: 2
  label_smoothing: 0.10
  patience: 8
  early_stopping_enabled: true
augmentation:
  brightness: [0.85, 1.15]
  contrast: [0.85, 1.15]
  gamma: [0.85, 1.15]
  noise_std_max: 0.025
  blur_probability: 0.20
  blur_kernel_size: 3
  blur_sigma: [0.10, 1.20]
```

- [ ] **Step 4: Run the runner-contract tests**

Run: `python -m pytest tests/test_x3d_s_fold0_dev.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the isolated experiment scaffold**

```powershell
git add scripts/run_x3d_s_fold0_dev.py tests/test_x3d_s_fold0_dev.py configs/experiments/x3d_s_ir_context_fold0_dev_a1.yaml
git commit -m "Add protected X3D fold0 development runner"
```

### Task 2: Add Selective X3D Backbone Unfreezing

**Files:**
- Modify: `src/models/x3d_s_visual_expert.py`
- Modify: `src/train_x3d_s_visual_expert.py`
- Modify: `tests/test_x3d_s_visual_expert.py`
- Modify: `tests/test_x3d_s_trainer_contract.py`

**Interfaces:**
- Consumes: the existing `X3DSVisualExpert.backbone.blocks` sequence.
- Produces: `set_backbone_trainable(enabled: bool, *, last_blocks: int | None = None) -> None` and configuration key `training.unfrozen_backbone_blocks`.

- [ ] **Step 1: Write failing selective-unfreeze tests**

Use a tiny backbone exposing four ordered blocks. Assert warmup freezes all backbone parameters; `last_blocks=2` enables only blocks 2 and 3; BatchNorm running statistics remain frozen; and invalid values 0 or values larger than the block count raise `ValueError`.

```python
model.set_backbone_trainable(True, last_blocks=2)
assert not any(p.requires_grad for p in model.backbone.blocks[0].parameters())
assert not any(p.requires_grad for p in model.backbone.blocks[1].parameters())
assert all(p.requires_grad for p in model.backbone.blocks[2].parameters())
assert all(p.requires_grad for p in model.backbone.blocks[3].parameters())
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_x3d_s_visual_expert.py tests/test_x3d_s_trainer_contract.py -q`

Expected: FAIL because the method has no `last_blocks` contract and validation fixes warmup at 2.

- [ ] **Step 3: Implement partial unfreezing and configuration validation**

Preserve existing boolean callers by treating `last_blocks=None` as all blocks. During training and finalization call:

```python
model.set_backbone_trainable(
    epoch > warmup_epochs,
    last_blocks=(unfrozen_backbone_blocks if epoch > warmup_epochs else None),
)
```

Allow positive `warmup_epochs < epochs`, and validate `unfrozen_backbone_blocks` as a positive integer when supplied. Record the trainable backbone parameter count and unfreeze depth in history/summary/provenance.

- [ ] **Step 4: Run focused model/trainer tests**

Run: `python -m pytest tests/test_x3d_s_visual_expert.py tests/test_x3d_s_trainer_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit selective unfreezing**

```powershell
git add src/models/x3d_s_visual_expert.py src/train_x3d_s_visual_expert.py tests/test_x3d_s_visual_expert.py tests/test_x3d_s_trainer_contract.py
git commit -m "Support selective X3D backbone fine tuning"
```

### Task 3: Add Deterministic IR Augmentation and Trial-Level Label Smoothing

**Files:**
- Modify: `src/data/x3d_clip_dataset.py`
- Modify: `src/train_x3d_s_visual_expert.py`
- Modify: `tests/test_x3d_clip_dataset.py`
- Modify: `tests/test_x3d_s_trainer_contract.py`

**Interfaces:**
- Consumes: `augmentation` config mapping and the dataset's per-clip seeded `torch.Generator`.
- Produces: `IRAugmentationConfig`, `_apply_ir_photometric_augmentation()`, and `_trial_nll_loss(log_probs, labels, label_smoothing, reduction)`.

- [ ] **Step 1: Write failing augmentation and loss tests**

Require identical outputs for the same seed/epoch, changed training pixels when enabled, unchanged validation preprocessing, clip-consistent brightness/contrast/gamma/blur parameters, finite bounded pixels before normalization, and exact equivalence to `F.nll_loss` at smoothing 0.

```python
expected = F.nll_loss(log_probs, labels, reduction="sum")
actual = _trial_nll_loss(log_probs, labels, label_smoothing=0.0, reduction="sum")
torch.testing.assert_close(actual, expected)
```

For smoothing 0.1, compare against the explicit uniform-target formula:

```python
expected = -((0.9 * log_probs[range(n), labels]) + (0.1 * log_probs.mean(dim=1))).sum()
```

- [ ] **Step 2: Verify focused tests fail**

Run: `python -m pytest tests/test_x3d_clip_dataset.py tests/test_x3d_s_trainer_contract.py -q`

Expected: FAIL because the augmentation config and smoothed trial loss do not exist.

- [ ] **Step 3: Implement clip-consistent photometric augmentation**

Parse and validate all numeric ranges once in the dataset constructor. Sample one brightness, contrast, gamma, blur decision, and blur sigma per clip; apply those parameters to all 13 frames. Add independent low-amplitude Gaussian sensor noise per frame using the same deterministic generator stream. Clamp to `[0,1]` before repeating grayscale into three channels and applying X3D normalization. Validation ignores augmentation settings.

- [ ] **Step 4: Implement label smoothing without breaking equal-trial accumulation**

Replace only the summed loss computation inside `run_model_epoch()`:

```python
loss_sum = _trial_nll_loss(
    output.main_logits,
    labels,
    label_smoothing=label_smoothing,
    reduction="sum",
)
```

Continue dividing accumulated gradients by `accumulated_trial_count`, so every trial retains equal weight regardless of microbatch size or clip count. Pass `label_smoothing` explicitly from configuration and record it in summaries.

- [ ] **Step 5: Run focused data/trainer tests**

Run: `python -m pytest tests/test_x3d_clip_dataset.py tests/test_x3d_s_trainer_contract.py -q`

Expected: PASS.

- [ ] **Step 6: Commit regularization behavior**

```powershell
git add src/data/x3d_clip_dataset.py src/train_x3d_s_visual_expert.py tests/test_x3d_clip_dataset.py tests/test_x3d_s_trainer_contract.py
git commit -m "Regularize X3D fold0 development training"
```

### Task 4: Verify Contracts and Launch A1

**Files:**
- Modify: `docs/superpowers/plans/2026-08-14-x3d-fold0-generalization-tuning.md`
- Create at runtime: `outputs/x3d_s_ir_context_fold0_dev/<run-id>/development_provenance.json`
- Create at runtime: `outputs/x3d_s_ir_context_fold0_dev/<run-id>/resolved_config.yaml`

**Interfaces:**
- Consumes: Tasks 1-3 and `metadata/splits/train14_oof_3fold.json`.
- Produces: A1 run directory, process log, checkpoints, prediction archives, and development metrics.

- [ ] **Step 1: Run the complete relevant test suite**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest `
  tests/test_x3d_clip_dataset.py `
  tests/test_x3d_s_visual_expert.py `
  tests/test_x3d_s_trainer_contract.py `
  tests/test_x3d_s_fold0_dev.py `
  tests/test_x3d_s_phase4_freeze.py `
  tests/test_verify_x3d_s_phase4_checkpoints.py `
  -q
```

Expected: PASS with no canonical artifact changes.

- [ ] **Step 2: Run a CUDA smoke experiment**

Use the development runner with a fresh smoke run ID and one train/validation batch. Verify finite loss, head gradients during warmup, output isolation, and preserved canonical hashes.

- [ ] **Step 3: Re-run the focused tests after smoke output creation**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_fold0_dev.py tests/test_x3d_s_phase4_freeze.py -q`

Expected: PASS.

- [ ] **Step 4: Launch the fixed A1 experiment**

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.run_x3d_s_fold0_dev `
  --config configs/experiments/x3d_s_ir_context_fold0_dev_a1.yaml `
  --oof-fold-assignment metadata/splits/train14_oof_3fold.json `
  --run-id x3d_s_ir_context_fold0_dev_a1_seed20260715 `
  --seed 20260715
```

The process writes stdout/stderr to an experiment-specific log. Do not reuse the run ID.

- [ ] **Step 5: Commit implementation and launch provenance**

```powershell
git add docs/superpowers/plans/2026-08-14-x3d-fold0-generalization-tuning.md
git commit -m "Launch X3D fold0 generalization experiment"
```

### Task 5: Evaluate A1 and Apply the Frozen Decision Rule

**Files:**
- Create: `scripts/report_x3d_s_fold0_dev.py`
- Create: `tests/test_report_x3d_s_fold0_dev.py`
- Create at runtime: `reports/x3d_s_fold0_generalization_tuning.md`
- Create at runtime: `reports/x3d_s_fold0_generalization_tuning.json`

**Interfaces:**
- Consumes: `history.csv`, `val_predictions_best_accuracy.npz`, and `run_summary.json` from the A1 run.
- Produces: `evaluate_candidate(metrics, parent_metrics) -> str` returning `target_met`, `continue_stage_a`, or `human_review_regression`.

- [ ] **Step 1: Write failing decision-rule tests**

Cover simultaneous target success, Accuracy >= 0.63 with failed Macro-F1, failed worst-user, and a greater-than-0.02 Accuracy regression requiring human review.

- [ ] **Step 2: Implement the fold0 report**

Recompute trial metrics from the saved prediction archive, including per-user and duration buckets. Never trust summary-only metrics. Include canonical baseline deltas, target gaps, selected epoch, trainable parameter count, and hashes.

- [ ] **Step 3: Run report tests and generate the report**

Run: `python -m pytest tests/test_report_x3d_s_fold0_dev.py -q`

Expected: PASS.

Then run the reporter against A1. If all three targets pass, stop. If Accuracy regresses by more than 0.02, preserve everything and stop for human review. Otherwise proceed only to the bounded A2 candidate set described by the approved design.

- [ ] **Step 4: Commit the evidence report**

```powershell
git add scripts/report_x3d_s_fold0_dev.py tests/test_report_x3d_s_fold0_dev.py reports/x3d_s_fold0_generalization_tuning.md reports/x3d_s_fold0_generalization_tuning.json
git commit -m "Report X3D fold0 tuning evidence"
```
