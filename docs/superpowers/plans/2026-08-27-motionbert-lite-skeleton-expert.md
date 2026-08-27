# MotionBERT-Lite Skeleton Expert P6-B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify one pretrained MotionBERT-Lite Skeleton expert on the fixed train12 to user6/user7 development boundary and stop unless it supplies sufficient unique visual rescues.

**Architecture:** A train12-fit selected Skeleton identity/projection produces one longest-segment T=96 projected pose sequence per trial. The frozen MotionBERT-Lite backbone first generates cached 512D embeddings for a low-cost linear qualification head. Only a passing B1 result may unfreeze the final two DSTformer block pairs for B2.

**Tech Stack:** Python 3.12, PyTorch 2.7, NumPy, pandas, torchvision/PIL, official MotionBERT-Lite Apache-2.0 source and checkpoint, pytest, JSON/Markdown/NPZ evidence.

**Spec:** `docs/superpowers/specs/2026-08-27-motionbert-lite-skeleton-expert-design.md`

## Global Constraints

- Exactly one backbone candidate: MotionBERT-Lite.
- Fixed train12 population is 2,039; fixed user6/user7 validation is 388; class order is `0..39`.
- Skeleton-supported rows are 1,956 train and 385 validation; unsupported canonical rows use train12 prior fallback in metrics.
- No three-fold, multiple seed, P6-A, CTR-GCN, PoseC3D, full MotionBERT, fusion, or distillation.
- Upstream source commit is `705d3a95354db8bdb696b3492e47a3b5537174ff`, Apache-2.0.
- Checkpoint revision is `370a9196aa3c89198b134c82476143b01c0fb32c`; bytes `64,099,897`; SHA-256/LFS OID `6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461`.
- MotionBERT input is exactly `[T=96,J=17,C=3]` projected normalized xy plus confidence.
- B1 is frozen-backbone only. B2 may run only from an atomically written B1 pass decision.
- Use `D:\Anaconda\envs\PyTorch2.7\python.exe` for every command.
- Formal outputs use new paths and refuse overwrite; experiment crashes are never retried without reporting.

---

## Planned File Structure

| File | Responsibility |
|---|---|
| `configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml` | Frozen source, data, model, B0/B1/B2, gates, outputs |
| `third_party/motionbert/LICENSE` | Upstream Apache-2.0 license |
| `third_party/motionbert/NOTICE.md` | Source commit, modified-file notice, provenance |
| `third_party/motionbert/DSTformer.py` | Minimal attributed upstream DSTformer implementation |
| `third_party/motionbert/drop.py` | Minimal attributed DropPath dependency |
| `scripts/fetch_motionbert_lite_checkpoint.py` | Atomic pinned-revision download and size/SHA verification |
| `src/data/motionbert_skeleton_dataset.py` | Longest-segment T=96 projected pose contract and canonical fallback |
| `src/models/motionbert_lite_skeleton.py` | Checkpoint adapter, coverage audit, pooled embedding and head |
| `src/train_motionbert_lite_skeleton_expert.py` | Smoke, embedding cache, B1, conditional B2 primitives |
| `scripts/run_motionbert_lite_skeleton_expert.py` | CLI orchestration and overwrite/resume policy |
| `scripts/report_motionbert_lite_skeleton_expert.py` | Archive metric/hash recomputation and promotion report |
| `tests/test_motionbert_skeleton_dataset.py` | Data contract and gap/ownership tests |
| `tests/test_motionbert_lite_skeleton.py` | Model/checkpoint/freeze tests |
| `tests/test_run_motionbert_lite_skeleton_expert.py` | Smoke, B1/B2 gate and artifact tests |
| `tests/test_report_motionbert_lite_skeleton_expert.py` | Metric and hash recomputation tests |

---

### Task 1: Freeze P6-B contract and vendor the pinned minimal upstream source

**Files:**
- Create: `configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml`
- Create: `third_party/motionbert/LICENSE`
- Create: `third_party/motionbert/NOTICE.md`
- Create: `third_party/motionbert/DSTformer.py`
- Create: `third_party/motionbert/drop.py`
- Create: `scripts/fetch_motionbert_lite_checkpoint.py`
- Create: `tests/test_motionbert_lite_contract.py`

**Interfaces:**
- Produces: `load_motionbert_p6b_config(path: Path) -> dict[str, Any]`
- Produces: `fetch_motionbert_lite_checkpoint(config_path: Path) -> dict[str, Any]`

- [ ] **Step 1: Write the failing frozen-contract test**

```python
def test_motionbert_contract_freezes_one_candidate_and_fixed_boundary() -> None:
    config = load_motionbert_p6b_config(CONFIG)
    assert config["stage"] == "P6-B"
    assert config["candidate"] == "motionbert_lite"
    assert config["population"]["train_samples"] == 2039
    assert config["population"]["validation_samples"] == 388
    assert config["input"] == {
        "frames": 96,
        "joints": 17,
        "channels": ["projected_x", "projected_y", "confidence"],
        "segment_policy": "longest_retained_then_smallest_index",
    }
    assert config["policy"]["grouped_cv_allowed"] is False
    assert config["policy"]["multiple_seeds_allowed"] is False
```

- [ ] **Step 2: Run the test and verify RED**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_motionbert_lite_contract.py
```

Expected: FAIL because the config/loader/source adapter is absent.

- [ ] **Step 3: Implement the config validator and vendor exact upstream files**

Copy only `lib/model/DSTformer.py`, `lib/model/drop.py`, and `LICENSE` from
upstream commit `705d3a95354db8bdb696b3492e47a3b5537174ff`. Modify imports only so
the files work as a local package. `NOTICE.md` records upstream URL, commit,
Apache-2.0, copied files, and the import modification.

The config freezes all Spec values, including B1/B2 optimizer settings and
gates. The loader rejects any changed source commit, weight revision/path,
bytes, SHA, population, modality, class order, candidate, or authorization.

- [ ] **Step 4: Write RED tests for atomic weight verification**

```python
def test_fetch_rejects_wrong_bytes_or_sha(tmp_path: Path) -> None:
    target = tmp_path / "latest_epoch.bin"
    target.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="checkpoint provenance"):
        verify_motionbert_checkpoint(target, EXPECTED_BYTES, EXPECTED_SHA256)
```

The fetch script downloads only the pinned Hugging Face revision URL to a
temporary file, verifies bytes and SHA, then atomically moves it into the user
cache. An existing verified file is reused; an existing invalid file is never
overwritten silently.

- [ ] **Step 5: Verify and commit Task 1**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_motionbert_lite_contract.py
D:\Anaconda\envs\PyTorch2.7\python.exe -m compileall -q third_party/motionbert scripts/fetch_motionbert_lite_checkpoint.py
git diff --check
git add configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml third_party/motionbert scripts/fetch_motionbert_lite_checkpoint.py tests/test_motionbert_lite_contract.py
git commit -m "experiment: freeze MotionBERT Lite skeleton contract"
```

---

### Task 2: Build the longest-segment T=96 MotionBERT dataset

**Files:**
- Create: `src/data/motionbert_skeleton_dataset.py`
- Create: `tests/test_motionbert_skeleton_dataset.py`

**Interfaces:**
- Produces: `MotionBERTSkeletonSample(sequence, available, quality)`
- Produces: `MotionBERTSkeletonDataset(config, partition) -> Dataset[dict[str, object]]`

- [ ] **Step 1: Write failing segment and projection tests**

```python
def test_longest_segment_is_selected_without_cross_gap_interpolation() -> None:
    rows = fixture_rows(segment_lengths={0: 4, 1: 7, 2: 7})
    result = build_motionbert_sequence(rows, projection=PROJECTION, frames=96)
    assert result.selected_segment_index == 1
    assert result.source_frame_ids.min() >= SEGMENT_ONE_FIRST_FRAME
    assert result.source_frame_ids.max() <= SEGMENT_ONE_LAST_FRAME
    assert result.sequence.shape == (96, 17, 3)


def test_projection_and_normalization_are_fit_scope_immutable() -> None:
    train = MotionBERTSkeletonDataset(CONFIG, partition="train")
    validation = MotionBERTSkeletonDataset(CONFIG, partition="validation")
    assert train.projection_sha256 == validation.projection_sha256
    assert set(train.projection_fit_user_ids).isdisjoint({"user6", "user7"})
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_motionbert_skeleton_dataset.py
```

- [ ] **Step 3: Implement canonical dataset and exact output**

Use canonical trials from `build_canonical_trials`; never filter the Dataset.
Join the selected-final clean view by sample ID. Supported rows load the frozen
candidate JSON, root/scale normalize, project, select the deterministic longest
segment, and resample inside it. Unsupported rows return zero `[96,17,3]`,
`available=False`, and an auditable reason. Output includes label, user,
sample ID, source frames, selected segment, discarded fraction, and quality.

- [ ] **Step 4: Run synthetic plus real-population tests**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_motionbert_skeleton_dataset.py
```

Expected: train/validation lengths `2039/388`, supported counts `1956/385`, all
40 labels retained, no user6/user7 projection fit, finite sequences.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/data/motionbert_skeleton_dataset.py tests/test_motionbert_skeleton_dataset.py
git commit -m "feat: add MotionBERT skeleton input contract"
```

---

### Task 3: Adapt MotionBERT-Lite checkpoint and frozen classification head

**Files:**
- Create: `src/models/motionbert_lite_skeleton.py`
- Create: `tests/test_motionbert_lite_skeleton.py`

**Interfaces:**
- Produces: `MotionBERTLiteSkeletonExpert.forward(sequence, available) -> dict[str, Tensor]`
- Produces: `load_motionbert_lite_backbone(checkpoint, config) -> CheckpointCoverage`
- Produces: `set_motionbert_train_stage(model, stage: Literal["B1", "B2"]) -> None`

- [ ] **Step 1: Write failing shape, mask, and freeze tests**

```python
def test_motionbert_expert_returns_embedding_and_prior_safe_logits() -> None:
    model = tiny_motionbert_expert()
    output = model(SEQUENCE, AVAILABLE)
    assert output["sequence_features"].shape == (2, 96, 17, 512)
    assert output["embedding"].shape == (2, 512)
    assert output["logits"].shape == (2, 40)
    assert torch.isfinite(output["logits"]).all()


def test_b1_changes_only_head_and_b2_unfreezes_last_two_block_pairs() -> None:
    model = tiny_motionbert_expert()
    set_motionbert_train_stage(model, "B1")
    assert trainable_names(model) == HEAD_NAMES
    set_motionbert_train_stage(model, "B2")
    assert trainable_names(model) == B2_NAMES
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_motionbert_lite_skeleton.py
```

- [ ] **Step 3: Implement the adapter and strict checkpoint translation**

Load `checkpoint["model_pos"]`. Strip only documented distributed prefixes.
Require shape equality and at least 99% pretrained backbone element coverage;
record every missing/unexpected tensor. Pool only available rows and multiply
unsupported embeddings/logits by the availability mask before the runner
applies the train-only prior.

- [ ] **Step 4: Verify and commit Task 3**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_motionbert_lite_skeleton.py
git diff --check
git add src/models/motionbert_lite_skeleton.py tests/test_motionbert_lite_skeleton.py
git commit -m "feat: adapt pretrained MotionBERT Lite expert"
```

---

### Task 4: Run B0 real-data resource smoke

**Files:**
- Create: `src/train_motionbert_lite_skeleton_expert.py`
- Create: `scripts/run_motionbert_lite_skeleton_expert.py`
- Create: `tests/test_run_motionbert_lite_skeleton_expert.py`

**Interfaces:**
- Produces: `run_motionbert_smoke(config_path: Path, output_root: Path) -> dict[str, Any]`

- [ ] **Step 1: Write the failing smoke contract test**

```python
def test_smoke_loads_pretraining_and_updates_head_only(tmp_path: Path) -> None:
    report = run_motionbert_smoke(CONFIG, output_root=tmp_path / "smoke")
    assert report["status"] == "smoke_passed"
    assert report["pretrained_element_coverage"] >= 0.99
    assert report["finite_forward_backward"] is True
    assert report["changed_parameter_groups"] == ["head"]
    assert report["pretrained_random_embedding_max_abs_delta"] > 0
    assert set(report["gradient_user_ids"]).isdisjoint({"user6", "user7"})
    assert report["peak_cuda_mib"] < 8151
```

- [ ] **Step 2: Run test and verify RED**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_run_motionbert_lite_skeleton_expert.py::test_smoke_loads_pretraining_and_updates_head_only
```

- [ ] **Step 3: Implement B0 and run formal smoke**

The smoke selects two supported train rows and two supported validation rows but
only train rows enter backward. Use CUDA AMP, save resolved config, exact
checkpoint/source/data hashes, coverage, memory, time, parameter groups, finite
checks, and a reload check. Refuse an existing formal smoke path.

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/fetch_motionbert_lite_checkpoint.py
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/run_motionbert_lite_skeleton_expert.py --mode smoke
```

- [ ] **Step 4: Verify smoke and commit evidence**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_run_motionbert_lite_skeleton_expert.py
git add src/train_motionbert_lite_skeleton_expert.py scripts/run_motionbert_lite_skeleton_expert.py tests/test_run_motionbert_lite_skeleton_expert.py reports/motionbert_lite_skeleton_expert_p6b_smoke.json
git commit -m "experiment: qualify MotionBERT Lite runtime"
```

Stop and report if B0 fails. Do not auto-retry.

---

### Task 5: Cache embeddings and run B1 frozen-backbone qualification

**Files:**
- Modify: `src/train_motionbert_lite_skeleton_expert.py`
- Modify: `scripts/run_motionbert_lite_skeleton_expert.py`
- Modify: `tests/test_run_motionbert_lite_skeleton_expert.py`

**Interfaces:**
- Produces: `cache_motionbert_embeddings(config_path) -> dict[str, Any]`
- Produces: `run_motionbert_b1(config_path) -> dict[str, Any]`

- [ ] **Step 1: Write failing ownership, cache, and fixed-epoch tests**

```python
def test_b1_cache_and_predictions_preserve_canonical_union(tmp_path: Path) -> None:
    result = run_tiny_b1(tmp_path)
    assert result["epochs_completed"] == 20
    assert result["validation_evaluation_count"] == 1
    assert len(result["train_sample_ids"]) == 2039
    assert len(result["validation_sample_ids"]) == 388
    assert result["validation_users_entered_training"] is False
    assert result["cache"]["embedding_shape"] == [2427, 512]
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_run_motionbert_lite_skeleton_expert.py
```

- [ ] **Step 3: Implement atomic cache, B1, and gate**

Cache train and validation embeddings in canonical order with availability,
quality, labels, users, and sample IDs. B1 trains only the head on supported
train embeddings for 20 epochs and evaluates train/validation once. Join the
frozen `visual_only/validation_predictions.npz` by sample ID; compute paired
rescue/harm and oracle. Atomically write `b1_decision.json` before any B2 call.

- [ ] **Step 4: Execute B1 and obey the gate**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/run_motionbert_lite_skeleton_expert.py --mode b1
```

If any B1 gate fails, stop P6-B, write the final rejection report, and do not
start B2. If all pass, proceed to Task 6 without asking for another model choice
because the user already authorized the conditional design.

- [ ] **Step 5: Commit B1 evidence**

```powershell
git add reports/motionbert_lite_skeleton_expert_p6b_b1.json outputs/motionbert_lite_skeleton_expert_p6b/b1_decision.json
git commit -m "report: qualify frozen MotionBERT Lite expert"
```

---

### Task 6: Conditionally run B2 final-two-block fine-tuning

**Files:**
- Modify: `src/train_motionbert_lite_skeleton_expert.py`
- Modify: `scripts/run_motionbert_lite_skeleton_expert.py`
- Modify: `tests/test_run_motionbert_lite_skeleton_expert.py`

**Interfaces:**
- Produces: `run_motionbert_b2(config_path: Path) -> dict[str, Any]`

- [ ] **Step 1: Write failing gate and trainable-parameter tests**

```python
def test_b2_refuses_missing_or_failed_b1_decision(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="B1 pass"):
        run_motionbert_b2(CONFIG, output_root=tmp_path)


def test_b2_uses_only_last_two_block_pairs_and_head(tmp_path: Path) -> None:
    result = run_tiny_b2_with_pass(tmp_path)
    assert result["epochs_completed"] == 10
    assert result["trainable_block_indices"] == [3, 4]
    assert result["validation_evaluation_count"] == 1
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_run_motionbert_lite_skeleton_expert.py
```

- [ ] **Step 3: Implement strict conditional B2 and checkpoint resume**

Read and hash-verify B1 decision and B1 head. Train supported train rows for
exactly 10 epochs with the frozen optimizer groups. Checkpoint each epoch with
model, optimizer, epoch, history, RNG, source/config/input/decision hashes.
Resume only an exact matching incomplete checkpoint. Evaluate validation once
after epoch 10 and compute the fixed fusion-qualification gate.

- [ ] **Step 4: Run B2 only if B1 passed**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/run_motionbert_lite_skeleton_expert.py --mode b2
```

- [ ] **Step 5: Commit conditional B2 evidence**

```powershell
git add reports/motionbert_lite_skeleton_expert_p6b_b2.json
git commit -m "report: evaluate partial MotionBERT Lite fine tuning"
```

---

### Task 7: Recompute reports, audit, and hand off

**Files:**
- Create: `scripts/report_motionbert_lite_skeleton_expert.py`
- Create: `tests/test_report_motionbert_lite_skeleton_expert.py`
- Create: `reports/motionbert_lite_skeleton_expert_p6b.md`

**Interfaces:**
- Produces: `build_motionbert_p6b_report(config_path: Path) -> dict[str, Any]`

- [ ] **Step 1: Write failing archive/hash recomputation tests**

```python
def test_report_recomputes_metrics_and_visual_oracle() -> None:
    report = build_motionbert_p6b_report(CONFIG)
    assert report["metrics_recomputed_from_archives"] is True
    assert report["development_validation"] is True
    assert report["independent_final_test"] is False
    assert report["fusion_qualified"] == all(report["final_gate"].values())
```

- [ ] **Step 2: Implement final report and run verification**

The reporter rejects changed hashes, duplicate/missing canonical sample IDs,
non-finite values, incomplete classes, mismatched metrics, or an unauthorized B2
artifact. It reports old E1/D1 context without importing their predictions into
the new selection rule.

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/report_motionbert_lite_skeleton_expert.py
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_report_motionbert_lite_skeleton_expert.py
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q
D:\Anaconda\envs\PyTorch2.7\python.exe -m compileall -q src scripts tests third_party/motionbert
git diff --check
```

- [ ] **Step 3: Perform the unified Standards/Spec audit and commit**

Resolve all P1/P2 scientific-integrity findings, rerun verification, then commit
only P6-B code, config, tests, reports, and attribution. Preserve unrelated
untracked IR/Depth reports.

- [ ] **Step 4: Update the temporary handoff document**

Record final P6-B gate, artifacts, commits, failures/recoveries, and the next
authorized boundary. Do not duplicate the Spec or Plan.

---

## Plan Self-Review

### Spec coverage

- One MotionBERT-Lite candidate and no three-fold: Tasks 1 and 5-6.
- Frozen source/license/checkpoint provenance: Tasks 1, 3, and 4.
- Longest-segment T=96 projected input: Task 2.
- B0 resource and pretrained-coverage gate: Task 4.
- Frozen B1 and exact promotion gate: Task 5.
- Conditional final-two-block B2: Task 6.
- Canonical metrics, visual complementarity, hashes, and reporting: Task 7.
- No fusion/distillation implementation: global constraints and Spec non-goals.

### Placeholder scan

The plan contains no open implementation choice. Candidate, source commit,
license, weight revision/path/bytes/SHA, populations, input shape, segment rule,
model dimensions, trainable blocks, epochs, optimizers, gates, output paths, and
commands are explicit.

### Type consistency

The dataset produces `[96,17,3]`; MotionBERT-Lite consumes that exact tensor and
returns `[96,17,512]`, pooled `[512]`, and `[40]` logits. B1 caches `[512]` and
trains only the same classification head later reused by B2. B1/B2 prediction
archives share canonical sample IDs and the final reporter consumes both through
one metric contract.
