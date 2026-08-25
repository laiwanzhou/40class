# Hierarchical Multimodal Mid-Fusion Stage-1 Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate three pre-registered IR+Depth+Skeleton+raw-IMU hierarchical mid-fusion teacher candidates on the fixed 2,039-row train12 to 388-row user6/user7 development boundary without training-data leakage.

**Architecture:** Aligned IR+Depth feature maps produce persistent context and wrist segment tokens. Gap-aware Skeleton and role-aware raw IMU produce body-motion segment tokens. Forty learned action queries fuse the three semantic groups through masked cross-attention; auxiliary heads and group dropout prevent token starvation. The fixed-validation runner trains every candidate on train12, transfers train12-only normalization to a separate user6/user7 Dataset, evaluates train and validation once after epoch 15, and labels selection as development-set selection.

**Tech Stack:** Python 3.12, PyTorch 2.7, NumPy, pandas, torchvision, existing VideoMAE V2 and YOLO pose assets, pytest, YAML/JSON/NPZ evidence artifacts.

**Spec:** `docs/superpowers/specs/2026-08-25-hierarchical-multimodal-midfusion-design.md`

## Global Constraints

- Stage-1 modalities are exactly `IR`, `Depth_Color`, `Skeleton`, and raw `IMU`.
- Canonical train12 population is exactly 2,039 rows; user6/user7 population is exactly 388 rows.
- Primary metrics include all 388 rows, including the three Thermal-only rows with no Stage-1 modality.
- A visual trial may degrade to global-only only for the dedicated no-valid-YOLO-person-pose condition; all other visual contract errors remain fatal.
- Normalized segment count is exactly `8`.
- Class order is exactly `0..39`.
- The default protocol is exactly `fixed_user6_user7`; grouped CV is unauthorized.
- Grouped CV may run only after every fold fit and validation scope contains class IDs `0..39` and the user explicitly authorizes it.
- All three pre-registered candidates evaluate user6/user7 once for development selection.
- No user6/user7 row may enter gradients, normalization, sampling, epoch selection, loss design, threshold fitting, fallback fitting, or distillation-target generation.
- Final reporting sets `independent_final_test=false` and never describes user6/user7 as an untouched test set.
- Thermal and Radar models, logits, and embeddings are forbidden in this plan.
- No frame-level Skeleton/image joint alignment is introduced.
- Student distillation is a separate plan and begins only after a `full_teacher_worthy` teacher result.
- The exact student inference bundle target remains below `95,000,000` serialized bytes; Stage 1 must still estimate the deployable student proxy.
- All real experiment outputs use new run IDs and refuse overwrite.
- Use `D:\Anaconda\envs\PyTorch2.7\python.exe` for tests and experiment commands.

---

## Planned File Structure

| File | Responsibility |
|---|---|
| `configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml` | Frozen population, modalities, model dimensions, training policy, gates, paths |
| `metadata/splits/train12_grouped_3fold_midfusion.json` | Non-executable audit artifact documenting the rejected incomplete-class split |
| `src/experiments/hierarchical_midfusion_config.py` | Frozen fixed-validation contract and grouped-execution hard gate |
| `src/data/multimodal_segment_contract.py` | Segment token dataclasses and mask validation |
| `src/data/canonical_multimodal_index.py` | Canonical-union membership, availability, fallback and split validation |
| `src/data/clean_skeleton_segments.py` | Ported strict Skeleton cleaning, scale normalization, velocity and gap-aware eight-segment loading |
| `scripts/build_midfusion_skeleton_clean_views.py` | Build the train12-fit `selected_final` Skeleton view; fold scopes require explicit authorization |
| `src/data/raw_imu_segments.py` | Role-aware raw IMU loading, timestamp sorting, masks and eight-segment loading |
| `src/data/hierarchical_multimodal_dataset.py` | Trial-level join of visual, Skeleton and IMU inputs without deleting canonical rows |
| `src/models/multimodal_token_contract.py` | `GroupTokens` model-side interface |
| `src/models/structured_ir_depth_visual_encoder.py` | Aligned IR+Depth spatial fusion and persistent context/wrist segment tokens |
| `src/models/body_motion_segment_encoder.py` | Skeleton/IMU encoders and within-segment bidirectional cross-attention |
| `src/models/hierarchical_action_query_fusion.py` | Forty action queries, masks, group/segment attention, auxiliary group pooling |
| `src/models/hierarchical_multimodal_teacher.py` | End-to-end teacher and training-only auxiliary heads |
| `src/training/hierarchical_multimodal_losses.py` | CE, auxiliary CE, warm-start entropy floor and dropout weighting |
| `src/data/body_normalization_state.py` | Fit train-only Skeleton/IMU normalization and apply the same state to separate Datasets |
| `src/train_hierarchical_multimodal_teacher.py` | Fixed-candidate split train/evaluate/cache primitives |
| `scripts/run_hierarchical_multimodal_teacher.py` | Authorization-gated CLI and fixed user6/user7 candidate orchestration |
| `scripts/report_hierarchical_multimodal_teacher.py` | Recompute metrics, ablations, rescue/harm, size and provenance from saved archives |

---

## Protocol Amendment Execution Boundary

Tasks 1-10 are completed historical implementation. The grouped-CV code and
fold-specific Skeleton views created while implementing them remain testable
audit artifacts, but they are not authorized experiment entry points. Tasks
11-13 below supersede the original grouped selection and selected-final tasks.
No formal `fixed-validation` training command may run until Task 13 verification
passes and the user separately confirms the launch.

The rejected split audit is frozen as:

| Fold | Fit rows/classes | Validation rows/classes | Missing fit class | Missing validation class |
|---|---:|---:|---|---|
| 0 | 1,414 / 39 | 625 / 39 | 25 | 26 |
| 1 | 1,302 / 40 | 737 / 39 | none | 25 |
| 2 | 1,362 / 40 | 677 / 39 | none | 25 |

Class 25 occurs only three times and only for `user1`, so the stored strict
user-held-out split cannot be repaired by rearranging the same 12 validation
owners. A replacement three-fold plan is admissible only when every fit and
validation scope has classes `0..39` and the user explicitly authorizes it.

### Task 1: Freeze the Stage-1 experiment contract and train12 grouped split

**Files:**
- Create: `configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml`
- Create: `metadata/splits/train12_grouped_3fold_midfusion.json`
- Create: `src/experiments/hierarchical_midfusion_config.py`
- Create: `tests/test_hierarchical_multimodal_contract.py`

**Interfaces:**
- Produces: `load_midfusion_config(path: Path) -> dict[str, Any]`
- Produces: persisted fold objects with `fold`, `fit_user_ids`, and `validation_user_ids`

- [ ] **Step 1: Write the failing contract test**

```python
def test_midfusion_contract_freezes_population_modalities_and_folds() -> None:
    config = load_midfusion_config(CONFIG)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))

    assert config["stage"] == "P5-HMF0"
    assert config["modalities"] == ["ir", "depth_color", "skeleton", "imu"]
    assert config["segment_count"] == 8
    assert config["population"]["train_samples"] == 2039
    assert config["population"]["validation_samples"] == 388
    assert config["population"]["class_count"] == 40
    assert config["policy"]["validation_users_enter_training"] is False
    assert config["policy"]["nonselected_candidates_enter_final_evaluation"] is False
    fold_users = [user for fold in split["folds"] for user in fold["validation_user_ids"]]
    assert len(fold_users) == len(set(fold_users)) == 12
    assert set(fold_users).isdisjoint({"user6", "user7"})
```

- [ ] **Step 2: Run the test and verify the contract is absent**

Run:

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_multimodal_contract.py::test_midfusion_contract_freezes_population_modalities_and_folds
```

Expected: FAIL because the config, loader, and split do not exist.

- [ ] **Step 3: Create the frozen config and deterministic split loader**

The config must contain these exact core values:

```yaml
schema_version: 1
stage: P5-HMF0
modalities: [ir, depth_color, skeleton, imu]
segment_count: 8
class_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
population:
  manifest: metadata/manifest.csv
  development_split: metadata/splits/train12_val2_user6_user7_development.json
  grouped_split: metadata/splits/train12_grouped_3fold_midfusion.json
  train_samples: 2039
  validation_samples: 388
  train_user_ids: [user1, user2, user3, user5, user8, user9, user16, user18, user19, user20, user21, user22]
  validation_user_ids: [user6, user7]
  class_count: 40
model:
  teacher_dim: 256
  action_queries: 40
  fusion_layers: 2
  attention_heads: 8
training:
  seed: 20260715
  fixed_epochs: 15
policy:
  validation_users_enter_training: false
  validation_users_enter_normalization: false
  validation_users_enter_sampler: false
  validation_users_enter_selection: false
  nonselected_candidates_enter_final_evaluation: false
  thermal_allowed: false
  radar_allowed: false
```

Create the grouped split from the already approved three train12 fold groups:

```json
{
  "schema_version": 1,
  "seed": 20260715,
  "folds": [
    {"fold": 0, "validation_user_ids": ["user1", "user18", "user21", "user5"]},
    {"fold": 1, "validation_user_ids": ["user16", "user19", "user22", "user8"]},
    {"fold": 2, "validation_user_ids": ["user2", "user20", "user3", "user9"]}
  ]
}
```

The loader derives each fold's `fit_user_ids` as train12 minus validation users
and rejects overlap, missing users, repeated ownership, changed counts, Thermal,
Radar, or class maps other than 0..39.

- [ ] **Step 4: Run the complete contract test file**

Run:

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_multimodal_contract.py
```

Expected: PASS.

- [ ] **Step 5: Commit the contract**

```powershell
git add configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml metadata/splits/train12_grouped_3fold_midfusion.json src/experiments/hierarchical_midfusion_config.py tests/test_hierarchical_multimodal_contract.py docs/superpowers/plans/2026-08-25-hierarchical-multimodal-midfusion-stage1.md
git commit -m "experiment: freeze hierarchical midfusion contract"
```

---

### Task 2: Implement canonical-union and segment interfaces

**Files:**
- Create: `src/data/multimodal_segment_contract.py`
- Create: `src/data/canonical_multimodal_index.py`
- Create: `tests/test_canonical_multimodal_index.py`

**Interfaces:**
- Produces: `SegmentBatch(tokens, token_mask, quality, quality_mask)`
- Produces: `CanonicalTrial(sample_id, user_id, class_id, paths, availability)`
- Produces: `build_canonical_trials(...) -> list[CanonicalTrial]`
- Produces: `normalized_segment_bounds(length: int, segments: int = 8) -> np.ndarray`

- [ ] **Step 1: Write failing tests for masks, bounds, and population preservation**

```python
def test_normalized_segment_bounds_cover_complete_sequence() -> None:
    bounds = normalized_segment_bounds(17, segments=8)
    assert bounds.tolist() == [[0, 2], [2, 4], [4, 6], [6, 8], [8, 11], [11, 13], [13, 15], [15, 17]]
    assert bounds[0, 0] == 0 and bounds[-1, 1] == 17


def test_canonical_index_retains_thermal_only_rows_as_core_unavailable() -> None:
    rows = build_canonical_trials(MANIFEST, DEVELOPMENT_SPLIT, partition="validation")
    assert len(rows) == 388
    assert sum(row.core_available for row in rows) == 385
    assert sum(not row.core_available for row in rows) == 3
    assert {row.class_id for row in rows} == set(range(40))
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_canonical_multimodal_index.py
```

Expected: FAIL on missing interfaces.

- [ ] **Step 3: Implement immutable contracts and strict joins**

Use these public types:

```python
@dataclass(frozen=True)
class CanonicalTrial:
    sample_id: str
    user_id: str
    class_id: int
    paths: Mapping[str, Path | None]
    availability: Mapping[str, bool]

    @property
    def core_available(self) -> bool:
        return any(self.availability[name] for name in ("ir", "depth_color", "skeleton", "imu"))


@dataclass(frozen=True)
class SegmentBatch:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    quality: torch.Tensor
    quality_mask: torch.Tensor

    def validate(self, *, batch: int, segments: int, streams: int, dim: int) -> None:
        if self.tokens.shape != (batch, segments, streams, dim):
            raise ValueError("segment token shape changed")
        if self.token_mask.shape != (batch, segments, streams) or self.token_mask.dtype != torch.bool:
            raise ValueError("segment token mask changed")
```

`build_canonical_trials` left-joins modality paths to every canonical manifest
row and never filters on modality presence. It validates exact train/validation
users and counts before returning.

- [ ] **Step 4: Run unit and real-population tests**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_canonical_multimodal_index.py
```

Expected: PASS with 2,039/388 counts.

- [ ] **Step 5: Commit canonical indexing**

```powershell
git add src/data/multimodal_segment_contract.py src/data/canonical_multimodal_index.py tests/test_canonical_multimodal_index.py
git commit -m "feat: add canonical multimodal segment contract"
```

---

### Task 3: Port strict Skeleton preprocessing to eight segments

**Files:**
- Create: `src/data/clean_skeleton_segments.py`
- Create: `scripts/build_midfusion_skeleton_clean_views.py`
- Create: `tests/test_clean_skeleton_segments.py`

**Interfaces:**
- Consumes: fold-specific strict clean-view CSVs generated by the Skeleton branch
- Produces: `load_skeleton_segments(trial, clean_rows, data_root) -> SkeletonSegments`
- Produces: `SkeletonSegments(features [8,17,6], mask [8], quality [8,4])`
- Produces: `outputs/midfusion_skeleton_clean_views/fold_{k}/clean_view.csv` and `provenance.json`
- Produces: `outputs/midfusion_skeleton_clean_views/selected_final/clean_view.csv` fitted on train12 only
- Produces: `reports/midfusion_skeleton_clean_views.json` with fold ownership and hashes but no frame rows

- [ ] **Step 1: Write tests for gap isolation and retained velocity**

```python
def test_skeleton_segment_loader_never_interpolates_across_gap() -> None:
    frames = np.asarray([0, 1, 10, 11])
    segment_ids = np.asarray([0, 0, 1, 1])
    poses = synthetic_h36m_poses(frames)
    result = resample_skeleton_segments(frames, segment_ids, poses, segment_count=8)
    assert not result.mask[3:5].any()
    assert np.count_nonzero(result.features[~result.mask]) == 0


def test_skeleton_features_are_xyz_plus_segment_local_velocity() -> None:
    result = resample_skeleton_segments(FRAMES, SEGMENTS, POSES, segment_count=8)
    assert result.features.shape == (8, 17, 6)
    first_valid = np.flatnonzero(result.mask)[0]
    assert np.allclose(result.features[first_valid, :, 3:], 0.0)


def test_clean_view_projection_never_fits_fold_validation_users(tmp_path: Path) -> None:
    reports = build_midfusion_clean_views(CONFIG, output_root=tmp_path)
    for report in reports:
        assert set(report["projection_fit_user_ids"]).isdisjoint(
            report["scope_validation_user_ids"]
        )
        assert set(report["projection_fit_user_ids"]) == set(report["fold_fit_user_ids"])
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_clean_skeleton_segments.py
```

Expected: FAIL because the loader is absent.

- [ ] **Step 3: Port and narrow the accepted Skeleton logic**

Port these behaviors from branch `skeleton_raw_data` at commit `bdd5d73`:

- `H36M_EDGES`;
- median bone-scale normalization;
- segment-local velocity;
- gap-aware interpolation restricted to one retained segment;
- strict candidate identity from the fold-specific clean view.

Port the projection/candidate-selection seam from
`scripts/build_strict_oof_skeleton_clean_views.py`, but bind it to
`metadata/splits/train12_grouped_3fold_midfusion.json`. For each fold, fit the
projection on `fit_user_ids`, build one clean view over fit plus fold-validation
users, and persist provenance whose fold-0 ownership satisfies:

```python
assert report["fold"] == 0
assert report["projection_fit_user_ids"] == [
    "user16", "user19", "user2", "user20", "user22", "user3", "user8", "user9"
]
assert report["scope_validation_user_ids"] == ["user1", "user18", "user21", "user5"]
assert report["fit_validation_overlap"] == []
assert np.asarray(report["projection_matrix"]).shape == (3, 2)
assert report["calibration_correspondences"] >= 100
assert re.fullmatch(r"[0-9a-f]{64}", report["clean_view_sha256"])
```

The implementation rejects fewer than 100 calibration correspondences,
duplicate sample/frame keys, user overlap, or a clean view whose validation-user
set differs from the grouped split.

Do not port Skeleton classifier code or its final probabilities. Collapse the
accepted 64-step view into eight bins while preserving an invalid bin whenever
no retained Skeleton segment overlaps that normalized interval.

Use the exact result type:

```python
@dataclass(frozen=True)
class SkeletonSegments:
    features: torch.Tensor  # [8,17,6]
    mask: torch.Tensor      # bool [8]
    quality: torch.Tensor   # [8,4]: coverage, retained-count, gap flag, scale stability
```

- [ ] **Step 4: Run synthetic and two-real-trial tests**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_clean_skeleton_segments.py
```

Expected: PASS; no cross-gap interpolation.

- [ ] **Step 5: Commit Skeleton segment loading**

```powershell
git add src/data/clean_skeleton_segments.py scripts/build_midfusion_skeleton_clean_views.py tests/test_clean_skeleton_segments.py
git commit -m "feat: add strict skeleton segment loader"
```

---

### Task 4: Implement role-aware raw IMU segments

**Files:**
- Create: `src/data/raw_imu_segments.py`
- Create: `tests/test_raw_imu_segments.py`

**Interfaces:**
- Produces: `load_raw_imu_segments(trial_path: Path, segment_count: int = 8) -> IMUSegments`
- Produces: `IMUSegments(features [8,5,16], role_mask [8,5], quality [8,5,3])`

- [ ] **Step 1: Write failing tests for timestamp ordering and missing roles**

```python
def test_imu_loader_preserves_device_roles_and_masks_missing_roles(tmp_path: Path) -> None:
    write_imu_csv(tmp_path / "a.csv", role="WTRA", times=[3, 1, 2])
    write_imu_csv(tmp_path / "b.csv", role="WTLL", times=[1, 2, 3])
    result = load_raw_imu_segments(tmp_path, segment_count=8)
    assert result.features.shape == (8, 5, 16)
    assert result.role_mask[:, 0].all()
    assert result.role_mask[:, 4].all()
    assert not result.role_mask[:, 1:4].any()
    assert torch.count_nonzero(result.features[:, 1:4]) == 0


def test_imu_segment_values_retain_temporal_direction(tmp_path: Path) -> None:
    write_ramp_imu_trial(tmp_path)
    result = load_raw_imu_segments(tmp_path, segment_count=8)
    assert result.features[0, 0, 0] < result.features[-1, 0, 0]
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_raw_imu_segments.py
```

Expected: FAIL on missing loader.

- [ ] **Step 3: Implement the raw role-aware loader**

Reuse the accepted constants from `src/data/imu_dataset.py`:

```python
DEVICE_ROLES = ("WTRA", "WTLA", "WTC", "WTRL", "WTLL")
SENSOR_COLUMNS = (...16 accepted columns...)
```

Parse timestamps, stable-sort each role, partition each role's observed time
range into eight normalized intervals, and compute the interval mean. Do not use
the 2,310 RF summary features. Quality channels are:

```text
observed_count, timestamp_span_fraction, finite_fraction
```

Normalization is a separate `fit_imu_normalization(dataset, fit_indices)` call
and must ignore masked role/segment entries.

- [ ] **Step 4: Run IMU loader tests**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_raw_imu_segments.py
```

Expected: PASS.

- [ ] **Step 5: Commit raw IMU segments**

```powershell
git add src/data/raw_imu_segments.py tests/test_raw_imu_segments.py
git commit -m "feat: add raw role-aware imu segments"
```

---

### Task 5: Build the canonical multimodal dataset and fallback behavior

**Files:**
- Create: `src/data/hierarchical_multimodal_dataset.py`
- Create: `tests/test_hierarchical_multimodal_dataset.py`

**Interfaces:**
- Consumes: `CanonicalTrial`, visual ROI assets, `SkeletonSegments`, `IMUSegments`
- Produces: `HierarchicalMultimodalDataset.__getitem__() -> dict[str, object]`

- [ ] **Step 1: Write failing tests for complete, missing-IMU, and Thermal-only rows**

```python
def test_dataset_keeps_natural_missing_patterns() -> None:
    dataset = build_fixture_dataset(patterns=["complete", "missing_imu", "thermal_only"])
    complete, missing_imu, thermal_only = dataset[0], dataset[1], dataset[2]
    assert complete["availability"].tolist() == [True, True, True, True]
    assert missing_imu["availability"].tolist() == [True, True, True, False]
    assert thermal_only["availability"].tolist() == [False, False, False, False]
    assert thermal_only["core_available"].item() is False
    assert len(dataset) == 3


def test_real_population_counts_are_not_filtered() -> None:
    train = make_dataset(CONFIG, partition="train", decode_visual=False)
    validation = make_dataset(CONFIG, partition="validation", decode_visual=False)
    assert len(train) == 2039
    assert len(validation) == 388
    assert sum(bool(validation[i]["core_available"]) for i in range(len(validation))) == 385
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_multimodal_dataset.py
```

Expected: FAIL on missing dataset.

- [ ] **Step 3: Implement a join-only metadata mode and decoded mode**

The dataset returns exactly:

```python
{
    "visual": visual_views,                 # [2,4,3,16,H,W] or empty masked tensor
    "visual_segment_indices": indices,      # [8,2]
    "skeleton": skeleton.features,          # [8,17,6]
    "skeleton_mask": skeleton.mask,         # [8]
    "imu": imu.features,                    # [8,5,16]
    "imu_role_mask": imu.role_mask,         # [8,5]
    "quality": quality,                     # modality-native quality bundle
    "availability": torch.bool [4],         # ir, depth, skeleton, imu
    "core_available": torch.bool [],
    "sample_id": str,
    "user_id": str,
    "label": int,
}
```

Use current P0 ROI configuration for visual crops. The metadata-only mode must
not open images or sensor files and exists for population/audit tests.

- [ ] **Step 4: Run dataset tests and a two-real-sample decode smoke**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_multimodal_dataset.py
```

Expected: PASS and finite real tensors.

- [ ] **Step 5: Commit the joined dataset**

```powershell
git add src/data/hierarchical_multimodal_dataset.py tests/test_hierarchical_multimodal_dataset.py
git commit -m "feat: join canonical multimodal trials"
```

---

### Task 6: Implement persistent context/wrist IR+Depth visual tokens

**Files:**
- Create: `src/models/multimodal_token_contract.py`
- Create: `src/models/structured_ir_depth_visual_encoder.py`
- Create: `tests/test_structured_ir_depth_visual_encoder.py`

**Interfaces:**
- Produces: `GroupTokens(tokens [B,8,S,D], mask [B,8,S], quality, quality_mask)`
- Produces: `StructuredIRDepthVisualEncoder.forward(...) -> GroupTokens` with `S=2`

- [ ] **Step 1: Write failing tiny-backbone tests**

```python
def test_visual_encoder_keeps_context_and_wrist_in_training_graph() -> None:
    model = StructuredIRDepthVisualEncoder(backbone=TinyTemporalBackbone(dim=16), output_dim=32)
    result = model(ir=IR, depth=DEPTH, availability=torch.ones(2, 2, 4, dtype=torch.bool))
    assert result.tokens.shape == (2, 8, 2, 32)
    loss = result.tokens[:, :, 0].sum() + result.tokens[:, :, 1].sum()
    loss.backward()
    assert model.context_router.weight.grad.abs().sum() > 0
    assert model.wrist_router.weight.grad.abs().sum() > 0


def test_zero_depth_adapter_recovers_ir_tokens_exactly() -> None:
    model = StructuredIRDepthVisualEncoder(backbone=TinyTemporalBackbone(dim=16), output_dim=32)
    first = model(ir=IR, depth=DEPTH_A, availability=ALL_AVAILABLE)
    second = model(ir=IR, depth=DEPTH_B, availability=ALL_AVAILABLE)
    assert torch.equal(first.tokens, second.tokens)
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_structured_ir_depth_visual_encoder.py
```

Expected: FAIL on missing model.

- [ ] **Step 3: Implement the visual group interface**

Use:

```python
@dataclass
class GroupTokens:
    tokens: torch.Tensor
    mask: torch.Tensor
    quality: torch.Tensor
    quality_mask: torch.Tensor
```

The encoder produces per-view, per-segment features; adds a zero-initialized
Depth residual to IR; computes a two-way softmax inside `(global, person)` and
inside `(left, right)`; and returns both subgroup tokens. Availability is applied
before each subgroup softmax. If one subgroup member is missing, the remaining
member receives weight one. If both are missing, the subgroup mask is false.

- [ ] **Step 4: Run visual unit tests and a real one-batch smoke**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_structured_ir_depth_visual_encoder.py
```

Expected: PASS, finite `[B,8,2,D]` output, nonzero gradients for both subgroups.

- [ ] **Step 5: Commit visual token encoding**

```powershell
git add src/models/multimodal_token_contract.py src/models/structured_ir_depth_visual_encoder.py tests/test_structured_ir_depth_visual_encoder.py
git commit -m "feat: add structured ir depth visual tokens"
```

---

### Task 7: Implement Skeleton/IMU body-motion tokens

**Files:**
- Create: `src/models/body_motion_segment_encoder.py`
- Create: `tests/test_body_motion_segment_encoder.py`

**Interfaces:**
- Consumes: Skeleton `[B,8,17,6]`, IMU `[B,8,5,16]`, masks and quality
- Produces: `GroupTokens` with `tokens [B,8,1,256]`

- [ ] **Step 1: Write failing modality-presence and interaction tests**

```python
def test_body_encoder_supports_skeleton_only_imu_only_and_both() -> None:
    model = BodyMotionSegmentEncoder(output_dim=32, heads=4)
    both = model(SKELETON, IMU, SKELETON_MASK, IMU_MASK)
    skeleton_only = model(SKELETON, IMU, SKELETON_MASK, torch.zeros_like(IMU_MASK))
    imu_only = model(SKELETON, IMU, torch.zeros_like(SKELETON_MASK), IMU_MASK)
    assert both.tokens.shape == skeleton_only.tokens.shape == imu_only.tokens.shape == (2, 8, 1, 32)
    assert both.mask.all() and skeleton_only.mask.all() and imu_only.mask.all()


def test_unavailable_body_tokens_receive_no_attention() -> None:
    result = BodyMotionSegmentEncoder(output_dim=32)(ZERO_SKELETON, ZERO_IMU, FALSE_SKEL, FALSE_IMU)
    assert not result.mask.any()
    assert torch.count_nonzero(result.tokens) == 0
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_body_motion_segment_encoder.py
```

Expected: FAIL on missing encoder.

- [ ] **Step 3: Implement modality-native encoders and one cross-attention block**

Skeleton path:

```text
Linear(102,128) -> LayerNorm -> GELU -> depthwise temporal Conv1d -> 256
```

IMU path:

```text
role-shared Linear(16,64) -> masked role attention -> temporal Conv1d -> 256
```

When both paths are present, apply bidirectional cross-attention inside each of
the eight normalized segments and average the two residual outputs. When one is
absent, return the other without manufacturing an absent token.

- [ ] **Step 4: Run body-motion tests**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_body_motion_segment_encoder.py
```

Expected: PASS.

- [ ] **Step 5: Commit body-motion encoding**

```powershell
git add src/models/body_motion_segment_encoder.py tests/test_body_motion_segment_encoder.py
git commit -m "feat: add skeleton imu body motion tokens"
```

---

### Task 8: Implement masked action-query hierarchical fusion

**Files:**
- Create: `src/models/hierarchical_action_query_fusion.py`
- Create: `tests/test_hierarchical_action_query_fusion.py`

**Interfaces:**
- Consumes: visual `GroupTokens(S=2)`, body `GroupTokens(S=1)`
- Produces: logits, action features, group attention, segment attention

- [ ] **Step 1: Write failing mask, identity, and gradient tests**

```python
def test_action_queries_never_attend_to_masked_group() -> None:
    model = HierarchicalActionQueryFusion(dim=32, classes=40, heads=4, layers=2)
    output = model(visual=VISUAL, body=MASKED_BODY)
    assert torch.count_nonzero(output["group_attention"][:, :, 2]) == 0
    assert output["logits"].shape == (2, 40)


def test_all_available_groups_receive_gradient_during_warm_start() -> None:
    model = HierarchicalActionQueryFusion(dim=32, classes=40, heads=4, layers=2)
    output = model(visual=VISUAL_REQUIRES_GRAD, body=BODY_REQUIRES_GRAD)
    output["logits"].sum().backward()
    assert VISUAL_REQUIRES_GRAD.tokens.grad[:, :, 0].abs().sum() > 0
    assert VISUAL_REQUIRES_GRAD.tokens.grad[:, :, 1].abs().sum() > 0
    assert BODY_REQUIRES_GRAD.tokens.grad.abs().sum() > 0
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_action_query_fusion.py
```

Expected: FAIL on missing fusion module.

- [ ] **Step 3: Implement two masked cross-attention layers**

Flatten tokens in deterministic order:

```text
segment0/context, segment0/wrist, segment0/body, ..., segment7/body
```

Add learned group embeddings and sinusoidal normalized-segment positions. Use
40 learned queries `[40,D]`. Attention logits for masked tokens are set to the
dtype minimum before softmax. Return attention aggregated by group and segment
for audit; do not recompute it from labels.

The class logits interface is:

```python
logits = torch.einsum("bcd,cd->bc", action_features, classifier_weight) + classifier_bias
```

- [ ] **Step 4: Run fusion tests including all natural availability subsets**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_action_query_fusion.py
```

Expected: PASS; every unavailable group has zero attention.

- [ ] **Step 5: Commit hierarchical fusion**

```powershell
git add src/models/hierarchical_action_query_fusion.py tests/test_hierarchical_action_query_fusion.py
git commit -m "feat: add masked action query fusion"
```

---

### Task 9: Assemble the teacher and anti-collapse training loss

**Files:**
- Create: `src/models/hierarchical_multimodal_teacher.py`
- Create: `src/training/hierarchical_multimodal_losses.py`
- Create: `tests/test_hierarchical_multimodal_teacher.py`

**Interfaces:**
- Produces: `HierarchicalMultimodalTeacher.forward(batch, dropout_policy) -> dict[str, Tensor]`
- Produces: `hierarchical_teacher_loss(output, labels, epoch, natural_pattern) -> dict[str, Tensor]`

- [ ] **Step 1: Write failing tests for auxiliary gradients and dropout safety**

```python
def test_teacher_auxiliary_heads_prevent_group_starvation() -> None:
    model = tiny_teacher()
    output = model(COMPLETE_BATCH, dropout_policy=GroupDropout.disabled())
    losses = hierarchical_teacher_loss(output, LABELS, epoch=1, natural_pattern=True)
    losses["loss"].backward()
    assert model.visual_encoder.context_router.weight.grad.abs().sum() > 0
    assert model.visual_encoder.wrist_router.weight.grad.abs().sum() > 0
    assert model.body_encoder.skeleton_projection.weight.grad.abs().sum() > 0
    assert model.body_encoder.imu_projection.weight.grad.abs().sum() > 0


def test_group_dropout_never_removes_every_usable_group() -> None:
    policy = GroupDropout(context=1.0, wrist=1.0, body=1.0)
    output = tiny_teacher()(COMPLETE_BATCH, dropout_policy=policy)
    assert output["effective_group_mask"].any(dim=1).all()
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_multimodal_teacher.py
```

Expected: FAIL on missing teacher/loss.

- [ ] **Step 3: Implement teacher outputs and exact first-run loss weights**

The output keys are:

```text
logits, context_logits, wrist_logits, body_logits,
action_features, group_attention, segment_attention,
effective_group_mask, core_available
```

The first registered teacher loss is:

```text
1.00 fused CE
0.15 context CE
0.15 wrist CE
0.20 body CE
0.02 warm-start group entropy floor for epochs 1-2 only
```

Synthetic-dropout examples multiply the complete loss by `0.5`; natural
missing-pattern examples retain weight `1.0`. Rows with `core_available=False`
are excluded from neural CE and handled by the class-prior fallback in metrics.

- [ ] **Step 4: Run teacher tests and finite-gradient smoke**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_multimodal_teacher.py
```

Expected: PASS, finite gradients for every available group.

- [ ] **Step 5: Commit teacher/loss assembly**

```powershell
git add src/models/hierarchical_multimodal_teacher.py src/training/hierarchical_multimodal_losses.py tests/test_hierarchical_multimodal_teacher.py
git commit -m "feat: assemble hierarchical multimodal teacher"
```

---

### Task 10: Add deterministic real-data smoke and resource gates

**Files:**
- Create: `src/train_hierarchical_multimodal_teacher.py`
- Create: `scripts/run_hierarchical_multimodal_teacher.py`
- Create: `tests/test_run_hierarchical_multimodal_teacher.py`

**Interfaces:**
- Produces: `run_smoke(config_path: Path) -> dict[str, Any]`
- Produces: atomic `resolved_config.yaml`, `smoke_report.json`, and prediction NPZ

- [ ] **Step 1: Write failing smoke-contract tests**

```python
def test_smoke_uses_train_users_only_and_updates_every_group(tmp_path: Path) -> None:
    report = run_smoke(CONFIG, output_root=tmp_path)
    assert report["sample_users_entered_gradient"]
    assert set(report["sample_users_entered_gradient"]).isdisjoint({"user6", "user7"})
    assert report["finite_gradients"] is True
    assert report["changed_parameter_groups"] == ["visual", "skeleton", "imu", "fusion"]
    assert report["peak_cuda_mib"] < 8151
```

- [ ] **Step 2: Run the test to verify failure**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_run_hierarchical_multimodal_teacher.py
```

Expected: FAIL on missing runner.

- [ ] **Step 3: Implement a two-step real smoke**

The smoke must:

- verify official VideoMAE checkpoint SHA-256;
- load two complete train12 rows and one natural missing-IMU row;
- execute forward/backward with bfloat16 autocast;
- verify finite logits, losses, and gradients;
- prove all four parameter groups changed after one optimizer step;
- run one body-only synthetic-mask forward;
- run one no-core fallback forward;
- record peak CUDA memory and exact parameter/state bytes;
- refuse an existing output directory.

- [ ] **Step 4: Run the real smoke and its test**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/run_hierarchical_multimodal_teacher.py --smoke-test
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_run_hierarchical_multimodal_teacher.py
```

Expected: `status=smoke_passed`; no user6/user7 gradient access.

- [ ] **Step 5: Commit the smoke-qualified runner**

```powershell
git add src/train_hierarchical_multimodal_teacher.py scripts/run_hierarchical_multimodal_teacher.py tests/test_run_hierarchical_multimodal_teacher.py
git commit -m "experiment: qualify hierarchical teacher runtime"
```

---

### Task 11: Replace grouped execution with an authorization-gated fixed protocol

**Files:**
- Modify: `configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml`
- Modify: `src/experiments/hierarchical_midfusion_config.py`
- Modify: `scripts/build_midfusion_skeleton_clean_views.py`
- Modify: `scripts/run_hierarchical_multimodal_teacher.py`
- Modify: `tests/test_hierarchical_multimodal_contract.py`
- Modify: `tests/test_clean_skeleton_segments.py`

**Interfaces:**
- Produces: `assert_grouped_cv_authorized(config: dict[str, Any]) -> None`
- Produces: `evaluation_protocol=fixed_user6_user7`
- Consumes: the existing `selected_final` Skeleton clean view

- [ ] **Step 1: Write failing fixed-protocol and grouped-denial tests**

```python
def test_midfusion_contract_defaults_to_fixed_user6_user7() -> None:
    config = load_midfusion_config(CONFIG)
    assert config["evaluation_protocol"] == "fixed_user6_user7"
    assert config["policy"]["grouped_cv_authorized"] is False
    assert config["policy"]["validation_users_enter_selection"] is True
    assert "grouped_folds" not in config


def test_grouped_cv_requires_complete_classes_and_explicit_authorization() -> None:
    config = load_midfusion_config(CONFIG)
    with pytest.raises(PermissionError, match="explicit authorization"):
        assert_grouped_cv_authorized(config)
```

- [ ] **Step 2: Run the contract tests and verify RED**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_multimodal_contract.py
```

Expected: FAIL because the current loader requires grouped folds and rejects
validation-user candidate selection.

- [ ] **Step 3: Implement the protocol and hard gate**

Use these exact config values:

```yaml
evaluation_protocol: fixed_user6_user7
policy:
  validation_users_enter_training: false
  validation_users_enter_normalization: false
  validation_users_enter_sampler: false
  validation_users_enter_selection: true
  grouped_cv_authorized: false
```

The default loader does not expose grouped folds. The authorization gate rejects
the stored split because audited class counts are `fit=[39,40,40]` and
`validation=[39,39,39]`, even if a caller changes only the Boolean. Skeleton
scope generation returns only `selected_final`. The CLI must call the grouped
gate before creating an output directory, Dataset, checkpoint, or CUDA model.

- [ ] **Step 4: Verify the protocol and Skeleton scopes**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_multimodal_contract.py tests/test_clean_skeleton_segments.py
```

Expected: PASS; existing `selected_final` artifacts remain valid.

- [ ] **Step 5: Commit the protocol hard gate**

```powershell
git add configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml src/experiments/hierarchical_midfusion_config.py scripts/build_midfusion_skeleton_clean_views.py scripts/run_hierarchical_multimodal_teacher.py tests/test_hierarchical_multimodal_contract.py tests/test_clean_skeleton_segments.py
git commit -m "experiment: fix user6 user7 validation protocol"
```

---

### Task 12: Transfer train-only normalization across separate Datasets

**Files:**
- Create: `src/data/body_normalization_state.py`
- Modify: `src/train_hierarchical_multimodal_teacher.py`
- Create: `tests/test_fixed_validation_training.py`
- Modify: `tests/test_hierarchical_multimodal_grouped_cv.py`

**Interfaces:**
- Produces: immutable `BodyNormalizationState`
- Produces: `fit_body_normalization_state(dataset, indices) -> BodyNormalizationState`
- Produces: `apply_body_normalization_state(dataset, state) -> None`
- Produces: `train_candidate_split(*, config: dict[str, Any], candidate: str, train_dataset: Dataset[dict[str, object]], validation_dataset: Dataset[dict[str, object]], run_dir: Path, model_factory: ModelFactory | None = None, device: torch.device | None = None) -> dict[str, Any]`
- Preserves: the existing `train_candidate_fold` signature as an audit wrapper

- [ ] **Step 1: Write the failing cross-Dataset normalization test**

```python
def test_normalization_state_fits_train_and_applies_to_validation() -> None:
    train, validation = distinct_tiny_datasets(train_value=2.0, validation_value=100.0)
    state = fit_body_normalization_state(train, np.arange(len(train)))
    apply_body_normalization_state(train, state)
    apply_body_normalization_state(validation, state)
    assert set(state.fit_user_ids).isdisjoint({"user6", "user7"})
    assert np.allclose(train.imu_loader.normalization[0], 2.0)
    assert np.allclose(validation.imu_loader.normalization[0], 2.0)
```

- [ ] **Step 2: Run the test and verify RED**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_fixed_validation_training.py::test_normalization_state_fits_train_and_applies_to_validation
```

Expected: FAIL because normalization arrays are private to one Dataset loader.

- [ ] **Step 3: Implement immutable normalization state**

```python
@dataclass(frozen=True)
class BodyNormalizationState:
    skeleton_mean: np.ndarray
    skeleton_std: np.ndarray
    imu_mean: np.ndarray
    imu_std: np.ndarray
    fit_sample_ids: tuple[str, ...]
    fit_user_ids: tuple[str, ...]

```

`body_normalization_provenance(state: BodyNormalizationState) -> dict[str, Any]`
returns SHA-256 values for all four arrays plus the exact fit users and samples.

Fit only supplied train indices. Apply the same arrays to train and validation
without reading validation values. Persist arrays in NPZ and their hashes in
JSON so resume applies an identical state.

- [ ] **Step 4: Write and verify RED for split training**

```python
def test_candidate_split_isolates_user67_and_evaluates_each_scope_once(tmp_path: Path) -> None:
    result = train_candidate_split(
        config=TINY_CONFIG,
        candidate="visual_skeleton_imu",
        train_dataset=TRAIN_DATASET,
        validation_dataset=USER67_DATASET,
        run_dir=tmp_path / "candidate",
        model_factory=tiny_model,
        device=torch.device("cpu"),
    )
    assert result["validation_user_ids"] == ["user6", "user7"]
    assert result["train_evaluation_count"] == 1
    assert result["validation_evaluation_count"] == 1
    assert set(result["fit_sample_ids"]).isdisjoint(result["validation_sample_ids"])
```

Run the file and expect failure because only a same-Dataset fold primitive
exists.

- [ ] **Step 5: Implement and verify the generic split primitive**

Extract the current loop without duplicating optimizer behavior. It samples only
train data, fits the class prior from train labels, excludes no-core rows from
gradients, evaluates dropout-disabled train and validation once after epoch 15,
and writes separate prediction archives. Resume verifies candidate, protocol,
fit/validation users, sample hashes, config hash, normalization hash, optimizer,
checkpoint, and RNG state.

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_fixed_validation_training.py tests/test_hierarchical_multimodal_grouped_cv.py
```

- [ ] **Step 6: Commit normalization and split training**

```powershell
git add src/data/body_normalization_state.py src/train_hierarchical_multimodal_teacher.py tests/test_fixed_validation_training.py tests/test_hierarchical_multimodal_grouped_cv.py
git commit -m "feat: add fixed validation candidate training"
```

---

### Task 13: Orchestrate three candidates and freeze evidence before authorization

**Files:**
- Modify: `configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml`
- Modify: `src/train_hierarchical_multimodal_teacher.py`
- Modify: `scripts/run_hierarchical_multimodal_teacher.py`
- Create: `scripts/report_hierarchical_multimodal_teacher.py`
- Create: `tests/test_fixed_validation_runner.py`
- Create: `tests/test_report_hierarchical_multimodal_teacher.py`

**Interfaces:**
- Produces: `run_fixed_validation(config_path: Path, output_root: Path | None = None, dataset_factory: FixedDatasetFactory | None = None, model_factory: ModelFactory | None = None, device: torch.device | None = None) -> dict[str, Any]`
- Produces: `build_fixed_validation_report(run_report: Path) -> dict[str, Any]`
- Produces: three train archives, three 388-row validation archives, and one atomic report

- [ ] **Step 1: Write failing orchestration and archive tests**

```python
def test_fixed_runner_uses_exact_populations_and_three_candidates(tmp_path: Path) -> None:
    report = run_tiny_fixed_validation(output_root=tmp_path)
    assert report["evaluation_protocol"] == "fixed_user6_user7"
    assert report["train_population_samples"] == 2039
    assert report["validation_population_samples"] == 388
    assert list(report["candidate_results"]) == list(CANDIDATE_MODALITIES)
    assert report["validation_users_entered_training"] is False
    assert report["independent_final_test"] is False


def test_validation_archive_keeps_diagnostic_evidence() -> None:
    archive = np.load(VALIDATION_ARCHIVE)
    assert archive["logits"].shape == (388, 40)
    assert archive["group_attention"].shape == (388, 40, 3)
    assert archive["segment_attention"].shape == (388, 40, 8)
    assert archive["context_logits"].shape == (388, 40)
    assert archive["wrist_logits"].shape == (388, 40)
    assert archive["body_logits"].shape == (388, 40)
    assert len(np.unique(archive["sample_ids"])) == 388
    assert set(archive["labels"]) == set(range(40))
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_fixed_validation_runner.py tests/test_report_hierarchical_multimodal_teacher.py
```

Expected: FAIL because the runner and evidence report do not exist.

- [ ] **Step 3: Implement fixed-validation orchestration**

Build separate train and validation Datasets with the existing
`selected_final/clean_view.csv`, using `training=True` only for train. Fit
normalization once on train12 and apply it to both. Train candidates in frozen
order and select by Accuracy, Macro-F1, worst-user Accuracy, negative NLL, then
fixed order.

Both train and validation archives save logits, labels, sample/user IDs, core
availability, effective group masks, group/segment attention, auxiliary group
logits, source availability, quality, and masks. The atomic report saves each
candidate's train/validation metrics and generalization gap, comparison to
`visual_only`, all hashes, `development_validation=true`, and
`independent_final_test=false`.

Formal paths are new and overwrite-protected:

```text
outputs/hierarchical_multimodal_midfusion_stage1/fixed_user6_user7/
reports/hierarchical_multimodal_teacher_fixed_user6_user7.json
reports/hierarchical_multimodal_teacher_fixed_user6_user7.md
```

- [ ] **Step 4: Implement metric recomputation and verify tests**

The report helper rejects duplicate or changed sample IDs, incomplete class
coverage, non-finite logits, metric mismatches, and hash mismatches. It assigns
the Spec Section 10 category and authorizes student planning only for
`full_teacher_worthy` or `teacher_target_reached`.

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_fixed_validation_runner.py tests/test_report_hierarchical_multimodal_teacher.py
```

- [ ] **Step 5: Run and commit the pre-training verification gate**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q
D:\Anaconda\envs\PyTorch2.7\python.exe -m compileall -q src scripts tests
git diff --check
```

Verify that no fixed-validation process is running and formal output/report
paths do not exist. Commit code, config, tests, Spec/Plan, and pre-training audit
evidence only.

- [ ] **Step 6: Stop for explicit training authorization**

Do not run this command yet:

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/run_hierarchical_multimodal_teacher.py --mode fixed-validation
```

Report the verified state, expected `41-44` GPU-hour envelope, and exact output
paths. Ask the user to authorize the formal launch. Status or audit requests are
not authorization.

---

### Historical Task 11: Grouped candidate selection (superseded; do not execute)

**Files:**
- Modify: `src/train_hierarchical_multimodal_teacher.py`
- Modify: `scripts/run_hierarchical_multimodal_teacher.py`
- Create: `tests/test_hierarchical_multimodal_grouped_cv.py`

**Interfaces:**
- Produces: `run_grouped_cv(config) -> GroupedCVResult`
- Produces: pooled predictions for the fixed candidates `visual_only`, `visual_skeleton`, `visual_imu`, `visual_skeleton_imu`

- [ ] **Step 1: Write failing selected-only and fold-ownership tests**

```python
def test_grouped_cv_predictions_are_user_held_out() -> None:
    result = run_tiny_grouped_cv(SYNTHETIC_CACHE)
    for row in result.prediction_rows:
        assert row.user_id not in row.fit_user_ids
    assert len({row.sample_id for row in result.prediction_rows}) == len(result.prediction_rows)


def test_final_evaluation_contract_returns_only_selected_candidate() -> None:
    assert final_evaluation_candidates("visual_skeleton_imu") == ("visual_skeleton_imu",)
```

- [ ] **Step 2: Run tests to verify failure**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_multimodal_grouped_cv.py
```

Expected: FAIL on missing grouped runner.

- [ ] **Step 3: Implement the four pre-registered candidates**

All candidates share the same visual architecture, optimizer, epochs, sampler,
augmentation, and seed. Candidate masks are:

```python
CANDIDATE_MODALITIES = {
    "visual_only": ("ir", "depth_color"),
    "visual_skeleton": ("ir", "depth_color", "skeleton"),
    "visual_imu": ("ir", "depth_color", "imu"),
    "visual_skeleton_imu": ("ir", "depth_color", "skeleton", "imu"),
}
```

Train each candidate for exactly 15 epochs in every train12 fold. Do not inspect
fold metrics per epoch; evaluate once after epoch 15. Pool fold predictions and
select by:

```text
Accuracy, Macro-F1, worst-user Accuracy, negative NLL, fixed candidate order
```

Candidate order is the mapping order above. Save hashes for fit users,
normalization, config, source, checkpoint, and predictions.

- [ ] **Step 4: Run grouped-CV tests and then the formal train12 CV**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_hierarchical_multimodal_grouped_cv.py
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/run_hierarchical_multimodal_teacher.py --mode grouped-cv
```

Expected: four complete pooled train12 archives; no user6/user7 prediction.

- [ ] **Step 5: Freeze the selected candidate in a commit**

```powershell
git add reports/hierarchical_multimodal_teacher_grouped_cv.json reports/hierarchical_multimodal_teacher_grouped_cv.md configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml
git commit -m "report: select hierarchical teacher candidate"
```

The commit records the selected candidate before any final development
evaluation.

---

### Historical Task 12: Selected-only evaluation (superseded; do not execute)

**Files:**
- Modify: `scripts/run_hierarchical_multimodal_teacher.py`
- Create: `scripts/report_hierarchical_multimodal_teacher.py`
- Create: `tests/test_report_hierarchical_multimodal_teacher.py`

**Interfaces:**
- Produces: one selected checkpoint and one 388-row prediction archive
- Produces: final JSON/Markdown report with ablations derived without additional user6/user7 model runs

- [ ] **Step 1: Write failing artifact and metric-recomputation tests**

```python
def test_final_archive_contains_canonical_union_and_one_candidate() -> None:
    grouped = json.loads(GROUPED_REPORT.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    archive = np.load(PREDICTIONS)
    assert report["selected_candidate"] == grouped["selected_candidate"]
    assert list(report["final_results"]) == [report["selected_candidate"]]
    assert len(archive["sample_ids"]) == 388
    assert len(np.unique(archive["sample_ids"])) == 388
    assert set(archive["labels"]) == set(range(40))


def test_report_metrics_recompute_from_predictions() -> None:
    report = load_report(REPORT)
    archive = np.load(PREDICTIONS)
    metrics = fixed_40_metrics(archive["labels"], archive["logits"], archive["user_ids"])
    assert metrics["accuracy"] == report["final_results"][report["selected_candidate"]]["accuracy"]
    assert sha256(PREDICTIONS) == report["artifacts"]["predictions"]["sha256"]
```

The first assertion binds final evaluation to the grouped-selected candidate
written in Task 11 without introducing a candidate-specific test edit.

- [ ] **Step 2: Run report tests to verify missing final artifacts**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_report_hierarchical_multimodal_teacher.py
```

Expected: FAIL because selected-only artifacts do not exist.

- [ ] **Step 3: Train the frozen candidate on all train12 and evaluate once**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/run_hierarchical_multimodal_teacher.py --mode selected-final
```

The command must:

- read the selected candidate from the committed grouped report;
- train exactly 15 epochs on all train12 users;
- fit visual/Skeleton/IMU normalization on train12 only;
- evaluate exactly 388 canonical validation rows;
- use train12 class prior for the three no-core rows;
- save logits, labels, sample IDs, users, core availability, group/segment
  attention, quality, masks, and per-group auxiliary logits;
- refuse any second final-evaluation event for the same run ID.

- [ ] **Step 4: Generate the report and run all verification gates**

```powershell
D:\Anaconda\envs\PyTorch2.7\python.exe scripts/report_hierarchical_multimodal_teacher.py
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q tests/test_report_hierarchical_multimodal_teacher.py
D:\Anaconda\envs\PyTorch2.7\python.exe -m pytest -q
git diff --check
```

The report computes the Spec Section 10 category and explicitly states whether
student planning is authorized. A result below `full_teacher_worthy` does not
authorize distillation implementation.

- [ ] **Step 5: Request two-axis review and commit final evidence**

Run Standards and Spec reviews against the implementation branch base. Resolve
all P1/P2 scientific-integrity findings, rerun the full suite, then commit:

```powershell
git add reports/hierarchical_multimodal_teacher_result.json reports/hierarchical_multimodal_teacher_result.md scripts/report_hierarchical_multimodal_teacher.py tests/test_report_hierarchical_multimodal_teacher.py
git commit -m "report: freeze hierarchical multimodal teacher result"
```

---

## Plan Self-Review

### Spec coverage

- Canonical 2,039/388 union: Tasks 1, 2, 5, 12.
- Eight normalized segments: Tasks 2, 3, 4, 6, 7.
- Early aligned IR+Depth fusion: Task 6.
- Persistent context/wrist groups without hard Top-k: Task 6.
- Sequence-level Skeleton/IMU fusion: Tasks 3, 4, 7.
- Masked 40-query cross-group fusion: Task 8.
- Auxiliary heads and anti-collapse: Task 9.
- Fixed train12 to user6/user7 isolation, grouped hard gate, and three-candidate
  development selection: Tasks 11-13.
- Canonical missing-pattern fallback: Tasks 2, 5, 12, 13.
- Metrics, rescue/harm, attention, size, and provenance: Task 13.
- Thermal/Radar exclusion and admission gates: global constraints and Spec;
  there is intentionally no Stage-1 implementation task for them.
- Student distillation: intentionally excluded from this implementation plan;
  the Spec requires a separate plan only after `full_teacher_worthy`.

### Placeholder scan

The plan contains no open implementation placeholder. Candidate names, shapes,
loss weights, split ownership, authorization conditions, commands, counts,
paths, and commit boundaries are explicit. Historical Tasks 11-12 are retained
only to explain existing grouped audit code and are explicitly non-executable.

### Type consistency

`SegmentBatch` is the data-side raw segment contract. `GroupTokens` is the
model-side encoded token contract. Visual and body encoders both produce
`GroupTokens`; hierarchical fusion consumes exactly those objects. The generic
split primitive and fixed-validation runner consume the same
`HierarchicalMultimodalTeacher` output keys. The grouped wrapper remains an
authorization-gated audit consumer of that primitive.
