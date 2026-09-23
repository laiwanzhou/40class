# Teammate-Style Single-Teacher Fixed-Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild a teammate-style privileged visual teacher → compact Visual/Skeleton/IMU Student pipeline on the 12-train/2-development/14-refit/4-final fixed split, while excluding all historical multi-teacher voting artifacts.

**Architecture:** New orchestration and adapters live in the existing `40class` code layout while the uploaded `teacher` branch remains a read-only source reference. Every stage writes immutable, hashed artifacts beneath one run root. The four final-evaluation users remain label-free until all A1–A9 candidate predictions are closed and hashed.

**Tech Stack:** Python 3.12, PyTorch 2.7/CUDA 12.8, torchvision, transformers, scikit-learn, NumPy, pandas, OpenCV, Ultralytics, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md`

## Global Constraints

- Training users: `user1,user2,user3,user5,user8,user9,user16,user18,user19,user20,user21,user22`.
- Development users: `user6,user7`.
- Final-refit users: the preceding 14 users.
- Final-evaluation users: `user4,user17,user23,user24`.
- Final-evaluation labels are unavailable to generation code and read once by the evaluator.
- Internal training and final refit must each contain all class IDs 0–39.
- No historical teammate checkpoint, generated cache, P310 target, Kaggle prediction, or expert-bank probability may be consumed.
- One visual teacher only: MCG-NJU VideoMAE-Large revision `0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4`.
- Deployment modalities are Visual, Skeleton, and IMU; Thermal and Radar are excluded.
- New artifacts live beneath the configurable run root `outputs/teammate_single_teacher_fixed_split/`.
- Execution requires at least 20 GiB free at the configured run root.
- The `teacher` branch source tree is read-only and verified against its original manifest before use.
- All stochastic stages use explicit seeds and save RNG/config/source provenance.

## Review Focus

- A final-evaluation input with a `label`, `class_id`, `correct`, or confusion-derived field must be rejected before generation.
- The 18 final rows with none of Visual/Skeleton/IMU available must remain in output order and receive the frozen training-prior fallback.
- A resumed stage with a changed source/config/input hash must fail instead of reusing stale artifacts.
- Every final-refit stage must include all 14 users and all 40 classes; every final candidate must include all 609 rows exactly once.
- A9 must fail if any P310 or expert-bank path is supplied, even when the file exists.

---

### Task 1: Freeze the experiment protocol and source adapter

**Files:**
- Create: `configs/experiments/teammate_single_teacher_fixed_split.yaml`
- Create: `src/experiments/no_vote_protocol.py`
- Create: `src/experiments/teammate_source.py`
- Create: `tests/test_no_vote_protocol.py`
- Create: `tests/test_teammate_source.py`

**Interfaces:**
- Consumes: repository manifest, split JSONs, teammate source root and `source_manifest.json`.
- Produces: `NoVoteProtocol`, `Partition`, `load_teammate_module(name)`, protocol identity hash, preflight report.

- [ ] **Step 1: Write failing split and preflight tests**

```python
def test_fixed_population_contract():
    protocol = NoVoteProtocol.from_yaml(CONFIG)
    assert protocol.train_users == frozenset(TRAIN12)
    assert protocol.development_users == frozenset({"user6", "user7"})
    assert protocol.refit_users == frozenset((*TRAIN12, "user6", "user7"))
    assert protocol.final_users == frozenset({"user4", "user17", "user23", "user24"})
    assert not protocol.refit_users & protocol.final_users

def test_preflight_rejects_low_space(monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda _: (100, 95, 5))
    with pytest.raises(RuntimeError, match="20 GiB"):
        preflight_run_root(Path("outputs/run"), minimum_free_gib=20)
```

- [ ] **Step 2: Run tests and confirm missing interfaces**

Run:

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_protocol.py tests/test_teammate_source.py -v
```

Expected: collection/import failure because the two modules do not exist.

- [ ] **Step 3: Implement immutable protocol dataclasses**

```python
@dataclass(frozen=True)
class Partition:
    name: Literal["train12", "development2", "refit14", "final4"]
    users: frozenset[str]
    labels_allowed: bool

@dataclass(frozen=True)
class NoVoteProtocol:
    train: Partition
    development: Partition
    refit: Partition
    final: Partition
    run_root: Path
    teammate_source_root: Path
    minimum_free_gib: int

    @classmethod
    def from_yaml(cls, path: Path) -> "NoVoteProtocol":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        parts = raw["partitions"]
        return cls(
            train=Partition("train12", frozenset(parts["train12"]), True),
            development=Partition("development2", frozenset(parts["development2"]), True),
            refit=Partition("refit14", frozenset(parts["refit14"]), True),
            final=Partition("final4", frozenset(parts["final4"]), False),
            run_root=Path(raw["run_root"]),
            teammate_source_root=Path(raw["teammate_source_root"]),
            minimum_free_gib=int(raw["minimum_free_gib"]),
        )

    def identity(self) -> str:
        payload = json.dumps(asdict(self), default=str, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()
```

The YAML records all users, seeds, source roots, pretrained revisions, run root, candidate grids, and stage constants from the spec. `identity()` hashes a canonical JSON representation.

- [ ] **Step 4: Implement source-manifest verification and imports**

```python
def verify_teammate_source(root: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = []
    for item in manifest["files"]:
        path = root / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if digest != item["sha256"] or (path.is_file() and path.stat().st_size != item["bytes"]):
            bad.append(item["path"])
    if bad:
        raise RuntimeError(f"teammate source mismatch: {bad[:5]}")
    return {"files": len(manifest["files"]), "manifest_sha256": sha256(manifest_path)}

def load_teammate_module(root: Path, module_name: str) -> ModuleType:
    verify_teammate_source(root, root.parent / "source_manifest.json")
    sys.path.insert(0, str(root / "aligned_multimodal"))
    return importlib.import_module(module_name)
```

Verification requires all 1,167 listed training files, matching sizes/SHA256, and no missing requested module. Imports occur only after verification.

- [ ] **Step 5: Run focused tests**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 6: Commit protocol scaffolding**

```powershell
git add configs/experiments/teammate_single_teacher_fixed_split.yaml src/experiments/no_vote_protocol.py src/experiments/teammate_source.py tests/test_no_vote_protocol.py tests/test_teammate_source.py
git commit -m "feat: freeze no-vote experiment protocol"
```

### Task 2: Build label-separated manifests and artifact records

**Files:**
- Create: `src/data/no_vote_manifest.py`
- Create: `src/experiments/artifact_record.py`
- Create: `scripts/build_no_vote_manifests.py`
- Create: `tests/test_no_vote_manifest.py`
- Create: `tests/test_artifact_record.py`

**Interfaces:**
- Consumes: canonical `metadata/manifest.csv`, `NoVoteProtocol`.
- Produces: labeled train/development/refit manifests, unlabeled final manifest, `ArtifactRecord.write/read/verify`.

- [ ] **Step 1: Write failing label-isolation and row-ownership tests**

```python
def test_final_manifest_has_no_label_fields(tmp_path):
    outputs = build_partition_manifests(MANIFEST, protocol(), tmp_path)
    rows = list(csv.DictReader(outputs.final.open(encoding="utf-8-sig")))
    forbidden = {"label", "labels", "class_id", "class_name", "correct"}
    assert not forbidden & set(rows[0])
    assert len(rows) == 609
    assert len({row["sample_id"] for row in rows}) == 609

def test_all_18_users_owned_once(tmp_path):
    outputs = build_partition_manifests(MANIFEST, protocol(), tmp_path)
    ownership = read_ownership(outputs.ownership)
    assert set(ownership) == set(ALL_18_USERS)
    assert all(len(value) == 1 for value in ownership.values())
```

- [ ] **Step 2: Run tests and observe import failure**

Run:

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_manifest.py tests/test_artifact_record.py -v
```

- [ ] **Step 3: Implement manifest construction**

```python
@dataclass(frozen=True)
class PartitionManifests:
    train: Path
    development: Path
    refit: Path
    final: Path
    ownership: Path

def build_partition_manifests(source: Path, protocol: NoVoteProtocol, output: Path) -> PartitionManifests:
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig")))
    mapping = {
        "train": protocol.train,
        "development": protocol.development,
        "refit": protocol.refit,
        "final": protocol.final,
    }
    paths = {}
    for name, part in mapping.items():
        selected = [dict(row) for row in rows if row["user_id"] in part.users]
        if not part.labels_allowed:
            for row in selected:
                for key in ("class_id", "class_name", "label", "correct"):
                    row.pop(key, None)
        paths[name] = write_csv_atomic(output / f"{name}.csv", selected)
    assert len({row["user_id"] for row in rows}) == 18
    return PartitionManifests(paths["train"], paths["development"], paths["refit"], paths["final"], write_ownership(output, mapping))
```

The function preserves canonical sample order, resolves raw modality paths, writes labels only to train/development/refit files, and asserts train12/refit14 each cover all 40 classes.

- [ ] **Step 4: Implement artifact provenance**

```python
@dataclass(frozen=True)
class ArtifactRecord:
    stage: str
    protocol_sha256: str
    config_sha256: str
    parent_sha256: tuple[str, ...]
    files: dict[str, str]
    users: tuple[str, ...]
    rows: int

    def write(self, path: Path) -> None:
        atomic_json(path, asdict(self))

    @classmethod
    def read(cls, path: Path) -> "ArtifactRecord":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def verify(self, root: Path, protocol_sha256: str) -> None:
        if self.protocol_sha256 != protocol_sha256:
            raise RuntimeError("protocol identity changed")
        bad = [name for name, digest in self.files.items() if sha256(root / name) != digest]
        if bad:
            raise RuntimeError(f"artifact hash mismatch: {bad}")
```

- [ ] **Step 5: Run tests and manifest smoke command**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_manifest.py tests/test_artifact_record.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/build_no_vote_manifests.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --smoke
```

Expected: four manifests, 18 unique users, no final labels.

- [ ] **Step 6: Commit manifest contracts**

```powershell
git add src/data/no_vote_manifest.py src/experiments/artifact_record.py scripts/build_no_vote_manifests.py tests/test_no_vote_manifest.py tests/test_artifact_record.py
git commit -m "feat: add fixed-split label-isolated manifests"
```

### Task 3: Rebuild teammate pose and ROI geometry

**Files:**
- Create: `src/experiments/pose_roi_adapter.py`
- Create: `scripts/build_no_vote_pose_roi.py`
- Create: `tests/test_no_vote_pose_roi.py`

**Interfaces:**
- Consumes: partition manifests, embedded/verified YOLO11n-pose weight, teammate P28/P29 modules.
- Produces: partition-neutral pose and multiscale ROI caches with label-free sample IDs.

- [ ] **Step 1: Write failing timestamp and label-isolation tests**

```python
def test_timestamped_skeleton_aligns_to_ir(tmp_path):
    mapping = compatible_frame_map(FIXTURE_SKELETON, "skeleton", sibling_ir=FIXTURE_IR)
    assert "2025-06-13_15-36-16.216_00000103" in mapping

def test_pose_cache_contains_no_label(tmp_path):
    cache = run_pose_smoke(FINAL_MANIFEST, tmp_path)
    with np.load(cache, allow_pickle=False) as values:
        assert not {"label", "class_id", "correct"} & set(values.files)
```

- [ ] **Step 2: Run the focused test and confirm missing adapter**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_pose_roi.py -v
```

- [ ] **Step 3: Implement verified wrappers around P28/P29**

```python
def compatible_frame_map(path: Path, modality: str, sibling_ir: Path | None = None) -> dict[str, Path]:
    raw = original_frame_map(path, modality)
    if modality != "skeleton" or not raw or sibling_ir is None:
        return raw
    ir = original_frame_map(sibling_ir, "ir")
    by_counter = {key.rsplit("_", 1)[-1]: key for key in ir}
    return {by_counter[key.rsplit("_", 1)[-1]]: value for key, value in raw.items()
            if key.rsplit("_", 1)[-1] in by_counter}

def build_pose_roi(manifest: Path, pose_weight: Path, output: Path, device: str) -> ArtifactRecord:
    p28 = output / "p28"
    p29 = output / "p29"
    run_module("build_adaptive_yolo11_pose_skeleton_cache", ["--manifest", manifest, "--model", pose_weight, "--output-dir", p28, "--device", device])
    run_module("build_multiscale_dir_rois", ["--manifest", manifest, "--pose-run", p28, "--output-dir", p29])
    return record_tree("pose_roi", output)
```

Reuse the timestamp-restoration behavior from the accepted inference `stage_runner`, then invoke teammate `build_adaptive_yolo11_pose_skeleton_cache` and `build_multiscale_dir_rois` with explicit paths.

- [ ] **Step 4: Run tests and a one-sample smoke extraction**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_pose_roi.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/build_no_vote_pose_roi.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition development2 --max-trials 1
```

Expected: synchronized IR/Depth/Skeleton geometry and finite ROI boxes.

- [ ] **Step 5: Commit pose/ROI adapter**

```powershell
git add src/experiments/pose_roi_adapter.py scripts/build_no_vote_pose_roi.py tests/test_no_vote_pose_roi.py
git commit -m "feat: rebuild teammate pose ROI geometry"
```

### Task 4: Train and refit the single VideoMAE-Large teacher

**Files:**
- Create: `src/experiments/visual_teacher.py`
- Create: `scripts/run_no_vote_visual_teacher.py`
- Create: `tests/test_no_vote_visual_teacher.py`

**Interfaces:**
- Consumes: label-separated manifests, ROI caches, VideoMAE-Large revision.
- Produces: train/development/refit/final feature caches, selected Ridge recipe, all-14 refit head, A1 probabilities.

- [ ] **Step 1: Write failing feature-contract and grid-selection tests**

```python
def test_teacher_feature_contract():
    assert extract_smoke().features.shape == (1, 2, 3, 1024)
    assert extract_smoke().kinetics_logits.shape == (1, 2, 3, 400)

def test_grid_tie_breaker():
    rows = [candidate(acc=.8, f1=.7, worst=.6, dim=6144, alpha=1000),
            candidate(acc=.8, f1=.7, worst=.6, dim=2048, alpha=3000)]
    assert select_ridge(rows).dimensions == 2048
```

- [ ] **Step 2: Run test and confirm missing module**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_visual_teacher.py -v
```

- [ ] **Step 3: Implement frozen extraction and Ridge grid**

```python
def extract_videomae_large(manifest: Path, roi_root: Path, model_revision: str, output: Path) -> ArtifactRecord:
    model_path = snapshot_download("MCG-NJU/videomae-large-finetuned-kinetics", revision=model_revision, local_files_only=True)
    run_module("build_p46_videomae_multiclip_cache", ["--manifest", manifest, "--p29-run", roi_root, "--model", model_path, "--output-dir", output])
    return record_tree("visual_teacher_features", output)

def fit_ridge_grid(train_cache: Path, development_cache: Path, output: Path) -> dict[str, object]:
    candidates = evaluate_ridge_grid(train_cache, development_cache,
                                     features=("early", "late", "window_mean", "early_late", "temporal_delta", "kinetics"),
                                     powers=(0.0, 0.5, 0.75), alphas=(300.0, 1000.0, 3000.0, 10000.0))
    selected = max(candidates, key=lambda row: (row["accuracy"], row["macro_f1"], row["worst_user_accuracy"], -row["dimensions"], row["alpha"]))
    atomic_json(output / "selection.json", {"selected": selected, "candidates": candidates})
    return selected

def refit_ridge(selected: dict[str, object], refit_cache: Path, output: Path) -> Path:
    model = make_model(float(selected["alpha"]))
    features, labels, weights = load_selected_features(refit_cache, selected)
    model.fit(features, labels, ridge__sample_weight=weights)
    joblib.dump(model, output / "ridge.joblib")
    return output / "ridge.joblib"
```

Use early/late windows, scene/person/workspace views, teammate feature transforms, and the exact 72-candidate grid. Development labels are permitted; final labels are not loaded.

- [ ] **Step 4: Run unit tests and two-row GPU smoke**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_visual_teacher.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/run_no_vote_visual_teacher.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --stage extract --partition development2 --max-trials 2
```

- [ ] **Step 5: Commit visual teacher stage**

```powershell
git add src/experiments/visual_teacher.py scripts/run_no_vote_visual_teacher.py tests/test_no_vote_visual_teacher.py
git commit -m "feat: add single VideoMAE teacher stage"
```

### Task 5: Distill the compatible MC3 visual Student

**Files:**
- Create: `src/experiments/visual_student.py`
- Create: `scripts/run_no_vote_visual_student.py`
- Create: `tests/test_no_vote_visual_student.py`

**Interfaces:**
- Consumes: raw pixels/ROI geometry, A1 logits/features.
- Produces: selected A2 checkpoint, all-14 refit checkpoint, `[B,2,3,T,512]` sequence caches, A2 probabilities.

- [ ] **Step 1: Write failing sequence and frozen-selection tests**

```python
def test_visual_student_sequence_contract():
    output = build_student().encode_backbone_sequence(fake_batch())
    assert output.shape[:4] == (2, 2, 3, 16)
    assert output.shape[-1] == 512

def test_refit_uses_frozen_epoch_budget():
    selection = Selection(epoch=16, config_sha256="abc")
    refit = build_refit_command(selection)
    assert refit.fixed_epochs == 16
    assert refit.early_stopping is False
```

- [ ] **Step 2: Run test and confirm missing student adapter**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_visual_student.py -v
```

- [ ] **Step 3: Implement fixed-split wrapper for teammate P86 training**

```python
def train_visual_student(train_manifest: Path, development_manifest: Path,
                         teacher_train: Path, teacher_development: Path,
                         output: Path, config: StudentConfig) -> Selection:
    command = visual_student_command(train_manifest, development_manifest, teacher_train,
                                     teacher_development, output, config, mode="select")
    subprocess.run(command, check=True)
    return Selection.read(output / "selection.json")

def refit_visual_student(refit_manifest: Path, teacher_refit: Path,
                         selection: Selection, output: Path) -> Path:
    command = visual_student_command(refit_manifest, None, teacher_refit, None, output,
                                     selection.config, mode="refit", fixed_epochs=selection.epoch)
    subprocess.run(command, check=True)
    return output / "visual_student.pt"

def build_sequence_cache(checkpoint: Path, pixel_cache: Path, output: Path) -> ArtifactRecord:
    run_module("build_p86_mc3_sequence_cache", ["--checkpoint", checkpoint, "--pixel-cache", pixel_cache, "--output-dir", output])
    return record_tree("mc3_sequence", output)
```

Port teammate `train_p86_visual_pixel_oof` into a fixed train/development/refit wrapper while retaining subject-robust augmentation and loss coefficients.

- [ ] **Step 4: Run tests and two-batch smoke training**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_visual_student.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/run_no_vote_visual_student.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --stage select --smoke --max-train-batches 2 --max-eval-batches 2
```

- [ ] **Step 5: Commit visual Student stage**

```powershell
git add src/experiments/visual_student.py scripts/run_no_vote_visual_student.py tests/test_no_vote_visual_student.py
git commit -m "feat: distill compatible MC3 visual student"
```

### Task 6: Build Skeleton/IMU motion caches

**Files:**
- Create: `src/experiments/motion_cache.py`
- Create: `scripts/build_no_vote_motion_cache.py`
- Create: `tests/test_no_vote_motion_cache.py`

**Interfaces:**
- Consumes: partition manifests, P28/P29 geometry, A2 visual time grids.
- Produces: P31 source caches and P86 two-window motion caches for all four partitions.

- [ ] **Step 1: Write failing modality and normalization tests**

```python
def test_missing_imu_is_masked_not_dropped():
    row = build_motion_row(sample_without_imu())
    assert row["imu_mask"].sum() == 0
    assert np.isfinite(row["imu"]).all()

def test_normalization_uses_declared_fit_users():
    state = fit_motion_normalization(rows, fit_users=TRAIN12)
    assert state.fit_users == tuple(TRAIN12)
    assert not set(FINAL4) & set(state.fit_users)
```

- [ ] **Step 2: Run test and observe missing module**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_motion_cache.py -v
```

- [ ] **Step 3: Implement P31/P86 wrappers**

```python
def build_p31_cache(manifest: Path, geometry: Path, output: Path) -> ArtifactRecord:
    run_module("build_p31_skeleton_imu_cache", ["--manifest", manifest, "--p29-run", geometry, "--output-dir", output])
    return record_tree("p31_motion", output)

def fit_motion_normalization(cache: Path, fit_users: Collection[str]) -> NormalizationState:
    rows = read_motion_rows(cache)
    selected = [row for row in rows if row.user_id in set(fit_users)]
    return NormalizationState.fit(selected, fit_users=tuple(sorted(fit_users)))

def build_p86_motion_windows(p31: Path, visual_sequence: Path,
                             state: NormalizationState, output: Path) -> ArtifactRecord:
    state_path = output / "normalization.json"
    state.write(state_path)
    run_module("build_p86_motion_window_cache", ["--p31-run", p31, "--sequence-cache", visual_sequence, "--normalization", state_path, "--output-dir", output])
    return record_tree("p86_motion_windows", output)
```

- [ ] **Step 4: Run tests and one-row cache verification**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_motion_cache.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/build_no_vote_motion_cache.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition development2 --max-trials 1
```

- [ ] **Step 5: Commit motion cache stage**

```powershell
git add src/experiments/motion_cache.py scripts/build_no_vote_motion_cache.py tests/test_no_vote_motion_cache.py
git commit -m "feat: add aligned skeleton IMU motion caches"
```

### Task 7: Rebuild the statistical IMU teacher

**Files:**
- Create: `src/experiments/imu_rf_teacher.py`
- Create: `scripts/run_no_vote_imu_rf.py`
- Create: `tests/test_no_vote_imu_rf.py`

**Interfaces:**
- Consumes: P31/P86 IMU caches and train/development/refit manifests.
- Produces: selected RF configuration, all-14 refit model, A4 logits/probabilities/availability.

- [ ] **Step 1: Write failing device-dropout and 40-class tests**

```python
def test_device_dropout_is_seeded_and_role_aligned():
    left = device_dropout(features, seed=20260811)
    right = device_dropout(features, seed=20260811)
    assert np.array_equal(left, right)
    assert left.shape == features.shape

def test_rf_probability_has_all_classes():
    probability = aligned_probability(model_with_missing_fit_class(), values, classes=40)
    assert probability.shape == (len(values), 40)
```

- [ ] **Step 2: Run failing tests**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_imu_rf.py -v
```

- [ ] **Step 3: Implement fixed-split RF teacher**

```python
def fit_imu_rf(train: MotionCache, development: MotionCache,
               candidates: Sequence[RFConfig], output: Path) -> Selection:
    results = []
    for config in candidates:
        model = config.build(seed=20260811)
        model.fit(device_dropout(train.features, config.dropout, config.seed), train.labels)
        results.append(score_rf(model, development, config))
    selected = max(results, key=lambda row: (row.accuracy, row.macro_f1, row.worst_user_accuracy, -row.complexity))
    selected.write(output / "selection.json")
    return selected

def refit_imu_rf(refit: MotionCache, selected: Selection, output: Path) -> Path:
    model = selected.config.build(seed=20260811)
    model.fit(device_dropout(refit.features, selected.config.dropout, 20260811), refit.labels)
    joblib.dump(model, output / "imu_rf.joblib")
    return output / "imu_rf.joblib"

def predict_imu_rf(model: Path, cache: MotionCache, output: Path) -> ArtifactRecord:
    estimator = joblib.load(model)
    probability = align_probability(estimator.predict_proba(cache.features), estimator.classes_, 40)
    atomic_npz(output / "probabilities.npz", sample_ids=cache.sample_ids, probabilities=probability, availability=cache.imu_available)
    return record_tree("imu_rf_predictions", output)
```

Use teammate statistical features and aligned device-dropout recipe. Candidate ordering and tie-breaking are serialized before fitting.

- [ ] **Step 4: Run tests and RF smoke**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_imu_rf.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/run_no_vote_imu_rf.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --smoke --max-rows 64
```

- [ ] **Step 5: Commit RF teacher**

```powershell
git add src/experiments/imu_rf_teacher.py scripts/run_no_vote_imu_rf.py tests/test_no_vote_imu_rf.py
git commit -m "feat: rebuild selected IMU RF teacher"
```

### Task 8: Train MoBind Skeleton/IMU branches

**Files:**
- Create: `src/experiments/mobind_pretrain.py`
- Create: `scripts/run_no_vote_mobind_pretrain.py`
- Create: `tests/test_no_vote_mobind_pretrain.py`

**Interfaces:**
- Consumes: A1 teacher logits/features, motion caches, A4 IMU probabilities.
- Produces: selected and all-14-refit `P86MoBindLite` checkpoints, branch probabilities and embeddings.

- [ ] **Step 1: Write failing loss-mask and branch-disable tests**

```python
def test_missing_modality_losses_are_zero():
    losses = compute_motion_losses(batch_with_no_imu(), model_output(), config())
    assert losses["imu_ce"] == 0
    assert losses["imu_teacher"] == 0

def test_branch_disable_preserves_other_branch():
    full = model(batch())
    no_imu = model(batch(), enabled_modalities=("skeleton",))
    assert torch.allclose(full["skeleton_logits"], no_imu["skeleton_logits"])
```

- [ ] **Step 2: Run tests and confirm missing wrapper**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_mobind_pretrain.py -v
```

- [ ] **Step 3: Implement fixed-split P86 pretraining wrapper**

```python
def train_mobind_pretrain(train: MotionCache, development: MotionCache,
                          visual_teacher: TeacherArtifacts,
                          imu_teacher: TeacherArtifacts,
                          output: Path, config: MoBindPretrainConfig) -> Selection:
    subprocess.run(mobind_command("select", train, development, visual_teacher,
                                  imu_teacher, output, config), check=True)
    return Selection.read(output / "selection.json")

def refit_mobind_pretrain(refit: MotionCache, visual_teacher: TeacherArtifacts,
                          imu_teacher: TeacherArtifacts,
                          selected: Selection, output: Path) -> Path:
    subprocess.run(mobind_command("refit", refit, None, visual_teacher, imu_teacher,
                                  output, selected.config, fixed_epochs=selected.epoch), check=True)
    return output / "mobind_lite.pt"
```

Preserve width 96, alignment width 64, 24-epoch maximum, and teammate loss components. Select epoch on user6/user7, then refit all 14 users for that fixed budget.

- [ ] **Step 4: Run tests and two-batch smoke**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_mobind_pretrain.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/run_no_vote_mobind_pretrain.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --stage select --smoke --max-train-batches 2 --max-eval-batches 2
```

- [ ] **Step 5: Commit MoBind pretraining**

```powershell
git add src/experiments/mobind_pretrain.py scripts/run_no_vote_mobind_pretrain.py tests/test_no_vote_mobind_pretrain.py
git commit -m "feat: train fixed-split MoBind motion branches"
```

### Task 9: Add simple fusion controls

**Files:**
- Create: `src/experiments/simple_fusion.py`
- Create: `scripts/run_no_vote_simple_fusion.py`
- Create: `tests/test_no_vote_simple_fusion.py`

**Interfaces:**
- Consumes: A2/A5 visual, Skeleton, and IMU probabilities.
- Produces: A6-VS, A6-VI, A6-VSI probabilities and frozen temperatures.

- [ ] **Step 1: Write failing availability-weighted mean tests**

```python
def test_missing_branch_does_not_dilute_probability():
    fused = calibrated_mean([visual, skeleton], availability=[True, False])
    assert np.allclose(fused, visual)

def test_temperature_is_fit_on_training_only():
    selection = select_temperatures(train, development)
    assert selection.fit_users == tuple(TRAIN12)
    assert selection.selection_users == ("user6", "user7")
```

- [ ] **Step 2: Run failing tests**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_simple_fusion.py -v
```

- [ ] **Step 3: Implement calibrated means and stage reporting**

```python
def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    result = minimize_scalar(lambda value: log_loss(labels, softmax(logits / value)), bounds=(0.25, 4.0), method="bounded")
    return float(result.x)

def calibrated_mean(probabilities: Sequence[np.ndarray], availability: Sequence[np.ndarray]) -> np.ndarray:
    values = np.stack(probabilities, axis=1)
    mask = np.stack(availability, axis=1).astype(np.float32)
    weighted = (values * mask[:, :, None]).sum(axis=1)
    return weighted / mask.sum(axis=1, keepdims=True).clip(min=1.0)

def build_simple_fusions(visual: TeacherArtifacts, skeleton: TeacherArtifacts,
                         imu: TeacherArtifacts, temperatures: Temperatures) -> dict[str, np.ndarray]:
    calibrated = {name: softmax(item.logits / temperatures[name])
                  for name, item in {"v": visual, "s": skeleton, "i": imu}.items()}
    return {
        "vs": calibrated_mean([calibrated["v"], calibrated["s"]], [visual.available, skeleton.available]),
        "vi": calibrated_mean([calibrated["v"], calibrated["i"]], [visual.available, imu.available]),
        "vsi": calibrated_mean(list(calibrated.values()), [visual.available, skeleton.available, imu.available]),
    }
```

- [ ] **Step 4: Run tests and report-generation smoke**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_simple_fusion.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/run_no_vote_simple_fusion.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition development2
```

- [ ] **Step 5: Commit simple fusion controls**

```powershell
git add src/experiments/simple_fusion.py scripts/run_no_vote_simple_fusion.py tests/test_no_vote_simple_fusion.py
git commit -m "feat: add no-vote simple fusion controls"
```

### Task 10: Train teammate-style MoBind fusion and matched controls

**Files:**
- Create: `src/experiments/mobind_fusion.py`
- Create: `scripts/run_no_vote_mobind_fusion.py`
- Create: `tests/test_no_vote_mobind_fusion.py`

**Interfaces:**
- Consumes: A2 visual Student/checkpoints/sequences, A5 motion checkpoint/caches, A1 teacher targets.
- Produces: selected and all-14 A7 checkpoints, A7 and matched-control probabilities.

- [ ] **Step 1: Write failing parameter-scope and shuffle-control tests**

```python
def test_stage_a_freezes_pretrained_motion_encoders():
    model = build_fusion_model()
    configure_stage_a(model)
    assert all(not p.requires_grad for p in model.motion_residual.encoder.parameters())

def test_skeleton_shuffle_preserves_marginal_but_breaks_pairing():
    shuffled = shuffle_modality(batch(), "skeleton", seed=7)
    assert torch.equal(shuffled["skeleton"].sort().values, batch()["skeleton"].sort().values)
    assert not torch.equal(shuffled["skeleton"], batch()["skeleton"])
```

- [ ] **Step 2: Run failing tests**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_mobind_fusion.py -v
```

- [ ] **Step 3: Implement S4/A7 fixed-split wrapper**

```python
def train_mobind_fusion(train: FusionInputs, development: FusionInputs,
                        visual_checkpoint: Path, motion_checkpoint: Path,
                        output: Path, config: FusionConfig) -> Selection:
    subprocess.run(fusion_command("select", train, development, visual_checkpoint,
                                  motion_checkpoint, output, config), check=True)
    return Selection.read(output / "selection.json")

def refit_mobind_fusion(refit: FusionInputs, selected: Selection,
                        output: Path) -> Path:
    subprocess.run(fusion_command("refit", refit, None, selected.visual_checkpoint,
                                  selected.motion_checkpoint, output, selected.config,
                                  fixed_stage_a=4, fixed_stage_b=20), check=True)
    return output / "unified_student.pt"

def evaluate_controls(model: nn.Module, inputs: FusionInputs,
                      controls: Sequence[str]) -> dict[str, np.ndarray]:
    outputs = {}
    for control in controls:
        batch = apply_control(inputs, control, seed=20260811)
        outputs[control] = predict_probabilities(model, batch)
    return outputs
```

Use the exact stage budgets and loss coefficients from the spec. Control definitions are `mask_motion`, `shuffle_skeleton`, `shuffle_imu`, `zero_skeleton`, and `zero_imu`.

- [ ] **Step 4: Run tests and one-batch stage smoke**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_mobind_fusion.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/run_no_vote_mobind_fusion.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --stage select --smoke --max-train-batches 1 --max-eval-batches 1
```

- [ ] **Step 5: Commit learned fusion stage**

```powershell
git add src/experiments/mobind_fusion.py scripts/run_no_vote_mobind_fusion.py tests/test_no_vote_mobind_fusion.py
git commit -m "feat: train teammate style no-vote fusion"
```

### Task 11: Add fixed repeat/session processing

**Files:**
- Create: `src/experiments/session_repeat.py`
- Create: `scripts/run_no_vote_session_repeat.py`
- Create: `tests/test_no_vote_session_repeat.py`

**Interfaces:**
- Consumes: A7 probabilities, label-free recording metadata, training/development labels for selection only.
- Produces: selected A8 operator, all-14 refit transition model, unlabeled final probabilities.

- [ ] **Step 1: Write failing order-invariance and no-edge tests**

```python
def test_restored_outputs_are_row_order_invariant():
    first = process(rows, probabilities, config)
    second = process(rows[::-1], probabilities[::-1], config)
    assert np.allclose(first, second[::-1])

def test_failed_gate_returns_original_emission():
    out = repeat_consensus(probability, peers=[], threshold=.9)
    assert np.array_equal(out, probability)
```

- [ ] **Step 2: Run failing tests**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_session_repeat.py -v
```

- [ ] **Step 3: Implement fixed-split transition/repeat selection**

```python
def fit_transition(labels: np.ndarray, sessions: list[np.ndarray], alpha: float) -> TransitionModel:
    counts = np.full((40, 40), alpha, dtype=np.float64)
    for session in sessions:
        for left, right in zip(labels[session[:-1]], labels[session[1:]]):
            counts[left, right] += 1.0
    return TransitionModel(counts / counts.sum(axis=1, keepdims=True))

def build_repeat_groups(metadata: Metadata, probabilities: np.ndarray,
                        config: RepeatConfig) -> list[np.ndarray]:
    graph = candidate_graph(metadata, max_gap=config.max_gap,
                            duration_tolerance=config.duration_tolerance)
    return connected_groups(graph, probabilities,
                            minimum_similarity=config.minimum_similarity)

def select_session_repeat(train: SessionData, development: SessionData,
                          candidates: Sequence[SessionRepeatConfig]) -> Selection:
    results = [score_session_repeat(config, train, development) for config in candidates]
    return max(results, key=lambda row: (row.accuracy, row.macro_f1, -row.changed_rows))

def refit_and_apply(selection: Selection, refit: SessionData,
                    target: UnlabeledSessionData) -> np.ndarray:
    transition = fit_transition(refit.labels, refit.sessions, selection.config.alpha)
    groups = build_repeat_groups(target.metadata, target.probabilities, selection.config.repeat)
    return decode_and_propagate(target.probabilities, transition, target.sessions, groups, selection.config)
```

- [ ] **Step 4: Run tests and metadata smoke**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_session_repeat.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/run_no_vote_session_repeat.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition development2 --smoke
```

- [ ] **Step 5: Commit repeat/session stage**

```powershell
git add src/experiments/session_repeat.py scripts/run_no_vote_session_repeat.py tests/test_no_vote_session_repeat.py
git commit -m "feat: add single-emission session repeat processing"
```

### Task 12: Add 12-epoch and 40-epoch target adaptation

**Files:**
- Create: `src/experiments/target_adaptation.py`
- Create: `scripts/run_no_vote_target_adaptation.py`
- Create: `tests/test_no_vote_target_adaptation.py`

**Interfaces:**
- Consumes: all-14 A7 checkpoint, A8 unlabeled fixed probabilities, final pixel/motion caches.
- Produces: A9-12 and A9-40 checkpoints/probabilities with no labels.

- [ ] **Step 1: Write failing forbidden-source and parameter-scope tests**

```python
@pytest.mark.parametrize("value", ["p310", "expert_bank", "p309_union"])
def test_forbidden_target_source(value):
    with pytest.raises(ValueError, match="forbidden"):
        validate_target_source(Path(value + ".npz"))

def test_adaptation_parameter_scope():
    model = build_model()
    configure_adaptation(model, scope="heads_motion_encoder")
    assert not any(p.requires_grad for p in model.visual.backbone.parameters())
    assert any(p.requires_grad for p in model.motion_residual.encoder.parameters())
```

- [ ] **Step 2: Run failing tests**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_target_adaptation.py -v
```

- [ ] **Step 3: Implement fixed-target adaptation endpoints**

```python
def adapt_target(base_checkpoint: Path, fixed_targets: Path,
                 pixel_cache: Path, motion_cache: Path,
                 output: Path, config: AdaptationConfig) -> Path:
    validate_target_source(fixed_targets)
    command = adaptation_command(base_checkpoint, fixed_targets, pixel_cache,
                                 motion_cache, output, config)
    subprocess.run(command, check=True)
    return output / "unified_student.pt"
```

Implement two immutable configs: primary 12 epochs/seed 20260812 and exploratory 40 epochs/seed 20260826. Neither reads target labels or selects checkpoints.

- [ ] **Step 4: Run tests and two-batch adaptation smoke**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_target_adaptation.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/run_no_vote_target_adaptation.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --endpoint 12 --smoke --max-train-batches 2
```

- [ ] **Step 5: Commit target adaptation**

```powershell
git add src/experiments/target_adaptation.py scripts/run_no_vote_target_adaptation.py tests/test_no_vote_target_adaptation.py
git commit -m "feat: add no-vote target adaptation"
```

### Task 13: Close generation and reveal final labels once

**Files:**
- Create: `src/experiments/no_vote_evaluation.py`
- Create: `scripts/freeze_no_vote_generation.py`
- Create: `scripts/evaluate_no_vote_ablation.py`
- Create: `tests/test_no_vote_evaluation.py`

**Interfaces:**
- Consumes: all frozen A1–A9 predictions and their provenance; final labels only in evaluator.
- Produces: `generation_complete.json`, `evaluation.json`, `evaluation.md`, per-user/per-class CSVs.

- [ ] **Step 1: Write failing freeze/reveal tests**

```python
def test_generation_bundle_has_no_labels(tmp_path):
    bundle = freeze_generation(CANDIDATES, tmp_path)
    for path in bundle.files:
        assert not contains_forbidden_label_field(path)

def test_evaluator_rejects_hash_drift(tmp_path):
    bundle = freeze_generation(CANDIDATES, tmp_path)
    bundle.prediction_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash"):
        evaluate(bundle.manifest, FINAL_LABELS, tmp_path / "evaluation")
```

- [ ] **Step 2: Run failing tests**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_evaluation.py -v
```

- [ ] **Step 3: Implement generation closure and paired metrics**

```python
def freeze_generation(candidates: Mapping[str, Path], output: Path) -> GenerationBundle:
    files = {name: sha256(path) for name, path in sorted(candidates.items())}
    for path in candidates.values():
        reject_label_fields(path)
    manifest = output / "generation_complete.json"
    atomic_json(manifest, {"files": files, "closed_at": datetime.now(timezone.utc).isoformat()})
    return GenerationBundle(manifest=manifest, files=tuple(candidates.values()))

def evaluate(manifest: Path, labels: Path, output: Path) -> EvaluationReport:
    bundle = verify_generation_manifest(manifest)
    truth = load_final_labels(labels, expected_rows=609)
    metrics = {name: paired_metrics(truth, load_probability(path))
               for name, path in bundle.candidates.items()}
    report = EvaluationReport(metrics=metrics, generation_sha256=sha256(manifest))
    report.write(output)
    return report
```

Metrics include correct/609, correct/591, fixed-40 macro-F1, four per-user accuracies, per-class recall, rescue/harm/net/disagreement, NLL, Brier score, missing-modality subgroups, and A9/A8 agreement.

- [ ] **Step 4: Run evaluator tests and a synthetic end-to-end fixture**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_evaluation.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/freeze_no_vote_generation.py --fixture tests/fixtures/no_vote_candidates.json --output tmp/no_vote_fixture
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/evaluate_no_vote_ablation.py --manifest tmp/no_vote_fixture/generation_complete.json --labels tests/fixtures/no_vote_labels.csv --output tmp/no_vote_fixture/evaluation
```

- [ ] **Step 5: Commit evaluator**

```powershell
git add src/experiments/no_vote_evaluation.py scripts/freeze_no_vote_generation.py scripts/evaluate_no_vote_ablation.py tests/test_no_vote_evaluation.py
git commit -m "feat: add frozen reveal and ablation evaluator"
```

### Task 14: Add resumable orchestration and acceptance checks

**Files:**
- Create: `scripts/run_teammate_single_teacher_pipeline.py`
- Create: `tests/test_no_vote_orchestrator.py`
- Create: `docs/teammate_single_teacher_fixed_split.md`

**Interfaces:**
- Consumes: the config and every stage command/artifact contract.
- Produces: resumable `preflight`, `smoke`, `generate`, `freeze`, and `evaluate` commands.

- [ ] **Step 1: Write failing resume and stage-order tests**

```python
def test_resume_skips_matching_complete_stage(tmp_path):
    stage = completed_stage(tmp_path, protocol="abc")
    assert decide_stage(stage, protocol="abc") == "skip"
    assert decide_stage(stage, protocol="changed") == "error"

def test_evaluate_requires_generation_complete(tmp_path):
    with pytest.raises(RuntimeError, match="generation_complete"):
        run_pipeline("evaluate", config(), tmp_path)
```

- [ ] **Step 2: Run failing tests**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_orchestrator.py -v
```

- [ ] **Step 3: Implement explicit stage orchestration**

```python
STAGES = (
    "manifest", "pose_roi", "visual_teacher", "visual_student",
    "motion_cache", "imu_rf", "mobind_pretrain", "simple_fusion",
    "mobind_fusion", "session_repeat", "adaptation_12",
    "adaptation_40", "freeze", "evaluate",
)

def run_pipeline(mode: Literal["preflight", "smoke", "generate", "freeze", "evaluate"],
                 config: Path) -> None:
    protocol = NoVoteProtocol.from_yaml(config)
    if mode == "preflight":
        run_preflight(protocol)
        return
    if mode == "evaluate":
        require_file(protocol.run_root / "generation_complete.json")
        run_evaluator(protocol)
        return
    stop = "freeze" if mode == "freeze" else "adaptation_40"
    for stage in STAGES[: STAGES.index(stop) + 1]:
        run_or_resume(stage, protocol, smoke=(mode == "smoke"))
```

`generate` stops before labels; `freeze` closes hashes; `evaluate` is a separate explicit command.

- [ ] **Step 4: Run all new tests and smoke pipeline**

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -m pytest tests/test_no_vote_*.py tests/test_teammate_source.py -v
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/run_teammate_single_teacher_pipeline.py preflight --config configs/experiments/teammate_single_teacher_fixed_split.yaml
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' scripts/run_teammate_single_teacher_pipeline.py smoke --config configs/experiments/teammate_single_teacher_fixed_split.yaml
```

Expected: all tests pass; smoke reaches frozen unlabeled predictions for a tiny fixture without opening final labels.

- [ ] **Step 5: Document exact commands and resource estimates**

The documentation records environment setup, required 20 GiB free-space gate, expected 10–20 GPU-hours, input paths, each stage command, resume behavior, and interpretation boundary.

- [ ] **Step 6: Commit orchestration and docs**

```powershell
git add scripts/run_teammate_single_teacher_pipeline.py tests/test_no_vote_orchestrator.py docs/teammate_single_teacher_fixed_split.md
git commit -m "feat: orchestrate fixed-split teammate pipeline"
```

## Plan Self-Review

- Spec coverage: Tasks 1–14 cover split identity, label isolation, ROI, A1 visual teacher, A2 visual Student, motion caches, A4 RF teacher, A5 motion pretraining, A6 controls, A7 fusion/controls, A8 session/repeat, A9 adaptation, reveal/evaluation, and resumable orchestration.
- Placeholder scan: all interfaces, test commands, stage outputs, and commit commands are explicit.
- Type consistency: `NoVoteProtocol`, `ArtifactRecord`, `Selection`, `TeacherArtifacts`, and stage inputs are introduced before downstream use; the implementation must place shared dataclasses in the earliest owning module and import them thereafter.
- Review Focus: each of the five listed failure modes has a corresponding test in Tasks 1, 2, 6, 12, or 14.
- Scope: stages are sequential and share artifact contracts, so one integrated plan is preferable to independent plans; each task still ends in a testable commit.
