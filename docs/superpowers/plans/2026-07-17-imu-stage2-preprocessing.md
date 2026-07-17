# IMU Stage 2 Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved IMU Stage 2 v1 offline and online pipeline: exact-time feature-aware 10 Hz tensors, auditable artifacts, leak-free training normalization, variable-length masked batching, and raw-test inference.

**Architecture:** Official training conversion loads the accepted Stage 1 artifacts, while online test conversion calls a reusable Stage 1 in-memory adapter; both produce `Stage1ActionData` and enter one side-effect-free Stage 2 core. Focused IO, indexing, normalization, dataset/collate, model, and inference modules validate explicit contracts and hashes without changing the existing Stage 1 output or legacy IMU baseline.

**Tech Stack:** Python 3.12; standard-library `argparse`, `csv`, `dataclasses`, `decimal`, `enum`, `hashlib`, `json`, `logging`, `pathlib`, `re`, `shutil`, `tempfile`; NumPy, Pandas, PyTorch, PyYAML, pytest; Bash wrapper for `inference.sh`.

## Global Constraints

- Execute in the isolated `D:\work\2026.7.14_kaggle\40class\IMU` worktree after confirming branch and clean tracked state.
- Do not regenerate or overwrite the accepted training Stage 1 root at `D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU`.
- Preserve all existing Stage 1 behavior and all current Stage 1 tests.
- Sensor order is exactly `LL, RL, LA, RA, C`.
- Feature order is exactly the 16-column order in the approved design.
- Stage 2 uses only exact `int64 relative_time_ns`; never float seconds or rounded milliseconds as a key.
- Grid step is exactly `100_000_000 ns`; maximum interpolation endpoint gap is exactly `300_000_000 ns`; extrapolation is forbidden.
- Persist unstandardized variable-length arrays; do not crop, window, pad, clip, or silently skip samples.
- Invalid persisted cells are NaN; normalization changes only valid cells and then replaces invalid model inputs with zero.
- `sensor_mask`, `valid_mask`, `sequence_mask`, `usable_sensor_mask`, and IMU modality availability keep distinct meanings.
- `--dry-run` performs zero writes. Fresh, `--resume`, and `--overwrite` modes follow the design exactly.
- Online degradation catches only the explicit typed-error allowlist; unknown exceptions are global failures.
- Do not run the full dataset, formal Stage 2 generation, training, or competition inference until the corresponding plan checkpoint is explicitly approved.
- Use `apply_patch` for source edits. Do not commit generated data, audit outputs, checkpoints, normalization artifacts, or submissions.

## Planned file structure

- Create `src/data/imu_stage2_contracts.py`: fixed orders, dataclasses, enums, typed errors, canonical JSON, fingerprints, and contract validation.
- Create `src/data/imu_stage1_bridge.py`: official Stage 1 artifact discovery/loading and raw Stage 1 in-memory adaptation.
- Create `src/data/imu_stage2_core.py`: duplicate aggregation, segmentation, grid construction, interpolation, and action-result QC.
- Create `src/data/imu_stage2_io.py`: NPZ/schema/QC/manifest validation and staged publication.
- Create `scripts/preprocess_imu_stage2.py`: root safety, fresh/resume/overwrite/dry-run orchestration, logging, summary, and exit codes.
- Create `scripts/build_imu_training_index.py`: class order, user split, strict eligibility, canonical index, and hashes.
- Create `scripts/compute_imu_normalization.py`: fold-only streaming statistics and normalization artifacts.
- Create `src/data/imu_stage2_dataset.py`: validated action loading, valid-only normalization, length sampler, and dynamic collate.
- Create `src/models/imu_stage2_tcn.py`: mask-aware v1 IMU classifier and model-output contract.
- Create `src/inference/imu_stage2_pipeline.py` and `src/inference/__init__.py`: test discovery, source adaptation, bundle validation, typed degradation, batching, and submission publication.
- Create `scripts/infer_imu_stage2.py` and `inference.sh`: public online entry points.
- Create focused tests under `tests/imu_stage2/`; leave `src/data/imu_dataset.py`, `src/models/tcn.py`, and the legacy config unchanged.
- Modify `scripts/preprocess_imu_stage1.py` only where necessary to expose an exact-time in-memory result without changing its CLI or artifacts.
- Modify `tests/test_preprocess_imu_stage1.py` only for Stage 1 regression/replay coverage created by that extraction.

---

### Task 1: Lock Stage 2 contracts, hashes, and typed errors

**Files:**
- Create: `src/data/imu_stage2_contracts.py`
- Create: `tests/imu_stage2/test_contracts.py`
- Create: `tests/imu_stage2/__init__.py`

**Interfaces:**
- Produces: `SENSOR_ORDER`, `FEATURE_ORDER`, `Stage1ActionData`, `Stage2ActionResult`, `TestSampleDescriptor`, `ImuActionSource`, `InferenceSample`.
- Produces: `DataStatus`, `WriteStatus`, the seven online-degradable error classes, `canonical_json_bytes()`, `sha256_file()`, `contract_sha256()`.

- [ ] **Step 1: Write failing contract and hash tests**

```python
def test_contract_hash_ignores_provenance() -> None:
    contract = {"schema_version": "imu-stage2-v1", "grid_step_ns": 100_000_000}
    assert contract_sha256(contract) == contract_sha256(dict(reversed(list(contract.items()))))


def test_stage2_result_requires_nan_at_invalid_cells() -> None:
    result = make_minimal_result()
    result.values[0, 0, :] = 0.0
    with pytest.raises(ValueError, match="invalid cells must be NaN"):
        result.validate()


def test_degradable_errors_carry_structured_fields() -> None:
    error = MissingImuDirectoryError("SM_test_0001", "missing IMU")
    assert error.error_code == "missing_imu_directory"
    assert error.failure_stage == "source_adapter"
    assert error.sample_id == "SM_test_0001"
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/imu_stage2/test_contracts.py -q
```

Expected: collection fails because `src.data.imu_stage2_contracts` does not exist.

- [ ] **Step 3: Implement the exact public contracts**

```python
SENSOR_ORDER = ("LL", "RL", "LA", "RA", "C")
FEATURE_ORDER = (
    "acc_x_g", "acc_y_g", "acc_z_g",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "angle_x_deg", "angle_y_deg", "angle_z_deg",
    "mag_x_ut", "mag_y_ut", "mag_z_ut",
    "quat_0", "quat_1", "quat_2", "quat_3",
)


class DataStatus(str, Enum):
    SUCCESS = "success"
    SUCCESS_WITH_WARNINGS = "success_with_warnings"
    INCOMPLETE_SENSORS = "incomplete_sensors"
    NO_USABLE_GRID_CELLS = "no_usable_grid_cells"
    FAILED = "failed"


class WriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED_EXISTING = "skipped_existing"
    QC_ONLY = "qc_only"
    NOT_WRITTEN = "not_written"


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def contract_sha256(contract: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
```

Implement `Stage2ActionResult.validate()` with exact dtype, shape, time-grid,
finite/NaN, missing-sensor, and usable-mask checks. Define each degradable
exception with fixed `error_code`, `failure_stage`, `sample_id`, and
`safe_message`; do not define a generic degradable base that callers can use
to catch unknown exceptions.

- [ ] **Step 4: Verify GREEN and commit**

```powershell
python -m pytest tests/imu_stage2/test_contracts.py -q
git add src/data/imu_stage2_contracts.py tests/imu_stage2
git commit -m "feat(imu): define stage 2 contracts"
```

Expected: all Task 1 tests pass; commit contains contracts/tests only.

---

### Task 2: Build exact-time Stage 1 artifact and raw bridges

**Files:**
- Create: `src/data/imu_stage1_bridge.py`
- Create: `tests/imu_stage2/test_stage1_bridge.py`
- Modify: `scripts/preprocess_imu_stage1.py`
- Modify: `tests/test_preprocess_imu_stage1.py`

**Interfaces:**
- Consumes: accepted `manifest.csv`, `imu_merged.csv`, `qc.json`, and existing Stage 1 parsing/validation functions.
- Produces: `decimal_seconds_to_ns(text: str) -> np.int64`.
- Produces: `discover_stage1_artifacts(root: Path) -> list[Stage1ArtifactDescriptor]`.
- Produces: `load_stage1_action(descriptor: Stage1ArtifactDescriptor) -> Stage1ActionData`.
- Produces: `process_raw_imu_source(source: ImuActionSource) -> Stage1ActionData`.
- Produces: `build_in_memory_action_result(descriptor, csv_results, validated_results) -> InMemoryActionResult`.

- [ ] **Step 1: Add exact-decimal RED tests**

```python
@pytest.mark.parametrize(
    ("text", "expected"),
    [("0.091", 91_000_000), ("0.000001", 1_000), ("0.000000001", 1)],
)
def test_decimal_seconds_to_ns_is_exact(text: str, expected: int) -> None:
    assert decimal_seconds_to_ns(text) == expected


def test_decimal_seconds_to_ns_rejects_subnanosecond() -> None:
    with pytest.raises(ValueError, match="represented exactly"):
        decimal_seconds_to_ns("0.0000000001")
```

Add a synthetic accepted Stage 1 action and assert `relative_time_s` is read as
text, `relative_time_ns.dtype == np.int64`, row alignment is preserved, fixed
features are selected explicitly, and source file ranks follow natural input
file order.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_stage1_bridge.py -q
```

Expected: import failure for the missing bridge.

- [ ] **Step 3: Implement the offline loader and source fingerprints**

```python
def decimal_seconds_to_ns(text: str) -> np.int64:
    value = Decimal(text)
    if not value.is_finite() or value < 0:
        raise ValueError("Invalid relative time")
    nanoseconds = value * Decimal(1_000_000_000)
    if nanoseconds != nanoseconds.to_integral_value():
        raise ValueError("Relative time cannot be represented exactly in nanoseconds")
    integer = int(nanoseconds)
    if not 0 <= integer <= np.iinfo(np.int64).max:
        raise OverflowError("Relative time is outside int64 range")
    return np.int64(integer)
```

Read the CSV with `dtype={"relative_time_s": "string"}` and
`encoding="utf-8-sig"`. Build file ranks from the Stage 1 QC input list, map
every `source_file`, assign `_source_file_rank`, and assign
`_stage1_row_index=np.arange(N,dtype=np.int64)`. Later duplicate groups use the
exact stable key `source_file_rank, source_row_index, stage1_row_index`.
Hash the Stage 1 CSV/QC bytes and canonical selected manifest row.

- [ ] **Step 4: Extract the raw in-memory adapter without changing Stage 1 output**

Add a pure Stage 1 return boundary that retains exact parsed absolute time and
file rank until the bridge computes integer differences:

```python
def process_action_in_memory(descriptor: ActionDescriptor) -> InMemoryActionResult:
    """Return validated exact-time rows without writing Stage 1 artifacts."""
    csv_results = tuple(
        read_csv_robust(path)
        for path in descriptor.input_csv_files
    )
    validated_results = tuple(
        validate_dataframe(result)
        for result in csv_results
    )
    return build_in_memory_action_result(
        descriptor,
        csv_results,
        validated_results,
    )


def process_action(descriptor: ActionDescriptor) -> ActionResult:
    memory = process_action_in_memory(descriptor)
    return build_legacy_action_result(memory)
```

The extraction must reuse the existing robust CSV scanner and validation code.
The legacy adapter must reproduce the same `imu_merged.csv`, QC, manifest row,
warnings, and statuses byte-for-byte or value-for-value as existing tests
require. `process_raw_imu_source()` converts exact absolute-time deltas directly
to `int64` nanoseconds and constructs `Stage1ActionData` without writing.

- [ ] **Step 5: Run Stage 1 regression and bridge GREEN tests**

```powershell
python -m pytest tests/test_preprocess_imu_stage1.py tests/imu_stage2/test_stage1_bridge.py -q
```

Expected: every pre-existing Stage 1 test plus the bridge tests pass.

- [ ] **Step 6: Commit the isolated refactor**

```powershell
git diff --check
git add scripts/preprocess_imu_stage1.py tests/test_preprocess_imu_stage1.py src/data/imu_stage1_bridge.py tests/imu_stage2/test_stage1_bridge.py
git commit -m "refactor(imu): expose reusable stage 1 action core"
```

Expected: no Stage 1 schema, status, CLI, output, or safety behavior changes.

---

### Task 3: Implement duplicate aggregation and continuous segments

**Files:**
- Create: `src/data/imu_stage2_core.py`
- Create: `tests/imu_stage2/test_stage2_aggregation.py`

**Interfaces:**
- Consumes: `Stage1ActionData`.
- Produces: `validate_stage1_records()`, `aggregate_sensor_timestamps()`, `split_continuous_segments()`.
- Produces: `AggregatedSensorSeries` with increasing `time_ns`, `float64 values`, and QC counters.

- [ ] **Step 1: Add RED tests for all aggregation groups**

Create focused tests that assert arithmetic means for the nine ordinary
features, circular mean for `179/-179`, normalized sign-aligned quaternion
mean for `q/-q`, partial invalid-record exclusion, singleton fallback,
degenerate circular/quaternion timestamp deletion, stable first-reference
selection by rank, and original duplicate counts before exclusions.

```python
def test_duplicate_quaternions_align_sign_before_mean() -> None:
    series = aggregate_sensor_timestamps(make_stage1_group([Q, -Q]))
    assert np.allclose(np.linalg.norm(series.values[0, 12:16]), 1.0)
    assert np.allclose(series.values[0, 12:16], Q)
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_stage2_aggregation.py -q
```

Expected: import failure for the missing core.

- [ ] **Step 3: Implement record validation and feature-aware aggregation**

Use fixed slices `(0:6)`, `(6:9)`, `(9:12)`, and `(12:16)` only through named
constants derived from `FEATURE_ORDER`. Reject a whole record when any feature
is non-finite or quaternion norm is below `1e-8`. Compute group statistics
before rejection. Delete the sensor timestamp when any required aggregated
feature group degenerates.

- [ ] **Step 4: Add segment-boundary and quaternion-canonicalization RED tests**

```python
def test_continuity_restarts_after_gap_over_300_ms() -> None:
    series = make_series(times=[0, 300_000_000, 600_000_001])
    segments = split_continuous_segments(series, max_gap_ns=300_000_000)
    assert [segment.time_ns.tolist() for segment in segments] == [
        [0, 300_000_000], [600_000_001]
    ]
```

Assert each segment's first quaternion makes its first absolute-`>=1e-8`
component positive, later signs follow adjacent dot products, and angle unwrap
does not cross the segment boundary.

- [ ] **Step 5: Implement segment processing and verify GREEN**

```powershell
python -m pytest tests/imu_stage2/test_stage2_aggregation.py -q
git add src/data/imu_stage2_core.py tests/imu_stage2/test_stage2_aggregation.py
git commit -m "feat(imu): aggregate exact stage 2 timestamps"
```

Expected: aggregation and segment tests pass with no float-time grouping.

---

### Task 4: Implement the 10 Hz grid, feature-aware interpolation, and statuses

**Files:**
- Modify: `src/data/imu_stage2_core.py`
- Create: `tests/imu_stage2/test_stage2_grid.py`

**Interfaces:**
- Consumes: aggregated per-sensor segments.
- Produces: `build_action_grid(stage1_end_ns: int) -> np.ndarray`.
- Produces: `interpolate_sensor_on_grid(series: AggregatedSensorSeries, grid_ns: np.ndarray, max_gap_ns: int = 300_000_000) -> SensorGridResult`.
- Produces: `process_stage2_action(data: Stage1ActionData, hard_safety_limit_t: int) -> Stage2ActionResult`.

- [ ] **Step 1: Add grid-boundary RED tests**

```python
@pytest.mark.parametrize(
    ("end_ns", "expected_ms"),
    [(0, [0]), (99_999_999, [0]), (100_000_000, [0, 100]), (2_263_000_000, list(range(0, 2201, 100)))],
)
def test_grid_uses_floor_and_includes_endpoint(end_ns: int, expected_ms: list[int]) -> None:
    assert (build_action_grid(end_ns) // 1_000_000).tolist() == expected_ms
```

Add exact-hit, 300-ms-inclusive interpolation, over-300-ms rejection,
no-extrapolation, angle wrap, quaternion nlerp, and shared-endpoint tests.
Also assert the core raises `SequenceLengthSafetyError` before allocating the
grid when the computed `T` exceeds `hard_safety_limit_t`.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_stage2_grid.py -q
```

Expected: missing grid functions.

- [ ] **Step 3: Implement grid and per-segment lookup**

Construct the int64 grid with `np.arange(0, grid_end_ns + GRID_STEP_NS,
GRID_STEP_NS)`, but first calculate `T=grid_end_ns//GRID_STEP_NS+1` and compare
it with the safety limit. Use `np.searchsorted` on each segment. Exact hits bypass gap
checks. Non-hits require strict left/right endpoints in one segment and gap
`<=MAX_INTERPOLATION_GAP_NS`. Write a unit only after all 16 float64 results
pass validation; cast once to float32 and revalidate.

- [ ] **Step 4: Add result-status RED tests**

Assert precedence for failed, no usable cells, incomplete sensors, duplicate
warnings, and clean success. Assert missing-sensor columns are entirely false
and NaN. Assert `no_usable_grid_cells` retains `T>=1`, all-invalid arrays,
`imu_usable=False`, and null last-usable time.

- [ ] **Step 5: Implement QC/status assembly and verify GREEN**

```powershell
python -m pytest tests/imu_stage2/test_stage2_aggregation.py tests/imu_stage2/test_stage2_grid.py -q
git add src/data/imu_stage2_core.py tests/imu_stage2/test_stage2_grid.py
git commit -m "feat(imu): align actions to a masked 10 hz grid"
```

Expected: all Task 3-4 core tests pass.

---

### Task 5: Implement schema, NPZ/QC validation, and atomic action IO

**Files:**
- Create: `src/data/imu_stage2_io.py`
- Create: `tests/imu_stage2/test_stage2_io.py`

**Interfaces:**
- Produces: `build_stage2_schema()`, `load_stage2_schema()`, `write_json_atomic()`.
- Produces: `write_action_atomic()`, `load_and_validate_npz()`, `validate_existing_action()`.

- [ ] **Step 1: Add RED tests for contract/provenance separation and NPZ keys**

Assert provenance changes leave `contract_sha256` unchanged, exact NPZ keys
and dtypes reopen with `allow_pickle=False`, invalid cells remain NaN, JSON
rejects non-finite values, and a malformed NPZ fails validation.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_stage2_io.py -q
```

Expected: missing IO module.

- [ ] **Step 3: Implement schema and atomic JSON/NPZ validation**

Write JSON with the exact call below and use same-directory temporary files
plus `os.replace` for root JSON. Use uncompressed `np.savez` for actions and
verify exact key sets after reopen.

```python
json.dump(
    payload,
    handle,
    ensure_ascii=False,
    indent=2,
    allow_nan=False,
)
```

- [ ] **Step 4: Add staged-directory RED tests**

Test new write, overwrite restoration after injected install failure, no stale
files, and zero staging/backup residue. Test source-fingerprint mismatch and QC
count mismatch prevent `skipped_existing`.

- [ ] **Step 5: Implement staged action publication and verify GREEN**

The implementation must resolve and recheck staging, backup, destination, and
output root before rename, restore, or delete. It returns `WriteStatus` without
altering `DataStatus`.

```powershell
python -m pytest tests/imu_stage2/test_stage2_io.py -q
git add src/data/imu_stage2_io.py tests/imu_stage2/test_stage2_io.py
git commit -m "feat(imu): write validated stage 2 artifacts"
```

Expected: all IO tests pass and temporary paths are absent.

---

### Task 6: Build the offline Stage 2 CLI and run modes

**Files:**
- Create: `scripts/preprocess_imu_stage2.py`
- Create: `tests/imu_stage2/test_stage2_cli.py`

**Interfaces:**
- Produces: `validate_roots()`, `preflight_run_mode()`, `build_manifest()`, `main(argv) -> int`.
- CLI options: `--input-root`, `--output-root`, `--dry-run`, `--resume`, `--overwrite`.

- [ ] **Step 1: Add root-safety and fresh-mode RED tests**

Parameterize equal/ancestor/descendant roots, resolved symlink/junction overlap,
and non-empty fresh output. Assert exit 2 before logs, manifests, or action
writes.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_stage2_cli.py -q
```

Expected: CLI module missing.

- [ ] **Step 3: Implement argument and run-mode preflight**

Reject simultaneous `--resume` and `--overwrite`. Fresh mode requires empty or
missing output. Resume requires compatible schema plus full managed-tree and
fingerprint validation. Overwrite requires the same contract and never clears
the root.

- [ ] **Step 4: Add dry-run and exit-code RED tests**

Use synthetic accepted Stage 1 artifacts for success, incomplete, no-usable,
and failed actions. Assert dry-run performs transformations but leaves output
absent. Assert offline exit codes 0/1/2 and failed count fields remain blank.

- [ ] **Step 5: Implement sequential orchestration and root artifacts**

Process actions independently, emit best-effort failed QC when sample identity
exists, preserve `status` across resume skips, write `write_status` separately,
and atomically publish schema/manifest. Serialize sensor and warning fields in
fixed registry order and manifest in UTF-8 BOM.

- [ ] **Step 6: Add resume/overwrite RED tests and make them GREEN**

Cover verified skip, corrupt existing artifact, source hash mismatch, unknown
managed file, incompatible contract, action overwrite, and zero residue.

- [ ] **Step 7: Verify and commit**

```powershell
python -m pytest tests/imu_stage2/test_stage2_cli.py tests/imu_stage2/test_stage2_io.py -q
python scripts/preprocess_imu_stage2.py --help
git add scripts/preprocess_imu_stage2.py tests/imu_stage2/test_stage2_cli.py
git commit -m "feat(imu): add stage 2 offline cli"
```

Expected: focused suite passes; help lists all five options.

---

### Task 7: Generate class order and canonical training indexes

**Files:**
- Create: `scripts/build_imu_training_index.py`
- Create: `tests/imu_stage2/test_training_index.py`

**Interfaces:**
- Produces: `build_class_order()`, `build_training_index()`, `hash_training_index()`.
- Writes: `class_order.json`, `training_index.csv`, `training_index.json`.

- [ ] **Step 1: Add RED tests for label mapping and strict eligibility**

Use non-contiguous `class_id` values to prove label index is contract-derived,
not `class_id` arithmetic. Assert strict eligibility requires label,
`imu_usable`, all historical sensors, all usable sensors, and success/warning
status.

- [ ] **Step 2: Add RED tests for split semantics and hashes**

Assert selected iff split is train/validation, unselected split is blank,
train/validation users are disjoint, and swapping splits changes
`training_index_sha256` even when sample sets are unchanged. Assert a changed
class order or Stage 2 manifest digest invalidates metadata.

- [ ] **Step 3: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_training_index.py -q
```

Expected: training-index module missing.

- [ ] **Step 4: Implement canonical artifacts**

Map the approved split file's `train_users` and `val_users`. Validate every
`label_index` in `[0,num_classes)`. Hash canonical sample-ID-sorted behavior
rows `(sample_id,label_index,split,selected_for_run,
eligible_for_strict_training,stage2_npz_relpath)`. Save source manifest,
contract, split, class-order, index, and three sample-set digests.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/imu_stage2/test_training_index.py -q
git add scripts/build_imu_training_index.py tests/imu_stage2/test_training_index.py
git commit -m "feat(imu): build reproducible training indexes"
```

Expected: all index tests pass with no user overlap.

---

### Task 8: Compute leak-free 5x16 fold normalization

**Files:**
- Create: `scripts/compute_imu_normalization.py`
- Create: `tests/imu_stage2/test_normalization.py`

**Interfaces:**
- Produces: `StreamingMoments`, `compute_normalization()`, `validate_normalization_artifacts()`.
- Writes: `imu_normalization.npz`, `imu_normalization.json`.

- [ ] **Step 1: Add numerical and leakage RED tests**

Create train, validation, invalid-NaN, and padding sentinels. Assert only
selected train valid cells affect mean/std; per-sensor feature counts are
equal; population standard deviation matches NumPy float64; and a channel with
`raw_std<1e-6` receives scale 1.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_normalization.py -q
```

Expected: normalization module missing.

- [ ] **Step 3: Implement Chan/Welford accumulation**

Accumulate count, mean, M2, minimum, and maximum in float64 per sensor-feature.
Reject zero counts, non-finite results, or meaningful negative variance. Cast
saved arrays to the exact contract dtypes, record near-constant names, and bind
Stage 2 contract, fold, and exact training sample hash.

- [ ] **Step 4: Add artifact-tamper RED tests**

Change fold ID, sample hash, sensor order, Stage 2 contract, and NPZ bytes one
at a time; each load must fail before values are used.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/imu_stage2/test_normalization.py -q
git add scripts/compute_imu_normalization.py tests/imu_stage2/test_normalization.py
git commit -m "feat(imu): compute fold-only normalization"
```

Expected: exact numerical and tamper tests pass.

---

### Task 9: Add the variable-length dataset, sampler, and dynamic collate

**Files:**
- Create: `src/data/imu_stage2_dataset.py`
- Create: `tests/imu_stage2/test_stage2_dataset.py`

**Interfaces:**
- Produces: `IMUStage2Dataset`, `LengthBucketBatchSampler`, `collate_imu_stage2()`.
- Produces: `SequenceLengthSafetyError` from the contracts module.

- [ ] **Step 1: Add valid-only normalization and padding RED tests**

Assert the dataset validates NPZ/schema/normalization, standardizes only valid
cells, emits finite zeros for invalid cells, and keeps full length. Collate two
lengths and assert right padding values 0, masks false, timestamps -1, lengths
equal sequence-mask sums, and a real all-invalid time point keeps
`sequence_mask=True`.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_stage2_dataset.py -q
```

Expected: dataset module missing.

- [ ] **Step 3: Implement dataset and collate**

Return `values`, `valid_mask`, `sensor_mask`, `usable_sensor_mask`,
`timestamps_ms`, `length`, `sample_id`, and optional label. Allocate batch
arrays from the maximum real length and never consult legacy
`sequence_length: 256`.

- [ ] **Step 4: Add sampler RED tests**

Assert deterministic seed behavior, bucket-local batching, batch feature
budget, an over-budget singleton, no omissions/duplicates when `drop_last`
false, and exact declared omissions when true. Assert `T>hard_safety_limit_T`
raises the typed error without truncation.

- [ ] **Step 5: Implement sampler, verify, and commit**

```powershell
python -m pytest tests/imu_stage2/test_stage2_dataset.py -q
git add src/data/imu_stage2_dataset.py tests/imu_stage2/test_stage2_dataset.py
git commit -m "feat(imu): load variable stage 2 sequences"
```

Expected: dataset/sampler/collate tests pass.

---

### Task 10: Implement the mask-aware v1 model and invariance tests

**Files:**
- Create: `src/models/imu_stage2_tcn.py`
- Create: `tests/imu_stage2/test_stage2_model.py`
- Modify: `src/models/__init__.py`

**Interfaces:**
- Produces: `IMUStage2Classifier.forward(batch) -> {"embedding", "logits"}`.
- Consumes: collate output and `imu_modality_mask`.

- [ ] **Step 1: Add model-output and invariance RED tests**

Set `model.eval()` and test finite logits `[B,40]`. Compare logits after adding
right padding, randomizing invalid-cell placeholders, randomizing an unusable
sensor, and randomizing an unavailable-modality technical placeholder. Test a
sample alone and in different legal batch partitions. Use one explicit test
tolerance constant, `MODEL_INVARIANCE_ATOL = 1e-6`, with `rtol=0`.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_stage2_model.py -q
```

Expected: model module missing.

- [ ] **Step 3: Implement behavior-preserving mask handling**

Gate invalid raw inputs before encoding. Encode usable sensor branches, pool
each with its own `sequence_mask & valid_mask[:,:,s]`, fuse only sensors where
`usable_sensor_mask` is true, and select a packaged learned null embedding when
`imu_modality_mask` is false. Mask padded outputs after biased temporal blocks.
Return logits without applying softmax.

- [ ] **Step 4: Add deterministic decision tests**

Assert `argmax` selects the lower index for tied maxima, dropout is inactive in
eval mode, and two same-seed inference runs return identical labels.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/imu_stage2/test_stage2_model.py -q
git add src/models/imu_stage2_tcn.py src/models/__init__.py tests/imu_stage2/test_stage2_model.py
git commit -m "feat(imu): add mask-aware stage 2 classifier"
```

Expected: model shape and invariance tests pass.

---

### Task 11: Implement raw-test discovery, typed degradation, and bundle validation

**Files:**
- Create: `src/inference/__init__.py`
- Create: `src/inference/imu_stage2_pipeline.py`
- Create: `tests/imu_stage2/test_online_pipeline.py`

**Interfaces:**
- Produces: `discover_test_samples()`, `adapt_raw_imu_source()`, `load_inference_bundle()`.
- Produces: `preprocess_inference_sample()` and inference-only collate extension.

- [ ] **Step 1: Add discovery/source RED tests**

Create direct `SM_test_0001`, `SM_test_0002`, `.claude`, nested fake samples,
and one sample without IMU. Assert both legal IDs are discovered in numeric
order, missing IMU remains represented, and ignored directories are audited.

- [ ] **Step 2: Add explicit-exception RED tests**

Parameterize the seven allowlisted typed errors and assert they yield
`imu_available=False`. Parameterize `AssertionError`, `IndexError`, `KeyError`,
`MemoryError`, unknown `ValueError`, unknown `RuntimeError`, and generic
`Exception`; assert each escapes as a global failure.

- [ ] **Step 3: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_online_pipeline.py -q
```

Expected: inference module missing.

- [ ] **Step 4: Implement discovery, adaptation, and typed orchestration**

Match only direct `^SM_test_\d{4}$` directories. Build `InferenceSample` with
either a real Stage 2 result or no result. Catch each allowlisted type in an
explicit tuple; do not catch a base `Exception` around sample preprocessing.
Construct technical placeholders only inside inference collate and set length
zero, sequence mask false, timestamp -1, and modality mask false.

- [ ] **Step 5: Add bundle-tamper RED tests**

Build a temporary bundle manifest, verify success, then alter each file byte,
path, internal Stage 2 digest, normalization digest, class order, submission
contract, and training-index binding. Assert validation fails before model load.

- [ ] **Step 6: Implement bundle-first validation and commit**

```powershell
python -m pytest tests/imu_stage2/test_online_pipeline.py -q
git add src/inference tests/imu_stage2/test_online_pipeline.py
git commit -m "feat(imu): validate online stage 2 inference inputs"
```

Expected: discovery, error isolation, placeholder, and bundle tests pass.

---

### Task 12: Add submission contract, inference CLI, and atomic output

**Files:**
- Create: `scripts/infer_imu_stage2.py`
- Create: `inference.sh`
- Create: `tests/imu_stage2/test_inference_cli.py`
- Modify: `src/inference/imu_stage2_pipeline.py`

**Interfaces:**
- Produces: `derive_submission_contract(sample_submission_path: Path) -> dict[str, object]`.
- Produces: `validate_logits()`, `decode_predictions()`, `write_submission_atomic()`, `main(argv) -> int`.
- Public command: `bash inference.sh RAW_TEST_ROOT OUTPUT_CSV`.

- [ ] **Step 1: Add model-output and submission-contract RED tests**

Assert logits are finite `[B,num_classes]`, ties select the lower index, and
wrong shape/non-finite logits fail. Test `derive_submission_contract()` with a
fixture having the official adapter's sample-ID and prediction columns. Bundle
preparation must receive the organizer-provided sample-submission path through
an explicit `--sample-submission` argument; the derived contract fixes column
names, encoding, header, row order, and class representation and is then
validated without guessing.

- [ ] **Step 2: Add output-overwrite RED tests**

Assert an existing output fails before inference unless `--overwrite-output`
is explicit. Inject validation failure and prove the old file remains. Assert a
validated temporary file replaces it atomically and has one row per discovered
sample with no duplicates or extras.

- [ ] **Step 3: Verify RED**

```powershell
python -m pytest tests/imu_stage2/test_inference_cli.py -q
```

Expected: CLI and submission functions missing.

- [ ] **Step 4: Implement deterministic inference and output publication**

Call `model.eval()` under `torch.inference_mode()`, disable augmentation,
apply fixed seed/configuration, validate logits, use `argmax`, decode through
class order, and validate the complete output against the loaded submission
contract before `os.replace`. Record framework, device, deterministic setting,
seed, and batching configuration.

- [ ] **Step 5: Implement audit and intermediate-output rules**

Require a missing/empty audit root and create a unique run-ID child. Save only
the inference manifest, log, problematic QC, and summary by default. Require
`--audit-dir` with `--save-intermediates`; persist real online Stage 1 and
Stage 2 outputs under their own contracts and never persist technical
placeholders as data artifacts.

- [ ] **Step 6: Add exit-code and wrapper tests**

Assert code 0 publishes complete output even with handled unavailable IMU,
code 1 publishes nothing when a declared unavailable policy cannot predict,
and code 2 publishes nothing for bundle/global/model/output errors. Invoke the
Bash wrapper in a portable test environment and assert it forwards raw root and
output arguments without machine-specific paths.

- [ ] **Step 7: Verify and commit**

```powershell
python -m pytest tests/imu_stage2/test_inference_cli.py tests/imu_stage2/test_online_pipeline.py -q
git add scripts/infer_imu_stage2.py inference.sh src/inference/imu_stage2_pipeline.py tests/imu_stage2/test_inference_cli.py
git commit -m "feat(imu): run raw-test stage 2 inference"
```

Expected: inference and output-contract tests pass.

---

### Task 13: Prove offline/online replay equivalence and end-to-end behavior

**Files:**
- Create: `tests/imu_stage2/test_replay_equivalence.py`
- Create: `tests/imu_stage2/test_stage2_end_to_end.py`
- Create: `scripts/validate_imu_stage2_output.py`

**Interfaces:**
- Verifies all Stage 1 bridge, Stage 2, IO, training, and inference contracts.
- Produces a read-only formal validation CLI for a generated Stage 2 root.

- [ ] **Step 1: Add synthetic exact replay tests**

Generate raw CSV fixtures, run raw Stage 1 plus Stage 2, write/read the matching
Stage 1 artifact, then run artifact loader plus Stage 2. Assert exact equality
for timestamps, sensor mask, valid mask, and float32 values with NaNs, plus
status, warnings, usable sensors, grid length, and all duplicate/interpolation
counts.

- [ ] **Step 2: Verify the replay test and diagnose any mismatch**

```powershell
python -m pytest tests/imu_stage2/test_replay_equivalence.py -q
```

Expected: PASS with exact equality. Do not add a nonzero tolerance to hide a
mismatch.

- [ ] **Step 3: Add raw-test end-to-end cases**

Cover normal IMU, partial sensors, missing IMU directory, no valid Stage 1
record, no usable Stage 2 cell, safety-limit degradation, ignored `.claude`,
bad bundle, unavailable-policy failure, repeated-run byte equality, and batch
partition invariance.

- [ ] **Step 4: Implement the read-only validator**

Validate schema contract/provenance, manifest columns/status/write status,
source fingerprints, every NPZ/QC pair, failed/QC-only rows, count identities,
root temporary residue, and action/path containment. Emit JSON summary and
return 0/1/2 without modifying Stage 2 or Stage 1 data.

- [ ] **Step 5: Run the full synthetic suite and commit**

```powershell
python -m pytest tests/imu_stage2 -q
python -m pytest tests/test_preprocess_imu_stage1.py -q
python -m py_compile scripts/preprocess_imu_stage2.py scripts/build_imu_training_index.py scripts/compute_imu_normalization.py scripts/infer_imu_stage2.py scripts/validate_imu_stage2_output.py
git add tests/imu_stage2/test_replay_equivalence.py tests/imu_stage2/test_stage2_end_to_end.py scripts/validate_imu_stage2_output.py
git commit -m "test(imu): verify stage 2 end to end"
```

Expected: all Stage 1 and Stage 2 tests pass; all scripts compile.

---

### Task 14: Perform bounded real-data dry-run and final requirement audit

**Files:**
- Verify: all files created or modified by Tasks 1-13.
- Verify: `docs/superpowers/specs/2026-07-17-imu-stage2-preprocessing-design.md`.
- Verify: `docs/superpowers/plans/2026-07-17-imu-stage2-preprocessing.md`.

**Interfaces:**
- Verifies offline CLI, online replay, artifact validator, safety, and Git scope.

- [ ] **Step 1: Confirm worktree and input boundary**

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
```

Expected: branch `IMU`; only approved Stage 2 implementation/test/config work
is present. Record the current commit rather than assuming the design-doc SHA.

- [ ] **Step 2: Run all regression and focused checks fresh**

```powershell
python -m pytest tests/test_preprocess_imu_stage1.py tests/imu_stage2 -q
python -m py_compile scripts/preprocess_imu_stage1.py scripts/preprocess_imu_stage2.py scripts/build_imu_training_index.py scripts/compute_imu_normalization.py scripts/infer_imu_stage2.py scripts/validate_imu_stage2_output.py
python scripts/preprocess_imu_stage2.py --help
python scripts/infer_imu_stage2.py --help
git diff --check
```

Expected: all tests and compilation pass; both CLIs show their approved flags;
diff check is clean.

- [ ] **Step 3: Snapshot accepted Stage 1 inputs**

Hash `manifest.csv`, every `imu_merged.csv`, and every Stage 1 `qc.json` into an
audit directory outside the repository and dataset trees. Record file count,
relative path, size, and SHA-256. Abort the dry-run if snapshot creation fails.

- [ ] **Step 4: Execute the real Stage 2 dry-run only**

```powershell
python scripts/preprocess_imu_stage2.py `
  --input-root "D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU" `
  --output-root "D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU_stage2" `
  --dry-run
```

Expected: exit 0, 2,863 actions discovered, no output root or other write, and
real status/grid/duplicate/interpolation counts printed. Do not assume the
strict candidate count; record the measured value for a separately approved
formal-run validation script. Save the independently parsed console summary as
`dry_run_expected_summary.json` in the external Task 14 audit directory and
report its absolute path. The preprocessing process itself still writes
nothing.

- [ ] **Step 5: Re-hash Stage 1 and prove zero-write behavior**

Regenerate the accepted Stage 1 snapshot, compare it byte-for-byte, and assert
the proposed Stage 2 output root was not created. Any difference blocks formal
processing.

- [ ] **Step 6: Run bounded real replay samples**

Select documented representatives for clean, duplicate, incomplete, long-gap,
late/early sensor, isolated exact-hit, and varied-length behavior. Run both
entry paths in read-only mode and require exact array/QC equality.

- [ ] **Step 7: Audit every design requirement**

Map each spec heading to a passing focused test or fresh command. Explicitly
record unverified competition submission-format or final-checkpoint behavior
until the official artifacts exist; do not claim formal inference readiness
without them.

- [ ] **Step 8: Review Git scope and request formal-run authorization**

```powershell
git status --short
git diff --stat
git diff --check
```

Expected: no dataset/audit/generated artifact is tracked. Report measured
dry-run baselines, replay evidence, test counts, and remaining external model
bundle requirements. Do not run formal Stage 2 generation, training, or final
test inference without explicit authorization.

---

### Task 15: Run and validate formal Stage 2 generation after separate authorization

**Files:**
- Execute: `scripts/preprocess_imu_stage2.py`.
- Execute: `scripts/validate_imu_stage2_output.py`.
- Do not modify: accepted Stage 1 input, repository source, or tests during the formal run.

**Interfaces:**
- Consumes: the Task 14 dry-run expected-summary JSON and accepted Stage 1 root.
- Produces: the formal `new_IMU_stage2` data tree and an external audit directory.

- [ ] **Step 1: Reconfirm explicit formal-run authorization and Git state**

```powershell
Set-Location "D:\work\2026.7.14_kaggle\40class\IMU"
git branch --show-current
git rev-parse HEAD
git status --short
git diff --check
```

Expected: branch `IMU`, the reviewed implementation commit, clean tracked
worktree, and no unrelated changes. Stop if authorization or state is absent.

- [ ] **Step 2: Verify fresh output and create the external audit root**

```powershell
$InputRoot = "D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU"
$OutputRoot = "D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU_stage2"
$AuditRoot = "D:\work\2026.7.14_kaggle\imu-stage2-formal-audit-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
if (-not $env:IMU_STAGE2_DRY_RUN_EXPECTED_SUMMARY) {
    throw "Set IMU_STAGE2_DRY_RUN_EXPECTED_SUMMARY to the Task 14 summary path"
}
$DryRunExpectedSummary = (
    Resolve-Path -LiteralPath $env:IMU_STAGE2_DRY_RUN_EXPECTED_SUMMARY
).Path
```

Require `$OutputRoot` to be absent or empty and `$AuditRoot` to be new. Do not
delete a non-empty path. Snapshot every accepted Stage 1 managed artifact and
its SHA-256 into `$AuditRoot\stage1_snapshot_before.json`.

- [ ] **Step 3: Run formal Stage 2 without resume or overwrite**

```powershell
python scripts/preprocess_imu_stage2.py `
  --input-root "$InputRoot" `
  --output-root "$OutputRoot"
$Stage2ExitCode = $LASTEXITCODE
```

Expected: exit 0. On any nonzero exit, do not retry, delete output, add
`--resume`, or add `--overwrite`; retain artifacts and continue with read-only
validation.

- [ ] **Step 4: Validate every formal artifact against dry-run expectations**

```powershell
python scripts/validate_imu_stage2_output.py `
  --input-root "$InputRoot" `
  --output-root "$OutputRoot" `
  --expected-summary "$DryRunExpectedSummary" `
  --audit-output "$AuditRoot\formal_validation_summary.json"
$ValidationExitCode = $LASTEXITCODE
```

Expected: exit 0; manifest/status counts equal the frozen dry-run summary;
every tensor-bearing row passes NPZ/QC validation; failed rows follow QC-only
rules; source/contract fingerprints agree; no staging or backup remains.

- [ ] **Step 5: Prove Stage 1 inputs are unchanged**

Regenerate `$AuditRoot\stage1_snapshot_after.json` using the same canonical
relative-path, size, SHA-256 records and compare the complete arrays. On a
difference, write a precise added/removed/changed report and mark formal
validation failed without modifying either tree.

- [ ] **Step 6: Recheck Git and publish the factual formal-run report**

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
git diff --check
```

Report the two exit codes, exact output/audit roots, manifest and data-status
counts, `imu_usable` and strict-candidate counts, grid/duplicate/interpolation
statistics, NPZ/QC counts, source-integrity result, and Git state. Do not add
generated output to Git and do not begin training.
