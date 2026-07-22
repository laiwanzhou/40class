# IMU Stage 2 Preprocessing Design

## Goal

IMU Stage 2 converts the accepted Stage 1 action-long-table output into an
unstandardized, variable-length physical-time representation for training and
inference. Each usable action produces `values [T,5,16]`, `sensor_mask [5]`,
`valid_mask [T,5]`, and `timestamps_ms [T]` on a fixed 10 Hz grid. Missing
sensors, unavailable grid cells, and an unavailable IMU modality remain
explicit through masks and status fields rather than being silently filled,
trimmed, or discarded.

The same side-effect-free Stage 2 action core is used by offline training-data
generation and online test inference. The two paths intentionally have
different Stage 1 entry points and converge at `Stage1ActionData`:

```text
Offline training:
accepted new_IMU
-> Stage1ArtifactDiscovery
-> Stage1ArtifactLoader
-> Stage1ActionData
-> Stage 2 pure core
-> offline writer

Online test:
raw test root
-> TestSampleDiscovery
-> RawImuSourceAdapter
-> Stage 1 pure core
-> Stage1ActionData
-> Stage 2 pure core
-> frozen training normalization
-> model inference
```

Raw training CSV replay is a verification-only path. It compares
`raw training CSV -> Stage 1 pure core -> Stage 2 pure core` with the official
`new_IMU -> Stage1ArtifactLoader -> Stage 2 pure core` path and never replaces
or modifies the accepted Stage 1 output.

## Scope and non-goals

Stage 2 v1 includes:

- exact integer-time adaptation for both Stage 1 entry paths;
- feature-aware same-sensor duplicate aggregation;
- a global 10 Hz action grid without boundary extrapolation;
- feature-aware interpolation with a 300 ms maximum endpoint gap;
- variable-length, per-action uncompressed NPZ output and full QC;
- separate training-index, class-order, and fold-normalization artifacts;
- dynamic right padding and explicit model-mask behavior contracts;
- raw-test discovery, online preprocessing, artifact validation, and
  `inference.sh` integration;
- offline/online replay equivalence and data-safety validation.

Stage 2 v1 does not:

- modify or regenerate the accepted training `new_IMU` tree;
- require all five sensors at preprocessing or inference time;
- standardize values in the Stage 2 NPZ files;
- pad, crop, truncate, window, or silently skip persisted sequences;
- clip, winsorize, smooth, filter, or impose unverified physical ranges;
- implement training or execute the formal Stage 2 data run as part of this
  design-document change.

The accepted Stage 1 baseline remains 2,863 actions, including 2,785 actions
whose five sensors are present and 78 `incomplete_sensors` actions. Stage 2
must preserve all of them for audit. Because Stage 2 legality checks can alter
the final usable-sensor set, the strict Stage 2 candidate count must be
measured by dry-run rather than assumed to remain 2,785.

## Fixed orders and feature schema

The sensor order is fixed:

```text
LL, RL, LA, RA, C
```

The feature order is fixed:

```text
acc_x_g, acc_y_g, acc_z_g,
gyro_x_dps, gyro_y_dps, gyro_z_dps,
angle_x_deg, angle_y_deg, angle_z_deg,
mag_x_ut, mag_y_ut, mag_z_ut,
quat_0, quat_1, quat_2, quat_3
```

Every component selects columns explicitly by this schema. No loader, core,
writer, normalizer, collate function, or model may infer feature order from a
DataFrame's incidental column order.

## Core data contracts

### Test sample descriptor

Test discovery creates a descriptor for every valid sample directory whether
or not IMU exists:

```python
@dataclass(frozen=True)
class TestSampleDescriptor:
    sample_id: str
    sample_directory: Path
    source_relative_path: Path
```

### Raw IMU source

`ImuActionSource` exists only when a real, non-empty raw IMU CSV source has
been adapted:

```python
@dataclass(frozen=True)
class ImuActionSource:
    sample_id: str
    input_directory: Path
    input_csv_files: tuple[Path, ...]
    source_relative_path: Path
    class_id: int | None = None
    class_name: str | None = None
    user_id: str | None = None
    action_id: str | None = None
```

`sample_id` is non-empty and unique in its dataset. `input_csv_files` is
non-empty and uses deterministic natural order. Serialized relative paths use
POSIX separators. Optional training metadata never affects numerical
processing.

Task 11's raw-test discovery/adaptation layer owns that ordering contract. It
discovers direct IMU CSV files, removes duplicate paths, and sorts them by one
deterministic natural key before constructing `ImuActionSource`. Thus
`part2.csv` precedes `part10.csv`, rather than the lexicographic
`part10.csv, part2.csv` order, and `source_file_rank` is assigned from that
natural-order result before duplicate timestamp aggregation. Task 13 replay
fixtures must place same-timestamp records in multiple raw CSV files and prove
offline/online exact equality. Audit Fix P1 documents this later gate but does
not implement raw discovery or sorting.

### Unified Stage 1 action data

Both Stage 1 entry paths produce:

```python
@dataclass
class Stage1ActionData:
    sample_id: str
    dataframe: pd.DataFrame
    relative_time_ns: np.ndarray
    sensor_mask: np.ndarray
    source_metadata: dict[str, object]
    qc: dict[str, object]
    class_id: int | None = None
    class_name: str | None = None
    user_id: str | None = None
    action_id: str | None = None
```

`relative_time_ns` is `int64 [N]`, is non-negative, and corresponds one-to-one
with DataFrame rows. `sensor_mask` is `bool [5]` and describes whether Stage 1
contains at least one valid record for each fixed sensor. Stage 2 uses only
`relative_time_ns` for grouping, grid construction, and interpolation; it must
not use floating `relative_time_s` or rounded `relative_time_ms` as a time key.

Offline loading reads `relative_time_s` as its original decimal text and
constructs nanoseconds without passing through binary floating point:

```python
value = Decimal(relative_time_s_text)
if not value.is_finite() or value < 0:
    raise ValueError("Invalid relative time")
nanoseconds = value * Decimal("1000000000")
if nanoseconds != nanoseconds.to_integral_value():
    raise ValueError("Relative time cannot be represented exactly in nanoseconds")
relative_time_ns = int(nanoseconds)
if not (0 <= relative_time_ns <= np.iinfo(np.int64).max):
    raise OverflowError("Relative time is outside int64 range")
```

This exact conversion applies to finite decimal text of any length and must
not inherit rounding from the process-wide default `Decimal` context. Any
sub-nanosecond remainder is rejected, while extreme exponents and Decimal
arithmetic failures are normalized to the documented contract exceptions.
Task 13 starts by hardening this conversion and adding RED/GREEN coverage; it
may not proceed to replay equivalence until that coverage passes, and Task 14
may not start before the Task 13 Decimal gate is complete. Audit Fix P1 records
this mandatory timing but does not change the Decimal implementation.

Online Stage 1 instead subtracts the exact earliest valid absolute timestamp
from each exact absolute timestamp and emits the nanosecond differences
directly. This does not require a string round-trip and does not change the
accepted on-disk Stage 1 format.

For deterministic duplicate processing, `Stage1ActionData` also supplies or
can derive `_source_file_rank` and `_stage1_row_index`. The stable record key is:

```text
source_file_rank, source_row_index, stage1_row_index
```

`source_file_rank` is the position in the action's deterministically sorted
input-file list. These internal columns are not model features.

### Stage 2 action result

Every non-failed tensor result contains:

```text
values         float32 [T,5,16]
sensor_mask    bool    [5]
valid_mask     bool    [T,5]
timestamps_ms  int64   [T]
qc             mapping
status         data-status value
```

The invariants are:

```text
T >= 1
timestamps_ms[0] == 0
adjacent timestamps differ by exactly 100
values[valid_mask] is entirely finite
values[~valid_mask] is entirely NaN across all 16 features
```

`usable_sensor_mask = valid_mask.any(axis=0)` and
`imu_usable = bool(valid_mask.any())`. If `sensor_mask[s]` is false, then
`valid_mask[:,s]` contains no true value and `values[:,s,:]` is entirely NaN.
The reverse implication is not required: Stage 1 can contain a sensor whose
records produce no usable Stage 2 cell.

## Exact time, duplicate aggregation, and continuous segments

### Action time extent

Before any Stage 2 record exclusion:

```python
stage1_action_end_ns = relative_time_ns.max()
```

It describes source action duration even when the last record is later removed.
An action with no Stage 1 record fails. QC distinguishes
`stage1_action_end_ns`, `grid_end_ns`, and `last_usable_timestamp_ns`; the last
is JSON `null` and a blank manifest field when no usable cell exists.

### Record legality

All 16 features are converted to `float64` and must be finite. A quaternion
must additionally have a finite norm of at least `1e-8`. No physical clipping
or range filtering is performed. Invalid records are removed only from their
sensor/time aggregation group and are counted by reason.

### Duplicate grouping and stable order

Records are grouped by `(sensor_position, relative_time_ns)`. Original
duplicate statistics are calculated before Stage 2 legality exclusions:

```text
duplicate_group_count
duplicate_extra_record_count
duplicate_max_group_size
```

Excluded-record and aggregation-failure counts are separate. Within a group,
valid records use the stable key `source_file_rank`, `source_row_index`, then
`stage1_row_index`.

The nine acceleration, angular-velocity, and magnetic-field channels use a
`float64` arithmetic mean. Each angle uses a circular mean. If the circular
resultant length is below `1e-8`, the entire sensor timestamp is removed.
Angles are returned to `[-180,180)`.

Each valid quaternion is first normalized. The stable first quaternion is the
reference; quaternions with a negative dot product to it are sign-flipped.
Their mean is normalized, and a non-finite or below-`1e-8` mean norm removes
the entire sensor timestamp.

If exclusions leave two or more records, normal aggregation proceeds. If one
record remains, it is used after quaternion normalization. If none remain, or
any angle/quaternion group degenerates, the sensor timestamp is removed. A
timestamp failure does not remove the action.

### Continuous-segment processing

Each sensor's unique, increasing timestamp sequence is split whenever:

```python
current_time_ns - previous_time_ns > 300_000_000
```

Angle unwrap and quaternion sign continuity operate only inside a segment.
They never propagate across a gap that interpolation is forbidden to bridge.

At a segment start, quaternion sign is canonicalized by inspecting components
in schema order and making the first component whose absolute value is at
least `1e-8` positive. Later quaternions are sign-flipped when their dot
product with the preceding quaternion is negative. Angles are independently
unwrapped per channel and per segment.

## Fixed 10 Hz grid and interpolation

The global grid step is `100_000_000 ns`. The grid covers:

```text
0 through floor(stage1_action_end_ns / 100_000_000) * 100_000_000
```

including the endpoint. It is not cropped to a shared sensor interval and does
not use a reference sensor. A shorter-than-100-ms action still produces
`timestamps_ms=[0]` and `T=1`. The unrepresented tail must satisfy:

```text
0 <= stage1_action_end_ns - grid_end_ns < 100_000_000
```

For sensor `s` at grid point `g`, an exact aggregated timestamp hit is used
directly and is valid without checking neighboring gaps. Otherwise two strict
same-sensor endpoints must exist:

```text
t_left < g < t_right
t_right - t_left <= 300_000_000 ns
```

All 16 channels share the same endpoints and:

```python
alpha = (g - t_left) / (t_right - t_left)
```

The nine ordinary continuous features use linear interpolation. Angles use
linear interpolation in the segment's unwrapped space and are wrapped to
`[-180,180)` afterward. Quaternions use normalized endpoints, shortest-sign
alignment, nlerp, and normalization with the `1e-8` norm threshold.

Boundary extrapolation, forward fill, and backward fill are forbidden. A
missing endpoint, over-300-ms gap, missing sensor, non-finite result, or failed
quaternion makes the whole `(t,s)` unit invalid. All 16 features must succeed
for `valid_mask[t,s]` to be true. Stage 2 v1 has no per-feature mask.

Interpolation must not rescan the complete global grid for every continuous
segment. Each segment processes only the grid interval covered by its first
and last timestamp, and each grid cell belongs to at most one segment candidate
interval. Time and auxiliary memory must not grow as
`O(segment_count * grid_length)` or construct a corresponding two-dimensional
candidate matrix. The target is near `O(N+T)` for `N` aggregated timestamps and
`T` grid cells, allowing logarithmic `searchsorted` factors. This lookup
optimization cannot change numerical values, endpoint selection, masks,
status, QC, dtypes, shapes, or the Stage 2 schema contract/hash.

Calculations use `float64` and `int64`. Only after validation are values cast
to `float32`; finite/NaN invariants are then checked again. The pure core
returns memory objects and never writes files.

## Offline artifacts

### Layout

```text
train/new_IMU_stage2/
|- schema.json
|- manifest.csv
|- processing.log
`- <class>/<user>/<action>/
   |- imu_stage2.npz
   `- qc.json
```

The action path mirrors Stage 1. Manifest and QC paths are POSIX relative paths
whose field names identify whether they are relative to the Stage 1 root or
Stage 2 output root.

### NPZ

`imu_stage2.npz` is created with uncompressed `np.savez` and contains exactly:

```text
values, sensor_mask, valid_mask, timestamps_ms
```

It contains no object arrays and is always reopened with `allow_pickle=False`.
The reopened artifact must satisfy key, dtype, shape, grid, mask, and NaN
invariants. It stores unstandardized physical values and no padding.

### Schema contract and provenance

`schema.json` separates stable behavior from run provenance:

```json
{
  "contract": {
    "schema_version": "imu-stage2-v1",
    "stage1_contract_version": "imu-stage1-v1",
    "grid_frequency_hz": 10,
    "grid_step_ns": 100000000,
    "max_interpolation_gap_ns": 300000000,
    "hard_safety_limit_t": 10000,
    "sensor_order": ["LL", "RL", "LA", "RA", "C"],
    "feature_order": [
      "acc_x_g", "acc_y_g", "acc_z_g",
      "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
      "angle_x_deg", "angle_y_deg", "angle_z_deg",
      "mag_x_ut", "mag_y_ut", "mag_z_ut",
      "quat_0", "quat_1", "quat_2", "quat_3"
    ],
    "values_dtype": "float32",
    "sensor_mask_dtype": "bool",
    "valid_mask_dtype": "bool",
    "timestamps_dtype": "int64",
    "invalid_value": "NaN",
    "standardized": false,
    "angle_range": "[-180, 180)",
    "time_key": "relative_time_ns",
    "duplicate_timestamp_policy": "feature_aware_aggregation",
    "interpolation_policy": "feature_aware",
    "boundary_extrapolation": false,
    "sequence_storage": "variable_length_unpadded",
    "container": "uncompressed_npz"
  },
  "provenance": {
    "implementation_version": "implementation identifier",
    "generator_script": "repository-relative path",
    "git_commit": "commit identifier",
    "created_at": "ISO-8601 timestamp",
    "source_stage1_manifest": "manifest.csv",
    "source_stage1_manifest_sha256": "digest"
  },
  "contract_sha256": "digest"
}
```

`contract_sha256` hashes only canonical JSON for `contract`, with sorted keys,
UTF-8, no ASCII escaping, and separators `(',', ':')`. The digest does not hash
itself or provenance. Compatibility checks use `contract_sha256`; provenance
remains auditable without making identical contracts incompatible.

JSON files use UTF-8 without BOM and `allow_nan=False`; absent measures use
JSON `null`. `manifest.csv` uses UTF-8 BOM.

### Action source fingerprints

Every action QC and manifest row records:

```text
stage1_output_csv_sha256
stage1_qc_sha256
stage1_manifest_row_sha256
stage2_contract_sha256
```

The Stage 1 manifest-row digest uses a fixed field set and canonical
serialization. The loader reads the accepted Stage 1 manifest with
`dtype=str` and `keep_default_na=False`, requires its columns to equal the
frozen Stage 1 `MANIFEST_COLUMNS` in order, and hashes canonical JSON for the
mapping `{column: original_csv_text}` across every column in that order. This
avoids Pandas type inference and makes empty fields unambiguous. These
fingerprints are required for validated resume behavior.

### Manifest fields

Each Stage 1 action has one row containing identity, source/output relative
paths, data status, write status, errors/warnings, masks, grid statistics,
per-sensor valid counts, and duplicate/exclusion counts. The core fields are:

```text
sample_id, class_id, class_name, user_id, action_id, relative_action_path
stage1_output_csv_relpath, stage1_qc_relpath
stage2_npz_relpath, stage2_qc_relpath
status, write_status, imu_usable, error_message, warning_codes
sensor_mask, usable_sensor_mask, missing_sensors, usable_sensors
grid_length, duration_ms, stage1_action_end_ns, grid_end_ns
unrepresented_tail_ns, first_usable_timestamp_ns, last_usable_timestamp_ns
valid_cell_count, invalid_cell_count, valid_cell_ratio
all_sensor_valid_timestep_count, all_sensor_invalid_timestep_count
exact_hit_count, interpolated_count, invalid_count
ll_valid_count, rl_valid_count, la_valid_count, ra_valid_count, c_valid_count
duplicate_group_count, duplicate_extra_record_count, duplicate_max_group_size
excluded_record_count, aggregation_failed_timestamp_count
stage1_output_csv_sha256, stage1_qc_sha256
stage1_manifest_row_sha256, stage2_contract_sha256
```

Sensor-name lists use fixed sensor order and semicolons; empty lists are blank.
Warning codes use this registry order, followed by unknown extension codes in
lexicographic order:

```text
incomplete_sensors
duplicate_timestamps_aggregated
records_excluded
angle_aggregation_degenerate
quaternion_aggregation_degenerate
no_usable_grid_cells
```

For tensor-bearing actions only:

```text
valid_cell_count + invalid_cell_count == grid_length * 5
exact_hit_count + interpolated_count == valid_cell_count
invalid_count == invalid_cell_count
```

Failed rows leave uncomputed grid/count fields blank rather than zero.
`no_usable_grid_cells` has zero valid/exact/interpolated cells, `grid_length*5`
invalid cells, and `imu_usable=false`.

### QC

Each tensor-bearing action has `qc.json`. An action that enters processing and
fails must receive a best-effort failed QC and manifest row whenever its
`sample_id` is known. QC includes schema/sample identity, source fingerprints,
status/write status, warnings/errors, masks, modality usability, action/grid
extent, valid/exact/interpolated/invalid counts, per-sensor counts, and:

```text
duplicate_group_count
duplicate_extra_record_count
duplicate_max_group_size
duplicate_excluded_record_count
duplicate_aggregation_failed_timestamp_count
duplicate_groups_per_sensor
excluded_records_per_sensor
aggregation_failures_per_sensor
nonfinite_feature_record_count
invalid_quaternion_record_count
degenerate_angle_group_count
degenerate_quaternion_group_count
```

Global failures before action identity exists can end without action rows.

### Data status and write status

Data status is one of:

```text
success
success_with_warnings
incomplete_sensors
no_usable_grid_cells
failed
```

Precedence is `failed`, `no_usable_grid_cells`, `incomplete_sensors`,
`success_with_warnings`, then `success`. Ordinary duplicate aggregation is a
warning. Incomplete actions retain `incomplete_sensors` as the primary status
and carry additional warning codes separately.

Write status is independent:

```text
written
skipped_existing
qc_only
not_written
```

`imu_usable`, not `trainable`, means at least one Stage 2 grid cell is usable.
Training eligibility is defined later by a separate index.

## Fresh, resume, overwrite, dry-run, and atomic publication

Default fresh mode has neither `--resume` nor `--overwrite` and requires a
missing or empty output root. A non-empty root is rejected before action
writes.

`--resume` requires a compatible schema and performs a complete preflight of
existing managed actions. An action can be `skipped_existing` only when all
four source/contract fingerprints match, the NPZ contract passes, and QC
agrees with NPZ counts and status. A damaged action, unknown managed file,
source mismatch, or incompatible schema fails preflight before continuation.

`--overwrite` regenerates action directories under the same Stage 2 contract.
It never authorizes cross-contract overwrite or whole-root deletion; an
incompatible contract requires a new output root.

Each action is written to a unique sibling staging directory, reopened and
validated, then atomically installed. Overwrite first moves the existing action
to a verified sibling backup, installs staging, and removes backup only after
success; failure attempts restoration. Staging, backup, and final directories
must be within the same resolved output root and filesystem. No successful run
may leave `.staging-*` or `.backup-*` paths.

Root `schema.json` and `manifest.csv` are themselves written to same-directory
temporary files, reread, validated, and published with `os.replace`.
`processing.log` need not be atomically published but records whether it closed
normally.

`--dry-run` performs discovery, loading, time adaptation, transformation,
contract validation, status assignment, and aggregation, but creates no output
root, logs, manifests, schemas, action directories, staging, or backups.
With `--summary-format json`, stdout is exactly one canonical JSON summary and
contains no human log text. An external orchestrator may redirect that stdout
to an audit file without making the preprocessing process write. Formal
validation consumes this machine-readable summary and never parses human log
lines. The summary has these fixed top-level keys:

```text
summary_version
source_stage1_manifest_sha256
stage2_contract_sha256
action_count
data_status_counts
imu_usable_action_count
strict_5sensor_candidate_count
total_grid_length
valid_cell_count
invalid_cell_count
exact_hit_count
interpolated_count
all_sensor_valid_timestep_count
all_sensor_invalid_timestep_count
duplicate_group_count
duplicate_extra_record_count
duplicate_max_group_size
excluded_record_count
aggregation_failed_timestamp_count
```

`data_status_counts` contains every fixed data status, including zero counts,
in status-registry order. The summary excludes run timestamps, paths, write
statuses, and localized text so dry-run and formal data results can be compared
directly.

Offline exit codes are:

```text
0  No action failed and root artifacts passed validation.
1  Traversal completed, but at least one action has status=failed.
2  A global root, schema, discovery, resume-preflight, manifest, or system
   failure occurred.
```

## Training index and label contract

Stage 2 output remains split- and model-agnostic. Each run creates
`training_index.csv` plus `training_index.json`. The CSV includes:

```text
sample_id, class_id, class_name, user_id, action_id, label_index
stage2_npz_relpath, status, imu_usable, sensor_mask, usable_sensor_mask
eligible_for_strict_training, selected_for_run, split, exclusion_reason
```

`label_index` comes from a versioned `class_order.json`, never an assumption
about `class_id` or `class_id-1`. V1 derives the ordered class records from the
Stage 2 manifest's unique `(class_id,class_name)` pairs sorted by integer
`class_id`, verifies a one-to-one ID/name mapping, and assigns consecutive
`label_index` values by that explicit order. The number of classes is derived,
not hard-coded. The class-order digest hashes only canonical JSON for the
ordered contract records and excludes provenance and the digest field itself.
The class-order contract records version, SHA-256, and `num_classes`; every label satisfies
`0 <= label_index < num_classes`, and each class identity maps consistently.

Strict v1 eligibility requires a label, `imu_usable`, all five historical and
usable sensors, and data status `success` or `success_with_warnings`.
`selected_for_run` means the sample is actually included in train or validation:

```text
selected_for_run == (split is "train" or "validation")
```

Unselected rows have a blank split. Train and validation user sets are
disjoint. The v1 default split input is the tracked
`metadata/splits/fold_0.json`; the training-index CLI accepts an explicit
`--split-file` override, and always records the selected file's repository-
relative path and byte-level SHA-256.

The Stage 2 output root is intentionally external to the repository.
`source_stage2_manifest_path` is therefore Stage-2-root-relative and equals
`manifest.csv` in v1; it is never interpreted as repository-relative. Its
byte-level SHA-256 binds the index to the external Stage 2 root.

`training_index.json` records:

```text
source_stage2_manifest_sha256
stage2_contract_sha256
split_definition_sha256
class_order_sha256
training_index_sha256
train_sample_id_sha256
validation_sample_id_sha256
selected_sample_id_sha256
```

Sample-set hashes use sorted IDs joined with newline plus a final newline.
`training_index_sha256` hashes canonical, sample-ID-sorted rows containing at
least sample ID, label index, split, selection, strict eligibility, and Stage 2
NPZ relative path. Duplicate IDs, missing artifacts, unknown IDs, split
overlap, or hash mismatches are fatal.

## Fold-specific normalization

Normalization is computed separately for each of the 5x16 sensor-feature
combinations using exactly the selected training-side sample IDs and only
Stage 2 cells where `valid_mask` is true. Validation, test, unselected samples,
NaNs, invalid real cells, and padding never contribute. Each valid 10 Hz cell
has equal weight, so longer actions contribute more observations.

Float64 Welford or Chan accumulation computes population variance (`ddof=0`).
Artifacts contain:

```text
count               int64   [5,16]
mean                float32 [5,16]
raw_std             float32 [5,16]
applied_scale       float32 [5,16]
near_constant_mask  bool    [5,16]
minimum             float32 [5,16]
maximum             float32 [5,16]
```

All counts are positive and, in v1, equal across the 16 features of each
sensor. Values are finite, standard deviations non-negative, and minimum no
greater than maximum. Only numerically tiny negative variance may be clamped
to zero under a documented tolerance; a meaningful negative value fails.

If `raw_std < 1e-6`, `applied_scale=1.0`; otherwise scale equals raw standard
deviation. Near-constant sensor/feature names are recorded. No clipping is
performed. Angles in `[-180,180)` use ordinary z-score in v1; a future sin/cos
representation requires a new schema.

Each run writes `imu_normalization.npz` and `imu_normalization.json`. The JSON
separates a stable normalization contract from provenance and binds Stage 2
contract, fixed orders, `[5,16]`, `ddof=0`, threshold, run/fold, users, exact
training sample hash, source manifest, generator, Git commit, and creation
time. Validation and inference verify the contract digest, file digest, fold,
orders, shapes, and exact sample-set hash. They never re-estimate statistics.
`normalization_contract_sha256` hashes only canonical normalization-contract
JSON; `normalization_file_sha256` is the byte-level SHA-256 of
`imu_normalization.npz`. The inference bundle separately hashes the JSON file
itself.

Before reading any action NPZ, normalization loads the actual class-order and
split contracts, validates the complete training-index metadata, rebuilds the
expected index semantics from the actual `stage2_root/manifest.csv`, and
requires `stage2_schema == stage2_root/schema.json`. The actual manifest digest
must equal the training-index declaration and the normalization provenance.
Dataset construction repeats the Stage 2 root manifest and schema identity
checks before loading samples.

Loading order is: validate Stage 2 artifact, validate normalization artifact,
standardize only true valid cells, set invalid real cells to zero, and derive
`usable_sensor_mask`. The resulting values are entirely finite and zero at
every invalid cell.

## Dynamic collate and model mask behavior

Stage 2 persists full variable-length sequences. Collate right-pads each batch
to its longest real sequence and returns:

```text
values               float32 [B,batch_T,5,16]
valid_mask           bool    [B,batch_T,5]
sequence_mask        bool    [B,batch_T]
sensor_mask          bool    [B,5]
usable_sensor_mask   bool    [B,5]
timestamps_ms        int64   [B,batch_T]
lengths              int64   [B]
sample_id            sequence[str]
labels               optional
```

Padding is values `0`, both masks false, timestamp `-1`. Real action positions
have `sequence_mask=true`, even when every sensor is invalid there. Lengths
equal the sequence-mask sum. `usable_sensor_mask` equals valid-mask `any` over
time.

Length buckets and a `B*batch_T*5*16` feature budget may reduce padding.
Training shuffles within buckets and then shuffles batch order. No sample is
cropped or skipped; an ordinary over-budget sample forms a batch of one.
Sampler configuration records version, bucket boundaries, feature budget,
minimum/maximum batch size, seed, and `drop_last`. Each epoch verifies no
unexpected omissions or duplicates.

`hard_safety_limit_t` detects corruption rather than truncating data. V1 uses
the fixed default `10_000`, records it in the Stage 2 contract and inference
configuration, and exposes the same explicitly named CLI/config field at every
entry point. The Stage 2 core checks the computed grid length before allocating
the grid, and the dataset rechecks persisted lengths before allocation. A
mismatch between configured values is a contract error. The core/loader raises
`SequenceLengthSafetyError`. Offline Stage 2 marks that action failed and
continues; selected train/validation data stops the training run; online
inference degrades only that sample's IMU and continues.

Model implementations are architecture-neutral but must satisfy these
behaviors:

- invalid raw sensor values never act as observations;
- right padding cannot affect real-region output;
- unavailable sensors cannot affect sensor fusion;
- invalid time positions cannot affect final pooling;
- changing placeholder values under a false mask cannot change logits;
- adding right padding cannot change logits.

The packaged null embedding is learned through controlled training-time IMU
modality dropout. V1 records a non-zero dropout probability in the model
configuration; dropout applies only in training mode, while evaluation and
inference use the declared `imu_modality_mask` unchanged. A training-level
gradient test must prove that an unavailable-modality batch produces a
non-zero gradient for the null embedding.

Input gating, masked convolution, repeated block gating, mask input channels,
or masked attention are all allowed if they pass invariance tests. Padding
must be suppressed after biased temporal blocks where necessary.

For an already fused `[B,T,C]` representation, time pooling uses:

```python
temporal_valid_mask = sequence_mask & valid_mask.any(dim=2)
```

For independent sensor branches, each sensor pools with
`sequence_mask.unsqueeze(-1) & valid_mask`, then sensor fusion uses
`usable_sensor_mask`. A single valid sensor must never make another invalid
sensor branch participate in pooling.

Checkpoints bind Stage 2 contract, training-index, normalization-contract,
normalization-file, class-order, and submission-contract hashes.

## Raw-test discovery and online inference

### Discovery and source adaptation

`TestSampleDiscovery` examines only direct children of the test root and
accepts names matching `^SM_test_\d{4}$`. Every matching directory becomes a
descriptor regardless of IMU presence. IDs are unique and naturally sorted.
Non-samples such as `.claude` are ignored and audited. The observed official
count of 405 is an acceptance observation, not a hard-coded runtime condition.

`RawImuSourceAdapter` then checks `<sample>/IMU`, directory type, recognizable
CSV presence, and deterministic order. A missing or unusable source produces
an unavailable-modality reason but never removes the sample.

### Inference sample and missing-modality batching

The orchestrator uses:

```text
InferenceSample
|- sample_id
|- imu_result: Stage2ActionResult | None
|- imu_available: bool
`- modality_mask
```

The model package declares one of two mechanisms. It can encode only available
IMU samples and scatter embeddings back into the full batch, using a packaged
null embedding for unavailable samples. Alternatively, inference-only collate
can create a technical placeholder:

```text
values=0, valid_mask=false, sensor_mask=false,
usable_sensor_mask=false, sequence_mask=false,
timestamps_ms=-1, lengths=0, imu_modality_mask=false
```

This placeholder is not a Stage 2 result, does not satisfy real-action `T>=1`,
and is never persisted as `imu_stage2.npz`. Its contents must not affect logits
when the modality mask is false.

### Explicit degradable-error allowlist

Only these typed sample errors may degrade IMU availability online:

```text
MissingImuDirectoryError
ImuPathNotDirectoryError
NoRecognizableImuCsvError
NoValidStage1RecordsError
Stage1DataValidationError
NoUsableGridCellsError
SequenceLengthSafetyError
```

Each carries `error_code`, `failure_stage`, `sample_id`, and `safe_message`.
The orchestrator catches these types explicitly. `AssertionError`, `IndexError`,
`KeyError`, `MemoryError`, unknown `ValueError`/`RuntimeError`, generic
`Exception`, invariant failures, and model-forward failures are global errors.
No broad exception handler may disguise a code defect as missing input.

### Inference bundle

`inference_bundle_manifest.json` records bundle-root-relative POSIX paths and SHA-256
for checkpoint, model config, Stage 2 schema, normalization NPZ and JSON,
class-order JSON, submission-contract JSON, and inference config. Startup first
verifies every file and digest, then validates internal hash bindings, and only
then loads the model. This prevents damaged, replaced, or mixed-fold bundles.
A dedicated bundle-builder CLI creates this manifest from explicit input paths,
including an organizer-provided sample-submission file used to derive the
submission contract. The inference CLI only consumes a completed bundle and
never derives or rewrites bundle contracts at runtime.

The model output contract is fixed to finite logits `[B,num_classes]`.
Prediction is `argmax(dim=1)`; an exact tie selects the lowest index. Other
output types require a new model-output contract and are never guessed.

`submission_contract.json` defines version, columns, sample-ID and prediction
columns, encoding, header presence, row order, and prediction representation.
Its exact official values are part of the competition adapter and its digest
is bound by both checkpoint and bundle. Output validation follows this
contract rather than inferring CSV shape from class order.

### Determinism

Inference uses `model.eval()`, `torch.inference_mode()`, no training
augmentation, no random crop/sample, fixed required seeds, deterministic
sample order, and recorded batching configuration. Audit records seed,
framework version, device, eval mode, deterministic-algorithm setting, and
batching configuration.

Two same-environment runs must produce identical sample order, label indices,
and submission bytes. Batch-partition tests compare a sample alone, in another
batch, and under different legal batch sizes; logits or final decisions must
meet a predefined model-test tolerance. These tests catch active dropout,
BatchNorm batch-statistics use, and padding dependence.

### `inference.sh`

The public interface is:

```bash
bash inference.sh /path/to/raw_test_root /path/to/output.csv
```

It performs discovery, online Stage 1 and Stage 2, frozen normalization,
dynamic batching, model inference, class decoding, and submission publication.
It uses packaged relative paths, does not download artifacts, does not write
the test root, and does not require generated test intermediates.

The model bundle declares a deterministic `imu_unavailable_policy`, such as
modality-mask fusion or a packaged deterministic fallback. Every discovered
sample must receive a valid prediction. If the bundle cannot handle an
encountered unavailable modality, the run cannot publish output.

The default output path must not exist. `--overwrite-output` explicitly allows
replacement; the old file remains until a same-directory temporary output has
been reread and fully validated, then `os.replace` publishes it.

`--audit-dir` must identify a missing or empty directory; repeated runs should
create unique `run_id` children and never remove unknown files.
`--save-intermediates` requires an audit directory. Online Stage 1
intermediates follow the frozen Stage 1 contract, online Stage 2 NPZ/QC follows
the Stage 2 contract, and inference-only no-modality placeholders are never
saved as Stage 1 or Stage 2 artifacts. A source-missing sample receives only
sample-level failure QC.

Without full intermediates, audit can store `inference_manifest.csv`,
`processing.log`, `problematic_sample_qc.json`, and
`inference_summary.json`. The manifest records every test sample, source and
stage status, IMU availability, masks, failure reason, warnings, sequence
length, and prediction.

Online exit codes are:

```text
0  Every discovered sample has a validated prediction and the final file was
   atomically published. Declared handling of unavailable IMU is allowed.
1  Traversal began, but at least one sample could not receive a prediction
   under the declared policy. No final output is published.
2  A global/system/contract/unclassified/model/output error occurred. No final
   output is published.
```

Exit code 1 is expected zero times in formal acceptance because a valid bundle
must cover unavailable IMU.

## Safety

All input trees are read-only. Existing real paths, symlinks, junctions, and
reparse points are resolved sufficiently to reject equal or overlapping input
and output roots, path traversal, links outside the action/sample root, and
cleanup outside the resolved output root. Staging, backup, and destination
must share the same output root and filesystem; containment is rechecked before
delete or restore. No non-empty root or unknown file is automatically removed.

NPZ always loads with `allow_pickle=False`; shapes, dtypes, and length limits
are validated before large allocation. Formal runs hash input CSV or accepted
Stage 1 artifacts before and after processing and require equality. Git status
is recorded before and after data runs; generated data and audits are not added
to Git.

## Verification and acceptance

### Offline/online replay

Representative real training actions cover normal input, duplicates,
incomplete sensors, late start/early stop, over-300-ms gaps, isolated exact
hits, and varied lengths. Official-artifact and raw-replay paths must have
exactly equal timestamps, masks, and float32 values including NaNs, plus equal
data status, warnings, usable sensors, duplicate/exclusion statistics,
exact/interpolated/invalid counts, and grid length. Only paths, creation times,
and explicit provenance may differ. A mismatch is diagnosed rather than
hidden by widening numerical tolerance.

### Test levels

Unit tests cover exact decimal nanoseconds and sub-nanosecond rejection,
stable rank ordering, duplicate aggregation, segment boundaries, quaternion
canonicalization/nlerp, angle unwrap/wrap, grid endpoints, exact hits, the
300-ms threshold, no extrapolation, masks/NaNs, status/write status, hashes,
fresh/resume/overwrite, class order, and index hashing.

Property tests cover padding invariance, invalid-value invariance,
unavailable-sensor invariance, unavailable-modality-placeholder invariance,
repeated-run determinism, and batch-partition invariance.

Integration tests cover accepted Stage 1 loading, raw online Stage 1, dry-run
zero writes, per-action failure isolation, typed online degradation, resume
fingerprints, staged publication/recovery, dynamic collate without cropping,
normalization leakage prevention, missing IMU sample retention, and bundle
validation.

End-to-end tests run raw test structure through `inference.sh` with normal,
partial-sensor, missing-IMU, no-valid-Stage-1, no-usable-Stage-2, safety-limit,
ignored-directory, and bad-bundle cases.

### Formal offline acceptance

Before formal generation: snapshot accepted Stage 1 inputs, run a zero-write
dry-run with exit 0, freeze measured status/count expectations, verify fresh
output conditions, and record Git state. Afterward: validate every NPZ/QC and
root artifact, status/write status and fingerprints, absence of staging/backup,
unchanged input hashes, and no unintended tracked Git changes.

### Formal online acceptance

The final path starts from raw test data with no prepared test intermediates,
discovers every legal sample including missing-IMU samples, verifies bundle and
all contracts, isolates only allowlisted sample errors, produces exactly one
valid prediction per sample, atomically publishes a submission conforming to
the submission contract, preserves input hashes, and uses neither network nor
machine-specific absolute paths.

## Completion boundary

Stage 2 v1 implementation is complete only when it provides the Stage 1
artifact loader, online Stage 1 pure core, shared Stage 2 pure core, offline
writer, NPZ/schema/manifest/QC, training index and class-order binding,
fold-specific normalization, dynamic collate and mask-invariance tests,
offline/online replay equivalence, raw-test discovery/adaptation, validated
inference bundle, and end-to-end `inference.sh` path.

This design-document task does not implement or run those components.
