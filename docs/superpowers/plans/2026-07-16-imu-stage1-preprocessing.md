# IMU Stage 1 Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe, traceable CLI that merges each Small-Model-Track IMU action into a globally timed CSV with QC, rejected-row audit data, and a global manifest.

**Architecture:** Discovery, structural CSV inspection, DataFrame validation, and action processing return typed in-memory results without writing. A single action writer stages and swaps complete output directories, while the CLI owns root safety, logging, manifest generation, aggregate reporting, and exit codes.

**Tech Stack:** Python 3.12, standard-library `argparse`, `csv`, `dataclasses`, `json`, `logging`, `pathlib`, `re`, `shutil`, `tempfile`; Pandas 2.2; pytest 7.4.

## Global Constraints

- Work only in `D:\work\2026.7.14_kaggle\40class`; never modify source files under the dataset tree.
- Default input is `D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\IMU`.
- Default output is `D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU`.
- Read CSV as UTF-8 with optional BOM via `encoding="utf-8-sig"`; never fall back to GBK.
- `--dry-run` performs zero writes: no directories, logs, manifest, temporary files, or action files.
- Reject equal or ancestor/descendant input-output roots before file logging is configured.
- Do not interpolate, fill, filter, smooth, normalize, standardize, resample, pad, truncate, or deduplicate.
- Sort and detect duplicates using exact parsed timestamps, never rounded milliseconds.
- Required row data is time, device name, and 16 retained IMU feature values. Temperature, version, and battery are optional discarded metadata.
- Continue after action failures. Exit `0` for no failed actions, `1` when actions fail, and `2` for a global fatal error.
- Do not commit or push during execution unless the user explicitly authorizes Git history changes.

## File structure

- Create/complete `scripts/preprocess_imu_stage1.py`: constants, typed result records, five processing interfaces, writer, manifest, CLI, and logging.
- Create/complete `tests/test_preprocess_imu_stage1.py`: temporary-tree behavior tests through public Python interfaces and `main(argv) -> int`.
- Keep `docs/superpowers/specs/2026-07-16-imu-stage1-preprocessing-design.md` as the approved behavior contract.

The repository uses standalone scripts rather than a package for this workflow, so a single deployable script is preferred over introducing a new package hierarchy. The script must keep orchestration in `main()` short by delegating to the named functions.

---

### Task 1: Lock discovery and typed boundaries

**Files:**
- Modify: `scripts/preprocess_imu_stage1.py`
- Modify: `tests/test_preprocess_imu_stage1.py`

**Interfaces:**
- Produces: `ActionDescriptor`, `RejectedRow`, `FileError`, `CsvReadResult`, `ValidatedCsvResult`, `ActionResult`, `WriteResult`.
- Produces: `discover_action_directories(input_root: Path) -> list[ActionDescriptor]`.
- Preserves: `parse_sensor_position(device_name: object) -> str | None` and `SENSOR_ORDER`.

- [ ] **Step 1: Add a failing integer-class discovery test**

Replace direct function imports in the test module with
`import scripts.preprocess_imu_stage1 as imu` so the already-observed global
time RED checkpoint does not prevent collection of earlier focused nodes. Use
`imu.parse_sensor_position` in the mapping test and add:

```python
def test_action_discovery_sorts_class_ids_as_integers(tmp_path: Path) -> None:
    input_root = tmp_path / "IMU"
    for class_name in ("10_Ten", "2_Two", "1_One"):
        action = input_root / class_name / "user1" / "1-1-1"
        action.mkdir(parents=True)
        (action / "data.csv").write_text("时间,设备名称\n", encoding="utf-8")

    actions = imu.discover_action_directories(input_root)

    assert [action.class_id for action in actions] == [1, 2, 10]
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/test_preprocess_imu_stage1.py::test_action_discovery_sorts_class_ids_as_integers -q
```

Expected: FAIL because `discover_action_directories` is not defined.

- [ ] **Step 3: Add typed records, constants, natural sorting, and discovery**

Add imports and records with these exact fields:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ActionDescriptor:
    class_id: int
    class_name: str
    user_id: str
    action_id: str
    input_directory: Path
    relative_action_path: Path
    input_csv_files: tuple[Path, ...]


@dataclass(frozen=True)
class RejectedRow:
    source_file: str
    source_line_number: int | None
    source_row_index: int | None
    reject_stage: str
    reject_reason: str
    raw_row: str


@dataclass(frozen=True)
class FileError:
    source_file: str
    error_type: str
    source_line_number: int | None
    message: str


@dataclass
class CsvReadResult:
    source_file: str
    dataframe: pd.DataFrame
    rejected_rows: list[RejectedRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    file_errors: list[FileError] = field(default_factory=list)
    total_input_rows: int = 0


@dataclass
class ValidatedCsvResult:
    dataframe: pd.DataFrame
    rejected_rows: list[RejectedRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unknown_sensor_rows: int = 0


@dataclass
class ActionResult:
    descriptor: ActionDescriptor
    status: str
    merged: pd.DataFrame
    rejected: pd.DataFrame
    qc: dict[str, Any]
    manifest_row: dict[str, Any]


@dataclass(frozen=True)
class WriteResult:
    written: bool
    output_directory: Path
    error_message: str = ""
```

Implement natural sorting and discovery:

```python
def natural_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def discover_action_directories(input_root: Path) -> list[ActionDescriptor]:
    actions: list[ActionDescriptor] = []
    for directory in input_root.rglob("*"):
        if not directory.is_dir():
            continue
        csv_files = tuple(
            sorted(
                (path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() == ".csv"),
                key=lambda path: natural_key(path.name),
            )
        )
        if not csv_files:
            continue
        relative = directory.relative_to(input_root)
        if len(relative.parts) < 3:
            raise ValueError(f"Invalid action path: {relative.as_posix()}")
        class_text, class_name = relative.parts[0].split("_", 1)
        actions.append(
            ActionDescriptor(
                class_id=int(class_text),
                class_name=class_name,
                user_id=relative.parts[1],
                action_id=relative.parts[-1],
                input_directory=directory,
                relative_action_path=relative,
                input_csv_files=csv_files,
            )
        )
    return sorted(
        actions,
        key=lambda item: (
            item.class_id,
            natural_key(item.user_id),
            natural_key(item.action_id),
            item.relative_action_path.as_posix().casefold(),
        ),
    )
```

- [ ] **Step 4: Verify GREEN for the Task 1 nodes**

Run:

```powershell
python -m pytest tests/test_preprocess_imu_stage1.py::test_device_names_map_to_sensor_positions tests/test_preprocess_imu_stage1.py::test_action_discovery_sorts_class_ids_as_integers -q
```

Expected: discovery and device mapping nodes pass. The already-observed
global-time tracer remains the explicit RED checkpoint reserved for Task 4.

- [ ] **Step 5: Inspect the checkpoint**

Run `git diff -- scripts/preprocess_imu_stage1.py tests/test_preprocess_imu_stage1.py` and confirm only typed boundaries and discovery were added. Do not commit without user authorization.

---

### Task 2: Structural CSV inspection with exact trace indices

**Files:**
- Modify: `scripts/preprocess_imu_stage1.py`
- Modify: `tests/test_preprocess_imu_stage1.py`

**Interfaces:**
- Consumes: `RejectedRow`, `FileError`, `CsvReadResult`.
- Produces: `normalize_column_name(name: object) -> str`.
- Produces: `read_csv_robust(path: Path) -> CsvReadResult`.

- [ ] **Step 1: Add one failing recoverable-structure test**

Create a UTF-8 file containing a valid header, valid data, a blank record, a BOM/space-normalized repeated header, a short record, and another valid record. Assert:

```python
result = read_csv_robust(source)
assert result.total_input_rows == 5
assert result.dataframe["source_row_index"].tolist() == [1, 5]
assert [row.reject_reason for row in result.rejected_rows] == [
    "blank_row",
    "repeated_header",
    "field_count_mismatch",
]
assert [row.source_row_index for row in result.rejected_rows] == [2, 3, 4]
```

- [ ] **Step 2: Verify RED**

Name the test
`test_structural_inspection_rejects_rows_and_preserves_indices` and run:

```powershell
python -m pytest tests/test_preprocess_imu_stage1.py::test_structural_inspection_rejects_rows_and_preserves_indices -q
```

Expected: FAIL because `read_csv_robust` is not defined.

- [ ] **Step 3: Implement normalized headers and one-pass structural parsing**

Define the canonical source mapping for time, device name, 16 features, and optional metadata. Normalize with Unicode-aware removal of BOM/control/format characters, trim whitespace, translate `（ ） μ µ` to `( ) u u`, and casefold for comparison. Implement `csv.reader(handle, strict=True)` so that:

```python
data_row_index = 0
record_start_line = reader.line_num + 1
row = next(reader)
data_row_index += 1
```

For every record after the header, increment before classification. Store
`source_line_number=record_start_line`, `source_row_index=data_row_index`,
`source_file=path.name`, and `raw_row=json.dumps(row, ensure_ascii=False)`.
Append only structurally valid records to a list, then construct exactly once:

```python
frame = pd.DataFrame(valid_records, columns=normalized_header)
frame["source_file"] = path.name
frame["source_line_number"] = valid_line_numbers
frame["source_row_index"] = valid_data_indices
```

Catch `UnicodeDecodeError` as `utf8_decode_error` and `csv.Error` as
`csv_syntax_error`. Record `reader.line_num` when available. Treat missing
required columns and an empty header as fatal file errors; optional metadata
columns add warnings only.

- [ ] **Step 4: Verify GREEN**

Run the recoverable-structure test and the mapping/discovery tests. Expected: PASS.

- [ ] **Step 5: Add and run a multiline-record test**

Write a quoted field containing a newline, followed by another record. Assert
the quoted record's `source_line_number` is its first physical line and the
following record starts after the quoted record's final physical line. Watch
the test fail, fix start-line tracking, then rerun to PASS.

- [ ] **Step 6: Add and run a fatal syntax test**

Write an unclosed quoted record. Assert one `FileError` has
`error_type == "csv_syntax_error"` and a detected line number, then perform the
RED-GREEN cycle.

- [ ] **Step 7: Inspect the checkpoint**

Run the focused structural tests and inspect `git diff`. Do not commit without user authorization.

---

### Task 3: DataFrame content validation

**Files:**
- Modify: `scripts/preprocess_imu_stage1.py`
- Modify: `tests/test_preprocess_imu_stage1.py`

**Interfaces:**
- Consumes: `CsvReadResult`, canonical source-column mapping.
- Produces: `validate_dataframe(result: CsvReadResult) -> ValidatedCsvResult`.

- [ ] **Step 1: Add a failing combined-content-reason test**

Build a `CsvReadResult` with rows containing invalid time, unknown device, and
invalid retained feature values. Assert the rejected audit row has
`reject_stage == "content"` and an ordered semicolon reason string such as:

```text
invalid_time;unknown_sensor;non_numeric_acc_x_g
```

Also assert `unknown_sensor_rows` counts every row whose device prefix is unknown.

- [ ] **Step 2: Verify RED**

Run the new test node. Expected: FAIL because `validate_dataframe` is absent.

- [ ] **Step 3: Implement vectorized Pandas conversion and rejection**

Use these exact retained output feature mappings:

```python
FEATURE_COLUMN_MAP = {
    "加速度X(g)": "acc_x_g", "加速度Y(g)": "acc_y_g", "加速度Z(g)": "acc_z_g",
    "角速度X(°/s)": "gyro_x_dps", "角速度Y(°/s)": "gyro_y_dps", "角速度Z(°/s)": "gyro_z_dps",
    "角度X(°)": "angle_x_deg", "角度Y(°)": "angle_y_deg", "角度Z(°)": "angle_z_deg",
    "磁场X(uT)": "mag_x_ut", "磁场Y(uT)": "mag_y_ut", "磁场Z(uT)": "mag_z_ut",
    "四元数0()": "quat_0", "四元数1()": "quat_1", "四元数2()": "quat_2", "四元数3()": "quat_3",
}
```

Create `absolute_time` with `pd.to_datetime(..., errors="coerce")`, convert
each retained feature with `pd.to_numeric(..., errors="coerce")`, and map
device names using `parse_sensor_position`. Build reasons in deterministic
time, sensor, then feature-column order. Preserve raw source values long enough
to JSON-encode content-rejected rows.

- [ ] **Step 4: Verify GREEN**

Run the content validation test and all Task 1-2 tests. Expected: PASS.

- [ ] **Step 5: Add a failing optional-metadata test**

Create a valid IMU row with invalid temperature and battery strings. Assert it
remains in `ValidatedCsvResult.dataframe`, warnings mention both invalid
metadata values, and no rejected row is added.

- [ ] **Step 6: Implement optional metadata warning counts and verify GREEN**

Convert present temperature and battery columns with `pd.to_numeric`; count
coerced values and append deterministic warnings. Never include temperature,
version, or battery in the validated feature frame. Run the focused test to PASS.

- [ ] **Step 7: Inspect the checkpoint**

Run all Task 1-3 tests and inspect the diff. Do not commit without user authorization.

---

### Task 4: Action processing, exact chronology, QC, and manifest row

**Files:**
- Modify: `scripts/preprocess_imu_stage1.py`
- Modify: `tests/test_preprocess_imu_stage1.py`

**Interfaces:**
- Consumes: `read_csv_robust()`, `validate_dataframe()`, `ActionDescriptor`.
- Produces: `process_action_directory(action_dir: Path, input_root: Path, output_root: Path) -> ActionResult` for test convenience, internally constructing an `ActionDescriptor` and delegating to `process_action(descriptor: ActionDescriptor) -> ActionResult`.

- [ ] **Step 1: Use the existing failing global-time tracer as RED**

Run:

```powershell
python -m pytest tests/test_preprocess_imu_stage1.py::test_all_sensors_share_the_actions_earliest_time_zero -q
```

Expected: FAIL because `process_action_directory` is absent.

- [ ] **Step 2: Implement minimal complete action processing**

Define:

```python
OUTPUT_COLUMNS = [
    "relative_time_s", "relative_time_ms", "sensor_position",
    "acc_x_g", "acc_y_g", "acc_z_g",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "angle_x_deg", "angle_y_deg", "angle_z_deg",
    "mag_x_ut", "mag_y_ut", "mag_z_ut",
    "quat_0", "quat_1", "quat_2", "quat_3",
    "source_file", "source_row_index",
]
```

For each action CSV, collect structural and content results. If any
`file_errors` exist, return `failed` with an empty merged frame. Otherwise
concatenate valid frames, compute:

```python
action_start_time = merged["absolute_time"].min()
action_end_time = merged["absolute_time"].max()
relative_time_s = (merged["absolute_time"] - action_start_time).dt.total_seconds()
relative_time_ms = (relative_time_s * 1000).round().astype("int64")
```

Sort with `kind="mergesort"` by exact `absolute_time`, `sensor_order`,
`source_file`, `source_row_index`. Count duplicates with
`duplicated(["sensor_position", "absolute_time"], keep="first")` grouped per
sensor. Drop temporary columns and select `OUTPUT_COLUMNS` in exact order.

- [ ] **Step 3: Verify the global-time tracer is GREEN**

Run the existing node. Expected: PASS with `[0, 150]` and `LL, RL` order.

- [ ] **Step 4: Add exact sub-millisecond sort and duplicate tests**

Use one sensor with timestamps `.1004`, `.1002`, and an exact repeat of `.1002`.
Assert output order follows `.1002`, `.1002`, `.1004`; all rounded milliseconds
may match, but duplicate count is exactly one. First run RED, implement/fix,
then run GREEN.

- [ ] **Step 5: Add cross-sensor equal-timestamp test**

Use LL and RL at the exact same timestamp. Assert both rows remain and both
per-sensor duplicate counts are zero. Perform RED-GREEN.

- [ ] **Step 6: Add incomplete-sensor and output-schema tests**

Process an action containing only LL and RL. Assert status is
`incomplete_sensors`, missing sensors are `LA`, `RA`, `C`, and merged columns
equal `OUTPUT_COLUMNS`. Perform RED-GREEN.

- [ ] **Step 7: Build complete QC and manifest dictionaries**

Populate QC in the exact design-specified order and types, including start/end
ISO timestamps, structural/content counts, per-sensor row/duplicate objects,
warnings, and file errors. Populate the exact 28 manifest columns. Serialize
relative paths with `.as_posix()` and sensor lists with semicolons in fixed
sensor order.

- [ ] **Step 8: Add an input-byte immutability test**

Hash every source CSV before and after `process_action_directory`; assert hashes,
paths, and directory entries are identical. Perform RED-GREEN.

- [ ] **Step 9: Verify all processing tests**

Run `python -m pytest tests/test_preprocess_imu_stage1.py -q`. Expected: all tests through Task 4 pass with no warnings.

---

### Task 5: Staged writer, overwrite cleanup, and skip behavior

**Files:**
- Modify: `scripts/preprocess_imu_stage1.py`
- Modify: `tests/test_preprocess_imu_stage1.py`

**Interfaces:**
- Consumes: `ActionResult`.
- Produces: `write_action_result(result: ActionResult, output_root: Path, overwrite: bool) -> WriteResult`.

- [ ] **Step 1: Add a failing normal-write test**

Write a successful in-memory result and assert the mirrored directory contains
`imu_merged.csv` and `qc.json`; add one rejected row and assert it also contains
`rejected_rows.csv`. Read merged CSV with Pandas and assert exact columns and
values. Assert the first three bytes are the UTF-8 BOM `b"\xef\xbb\xbf"`.

- [ ] **Step 2: Verify RED**

Run the writer test. Expected: FAIL because `write_action_result` is absent.

- [ ] **Step 3: Implement staged action-directory installation**

Use a unique sibling staging directory under the resolved output root. Write
managed artifacts there, validate its filename set against status/rejection
state, and then install it. For overwrite, rename the old action directory to
a unique sibling backup, rename staging into place, and remove the verified
backup with `shutil.rmtree`. On installation failure, restore backup if the
destination is absent. Verify staging, backup, and destination resolve below
the output root before any cleanup.

- [ ] **Step 4: Verify GREEN**

Run the normal writer test. Expected: PASS.

- [ ] **Step 5: Add stale merged-output overwrite test**

First write success, then overwrite with a failed result. Assert final managed
files are only `qc.json` plus `rejected_rows.csv` when the failed result has
rejections; assert old `imu_merged.csv` is absent. Perform RED-GREEN.

- [ ] **Step 6: Add stale rejected-output overwrite test**

First write success with rejected rows, then overwrite clean success. Assert
`rejected_rows.csv` is absent and only merged CSV plus QC remain. Perform RED-GREEN.

- [ ] **Step 7: Add skip immutability test**

Snapshot names and bytes in an existing action output, call writer with
`overwrite=False`, assert `written is False`, and compare the exact snapshot.
Perform RED-GREEN.

- [ ] **Step 8: Verify writer tests and inspect the checkpoint**

Run all tests. Inspect temporary directories are absent after success. Do not commit without user authorization.

---

### Task 6: Root safety, dry-run, CLI reporting, and manifest

**Files:**
- Modify: `scripts/preprocess_imu_stage1.py`
- Modify: `tests/test_preprocess_imu_stage1.py`

**Interfaces:**
- Produces: `validate_roots(input_root: Path, output_root: Path) -> None`.
- Produces: `build_manifest(results: list[ActionResult]) -> pd.DataFrame`.
- Produces: `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Add a failing root-overlap zero-write test**

Parameterize equal roots, output below input, and input below output. Invoke
`main([...])`; assert exit code `2` and assert no output directory,
`processing.log`, manifest, or temporary path was created.

- [ ] **Step 2: Verify RED**

Run the node. Expected: FAIL because CLI root validation is absent.

- [ ] **Step 3: Implement argument parsing and pre-logging root safety**

Define exact arguments and defaults. Resolve both paths with `strict=False`.
Reject equality or either result appearing in the other's `.parents`. Validate
input existence/directory before `output_root.mkdir()` or file-handler setup.

- [ ] **Step 4: Verify root safety GREEN**

Run the root test. Expected: PASS and zero filesystem writes.

- [ ] **Step 5: Add a failing dry-run integration test**

Build one complete synthetic action, invoke `main` with `--dry-run`, assert
exit `0`, capture the Chinese summary labels, and assert output root does not
exist. Add a failed synthetic action and assert dry-run scans all actions,
still writes nothing, and returns `1`.

- [ ] **Step 6: Implement dry-run orchestration and console-only logging**

In dry-run, discover and process each action but never call writer or root
manifest/log setup. Keep only compact ActionResult metadata after aggregating
counts. Print all required totals and LL/RL/LA/RA/C missing counts.

- [ ] **Step 7: Verify dry-run GREEN**

Run the dry-run tests. Expected: PASS with no output tree.

- [ ] **Step 8: Add a normal-run manifest/log integration test**

Use synthetic success, incomplete, failed, and pre-existing skipped actions.
Assert `processing.log` and `manifest.csv` exist, manifest columns exactly
match the 28-column specification, rows sort by class IDs `1, 2, 10`, and the
failed action is present. Assert failed action QC exists without merged output.

- [ ] **Step 9: Implement normal orchestration and manifest writing**

Configure one console handler and one UTF-8 file handler after root validation.
Process actions sequentially, write each result, release large frames, build
and integer/natural-sort manifest, and write it with `encoding="utf-8-sig"`,
`index=False`. For skipped actions, load existing QC when readable, report
`skipped_existing`, and do not invoke the action writer.

- [ ] **Step 10: Implement and test exit codes**

Return `0` when no action is failed, `1` after completing all actions when any
action failed, and `2` for root/discovery/logging/manifest fatal errors. Catch
global exceptions only at the CLI boundary and log a clear message when logging
is available. Verify all three paths in focused tests.

- [ ] **Step 11: Add CLI entry point and help test**

End the script with:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

Run `python scripts/preprocess_imu_stage1.py --help` and assert the four options appear.

- [ ] **Step 12: Verify all Task 6 tests**

Run the complete test module. Expected: all tests pass with no unhandled warnings.

---

### Task 7: Final verification and requirement audit

**Files:**
- Verify: `scripts/preprocess_imu_stage1.py`
- Verify: `tests/test_preprocess_imu_stage1.py`
- Verify: `docs/superpowers/specs/2026-07-16-imu-stage1-preprocessing-design.md`

**Interfaces:**
- Verifies all public and CLI interfaces from Tasks 1-6.

- [ ] **Step 1: Run the focused suite fresh**

```powershell
python -m pytest tests/test_preprocess_imu_stage1.py -q
```

Expected: exit `0`, all tests passed, no warnings.

- [ ] **Step 2: Compile and inspect CLI help**

```powershell
python -m py_compile scripts/preprocess_imu_stage1.py
python scripts/preprocess_imu_stage1.py --help
```

Expected: compile exit `0`; help lists `--input-root`, `--output-root`, `--overwrite`, `--dry-run`.

- [ ] **Step 3: Run a fresh synthetic dry-run proof**

Use the pytest dry-run integration node with `-vv`; confirm the asserted output root remains absent and exit-code paths pass.

- [ ] **Step 4: Run a fresh synthetic normal-run proof**

Use the pytest manifest/log integration node with `-vv`; confirm expected action files, failed QC, manifest ordering, and no writes to input.

- [ ] **Step 5: Audit every requested output field and summary counter**

Compare `OUTPUT_COLUMNS`, QC key order, manifest key order, status precedence,
sensor order, summary labels, and exit codes directly against the approved
design spec. Record any unverified item rather than implying coverage.

- [ ] **Step 6: Inspect source safety and worktree scope**

```powershell
git status --short
git diff -- scripts/preprocess_imu_stage1.py tests/test_preprocess_imu_stage1.py docs/superpowers/specs/2026-07-16-imu-stage1-preprocessing-design.md docs/superpowers/plans/2026-07-16-imu-stage1-preprocessing.md
```

Expected: only the requested script/test and Superpowers design/plan documents are changed; no dataset file appears.

- [ ] **Step 7: Report without committing**

Summarize created files, exact run commands, tests and their fresh counts,
dry-run zero-write behavior, overwrite behavior, exact-time rules, rejected-row
audit rules, and any remaining limitations. Do not commit or push unless the
user separately authorizes it.
