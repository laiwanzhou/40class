# IMU Stage 1 Preprocessing Design

## Goal and scope

Create `scripts/preprocess_imu_stage1.py` to convert the Small-Model-Track
training IMU CSV files into one traceable, time-sorted CSV per action. This
stage preserves valid measurements and reports source defects. It does not
interpolate, fill, filter, smooth, normalize, standardize, resample, pad,
truncate, or deduplicate records.

The script never modifies the input tree. In normal mode it mirrors action
paths below the output root and writes action results incrementally. In
`--dry-run` mode it performs the same discovery and validation but makes no
filesystem changes of any kind.

## Public interface and safety

The command accepts `--input-root`, `--output-root`, `--overwrite`, and
`--dry-run`. The exact defaults are:

```text
input:  D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\IMU
output: D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train\new_IMU
```

Input and output roots must not be equal or overlap by ancestry; this stronger
check prevents output creation from changing the input tree or mixing generated
files with source discovery. The check happens before logging creates a file.

An existing `imu_merged.csv` causes the action to be reported as
`skipped_existing` unless `--overwrite` is present. Normal execution writes
console and file logs, with the file log at `output_root/processing.log`.
Dry-run configures console logging only and creates neither the output root,
logs, manifest, action directories, temporary files, nor other artifacts.

## Components

### 1. Discovery

`discover_action_directories()` recursively finds directories that directly
contain one or more visible `.csv` files. It parses the first relative path
component as `<class_id>_<class_name>`, the second as `user_id`, and the final
component as `action_id`. Discovery sorts classes by integer `class_id`, then
users and actions with a natural numeric key. A malformed action path is a
global discovery-contract error and produces exit code 2 before action writes.

### 2. Structural CSV inspection

`read_csv_robust()` opens each file with `encoding="utf-8-sig"`,
`newline=""`, and strict `csv.reader` parsing. There is no fallback to GBK or
another encoding. A decode or CSV syntax error becomes a file-level error and
is reported without aborting other actions.

The first CSV record is the header. Header cells are normalized by removing
BOMs, surrounding whitespace, Unicode format/control characters, and known
presentation differences such as full-width parentheses and micro-symbol
variants. The normalized header must contain time, device name, and all 16
retained IMU feature columns: acceleration X/Y/Z, angular velocity X/Y/Z,
angle X/Y/Z, magnetic field X/Y/Z, and quaternion 0/1/2/3. Missing any of
these required columns fails that file. Temperature, version, and battery are
discarded metadata columns: missing ones produce schema warnings but do not
fail the file.

The scanner records, for every data record:

- `source_line_number`: 1-based physical starting line in the original file.
- `source_row_index`: 1-based record number in the data area.

The data record counter advances for every candidate record, including blank,
malformed, and repeated-header records. This keeps indices stable when records
are rejected. For a quoted record spanning multiple lines,
`source_line_number` is its first physical line.

Recoverable row-level structural rejection reasons are `blank_row`,
`repeated_header`, and `field_count_mismatch`. Repeated-header detection
compares normalized header fingerprints rather than raw strings.

`utf8_decode_error`, `csv_syntax_error`, `missing_required_columns`, and
`empty_or_missing_header` are file-level fatal errors. A file error records
the source filename, error type, detected physical line number when available,
and message. If any directly contained CSV has a fatal file error, the entire
action is `failed`, no new partial `imu_merged.csv` is produced, and processing
continues with the next action even if another CSV in the same action yielded
valid rows.

Valid records are collected in memory and passed once to
`pd.DataFrame(records, columns=normalized_header)`; Pandas is not called once
per row and the CSV is not parsed a second time.

### 3. DataFrame validation

Column normalization maps the required Chinese source names to canonical
names. Pandas converts absolute time with `pd.to_datetime(errors="coerce")`.
It converts the 16 retained IMU feature columns and any present temperature or
battery columns with `pd.to_numeric(errors="coerce")`.

Device names are matched case-insensitively by complete prefixes only:
`WTLL` to `LL`, `WTRL` to `RL`, `WTLA` to `LA`, `WTRA` to `RA`, and `WTC` to
`C`. Text in parentheses is ignored. Filenames are never used to infer a
sensor.

A row is content-rejected if its time is invalid, its sensor is unknown, or
any of the 16 retained IMU feature values cannot be parsed. Numeric values are
never replaced with zero. Invalid temperature or battery values produce
warnings and missing-value counts but do not reject an otherwise valid IMU
row. Version is read only as metadata. Temperature, version, and battery are
all removed before final output. All applicable rejection reasons are combined
with semicolons.
Rejected records use this audit schema:

```text
source_file, source_line_number, source_row_index,
reject_stage, reject_reason, raw_row
```

`reject_stage` is `structural` or `content`. `raw_row` is a JSON-encoded array
of original field values, preserving delimiters and non-ASCII text without
inventing additional CSV columns.

### 4. Action processing

`process_action_directory()` is side-effect free and returns an
`ActionResult`. It reads every CSV directly inside one action directory,
combines valid rows, and uses the earliest valid timestamp across all files
and all sensors as the single action start time. Relative seconds use exact
timedelta seconds; relative milliseconds use rounded values stored as
`int64`.

Rows are stably sorted internally by the exact parsed `absolute_time`,
temporary `sensor_order`, `source_file`, and `source_row_index`.
`relative_time_ms` is only an output convenience field; it is never the
primary chronological sort key or a duplicate-detection key. Equal exact
timestamps from different sensors are normal. Duplicate counts are calculated
independently within each sensor using exact parsed `absolute_time` values and
count rows participating beyond the first occurrence of a sensor/time pair.
Duplicate records remain in output. Exact absolute time is removed only after
sorting and duplicate detection are complete.

The final CSV contains only:

```text
relative_time_s, relative_time_ms, sensor_position,
acc_x_g, acc_y_g, acc_z_g,
gyro_x_dps, gyro_y_dps, gyro_z_dps,
angle_x_deg, angle_y_deg, angle_z_deg,
mag_x_ut, mag_y_ut, mag_z_ut,
quat_0, quat_1, quat_2, quat_3,
source_file, source_row_index
```

Absolute time, raw device name, temperature, version, battery, temporary
sensor order, and physical line number are excluded from training output.
Physical line numbers remain available in rejected-row audits.

Status precedence is:

1. `failed` when the action cannot produce any valid output or has a fatal
   schema/file error that prevents a trustworthy action result.
2. `incomplete_sensors` when one or more of `LL`, `RL`, `LA`, `RA`, or `C` is
   absent from valid output.
3. `success_with_warnings` when all sensors are present but rejected rows,
   duplicate timestamps, or non-fatal file warnings exist.
4. `success` otherwise.

Unknown sensor rows are rejected and counted separately. A failed action still
returns complete metadata for the manifest and QC report.

### 5. Writer and reporting

`write_action_result()` is the only action-artifact writer. In normal mode it
writes `imu_merged.csv` with UTF-8 BOM and no index, always writes `qc.json`,
and writes `rejected_rows.csv` only when rejected records exist. It writes
failed-action QC even when there is no merged CSV. Each action is written and
its large DataFrames are then released before processing the next action.

For a new or overwritten action, the writer first creates a complete staging
directory beside the destination action directory and validates its managed
artifact set. The managed final states are exact:

| Current result | Managed files that must exist |
| --- | --- |
| Success without rejected rows | `imu_merged.csv`, `qc.json` |
| Success with rejected rows | `imu_merged.csv`, `qc.json`, `rejected_rows.csv` |
| Failed without rejected rows | `qc.json` |
| Failed with rejected rows | `qc.json`, `rejected_rows.csv` |
| Skipped existing | Existing action directory remains byte-for-byte unchanged |

With `--overwrite`, the existing destination is renamed to a unique sibling
backup, the validated staging directory is renamed into place, and then the
backup is removed. If installation fails after the backup rename, the writer
restores the backup before reporting a write failure. Staging and backup paths
are resolved and verified to be siblings below the output root before cleanup.
This action-directory swap prevents stale `imu_merged.csv` or
`rejected_rows.csv` files from surviving a changed result. Temporary paths are
used only in normal mode and are cleaned or recovered on failure. The original
input tree is never a write target.

The driver accumulates only manifest rows and aggregate counters. At the end,
`manifest.csv` contains every discovered action, including failed and skipped
actions, with exactly this column order:

```text
sample_id
class_id
class_name
user_id
action_id
relative_action_path
output_csv
status
csv_file_count
total_input_rows
valid_output_rows
rejected_rows
unknown_sensor_rows
present_sensors
missing_sensors
ll_rows
rl_rows
la_rows
ra_rows
c_rows
ll_duplicate_timestamps
rl_duplicate_timestamps
la_duplicate_timestamps
ra_duplicate_timestamps
c_duplicate_timestamps
duration_s
warning_count
error_message
```

`sample_id` is `<class_id>__<user_id>__<action_id>`. Relative paths use POSIX
forward slashes; `output_csv` is empty when no merged CSV exists. Sensor lists
are semicolon-delimited strings serialized in fixed `LL;RL;LA;RA;C` order,
omitting sensors that are not members of the list. `error_message` is a single
line. Manifest rows are sorted by integer `class_id` and then natural
user/action keys.

The console summary reports class, user, and action counts; all status counts;
input, valid, and rejected rows; and missing-action counts for all five sensors.

## QC and overwrite behavior

`qc.json` contains exactly these top-level fields, in this order:

```text
class_id
class_name
user_id
action_id
input_directory
input_csv_files
csv_file_count
status
total_input_rows
valid_output_rows
rejected_rows
unknown_sensor_rows
present_sensors
missing_sensors
rows_per_sensor
duplicate_timestamp_count_per_sensor
min_relative_time_ms
max_relative_time_ms
duration_s
columns_written
warnings
error_message
action_start_time
action_end_time
structural_rejected_rows
content_rejected_rows
file_errors
```

`input_csv_files`, `present_sensors`, `missing_sensors`, `columns_written`,
`warnings`, and `file_errors` are JSON arrays. Sensor arrays and the keys of
the two per-sensor objects use fixed `LL`, `RL`, `LA`, `RA`, `C` order.
`input_directory` is absolute. JSON timestamps are ISO 8601 strings; absent
measures and timestamps are JSON `null`. `file_errors` entries contain
`source_file`, `error_type`, `source_line_number`, and `message`.

When an action output already exists and overwrite is disabled, the driver
loads its existing `qc.json` when possible to preserve row statistics in the
new manifest. It reports `skipped_existing` for this run and never rewrites
that action directory. Missing or unreadable existing QC produces a skip
warning rather than permission to overwrite.

## Error handling

Failures are isolated per action. Exceptions are logged with the input action
path, converted to failed `ActionResult` values, written as QC in normal mode,
and included in the manifest. Fatal CSV errors include their filename, error
type, detected line number, and message in `file_errors`. A failure writing one
action also becomes a failed manifest row and processing continues.

The CLI exit-code contract, including dry-run, is:

```text
0  Run completed and no action has status failed.
1  Run completed after scanning all actions, with one or more failed actions.
2  A global fatal error occurred, including root safety validation, discovery,
   logging setup, or manifest/report writing.
```

## Internal interfaces

The implementation plan and tests use these typed boundaries:

```python
discover_action_directories(...) -> list[ActionDescriptor]
read_csv_robust(...) -> CsvReadResult
validate_dataframe(...) -> ValidatedCsvResult
process_action_directory(...) -> ActionResult
write_action_result(...) -> WriteResult
```

The first four stages are side-effect free with respect to the input and output
trees. Only `write_action_result()` and the root-level manifest/log reporting
layer may write, and neither is invoked for dry-run.

## Test strategy

Tests use temporary input and output trees and public functions or the CLI.
They cover:

1. All five device-prefix mappings and an unknown device.
2. A shared global earliest time across sensors and files.
3. Stable sorting after deliberately shuffled input.
4. Equal cross-sensor timestamps excluded from duplicate counts, while
   same-sensor duplicates are counted and retained.
5. Missing sensors produce `incomplete_sensors` and correct QC fields.
6. Class directories sort by integer ID.
7. Input file bytes and paths remain unchanged after normal processing.
8. Structural/content rejected rows retain both trace indices and combined
   reasons.
9. Existing outputs skip unless overwrite is explicit.
10. Dry-run creates no output path or files.
11. Failed actions still appear in QC and manifest during normal CLI runs.
12. Records less than 1 ms apart retain exact chronological order even when
    their rounded `relative_time_ms` values match.
13. Different exact timestamps within one sensor are not duplicates merely
    because their rounded milliseconds match.
14. Invalid temperature or battery metadata does not reject a row whose time,
    sensor, and retained IMU features are valid.
15. A `csv_syntax_error` in one action CSV fails the whole action and prevents
    partial merged output.
16. Overwriting a formerly successful action with a failed result removes the
    stale merged CSV through action-directory replacement.
17. Overwriting a formerly rejected action with a clean result removes the
    stale rejected-row CSV.
18. Equal roots and either ancestor/descendant root relationship are rejected
    before any output path or log is created.
19. Blank, repeated-header, and field-count-invalid records still advance the
    1-based `source_row_index`.
20. A quoted CSV record spanning physical lines records its first physical
    line as `source_line_number`.
21. Output CSV columns and order exactly match the final schema.
22. Exit codes distinguish clean completion, action failures, and global fatal
    errors in both normal and dry-run modes.

Implementation follows one red-green-refactor cycle per behavior. Final
verification runs the focused test module, the repository's existing smoke
test entry points that are safe and relevant, CLI help, a synthetic dry-run,
and a small synthetic normal run without processing the full training set.
