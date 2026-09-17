from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ALIGNED = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_TRAIN_UNION = PROJECT_DIR / "data/six_modality_audit/train_union_manifest.csv"
DEFAULT_TEST_UNION = PROJECT_DIR / "data/p46_test_union_manifest.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data/p85_recording_metadata"
TIME_PATTERN = re.compile(
    r"^\s*(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\s+"
    r"(\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.(\d+))?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract label-free recording time/device metadata from train and anonymous Test IMU files."
    )
    parser.add_argument("--aligned", type=Path, default=DEFAULT_ALIGNED)
    parser.add_argument("--train-union", type=Path, default=DEFAULT_TRAIN_UNION)
    parser.add_argument("--test-union", type=Path, default=DEFAULT_TEST_UNION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_time(value: str) -> dt.datetime | None:
    match = TIME_PATTERN.match(value)
    if match is None:
        return None
    year, month, day, hour, minute, second, fraction = match.groups()
    microsecond = int(((fraction or "") + "000000")[:6])
    try:
        return dt.datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second), microsecond
        )
    except ValueError:
        return None


def inspect_imu(folder_value: str) -> dict[str, Any]:
    folder = Path(folder_value) if folder_value else Path("__missing__")
    timestamps: list[dt.datetime] = []
    devices: set[str] = set()
    source_files = 0
    parsed_rows = 0
    if folder.is_dir():
        for path in sorted(folder.glob("*.csv")):
            source_files += 1
            try:
                lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
            except OSError:
                continue
            for line in lines[1:]:
                fields = line.split(",", 2)
                if len(fields) < 2:
                    continue
                timestamp = parse_time(fields[0])
                if timestamp is None:
                    continue
                timestamps.append(timestamp)
                devices.add(fields[1].strip())
                parsed_rows += 1
    if timestamps:
        first = min(timestamps)
        last = max(timestamps)
        date = first.date().isoformat()
        seconds = (
            first.hour * 3600
            + first.minute * 60
            + first.second
            + first.microsecond / 1_000_000
        )
        duration = (last - first).total_seconds()
        first_iso = first.isoformat(timespec="milliseconds")
        last_iso = last.isoformat(timespec="milliseconds")
    else:
        date = ""
        seconds = ""
        duration = ""
        first_iso = ""
        last_iso = ""
    device_text = "|".join(sorted(devices))
    device_hash = hashlib.sha1(device_text.encode("utf-8")).hexdigest()[:12] if devices else ""
    return {
        "timestamp_available": int(bool(timestamps)),
        "recording_date": date,
        "start_timestamp": first_iso,
        "end_timestamp": last_iso,
        "start_seconds": seconds,
        "duration_seconds": duration,
        "imu_csv_files": source_files,
        "imu_parsed_rows": parsed_rows,
        "device_count": len(devices),
        "device_signature": device_hash,
        "device_ids": device_text,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stable_key(row: dict[str, str]) -> tuple[int, str, str]:
    return int(row["class_id"]), row["user_id"], row["trial_id"]


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    aligned = read_csv(args.aligned)
    train_union_rows = read_csv(args.train_union)
    train_union = {stable_key(row): row for row in train_union_rows}
    if len(train_union) != len(train_union_rows):
        raise RuntimeError("Train union has duplicate class/user/trial keys")
    test_union = read_csv(args.test_union)
    if len(aligned) != 2914 or len(test_union) != 405:
        raise RuntimeError("Frozen train/Test counts changed")
    train_rows: list[dict[str, Any]] = []
    for row in aligned:
        union = train_union[stable_key(row)]
        train_rows.append(
            {
                "sample_id": row["sample_id"],
                "class_id": int(row["class_id"]),
                "user_id": row["user_id"],
                "fold0_split": row["split"],
                "trial_id": row["trial_id"],
                **inspect_imu(union["imu_path"]),
            }
        )
    test_rows: list[dict[str, Any]] = []
    for row in test_union:
        test_rows.append(
            {
                "sample_id": row["official_sample_id"],
                "official_path": row["official_path"],
                **inspect_imu(row["imu_path"]),
            }
        )
    write_csv(output / "train_recording_metadata.csv", train_rows)
    write_csv(output / "test_recording_metadata.csv", test_rows)
    summary = {
        "protocol": "label-free IMU recording metadata; no Test label or manual Test annotation",
        "train_rows": len(train_rows),
        "train_timestamp_available": sum(row["timestamp_available"] for row in train_rows),
        "train_users": len({row["user_id"] for row in train_rows}),
        "test_rows": len(test_rows),
        "test_timestamp_available": sum(row["timestamp_available"] for row in test_rows),
        "test_dates": sorted(
            {
                row["recording_date"]
                for row in test_rows
                if row["recording_date"]
            }
        ),
        "note": (
            "Metadata is only a candidate auxiliary signal. It must improve frozen "
            "subject-disjoint OOF before it may affect Test predictions."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
