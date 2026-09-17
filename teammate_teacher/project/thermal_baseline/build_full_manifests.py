from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Thermal full-training and test manifests from six-modality audit."
    )
    parser.add_argument("--train-union", type=Path, required=True)
    parser.add_argument("--test-union", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    train_union = read_rows(args.train_union)
    test_union = read_rows(args.test_union)
    train_rows = []
    for row in train_union:
        if row["thermal_usable"] != "1":
            continue
        class_id = int(row["class_id"])
        train_rows.append(
            {
                "sample_id": (
                    f"train__c{class_id:02d}__{row['user_id']}__{row['trial_id']}"
                ),
                "split": "train",
                "class_id": class_id,
                "class_name": row["class_name"],
                "class_label": row["class_name"].split("_", 1)[-1],
                "user_id": row["user_id"],
                "trial_id": row["trial_id"],
                "trial_dir": row["thermal_path"],
                "num_frames": int(row["thermal_data_file_count"]),
            }
        )
    test_rows = []
    for row in test_union:
        if row["thermal_usable"] != "1":
            continue
        test_rows.append(
            {
                "sample_id": row["sample_id"],
                "split": "test",
                "class_id": -1,
                "class_name": "",
                "class_label": "",
                "user_id": "",
                "trial_id": row["sample_id"],
                "trial_dir": row["thermal_path"],
                "num_frames": int(row["thermal_data_file_count"]),
            }
        )
    if len({row["sample_id"] for row in train_rows}) != len(train_rows):
        raise ValueError("Duplicate full-training sample IDs.")
    if len({row["sample_id"] for row in test_rows}) != len(test_rows):
        raise ValueError("Duplicate test sample IDs.")
    if set(int(row["class_id"]) for row in train_rows) != set(range(40)):
        raise ValueError("Full-training manifest does not contain all 40 classes.")
    write_rows(args.train_output, train_rows)
    write_rows(args.test_output, test_rows)
    summary = {
        "protocol": (
            "All Thermal-usable training trials are used for fixed-epoch refit. "
            "Thermal-missing test trials are omitted here and must fall back to "
            "the non-Thermal base."
        ),
        "train_union_rows": len(train_union),
        "train_thermal_usable": len(train_rows),
        "train_subjects": len(set(row["user_id"] for row in train_rows)),
        "train_classes": len(set(int(row["class_id"]) for row in train_rows)),
        "test_union_rows": len(test_union),
        "test_thermal_usable": len(test_rows),
        "test_thermal_missing": len(test_union) - len(test_rows),
        "train_output": str(args.train_output.resolve()),
        "test_output": str(args.test_output.resolve()),
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
