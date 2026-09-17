from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_UNION_MANIFEST = (
    REPO_DIR
    / "aligned_multimodal"
    / "data"
    / "six_modality_audit"
    / "train_union_manifest.csv"
)
DEFAULT_FOLD_SUMMARY = (
    REPO_DIR / "aligned_multimodal" / "data" / "subject_folds" / "folds_summary.json"
)
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "subject_folds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Thermal-only manifests from the fixed subject folds"
    )
    parser.add_argument("--union-manifest", type=Path, default=DEFAULT_UNION_MANIFEST)
    parser.add_argument("--fold-summary", type=Path, default=DEFAULT_FOLD_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_id(row: dict[str, str]) -> str:
    return (
        f"train__c{int(row['class_id']):02d}__"
        f"{row['user_id']}__{row['trial_id']}"
    )


def main() -> None:
    args = parse_args()
    union_path = args.union_manifest.resolve()
    fold_summary_path = args.fold_summary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    union_rows = read_csv(union_path)
    usable = [
        row
        for row in union_rows
        if row["split"] == "train"
        and int(row["thermal_usable"])
        and Path(row["thermal_path"]).is_dir()
    ]
    folds = json.loads(
        fold_summary_path.read_text(encoding="utf-8")
    )["folds"]
    fieldnames = [
        "sample_id",
        "split",
        "class_id",
        "class_name",
        "class_label",
        "user_id",
        "trial_id",
        "trial_dir",
        "num_frames",
    ]
    summaries = []
    for fold in folds:
        train_users = set(fold["train_users"])
        val_users = set(fold["val_users"])
        rows = []
        for source in usable:
            if source["user_id"] in train_users:
                split = "train"
            elif source["user_id"] in val_users:
                split = "val"
            else:
                continue
            rows.append(
                {
                    "sample_id": sample_id(source),
                    "split": split,
                    "class_id": int(source["class_id"]),
                    "class_name": source["class_name"],
                    "class_label": source["class_name"].split("_", 1)[-1],
                    "user_id": source["user_id"],
                    "trial_id": source["trial_id"],
                    "trial_dir": source["thermal_path"],
                    "num_frames": int(source["thermal_file_count"]),
                }
            )
        rows.sort(
            key=lambda row: (
                row["split"],
                int(row["class_id"]),
                int(row["user_id"][4:]),
                row["trial_id"],
            )
        )
        manifest_path = output_dir / f"fold_{int(fold['fold'])}.csv"
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        train_rows = [row for row in rows if row["split"] == "train"]
        val_rows = [row for row in rows if row["split"] == "val"]
        train_counts = Counter(int(row["class_id"]) for row in train_rows)
        val_counts = Counter(int(row["class_id"]) for row in val_rows)
        summaries.append(
            {
                "fold": int(fold["fold"]),
                "manifest": str(manifest_path),
                "train_users": sorted(train_users, key=lambda value: int(value[4:])),
                "val_users": sorted(val_users, key=lambda value: int(value[4:])),
                "train_trials": len(train_rows),
                "val_trials": len(val_rows),
                "missing_train_classes": [
                    class_id for class_id in range(40) if train_counts[class_id] == 0
                ],
                "missing_val_classes": [
                    class_id for class_id in range(40) if val_counts[class_id] == 0
                ],
                "val_class_min": min(val_counts.values()),
                "val_class_max": max(val_counts.values()),
            }
        )

    summary = {
        "protocol": (
            "Existing three subject-disjoint folds; Thermal-usable trials from the "
            "six-modality union manifest; normalized trial-progress frame sampling."
        ),
        "union_trials": len(union_rows),
        "thermal_usable_trials": len(usable),
        "thermal_missing_or_unusable_trials": len(union_rows) - len(usable),
        "folds": summaries,
    }
    (output_dir / "folds_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
