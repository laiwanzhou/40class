from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "subject_folds"
DEFAULT_FOLDS = (
    ("user3", "user20", "user22", "user24", "user4", "user9"),
    ("user2", "user8", "user1", "user17", "user6", "user23"),
    ("user5", "user16", "user19", "user7", "user21", "user18"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从统一 manifest 生成固定的 subject-disjoint 三折清单")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "user_id" not in fieldnames or "split" not in fieldnames:
        raise ValueError(f"manifest 格式不正确：{manifest}")

    all_users = {row["user_id"] for row in rows}
    fold_users = [set(users) for users in DEFAULT_FOLDS]
    if set.union(*fold_users) != all_users:
        missing = sorted(all_users - set.union(*fold_users))
        extra = sorted(set.union(*fold_users) - all_users)
        raise ValueError(f"三折 subject 与 manifest 不一致，missing={missing}，extra={extra}")
    if sum(len(users) for users in fold_users) != len(set.union(*fold_users)):
        raise ValueError("三折之间存在重复 subject")

    summary: dict[str, object] = {"source_manifest": str(manifest), "folds": []}
    for fold_index, val_users in enumerate(fold_users):
        fold_rows = []
        for row in rows:
            updated = dict(row)
            updated["split"] = "val" if row["user_id"] in val_users else "train"
            fold_rows.append(updated)
        output = output_dir / f"fold_{fold_index}.csv"
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fold_rows)

        split_counts = Counter(row["split"] for row in fold_rows)
        val_class_counts = Counter(
            int(row["class_id"]) for row in fold_rows if row["split"] == "val"
        )
        train_users = sorted(all_users - val_users, key=lambda value: int(value[4:]))
        fold_summary = {
            "fold": fold_index,
            "manifest": str(output),
            "train_users": train_users,
            "val_users": sorted(val_users, key=lambda value: int(value[4:])),
            "train_trials": split_counts["train"],
            "val_trials": split_counts["val"],
            "val_class_min": min(val_class_counts.values()),
            "val_class_max": max(val_class_counts.values()),
            "missing_val_classes": [i for i in range(40) if val_class_counts[i] == 0],
        }
        summary["folds"].append(fold_summary)
        print(
            f"fold {fold_index}: train={split_counts['train']}，val={split_counts['val']}，"
            f"val_users={fold_summary['val_users']}，"
            f"class_range={fold_summary['val_class_min']}..{fold_summary['val_class_max']}",
            flush=True,
        )

    (output_dir / "folds_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
