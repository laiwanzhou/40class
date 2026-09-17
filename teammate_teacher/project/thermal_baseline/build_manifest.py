from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = PROJECT_DIR.parent / "Training" / "data" / "HAR" / "data" / "Thermal"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_VAL_USERS = ("user1", "user16", "user19", "user22")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立 Thermal trial 级数据清单")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--val-users",
        nargs="+",
        default=list(DEFAULT_VAL_USERS),
        help="整名用户划入验证集，避免 trial 级随机泄漏",
    )
    return parser.parse_args()


def class_sort_key(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    val_users = set(args.val_users)

    if not raw_root.is_dir():
        raise FileNotFoundError(f"找不到 Thermal 原始数据：{raw_root}")

    rows: list[dict[str, str | int]] = []
    empty_trials: list[str] = []

    class_dirs = sorted(
        (path for path in raw_root.iterdir() if path.is_dir()),
        key=class_sort_key,
    )
    for class_dir in class_dirs:
        class_id_text, class_label = class_dir.name.split("_", 1)
        class_id = int(class_id_text)
        for user_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            for trial_dir in sorted(path for path in user_dir.iterdir() if path.is_dir()):
                frame_count = sum(
                    1
                    for path in trial_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                if frame_count == 0:
                    empty_trials.append(str(trial_dir))
                    continue
                split = "val" if user_dir.name in val_users else "train"
                rows.append(
                    {
                        "sample_id": f"train__c{class_id:02d}__{user_dir.name}__{trial_dir.name}",
                        "split": split,
                        "class_id": class_id,
                        "class_name": class_dir.name,
                        "class_label": class_label,
                        "user_id": user_dir.name,
                        "trial_id": trial_dir.name,
                        "trial_dir": str(trial_dir.resolve()),
                        "num_frames": frame_count,
                    }
                )

    rows.sort(key=lambda row: (str(row["split"]), int(row["class_id"]), str(row["user_id"]), str(row["trial_id"])))
    output.parent.mkdir(parents=True, exist_ok=True)
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
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter(str(row["split"]) for row in rows)
    train_class_counts = Counter(int(row["class_id"]) for row in rows if row["split"] == "train")
    val_class_counts = Counter(int(row["class_id"]) for row in rows if row["split"] == "val")
    users = sorted({str(row["user_id"]) for row in rows})
    summary = {
        "raw_root": str(raw_root),
        "manifest": str(output),
        "total_trials": len(rows),
        "split_counts": dict(split_counts),
        "all_users": users,
        "val_users": sorted(val_users),
        "train_users": sorted(set(users) - val_users),
        "train_class_counts": {str(i): train_class_counts[i] for i in range(40)},
        "val_class_counts": {str(i): val_class_counts[i] for i in range(40)},
        "empty_trials": empty_trials,
    }
    summary_path = output.with_name("manifest_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    missing_train = [i for i in range(40) if train_class_counts[i] == 0]
    missing_val = [i for i in range(40) if val_class_counts[i] == 0]
    print(f"清单已保存：{output}")
    print(f"总 trial：{len(rows)}，训练：{split_counts['train']}，验证：{split_counts['val']}")
    print(f"验证用户：{', '.join(sorted(val_users))}")
    print(f"训练集缺失类别：{missing_train or '无'}")
    print(f"验证集缺失类别：{missing_val or '无'}")
    print(f"空 trial：{len(empty_trials)}")


if __name__ == "__main__":
    main()
