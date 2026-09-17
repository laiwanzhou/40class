from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_DIR.parent / "Training" / "data" / "HAR" / "data"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_VAL_USERS = ("user1", "user16", "user19", "user22")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立 Depth/IR/Skeleton 同步 trial 清单")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-users", nargs="+", default=list(DEFAULT_VAL_USERS))
    return parser.parse_args()


def scan_trials(modality_root: Path) -> dict[tuple[int, str, str], Path]:
    trials: dict[tuple[int, str, str], Path] = {}
    for class_dir in modality_root.iterdir():
        if not class_dir.is_dir():
            continue
        class_id = int(class_dir.name.split("_", 1)[0])
        for user_dir in class_dir.iterdir():
            if not user_dir.is_dir():
                continue
            for trial_dir in user_dir.iterdir():
                if trial_dir.is_dir():
                    trials[(class_id, user_dir.name, trial_dir.name)] = trial_dir
    return trials


def frame_id(path: Path, modality: str) -> str:
    stem = path.stem
    if modality == "depth":
        return stem[len("Depth_") : -len("_Color")]
    if modality == "ir":
        return stem[len("IR_") :]
    if modality == "skeleton":
        return stem[len("Color_") :]
    raise ValueError(modality)


def frame_ids(trial_dir: Path, modality: str) -> set[str]:
    if modality == "skeleton":
        files = (trial_dir / "predictions").glob("*.json")
    else:
        files = (path for path in trial_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return {frame_id(path, modality) for path in files if path.is_file()}


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    val_users = set(args.val_users)
    roots = {
        "depth": data_root / "Depth_Color",
        "ir": data_root / "IR",
        "skeleton": data_root / "Skeleton",
    }
    for modality, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"找不到 {modality}：{root}")

    maps = {modality: scan_trials(root) for modality, root in roots.items()}
    common_keys = set.intersection(*(set(mapping) for mapping in maps.values()))
    rows: list[dict[str, str | int]] = []
    frame_mismatch_trials = 0

    for class_id, user_id, trial_id in sorted(common_keys):
        paths = {modality: maps[modality][(class_id, user_id, trial_id)] for modality in maps}
        ids = {modality: frame_ids(paths[modality], modality) for modality in maps}
        aligned_ids = set.intersection(*(set(values) for values in ids.values()))
        if not aligned_ids:
            continue
        if not (ids["depth"] == ids["ir"] == ids["skeleton"]):
            frame_mismatch_trials += 1
        class_name = paths["depth"].parents[1].name
        split = "val" if user_id in val_users else "train"
        rows.append(
            {
                "sample_id": f"train__c{class_id:02d}__{user_id}__{trial_id}",
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "user_id": user_id,
                "trial_id": trial_id,
                "depth_dir": str(paths["depth"].resolve()),
                "ir_dir": str(paths["ir"].resolve()),
                "skeleton_dir": str(paths["skeleton"].resolve()),
                "num_aligned_frames": len(aligned_ids),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "class_id",
        "class_name",
        "user_id",
        "trial_id",
        "depth_dir",
        "ir_dir",
        "skeleton_dir",
        "num_aligned_frames",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter(str(row["split"]) for row in rows)
    class_counts = {
        split: Counter(int(row["class_id"]) for row in rows if row["split"] == split)
        for split in ("train", "val")
    }
    summary = {
        "data_root": str(data_root),
        "modality_trial_counts": {key: len(value) for key, value in maps.items()},
        "common_trial_count": len(common_keys),
        "usable_trial_count": len(rows),
        "split_counts": dict(split_counts),
        "val_users": sorted(val_users),
        "frame_mismatch_trials": frame_mismatch_trials,
        "class_counts": {
            split: {str(i): class_counts[split][i] for i in range(40)} for split in class_counts
        },
    }
    output.with_name("manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"清单：{output}")
    print(f"共同 trial：{len(common_keys)}，可用：{len(rows)}")
    print(f"训练/验证：{split_counts['train']}/{split_counts['val']}")
    print(f"帧键不完全一致的 trial：{frame_mismatch_trials}（已取交集）")
    print("训练缺失类别：", [i for i in range(40) if class_counts["train"][i] == 0] or "无")
    print("验证缺失类别：", [i for i in range(40) if class_counts["val"][i] == 0] or "无")


if __name__ == "__main__":
    main()
