from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_TRAIN_ROOT = REPO_DIR / "Training" / "data" / "HAR" / "data"
DEFAULT_TEST_ROOT = REPO_DIR / "Testing" / "data"
DEFAULT_TEST_CSV = REPO_DIR / "Testing" / "test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "six_modality_audit"

MODALITY_DIRS = {
    "depth_color": "Depth_Color",
    "ir": "IR",
    "thermal": "Thermal",
    "skeleton": "Skeleton",
    "imu": "IMU",
    "radar": "Radar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立六模态并集索引并审计缺失模态 shortcut")
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def class_id_from_name(name: str) -> int:
    prefix = name.split("_", 1)[0]
    if not prefix.isdigit():
        raise ValueError(f"无法从类别目录解析 action_id：{name}")
    return int(prefix)


def iter_train_trials(modality_root: Path) -> Iterable[tuple[tuple[str, str, str], Path]]:
    for class_dir in sorted((p for p in modality_root.iterdir() if p.is_dir()), key=lambda p: class_id_from_name(p.name)):
        for user_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            for trial_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
                yield (class_dir.name, user_dir.name, trial_dir.name), trial_dir


def csv_has_data(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.readline()
            if not header:
                return False
            while True:
                line = handle.readline()
                if not line:
                    return False
                if line.strip():
                    return True
    except OSError:
        return False


def trial_file_stats(path: Path | None, modality: str) -> dict[str, object]:
    if path is None or not path.is_dir():
        return {
            "path": "",
            "present": 0,
            "usable": 0,
            "file_count": 0,
            "data_file_count": 0,
            "total_bytes": 0,
        }

    if modality == "depth_color":
        files = sorted(path.glob("*.png"))
    elif modality == "ir":
        files = sorted(path.glob("*.png"))
    elif modality == "thermal":
        files = sorted(path.glob("*.jpg"))
    elif modality == "skeleton":
        files = sorted(path.rglob("*.json"))
    else:
        files = sorted(path.glob("*.csv"))

    sizes = []
    for file_path in files:
        try:
            sizes.append(file_path.stat().st_size)
        except OSError:
            sizes.append(0)

    if modality in {"imu", "radar"}:
        data_file_count = sum(csv_has_data(file_path) for file_path in files)
    else:
        data_file_count = sum(size > 0 for size in sizes)

    usable = int(bool(files) and data_file_count > 0)
    return {
        "path": str(path.resolve()),
        "present": 1,
        "usable": usable,
        "file_count": len(files),
        "data_file_count": int(data_file_count),
        "total_bytes": int(sum(sizes)),
    }


def recording_group(trial_id: str) -> str:
    # 数据没有显式 session 字段；只保留 trial 名的前两段作为“录制组代理”，
    # 避免把未经证实的目录语义写成真实 session。
    parts = trial_id.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else trial_id


def flatten_record(base: dict[str, object], paths: dict[str, Path | None]) -> dict[str, object]:
    row = dict(base)
    for modality in MODALITY_DIRS:
        stats = trial_file_stats(paths.get(modality), modality)
        for key, value in stats.items():
            row[f"{modality}_{key}"] = value
    present_bits = "".join(str(row[f"{m}_present"]) for m in MODALITY_DIRS)
    usable_bits = "".join(str(row[f"{m}_usable"]) for m in MODALITY_DIRS)
    row["present_pattern"] = present_bits
    row["usable_pattern"] = usable_bits
    row["present_modalities"] = sum(int(row[f"{m}_present"]) for m in MODALITY_DIRS)
    row["usable_modalities"] = sum(int(row[f"{m}_usable"]) for m in MODALITY_DIRS)
    return row


def build_train_rows(train_root: Path) -> list[dict[str, object]]:
    union: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    for modality, directory in MODALITY_DIRS.items():
        root = train_root / directory
        if not root.is_dir():
            raise FileNotFoundError(root)
        for key, path in iter_train_trials(root):
            union[key][modality] = path

    rows: list[dict[str, object]] = []
    for class_name, user_id, trial_id in sorted(
        union,
        key=lambda item: (class_id_from_name(item[0]), item[1], item[2]),
    ):
        class_id = class_id_from_name(class_name)
        rows.append(
            flatten_record(
                {
                    "split": "train",
                    "sample_id": f"{class_name}/{user_id}/{trial_id}",
                    "official_path": "",
                    "class_id": class_id,
                    "class_name": class_name,
                    "user_id": user_id,
                    "trial_id": trial_id,
                    "recording_group_proxy": recording_group(trial_id),
                },
                union[(class_name, user_id, trial_id)],
            )
        )
    return rows


def build_test_rows(test_csv: Path, test_root: Path) -> list[dict[str, object]]:
    with test_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        official_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for official in official_rows:
        official_path = official["path"]
        trial_dir = (test_root / official_path).resolve()
        if not trial_dir.is_dir():
            raise FileNotFoundError(trial_dir)
        paths = {
            modality: (trial_dir / directory if (trial_dir / directory).is_dir() else None)
            for modality, directory in MODALITY_DIRS.items()
        }
        rows.append(
            flatten_record(
                {
                    "split": "test",
                    "sample_id": trial_dir.name,
                    "official_path": official_path,
                    "class_id": -1,
                    "class_name": "",
                    "user_id": "",
                    "trial_id": trial_dir.name,
                    "recording_group_proxy": "",
                },
                paths,
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_missing_rates(rows: list[dict[str, object]], group_key: str, modality: str) -> list[dict[str, object]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(int(row[f"{modality}_usable"]))
    result = []
    for group, values in grouped.items():
        result.append(
            {
                "group": group,
                "samples": len(values),
                "missing_or_unusable": int(len(values) - sum(values)),
                "missing_or_unusable_rate": float(1.0 - np.mean(values)),
            }
        )
    return sorted(result, key=lambda item: (-float(item["missing_or_unusable_rate"]), str(item["group"])))


def availability_classifier(rows: list[dict[str, object]], suffix: str) -> dict[str, object]:
    x = np.asarray([[int(row[f"{m}_{suffix}"]) for m in MODALITY_DIRS] for row in rows])
    y = np.asarray([int(row["class_id"]) for row in rows])
    groups = np.asarray([str(row["user_id"]) for row in rows])
    predictions = np.full(len(rows), -1, dtype=np.int64)
    majority_predictions = np.full(len(rows), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=3)
    fold_metrics = []
    for fold, (train_indices, val_indices) in enumerate(splitter.split(x, y, groups)):
        classifier = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=20260723 + fold,
        )
        classifier.fit(x[train_indices], y[train_indices])
        predictions[val_indices] = classifier.predict(x[val_indices])
        majority = Counter(y[train_indices].tolist()).most_common(1)[0][0]
        majority_predictions[val_indices] = majority
        fold_metrics.append(
            {
                "fold": fold,
                "validation_subjects": sorted(set(groups[val_indices].tolist())),
                "samples": len(val_indices),
                "tree_accuracy": float(accuracy_score(y[val_indices], predictions[val_indices])),
                "majority_accuracy": float(accuracy_score(y[val_indices], majority_predictions[val_indices])),
            }
        )
    return {
        "features": [f"{m}_{suffix}" for m in MODALITY_DIRS],
        "method": "3-fold subject-disjoint depth-4 decision tree; diagnostic only",
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "majority_accuracy": float(accuracy_score(y, majority_predictions)),
        "folds": fold_metrics,
    }


def build_summary(train_rows: list[dict[str, object]], test_rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "train_samples_union": len(train_rows),
        "test_samples": len(test_rows),
        "modality_order": list(MODALITY_DIRS),
        "train_complete_present": sum(int(row["present_modalities"]) == 6 for row in train_rows),
        "train_complete_usable": sum(int(row["usable_modalities"]) == 6 for row in train_rows),
        "test_complete_present": sum(int(row["present_modalities"]) == 6 for row in test_rows),
        "test_complete_usable": sum(int(row["usable_modalities"]) == 6 for row in test_rows),
        "train_present_patterns": dict(Counter(str(row["present_pattern"]) for row in train_rows).most_common()),
        "train_usable_patterns": dict(Counter(str(row["usable_pattern"]) for row in train_rows).most_common()),
        "test_present_patterns": dict(Counter(str(row["present_pattern"]) for row in test_rows).most_common()),
        "test_usable_patterns": dict(Counter(str(row["usable_pattern"]) for row in test_rows).most_common()),
        "modalities": {},
    }
    modality_summary: dict[str, object] = {}
    for modality in MODALITY_DIRS:
        train_present = sum(int(row[f"{modality}_present"]) for row in train_rows)
        train_usable = sum(int(row[f"{modality}_usable"]) for row in train_rows)
        test_present = sum(int(row[f"{modality}_present"]) for row in test_rows)
        test_usable = sum(int(row[f"{modality}_usable"]) for row in test_rows)
        overall_missing_rate = 1.0 - train_usable / len(train_rows)
        by_class = grouped_missing_rates(train_rows, "class_id", modality)
        by_subject = grouped_missing_rates(train_rows, "user_id", modality)
        by_recording_group = grouped_missing_rates(train_rows, "recording_group_proxy", modality)
        modality_summary[modality] = {
            "train_present": train_present,
            "train_usable": train_usable,
            "train_missing_or_unusable": len(train_rows) - train_usable,
            "test_present": test_present,
            "test_usable": test_usable,
            "test_missing_or_unusable": len(test_rows) - test_usable,
            "train_file_count": int(sum(int(row[f"{modality}_file_count"]) for row in train_rows)),
            "test_file_count": int(sum(int(row[f"{modality}_file_count"]) for row in test_rows)),
            "train_missing_rate_range_by_class": [
                float(min(item["missing_or_unusable_rate"] for item in by_class)),
                float(max(item["missing_or_unusable_rate"] for item in by_class)),
            ],
            "train_max_abs_class_deviation_from_overall": float(
                max(abs(float(item["missing_or_unusable_rate"]) - overall_missing_rate) for item in by_class)
            ),
            "highest_missing_classes": by_class[:8],
            "highest_missing_subjects": by_subject[:8],
            "highest_missing_recording_groups_proxy": by_recording_group[:8],
        }
    summary["modalities"] = modality_summary
    summary["availability_shortcut_diagnostic"] = {
        "present_mask": availability_classifier(train_rows, "present"),
        "usable_mask": availability_classifier(train_rows, "usable"),
        "interpretation": (
            "若仅用六个二值 mask 的 subject-disjoint 分类显著高于多数类基线，"
            "说明 mask 可能携带类别 shortcut；即便不高，训练时仍应随机模拟缺模态并做消融。"
        ),
    }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("扫描训练集六模态并集……", flush=True)
    train_rows = build_train_rows(args.train_root.resolve())
    print(f"训练集并集：{len(train_rows)}", flush=True)
    print("按官方顺序扫描 Test……", flush=True)
    test_rows = build_test_rows(args.test_csv.resolve(), args.test_root.resolve())
    write_csv(output_dir / "train_union_manifest.csv", train_rows)
    write_csv(output_dir / "test_union_manifest.csv", test_rows)
    summary = build_summary(train_rows, test_rows)
    (output_dir / "availability_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
