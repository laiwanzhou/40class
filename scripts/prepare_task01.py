from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_DATA_ROOT = Path(
    r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train"
)
MODALITY_COLUMNS = {
    "Depth_Color": "depth_color_path",
    "IMU": "imu_path",
    "IR": "ir_path",
    "Radar": "radar_path",
    "Skeleton": "skeleton_path",
    "Thermal": "thermal_path",
}
MANIFEST_COLUMNS = [
    "sample_id",
    "class_id",
    "action_name",
    "user_id",
    "trial_id",
    *MODALITY_COLUMNS.values(),
]
IGNORED_NAMES = {
    "__MACOSX",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    ".claude",
}
EXPECTED_CLASS_IDS = set(range(40))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the task-01 training manifest and fixed user split."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    """Return a comparison key that sorts embedded numbers numerically."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def is_ignored(path: Path) -> bool:
    name = path.name
    return name in IGNORED_NAMES or name.startswith(".") or name.startswith("._")


def visible_directories(path: Path) -> list[Path]:
    return sorted(
        (item for item in path.iterdir() if item.is_dir() and not is_ignored(item)),
        key=lambda item: natural_key(item.name),
    )


def parse_class_directory(path: Path) -> tuple[int, str]:
    try:
        class_id_text, action_name = path.name.split("_", 1)
        class_id = int(class_id_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid class directory {path}: expected '<number>_<action_name>'."
        ) from exc
    if not action_name:
        raise ValueError(f"Empty action name in class directory: {path}")
    return class_id, action_name


def find_class_mapping(data_root: Path) -> Path | None:
    search_roots = (data_root, data_root.parent, data_root.parent.parent)
    matches = [root / "class_mapping.csv" for root in search_roots]
    existing = [path for path in matches if path.is_file()]
    if len(existing) > 1:
        raise ValueError(
            "Multiple class_mapping.csv files found in the required search scope: "
            + ", ".join(str(path) for path in existing)
        )
    return existing[0] if existing else None


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Could not decode {path}: {'; '.join(errors)}")


def load_official_mapping(path: Path) -> dict[int, str]:
    frame = read_csv_with_fallback(path)
    normalized = {str(column).strip().casefold(): column for column in frame.columns}
    id_aliases = ("class_id", "classid", "id", "label", "index")
    name_aliases = ("action_name", "action", "class_name", "classname", "name")
    id_column = next((normalized[name] for name in id_aliases if name in normalized), None)
    name_column = next(
        (normalized[name] for name in name_aliases if name in normalized), None
    )
    if id_column is None or name_column is None:
        raise ValueError(
            f"Could not identify class id/name columns in {path}; columns={list(frame.columns)}"
        )

    mapping: dict[int, str] = {}
    for row in frame[[id_column, name_column]].itertuples(index=False, name=None):
        class_id = int(row[0])
        action_name = str(row[1]).strip()
        if class_id in mapping and mapping[class_id] != action_name:
            raise ValueError(f"Conflicting official names for class_id={class_id}")
        mapping[class_id] = action_name
    if set(mapping) != EXPECTED_CLASS_IDS:
        raise ValueError(
            f"Official class mapping IDs must be 0-39; got {sorted(mapping)}"
        )
    if any(not name for name in mapping.values()):
        raise ValueError("Official class mapping contains an empty action name.")
    return mapping


def scan_modalities(
    data_root: Path, official_mapping: dict[int, str] | None
) -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[str, int]]:
    records: dict[tuple[int, str, str], dict[str, Any]] = {}
    modality_counts: dict[str, int] = {}
    reference_mapping: dict[int, str] | None = None

    for modality, path_column in MODALITY_COLUMNS.items():
        modality_root = data_root / modality
        if not modality_root.is_dir():
            raise FileNotFoundError(f"Required modality directory not found: {modality_root}")

        modality_mapping: dict[int, str] = {}
        count = 0
        for class_dir in visible_directories(modality_root):
            class_id, directory_action = parse_class_directory(class_dir)
            expected_action = (
                official_mapping[class_id] if official_mapping is not None else directory_action
            )
            if official_mapping is not None and directory_action != expected_action:
                raise ValueError(
                    f"Class mapping mismatch at {class_dir}: directory={directory_action!r}, "
                    f"official={expected_action!r}"
                )
            if class_id in modality_mapping and modality_mapping[class_id] != directory_action:
                raise ValueError(
                    f"Duplicate class_id={class_id} with inconsistent names in {modality}"
                )
            modality_mapping[class_id] = directory_action

            for user_dir in visible_directories(class_dir):
                for trial_dir in visible_directories(user_dir):
                    key = (class_id, user_dir.name, trial_dir.name)
                    relative_path = trial_dir.relative_to(data_root).as_posix()
                    record = records.setdefault(
                        key,
                        {
                            "class_id": class_id,
                            "action_name": expected_action,
                            "user_id": user_dir.name,
                            "trial_id": trial_dir.name,
                            **{column: "" for column in MODALITY_COLUMNS.values()},
                        },
                    )
                    if record["action_name"] != expected_action:
                        raise ValueError(f"Action-name mismatch for sample key {key}")
                    if record[path_column]:
                        raise ValueError(
                            f"Duplicate trial for {modality} and sample key {key}: "
                            f"{record[path_column]} and {relative_path}"
                        )
                    record[path_column] = relative_path
                    count += 1

        if set(modality_mapping) != EXPECTED_CLASS_IDS:
            raise ValueError(
                f"{modality} class IDs must be 0-39; got {sorted(modality_mapping)}"
            )
        if reference_mapping is None:
            reference_mapping = modality_mapping
        elif modality_mapping != reference_mapping:
            raise ValueError(f"Class mapping in {modality} differs from other modalities.")
        modality_counts[modality] = count

    return records, modality_counts


def build_manifest(records: dict[tuple[int, str, str], dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_id, user_id, trial_id in sorted(
        records, key=lambda key: (key[0], natural_key(key[1]), natural_key(key[2]))
    ):
        row = records[(class_id, user_id, trial_id)].copy()
        row["sample_id"] = f"train__c{class_id:02d}__{user_id}__{trial_id}"
        rows.append(row)
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def sorted_users(users: Iterable[str]) -> list[str]:
    return sorted(set(users), key=natural_key)


def build_fold(manifest: pd.DataFrame) -> dict[str, Any]:
    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=20260715,
    )
    train_indices, val_indices = next(
        splitter.split(manifest, manifest["class_id"], manifest["user_id"])
    )
    train_frame = manifest.iloc[train_indices]
    val_frame = manifest.iloc[val_indices]
    return {
        "fold": 0,
        "method": "StratifiedGroupKFold",
        "n_splits": 3,
        "shuffle": True,
        "random_state": 20260715,
        "group_field": "user_id",
        "label_field": "class_id",
        "train_users": sorted_users(train_frame["user_id"]),
        "val_users": sorted_users(val_frame["user_id"]),
        "train_sample_count": int(len(train_frame)),
        "val_sample_count": int(len(val_frame)),
        "val_class_count": int(val_frame["class_id"].nunique()),
    }


def validate_manifest(manifest: pd.DataFrame, data_root: Path) -> None:
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(f"Unexpected manifest columns: {list(manifest.columns)}")
    if manifest.empty:
        raise ValueError("manifest.csv is empty.")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("manifest.csv contains duplicate sample_id values.")
    if manifest[["class_id", "user_id", "trial_id"]].duplicated().any():
        raise ValueError("manifest.csv contains duplicate sample keys.")
    if not set(manifest["class_id"].astype(int)).issubset(EXPECTED_CLASS_IDS):
        raise ValueError("manifest.csv contains class_id values outside 0-39.")
    for column in ("action_name", "user_id", "trial_id"):
        if manifest[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"manifest.csv contains an empty {column}.")

    path_columns = list(MODALITY_COLUMNS.values())
    normalized_paths = manifest[path_columns].fillna("").astype(str)
    if normalized_paths.apply(lambda row: all(not value for value in row), axis=1).any():
        raise ValueError("A manifest row has no modality path.")

    column_modalities = {column: modality for modality, column in MODALITY_COLUMNS.items()}
    for column in path_columns:
        for value in normalized_paths[column]:
            if not value:
                continue
            posix_path = PurePosixPath(value)
            if posix_path.is_absolute() or Path(value).is_absolute():
                raise ValueError(f"Absolute path found in {column}: {value}")
            if "\\" in value:
                raise ValueError(f"Backslash found in {column}: {value}")
            if posix_path.parts[0] != column_modalities[column]:
                raise ValueError(f"Unexpected modality root in {column}: {value}")
            if any(part in IGNORED_NAMES or part.startswith(".") for part in posix_path.parts):
                raise ValueError(f"System/hidden path found in {column}: {value}")
            if any(part.casefold() in {"test", "testing"} for part in posix_path.parts):
                raise ValueError(f"Test-set path found in {column}: {value}")
            if not (data_root / Path(*posix_path.parts)).is_dir():
                raise ValueError(f"Trial directory does not exist for {column}: {value}")


def validate_fold(fold: dict[str, Any], manifest: pd.DataFrame) -> None:
    train_users = set(fold["train_users"])
    val_users = set(fold["val_users"])
    manifest_users = set(manifest["user_id"])
    if not train_users or not val_users:
        raise ValueError("train_users and val_users must both be non-empty.")
    if train_users & val_users:
        raise ValueError("train_users and val_users overlap.")
    if train_users | val_users != manifest_users:
        raise ValueError("Every manifest user must belong to exactly one split side.")

    actual_train = manifest[manifest["user_id"].isin(train_users)]
    actual_val = manifest[manifest["user_id"].isin(val_users)]
    if len(actual_train) != int(fold["train_sample_count"]):
        raise ValueError("train_sample_count does not match manifest filtering.")
    if len(actual_val) != int(fold["val_sample_count"]):
        raise ValueError("val_sample_count does not match manifest filtering.")
    val_class_count = int(actual_val["class_id"].nunique())
    if val_class_count != int(fold["val_class_count"]):
        raise ValueError("val_class_count does not match manifest filtering.")
    if val_class_count != 40:
        raise ValueError(f"Validation split covers {val_class_count} classes, expected 40.")


def write_outputs(
    manifest: pd.DataFrame, fold: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    manifest_path = output_dir / "manifest.csv"
    fold_path = output_dir / "splits" / "fold_0.json"
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    fold_path.write_text(
        json.dumps(fold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path, fold_path


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = (args.output_dir or data_root / "metadata").resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Training data root does not exist: {data_root}")

    print(f"Training data root: {data_root}")
    mapping_path = find_class_mapping(data_root)
    if mapping_path is None:
        print(
            "WARNING: class_mapping.csv was not found in the train directory, its "
            "parent, or grandparent. Using numeric directory prefixes and verifying "
            "that all six modality mappings match."
        )
        official_mapping = None
    else:
        print(f"Using official class mapping: {mapping_path}")
        official_mapping = load_official_mapping(mapping_path)

    records, modality_counts = scan_modalities(data_root, official_mapping)
    manifest = build_manifest(records)
    validate_manifest(manifest, data_root)
    fold = build_fold(manifest)
    validate_fold(fold, manifest)
    manifest_path, fold_path = write_outputs(manifest, fold, output_dir)

    reloaded_manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype={
        "user_id": str,
        "trial_id": str,
    })
    reloaded_fold = json.loads(fold_path.read_text(encoding="utf-8"))
    validate_manifest(reloaded_manifest, data_root)
    validate_fold(reloaded_fold, reloaded_manifest)

    print(f"manifest rows: {len(reloaded_manifest)}")
    print(f"class_id count: {reloaded_manifest['class_id'].nunique()}")
    print(f"user_id count: {reloaded_manifest['user_id'].nunique()}")
    for modality, path_column in MODALITY_COLUMNS.items():
        non_empty = int(reloaded_manifest[path_column].fillna("").ne("").sum())
        print(f"{modality}: {non_empty} paths (scanned trials: {modality_counts[modality]})")
    print(f"train_users: {reloaded_fold['train_users']}")
    print(f"val_users: {reloaded_fold['val_users']}")
    print(f"train_sample_count: {reloaded_fold['train_sample_count']}")
    print(f"val_sample_count: {reloaded_fold['val_sample_count']}")
    print(f"val_class_count: {reloaded_fold['val_class_count']}")
    print("All manifest and fold validations passed.")
    print(f"manifest.csv: {manifest_path}")
    print(f"fold_0.json: {fold_path}")


if __name__ == "__main__":
    main()
