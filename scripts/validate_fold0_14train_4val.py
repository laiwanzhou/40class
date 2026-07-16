from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "manifest.csv"
SPLIT_DIR = PROJECT_ROOT / "metadata" / "splits"
FOLD_PATH = SPLIT_DIR / "fold_0.json"
OLD_FOLD_PATH = SPLIT_DIR / "fold_0_12train_6val_20260715.json"
CANDIDATES_PATH = SPLIT_DIR / "fold_0_14train_4val_candidates.csv"
REPORT_PATH = SPLIT_DIR / "fold_0_14train_4val_selection_report.md"
MODALITY_COLUMNS = {
    "Depth_Color": "depth_color_path",
    "IMU": "imu_path",
    "IR": "ir_path",
    "Radar": "radar_path",
    "Skeleton": "skeleton_path",
    "Thermal": "thermal_path",
}


def present_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-11):
        raise ValueError(f"{label} mismatch: actual={actual}, expected={expected}")


def main() -> None:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8-sig",
        dtype={"user_id": str, "trial_id": str},
    )
    fold = json.loads(FOLD_PATH.read_text(encoding="utf-8"))
    archived = json.loads(OLD_FOLD_PATH.read_text(encoding="utf-8"))
    candidates = pd.read_csv(CANDIDATES_PATH, encoding="utf-8-sig")
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(REPORT_PATH)

    manifest_users = set(manifest["user_id"])
    train_users = set(fold["train_users"])
    val_users = set(fold["val_users"])
    if len(manifest_users) != 18:
        raise ValueError(f"Expected 18 manifest users, found {len(manifest_users)}")
    if len(train_users) != 14 or len(val_users) != 4:
        raise ValueError("New fold must contain 14 train users and 4 validation users.")
    if train_users & val_users:
        raise ValueError("Train and validation users overlap.")
    if train_users | val_users != manifest_users:
        raise ValueError("Train and validation users do not cover the manifest.")
    if fold.get("selection_uses_model_results") is not False:
        raise ValueError("Fold must declare that model results were not used.")
    if fold.get("selection_source") != "metadata/manifest.csv only":
        raise ValueError("Unexpected selection source.")

    train = manifest.loc[manifest["user_id"].isin(train_users)]
    val = manifest.loc[manifest["user_id"].isin(val_users)]
    if train["class_id"].nunique() != 40 or val["class_id"].nunique() != 40:
        raise ValueError("Union train and validation splits must both cover 40 classes.")
    if len(train) != int(fold["train_sample_count"]):
        raise ValueError("Union train sample count does not match JSON.")
    if len(val) != int(fold["val_sample_count"]):
        raise ValueError("Union validation sample count does not match JSON.")
    assert_close(len(val) / len(manifest), float(fold["val_sample_ratio"]), "union ratio")
    if int(fold["train_class_count"]) != 40 or int(fold["val_class_count"]) != 40:
        raise ValueError("JSON union class counts are invalid.")

    print(f"union: train={len(train)} val={len(val)} classes=40/40")
    for modality, column in MODALITY_COLUMNS.items():
        modality_frame = manifest.loc[present_mask(manifest[column])]
        modality_train = modality_frame.loc[modality_frame["user_id"].isin(train_users)]
        modality_val = modality_frame.loc[modality_frame["user_id"].isin(val_users)]
        actual = fold["per_modality_counts"][modality]
        if modality_train.empty or modality_val.empty:
            raise ValueError(f"Empty {modality} split.")
        if modality_train["class_id"].nunique() != 40:
            raise ValueError(f"{modality} train split does not cover 40 classes.")
        if len(modality_train) != int(actual["train_samples"]):
            raise ValueError(f"{modality} train sample count mismatch.")
        if len(modality_val) != int(actual["val_samples"]):
            raise ValueError(f"{modality} validation sample count mismatch.")
        val_class_count = int(modality_val["class_id"].nunique())
        if val_class_count != int(actual["val_class_count"]):
            raise ValueError(f"{modality} validation class count mismatch.")
        assert_close(
            len(modality_val) / len(modality_frame),
            float(actual["val_ratio"]),
            f"{modality} ratio",
        )
        val_class_counts = (
            modality_val.groupby("class_id").size().reindex(range(40), fill_value=0)
        )
        if int(val_class_counts.min()) != int(actual["min_val_class_samples"]):
            raise ValueError(f"{modality} minimum validation class count mismatch.")
        if int(val_class_counts.max()) != int(actual["max_val_class_samples"]):
            raise ValueError(f"{modality} maximum validation class count mismatch.")
        assert_close(
            float(val_class_counts.mean()),
            float(actual["mean_val_class_samples"]),
            f"{modality} mean validation class count",
        )
        print(
            f"{modality}: train={len(modality_train)} val={len(modality_val)} "
            f"classes=40/{val_class_count} ratio={len(modality_val) / len(modality_frame):.6f}"
        )

    if candidates.empty or int(candidates.iloc[0]["rank"]) != 1:
        raise ValueError("Candidate CSV does not begin with rank 1.")
    first = candidates.iloc[0]
    candidate_val_users = set(str(first["val_users"]).split(";"))
    candidate_train_users = set(str(first["train_users"]).split(";"))
    if candidate_val_users != val_users or candidate_train_users != train_users:
        raise ValueError("Candidate rank 1 does not match fold_0.json users.")
    assert_close(float(first["total_score"]), float(fold["selection_score"]), "selection score")
    if str(first["status"]) != "passed":
        raise ValueError("Rank 1 candidate did not pass hard constraints.")

    if len(archived.get("train_users", [])) != 12 or len(archived.get("val_users", [])) != 6:
        raise ValueError("Archived fold is not the original 12/6 split.")
    if set(archived["train_users"]) & set(archived["val_users"]):
        raise ValueError("Archived fold has overlapping users.")

    if not manifest["sample_id"].astype(str).str.startswith("train__").all():
        raise ValueError("Manifest contains non-training sample IDs.")
    for column in MODALITY_COLUMNS.values():
        for value in manifest.loc[present_mask(manifest[column]), column].astype(str):
            if any(
                part.casefold() in {"test", "testing"}
                for part in PurePosixPath(value).parts
            ):
                raise ValueError(f"Test path found in manifest: {value}")

    print(f"rank-1 val_users: {fold['val_users']}")
    print(f"selection score: {fold['selection_score']:.12f}")
    print("All 14-train/4-validation fold checks passed; no test path was accessed.")


if __name__ == "__main__":
    main()
