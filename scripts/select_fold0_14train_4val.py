from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

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
TARGET_VAL_USERS = 4
TARGET_TRAIN_USERS = 14
TARGET_VAL_RATIO = 0.20
EXPECTED_USERS = 18
EXPECTED_CLASSES = 40
TOP_CANDIDATES = 50

# Total score is the weighted sum of these structural terms. Model results are
# deliberately unavailable to this script.
WEIGHTS = {
    "coverage_penalty_per_missing_val_modality_class": 100.0,
    "union_ratio_error": 8.0,
    "mean_modality_ratio_error": 8.0,
    "union_class_distribution_error": 4.0,
    "mean_modality_class_distribution_error": 4.0,
    "user_count_imbalance": 1.0,
    "modality_missing_rate_error": 4.0,
    "min_class_penalty": 0.5,
}


def natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def sorted_users(users: Iterable[str]) -> list[str]:
    return sorted(set(users), key=natural_key)


def present_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8-sig",
        dtype={"user_id": str, "trial_id": str},
    )
    required = {"sample_id", "class_id", "user_id", *MODALITY_COLUMNS.values()}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    manifest["class_id"] = manifest["class_id"].astype(int)
    users = sorted_users(manifest["user_id"])
    if len(users) != EXPECTED_USERS:
        raise ValueError(f"Expected 18 manifest users, found {len(users)}: {users}")
    classes = set(manifest["class_id"])
    if classes != set(range(EXPECTED_CLASSES)):
        raise ValueError(f"Expected class IDs 0-39, found {sorted(classes)}")
    if not manifest["sample_id"].astype(str).str.startswith("train__").all():
        raise ValueError("Manifest contains a non-training sample_id.")
    for column in MODALITY_COLUMNS.values():
        for value in manifest.loc[present_mask(manifest[column]), column].astype(str):
            parts = PurePosixPath(value).parts
            if any(part.casefold() in {"test", "testing"} for part in parts):
                raise ValueError(f"Test path found in training manifest: {value}")
    return manifest


def archive_old_fold() -> str:
    if not FOLD_PATH.is_file():
        raise FileNotFoundError(f"Current fold is missing: {FOLD_PATH}")
    if not OLD_FOLD_PATH.exists():
        old_bytes = FOLD_PATH.read_bytes()
        old_fold = json.loads(old_bytes.decode("utf-8"))
        if len(old_fold.get("train_users", [])) != 12 or len(old_fold.get("val_users", [])) != 6:
            raise ValueError("Current fold is not the expected 12-train/6-val fold.")
        OLD_FOLD_PATH.write_bytes(old_bytes)
    archived = json.loads(OLD_FOLD_PATH.read_text(encoding="utf-8"))
    if len(archived.get("train_users", [])) != 12 or len(archived.get("val_users", [])) != 6:
        raise ValueError("Archived fold is not the expected 12-train/6-val fold.")
    return hashlib.sha256(OLD_FOLD_PATH.read_bytes()).hexdigest().upper()


def class_counts(frame: pd.DataFrame) -> np.ndarray:
    return np.bincount(
        frame["class_id"].to_numpy(dtype=np.int64), minlength=EXPECTED_CLASSES
    ).astype(np.int64)


def distribution_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    if candidate.sum() <= 0 or reference.sum() <= 0:
        return 1.0
    candidate_distribution = candidate / candidate.sum()
    reference_distribution = reference / reference.sum()
    return float(np.abs(candidate_distribution - reference_distribution).mean())


def build_statistics(manifest: pd.DataFrame) -> dict[str, Any]:
    users = sorted_users(manifest["user_id"])
    user_index = {user: index for index, user in enumerate(users)}

    def aggregate(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        counts = np.zeros(len(users), dtype=np.int64)
        classes = np.zeros((len(users), EXPECTED_CLASSES), dtype=np.int64)
        for row in frame[["user_id", "class_id"]].itertuples(index=False):
            index = user_index[str(row.user_id)]
            counts[index] += 1
            classes[index, int(row.class_id)] += 1
        return counts, classes

    union_counts, union_classes = aggregate(manifest)
    modalities: dict[str, dict[str, Any]] = {}
    for modality, column in MODALITY_COLUMNS.items():
        frame = manifest.loc[present_mask(manifest[column])]
        counts, classes = aggregate(frame)
        modalities[modality] = {
            "column": column,
            "user_counts": counts,
            "user_classes": classes,
            "total_samples": int(len(frame)),
            "total_classes": class_counts(frame),
        }
    return {
        "users": users,
        "union_user_counts": union_counts,
        "union_user_classes": union_classes,
        "union_total_samples": int(len(manifest)),
        "union_total_classes": class_counts(manifest),
        "modalities": modalities,
    }


def evaluate_candidate(statistics: dict[str, Any], indices: tuple[int, ...]) -> dict[str, Any]:
    users: list[str] = statistics["users"]
    index_set = set(indices)
    val_users = [users[index] for index in indices]
    train_users = [user for index, user in enumerate(users) if index not in index_set]

    union_user_counts: np.ndarray = statistics["union_user_counts"]
    union_user_classes: np.ndarray = statistics["union_user_classes"]
    union_val_samples = int(union_user_counts[list(indices)].sum())
    union_train_samples = int(statistics["union_total_samples"] - union_val_samples)
    union_val_classes = union_user_classes[list(indices)].sum(axis=0)
    union_train_classes = statistics["union_total_classes"] - union_val_classes
    union_val_class_count = int(np.count_nonzero(union_val_classes))
    union_train_class_count = int(np.count_nonzero(union_train_classes))
    union_val_ratio = union_val_samples / statistics["union_total_samples"]
    union_ratio_error = abs(union_val_ratio - TARGET_VAL_RATIO)
    union_distribution_error = distribution_error(
        union_val_classes, statistics["union_total_classes"]
    )

    selected_user_counts = union_user_counts[list(indices)].astype(np.float64)
    user_count_imbalance = float(
        selected_user_counts.std(ddof=0) / selected_user_counts.mean()
    )
    modality_rows: dict[str, dict[str, Any]] = {}
    modality_ratio_errors: list[float] = []
    modality_distribution_errors: list[float] = []
    missing_rate_errors: list[float] = []
    modality_min_class_samples: list[int] = []
    missing_val_modality_classes = 0
    hard_errors: list[str] = []

    if len(train_users) != TARGET_TRAIN_USERS or len(val_users) != TARGET_VAL_USERS:
        hard_errors.append("user_count")
    if union_train_class_count != EXPECTED_CLASSES:
        hard_errors.append("union_train_class_coverage")
    if union_val_class_count != EXPECTED_CLASSES:
        hard_errors.append("union_val_class_coverage")

    for modality, values in statistics["modalities"].items():
        user_counts: np.ndarray = values["user_counts"]
        user_classes: np.ndarray = values["user_classes"]
        val_samples = int(user_counts[list(indices)].sum())
        train_samples = int(values["total_samples"] - val_samples)
        val_classes = user_classes[list(indices)].sum(axis=0)
        train_classes = values["total_classes"] - val_classes
        val_class_count = int(np.count_nonzero(val_classes))
        train_class_count = int(np.count_nonzero(train_classes))
        val_ratio = val_samples / values["total_samples"]
        ratio_error = abs(val_ratio - TARGET_VAL_RATIO)
        modality_distribution_error = distribution_error(
            val_classes, values["total_classes"]
        )
        full_missing_rate = 1.0 - values["total_samples"] / statistics["union_total_samples"]
        val_missing_rate = 1.0 - val_samples / union_val_samples
        missing_rate_error = abs(val_missing_rate - full_missing_rate)
        min_class_samples = int(val_classes.min())

        if train_samples <= 0 or val_samples <= 0:
            hard_errors.append(f"{modality}_empty")
        if train_class_count != EXPECTED_CLASSES:
            hard_errors.append(f"{modality}_train_class_coverage")
        missing_val_modality_classes += EXPECTED_CLASSES - val_class_count
        modality_ratio_errors.append(ratio_error)
        modality_distribution_errors.append(modality_distribution_error)
        missing_rate_errors.append(missing_rate_error)
        modality_min_class_samples.append(min_class_samples)
        modality_rows[modality] = {
            "train_samples": train_samples,
            "val_samples": val_samples,
            "val_ratio": val_ratio,
            "train_class_count": train_class_count,
            "val_class_count": val_class_count,
            "min_val_class_samples": min_class_samples,
            "max_val_class_samples": int(val_classes.max()),
            "mean_val_class_samples": float(val_classes.mean()),
        }

    coverage_penalty = (
        missing_val_modality_classes
        * WEIGHTS["coverage_penalty_per_missing_val_modality_class"]
    )
    mean_modality_ratio_error = float(np.mean(modality_ratio_errors))
    mean_modality_distribution_error = float(np.mean(modality_distribution_errors))
    missing_rate_error = float(np.mean(missing_rate_errors))
    min_val_class_samples_union = int(union_val_classes.min())
    min_class_penalty = 1.0 / (1.0 + min_val_class_samples_union) + float(
        np.mean([1.0 / (1.0 + value) for value in modality_min_class_samples])
    )
    total_score = (
        coverage_penalty
        + WEIGHTS["union_ratio_error"] * union_ratio_error
        + WEIGHTS["mean_modality_ratio_error"] * mean_modality_ratio_error
        + WEIGHTS["union_class_distribution_error"] * union_distribution_error
        + WEIGHTS["mean_modality_class_distribution_error"]
        * mean_modality_distribution_error
        + WEIGHTS["user_count_imbalance"] * user_count_imbalance
        + WEIGHTS["modality_missing_rate_error"] * missing_rate_error
        + WEIGHTS["min_class_penalty"] * min_class_penalty
    )
    return {
        "val_users_list": val_users,
        "train_users_list": train_users,
        "val_users": ";".join(val_users),
        "train_users": ";".join(train_users),
        "total_score": total_score,
        "coverage_penalty": coverage_penalty,
        "union_train_samples": union_train_samples,
        "union_val_samples": union_val_samples,
        "union_val_ratio": union_val_ratio,
        "union_train_class_count": union_train_class_count,
        "union_val_class_count": union_val_class_count,
        "min_val_class_samples_union": min_val_class_samples_union,
        "union_max_val_class_samples": int(union_val_classes.max()),
        "union_mean_val_class_samples": float(union_val_classes.mean()),
        "union_ratio_error": union_ratio_error,
        "mean_modality_ratio_error": mean_modality_ratio_error,
        "union_class_distribution_error": union_distribution_error,
        "mean_modality_class_distribution_error": mean_modality_distribution_error,
        "class_distribution_error": (
            union_distribution_error + mean_modality_distribution_error
        ),
        "user_count_imbalance": user_count_imbalance,
        "missing_rate_error": missing_rate_error,
        "min_class_penalty": min_class_penalty,
        "modalities": modality_rows,
        "status": "passed" if not hard_errors else "rejected:" + ";".join(hard_errors),
    }


def evaluate_all(statistics: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        evaluate_candidate(statistics, indices)
        for indices in itertools.combinations(range(len(statistics["users"])), TARGET_VAL_USERS)
    ]
    candidates.sort(
        key=lambda row: (
            row["status"] != "passed",
            float(row["total_score"]),
            tuple(sorted(row["val_users_list"])),
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    return candidates


def candidate_csv_row(candidate: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: candidate[key]
        for key in (
            "rank",
            "val_users",
            "train_users",
            "total_score",
            "coverage_penalty",
            "union_train_samples",
            "union_val_samples",
            "union_val_ratio",
            "union_train_class_count",
            "union_val_class_count",
        )
    }
    for modality in MODALITY_COLUMNS:
        prefix = modality.casefold()
        values = candidate["modalities"][modality]
        for field in (
            "train_samples",
            "val_samples",
            "val_ratio",
            "train_class_count",
            "val_class_count",
            "min_val_class_samples",
            "max_val_class_samples",
            "mean_val_class_samples",
        ):
            row[f"{prefix}_{field}"] = values[field]
    for key in (
        "min_val_class_samples_union",
        "union_ratio_error",
        "mean_modality_ratio_error",
        "union_class_distribution_error",
        "mean_modality_class_distribution_error",
        "class_distribution_error",
        "user_count_imbalance",
        "missing_rate_error",
        "min_class_penalty",
        "status",
    ):
        row[key] = candidate[key]
    return row


def build_fold(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "fold": 0,
        "method": "ExhaustiveUserGroupHoldout",
        "selection_version": "14train_4val_v1",
        "group_field": "user_id",
        "label_field": "class_id",
        "target_train_user_count": TARGET_TRAIN_USERS,
        "target_val_user_count": TARGET_VAL_USERS,
        "target_val_sample_ratio": TARGET_VAL_RATIO,
        "selection_source": "metadata/manifest.csv only",
        "selection_uses_model_results": False,
        "train_users": candidate["train_users_list"],
        "val_users": candidate["val_users_list"],
        "train_sample_count": candidate["union_train_samples"],
        "val_sample_count": candidate["union_val_samples"],
        "val_sample_ratio": candidate["union_val_ratio"],
        "train_class_count": candidate["union_train_class_count"],
        "val_class_count": candidate["union_val_class_count"],
        "per_modality_counts": candidate["modalities"],
        "selection_score": candidate["total_score"],
        "candidate_rank": 1,
    }


def format_users(users: list[str]) -> str:
    return ", ".join(users)


def build_report(
    statistics: dict[str, Any],
    candidates: list[dict[str, Any]],
    backup_sha256: str,
) -> str:
    winner = candidates[0]
    runner_up = candidates[1]
    eligible_count = sum(candidate["status"] == "passed" for candidate in candidates)
    lines = [
        "# fold_0 14-train / 4-validation user selection",
        "",
        "## Scope",
        "",
        f"The manifest contains {len(statistics['users'])} users: {format_users(statistics['users'])}.",
        f"All C(18,4) = {len(candidates)} validation-user combinations were evaluated; {eligible_count} passed the hard constraints.",
        "The selection uses only metadata/manifest.csv structure. It does not use model predictions, Accuracy, loss, checkpoints, confusion matrices, or any test-set information.",
        "",
        "## Hard constraints",
        "",
        "- Exactly 14 train users and 4 validation users, with no overlap and complete 18-user coverage.",
        "- Union train and validation splits must each cover all 40 classes.",
        "- Every modality must have non-empty train and validation samples.",
        "- Every modality train split must cover all 40 classes.",
        "- Missing validation classes in an individual modality are heavily penalized; full 40-class modality validation coverage is preferred.",
        "- Modality presence is defined exactly as series.fillna(\"\").astype(str).str.strip().ne(\"\").",
        "",
        "## Score",
        "",
        "Class-distribution error is mean absolute difference: MAD(p,q) = (1/40) * sum_c |p_c - q_c|, comparing normalized candidate-validation counts with normalized full-manifest counts.",
        "User-count imbalance is the coefficient of variation of the four users' union sample counts. Missing-rate error is the mean absolute difference between candidate-validation and full-manifest modality missing rates.",
        "The minimum-class penalty is 1/(1+union minimum) plus the mean of 1/(1+modality minimum), so larger minimum class counts are preferred.",
        "",
        "total_score = coverage_penalty + 8*union_ratio_error + 8*mean_modality_ratio_error + 4*union_class_distribution_error + 4*mean_modality_class_distribution_error + 1*user_count_imbalance + 4*missing_rate_error + 0.5*min_class_penalty.",
        "coverage_penalty = 100 times the number of missing validation classes summed across the six modalities. Lower score is better; exact ties use the lexicographically sorted val_users tuple.",
        "",
        "| Weight | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {value:g} |" for key, value in WEIGHTS.items())
    lines.extend(
        [
            "",
            "## Selected split",
            "",
            f"- Validation users: {format_users(winner['val_users_list'])}",
            f"- Train users: {format_users(winner['train_users_list'])}",
            f"- Union train/validation: {winner['union_train_samples']}/{winner['union_val_samples']} (validation ratio {winner['union_val_ratio']:.6f})",
            f"- Selection score: {winner['total_score']:.12f}",
            "",
            "| Modality | Train | Validation | Validation ratio | Train classes | Validation classes | Minimum validation class |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for modality in MODALITY_COLUMNS:
        values = winner["modalities"][modality]
        lines.append(
            f"| {modality} | {values['train_samples']} | {values['val_samples']} | "
            f"{values['val_ratio']:.6f} | {values['train_class_count']} | "
            f"{values['val_class_count']} | {values['min_val_class_samples']} |"
        )
    lines.extend(
        [
            "",
            "## Validation class-count summary",
            "",
            "| Population | Minimum | Maximum | Mean | Class coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Union | {winner['min_val_class_samples_union']} | {winner['union_max_val_class_samples']} | {winner['union_mean_val_class_samples']:.3f} | {winner['union_val_class_count']} |",
        ]
    )
    for modality in MODALITY_COLUMNS:
        values = winner["modalities"][modality]
        lines.append(
            f"| {modality} | {values['min_val_class_samples']} | "
            f"{values['max_val_class_samples']} | {values['mean_val_class_samples']:.3f} | "
            f"{values['val_class_count']} |"
        )
    lines.extend(
        [
            "",
            "## Top 10 candidates",
            "",
            "| Rank | Validation users | Score | Union ratio | Mean modality ratio error | Class distribution error | Missing-rate error | Union minimum class |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in candidates[:10]:
        lines.append(
            f"| {candidate['rank']} | {candidate['val_users'].replace(';', ', ')} | "
            f"{candidate['total_score']:.12f} | {candidate['union_val_ratio']:.6f} | "
            f"{candidate['mean_modality_ratio_error']:.6f} | "
            f"{candidate['class_distribution_error']:.6f} | "
            f"{candidate['missing_rate_error']:.6f} | "
            f"{candidate['min_val_class_samples_union']} |"
        )
    lines.extend(
        [
            "",
            "## Selection rationale",
            "",
            f"Candidate 1 has the lowest deterministic structural score ({winner['total_score']:.12f}), ahead of candidate 2 ({runner_up['total_score']:.12f}) by {runner_up['total_score'] - winner['total_score']:.12f}. The ranking jointly balances sample ratios, class distributions, per-user sample imbalance, modality missing rates, class coverage, and minimum class support; it is not a model-performance ranking.",
            "",
            "## Provenance and archive",
            "",
            f"The old 12-train/6-validation fold was archived byte-for-byte at metadata/splits/{OLD_FOLD_PATH.name} (SHA-256 {backup_sha256}).",
            "No test directory was read. No model result or prediction artifact was read. The old formal Baseline reports and outputs remain unchanged.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    manifest = load_manifest()
    backup_sha256 = archive_old_fold()
    statistics = build_statistics(manifest)
    candidates = evaluate_all(statistics)
    expected_combinations = 3060
    if len(candidates) != expected_combinations:
        raise ValueError(f"Expected 3060 combinations, found {len(candidates)}")
    if candidates[0]["status"] != "passed":
        raise ValueError("No candidate passed the hard constraints.")

    candidate_rows = [candidate_csv_row(row) for row in candidates[:TOP_CANDIDATES]]
    pd.DataFrame(candidate_rows).to_csv(
        CANDIDATES_PATH,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format="%.12f",
    )
    FOLD_PATH.write_text(
        json.dumps(build_fold(candidates[0]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        build_report(statistics, candidates, backup_sha256), encoding="utf-8"
    )

    winner = candidates[0]
    print(f"Evaluated combinations: {len(candidates)}")
    print(f"Passed hard constraints: {sum(row['status'] == 'passed' for row in candidates)}")
    print(f"val_users: {winner['val_users_list']}")
    print(f"train_users: {winner['train_users_list']}")
    print(f"union validation ratio: {winner['union_val_ratio']:.6f}")
    print(f"selection score: {winner['total_score']:.12f}")
    print(f"old fold archive SHA-256: {backup_sha256}")
    print(FOLD_PATH)
    print(CANDIDATES_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
