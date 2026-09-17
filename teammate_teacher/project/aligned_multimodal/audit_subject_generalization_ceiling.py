from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
RESEARCH_DOCS = REPO_ROOT / "docs" / "research"
MODALITIES = ("skeleton", "depth", "thermal", "imu")
DISTANCES = ("cosine", "standardized_euclidean")
FEATURE_NAMES = (
    "skeleton_embedding",
    "skeleton_logits",
    "depth_embedding",
    "depth_logits",
    "thermal_embedding",
    "thermal_logits",
    "imu_embedding",
    "imu_logits",
    "multimodal_embedding",
    "multimodal_logits",
)
PRIMARY_FEATURE = "multimodal_embedding"
PRIMARY_DISTANCE = "standardized_euclidean"
FOCUS_CLASS_IDS = (19, 24, 26, 25, 37, 21, 22, 9, 10, 12, 13)
FOCUS_PAIRS = (
    (19, 24),
    (24, 26),
    (25, 26),
    (37, 6),
    (21, 22),
    (9, 10),
    (12, 13),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P23 read-only same-subject separability and cross-subject prototype audit"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p22_joint_pooled_fusion" / "cache",
    )
    parser.add_argument(
        "--p22-config",
        type=Path,
        default=PROJECT_DIR / "configs" / "p22_joint_pooled_fusion.json",
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "subject_folds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p23_subject_generalization_audit",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=RESEARCH_DOCS
        / "10_local_and_domain"
        / "24_P23同主体可分性与跨主体迁移审计.md",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_class_names(fold_dir: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for path in sorted(fold_dir.glob("fold_*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                names[int(row["class_id"])] = row["class_name"].split("_", 1)[-1]
    if sorted(names) != list(range(40)):
        raise ValueError(f"Expected class IDs 0..39, got {sorted(names)}")
    return names


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_and_align_caches(
    cache_dir: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    paths = [cache_dir / f"fold_{fold}_cache.npz" for fold in range(3)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = {fold: load_npz(path) for fold, path in enumerate(paths)}
    required = {
        "sample_ids",
        "labels",
        "subjects",
        "folds",
        "outer_fold",
        "is_outer_train",
        "presence",
        "skeleton_embedding",
        "depth_embedding",
        "thermal_embedding",
        "imu_embedding",
        "per_modality_logits",
        "modality_order",
        "checkpoint_sha256",
        "manifest_sha256",
        "code_commit",
    }
    for fold, cache in raw.items():
        missing = required - set(cache)
        if missing:
            raise ValueError(f"fold {fold} cache missing fields: {sorted(missing)}")

    canonical_ids = np.asarray(
        sorted(raw[0]["sample_ids"].astype(str).tolist()),
        dtype=str,
    )
    if len(np.unique(canonical_ids)) != len(canonical_ids):
        raise ValueError("Duplicate sample_id in canonical cache")

    aligned: dict[int, dict[str, np.ndarray]] = {}
    identities: dict[str, Any] = {}
    metadata_fields = ("labels", "subjects", "folds", "presence")
    reference: dict[str, np.ndarray] | None = None
    for fold, path in enumerate(paths):
        cache = raw[fold]
        sample_ids = cache["sample_ids"].astype(str)
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError(f"Duplicate sample_id in fold {fold} cache")
        index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        if set(index_by_id) != set(canonical_ids.tolist()):
            raise ValueError(f"fold {fold} sample_id universe differs")
        order = np.asarray([index_by_id[sample_id] for sample_id in canonical_ids])

        reordered: dict[str, np.ndarray] = {}
        for key, value in cache.items():
            if value.ndim > 0 and value.shape[0] == len(sample_ids):
                reordered[key] = value[order]
            else:
                reordered[key] = value
        reordered["sample_ids"] = canonical_ids.copy()
        if int(reordered["outer_fold"]) != fold:
            raise ValueError(f"fold {fold} outer_fold metadata mismatch")
        expected_outer_train = (reordered["folds"].astype(int) != fold).astype(np.uint8)
        if not np.array_equal(reordered["is_outer_train"], expected_outer_train):
            raise ValueError(f"fold {fold} is_outer_train mismatch")
        if tuple(reordered["modality_order"].astype(str).tolist()) != MODALITIES:
            raise ValueError(f"fold {fold} modality order mismatch")

        if reference is None:
            reference = reordered
        else:
            for field in metadata_fields:
                if not np.array_equal(reference[field], reordered[field]):
                    raise ValueError(
                        f"fold {fold} metadata differs after sample_id join: {field}"
                    )
        aligned[fold] = reordered
        identities[str(fold)] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "cache_code_commit": str(reordered["code_commit"]),
            "checkpoint_sha256": json.loads(str(reordered["checkpoint_sha256"])),
            "manifest_sha256": json.loads(str(reordered["manifest_sha256"])),
        }

    assert reference is not None
    subjects = reference["subjects"].astype(str)
    folds = reference["folds"].astype(int)
    subject_folds: dict[str, int] = {}
    for subject in np.unique(subjects):
        values = np.unique(folds[subjects == subject])
        if len(values) != 1:
            raise ValueError(f"subject {subject} spans multiple folds: {values}")
        subject_folds[str(subject)] = int(values[0])

    audit = {
        "sample_alignment": "explicit sample_id dictionary join; no array-order assumption",
        "samples": int(len(canonical_ids)),
        "unique_sample_ids": int(len(np.unique(canonical_ids))),
        "subjects": int(len(subject_folds)),
        "subject_folds": subject_folds,
        "caches": identities,
    }
    return aligned, audit


def trial_parts(sample_id: str) -> tuple[str, int] | None:
    trial_id = sample_id.rsplit("__", 1)[-1]
    parts = trial_id.rsplit("-", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1])


def build_repeat_groups(
    sample_ids: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, list[tuple[int, int, int]]], dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[tuple[int, int]]] = defaultdict(list)
    unparsable: list[str] = []
    for index, (sample_id, subject, label) in enumerate(
        zip(sample_ids.astype(str), subjects.astype(str), labels.astype(int))
    ):
        parsed = trial_parts(sample_id)
        if parsed is None:
            unparsable.append(sample_id)
            continue
        stem, repeat = parsed
        grouped[(subject, int(label), stem)].append((repeat, index))

    complete_by_subject: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    incomplete_groups = 0
    incomplete_samples = 0
    complete_groups = 0
    for (subject, _label, _stem), members in sorted(grouped.items()):
        ordered = sorted(members)
        repeats = [repeat for repeat, _ in ordered]
        if len(ordered) == 3 and repeats == [1, 2, 3]:
            complete_by_subject[subject].append(
                tuple(index for _repeat, index in ordered)
            )
            complete_groups += 1
        else:
            incomplete_groups += 1
            incomplete_samples += len(ordered)

    coverage = {
        "all_groups": int(len(grouped)),
        "complete_triplets": int(complete_groups),
        "complete_triplet_samples": int(complete_groups * 3),
        "incomplete_groups": int(incomplete_groups),
        "incomplete_group_samples": int(incomplete_samples),
        "unparsable_sample_ids": unparsable,
    }
    return dict(complete_by_subject), coverage


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values, dtype=np.float32),
        where=norms > 1e-12,
    )


def build_feature(
    cache: dict[str, np.ndarray],
    feature_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    presence = cache["presence"].astype(bool)
    if feature_name.endswith("_embedding") and not feature_name.startswith("multimodal"):
        modality = feature_name.removesuffix("_embedding")
        index = MODALITIES.index(modality)
        return (
            cache[feature_name].astype(np.float32),
            presence[:, index],
        )
    if feature_name.endswith("_logits") and not feature_name.startswith("multimodal"):
        modality = feature_name.removesuffix("_logits")
        index = MODALITIES.index(modality)
        return (
            cache["per_modality_logits"][:, index].astype(np.float32),
            presence[:, index],
        )
    if feature_name == "multimodal_embedding":
        blocks: list[np.ndarray] = []
        for index, modality in enumerate(MODALITIES):
            block = normalize_rows(cache[f"{modality}_embedding"].astype(np.float32))
            block *= presence[:, index : index + 1]
            blocks.append(block)
        blocks.append(presence.astype(np.float32))
        return np.concatenate(blocks, axis=1), np.ones(len(presence), dtype=bool)
    if feature_name == "multimodal_logits":
        blocks = []
        for index, _modality in enumerate(MODALITIES):
            block = normalize_rows(
                cache["per_modality_logits"][:, index].astype(np.float32)
            )
            block *= presence[:, index : index + 1]
            blocks.append(block)
        blocks.append(presence.astype(np.float32))
        return np.concatenate(blocks, axis=1), np.ones(len(presence), dtype=bool)
    raise ValueError(f"Unknown feature: {feature_name}")


def prototype_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    distance: str,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(train_labels.astype(int))
    if distance == "cosine":
        train_transformed = normalize_rows(train_features.astype(np.float32))
        test_transformed = normalize_rows(test_features.astype(np.float32))
        prototypes = np.stack(
            [train_transformed[train_labels == class_id].mean(axis=0) for class_id in classes]
        )
        prototypes = normalize_rows(prototypes.astype(np.float32))
        scores = test_transformed @ prototypes.T
        predictions = classes[np.argmax(scores, axis=1)]
        return predictions.astype(np.int64), classes
    if distance == "standardized_euclidean":
        mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        train_transformed = (train_features - mean) / scale
        test_transformed = (test_features - mean) / scale
        prototypes = np.stack(
            [train_transformed[train_labels == class_id].mean(axis=0) for class_id in classes]
        ).astype(np.float32)
        test_norm = np.sum(test_transformed * test_transformed, axis=1, keepdims=True)
        prototype_norm = np.sum(prototypes * prototypes, axis=1)[None, :]
        distances = test_norm + prototype_norm - 2.0 * (test_transformed @ prototypes.T)
        predictions = classes[np.argmin(distances, axis=1)]
        return predictions.astype(np.int64), classes
    raise ValueError(distance)


def evaluate_same_subject(
    features_by_fold: dict[int, np.ndarray],
    valid_by_fold: dict[int, np.ndarray],
    reference: dict[str, np.ndarray],
    subject_folds: dict[str, int],
    complete_groups: dict[str, list[tuple[int, int, int]]],
    distance: str,
) -> dict[str, Any]:
    sample_ids = reference["sample_ids"].astype(str)
    labels = reference["labels"].astype(int)
    output_ids: list[str] = []
    output_labels: list[int] = []
    output_predictions: list[int] = []
    output_subjects: list[str] = []
    subject_coverage: dict[str, Any] = {}

    for subject in sorted(subject_folds):
        fold = subject_folds[subject]
        features = features_by_fold[fold]
        valid = valid_by_fold[fold]
        all_groups = complete_groups.get(subject, [])
        eligible_groups = [
            group for group in all_groups if bool(valid[np.asarray(group)].all())
        ]
        candidate_counts: list[int] = []
        for held_repeat in range(3):
            train_indices: list[int] = []
            test_indices: list[int] = []
            for group in eligible_groups:
                test_indices.append(group[held_repeat])
                train_indices.extend(
                    group[position] for position in range(3) if position != held_repeat
                )
            if not test_indices:
                continue
            train_array = np.asarray(train_indices, dtype=np.int64)
            test_array = np.asarray(test_indices, dtype=np.int64)
            predictions, classes = prototype_predict(
                features[train_array],
                labels[train_array],
                features[test_array],
                distance,
            )
            candidate_counts.append(int(len(classes)))
            output_ids.extend(sample_ids[test_array].tolist())
            output_labels.extend(labels[test_array].tolist())
            output_predictions.extend(predictions.tolist())
            output_subjects.extend([subject] * len(test_array))

        subject_coverage[subject] = {
            "complete_triplets": int(len(all_groups)),
            "feature_complete_triplets": int(len(eligible_groups)),
            "evaluated_samples": int(len(eligible_groups) * 3),
            "candidate_classes": (
                int(candidate_counts[0]) if candidate_counts else 0
            ),
        }

    result_ids = np.asarray(output_ids, dtype=str)
    if len(np.unique(result_ids)) != len(result_ids):
        raise ValueError("same-subject evaluation emitted duplicate test sample IDs")
    return {
        "sample_ids": result_ids,
        "labels": np.asarray(output_labels, dtype=np.int64),
        "predictions": np.asarray(output_predictions, dtype=np.int64),
        "subjects": np.asarray(output_subjects, dtype=str),
        "coverage": {
            "evaluated_samples": int(len(result_ids)),
            "feature_complete_triplets": int(len(result_ids) // 3),
            "subjects": subject_coverage,
            "candidate_class_rule": (
                "classes with a complete, modality-present triplet for that subject"
            ),
        },
    }


def evaluate_cross_subject(
    features_by_fold: dict[int, np.ndarray],
    valid_by_fold: dict[int, np.ndarray],
    reference: dict[str, np.ndarray],
    subject_folds: dict[str, int],
    distance: str,
) -> dict[str, np.ndarray]:
    sample_ids = reference["sample_ids"].astype(str)
    labels = reference["labels"].astype(int)
    subjects = reference["subjects"].astype(str)
    output_ids: list[str] = []
    output_labels: list[int] = []
    output_predictions: list[int] = []
    output_subjects: list[str] = []
    prototype_classes: dict[str, int] = {}

    for held_subject in sorted(subject_folds):
        fold = subject_folds[held_subject]
        features = features_by_fold[fold]
        valid = valid_by_fold[fold]
        train_mask = (subjects != held_subject) & valid
        test_mask = (subjects == held_subject) & valid
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        predictions, classes = prototype_predict(
            features[train_indices],
            labels[train_indices],
            features[test_indices],
            distance,
        )
        if len(classes) != 40:
            raise ValueError(
                f"held subject {held_subject} has only {len(classes)} cross-subject prototypes"
            )
        prototype_classes[held_subject] = int(len(classes))
        output_ids.extend(sample_ids[test_indices].tolist())
        output_labels.extend(labels[test_indices].tolist())
        output_predictions.extend(predictions.tolist())
        output_subjects.extend([held_subject] * len(test_indices))

    result_ids = np.asarray(output_ids, dtype=str)
    if len(np.unique(result_ids)) != len(result_ids):
        raise ValueError("cross-subject evaluation emitted duplicate test sample IDs")
    return {
        "sample_ids": result_ids,
        "labels": np.asarray(output_labels, dtype=np.int64),
        "predictions": np.asarray(output_predictions, dtype=np.int64),
        "subjects": np.asarray(output_subjects, dtype=str),
        "prototype_classes": prototype_classes,
    }


def select_ids(result: dict[str, Any], selected_ids: set[str]) -> dict[str, np.ndarray]:
    mask = np.asarray(
        [sample_id in selected_ids for sample_id in result["sample_ids"].astype(str)]
    )
    return {
        key: value[mask]
        for key, value in result.items()
        if isinstance(value, np.ndarray) and value.shape[:1] == mask.shape
    }


def core_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    if len(labels) == 0:
        return {
            "samples": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
        }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return {
            "samples": int(len(labels)),
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
        }


def metrics_bundle(
    labels: np.ndarray,
    predictions: np.ndarray,
    small_ids: list[int],
    hard_ids: list[int],
) -> dict[str, Any]:
    small_mask = np.isin(labels, np.asarray(small_ids))
    hard_mask = np.isin(labels, np.asarray(hard_ids))
    return {
        "all": core_metrics(labels, predictions),
        "small": core_metrics(labels[small_mask], predictions[small_mask]),
        "hard": core_metrics(labels[hard_mask], predictions[hard_mask]),
    }


def gap_pp(same: dict[str, Any], cross: dict[str, Any], subset: str) -> float | None:
    left = same[subset]["accuracy"]
    right = cross[subset]["accuracy"]
    if left is None or right is None:
        return None
    return float((left - right) * 100.0)


def per_class_rows(
    feature: str,
    distance: str,
    protocol: str,
    result: dict[str, np.ndarray],
    class_names: dict[int, str],
    small_ids: set[int],
    hard_ids: set[int],
) -> list[dict[str, Any]]:
    labels = result["labels"]
    predictions = result["predictions"]
    matrix = confusion_matrix(labels, predictions, labels=np.arange(40))
    rows: list[dict[str, Any]] = []
    for class_id in range(40):
        samples = int(matrix[class_id].sum())
        correct = int(matrix[class_id, class_id])
        rows.append(
            {
                "feature": feature,
                "distance": distance,
                "protocol": protocol,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "samples": samples,
                "correct": correct,
                "recall": (float(correct / samples) if samples else None),
                "is_small": int(class_id in small_ids),
                "is_hard": int(class_id in hard_ids),
                "is_focus": int(class_id in FOCUS_CLASS_IDS),
            }
        )
    return rows


def per_subject_rows(
    feature: str,
    distance: str,
    same: dict[str, np.ndarray],
    cross_matched: dict[str, np.ndarray],
    cross_all: dict[str, np.ndarray],
    small_ids: list[int],
    hard_ids: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subjects = sorted(np.unique(same["subjects"]).tolist())
    for subject in subjects:
        same_mask = same["subjects"] == subject
        matched_mask = cross_matched["subjects"] == subject
        all_mask = cross_all["subjects"] == subject
        same_metrics = metrics_bundle(
            same["labels"][same_mask],
            same["predictions"][same_mask],
            small_ids,
            hard_ids,
        )
        matched_metrics = metrics_bundle(
            cross_matched["labels"][matched_mask],
            cross_matched["predictions"][matched_mask],
            small_ids,
            hard_ids,
        )
        all_metrics = metrics_bundle(
            cross_all["labels"][all_mask],
            cross_all["predictions"][all_mask],
            small_ids,
            hard_ids,
        )
        rows.append(
            {
                "feature": feature,
                "distance": distance,
                "subject": subject,
                "same_samples": same_metrics["all"]["samples"],
                "same_accuracy": same_metrics["all"]["accuracy"],
                "cross_matched_samples": matched_metrics["all"]["samples"],
                "cross_matched_accuracy": matched_metrics["all"]["accuracy"],
                "gap_pp": gap_pp(same_metrics, matched_metrics, "all"),
                "same_small_accuracy": same_metrics["small"]["accuracy"],
                "cross_matched_small_accuracy": matched_metrics["small"]["accuracy"],
                "same_hard_accuracy": same_metrics["hard"]["accuracy"],
                "cross_matched_hard_accuracy": matched_metrics["hard"]["accuracy"],
                "cross_all_samples": all_metrics["all"]["samples"],
                "cross_all_accuracy": all_metrics["all"]["accuracy"],
            }
        )
    return rows


def top_confusion_rows(
    feature: str,
    distance: str,
    protocol: str,
    result: dict[str, np.ndarray],
    class_names: dict[int, str],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(result["labels"], result["predictions"], labels=np.arange(40))
    pairs: list[dict[str, Any]] = []
    focus_pairs = {tuple(sorted(pair)) for pair in FOCUS_PAIRS}
    for left in range(40):
        for right in range(left + 1, 40):
            left_to_right = int(matrix[left, right])
            right_to_left = int(matrix[right, left])
            total = left_to_right + right_to_left
            if total:
                pairs.append(
                    {
                        "feature": feature,
                        "distance": distance,
                        "protocol": protocol,
                        "class_a": left,
                        "class_a_name": class_names[left],
                        "class_b": right,
                        "class_b_name": class_names[right],
                        "a_to_b": left_to_right,
                        "b_to_a": right_to_left,
                        "total": total,
                        "is_focus_pair": int((left, right) in focus_pairs),
                    }
                )
    return sorted(
        pairs,
        key=lambda row: (-int(row["total"]), int(row["class_a"]), int(row["class_b"])),
    )[:top_k]


def focus_pair_rows(
    feature: str,
    distance: str,
    protocol: str,
    result: dict[str, np.ndarray],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(result["labels"], result["predictions"], labels=np.arange(40))
    rows: list[dict[str, Any]] = []
    for left, right in FOCUS_PAIRS:
        rows.append(
            {
                "feature": feature,
                "distance": distance,
                "protocol": protocol,
                "class_a": left,
                "class_a_name": class_names[left],
                "class_b": right,
                "class_b_name": class_names[right],
                "a_to_b": int(matrix[left, right]),
                "b_to_a": int(matrix[right, left]),
                "total": int(matrix[left, right] + matrix[right, left]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.2f}%"


def fmt_pp(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.2f} pp"


def render_report(
    summary: dict[str, Any],
    class_rows: list[dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# 24_P23同主体可分性与跨主体迁移审计",
        "",
        f"**协议：** `{summary['protocol']}`",
        "",
        f"**分析代码提交：** `{summary['code_commit']}`",
        "",
        "**限制遵守：** 未训练神经网络、未运行 encoder、未修改 checkpoint、"
        "未覆盖 OOF、未启动 P22-B。",
        "",
        "## 方法",
        "",
        "- 三份 P22 fold-conditioned cache 通过 sample_id 字典显式对齐；每个 held subject "
        "始终使用其所属 outer fold 的同一份 cache，使测试样本与其他主体 prototype 位于同一特征空间。",
        "- 同主体以 `(subject, class, trial stem)` 识别 1/2/3 三次重复；每轮两次建 prototype、"
        "一次测试。组内不足三次或模态缺失导致无法形成三连的样本跳过并计数。",
        "- 跨主体对每个 held subject 使用其余 17 个 subject 的标签与特征建立 40 类 prototype；"
        "held subject 的特征只用于测试、标签只用于最终计分。",
        "- cosine 使用逐样本 L2 归一化；标准化欧氏距离的 mean/std 只由当轮 prototype 训练样本估计。"
        "多模态拼接先分别 L2 归一化四个 present 模态块，再拼接四位 presence。",
        "- 主差值使用与同主体实验完全相同的 sample_id 子集；`cross_all` 作为全量覆盖结果另存。",
        "",
        "## 数据覆盖",
        "",
        f"- 样本：`{summary['input_audit']['samples']}`；subject："
        f"`{summary['input_audit']['subjects']}`；sample_id 唯一："
        f"`{summary['input_audit']['unique_sample_ids']}`。",
        f"- 重复组：`{summary['repeat_coverage']['all_groups']}`；完整三连："
        f"`{summary['repeat_coverage']['complete_triplets']}` 组 / "
        f"`{summary['repeat_coverage']['complete_triplet_samples']}` 样本；"
        f"不足三次：`{summary['repeat_coverage']['incomplete_groups']}` 组 / "
        f"`{summary['repeat_coverage']['incomplete_group_samples']}` 样本。",
        "- P22-F 的 256D 投影未保存在 cache，按“如可直接读取”限制排除；"
        "本次没有加载 P22-F checkpoint 或执行神经头。",
        "",
        "## Same-subject 与 cross-subject",
        "",
        "主表中的 cross 为匹配 same-subject 可评估 sample_id 后的严格跨主体 40 类 prototype 结果。",
        "",
        "| 特征 | 距离 | Same | Cross | 差值 | Same小动作 | Cross小动作 | Same困难类 | Cross困难类 | Same覆盖 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature in FEATURE_NAMES:
        for distance in DISTANCES:
            item = summary["results"][feature][distance]
            same = item["same_subject"]
            cross = item["cross_subject_matched"]
            lines.append(
                f"| {feature} | {distance} | {fmt_pct(same['all']['accuracy'])} | "
                f"{fmt_pct(cross['all']['accuracy'])} | {fmt_pp(item['gap_pp']['all'])} | "
                f"{fmt_pct(same['small']['accuracy'])} | {fmt_pct(cross['small']['accuracy'])} | "
                f"{fmt_pct(same['hard']['accuracy'])} | {fmt_pct(cross['hard']['accuracy'])} | "
                f"{same['all']['samples']}/{summary['input_audit']['samples']} |"
            )

    primary = summary["results"][PRIMARY_FEATURE][PRIMARY_DISTANCE]
    lines.extend(
        [
            "",
            "## 主分析与逐 subject",
            "",
            f"预注册主分析为 `{PRIMARY_FEATURE}` + `{PRIMARY_DISTANCE}`："
            f"overall 差值 `{fmt_pp(primary['gap_pp']['all'])}`，"
            f"小动作 `{fmt_pp(primary['gap_pp']['small'])}`，"
            f"困难类 `{fmt_pp(primary['gap_pp']['hard'])}`。",
            "",
            "| subject | Same | Cross matched | 差值 | Cross all | 样本 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    primary_subject_rows = [
        row
        for row in subject_rows
        if row["feature"] == PRIMARY_FEATURE and row["distance"] == PRIMARY_DISTANCE
    ]
    for row in sorted(primary_subject_rows, key=lambda item: str(item["subject"])):
        lines.append(
            f"| {row['subject']} | {fmt_pct(row['same_accuracy'])} | "
            f"{fmt_pct(row['cross_matched_accuracy'])} | {fmt_pp(row['gap_pp'])} | "
            f"{fmt_pct(row['cross_all_accuracy'])} | {row['same_samples']} |"
        )

    lines.extend(
        [
            "",
            "## 重点类别",
            "",
            f"以下为 `{PRIMARY_FEATURE}` 的同主体与匹配跨主体 recall。",
            "",
            "| 类别 | 距离 | Same recall | Cross recall | 差值 | Same样本 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    lookup = {
        (row["distance"], row["protocol"], row["class_id"]): row
        for row in class_rows
        if row["feature"] == PRIMARY_FEATURE
    }
    class_names = summary["class_names"]
    for class_id in FOCUS_CLASS_IDS:
        for distance in DISTANCES:
            same = lookup[(distance, "same_subject", class_id)]
            cross = lookup[(distance, "cross_subject_matched", class_id)]
            same_recall = same["recall"]
            cross_recall = cross["recall"]
            delta = (
                None
                if same_recall is None or cross_recall is None
                else (same_recall - cross_recall) * 100.0
            )
            lines.append(
                f"| {class_id} {class_names[str(class_id)]} | {distance} | "
                f"{fmt_pct(same_recall)} | {fmt_pct(cross_recall)} | "
                f"{fmt_pp(delta)} | {same['samples']} |"
            )

    primary_confusions = [
        row
        for row in confusion_rows
        if row["feature"] == PRIMARY_FEATURE
        and row["distance"] == PRIMARY_DISTANCE
        and row["protocol"] == "cross_subject_matched"
    ][:10]
    lines.extend(
        [
            "",
            "## 跨主体前 10 个双向混淆",
            "",
            "| 类别 A | 类别 B | A→B | B→A | 合计 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in primary_confusions:
        lines.append(
            f"| {row['class_a']} {row['class_a_name']} | "
            f"{row['class_b']} {row['class_b_name']} | {row['a_to_b']} | "
            f"{row['b_to_a']} | {row['total']} |"
        )

    decision = summary["decision"]
    lines.extend(
        [
            "",
            "## A/B/C 裁决",
            "",
            f"**裁决：{decision['code']} — {decision['title']}。**",
            "",
            decision["reason"],
            "",
            "## 后续唯一建议",
            "",
            decision["next_recommendation"],
            "",
            "完整主指标、每类 recall、逐 subject、前 20 双向混淆、重点 pair、"
            "覆盖率和逐样本预测保存在 "
            "`aligned_multimodal/runs/p23_subject_generalization_audit/`。",
            "",
        ]
    )
    return "\n".join(lines)


def decide(
    primary: dict[str, Any],
    primary_class_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    same_accuracy = primary["same_subject"]["all"]["accuracy"]
    same_hard = primary["same_subject"]["hard"]["accuracy"]
    overall_gap = primary["gap_pp"]["all"]
    if same_accuracy is None or same_hard is None or overall_gap is None:
        raise ValueError("Primary decision metrics are unavailable")

    evaluable = [row for row in primary_class_rows if row["samples"] > 0]
    high_classes = [
        row for row in evaluable if row["recall"] is not None and row["recall"] >= 0.50
    ]
    focus_low = [
        row["class_id"]
        for row in evaluable
        if row["class_id"] in FOCUS_CLASS_IDS
        and row["recall"] is not None
        and row["recall"] < 0.40
    ]
    high_class_fraction = len(high_classes) / len(evaluable) if evaluable else 0.0

    if overall_gap >= 15.0 and same_hard >= 0.50:
        return {
            "code": "A",
            "title": "主体域偏移为主",
            "reason": (
                f"主分析同主体比匹配跨主体高 {overall_gap:.2f} pp，且同主体困难类"
                f"准确率为 {same_hard * 100:.2f}%，达到预先固定的“差值≥15 pp 且"
                "困难类同主体≥50%”条件。"
            ),
            "high_class_fraction": high_class_fraction,
            "focus_low_same_subject": focus_low,
            "next_recommendation": (
                "只优先做一项：建立主体不变的 Skeleton 身体尺度/朝向与 IMU 方向/幅度"
                "归一化表示，再用同一 P23 prototype 协议验证跨主体差距是否收窄；本次不启动训练。"
            ),
        }
    if same_accuracy < 0.50 or same_hard < 0.40:
        return {
            "code": "B",
            "title": "信息本身不足",
            "reason": (
                f"主分析同主体 overall 为 {same_accuracy * 100:.2f}%，困难类为"
                f" {same_hard * 100:.2f}%，触发“overall<50% 或困难类<40%”规则；"
                "即使主体相同，当前 pooled embedding 也不足以稳定分开动作。"
            ),
            "high_class_fraction": high_class_fraction,
            "focus_low_same_subject": focus_low,
            "next_recommendation": (
                "只优先做一项：停止 pooled embedding 扩展，针对 G1/G2/G3 构造手—头、"
                "手—嘴、双手关系、停留时间、速度与 IMU 时序等显式特征并先做只读可分性验证。"
            ),
        }
    if same_accuracy >= 0.60 and high_class_fraction >= 0.60 and focus_low:
        return {
            "code": "C",
            "title": "只有少数类别仍不可分",
            "reason": (
                f"同主体 overall 为 {same_accuracy * 100:.2f}%，"
                f"{high_class_fraction * 100:.1f}% 的可评估类别 recall≥50%，但重点类别"
                f" {focus_low} 仍低于 40%，符合少数语义模糊/证据不足类别主导的模式。"
            ),
            "high_class_fraction": high_class_fraction,
            "focus_low_same_subject": focus_low,
            "next_recommendation": (
                "只优先做一项：对 Watch TV、Play games 及其 Phone/Read 邻类进行人工完整"
                "序列标签审计，先确认动作是否完整发生与标签边界，再决定是否建小专家。"
            ),
        }
    if overall_gap >= 15.0:
        return {
            "code": "A",
            "title": "主体域偏移为主（伴随残余类内不可分）",
            "reason": (
                f"同主体与匹配跨主体差值为 {overall_gap:.2f} pp，超过 15 pp；"
                f"同主体困难类为 {same_hard * 100:.2f}%，说明主体域偏移是主要损失源，"
                "但困难类仍有残余信息不足。"
            ),
            "high_class_fraction": high_class_fraction,
            "focus_low_same_subject": focus_low,
            "next_recommendation": (
                "只优先做一项：先验证无训练的 Skeleton 尺度/朝向与 IMU 方向/幅度归一化，"
                "并要求同一 P23 跨主体 prototype 指标改善后才考虑主体不变训练。"
            ),
        }
    return {
        "code": "C",
        "title": "少数类别与混合因素主导",
        "reason": (
            f"同主体 overall 为 {same_accuracy * 100:.2f}%，跨主体差值为"
            f" {overall_gap:.2f} pp，未触发纯 A 或纯 B；低 recall 重点类别为 {focus_low}。"
        ),
        "high_class_fraction": high_class_fraction,
        "focus_low_same_subject": focus_low,
        "next_recommendation": (
            "只优先做一项：对低 recall 重点类别进行人工完整序列标签审计，"
            "不扩展 pooled embedding 或启动后续训练。"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite P23 output directory: {args.output_dir}"
        )
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite P23 report: {args.report}")

    config = load_json(args.p22_config)
    small_ids = [int(value) for value in config["evaluation"]["small_action_ids"]]
    hard_ids = [int(value) for value in config["evaluation"]["hard_class_ids"]]
    small_set = set(small_ids)
    hard_set = set(hard_ids)
    class_names = load_class_names(args.fold_dir)
    caches, input_audit = load_and_align_caches(args.cache_dir)
    reference = caches[0]
    subject_folds = {
        subject: int(fold)
        for subject, fold in input_audit["subject_folds"].items()
    }
    complete_groups, repeat_coverage = build_repeat_groups(
        reference["sample_ids"],
        reference["subjects"],
        reference["labels"],
    )

    results: dict[str, Any] = {}
    all_class_rows: list[dict[str, Any]] = []
    all_subject_rows: list[dict[str, Any]] = []
    all_confusion_rows: list[dict[str, Any]] = []
    all_focus_pair_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    print(
        f"P23 input passed: {input_audit['samples']} sample_ids, "
        f"{repeat_coverage['complete_triplets']} complete triplets"
    )

    for feature_index, feature_name in enumerate(FEATURE_NAMES, start=1):
        print(f"[{feature_index}/{len(FEATURE_NAMES)}] {feature_name}", flush=True)
        features_by_fold: dict[int, np.ndarray] = {}
        valid_by_fold: dict[int, np.ndarray] = {}
        for fold, cache in caches.items():
            features, valid = build_feature(cache, feature_name)
            features_by_fold[fold] = features
            valid_by_fold[fold] = valid
        results[feature_name] = {}

        for distance in DISTANCES:
            same = evaluate_same_subject(
                features_by_fold,
                valid_by_fold,
                reference,
                subject_folds,
                complete_groups,
                distance,
            )
            cross_all = evaluate_cross_subject(
                features_by_fold,
                valid_by_fold,
                reference,
                subject_folds,
                distance,
            )
            same_ids = set(same["sample_ids"].astype(str).tolist())
            cross_matched = select_ids(cross_all, same_ids)
            if set(cross_matched["sample_ids"].astype(str).tolist()) != same_ids:
                raise ValueError(
                    f"{feature_name}/{distance} cross result cannot match same sample IDs"
                )
            same_metrics = metrics_bundle(
                same["labels"], same["predictions"], small_ids, hard_ids
            )
            cross_matched_metrics = metrics_bundle(
                cross_matched["labels"],
                cross_matched["predictions"],
                small_ids,
                hard_ids,
            )
            cross_all_metrics = metrics_bundle(
                cross_all["labels"], cross_all["predictions"], small_ids, hard_ids
            )
            feature_result = {
                "same_subject": same_metrics,
                "cross_subject_matched": cross_matched_metrics,
                "cross_subject_all": cross_all_metrics,
                "gap_pp": {
                    subset: gap_pp(same_metrics, cross_matched_metrics, subset)
                    for subset in ("all", "small", "hard")
                },
                "coverage": {
                    "same_subject": same["coverage"],
                    "cross_subject_all_samples": int(len(cross_all["sample_ids"])),
                    "cross_subject_matched_samples": int(
                        len(cross_matched["sample_ids"])
                    ),
                    "cross_prototype_classes_per_subject": cross_all[
                        "prototype_classes"
                    ],
                },
            }
            results[feature_name][distance] = feature_result

            protocol_results = {
                "same_subject": same,
                "cross_subject_matched": cross_matched,
                "cross_subject_all": cross_all,
            }
            for protocol, result in protocol_results.items():
                all_class_rows.extend(
                    per_class_rows(
                        feature_name,
                        distance,
                        protocol,
                        result,
                        class_names,
                        small_set,
                        hard_set,
                    )
                )
                all_confusion_rows.extend(
                    top_confusion_rows(
                        feature_name,
                        distance,
                        protocol,
                        result,
                        class_names,
                    )
                )
                all_focus_pair_rows.extend(
                    focus_pair_rows(
                        feature_name,
                        distance,
                        protocol,
                        result,
                        class_names,
                    )
                )
            all_subject_rows.extend(
                per_subject_rows(
                    feature_name,
                    distance,
                    same,
                    cross_matched,
                    cross_all,
                    small_ids,
                    hard_ids,
                )
            )

            key = f"{feature_name}__{distance}"
            for protocol, result in protocol_results.items():
                for field in ("sample_ids", "labels", "subjects", "predictions"):
                    predictions[f"{key}__{protocol}__{field}"] = result[field]

    primary_class_rows = [
        row
        for row in all_class_rows
        if row["feature"] == PRIMARY_FEATURE
        and row["distance"] == PRIMARY_DISTANCE
        and row["protocol"] == "same_subject"
    ]
    decision = decide(
        results[PRIMARY_FEATURE][PRIMARY_DISTANCE],
        primary_class_rows,
    )
    commit = git_commit()
    summary = {
        "status": "complete",
        "protocol": "p23-subject-generalization-ceiling-v1-fixed-before-run",
        "code_commit": commit,
        "script_sha256": sha256(Path(__file__)),
        "restrictions": {
            "neural_network_training": False,
            "encoder_inference": False,
            "checkpoint_modified": False,
            "existing_oof_overwritten": False,
            "p22_b_started": False,
            "post_result_tuning": False,
        },
        "input_audit": input_audit,
        "repeat_coverage": repeat_coverage,
        "feature_protocol": {
            "features": list(FEATURE_NAMES),
            "distances": list(DISTANCES),
            "same_subject": (
                "complete repeat triplets by subject/class/trial stem; two repeats "
                "form prototypes and the third is held out, rotating 3 ways"
            ),
            "cross_subject": (
                "held subject excluded from prototype features and labels; 40 class "
                "prototypes; standardization fitted only on other subjects"
            ),
            "gap": (
                "same-subject accuracy minus cross-subject accuracy on exactly the "
                "same evaluable sample_ids"
            ),
            "multimodal": (
                "per-modality row L2 normalization, missing block zero, concatenate "
                "four blocks and four presence bits"
            ),
            "p22f_projected": (
                "excluded because 256D projected features are not stored in P22 caches; "
                "P22-F checkpoint was not executed"
            ),
            "primary_decision_feature": PRIMARY_FEATURE,
            "primary_decision_distance": PRIMARY_DISTANCE,
            "decision_thresholds": {
                "A": "overall gap >= 15 pp and same-subject hard accuracy >= 50%",
                "B": "same-subject overall < 50% or hard accuracy < 40%",
                "C": (
                    "same-subject overall >= 60%, >=60% evaluable classes recall >=50%, "
                    "and at least one focus class recall <40%"
                ),
            },
        },
        "class_names": {str(key): value for key, value in class_names.items()},
        "small_action_ids": small_ids,
        "hard_class_ids": hard_ids,
        "focus_class_ids": list(FOCUS_CLASS_IDS),
        "focus_pairs": [list(pair) for pair in FOCUS_PAIRS],
        "results": results,
        "decision": decision,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "metrics.csv", [
        {
            "feature": feature,
            "distance": distance,
            "same_samples": results[feature][distance]["same_subject"]["all"]["samples"],
            "same_accuracy": results[feature][distance]["same_subject"]["all"]["accuracy"],
            "cross_matched_samples": results[feature][distance]["cross_subject_matched"]["all"]["samples"],
            "cross_matched_accuracy": results[feature][distance]["cross_subject_matched"]["all"]["accuracy"],
            "gap_pp": results[feature][distance]["gap_pp"]["all"],
            "same_small_accuracy": results[feature][distance]["same_subject"]["small"]["accuracy"],
            "cross_matched_small_accuracy": results[feature][distance]["cross_subject_matched"]["small"]["accuracy"],
            "small_gap_pp": results[feature][distance]["gap_pp"]["small"],
            "same_hard_accuracy": results[feature][distance]["same_subject"]["hard"]["accuracy"],
            "cross_matched_hard_accuracy": results[feature][distance]["cross_subject_matched"]["hard"]["accuracy"],
            "hard_gap_pp": results[feature][distance]["gap_pp"]["hard"],
            "cross_all_samples": results[feature][distance]["cross_subject_all"]["all"]["samples"],
            "cross_all_accuracy": results[feature][distance]["cross_subject_all"]["all"]["accuracy"],
        }
        for feature in FEATURE_NAMES
        for distance in DISTANCES
    ])
    write_csv(args.output_dir / "per_class_recall.csv", all_class_rows)
    write_csv(args.output_dir / "per_subject_accuracy.csv", all_subject_rows)
    write_csv(
        args.output_dir / "focus_classes.csv",
        [row for row in all_class_rows if row["is_focus"]],
    )
    write_csv(args.output_dir / "top20_bidirectional_confusions.csv", all_confusion_rows)
    write_csv(args.output_dir / "focus_pair_confusions.csv", all_focus_pair_rows)
    np.savez_compressed(args.output_dir / "predictions.npz", **predictions)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_text = render_report(
        summary,
        all_class_rows,
        all_subject_rows,
        all_confusion_rows,
    )
    args.report.write_text(report_text, encoding="utf-8")
    print(
        f"P23 complete: decision {decision['code']}; "
        f"report={args.report}; output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
