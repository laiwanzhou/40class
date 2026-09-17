from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax

from train_p46_videomae_head import l2_normalize
from train_p85_videomae_full40_head import metrics


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_FEATURES = (
    PROJECT_DIR / "runs/p85_videomae_large_fullwindow_full40_v1/complete_features.npz"
)
DEFAULT_TEST_FEATURES = (
    PROJECT_DIR / "runs/p85_videomae_large_fullwindow_test_v1/complete_features.npz"
)
DEFAULT_FUSION = PROJECT_DIR / "runs/p85_multiexpert_submission_v1/fusion_logits.npz"
DEFAULT_TEST_CSV = PROJECT_DIR.parent / "Testing/test.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3"
NEIGHBORS = 3
NEIGHBOR_WEIGHT = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the strongest frozen P85 teacher targets by applying anonymous "
            "full-window visual-neighbour consistency to the global fusion."
        )
    )
    parser.add_argument("--train-features", type=Path, default=DEFAULT_TRAIN_FEATURES)
    parser.add_argument("--test-features", type=Path, default=DEFAULT_TEST_FEATURES)
    parser.add_argument("--fusion", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def unit_features(values: np.ndarray) -> np.ndarray:
    features = l2_normalize(np.asarray(values, dtype=np.float32)).reshape(len(values), -1)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)


def top_neighbors(features: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    indices = np.argpartition(-similarity, count, axis=1)[:, :count]
    values = np.take_along_axis(similarity, indices, axis=1)
    order = np.argsort(-values, axis=1)
    return (
        np.take_along_axis(indices, order, axis=1),
        np.take_along_axis(values, order, axis=1),
    )


def smooth(
    probability: np.ndarray, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbors, similarities = top_neighbors(features, NEIGHBORS)
    nearby = probability[neighbors].mean(axis=1)
    result = (1.0 - NEIGHBOR_WEIGHT) * probability + NEIGHBOR_WEIGHT * nearby
    result /= result.sum(axis=1, keepdims=True)
    return result, neighbors, similarities


def official_id(path_value: str) -> str:
    return path_value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = load(args.train_features)
    test = load(args.test_features)
    fusion = load(args.fusion)

    train_ids = np.asarray(train["sample_ids"]).astype(str)
    fusion_train_ids = np.asarray(fusion["oof_sample_ids"]).astype(str)
    if not np.array_equal(train_ids, fusion_train_ids):
        raise RuntimeError("Full-window train features and fusion OOF order changed")
    labels = np.asarray(fusion["oof_labels"], dtype=np.int64)
    folds = np.asarray(fusion["oof_folds"], dtype=np.int64)
    oof_global_scores = np.asarray(fusion["oof_crossfit_scores"], dtype=np.float64)
    oof_global_probability = softmax(oof_global_scores, axis=1)
    oof_probability, oof_neighbors, oof_similarities = smooth(
        oof_global_probability, unit_features(train["features"])
    )
    oof_result = metrics(labels, oof_probability.argmax(axis=1))
    if int(oof_result["correct"]) != 2241:
        raise RuntimeError(f"Frozen full-window KNN OOF result changed: {oof_result}")

    test_feature_ids = np.asarray(test["sample_ids"]).astype(str)
    test_fusion_ids = np.asarray(fusion["test_sample_ids"]).astype(str)
    lookup = {value: index for index, value in enumerate(test_feature_ids)}
    if set(test_fusion_ids) != set(lookup):
        raise RuntimeError("Full-window Test features and fusion rows differ")
    feature_order = np.asarray([lookup[value] for value in test_fusion_ids], dtype=np.int64)
    test_global_scores = np.asarray(fusion["test_fused_scores"], dtype=np.float64)
    test_global_probability = softmax(test_global_scores, axis=1)
    test_probability, test_neighbors, test_similarities = smooth(
        test_global_probability, unit_features(test["features"])[feature_order]
    )

    target_path = output / "teacher_targets.npz"
    temporary = target_path.with_suffix(target_path.suffix + ".building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            oof_sample_ids=train_ids,
            oof_labels=labels,
            oof_folds=folds,
            oof_teacher_probability=oof_probability.astype(np.float32),
            oof_teacher_log_probability=np.log(
                np.clip(oof_probability, 1e-8, 1.0)
            ).astype(np.float32),
            oof_global_scores=oof_global_scores.astype(np.float32),
            oof_neighbors=oof_neighbors.astype(np.int32),
            oof_neighbor_similarities=oof_similarities.astype(np.float32),
            test_sample_ids=test_fusion_ids,
            test_teacher_probability=test_probability.astype(np.float32),
            test_teacher_log_probability=np.log(
                np.clip(test_probability, 1e-8, 1.0)
            ).astype(np.float32),
            test_global_scores=test_global_scores.astype(np.float32),
            test_neighbors=test_neighbors.astype(np.int32),
            test_neighbor_similarities=test_similarities.astype(np.float32),
        )
    temporary.replace(target_path)

    all_ids = np.asarray(fusion["test_all_sample_ids"]).astype(str)
    base_prediction = np.asarray(fusion["test_base_predictions"], dtype=np.int64)
    final_prediction = base_prediction.copy()
    all_lookup = {value: index for index, value in enumerate(all_ids)}
    visual_indices = np.asarray([all_lookup[value] for value in test_fusion_ids])
    final_prediction[visual_indices] = test_probability.argmax(axis=1)
    with args.test_csv.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        official_rows = list(csv.DictReader(handle))
    official_ids = [official_id(row["path"]) for row in official_rows]
    if len(official_ids) != 405 or set(official_ids) != set(all_lookup):
        raise RuntimeError("Official Test order changed")
    official_order = np.asarray([all_lookup[value] for value in official_ids])
    submission = output / "submission_p85_fullwindow_knn_teacher_v3.csv"
    write_csv(
        submission,
        [
            {"path": row["path"], "prediction": int(prediction)}
            for row, prediction in zip(
                official_rows, final_prediction[official_order], strict=True
            )
        ],
    )

    summary = {
        "protocol": (
            "Frozen global fusion plus anonymous full-window three-neighbour "
            "consistency; targets are intended for student distillation"
        ),
        "deployment_rule_note": (
            "The public Large VideoMAE model is permitted as a distillation teacher. "
            "This direct-teacher diagnostic CSV is not a compliant final deployment; "
            "the final student must not load teacher weights and all deployed weights "
            "must be packaged below 100 MB."
        ),
        "neighbors": NEIGHBORS,
        "neighbor_weight": NEIGHBOR_WEIGHT,
        "oof": oof_result,
        "teacher_targets": str(target_path),
        "diagnostic_submission": str(submission),
        "test_visual_rows": int(len(test_fusion_ids)),
        "test_missing_ir_p12_fallback": int(len(all_ids) - len(test_fusion_ids)),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
