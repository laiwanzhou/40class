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
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
)
DEFAULT_TEST_FEATURES = (
    PROJECT_DIR / "runs/p46_videomae_large_multiclip_test_v1/complete_features.npz"
)
DEFAULT_FUSION = PROJECT_DIR / "runs/p85_multiexpert_submission_v1/fusion_logits.npz"
DEFAULT_TEST_CSV = PROJECT_DIR.parent / "Testing/test.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p85_knn_consistency_submission_v2"
NEIGHBORS = 2
NEIGHBOR_WEIGHT = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply label-free Test-batch visual-neighbour consistency to the P85 fusion."
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
    indices = np.take_along_axis(indices, order, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    return indices, values


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
        raise RuntimeError("Training features and fusion OOF order changed")
    train_features = unit_features(train["features"])
    train_scores = np.asarray(fusion["oof_crossfit_scores"], dtype=np.float64)
    train_probability = softmax(train_scores, axis=1)
    train_neighbors, _ = top_neighbors(train_features, NEIGHBORS)
    train_near_probability = train_probability[train_neighbors].mean(axis=1)
    train_smoothed = (
        (1.0 - NEIGHBOR_WEIGHT) * train_probability
        + NEIGHBOR_WEIGHT * train_near_probability
    )
    labels = np.asarray(fusion["oof_labels"], dtype=np.int64)
    oof_result = metrics(labels, train_smoothed.argmax(axis=1))
    if int(oof_result["correct"]) != 2234:
        raise RuntimeError(f"Frozen KNN OOF result changed: {oof_result}")

    test_feature_ids = np.asarray(test["sample_ids"]).astype(str)
    test_fusion_ids = np.asarray(fusion["test_sample_ids"]).astype(str)
    lookup = {value: index for index, value in enumerate(test_feature_ids)}
    if set(test_fusion_ids) != set(lookup):
        raise RuntimeError("Test feature and fusion coverage changed")
    feature_order = np.asarray([lookup[value] for value in test_fusion_ids], dtype=np.int64)
    test_features = unit_features(test["features"])[feature_order]
    test_scores = np.asarray(fusion["test_fused_scores"], dtype=np.float64)
    test_probability = softmax(test_scores, axis=1)
    test_neighbors, similarities = top_neighbors(test_features, NEIGHBORS)
    near_probability = test_probability[test_neighbors].mean(axis=1)
    smoothed = (
        (1.0 - NEIGHBOR_WEIGHT) * test_probability
        + NEIGHBOR_WEIGHT * near_probability
    )
    available_prediction = smoothed.argmax(axis=1)

    all_ids = np.asarray(fusion["test_all_sample_ids"]).astype(str)
    base_prediction = np.asarray(fusion["test_base_predictions"], dtype=np.int64)
    final_prediction = base_prediction.copy()
    all_lookup = {value: index for index, value in enumerate(all_ids)}
    available_all_indices = np.asarray([all_lookup[value] for value in test_fusion_ids])
    global_prediction = base_prediction.copy()
    global_prediction[available_all_indices] = test_scores.argmax(axis=1)
    final_prediction[available_all_indices] = available_prediction

    with args.test_csv.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        official_rows = list(csv.DictReader(handle))
    official_ids = [official_id(row["path"]) for row in official_rows]
    if len(official_rows) != 405 or set(official_ids) != set(all_lookup):
        raise RuntimeError("Official Test order changed")
    official_order = np.asarray([all_lookup[value] for value in official_ids])
    submission_rows = [
        {"path": row["path"], "prediction": int(value)}
        for row, value in zip(official_rows, final_prediction[official_order], strict=True)
    ]
    submission = output / "submission_p85_knn_consistency_v2.csv"
    write_csv(submission, submission_rows)

    audit_rows = []
    for index, sample_id in enumerate(test_fusion_ids):
        audit_rows.append(
            {
                "sample_id": sample_id,
                "global_prediction": int(test_scores[index].argmax()),
                "knn_prediction": int(available_prediction[index]),
                "neighbor_1": test_fusion_ids[test_neighbors[index, 0]],
                "neighbor_1_similarity": float(similarities[index, 0]),
                "neighbor_2": test_fusion_ids[test_neighbors[index, 1]],
                "neighbor_2_similarity": float(similarities[index, 1]),
            }
        )
    write_csv(output / "test_knn_audit.csv", audit_rows)
    summary = {
        "protocol": "anonymous Test-batch two-neighbour consistency using only IR embeddings and model probabilities",
        "deployment_rule_note": (
            "The public Large VideoMAE model is allowed as a knowledge-distillation "
            "teacher, but this direct-teacher inference artifact is not a compliant "
            "final deployment. Final inference weights, including ensembles, must be "
            "packaged below 100 MB."
        ),
        "neighbors": NEIGHBORS,
        "neighbor_weight": NEIGHBOR_WEIGHT,
        "oof": oof_result,
        "test": {
            "rows": len(all_ids),
            "visual_rows": len(test_fusion_ids),
            "changed_from_global_fusion": int(np.sum(final_prediction != global_prediction)),
            "changed_from_p12": int(np.sum(final_prediction != base_prediction)),
            "missing_ir_p12_fallback": int(len(all_ids) - len(test_fusion_ids)),
            "submission": str(submission),
            "audit": str(output / "test_knn_audit.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
