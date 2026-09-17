from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import softmax

from p46_protocol import HARD_CLASS_IDS
from train_p46_videomae_head import aligned_scores, l2_normalize, make_model
from train_p46_videomae_large_weighted import sample_weights


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_FEATURES = (
    PROJECT_DIR / "runs/p46_videomae_large_multiclip_v1/complete_features.npz"
)
DEFAULT_TEST_FEATURES = (
    PROJECT_DIR / "runs/p46_videomae_large_multiclip_test_v1/complete_features.npz"
)
DEFAULT_P12_TEST = PROJECT_DIR / "runs/p11_final_package/test_candidate/test_logits.npz"
DEFAULT_TEST_CSV = PROJECT_DIR.parent / "Testing/test.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_test_deploy_v1"

# This was selected by grouped CV on the 14 P46 training users.  Among the
# independently deployable temporal heads it then obtained 193/290 = 66.55%
# on the four untouched P46 validation users.
ALPHA = 3000.0
CLASS_WEIGHT_POWER = 0.75
TEMPERATURE = 0.2002252480466578
BLENDS = {
    # Selected on the four-user development validation set for final deployment.
    "development_w055": 0.55,
    # Protocol-strict alternative selected from training-user OOF behaviour,
    # without using development labels to choose the blend strength.
    "strict_w070": 0.70,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the frozen P46 early+late head on all labeled Detail21 data and "
            "safely re-rank only P12 predictions already inside Detail21."
        )
    )
    parser.add_argument("--train-features", type=Path, default=DEFAULT_TRAIN_FEATURES)
    parser.add_argument("--test-features", type=Path, default=DEFAULT_TEST_FEATURES)
    parser.add_argument("--p12-test", type=Path, default=DEFAULT_P12_TEST)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def early_late_matrix(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (2, 3, 1024):
        raise RuntimeError(f"Expected [N,2,3,1024] early/late features, got {values.shape}")
    return l2_normalize(values).reshape(len(values), -1)


def official_id(path_value: str) -> str:
    clean = path_value.replace("\\", "/").rstrip("/")
    return clean.rsplit("/", 1)[-1]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hard_ids = np.asarray(HARD_CLASS_IDS, dtype=np.int64)
    hard_set = set(hard_ids.tolist())
    class_to_index = {class_id: index for index, class_id in enumerate(HARD_CLASS_IDS)}

    train = load_npz(args.train_features)
    train_labels_40 = np.asarray(train["labels"], dtype=np.int64)
    if len(train_labels_40) != 1384 or not set(train_labels_40.tolist()).issubset(hard_set):
        raise RuntimeError("The final P46 refit universe must contain 1384 Detail21 rows")
    train_labels = np.asarray(
        [class_to_index[value] for value in train_labels_40], dtype=np.int64
    )
    train_values = early_late_matrix(train["features"])

    model = make_model(ALPHA)
    model.fit(
        train_values,
        train_labels,
        ridge__sample_weight=sample_weights(train_labels, CLASS_WEIGHT_POWER),
    )
    joblib.dump(model, output / "p46_early_late_refit_all1384.joblib", compress=3)

    test = load_npz(args.test_features)
    test_values = early_late_matrix(test["features"])
    p46_scores = aligned_scores(model, test_values) / TEMPERATURE
    p46_probability = softmax(p46_scores, axis=1)
    # The anonymous Test cache deliberately stores official IDs separately from
    # its proxy `test/anonymous/...` paths.  Use the former for alignment with
    # Kaggle's test.csv and the saved P12 logits.
    test_source_ids = np.asarray(test["sample_ids"]).astype(str)
    if len(set(test_source_ids.tolist())) != len(test_source_ids):
        raise RuntimeError("P46 test cache contains duplicate official sample IDs")
    p46_lookup = {
        sample_id: p46_probability[index]
        for index, sample_id in enumerate(test_source_ids)
    }

    p12 = load_npz(args.p12_test)
    p12_ids = np.asarray(p12["sample_ids"]).astype(str)
    route = np.asarray(p12["route_to_candidate"]).astype(bool)
    base_logits = np.asarray(p12["base_logits"], dtype=np.float64)
    candidate_logits = np.asarray(p12["fixed_candidate_logits"], dtype=np.float64)
    routed_logits = np.where(route[:, None], candidate_logits, base_logits)
    routed_probability = softmax(routed_logits, axis=1)
    routed_argmax = routed_probability.argmax(axis=1).astype(np.int64)
    saved_prediction = np.asarray(p12["routed_predictions"], dtype=np.int64)
    if not np.array_equal(routed_argmax, saved_prediction):
        mismatch = int(np.sum(routed_argmax != saved_prediction))
        raise RuntimeError(f"Reconstructed P12 route disagrees on {mismatch} rows")
    p12_lookup = {sample_id: index for index, sample_id in enumerate(p12_ids)}

    with args.test_csv.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        official_rows = list(csv.DictReader(handle))
    if len(official_rows) != 405:
        raise RuntimeError(f"Official Test row count changed: {len(official_rows)}")
    official_ids = [official_id(row["path"]) for row in official_rows]
    if set(official_ids) != set(p12_lookup):
        raise RuntimeError("Official Test CSV and P12 test predictions do not align")
    if set(test_source_ids) - set(official_ids):
        raise RuntimeError("P46 cache contains an unknown Test ID")

    base_predictions = np.asarray(
        [saved_prediction[p12_lookup[sample_id]] for sample_id in official_ids],
        dtype=np.int64,
    )
    base_probabilities = np.stack(
        [routed_probability[p12_lookup[sample_id]] for sample_id in official_ids]
    )
    available = np.asarray([sample_id in p46_lookup for sample_id in official_ids])
    gate = available & np.isin(base_predictions, hard_ids)

    variants: dict[str, dict[str, Any]] = {}
    detailed_rows: list[dict[str, Any]] = []
    variant_predictions: dict[str, np.ndarray] = {}
    for name, weight in BLENDS.items():
        prediction = base_predictions.copy()
        p46_top = np.full(len(official_ids), -1, dtype=np.int64)
        p46_confidence = np.full(len(official_ids), np.nan, dtype=np.float64)
        for index in np.flatnonzero(gate):
            sample_id = official_ids[index]
            p46_conditional = p46_lookup[sample_id]
            base_conditional = base_probabilities[index, hard_ids]
            base_conditional /= max(float(base_conditional.sum()), 1e-12)
            blended = (1.0 - weight) * base_conditional + weight * p46_conditional
            prediction[index] = hard_ids[int(blended.argmax())]
            p46_top[index] = hard_ids[int(p46_conditional.argmax())]
            p46_confidence[index] = float(p46_conditional.max())
        variant_predictions[name] = prediction
        submission_rows = [
            {"path": row["path"], "prediction": int(value)}
            for row, value in zip(official_rows, prediction, strict=True)
        ]
        submission_path = output / f"submission_p12_p46_{name}.csv"
        write_csv(submission_path, submission_rows, ["path", "prediction"])
        variants[name] = {
            "blend_weight": weight,
            "submission": str(submission_path),
            "changed_from_p12": int(np.sum(prediction != base_predictions)),
            "unchanged_from_p12": int(np.sum(prediction == base_predictions)),
            "changed_outside_gate": int(np.sum((prediction != base_predictions) & ~gate)),
            "prediction_histogram": {
                str(class_id): int(np.sum(prediction == class_id))
                for class_id in range(40)
            },
        }

    development = variant_predictions["development_w055"]
    strict = variant_predictions["strict_w070"]
    for index, (row, sample_id) in enumerate(
        zip(official_rows, official_ids, strict=True)
    ):
        q = p46_lookup.get(sample_id)
        detailed_rows.append(
            {
                "path": row["path"],
                "official_sample_id": sample_id,
                "p12_prediction": int(base_predictions[index]),
                "p12_confidence": float(base_probabilities[index].max()),
                "p46_available": int(available[index]),
                "p46_gate": int(gate[index]),
                "p46_prediction": "" if q is None else int(hard_ids[int(q.argmax())]),
                "p46_confidence": "" if q is None else float(q.max()),
                "development_w055_prediction": int(development[index]),
                "strict_w070_prediction": int(strict[index]),
            }
        )
    write_csv(
        output / "test_prediction_audit.csv",
        detailed_rows,
        list(detailed_rows[0]),
    )

    unavailable_ids = [
        sample_id for sample_id, is_available in zip(official_ids, available, strict=True)
        if not is_available
    ]
    summary = {
        "protocol": (
            "P12/P11 routed 40-class base; P46 early+late may only re-rank Detail21 "
            "when the P12 top-1 class is already inside Detail21"
        ),
        "official_test_rows": len(official_rows),
        "p46_model": {
            "feature_set": "early_late",
            "fit_rows": len(train_labels),
            "fit_policy": "refit on all 1384 labeled Detail21 rows after freezing hyperparameters",
            "alpha": ALPHA,
            "class_weight_power": CLASS_WEIGHT_POWER,
            "temperature_from_training_user_oof": TEMPERATURE,
            "frozen_validation_evidence": {
                "correct": 193,
                "total": 290,
                "accuracy": 193 / 290,
            },
        },
        "p12_base": {
            "source": str(args.p12_test.resolve()),
            "known_kaggle_public_score": 0.53233,
            "p12_top1_inside_detail21": int(np.sum(np.isin(base_predictions, hard_ids))),
        },
        "p46_test": {
            "feature_rows": len(test_source_ids),
            "available_and_gated_rows": int(gate.sum()),
            "unavailable_rows": len(unavailable_ids),
            "unavailable_ids": unavailable_ids,
        },
        "safety_invariant": (
            "All P12 top-1 predictions outside Detail21 and every P46-unavailable row "
            "remain exactly unchanged."
        ),
        "variants": variants,
        "audit_csv": str(output / "test_prediction_audit.csv"),
        "accuracy_note": "Official Test labels are unavailable; a real score requires submission.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
