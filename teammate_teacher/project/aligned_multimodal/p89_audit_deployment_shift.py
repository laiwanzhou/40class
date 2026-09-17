from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np

import p89_build_final_test_submissions_nometa  # noqa: F401
import p89_build_final_test_submissions as pipeline


PROJECT_DIR = Path(__file__).resolve().parent
INPUT = PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2"
OUTPUT = PROJECT_DIR / "runs/p89_deployment_shift_audit_v1"
EXPERT_COUNT = 33
CLASS_COUNT = 21


def read_submission(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray(
            [int(row["prediction"]) for row in csv.DictReader(handle)],
            dtype=np.int64,
        )


def distribution_statistics(probability: np.ndarray) -> dict[str, float]:
    ordered = np.sort(probability, axis=1)
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, 1e-12)), axis=1
    )
    return {
        "mean_max_probability": float(ordered[:, -1].mean()),
        "mean_margin": float((ordered[:, -1] - ordered[:, -2]).mean()),
        "mean_entropy": float(entropy.mean()),
    }


def main() -> None:
    reference = pipeline.load(
        PROJECT_DIR / "runs/p46_validation70_final_v1/crossfit_logits.npz"
    )
    train_ids = reference["sample_ids"].astype(str)
    train_x, train_names = pipeline.training_features(train_ids)
    test_source = pipeline.load(
        PROJECT_DIR / "runs/p89_multiexpert_test_logits_v1/p46_test_logits.npz"
    )
    test_ids = test_source["sample_ids"].astype(str)
    test_x, test_names = pipeline.test_features(test_ids)
    if train_names != test_names:
        raise RuntimeError("training/Test expert feature order differs")
    if len(train_names) != EXPERT_COUNT:
        raise RuntimeError(f"expected {EXPERT_COUNT} experts, found {len(train_names)}")

    train_probability = train_x[:, : EXPERT_COUNT * CLASS_COUNT].reshape(
        len(train_x), EXPERT_COUNT, CLASS_COUNT
    )
    test_probability = test_x[:, : EXPERT_COUNT * CLASS_COUNT].reshape(
        len(test_x), EXPERT_COUNT, CLASS_COUNT
    )
    experts = []
    for index, name in enumerate(train_names):
        train_stats = distribution_statistics(train_probability[:, index])
        test_stats = distribution_statistics(test_probability[:, index])
        experts.append(
            {
                "name": name,
                "train_oof": train_stats,
                "test_refit": test_stats,
                "delta_max_probability": (
                    test_stats["mean_max_probability"]
                    - train_stats["mean_max_probability"]
                ),
                "delta_margin": test_stats["mean_margin"] - train_stats["mean_margin"],
                "delta_entropy": test_stats["mean_entropy"] - train_stats["mean_entropy"],
            }
        )

    artifact = joblib.load(INPUT / "detail21_stacker.joblib")
    scaler = artifact["scaler"]
    model = artifact["model"]
    scaled_test = scaler.transform(test_x)
    mean_absolute_z = np.mean(np.abs(scaled_test), axis=0)
    max_absolute_z = np.max(np.abs(scaled_test), axis=0)
    outlier_rate3 = np.mean(np.abs(scaled_test) > 3.0, axis=0)
    outlier_rate5 = np.mean(np.abs(scaled_test) > 5.0, axis=0)
    feature_report = {
        "mean_absolute_z": float(np.mean(np.abs(scaled_test))),
        "p95_absolute_z": float(np.quantile(np.abs(scaled_test), 0.95)),
        "p99_absolute_z": float(np.quantile(np.abs(scaled_test), 0.99)),
        "maximum_absolute_z": float(np.max(np.abs(scaled_test))),
        "feature_fraction_mean_abs_z_gt_1": float(np.mean(mean_absolute_z > 1.0)),
        "feature_fraction_any_abs_z_gt_5": float(np.mean(max_absolute_z > 5.0)),
        "cell_fraction_abs_z_gt_3": float(np.mean(outlier_rate3)),
        "cell_fraction_abs_z_gt_5": float(np.mean(outlier_rate5)),
    }

    probability = pipeline.softmax(model.decision_function(scaled_test))
    p87 = read_submission(
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    detail = read_submission(INPUT / "submission_p89_detail_global.csv")
    robust = read_submission(INPUT / "submission_p89_robust.csv")
    template = read_submission(INPUT / "submission_p89_template.csv")
    report = {
        "stage": "P89_OOF_to_Test_deployment_shift_audit_v1",
        "train_rows": int(len(train_x)),
        "test_detail_rows": int(len(test_x)),
        "experts": experts,
        "stacker_feature_shift": feature_report,
        "stacker_output": distribution_statistics(probability),
        "submission_changes_vs_p87": {
            "detail_global": int(np.sum(detail != p87)),
            "robust": int(np.sum(robust != p87)),
            "template": int(np.sum(template != p87)),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
