from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_missing_ir_fallback_v1"
SAFE_SUBMISSION = (
    PROJECT_DIR
    / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
)
H1_USERS = {"user6", "user8", "user17", "user23"}
H2_USERS = {"user5", "user7", "user16", "user18", "user19"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    correct = int(np.sum(labels == prediction))
    return {
        "correct": correct,
        "total": int(len(labels)),
        "accuracy": float(correct / len(labels)),
    }


def sliced_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
    folds: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {"all": metrics(labels, prediction)}
    result["H1_users"] = metrics(labels[np.isin(users, list(H1_USERS))], prediction[np.isin(users, list(H1_USERS))])
    result["H2_users"] = metrics(labels[np.isin(users, list(H2_USERS))], prediction[np.isin(users, list(H2_USERS))])
    result["per_fold"] = {
        str(fold): metrics(labels[folds == fold], prediction[folds == fold])
        for fold in sorted(np.unique(folds).tolist())
    }
    return result


def main() -> None:
    # P12 is independent of IR: Skeleton + Depth + IMU + Thermal.  Its OOF file
    # is cross-fitted by subject, so these are genuine unseen-subject estimates.
    with np.load(PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz") as oof:
        labels = np.asarray(oof["labels"], dtype=np.int64)
        folds = np.asarray(oof["folds"], dtype=np.int64)
        oof_predictions = {
            "depth": np.asarray(oof["depth_logits"]).argmax(axis=1),
            "skeleton": np.asarray(oof["skeleton_logits"]).argmax(axis=1),
            "sd": np.asarray(oof["sd_logits"]).argmax(axis=1),
            "imu": np.asarray(oof["imu_logits"]).argmax(axis=1),
            "sd_imu": np.asarray(oof["sd_imu_logits"]).argmax(axis=1),
            "thermal": np.asarray(oof["thermal_logits"]).argmax(axis=1),
            "thermal_candidate": np.asarray(oof["thermal_candidate_logits"]).argmax(axis=1),
            "p12_final": np.asarray(oof["final_predictions"], dtype=np.int64),
        }
    row_metadata = read_csv(PROJECT_DIR / "runs/p12_complete_oof/complete_oof_rows.csv")
    users = np.asarray([row["user_id"] for row in row_metadata]).astype(str)
    if len(users) != len(labels):
        raise RuntimeError("P12 OOF row metadata length changed")

    validation = {
        name: sliced_metrics(labels, prediction, users, folds)
        for name, prediction in oof_predictions.items()
    }
    p12_prediction = oof_predictions["p12_final"]
    predicted_class_reliability = {}
    for class_id in (7, 9):
        rows = p12_prediction == class_id
        predicted_class_reliability[str(class_id)] = {
            "predicted_samples": int(np.sum(rows)),
            "precision": float(np.mean(labels[rows] == class_id)),
            "true_samples": int(np.sum(labels == class_id)),
            "recall": float(np.mean(p12_prediction[labels == class_id] == class_id)),
        }

    union_rows = read_csv(PROJECT_DIR / "data/p46_test_union_manifest.csv")
    missing_ir_ids = sorted(
        row["official_sample_id"]
        for row in union_rows
        if row["p46_ir_readable"] == "0"
    )
    if len(union_rows) != 405 or len(missing_ir_ids) != 4:
        raise RuntimeError(
            f"Test IR contract changed: rows={len(union_rows)}, missing={missing_ir_ids}"
        )

    with np.load(PROJECT_DIR / "runs/p11_final_package/test_candidate/test_logits.npz") as p12_test:
        p12_ids = p12_test["sample_ids"].astype(str)
        p12_lookup = {sample_id: index for index, sample_id in enumerate(p12_ids)}
        routed = np.asarray(p12_test["routed_predictions"], dtype=np.int64)
        fixed = np.asarray(p12_test["fixed_candidate_logits"]).argmax(axis=1)
        thermal = np.asarray(p12_test["thermal_logits"]).argmax(axis=1)
    with np.load(PROJECT_DIR / "runs/p11_final_package/test_logits_sd_fp16.npz") as sd_test:
        sd_ids = sd_test["sample_ids"].astype(str)
        sd_lookup = {sample_id: index for index, sample_id in enumerate(sd_ids)}
        depth = np.asarray(sd_test["depth_logits"]).argmax(axis=1)
        skeleton = np.asarray(sd_test["skeleton_logits"]).argmax(axis=1)
    with np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz") as imu_test:
        imu_ids = imu_test["sample_ids"].astype(str)
        imu_lookup = {sample_id: index for index, sample_id in enumerate(imu_ids)}
        imu = np.asarray(imu_test["imu_logits"]).argmax(axis=1)
    with np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    ) as p87_test:
        base_ids = p87_test["sample_ids"].astype(str)
        base_lookup = {sample_id: index for index, sample_id in enumerate(base_ids)}
        base_probability = np.asarray(p87_test["base_probability"], dtype=np.float64)

    source_rows = submission_io.read_rows(SAFE_SUBMISSION)
    safe_prediction = submission_io.read_prediction(SAFE_SUBMISSION)
    official_ids = np.asarray(
        [Path(row["path"].rstrip("/\\")).name for row in source_rows]
    ).astype(str)
    if len(set(official_ids.tolist())) != 405:
        raise RuntimeError("safe submission IDs are not unique")
    official_lookup = {sample_id: index for index, sample_id in enumerate(official_ids)}
    prediction = safe_prediction.copy()
    details = []
    for sample_id in missing_ir_ids:
        row_index = official_lookup[sample_id]
        p12_index = p12_lookup[sample_id]
        sd_index = sd_lookup[sample_id]
        imu_index = imu_lookup[sample_id]
        base_index = base_lookup[sample_id]
        replacement = int(routed[p12_index])
        before = int(prediction[row_index])
        prediction[row_index] = replacement
        details.append(
            {
                "sample_id": sample_id,
                "safe_prediction": before,
                "p87_base_prediction": int(base_probability[base_index].argmax()),
                "p87_base_confidence": float(base_probability[base_index].max()),
                "fallback_prediction": replacement,
                "p12_fixed_prediction": int(fixed[p12_index]),
                "depth_prediction": int(depth[sd_index]),
                "thermal_prediction": int(thermal[p12_index]),
                "imu_prediction": int(imu[imu_index]),
                "skeleton_prediction": int(skeleton[sd_index]),
                "changed": bool(before != replacement),
            }
        )

    changed = np.flatnonzero(prediction != safe_prediction)
    if any(official_ids[index] not in missing_ir_ids for index in changed):
        raise RuntimeError("fallback changed a row with readable IR")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_missing_ir_fallback.csv"
    submission_io.write_submission(submission, source_rows, prediction)
    reproduced = submission_io.read_prediction(submission)
    if not np.array_equal(reproduced, prediction):
        raise RuntimeError("submission round-trip failed")

    report = {
        "stage": "P89_missing_IR_nonIR_fallback_v1",
        "protocol": (
            "Define missingness only from the modality readability manifest. For "
            "p46_ir_readable=0, replace the decoded safe prediction with the P12 "
            "subject-disjoint non-IR fallback (Skeleton+Depth+IMU+Thermal). No Test "
            "labels or leaderboard feedback are used to choose rows or labels."
        ),
        "validation": validation,
        "predicted_class_reliability": predicted_class_reliability,
        "test": {
            "rows": 405,
            "ir_unreadable": len(missing_ir_ids),
            "ir_unreadable_fraction": len(missing_ir_ids) / 405.0,
            "details": details,
            "changes_vs_0_85572_safe": int(len(changed)),
            "changed_ids": official_ids[changed].tolist(),
        },
        "limitation": (
            "Training contains no natural Depth+Skeleton row with unreadable IR, so "
            "the missingness gate itself cannot be directly measured. The fallback "
            "model accuracy is nevertheless measured with strict subject-disjoint OOF."
        ),
        "submission": {
            "path": str(submission.resolve()),
            "sha256": submission_io.digest(submission),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
