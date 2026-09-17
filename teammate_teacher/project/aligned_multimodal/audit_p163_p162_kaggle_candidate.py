"""Freeze and audit the P162 raw Kaggle CSV without uploading it."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SOURCE = RUNS / "p163_p162_base_test_predictions_v1/submission_p87s_student_raw.csv"
INFERENCE_SUMMARY = RUNS / "p163_p162_base_test_predictions_v1/summary.json"
DECODER_AUDIT = RUNS / "p163_p162_decoder_oof_audit_v1/summary.json"
PACKAGE_AUDIT = RUNS / "p162_p150_student_final_package_v1/summary.json"
MODEL = RUNS / "p162_p150_student_final_package_v1/checkpoints/model.pth"
OFFICIAL = HERE.parent / "Testing/test.csv"
OLD = RUNS / "p90_p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
OUTPUT = RUNS / "p163_p162_kaggle_candidate_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source_rows = read_csv(SOURCE)
    official_rows = read_csv(OFFICIAL)
    if len(source_rows) != 405 or len(official_rows) != 405:
        raise RuntimeError("candidate and official Test must both contain 405 rows")
    if [row["path"] for row in source_rows] != [row["path"] for row in official_rows]:
        raise RuntimeError("candidate path order differs from official Test")
    if len({row["path"] for row in source_rows}) != 405:
        raise RuntimeError("candidate contains duplicate Test paths")
    prediction = [int(row["prediction"]) for row in source_rows]
    if any(value < 0 or value >= 40 for value in prediction):
        raise RuntimeError("candidate prediction is outside 0..39")
    decoder = json.loads(DECODER_AUDIT.read_text(encoding="utf-8"))
    if decoder["aggregate"]["preferred"] != "raw" or decoder["aggregate"]["decoder_net"] >= 0:
        raise RuntimeError("raw candidate is not supported by the OOF decoder audit")
    package = json.loads(PACKAGE_AUDIT.read_text(encoding="utf-8"))
    inference = json.loads(INFERENCE_SUMMARY.read_text(encoding="utf-8"))
    if not package["all_checks_passed"] or inference["test_rows"] != 405:
        raise RuntimeError("model package or Test inference audit failed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate = OUTPUT / "submission_p162_student_raw.csv"
    with candidate.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        writer.writerows(
            {"path": row["path"], "prediction": value}
            for row, value in zip(official_rows, prediction, strict=True)
        )
    old_changed = None
    if OLD.is_file():
        old_rows = read_csv(OLD)
        old_changed = sum(
            left != int(right["prediction"])
            for left, right in zip(prediction, old_rows, strict=True)
        )
    counts = {str(value): prediction.count(value) for value in sorted(set(prediction))}
    report = {
        "stage": "P163_P162_Kaggle_raw_candidate_audit",
        "status": "passed_local_not_uploaded",
        "primary_candidate": str(candidate),
        "candidate_sha256": sha256(candidate),
        "rows": len(prediction),
        "unique_paths": len({row["path"] for row in source_rows}),
        "prediction_min": min(prediction),
        "prediction_max": max(prediction),
        "predicted_class_count": len(set(prediction)),
        "missing_predicted_classes": sorted(set(range(40)) - set(prediction)),
        "class_histogram": counts,
        "model": str(MODEL),
        "model_bytes": MODEL.stat().st_size,
        "model_sha256": sha256(MODEL),
        "single_checkpoint_under_100MB": MODEL.stat().st_size < 100_000_000,
        "large_teacher_required_at_inference": False,
        "missing_visual_rows_supported": int(inference["visual_missing_rows"]),
        "missing_imu_rows_supported": int(inference["imu_missing_rows"]),
        "decoder_oof_net": int(decoder["aggregate"]["decoder_net"]),
        "candidate_choice": "raw because frozen leakage-safe decoder OOF lost 12 rows",
        "changed_vs_previous_p90_decoded": old_changed,
        "test_labels_read": False,
        "uploaded_to_kaggle": False,
        "accuracy_note": "Unknown until the user submits this CSV to Kaggle.",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
