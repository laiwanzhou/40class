"""Package and audit the <100 MB P150-distilled Student."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from p87s_deploy_model import (
    deployment_model_config,
    load_p87s_deploy_checkpoint,
)


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FOLDS = (
    RUNS / "p162_p150_student_h1_e40_v1",
    RUNS / "p162_p150_student_h2_e40_v1",
    RUNS / "p162_p150_student_h3_e40_v1",
)
REFIT = RUNS / "p162_p150_student_final_refit_v1"
OUTPUT = RUNS / "p162_p150_student_final_package_v1"
BASE_SUMMARY = RUNS / "p87s_fusion_all2914_v1/summary.json"
TARGET_SUMMARY = RUNS / "p162_p150_student_targets_v1/summary.json"
EQUIVALENCE = RUNS / "p162_p150_student_deployment_equivalence_v1/summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    fold_summaries = [load_json(path / "summary.json") for path in FOLDS]
    rows = sum(int(value["adapted_metrics"]["total"]) for value in fold_summaries)
    correct = sum(int(value["adapted_metrics"]["correct"]) for value in fold_summaries)
    if rows != 2470:
        raise RuntimeError(f"P162 OOF rows changed: {rows}")
    required = int(__import__("math").ceil(0.88 * rows))
    if correct < required:
        raise RuntimeError(f"P162 misses 88%: {correct}/{rows}, required={required}")
    if any(int(value["parameters"]) != 23_560_564 for value in fold_summaries):
        raise RuntimeError("fold Student parameter contract changed")
    if any(int(value["ground_truth_fields_seen_during_training"]) != 0 for value in fold_summaries):
        raise RuntimeError("ground truth entered a P162 adaptation batch")

    refit_checkpoint_path = REFIT / "model.pth"
    refit_checkpoint = torch.load(refit_checkpoint_path, map_location="cpu", weights_only=False)
    refit_summary = load_json(REFIT / "summary.json")
    base_summary = load_json(BASE_SUMMARY)
    deploy_config = deployment_model_config(
        refit_checkpoint["visual_config"],
        refit_checkpoint["pretrain_config"],
        refit_checkpoint["modality"],
        base_summary["config"],
    )
    package_dir = OUTPUT / "checkpoints"
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = package_dir / "model.pth"
    package = {
        "stage": "P162_P150_distilled_deployment",
        "deployment_model_config": deploy_config,
        "model_state": refit_checkpoint["model_state"],
        "provenance": {
            "teacher": "P150 strict OOF hard prediction with 0.02 smoothing",
            "teacher_accuracy": 2210 / 2470,
            "student_oof_accuracy": correct / rows,
            "adaptation_ground_truth_fields_seen": 0,
            "large_teacher_required_at_inference": False,
        },
    }
    torch.save(package, package_path)
    checkpoint_bytes = package_path.stat().st_size
    if checkpoint_bytes >= 100_000_000:
        raise RuntimeError(f"deployment checkpoint is too large: {checkpoint_bytes}")
    loaded_model, loaded_checkpoint = load_p87s_deploy_checkpoint(package_path)
    parameters = sum(parameter.numel() for parameter in loaded_model.parameters())
    if parameters != 23_560_564:
        raise RuntimeError(f"loaded parameter contract changed: {parameters}")
    if loaded_checkpoint.get("stage") != "P162_P150_distilled_deployment":
        raise RuntimeError("deployment stage changed after reload")

    teacher_summary = load_json(TARGET_SUMMARY)
    equivalence = load_json(EQUIVALENCE)
    if equivalence.get("status") != "passed" or equivalence.get("prediction_agreement") != 1.0:
        raise RuntimeError("raw-input deployment equivalence did not pass")
    report = {
        "stage": "P162_P150_student_final_package_audit",
        "status": "passed",
        "all_checks_passed": True,
        "validation": {
            "protocol": (
                "Three disjoint held cohorts. Each fold starts from a model trained "
                "without that cohort's labels, then consumes only P150 strict OOF "
                "pseudo targets for its held inputs."
            ),
            "cohorts": [
                {
                    "run": path.name,
                    "correct": int(summary["adapted_metrics"]["correct"]),
                    "rows": int(summary["adapted_metrics"]["total"]),
                    "accuracy": float(summary["adapted_metrics"]["accuracy"]),
                    "teacher_agreement": float(summary["audit"]["student_target_agreement"]),
                }
                for path, summary in zip(FOLDS, fold_summaries)
            ],
            "aggregate_correct": correct,
            "aggregate_rows": rows,
            "aggregate_accuracy": correct / rows,
            "required_accuracy": 0.88,
            "required_correct": required,
            "margin_correct": correct - required,
            "passes_0.88": correct >= required,
            "teacher_correct": int(teacher_summary["aggregate"]["teacher_correct_audit_only"]),
            "teacher_accuracy": float(teacher_summary["aggregate"]["teacher_accuracy_audit_only"]),
            "gap_to_teacher_correct": int(teacher_summary["aggregate"]["teacher_correct_audit_only"]) - correct,
        },
        "deployment": {
            "path": str(package_path),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_mb_decimal": checkpoint_bytes / 1_000_000,
            "checkpoint_sha256": sha256(package_path),
            "parameters": parameters,
            "single_checkpoint": True,
            "strict_decimal_100MB_passed": checkpoint_bytes < 100_000_000,
            "fresh_model_strict_reload_passed": True,
            "large_teacher_required_at_inference": False,
            "teacher_logits_or_targets_in_package": False,
            "raw_input_equivalence_passed": True,
            "raw_vs_cached_prediction_agreement": float(
                equivalence["prediction_agreement"]
            ),
            "raw_vs_cached_max_logit_error": float(
                equivalence["logit_max_absolute_error"]
            ),
        },
        "final_refit": {
            "target_rows": int(refit_summary["target_rows"]),
            "teacher_agreement_diagnostic": float(refit_summary["teacher_agreement_diagnostic"]),
            "ground_truth_fields_seen_during_adaptation": int(
                refit_summary["ground_truth_fields_seen_during_adaptation"]
            ),
            "note": refit_summary["validation_note"],
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
