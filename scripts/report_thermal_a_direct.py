from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.train_thermal_generation2 import fixed_label_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT_ROOT / "outputs/thermal_a_multistream_direct_train12_val2/run_manifest.json"
DEFAULT_ARCHIVE = PROJECT_ROOT / "outputs/thermal_a_multistream_direct_train12_val2/selected_predictions.npz"
DEFAULT_A2 = PROJECT_ROOT / "reports/thermal_generation2_environment_probe.json"
DEFAULT_JSON = PROJECT_ROOT / "reports/thermal_a_multistream_direct_train12_val2.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "reports/thermal_a_multistream_direct_train12_val2.md"


def confusion_pairs(matrix: list[list[int]], limit: int = 15) -> list[dict[str, int]]:
    array = np.asarray(matrix, dtype=np.int64)
    pairs = [
        {"true_class": int(left), "predicted_class": int(right), "count": int(array[left, right])}
        for left in range(40)
        for right in range(40)
        if left != right and array[left, right] > 0
    ]
    return sorted(pairs, key=lambda row: (-row["count"], row["true_class"], row["predicted_class"]))[:limit]


def stream_summary(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("selected archive must contain N x 4 stream norms")
    return {
        "names": ["full", "crop", "motion", "pose"],
        "mean": values.mean(axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "p95": np.quantile(values, 0.95, axis=0).tolist(),
    }


def build_a_direct_report(
    *, manifest: dict[str, Any], archive: dict[str, np.ndarray], a2_report: dict[str, Any]
) -> dict[str, Any]:
    if manifest.get("status") != "completed_50_epoch_hard_stop":
        raise ValueError("A-direct run did not complete the frozen 50-epoch hard stop")
    if manifest.get("teacher_logits_loaded") is not False:
        raise ValueError("A-direct must not load teacher logits")
    labels = np.asarray(archive["labels"], dtype=np.int64)
    logits = np.asarray(archive["logits"], dtype=np.float32)
    users = np.asarray(archive["users"], dtype=str)
    combined = fixed_label_metrics(labels=labels, logits=logits, users=users)
    per_user = {
        user: fixed_label_metrics(
            labels=labels[users == user], logits=logits[users == user], users=users[users == user]
        )
        for user in sorted(set(users.tolist()))
    }
    model_probe = a2_report["models"]["a_multistream"]
    package_bytes = int(model_probe["deployment"]["complete_package_bytes"])
    return {
        "schema_version": 1,
        "stage": "A3",
        "status": "completed_stopped_before_a4",
        "experiment_id": manifest["experiment_id"],
        "selected_epoch": int(manifest["selected_epoch"]),
        "epochs_completed": int(manifest["epochs_completed"]),
        "metrics": {
            "combined": combined,
            "per_user": per_user,
            "top_confusion_pairs": confusion_pairs(combined["confusion_matrix"]),
        },
        "route_evidence": {
            "availability_rate": np.asarray(archive["availability"], dtype=np.float32).mean(axis=0).tolist(),
            "quality_mean": np.asarray(archive["quality"], dtype=np.float32).mean(axis=0).tolist(),
            "stream_norms": stream_summary(archive["stream_norms"]),
        },
        "compute": {
            "total_seconds": float(manifest["total_seconds"]),
            "cuda_peak_allocated_mib": float(manifest["cuda_peak_allocated_mib"]),
            "cuda_peak_reserved_mib": float(manifest["cuda_peak_reserved_mib"]),
            "fp32_model_latency": model_probe["fp32_inference"]["latency"],
            "latency_scope": "model_only_preprocessed_tensors_from_A2",
        },
        "deployment": {
            "parameters": int(model_probe["parameters"]),
            "checkpoint_bytes": int(manifest["checkpoint_bytes"]),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "a2_complete_package_proxy_bytes": package_bytes,
            "limit_bytes_exclusive": 95_000_000,
            "under_95000000_bytes": package_bytes < 95_000_000,
        },
        "policy": {
            "thermal_only": True,
            "random_student_initialization": True,
            "pretrained_student_weights": False,
            "teacher_logits_loaded": False,
            "route_b_report_verified": False,
            "route_b_prerequisite_waived_by_user": True,
            "route_b_paired_comparison_available": False,
            "route_b_paired_comparison_reason": "Route B was explicitly waived and has no formal report",
            "heldout4_labels_read": False,
            "competition_test_read": False,
            "ir_depth_inputs_read": False,
            "automatic_resume": False,
            "automatic_extension": False,
        },
        "decision_gate": {
            "accuracy_min": 0.60,
            "macro_f1_min": 0.45,
            "worst_user_accuracy_min": 0.50,
            "zero_recall_classes_max": 10,
            "passed": bool(
                combined["accuracy"] >= 0.60
                and combined["macro_f1"] >= 0.45
                and combined["worst_user_accuracy"] >= 0.50
                and combined["zero_recall_classes"] <= 10
            ),
        },
        "next_action": "stop_for_human_review_before_A4",
    }


def markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]["combined"]
    lines = [
        "# Thermal A-direct Train12 / User6-User7 Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Selected epoch: `{report['selected_epoch']}` of `{report['epochs_completed']}`",
        f"- Accuracy: `{metrics['accuracy']:.5f}`",
        f"- Macro-F1 (fixed 0..39): `{metrics['macro_f1']:.5f}`",
        f"- Worst-user Accuracy: `{metrics['worst_user_accuracy']:.5f}`",
        f"- NLL: `{metrics['nll']:.5f}`",
        f"- Zero-recall classes: `{metrics['zero_recall_classes']}/40`",
        f"- Checkpoint: `{report['deployment']['checkpoint_bytes']}` bytes, SHA256 `{report['deployment']['checkpoint_sha256']}`",
        f"- A2 package proxy: `{report['deployment']['a2_complete_package_proxy_bytes']}` bytes (<95,000,000: `{report['deployment']['under_95000000_bytes']}`)",
        "",
        "## Per-user",
        "",
    ]
    for user, values in report["metrics"]["per_user"].items():
        lines.append(
            f"- {user}: Accuracy `{values['accuracy']:.5f}`, Macro-F1 `{values['macro_f1']:.5f}`, NLL `{values['nll']:.5f}`"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "Thermal-only, random student initialization, and no teacher logits were used. Route B remains unverified and was explicitly waived by the user, so no paired Route B comparison is claimed. No heldout-4 labels, competition test, IR/Depth inputs, or quarantined evidence were read.",
            "",
            "A3 stops here for human review. A4 was not started.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the completed frozen Thermal A-direct run.")
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--a2-report", type=Path, default=DEFAULT_A2)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    a2_report = json.loads(args.a2_report.read_text(encoding="utf-8"))
    with np.load(args.archive, allow_pickle=False) as payload:
        archive = {key: payload[key] for key in payload.files}
    report = build_a_direct_report(manifest=manifest, archive=archive, a2_report=a2_report)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected_epoch": report["selected_epoch"]}))


if __name__ == "__main__":
    main()
