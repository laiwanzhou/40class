from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.report_x3d_s_train12_val2_dev import build_report as build_standalone_report
import src.train_x3d_s_visual_expert as trainer


def evaluate_matched_candidate(
    metrics: Mapping[str, float], reference: Mapping[str, float]
) -> str:
    if float(metrics["accuracy"]) < float(reference["accuracy"]) - 0.02 - 1e-12:
        return "human_review_regression"
    names = ("accuracy", "macro_f1", "worst_user_accuracy")
    no_worse = all(float(metrics[name]) >= float(reference[name]) for name in names)
    strictly_better = any(float(metrics[name]) > float(reference[name]) for name in names)
    return "preferred" if no_worse and strictly_better else "not_preferred"


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_ids", "labels", "logits", "user_ids", "num_frames"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Prediction archive is missing fields: {sorted(missing)}")
        return {name: archive[name] for name in required}


def _matched_prediction_diagnostics(
    candidate_path: Path, reference_path: Path
) -> dict[str, Any]:
    candidate = _load_predictions(candidate_path)
    reference = _load_predictions(reference_path)
    for name in ("sample_ids", "labels", "user_ids", "num_frames"):
        if not np.array_equal(candidate[name], reference[name]):
            raise ValueError(f"Matched prediction archives differ at {name}")
    labels = candidate["labels"].astype(np.int64)
    candidate_predictions = candidate["logits"].argmax(axis=1)
    reference_predictions = reference["logits"].argmax(axis=1)
    candidate_probabilities = np.exp(candidate["logits"].astype(np.float64))
    reference_probabilities = np.exp(reference["logits"].astype(np.float64))

    def confidence(probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
        correct = predictions == labels
        return {
            "mean_all": float(probabilities.max(axis=1).mean()),
            "mean_correct": float(probabilities.max(axis=1)[correct].mean()),
            "mean_wrong": float(probabilities.max(axis=1)[~correct].mean()),
            "nll": float(
                -np.log(
                    np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
                ).mean()
            ),
        }

    candidate_correct = candidate_predictions == labels
    reference_correct = reference_predictions == labels
    return {
        "sample_alignment_exact": True,
        "prediction_disagreement_count": int(
            (candidate_predictions != reference_predictions).sum()
        ),
        "prediction_disagreement_fraction": float(
            (candidate_predictions != reference_predictions).mean()
        ),
        "candidate_only_correct": int((candidate_correct & ~reference_correct).sum()),
        "reference_only_correct": int((reference_correct & ~candidate_correct).sum()),
        "both_wrong": int((~candidate_correct & ~reference_correct).sum()),
        "candidate_confidence": confidence(candidate_probabilities, candidate_predictions),
        "reference_confidence": confidence(reference_probabilities, reference_predictions),
    }


def build_report(
    run_directory: Path,
    *,
    reference_report_path: Path,
    reference_run_directory: Path,
) -> dict[str, Any]:
    report = build_standalone_report(run_directory)
    reference_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
    if reference_report.get("role") != "train12_val2_development_tuning_report":
        raise ValueError("Reference report is not train12/val2 development evidence")
    reference_metrics = reference_report["metrics"]
    metrics = report["metrics"]
    decision = evaluate_matched_candidate(metrics, reference_metrics)

    candidate_archive = run_directory / "val_predictions_best_accuracy.npz"
    reference_archive = reference_run_directory / "val_predictions_best_accuracy.npz"
    diagnostics = _matched_prediction_diagnostics(candidate_archive, reference_archive)
    names = ("accuracy", "macro_f1", "worst_user_accuracy")
    report.update(
        {
            "role": "train12_val2_partial1_matched_report",
            "decision": decision,
            "interpretation": (
                "Matched capacity ablation against partial2. Only the number of unfrozen "
                "X3D backbone blocks changes from two to one."
            ),
            "matched_partial2_reference": {
                "run_id": reference_report["run_id"],
                "metrics": {name: float(reference_metrics[name]) for name in names},
                "report_path": str(reference_report_path.resolve()),
                "report_sha256": trainer._sha256_file(reference_report_path),
                "prediction_sha256": trainer._sha256_file(reference_archive),
            },
            "matched_delta": {
                name: float(metrics[name] - reference_metrics[name]) for name in names
            },
            "per_user_accuracy_delta": {
                user: float(
                    metrics["per_user_accuracy"][user]
                    - reference_metrics["per_user_accuracy"][user]
                )
                for user in metrics["per_user_accuracy"]
            },
            "duration_accuracy_delta": {
                bucket: float(
                    metrics["duration_buckets"][bucket]["accuracy"]
                    - reference_metrics["duration_buckets"][bucket]["accuracy"]
                )
                for bucket in metrics["duration_buckets"]
            },
            "matched_prediction_diagnostics": diagnostics,
            "human_review_floor": float(reference_metrics["accuracy"] - 0.02),
            "automatic_next_experiment_permitted": False,
        }
    )
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    reference = report["matched_partial2_reference"]["metrics"]
    delta = report["matched_delta"]
    training = report["training_log_at_selected_epoch"]
    diagnostics = report["matched_prediction_diagnostics"]
    lines = [
        "# X3D-S Train12/Val2 Partial1 Matched Result",
        "",
        "> Development-only matched capacity ablation. This is not unbiased OOF and does not replace canonical Phase 4/5 evidence.",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "| Metric | Partial1 | Partial2 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name in ("accuracy", "macro_f1", "worst_user_accuracy"):
        lines.append(
            f"| {name} | {metrics[name]:.6f} | {reference[name]:.6f} | {delta[name]:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Selected epoch: `{report['selected_epoch']}`. Train Accuracy: `{training['train_accuracy']:.6f}`. Train-minus-validation gap: `{training['train_minus_val_accuracy']:.6f}`.",
            "",
            "## Per User",
            "",
            "| User | Accuracy | Delta vs partial2 |",
            "|---|---:|---:|",
        ]
    )
    for user, accuracy in metrics["per_user_accuracy"].items():
        lines.append(
            f"| {user} | {accuracy:.6f} | {report['per_user_accuracy_delta'][user]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Duration",
            "",
            "| Frames | N | Accuracy | Delta vs partial2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for bucket, values in metrics["duration_buckets"].items():
        lines.append(
            f"| {bucket} | {values['sample_count']} | {values['accuracy']:.6f} | "
            f"{report['duration_accuracy_delta'][bucket]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Matched Prediction Diagnosis",
            "",
            f"The models disagree on {diagnostics['prediction_disagreement_count']}/324 trials. Partial1-only correct: {diagnostics['candidate_only_correct']}; partial2-only correct: {diagnostics['reference_only_correct']}; both wrong: {diagnostics['both_wrong']}.",
            "",
            f"Partial1 NLL is {diagnostics['candidate_confidence']['nll']:.6f}; partial2 NLL is {diagnostics['reference_confidence']['nll']:.6f}.",
            "",
            "The greater-than-two-point Accuracy regression rule is triggered. Preserve all artifacts and require human review; do not automatically launch another experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report matched partial1 X3D evidence")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-run-directory", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/x3d_s_train12_val2_partial1_report.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/x3d_s_train12_val2_partial1_report.md"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(
        args.run_directory,
        reference_report_path=args.reference_report,
        reference_run_directory=args.reference_run_directory,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
