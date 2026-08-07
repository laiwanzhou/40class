from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import classification_metrics
from src.target16_hierarchical_fusion import fusion_metrics, hierarchical_predictions


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in frame.to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e2-dir", type=Path,
        default=PROJECT_ROOT / "outputs/depth_ir_person_crop_target16_e2_fold0/target16_conditional_e2_14train_4val",
    )
    parser.add_argument(
        "--b2-predictions", type=Path,
        default=PROJECT_ROOT / (
            "outputs/depth_ir_person_crop_40class_256_fold0/"
            "depth_ir_person_crop_40class_256_14train_4val/val_predictions_best_epoch25_with_users.npz"
        ),
    )
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--report-prefix", default="target16_hierarchical_e2")
    parser.add_argument("--experiment-label", default="Target16 conditional E2")
    return parser.parse_args()


def _aligned_e2(b2: np.lib.npyio.NpzFile, e2: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    lookup = {str(sample_id): index for index, sample_id in enumerate(e2["sample_ids"])}
    if len(lookup) != len(e2["sample_ids"]):
        raise ValueError("Duplicate E2 sample IDs.")
    try:
        order = np.asarray([lookup[str(sample_id)] for sample_id in b2["sample_ids"]], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"E2 is missing B2 sample {error}.") from error
    if len(order) != len(e2["sample_ids"]):
        raise ValueError("B2 and E2 sample counts differ.")
    return {key: np.asarray(e2[key])[order] for key in e2.files if key not in {"original_class_ids"}}


def _gated_target_macro(labels: np.ndarray, predictions: np.ndarray, gate: np.ndarray, targets: np.ndarray) -> float:
    selected = gate & np.isin(labels, targets)
    mapping = {int(value): index for index, value in enumerate(targets)}
    true_local = np.asarray([mapping[int(value)] for value in labels[selected]])
    pred_local = np.asarray([mapping[int(value)] for value in predictions[selected]])
    _, _, f1, _ = precision_recall_fscore_support(
        true_local, pred_local, labels=np.arange(16), zero_division=0,
    )
    return float(f1.mean())


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.report_prefix)
    b2 = np.load(args.b2_predictions, allow_pickle=False)
    e2_raw = np.load(args.e2_dir / "val_predictions_all40_best_macro_f1.npz", allow_pickle=False)
    e2 = _aligned_e2(b2, e2_raw)
    sample_ids = np.asarray(b2["sample_ids"], dtype=str)
    labels = np.asarray(b2["labels"], dtype=np.int64)
    users = np.asarray(b2["user_ids"], dtype=str)
    b2_probabilities = np.asarray(b2["probabilities"], dtype=np.float64)
    e2_probabilities = np.asarray(e2["probabilities"], dtype=np.float64)
    targets = np.asarray(e2_raw["original_class_ids"], dtype=np.int64)
    if not np.array_equal(labels, np.asarray(e2["labels"], dtype=np.int64)):
        raise ValueError("B2 and E2 labels are not sample-aligned.")
    if len(labels) != 590 or len(targets) != 16:
        raise ValueError("Expected 590 validation samples and 16 target classes.")

    class_map = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_class_map.csv").sort_values("class_id")
    action_names = class_map["action_name"].tolist()
    b2_predictions = b2_probabilities.argmax(axis=1)
    target_truth = np.isin(labels, targets)
    target_gate = np.isin(b2_predictions, targets)
    base_correct = b2_predictions == labels
    audit = {
        "target_correct": int(np.sum(target_truth & base_correct)),
        "target_to_target_wrong": int(np.sum(target_truth & target_gate & ~base_correct)),
        "target_to_external": int(np.sum(target_truth & ~target_gate)),
        "external_to_target": int(np.sum(~target_truth & target_gate)),
    }
    if audit != {
        "target_correct": 56,
        "target_to_target_wrong": 123,
        "target_to_external": 43,
        "external_to_target": 43,
    }:
        raise ValueError(f"B2 gate audit changed unexpectedly: {audit}")

    summary_rows: list[dict[str, object]] = []
    user_rows: list[dict[str, object]] = []
    per_class: dict[float, np.ndarray] = {}
    prediction_by_alpha: dict[float, np.ndarray] = {}
    base_non_target_f1: np.ndarray | None = None
    for alpha in ALPHAS:
        predictions, gate = hierarchical_predictions(b2_probabilities, e2_probabilities, targets, alpha)
        metrics = fusion_metrics(labels, b2_predictions, predictions, gate, targets)
        prediction_by_alpha[alpha] = predictions
        per_class[alpha] = np.asarray(metrics["per_class_f1"], dtype=np.float64)
        non_target_ids = np.asarray([value for value in range(40) if value not in set(targets)], dtype=np.int64)
        non_target_f1 = per_class[alpha][non_target_ids]
        if base_non_target_f1 is None:
            base_non_target_f1 = non_target_f1
        elif not np.allclose(base_non_target_f1, non_target_f1, atol=1e-12):
            raise AssertionError("Hard hierarchy changed a non-target class F1 score.")
        summary_rows.append(
            {
                "alpha": alpha,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "target16_macro_f1": metrics["target16_macro_f1"],
                "gated_target_accuracy": float(np.mean(predictions[target_truth & gate] == labels[target_truth & gate])),
                "gated_target_macro_f1": _gated_target_macro(labels, predictions, gate, targets),
                "rescued": metrics["rescued"],
                "harmed": metrics["harmed"],
                "net_rescue": metrics["net_rescue"],
                "correct_count": int(np.sum(predictions == labels)),
                "gated_samples": metrics["gated_samples"],
                "gated_true_target_samples": metrics["gated_true_target_samples"],
                "gated_external_samples": metrics["gated_external_samples"],
            }
        )
        for user in sorted(np.unique(users)):
            selected = users == user
            user_metrics = classification_metrics(labels[selected], predictions[selected], 40)
            user_rows.append(
                {
                    "alpha": alpha,
                    "user_id": user,
                    "support": int(selected.sum()),
                    "accuracy": user_metrics["accuracy"],
                    "macro_f1_40class": user_metrics["macro_f1"],
                    "correct_count": int(np.sum(predictions[selected] == labels[selected])),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.report_dir / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    user_frame = pd.DataFrame(user_rows)
    user_frame.to_csv(
        args.report_dir / f"{prefix}_per_user.csv", index=False, encoding="utf-8-sig",
    )
    class_rows = []
    for class_id, action_name in enumerate(action_names):
        row: dict[str, object] = {
            "class_id": class_id,
            "action_name": action_name,
            "is_target16": class_id in set(targets),
            "train_support": int(class_map.iloc[class_id]["train_support"]),
            "val_support": int(class_map.iloc[class_id]["val_support"]),
        }
        for alpha in ALPHAS:
            row[f"f1_alpha_{alpha:.2f}"] = per_class[alpha][class_id]
        class_rows.append(row)
    class_frame = pd.DataFrame(class_rows)
    class_frame.to_csv(
        args.report_dir / f"{prefix}_per_class.csv", index=False, encoding="utf-8-sig",
    )

    outcome_rows = []
    for index in np.flatnonzero(target_gate):
        row = {
            "sample_id": sample_ids[index],
            "user_id": users[index],
            "true_class_id": int(labels[index]),
            "true_action": action_names[int(labels[index])],
            "b2_prediction": int(b2_predictions[index]),
            "b2_prediction_action": action_names[int(b2_predictions[index])],
            "b2_correct": bool(base_correct[index]),
        }
        for alpha in ALPHAS:
            prediction = int(prediction_by_alpha[alpha][index])
            row[f"prediction_alpha_{alpha:.2f}"] = prediction
            row[f"correct_alpha_{alpha:.2f}"] = prediction == labels[index]
        outcome_rows.append(row)
    pd.DataFrame(outcome_rows).to_csv(
        args.report_dir / f"{prefix}_gated_outcomes.csv", index=False, encoding="utf-8-sig",
    )

    best = summary.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    baseline = summary.loc[summary["alpha"] == 0.0].iloc[0]
    history = pd.read_csv(args.e2_dir / "history.csv")
    run_summary = json.loads((args.e2_dir / "run_summary.json").read_text(encoding="utf-8"))
    linear_residual = int(run_summary.get("trainable_parameters", -1)) == 3088
    best_e2 = history.loc[history["val_macro_f1"].idxmax()]
    final_e2 = history.iloc[-1]
    class_frame["delta_f1_alpha_1_minus_0"] = class_frame["f1_alpha_1.00"] - class_frame["f1_alpha_0.00"]
    target_changes = class_frame[class_frame["is_target16"]].sort_values(
        "delta_f1_alpha_1_minus_0", ascending=False,
    )
    gain_lines = [
        f"- {row.action_name}: {row['f1_alpha_0.00']:.3f} -> {row['f1_alpha_1.00']:.3f} "
        f"({row.delta_f1_alpha_1_minus_0:+.3f})."
        for _, row in target_changes.head(6).iterrows()
    ]
    harm_lines = [
        f"- {row.action_name}: {row['f1_alpha_0.00']:.3f} -> {row['f1_alpha_1.00']:.3f} "
        f"({row.delta_f1_alpha_1_minus_0:+.3f})."
        for _, row in target_changes.sort_values("delta_f1_alpha_1_minus_0").head(6).iterrows()
    ]
    user_base = user_frame[user_frame["alpha"] == 0.0].set_index("user_id")
    user_best = user_frame[user_frame["alpha"] == float(best["alpha"])].set_index("user_id")
    user_lines = [
        f"- {user}: accuracy {user_base.loc[user, 'accuracy']:.3f} -> {user_best.loc[user, 'accuracy']:.3f}, "
        f"correct-count delta {int(user_best.loc[user, 'correct_count'] - user_base.loc[user, 'correct_count']):+d}."
        for user in user_base.index
    ]
    if linear_residual:
        training_interpretation = (
            f"- By Epoch {int(final_e2['epoch'])}, validation Accuracy/Macro-F1 were "
            f"{final_e2['val_accuracy']:.6f}/{final_e2['val_macro_f1']:.6f}, with validation loss "
            f"{final_e2['val_loss']:.6f}. Only the 3,088-parameter residual head was trained; the large "
            "absolute train-validation gap is inherited from the frozen B2 representation rather than "
            "created by backbone fine-tuning."
        )
    else:
        training_interpretation = (
            f"- By Epoch {int(final_e2['epoch'])}, validation Accuracy/Macro-F1 had fallen to "
            f"{final_e2['val_accuracy']:.6f}/{final_e2['val_macro_f1']:.6f}, while validation loss reached "
            f"{final_e2['val_loss']:.6f}. This is clear late-stage overfitting."
        )
    improved = bool(
        float(best["accuracy"]) > float(baseline["accuracy"]) + 1e-12
        or float(best["macro_f1"]) > float(baseline["macro_f1"]) + 1e-12
    )
    if improved:
        assessment = (
            "The hard hierarchy succeeds on this validation fold: at least one overall metric improves, "
            "and the non-target 24-class decision surface is preserved. The improvement is not uniform "
            "across target actions or users. The selected alpha remains a validation diagnostic until "
            "confirmed on an independent fold or held-out calibration protocol."
        )
    else:
        assessment = (
            "The linear residual experiment does not improve the hard hierarchy on this validation fold. "
            "The strongest alpha is 0.0, meaning the unmodified B2 output remains preferable. Fully freezing "
            "B2 prevents additional representation overfitting, but its fixed 192-dimensional embedding is "
            "not linearly sufficient to repair the 123 target-group errors."
        )
    comparison_lines: list[str] = []
    previous_path = args.report_dir / "target16_hierarchical_e2_summary.csv"
    if linear_residual and previous_path.exists():
        previous = pd.read_csv(previous_path).sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
        comparison_lines = [
            "",
            "## Comparison with partially fine-tuned E2",
            "",
            f"- Previous E2 best: alpha={previous['alpha']:.2f}, Accuracy {previous['accuracy']:.6f}, "
            f"Macro-F1 {previous['macro_f1']:.6f}, net rescue {int(previous['net_rescue'])}.",
            f"- Linear residual best: alpha={best['alpha']:.2f}, Accuracy {best['accuracy']:.6f}, "
            f"Macro-F1 {best['macro_f1']:.6f}, net rescue {int(best['net_rescue'])}.",
            "- The capacity reduction removes the previous net gain; a useful next capacity point must lie "
            "between a 3,088-parameter head and the 1,124,977-parameter partial fine-tune.",
        ]
    report = [
        f"# {args.experiment_label} hierarchical fusion",
        "",
        "## Integrity",
        "",
        "- B2 checkpoint: Epoch 25.",
        "- E2 training/selection: 880 Target16 train samples and 222 Target16 validation samples.",
        "- Fusion evaluation: the same 590 validation samples, with no test access.",
        "- Gate audit: 56 correct target, 123 target-to-target errors, 43 target-to-external errors, 43 external-to-target errors.",
        "- Samples outside the B2 Target16 Top-1 gate are unchanged for every alpha.",
        "- All 24 non-target class F1 scores are numerically invariant across the alpha sweep.",
        "",
        "## E2 closed-set training",
        "",
        f"- Best checkpoint: Epoch {int(best_e2['epoch'])}, Target16 validation Accuracy "
        f"{best_e2['val_accuracy']:.6f}, Macro-F1 {best_e2['val_macro_f1']:.6f}.",
        "- B2 Target16 closed-set baseline on all 222 target samples: Accuracy 0.324324, Macro-F1 0.315527.",
        f"- E2 change over that baseline: Accuracy {best_e2['val_accuracy'] - 0.32432432432432434:+.6f}, "
        f"Macro-F1 {best_e2['val_macro_f1'] - 0.3155272729465266:+.6f}.",
        f"- At the selected epoch, train Accuracy was {best_e2['train_accuracy']:.6f}; the train-validation gap "
        f"was {best_e2['generalization_gap']:.6f}.",
        training_interpretation,
        "",
        "## Fixed alpha sweep",
        "",
        _markdown_table(summary),
        "",
        "## Diagnostic result",
        "",
        f"The strongest validation Macro-F1 in the fixed sweep is alpha={best['alpha']:.2f}: "
        f"Accuracy {best['accuracy']:.6f}, Macro-F1 {best['macro_f1']:.6f}, "
        f"rescued {int(best['rescued'])}, harmed {int(best['harmed'])}, net {int(best['net_rescue'])}.",
        f"Relative to B2, this is Accuracy {best['accuracy'] - baseline['accuracy']:+.6f}, "
        f"Macro-F1 {best['macro_f1'] - baseline['macro_f1']:+.6f}, and "
        f"Target16 Macro-F1 {best['target16_macro_f1'] - baseline['target16_macro_f1']:+.6f}.",
        "This is a validation diagnostic, not an independently tested deployment threshold.",
        "",
        "## Largest Target16 gains at alpha=1.0",
        "",
        *gain_lines,
        "",
        "## Largest Target16 harms at alpha=1.0",
        "",
        *harm_lines,
        "",
        "## Validation-user behavior",
        "",
        *user_lines,
        "",
        "## Assessment",
        "",
        assessment,
        *comparison_lines,
    ]
    (args.report_dir / f"{prefix}_experiment.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8",
    )
    print(summary.to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()
