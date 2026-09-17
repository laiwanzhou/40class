from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from analyze_peak_sampling_experiment import pair_errors, write_matrix
from analyze_thermal_oof_fusion import (
    align_thermal,
    cross_fitted_thermal_residual,
    load_thermal,
)
from evaluate_conditional_expert_routing import (
    bootstrap_delta,
    route_cross_fitted,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_COMPACT_BASE = (
    PROJECT_DIR
    / "runs"
    / "p20_tiny_imu_student_audit"
    / "complete_oof.npz"
)
DEFAULT_P12_REFERENCE = (
    PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
)
DEFAULT_THERMAL_ROOT = (
    REPO_DIR / "thermal_baseline" / "runs" / "p11_thermal_imagenet_fp16"
)
DEFAULT_STUDENT_SUMMARY = (
    PROJECT_DIR / "runs" / "p20_tiny_imu_student_oof" / "summary.json"
)
DEFAULT_RF_MODEL = (
    PROJECT_DIR
    / "runs"
    / "p3_sd_imu_rf_full18"
    / "imu_random_forest.joblib"
)
DEFAULT_HARD = PROJECT_DIR / "data" / "hard_local_v1" / "hard_action_protocol.json"
DEFAULT_TAXONOMY = (
    PROJECT_DIR / "data" / "six_modality_audit" / "small_action_taxonomy_v1.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "runs" / "p21_tiny_imu_thermal_integration_audit"
)
DEFAULT_GATE = DEFAULT_OUTPUT / "preregistered_gate.json"
SMALL_ACTION_IDS = np.asarray(
    (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39),
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate the P20 tiny IMU student base with the existing Thermal "
            "expert using cross-fitted calibration and routing"
        )
    )
    parser.add_argument(
        "--compact-base", type=Path, default=DEFAULT_COMPACT_BASE
    )
    parser.add_argument(
        "--p12-reference", type=Path, default=DEFAULT_P12_REFERENCE
    )
    parser.add_argument(
        "--thermal-root", type=Path, default=DEFAULT_THERMAL_ROOT
    )
    parser.add_argument("--thermal-logits-name", default="val_logits_fp16.npz")
    parser.add_argument(
        "--student-summary", type=Path, default=DEFAULT_STUDENT_SUMMARY
    )
    parser.add_argument("--rf-model", type=Path, default=DEFAULT_RF_MODEL)
    parser.add_argument("--hard", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true"
        )
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)
            ),
            "macro_f1": float(
                f1_score(
                    labels, predictions, average="macro", zero_division=0
                )
            ),
        }


def metric_bundle(
    labels: np.ndarray,
    predictions: np.ndarray,
    small_ids: np.ndarray,
    hard_ids: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    non_small_ids = np.asarray(
        [
            class_id
            for class_id in range(40)
            if class_id not in set(small_ids.tolist())
        ],
        dtype=np.int64,
    )
    result: dict[str, dict[str, float | int]] = {}
    for name, ids in (
        ("all", np.arange(40)),
        ("small", small_ids),
        ("hard", hard_ids),
        ("non_small", non_small_ids),
    ):
        mask = np.isin(labels, ids)
        result[name] = {
            "samples": int(mask.sum()),
            **metric_dict(labels[mask], predictions[mask]),
        }
    return result


def metric_deltas(
    primary: dict[str, dict[str, float | int]],
    baseline: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    return {
        subset: {
            metric + "_pp": 100.0
            * (
                float(primary[subset][metric])
                - float(baseline[subset][metric])
            )
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        for subset in primary
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = json.loads(args.gate.resolve().read_text(encoding="utf-8"))
    if not bool(gate["frozen_before_evaluation"]):
        raise ValueError("P21 gate is not frozen before evaluation")

    with np.load(args.compact_base.resolve(), allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str)
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int64)
        compact_base_logits = data["student_fused_logits"].astype(np.float64)
        imu_present = data["imu_present"].astype(bool)
    if len(sample_ids) != 2914 or len(np.unique(sample_ids)) != 2914:
        raise ValueError("Expected 2914 unique compact-base OOF rows")

    with np.load(args.p12_reference.resolve(), allow_pickle=False) as data:
        reference_ids = data["sample_ids"].astype(str)
        reference_labels = data["labels"].astype(np.int64)
        reference_folds = data["folds"].astype(np.int64)
        reference_logits = data["final_logits"].astype(np.float64)
        reference_predictions_saved = data["final_predictions"].astype(np.int64)
    if not np.array_equal(sample_ids, reference_ids):
        raise ValueError("P20 compact base and p12 reference IDs differ")
    if not np.array_equal(labels, reference_labels):
        raise ValueError("P20 compact base and p12 reference labels differ")
    if not np.array_equal(folds, reference_folds):
        raise ValueError("P20 compact base and p12 reference folds differ")
    reference_predictions = reference_logits.argmax(1)
    if not np.array_equal(reference_predictions, reference_predictions_saved):
        raise ValueError("Stored p12 logits and predictions differ")

    thermal = load_thermal(
        args.thermal_root.resolve(), str(args.thermal_logits_name)
    )
    thermal_present, thermal_logits = align_thermal(
        sample_ids, labels, thermal
    )
    thermal_result = cross_fitted_thermal_residual(
        "p20_tiny_imu_plus_thermal",
        sample_ids,
        labels,
        folds,
        compact_base_logits,
        thermal,
    )
    fixed_candidate_logits = np.asarray(
        thermal_result["logits"], dtype=np.float64
    )
    fixed_candidate_predictions = fixed_candidate_logits.argmax(1)
    router = route_cross_fitted(
        labels,
        folds,
        compact_base_logits,
        fixed_candidate_logits,
        thermal_present,
        include_class=True,
    )
    route_to_thermal = np.asarray(router["route_candidate"], dtype=bool)
    routed_predictions = np.asarray(router["predictions"], dtype=np.int64)
    routed_logits = compact_base_logits.copy()
    routed_logits[route_to_thermal] = fixed_candidate_logits[route_to_thermal]
    if not np.array_equal(routed_logits.argmax(1), routed_predictions):
        raise ValueError("P21 routed logits and predictions differ")

    hard_ids = np.asarray(
        json.loads(args.hard.resolve().read_text(encoding="utf-8"))[
            "hard_class_ids"
        ],
        dtype=np.int64,
    )
    small_ids = SMALL_ACTION_IDS.copy()
    compact_metrics = metric_bundle(
        labels, compact_base_logits.argmax(1), small_ids, hard_ids
    )
    fixed_metrics = metric_bundle(
        labels, fixed_candidate_predictions, small_ids, hard_ids
    )
    routed_metrics = metric_bundle(
        labels, routed_predictions, small_ids, hard_ids
    )
    reference_metrics = metric_bundle(
        labels, reference_predictions, small_ids, hard_ids
    )
    routed_vs_p20 = metric_deltas(routed_metrics, compact_metrics)
    routed_vs_p12 = metric_deltas(routed_metrics, reference_metrics)
    fixed_vs_p20 = metric_deltas(fixed_metrics, compact_metrics)

    fold_rows: list[dict[str, object]] = []
    for fold in range(3):
        mask = folds == fold
        compact_accuracy = float(
            np.mean(compact_base_logits[mask].argmax(1) == labels[mask])
        )
        fixed_accuracy = float(
            np.mean(fixed_candidate_predictions[mask] == labels[mask])
        )
        routed_accuracy = float(
            np.mean(routed_predictions[mask] == labels[mask])
        )
        reference_accuracy = float(
            np.mean(reference_predictions[mask] == labels[mask])
        )
        fold_rows.append(
            {
                "fold": fold,
                "samples": int(mask.sum()),
                "thermal_present": int(np.sum(mask & thermal_present)),
                "compact_p20_accuracy": compact_accuracy,
                "fixed_thermal_accuracy": fixed_accuracy,
                "routed_p21_accuracy": routed_accuracy,
                "p12_reference_accuracy": reference_accuracy,
                "p21_delta_vs_p20_pp": 100.0
                * (routed_accuracy - compact_accuracy),
                "p21_delta_vs_p12_pp": 100.0
                * (routed_accuracy - reference_accuracy),
            }
        )
    non_worse_fold_count_vs_p12 = sum(
        float(row["p21_delta_vs_p12_pp"]) >= 0.0 for row in fold_rows
    )

    bootstrap_vs_p20 = bootstrap_delta(
        labels,
        routed_predictions,
        compact_base_logits.argmax(1),
        sample_ids,
        int(args.bootstrap_repeats),
        int(args.seed),
    )
    bootstrap_vs_p12 = bootstrap_delta(
        labels,
        routed_predictions,
        reference_predictions,
        sample_ids,
        int(args.bootstrap_repeats),
        int(args.seed) + 1,
    )

    taxonomy = pd.read_csv(args.taxonomy.resolve())
    class_names = {
        int(row.class_id): str(row.action_name)
        for row in taxonomy[["class_id", "action_name"]].itertuples(index=False)
    }
    compact_predictions = compact_base_logits.argmax(1)
    pair_rows: list[dict[str, object]] = []
    for left_value, right_value in gate["target_pairs"]:
        left = int(left_value)
        right = int(right_value)
        compact_pair = pair_errors(
            labels, compact_predictions, left, right
        )
        routed_pair = pair_errors(labels, routed_predictions, left, right)
        reference_pair = pair_errors(
            labels, reference_predictions, left, right
        )
        pair_mask = np.isin(labels, [left, right])
        pair_rows.append(
            {
                "class_a": left,
                "action_a": class_names[left],
                "class_b": right,
                "action_b": class_names[right],
                "p20_bidirectional": compact_pair[2],
                "p21_bidirectional": routed_pair[2],
                "p12_bidirectional": reference_pair[2],
                "p21_error_reduction_vs_p20": compact_pair[2]
                - routed_pair[2],
                "p20_pair_accuracy": float(
                    np.mean(
                        compact_predictions[pair_mask] == labels[pair_mask]
                    )
                ),
                "p21_pair_accuracy": float(
                    np.mean(
                        routed_predictions[pair_mask] == labels[pair_mask]
                    )
                ),
                "p12_pair_accuracy": float(
                    np.mean(
                        reference_predictions[pair_mask] == labels[pair_mask]
                    )
                ),
            }
        )
    pd.DataFrame(pair_rows).to_csv(
        output_dir / "target_pair_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    compact_matrix = confusion_matrix(
        labels, compact_predictions, labels=np.arange(40)
    )
    routed_matrix = confusion_matrix(
        labels, routed_predictions, labels=np.arange(40)
    )
    union_ids = set(small_ids.tolist()) | set(hard_ids.tolist())
    ranked_pairs: list[tuple[int, int, int]] = []
    for left in range(40):
        for right in range(left + 1, 40):
            if left not in union_ids and right not in union_ids:
                continue
            count = int(
                compact_matrix[left, right] + compact_matrix[right, left]
            )
            ranked_pairs.append((count, left, right))
    ranked_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    top20_rows: list[dict[str, object]] = []
    for rank, (compact_count, left, right) in enumerate(
        ranked_pairs[:20], start=1
    ):
        pair_mask = np.isin(labels, [left, right])
        routed_count = int(
            routed_matrix[left, right] + routed_matrix[right, left]
        )
        top20_rows.append(
            {
                "rank": rank,
                "class_a": left,
                "action_a": class_names[left],
                "class_b": right,
                "action_b": class_names[right],
                "pair_samples": int(pair_mask.sum()),
                "p20_bidirectional": compact_count,
                "p21_bidirectional": routed_count,
                "error_reduction": compact_count - routed_count,
                "p20_pair_accuracy": float(
                    np.mean(
                        compact_predictions[pair_mask] == labels[pair_mask]
                    )
                ),
                "p21_pair_accuracy": float(
                    np.mean(
                        routed_predictions[pair_mask] == labels[pair_mask]
                    )
                ),
            }
        )
    pd.DataFrame(top20_rows).to_csv(
        output_dir / "p20_top20_pair_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_class_rows: list[dict[str, object]] = []
    for class_id in range(40):
        mask = labels == class_id
        p20_recall = float(
            np.mean(compact_predictions[mask] == labels[mask])
        )
        p21_recall = float(
            np.mean(routed_predictions[mask] == labels[mask])
        )
        p12_recall = float(
            np.mean(reference_predictions[mask] == labels[mask])
        )
        per_class_rows.append(
            {
                "class_id": class_id,
                "action_name": class_names[class_id],
                "samples": int(mask.sum()),
                "is_small": int(class_id in set(small_ids.tolist())),
                "is_hard": int(class_id in set(hard_ids.tolist())),
                "p20_recall": p20_recall,
                "p21_recall": p21_recall,
                "p12_recall": p12_recall,
                "p21_delta_vs_p20_pp": 100.0
                * (p21_recall - p20_recall),
                "p21_delta_vs_p12_pp": 100.0
                * (p21_recall - p12_recall),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / "per_class_recall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    subjects = np.asarray(
        [
            int(re.search(r"__user(\d+)__", sample_id).group(1))
            for sample_id in sample_ids
        ]
    )
    subject_rows: list[dict[str, object]] = []
    for subject in sorted(np.unique(subjects).tolist()):
        subject_mask = subjects == subject
        row: dict[str, object] = {"subject": f"user{subject}"}
        for subset_name, subset_ids in (
            ("all", np.arange(40)),
            ("small", small_ids),
            ("hard", hard_ids),
        ):
            mask = subject_mask & np.isin(labels, subset_ids)
            p20_accuracy = float(
                np.mean(compact_predictions[mask] == labels[mask])
            )
            p21_accuracy = float(
                np.mean(routed_predictions[mask] == labels[mask])
            )
            p12_accuracy = float(
                np.mean(reference_predictions[mask] == labels[mask])
            )
            row[f"{subset_name}_n"] = int(mask.sum())
            row[f"{subset_name}_p20_accuracy"] = p20_accuracy
            row[f"{subset_name}_p21_accuracy"] = p21_accuracy
            row[f"{subset_name}_p12_accuracy"] = p12_accuracy
            row[f"{subset_name}_p21_vs_p20_pp"] = 100.0 * (
                p21_accuracy - p20_accuracy
            )
            row[f"{subset_name}_p21_vs_p12_pp"] = 100.0 * (
                p21_accuracy - p12_accuracy
            )
        subject_rows.append(row)
    pd.DataFrame(subject_rows).to_csv(
        output_dir / "per_subject_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_sample_rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        p20_correct = compact_predictions[index] == labels[index]
        p21_correct = routed_predictions[index] == labels[index]
        p12_correct = reference_predictions[index] == labels[index]
        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "fold": int(folds[index]),
                "class_id": int(labels[index]),
                "action_name": class_names[int(labels[index])],
                "imu_present": int(imu_present[index]),
                "thermal_present": int(thermal_present[index]),
                "route_to_thermal": int(route_to_thermal[index]),
                "route_probability": float(router["probability"][index]),
                "p20_prediction": int(compact_predictions[index]),
                "p21_prediction": int(routed_predictions[index]),
                "p12_prediction": int(reference_predictions[index]),
                "p21_vs_p20_outcome": (
                    "win"
                    if (not p20_correct and p21_correct)
                    else "loss"
                    if (p20_correct and not p21_correct)
                    else "both_correct"
                    if p20_correct
                    else "both_wrong"
                ),
                "p21_vs_p12_outcome": (
                    "win"
                    if (not p12_correct and p21_correct)
                    else "loss"
                    if (p12_correct and not p21_correct)
                    else "both_correct"
                    if p12_correct
                    else "both_wrong"
                ),
            }
        )
    pd.DataFrame(per_sample_rows).to_csv(
        output_dir / "per_sample_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_matrix(
        output_dir / "small_actions_confusion_21x40.csv",
        labels,
        routed_predictions,
        small_ids.tolist(),
        class_names,
    )
    write_matrix(
        output_dir / "hard_classes_confusion_21x40.csv",
        labels,
        routed_predictions,
        hard_ids.tolist(),
        class_names,
    )
    np.savez_compressed(
        output_dir / "complete_oof.npz",
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        compact_p20_logits=compact_base_logits.astype(np.float32),
        imu_present=imu_present.astype(np.uint8),
        thermal_logits=thermal_logits.astype(np.float32),
        thermal_present=thermal_present.astype(np.uint8),
        fixed_thermal_logits=fixed_candidate_logits.astype(np.float32),
        route_probability=np.asarray(
            router["probability"], dtype=np.float32
        ),
        route_to_thermal=route_to_thermal.astype(np.uint8),
        routed_p21_logits=routed_logits.astype(np.float32),
        routed_p21_predictions=routed_predictions,
        p12_reference_logits=reference_logits.astype(np.float32),
        p12_reference_predictions=reference_predictions,
    )

    student_summary = json.loads(
        args.student_summary.resolve().read_text(encoding="utf-8")
    )
    student_size_mib = float(student_summary["max_checkpoint_size_mib"])
    rf_size_mib = args.rf_model.resolve().stat().st_size / 1024**2
    imu_size_reduction_mib = rf_size_mib - student_size_mib
    thresholds = gate["criteria"]
    bootstrap_lower = float(
        bootstrap_vs_p20["subject_cluster_bootstrap_95_ci_pp"][0]
    )
    criteria = {
        "overall_accuracy_delta_vs_p12": bool(
            routed_vs_p12["all"]["accuracy_pp"]
            >= float(thresholds["overall_accuracy_delta_vs_p12_pp_min"])
        ),
        "small_action_accuracy_delta_vs_p12": bool(
            routed_vs_p12["small"]["accuracy_pp"]
            >= float(
                thresholds["small_action_accuracy_delta_vs_p12_pp_min"]
            )
        ),
        "hard_class_accuracy_delta_vs_p12": bool(
            routed_vs_p12["hard"]["accuracy_pp"]
            >= float(
                thresholds["hard_class_accuracy_delta_vs_p12_pp_min"]
            )
        ),
        "non_worse_fold_count_vs_p12": bool(
            non_worse_fold_count_vs_p12
            >= int(thresholds["non_worse_fold_count_vs_p12_min"])
        ),
        "thermal_routed_overall_delta_vs_p20": bool(
            routed_vs_p20["all"]["accuracy_pp"]
            >= float(
                thresholds["thermal_routed_overall_delta_vs_p20_pp_min"]
            )
        ),
        "thermal_routed_small_delta_vs_p20": bool(
            routed_vs_p20["small"]["accuracy_pp"]
            >= float(
                thresholds["thermal_routed_small_delta_vs_p20_pp_min"]
            )
        ),
        "thermal_routed_hard_delta_vs_p20": bool(
            routed_vs_p20["hard"]["accuracy_pp"]
            >= float(
                thresholds["thermal_routed_hard_delta_vs_p20_pp_min"]
            )
        ),
        "thermal_vs_p20_subject_bootstrap_lower": bool(
            bootstrap_lower
            >= float(
                thresholds[
                    "thermal_vs_p20_subject_bootstrap_95_ci_lower_pp_min"
                ]
            )
        ),
        "student_checkpoint_size": bool(
            student_size_mib
            <= float(thresholds["student_checkpoint_size_mib_max"])
        ),
        "imu_model_size_reduction_vs_rf": bool(
            imu_size_reduction_mib
            >= float(
                thresholds["imu_model_size_reduction_vs_rf_mib_min"]
            )
        ),
    }
    p20_outcomes = {
        outcome: sum(
            row["p21_vs_p20_outcome"] == outcome for row in per_sample_rows
        )
        for outcome in ("win", "loss", "both_correct", "both_wrong")
    }
    p12_outcomes = {
        outcome: sum(
            row["p21_vs_p12_outcome"] == outcome for row in per_sample_rows
        )
        for outcome in ("win", "loss", "both_correct", "both_wrong")
    }
    summary = {
        "status": "complete",
        "experiment": "p16_visual_plus_tiny_imu_plus_thermal_router",
        "preregistered_gate": gate,
        "accuracy_replacement_gate_passed": bool(all(criteria.values())),
        "criteria_passed": criteria,
        "counts": {
            "samples": int(len(labels)),
            "imu_present": int(imu_present.sum()),
            "thermal_present": int(thermal_present.sum()),
            "both_present": int((imu_present & thermal_present).sum()),
            "routed_to_thermal": int(route_to_thermal.sum()),
        },
        "metrics": {
            "compact_p20_base": compact_metrics,
            "fixed_thermal_candidate": fixed_metrics,
            "routed_p21_candidate": routed_metrics,
            "p12_accuracy_reference": reference_metrics,
        },
        "deltas": {
            "fixed_thermal_vs_p20": fixed_vs_p20,
            "routed_p21_vs_p20": routed_vs_p20,
            "routed_p21_vs_p12": routed_vs_p12,
        },
        "folds": fold_rows,
        "non_worse_fold_count_vs_p12": non_worse_fold_count_vs_p12,
        "bootstrap": {
            "routed_p21_vs_p20": bootstrap_vs_p20,
            "routed_p21_vs_p12": bootstrap_vs_p12,
        },
        "thermal_protocols": thermal_result["protocols"],
        "router_protocols": router["protocols"],
        "model_size": {
            "student_checkpoint_mib": student_size_mib,
            "rf_reference_mib": rf_size_mib,
            "imu_model_size_reduction_mib": imu_size_reduction_mib,
        },
        "win_loss": {
            "vs_p20": p20_outcomes,
            "vs_p12": p12_outcomes,
        },
        "sources": {
            "compact_base": str(args.compact_base.resolve()),
            "p12_reference": str(args.p12_reference.resolve()),
            "thermal_root": str(args.thermal_root.resolve()),
            "thermal_logits_name": str(args.thermal_logits_name),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
