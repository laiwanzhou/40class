from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from analyze_local_depth_oof_fusion import SMALL_ACTION_IDS, late_fuse_cross_fitted
from analyze_peak_sampling_experiment import (
    align_candidate,
    metrics,
    pair_errors,
    subset_metrics,
    write_matrix,
)
from evaluate_visual_base_with_imu import fit_temperature


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_SHARED = (
    PROJECT_DIR / "runs" / "p16_shared_full_local_oracle_oof" / "oof_logits.npz"
)
DEFAULT_IMU = PROJECT_DIR / "runs" / "p20_tiny_imu_student_oof" / "oof_logits.npz"
DEFAULT_TRAINING_SUMMARY = (
    PROJECT_DIR / "runs" / "p20_tiny_imu_student_oof" / "summary.json"
)
DEFAULT_HARD = PROJECT_DIR / "data" / "hard_local_v1" / "hard_action_protocol.json"
DEFAULT_TAXONOMY = (
    PROJECT_DIR / "data" / "six_modality_audit" / "small_action_taxonomy_v1.csv"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p20_tiny_imu_student_audit"
DEFAULT_GATE = DEFAULT_OUTPUT / "preregistered_gate.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the tiny device-aware IMU student OOF experiment"
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--imu", type=Path, default=DEFAULT_IMU)
    parser.add_argument(
        "--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY
    )
    parser.add_argument("--hard", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def missing_aware_cross_fitted_fusion(
    labels: np.ndarray,
    folds: np.ndarray,
    visual_logits: np.ndarray,
    imu_logits: np.ndarray,
    imu_present: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    result = np.zeros_like(visual_logits, dtype=np.float64)
    protocols: list[dict[str, object]] = []
    weights = np.linspace(0.0, 0.5, 11)
    for held_fold in range(3):
        tune = (folds != held_fold) & imu_present
        target = folds == held_fold
        target_present = target & imu_present
        visual_temperature = fit_temperature(
            visual_logits[tune], labels[tune]
        )
        imu_temperature = fit_temperature(imu_logits[tune], labels[tune])
        candidates: list[tuple[float, float, float, float]] = []
        for weight in weights:
            candidate = (
                (1.0 - weight)
                * visual_logits[tune]
                / visual_temperature
                + weight * imu_logits[tune] / imu_temperature
            )
            prediction = candidate.argmax(1)
            candidate_metrics = metrics(labels[tune], candidate)
            candidates.append(
                (
                    float(candidate_metrics["accuracy"]),
                    float(candidate_metrics["macro_f1"]),
                    -float(weight),
                    float(weight),
                )
            )
        _, _, _, selected_weight = max(candidates)
        result[target] = visual_logits[target] / visual_temperature
        result[target_present] = (
            (1.0 - selected_weight)
            * visual_logits[target_present]
            / visual_temperature
            + selected_weight
            * imu_logits[target_present]
            / imu_temperature
        )
        protocols.append(
            {
                "held_fold": held_fold,
                "calibration_imu_present": int(tune.sum()),
                "target_samples": int(target.sum()),
                "target_imu_present": int(target_present.sum()),
                "visual_temperature": visual_temperature,
                "imu_temperature": imu_temperature,
                "selected_imu_weight": selected_weight,
            }
        )
    return result, protocols


def align_imu(
    base_ids: np.ndarray,
    base_labels: np.ndarray,
    base_folds: np.ndarray,
    imu_file: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    lookup = {
        sample_id: index
        for index, sample_id in enumerate(base_ids.astype(str).tolist())
    }
    imu_ids = imu_file["sample_ids"].astype(str)
    if len(set(imu_ids.tolist())) != len(imu_ids):
        raise ValueError("IMU OOF IDs are not unique")
    keep = np.asarray([sample_id in lookup for sample_id in imu_ids], dtype=bool)
    imu_indices = np.flatnonzero(keep)
    indices = np.asarray(
        [lookup[sample_id] for sample_id in imu_ids[keep]], dtype=np.int64
    )
    if not np.array_equal(
        base_labels[indices], imu_file["labels"].astype(int)[imu_indices]
    ):
        raise ValueError("IMU labels do not align with the visual base")
    if not np.array_equal(
        base_folds[indices], imu_file["held_fold"].astype(int)[imu_indices]
    ):
        raise ValueError("IMU folds do not align with the visual base")
    present = np.zeros(len(base_ids), dtype=bool)
    present[indices] = True
    student = np.zeros((len(base_ids), 40), dtype=np.float64)
    teacher = np.zeros((len(base_ids), 40), dtype=np.float64)
    student[indices] = imu_file["student_logits"].astype(np.float64)[imu_indices]
    teacher[indices] = imu_file["teacher_logits"].astype(np.float64)[imu_indices]
    return present, student, teacher, int((~keep).sum())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = json.loads(args.gate.resolve().read_text(encoding="utf-8"))
    if not bool(gate["frozen_before_training"]):
        raise ValueError("The preregistered gate is not frozen")
    training_summary = json.loads(
        args.training_summary.resolve().read_text(encoding="utf-8")
    )

    base = np.load(args.base.resolve(), allow_pickle=False)
    shared_file = np.load(args.shared.resolve(), allow_pickle=False)
    imu_file = np.load(args.imu.resolve(), allow_pickle=False)
    sample_ids = base["sample_ids"].astype(str)
    labels = base["labels"].astype(int)
    folds = base["folds"].astype(int)
    shared = align_candidate(sample_ids, labels, folds, shared_file)
    visual_logits, visual_protocol = late_fuse_cross_fitted(
        labels,
        folds,
        base["skeleton_logits"].astype(np.float64),
        shared["fused_logits"].astype(np.float64),
    )
    imu_present, student_logits, teacher_logits, imu_unmatched = align_imu(
        sample_ids, labels, folds, imu_file
    )
    student_fused, student_protocol = missing_aware_cross_fitted_fusion(
        labels, folds, visual_logits, student_logits, imu_present
    )
    teacher_fused, teacher_protocol = missing_aware_cross_fitted_fusion(
        labels, folds, visual_logits, teacher_logits, imu_present
    )
    visual_pred = visual_logits.argmax(1)
    student_pred = student_fused.argmax(1)
    teacher_pred = teacher_fused.argmax(1)

    taxonomy = pd.read_csv(args.taxonomy.resolve())
    class_names = {
        int(row.class_id): str(row.action_name)
        for row in taxonomy[["class_id", "action_name"]].itertuples(index=False)
    }
    small_ids = [int(value) for value in SMALL_ACTION_IDS.tolist()]
    hard_ids = [
        int(value)
        for value in json.loads(args.hard.resolve().read_text(encoding="utf-8"))[
            "hard_class_ids"
        ]
    ]
    small = np.asarray(small_ids, dtype=int)
    hard = np.asarray(hard_ids, dtype=int)
    non_small = np.asarray(
        [class_id for class_id in range(40) if class_id not in set(small_ids)],
        dtype=int,
    )
    subsets = {
        "all": np.arange(40),
        "small": small,
        "hard": hard,
        "non_small": non_small,
    }

    visual_metrics = {
        name: subset_metrics(labels, visual_logits, ids)
        for name, ids in subsets.items()
    }
    student_metrics = {
        name: subset_metrics(labels, student_fused, ids)
        for name, ids in subsets.items()
    }
    teacher_metrics = {
        name: subset_metrics(labels, teacher_fused, ids)
        for name, ids in subsets.items()
    }
    deltas: dict[str, dict[str, float]] = {}
    for name in subsets:
        deltas[name] = {
            key + "_pp": 100.0
            * (
                float(student_metrics[name][key])
                - float(visual_metrics[name][key])
            )
            for key in ("accuracy", "balanced_accuracy", "macro_f1")
        }

    student_standalone = metrics(
        labels[imu_present], student_logits[imu_present]
    )
    teacher_standalone = metrics(
        labels[imu_present], teacher_logits[imu_present]
    )
    teacher_gain = (
        teacher_metrics["all"]["accuracy"] - visual_metrics["all"]["accuracy"]
    )
    student_gain = (
        student_metrics["all"]["accuracy"] - visual_metrics["all"]["accuracy"]
    )
    teacher_gain_retention = (
        float(student_gain / teacher_gain) if teacher_gain > 0 else float("-inf")
    )

    fold_rows: list[dict[str, object]] = []
    for fold in range(3):
        mask = folds == fold
        visual_accuracy = float(np.mean(visual_pred[mask] == labels[mask]))
        student_accuracy = float(np.mean(student_pred[mask] == labels[mask]))
        teacher_accuracy = float(np.mean(teacher_pred[mask] == labels[mask]))
        fold_rows.append(
            {
                "fold": fold,
                "samples": int(mask.sum()),
                "imu_present": int(np.sum(mask & imu_present)),
                "visual_accuracy": visual_accuracy,
                "student_fused_accuracy": student_accuracy,
                "teacher_fused_accuracy": teacher_accuracy,
                "student_delta_pp": 100.0
                * (student_accuracy - visual_accuracy),
                "teacher_delta_pp": 100.0
                * (teacher_accuracy - visual_accuracy),
            }
        )
    positive_folds = sum(
        float(row["student_delta_pp"]) > 0.0 for row in fold_rows
    )

    target_pairs = [
        (int(pair[0]), int(pair[1])) for pair in gate["target_pairs"]
    ]
    pair_rows: list[dict[str, object]] = []
    improved_target_pairs = 0
    for left, right in target_pairs:
        baseline = pair_errors(labels, visual_pred, left, right)
        candidate = pair_errors(labels, student_pred, left, right)
        teacher_pair = pair_errors(labels, teacher_pred, left, right)
        improved = candidate[2] < baseline[2]
        improved_target_pairs += int(improved)
        pair_rows.append(
            {
                "class_a": left,
                "action_a": class_names[left],
                "class_b": right,
                "action_b": class_names[right],
                "visual_bidirectional": baseline[2],
                "student_bidirectional": candidate[2],
                "teacher_bidirectional": teacher_pair[2],
                "student_error_reduction": baseline[2] - candidate[2],
                "improved": int(improved),
            }
        )
    pd.DataFrame(pair_rows).to_csv(
        output_dir / "target_pair_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_matrix = confusion_matrix(labels, visual_pred, labels=np.arange(40))
    student_matrix = confusion_matrix(labels, student_pred, labels=np.arange(40))
    union_ids = set(small_ids) | set(hard_ids)
    ranked_pairs: list[tuple[int, int, int]] = []
    for left in range(40):
        for right in range(left + 1, 40):
            if left not in union_ids and right not in union_ids:
                continue
            count = int(baseline_matrix[left, right] + baseline_matrix[right, left])
            ranked_pairs.append((count, left, right))
    ranked_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    top20_rows: list[dict[str, object]] = []
    for rank, (baseline_count, left, right) in enumerate(ranked_pairs[:20], 1):
        pair_mask = np.isin(labels, [left, right])
        student_count = int(
            student_matrix[left, right] + student_matrix[right, left]
        )
        top20_rows.append(
            {
                "rank": rank,
                "class_a": left,
                "action_a": class_names[left],
                "class_b": right,
                "action_b": class_names[right],
                "pair_samples": int(pair_mask.sum()),
                "visual_bidirectional": baseline_count,
                "student_bidirectional": student_count,
                "error_reduction": baseline_count - student_count,
                "visual_pair_accuracy": float(
                    np.mean(visual_pred[pair_mask] == labels[pair_mask])
                ),
                "student_pair_accuracy": float(
                    np.mean(student_pred[pair_mask] == labels[pair_mask])
                ),
            }
        )
    pd.DataFrame(top20_rows).to_csv(
        output_dir / "baseline_top20_pair_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_sample_rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        visual_correct = bool(visual_pred[index] == labels[index])
        student_correct = bool(student_pred[index] == labels[index])
        subject_match = re.search(r"__user(\d+)__", sample_id)
        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "fold": int(folds[index]),
                "subject": (
                    f"user{subject_match.group(1)}" if subject_match else ""
                ),
                "class_id": int(labels[index]),
                "action_name": class_names[int(labels[index])],
                "imu_present": int(imu_present[index]),
                "is_small": int(labels[index] in small_ids),
                "is_hard": int(labels[index] in hard_ids),
                "visual_prediction": int(visual_pred[index]),
                "student_prediction": int(student_pred[index]),
                "teacher_prediction": int(teacher_pred[index]),
                "outcome": (
                    "win"
                    if (not visual_correct and student_correct)
                    else "loss"
                    if (visual_correct and not student_correct)
                    else "both_correct"
                    if visual_correct
                    else "both_wrong"
                ),
            }
        )
    pd.DataFrame(per_sample_rows).to_csv(
        output_dir / "per_sample_win_loss.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_class_rows: list[dict[str, object]] = []
    for class_id in range(40):
        mask = labels == class_id
        baseline_recall = float(np.mean(visual_pred[mask] == labels[mask]))
        student_recall = float(np.mean(student_pred[mask] == labels[mask]))
        teacher_recall = float(np.mean(teacher_pred[mask] == labels[mask]))
        per_class_rows.append(
            {
                "class_id": class_id,
                "action_name": class_names[class_id],
                "samples": int(mask.sum()),
                "is_small": int(class_id in small_ids),
                "is_hard": int(class_id in hard_ids),
                "visual_recall": baseline_recall,
                "student_recall": student_recall,
                "teacher_recall": teacher_recall,
                "student_delta_pp": 100.0
                * (student_recall - baseline_recall),
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
        for subset_name, subset_ids in [
            ("all", np.arange(40)),
            ("small", small),
            ("hard", hard),
        ]:
            mask = subject_mask & np.isin(labels, subset_ids)
            visual_accuracy = float(
                np.mean(visual_pred[mask] == labels[mask])
            )
            student_accuracy = float(
                np.mean(student_pred[mask] == labels[mask])
            )
            row[f"{subset_name}_n"] = int(mask.sum())
            row[f"{subset_name}_visual_accuracy"] = visual_accuracy
            row[f"{subset_name}_student_accuracy"] = student_accuracy
            row[f"{subset_name}_delta_pp"] = 100.0 * (
                student_accuracy - visual_accuracy
            )
        subject_rows.append(row)
    pd.DataFrame(subject_rows).to_csv(
        output_dir / "per_subject_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_matrix(
        output_dir / "small_actions_confusion_21x40.csv",
        labels,
        student_pred,
        small_ids,
        class_names,
    )
    write_matrix(
        output_dir / "hard_classes_confusion_21x40.csv",
        labels,
        student_pred,
        hard_ids,
        class_names,
    )
    np.savez_compressed(
        output_dir / "complete_oof.npz",
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        visual_logits=visual_logits.astype(np.float32),
        imu_present=imu_present.astype(np.uint8),
        student_imu_logits=student_logits.astype(np.float32),
        teacher_imu_logits=teacher_logits.astype(np.float32),
        student_fused_logits=student_fused.astype(np.float32),
        teacher_fused_logits=teacher_fused.astype(np.float32),
    )

    thresholds = gate["criteria"]
    model_size_mib = float(training_summary["max_checkpoint_size_mib"])
    criteria = {
        "overall_accuracy_delta": bool(
            deltas["all"]["accuracy_pp"]
            >= float(thresholds["overall_accuracy_delta_pp_min"])
        ),
        "small_action_accuracy_delta": bool(
            deltas["small"]["accuracy_pp"]
            >= float(thresholds["small_action_accuracy_delta_pp_min"])
        ),
        "hard_class_accuracy_delta": bool(
            deltas["hard"]["accuracy_pp"]
            >= float(thresholds["hard_class_accuracy_delta_pp_min"])
        ),
        "positive_fold_count": bool(
            positive_folds >= int(thresholds["positive_fold_count_min"])
        ),
        "improved_target_pair_count": bool(
            improved_target_pairs
            >= int(thresholds["improved_target_pair_count_min"])
        ),
        "non_small_balanced_accuracy_delta": bool(
            deltas["non_small"]["balanced_accuracy_pp"]
            >= float(
                thresholds["non_small_balanced_accuracy_delta_pp_min"]
            )
        ),
        "teacher_overall_fusion_gain_retention": bool(
            teacher_gain_retention
            >= float(thresholds["teacher_overall_fusion_gain_retention_min"])
        ),
        "student_standalone_accuracy": bool(
            student_standalone["accuracy"]
            >= float(thresholds["student_standalone_accuracy_min"])
        ),
        "deployment_model_size": bool(
            model_size_mib
            <= float(thresholds["deployment_model_size_mib_max"])
        ),
    }
    summary = {
        "status": "complete",
        "experiment": "tiny_device_aware_imu_student",
        "preregistered_gate": gate,
        "preregistered_gate_passed": bool(all(criteria.values())),
        "criteria_passed": criteria,
        "counts": {
            "samples": int(len(labels)),
            "imu_present": int(imu_present.sum()),
            "imu_missing_visual_fallback": int((~imu_present).sum()),
            "usable_imu_without_visual_base": imu_unmatched,
        },
        "visual_baseline": visual_metrics,
        "student_candidate": student_metrics,
        "rf_teacher_candidate": teacher_metrics,
        "student_delta_vs_visual": deltas,
        "student_standalone": student_standalone,
        "rf_teacher_standalone": teacher_standalone,
        "teacher_overall_fusion_gain_retention": teacher_gain_retention,
        "max_checkpoint_size_mib": model_size_mib,
        "parameters": int(training_summary["parameters"]),
        "folds": fold_rows,
        "positive_folds": positive_folds,
        "improved_target_pairs": improved_target_pairs,
        "cross_fitted_protocol": {
            "visual": visual_protocol,
            "student": student_protocol,
            "teacher": teacher_protocol,
        },
        "win_loss": {
            outcome: sum(row["outcome"] == outcome for row in per_sample_rows)
            for outcome in ("win", "loss", "both_correct", "both_wrong")
        },
        "teacher_reproduction_max_abs_probability_delta": float(
            max(
                float(row["teacher_reference_max_abs_probability_delta"])
                for row in training_summary["folds"]
            )
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
