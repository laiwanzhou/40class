from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from analyze_thermal_oof_fusion import cluster_bootstrap_delta, fit_temperature
from evaluate_conditional_expert_routing import route_cross_fitted


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_LOCAL = (
    PROJECT_DIR / "runs" / "p12_local_depth_oof_v2" / "oof_logits.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p12_local_depth_fusion_audit"
DEFAULT_REVIEWED60 = (
    PROJECT_DIR
    / "data"
    / "local_roi_annotation_v2"
    / "blind60_review_annotations.csv"
)
SMALL_ACTION_IDS = np.asarray(
    (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39),
    dtype=np.int64,
)
BASE_VARIANTS = (
    ("full_depth_branch", "depth_logits"),
    ("skeleton_depth", "sd_logits"),
    ("skeleton_depth_imu", "sd_imu_logits"),
    ("current_final", "final_logits"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict cross-fitted OOF audit of the Local Depth expert"
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviewed60", type=Path, default=DEFAULT_REVIEWED60)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)
            ),
            "macro_f1": float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
        }


def load_class_names() -> dict[int, str]:
    result: dict[int, str] = {}
    for path in sorted((PROJECT_DIR / "data" / "subject_folds").glob("fold_*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                result[int(row["class_id"])] = row["class_name"]
    return result


def validate_inputs(
    base: np.lib.npyio.NpzFile,
    local: np.lib.npyio.NpzFile,
) -> None:
    if not np.array_equal(base["sample_ids"].astype(str), local["sample_ids"].astype(str)):
        raise ValueError("Base and Local OOF sample ordering differs")
    if not np.array_equal(base["labels"], local["labels"]):
        raise ValueError("Base and Local OOF labels differ")
    if not np.array_equal(base["folds"], local["held_fold"]):
        raise ValueError("Base and Local held folds differ")
    if len(set(base["sample_ids"].astype(str).tolist())) != len(base["sample_ids"]):
        raise ValueError("Duplicate OOF sample IDs")


def late_fuse_cross_fitted(
    labels: np.ndarray,
    folds: np.ndarray,
    base_logits: np.ndarray,
    local_logits: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    result = np.zeros_like(base_logits, dtype=np.float64)
    protocols: list[dict[str, object]] = []
    weights = np.linspace(0.0, 0.8, 17)
    for held_fold in range(3):
        calibration = folds != held_fold
        target = folds == held_fold
        base_temperature = fit_temperature(
            base_logits[calibration], labels[calibration]
        )
        local_temperature = fit_temperature(
            local_logits[calibration], labels[calibration]
        )
        candidates: list[tuple[float, float, float, float]] = []
        for weight in weights:
            logits = (
                (1.0 - weight) * base_logits[calibration] / base_temperature
                + weight * local_logits[calibration] / local_temperature
            )
            predictions = logits.argmax(1)
            candidates.append(
                (
                    float(accuracy_score(labels[calibration], predictions)),
                    float(
                        f1_score(
                            labels[calibration],
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    -float(weight),
                    float(weight),
                )
            )
        calibration_accuracy, calibration_macro_f1, _, selected_weight = max(
            candidates
        )
        result[target] = (
            (1.0 - selected_weight)
            * base_logits[target]
            / base_temperature
            + selected_weight * local_logits[target] / local_temperature
        )
        protocols.append(
            {
                "held_fold": held_fold,
                "calibration_samples": int(calibration.sum()),
                "target_samples": int(target.sum()),
                "base_temperature": base_temperature,
                "local_temperature": local_temperature,
                "selected_local_weight": selected_weight,
                "calibration_accuracy": calibration_accuracy,
                "calibration_macro_f1": calibration_macro_f1,
            }
        )
    return result, protocols


def per_fold_metrics(
    labels: np.ndarray,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    return {
        str(fold): {
            "base": metric_dict(labels[folds == fold], baseline[folds == fold]),
            "candidate": metric_dict(
                labels[folds == fold], candidate[folds == fold]
            ),
            "accuracy_delta_pp": float(
                100
                * (
                    accuracy_score(
                        labels[folds == fold], candidate[folds == fold]
                    )
                    - accuracy_score(
                        labels[folds == fold], baseline[folds == fold]
                    )
                )
            ),
        }
        for fold in range(3)
    }


def summarize_fusion(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    base_logits: np.ndarray,
    local_logits: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
    fused_logits, protocols = late_fuse_cross_fitted(
        labels, folds, base_logits, local_logits
    )
    base_predictions = base_logits.argmax(1)
    local_predictions = local_logits.argmax(1)
    fused_predictions = fused_logits.argmax(1)
    small = np.isin(labels, SMALL_ACTION_IDS)
    return (
        {
            "base": metric_dict(labels, base_predictions),
            "local": metric_dict(labels, local_predictions),
            "fused": metric_dict(labels, fused_predictions),
            "small_actions": {
                "samples": int(small.sum()),
                "base": metric_dict(labels[small], base_predictions[small]),
                "local": metric_dict(labels[small], local_predictions[small]),
                "fused": metric_dict(labels[small], fused_predictions[small]),
            },
            "base_wrong_local_right": int(
                np.sum((base_predictions != labels) & (local_predictions == labels))
            ),
            "base_right_local_wrong": int(
                np.sum((base_predictions == labels) & (local_predictions != labels))
            ),
            "oracle_choose_base_or_local_accuracy": float(
                np.mean(
                    (base_predictions == labels) | (local_predictions == labels)
                )
            ),
            "per_fold": per_fold_metrics(
                labels, folds, base_predictions, fused_predictions
            ),
            "vs_base_subject_bootstrap": cluster_bootstrap_delta(
                labels,
                fused_predictions,
                base_predictions,
                sample_ids,
                repeats,
                seed,
            ),
            "cross_fit_protocols": protocols,
        },
        fused_logits,
        protocols,
    )


def subject_rows(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    base_predictions: np.ndarray,
    fused_predictions: np.ndarray,
) -> list[dict[str, object]]:
    users = np.asarray([sample_id.split("__")[2] for sample_id in sample_ids])
    rows: list[dict[str, object]] = []
    for user in sorted(set(users.tolist()), key=lambda value: int(value[4:])):
        selected = users == user
        base_accuracy = float(accuracy_score(labels[selected], base_predictions[selected]))
        fused_accuracy = float(
            accuracy_score(labels[selected], fused_predictions[selected])
        )
        rows.append(
            {
                "user_id": user,
                "samples": int(selected.sum()),
                "base_accuracy": base_accuracy,
                "fused_accuracy": fused_accuracy,
                "delta_pp": 100 * (fused_accuracy - base_accuracy),
            }
        )
    return rows


def class_rows(
    labels: np.ndarray,
    base_predictions: np.ndarray,
    local_predictions: np.ndarray,
    fused_predictions: np.ndarray,
    class_names: dict[int, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_id in range(40):
        selected = labels == class_id
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, ""),
                "is_small_action": int(class_id in SMALL_ACTION_IDS),
                "samples": int(selected.sum()),
                "base_accuracy": float(
                    accuracy_score(labels[selected], base_predictions[selected])
                ),
                "local_accuracy": float(
                    accuracy_score(labels[selected], local_predictions[selected])
                ),
                "fused_accuracy": float(
                    accuracy_score(labels[selected], fused_predictions[selected])
                ),
                "base_wrong_local_right": int(
                    np.sum(
                        selected
                        & (base_predictions != labels)
                        & (local_predictions == labels)
                    )
                ),
                "base_right_local_wrong": int(
                    np.sum(
                        selected
                        & (base_predictions == labels)
                        & (local_predictions != labels)
                    )
                ),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reviewed60_audit(
    path: Path,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    depth_predictions: np.ndarray,
    local_predictions: np.ndarray,
    final_predictions: np.ndarray,
) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}

    def summarize(selected_rows: list[dict[str, str]]) -> dict[str, object]:
        indices = np.asarray(
            [lookup[row["sample_id"]] for row in selected_rows], dtype=np.int64
        )
        selected_labels = labels[indices]
        depth = depth_predictions[indices]
        local = local_predictions[indices]
        final = final_predictions[indices]
        return {
            "samples": int(len(indices)),
            "full_depth_accuracy": float(
                accuracy_score(selected_labels, depth)
            ),
            "local_depth_accuracy": float(
                accuracy_score(selected_labels, local)
            ),
            "current_final_accuracy": float(
                accuracy_score(selected_labels, final)
            ),
            "current_final_wrong_local_right": int(
                np.sum((final != selected_labels) & (local == selected_labels))
            ),
        }

    result: dict[str, object] = {"all": summarize(rows)}
    for field in (
        "machine_box_assessment",
        "original_fallback",
        "final_box_source",
    ):
        groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            groups.setdefault(row[field], []).append(row)
        result[field] = {
            value: summarize(group_rows)
            for value, group_rows in sorted(groups.items())
        }
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.base.resolve(), allow_pickle=False) as base, np.load(
        args.local.resolve(), allow_pickle=False
    ) as local:
        validate_inputs(base, local)
        sample_ids = base["sample_ids"].astype(str)
        labels = base["labels"].astype(np.int64)
        folds = base["folds"].astype(np.int64)
        local_logits = local["logits"].astype(np.float64)
        base_arrays = {
            name: base[key].astype(np.float64) for name, key in BASE_VARIANTS
        }
        thermal_candidate_logits = base["thermal_candidate_logits"].astype(np.float64)

    summaries: dict[str, object] = {}
    fused_arrays: dict[str, np.ndarray] = {}
    for index, (name, _) in enumerate(BASE_VARIANTS):
        summary, fused_logits, _ = summarize_fusion(
            sample_ids,
            labels,
            folds,
            base_arrays[name],
            local_logits,
            int(args.bootstrap_repeats),
            int(args.seed) + index * 1000,
        )
        summaries[name] = summary
        fused_arrays[name] = fused_logits

    final_base_logits = base_arrays["current_final"]
    final_base_predictions = final_base_logits.argmax(1)
    local_predictions = local_logits.argmax(1)
    final_fused_predictions = fused_arrays["current_final"].argmax(1)
    routers = {
        "scalar_only": route_cross_fitted(
            labels,
            folds,
            final_base_logits,
            local_logits,
            np.ones(len(labels), dtype=bool),
            include_class=False,
        ),
        "scalar_plus_predicted_class": route_cross_fitted(
            labels,
            folds,
            final_base_logits,
            local_logits,
            np.ones(len(labels), dtype=bool),
            include_class=True,
        ),
    }
    router_summary = {
        name: {
            "metrics": metric_dict(labels, result["predictions"]),
            "route_to_local": int(result["route_candidate"].sum()),
            "per_fold": per_fold_metrics(
                labels, folds, final_base_predictions, result["predictions"]
            ),
            "protocols": result["protocols"],
        }
        for name, result in routers.items()
    }

    sd_imu_predictions = base_arrays["skeleton_depth_imu"].argmax(1)
    thermal_predictions = thermal_candidate_logits.argmax(1)
    sd_imu_wrong = sd_imu_predictions != labels
    thermal_rescue = sd_imu_wrong & (thermal_predictions == labels)
    local_rescue = sd_imu_wrong & (local_predictions == labels)
    rescue_union = thermal_rescue | local_rescue
    rescue_overlap = thermal_rescue & local_rescue
    complementarity = {
        "starting_point": "skeleton_depth_imu",
        "base_wrong_samples": int(sd_imu_wrong.sum()),
        "thermal_rescues": int(thermal_rescue.sum()),
        "local_rescues": int(local_rescue.sum()),
        "overlap_rescues": int(rescue_overlap.sum()),
        "thermal_only_rescues": int((thermal_rescue & ~local_rescue).sum()),
        "local_only_rescues": int((local_rescue & ~thermal_rescue).sum()),
        "rescue_jaccard": float(
            rescue_overlap.sum() / max(1, rescue_union.sum())
        ),
        "current_final_wrong_local_right": int(
            np.sum((final_base_predictions != labels) & (local_predictions == labels))
        ),
    }

    class_names = load_class_names()
    classes = class_rows(
        labels,
        final_base_predictions,
        local_predictions,
        final_fused_predictions,
        class_names,
    )
    subjects = subject_rows(
        sample_ids, labels, final_base_predictions, final_fused_predictions
    )
    write_rows(output_dir / "per_class.csv", classes)
    write_rows(output_dir / "per_subject.csv", subjects)
    with (output_dir / "oof_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "fold",
                "label",
                "current_final_prediction",
                "local_prediction",
                "fixed_low_weight_fused_prediction",
                "scalar_router_prediction",
                "class_router_prediction",
            ]
        )
        writer.writerows(
            zip(
                sample_ids,
                folds,
                labels,
                final_base_predictions,
                local_predictions,
                final_fused_predictions,
                routers["scalar_only"]["predictions"],
                routers["scalar_plus_predicted_class"]["predictions"],
            )
        )
    np.savez_compressed(
        output_dir / "local_fusion_oof.npz",
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        local_logits=local_logits.astype(np.float32),
        current_final_logits=final_base_logits.astype(np.float32),
        fixed_low_weight_fused_logits=fused_arrays["current_final"].astype(np.float32),
    )
    report = {
        "protocol": (
            "Local Depth is trained as three subject-disjoint OOF models. For every "
            "held fold, temperatures and a fixed Local weight are selected using "
            "only the other two folds. Conditional routers are also trained only "
            "on the other two folds. The reviewed 60 ROI samples are locator "
            "evaluation-only and never train the held-fold locator."
        ),
        "samples": int(len(labels)),
        "small_action_samples": int(np.isin(labels, SMALL_ACTION_IDS).sum()),
        "fusion_variants": summaries,
        "conditional_router_on_current_final": router_summary,
        "thermal_local_rescue_overlap": complementarity,
        "reviewed60_classification_audit": reviewed60_audit(
            args.reviewed60.resolve(),
            sample_ids,
            labels,
            base_arrays["full_depth_branch"].argmax(1),
            local_predictions,
            final_base_predictions,
        ),
        "current_final_subjects_improved": int(
            sum(float(row["delta_pp"]) > 0 for row in subjects)
        ),
        "current_final_subjects_harmed": int(
            sum(float(row["delta_pp"]) < 0 for row in subjects)
        ),
        "current_final_subjects_unchanged": int(
            sum(float(row["delta_pp"]) == 0 for row in subjects)
        ),
        "sources": {
            "base_oof": str(args.base.resolve()),
            "local_oof": str(args.local.resolve()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
