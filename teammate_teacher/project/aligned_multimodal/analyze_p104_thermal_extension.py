"""Audit Thermal as a sixth P104 modality without reopening the frozen pair plan."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


HERE = Path(__file__).resolve().parent
MODALITY_RUNS = {
    "GlobalV": HERE / "runs/p104_globalv_specialists_oof_v1",
    "LocalV": HERE / "runs/p104_localv_specialists_oof_v1",
    "Skeleton": HERE / "runs/p104_skeleton_specialists_oof_v1",
    "IMU": HERE / "runs/p104_imu_specialists_oof_v1",
    "Depth": HERE / "runs/p104_depth_specialists_oof_v1",
    "Thermal": HERE / "runs/p104_thermal_specialists_oof_v1",
}
DEFAULT_PLAN = HERE / "runs/p104_single_modality_audit_v1/pair_plan.json"
DEFAULT_CORE_FINAL = HERE / "runs/p104_final_audit_v1/summary.json"
DEFAULT_OUTPUT = HERE / "runs/p104_thermal_extension_audit_v1"

KNOWLEDGE_MAP = [
    {"family_key": "21__22", "best_evidence": "Thermal", "candidate_specialist": None, "decision": "physical evidence candidate; simple trigger is net-negative"},
    {"family_key": "24__26", "best_evidence": "LocalV", "candidate_specialist": None, "decision": "family head gain lacks sample-correspondence evidence"},
    {"family_key": "24__27", "best_evidence": None, "candidate_specialist": None, "decision": "all source-selected evidence is held-negative"},
    {"family_key": "32__34", "best_evidence": None, "candidate_specialist": None, "decision": "A baseline is stronger than current specialists"},
    {"family_key": "38__39", "best_evidence": "LocalV/GlobalV", "candidate_specialist": None, "decision": "positive visual evidence but modality identity and sample count are insufficient"},
    {"family_key": "3__5", "best_evidence": "LocalV+Skeleton", "candidate_specialist": "LocalV+Skeleton family head", "decision": "enter next-stage specialist research"},
    {"family_key": "6__37", "best_evidence": "LocalV", "candidate_specialist": None, "decision": "small, fold-unstable gain and net-negative trigger"},
    {"family_key": "7__37", "best_evidence": "LocalV", "candidate_specialist": "LocalV family head", "decision": "enter next-stage specialist research"},
    {"family_key": "7__8", "best_evidence": "GlobalV", "candidate_specialist": "GlobalV family head", "decision": "enter next-stage specialist research"},
    {"family_key": "8__10", "best_evidence": "LocalV (source-selected); Depth held diagnostic", "candidate_specialist": None, "decision": "source-selected LocalV ties shuffle; Depth cannot be chosen from held results"},
    {"family_key": "8__9", "best_evidence": "LocalV", "candidate_specialist": "LocalV family head", "decision": "enter next-stage specialist research"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in MODALITY_RUNS.items():
        parser.add_argument(f"--{name.lower()}-run", type=Path, default=path)
    parser.add_argument("--core-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--core-final", type=Path, default=DEFAULT_CORE_FINAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def source_metric_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["balanced_accuracy"]),
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
    )


def load_outer_predictions(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "outer_held_family":
                continue
            grouped[(row["family_key"], row["variant"])].append(
                {
                    "outer_fold": int(row["outer_fold"]),
                    "row_index": int(row["row_index"]),
                    "subject": row["subject"],
                    "label": int(row["label"]),
                    "a_prediction": int(row["a_prediction"]),
                    "specialist_prediction": int(row["specialist_prediction"]),
                }
            )
    return grouped


def classification_metrics(rows: list[dict[str, Any]], classes: list[int]) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    prediction = np.asarray(
        [row["specialist_prediction"] for row in rows], dtype=np.int64
    )
    recalls = {
        str(class_id): float(np.mean(prediction[labels == class_id] == class_id))
        for class_id in classes
    }
    return {
        "rows": len(rows),
        "correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "macro_f1": float(
            f1_score(labels, prediction, labels=classes, average="macro", zero_division=0)
        ),
        "per_class_recall": recalls,
    }


def subject_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["subject"]].append(row)
    records = []
    for subject, values in sorted(grouped.items()):
        a_correct = sum(value["a_prediction"] == value["label"] for value in values)
        specialist = sum(
            value["specialist_prediction"] == value["label"] for value in values
        )
        records.append(
            {
                "subject": subject,
                "rows": len(values),
                "a_correct": a_correct,
                "specialist_correct": specialist,
                "net": specialist - a_correct,
            }
        )
    return {
        "positive_subjects": sum(value["net"] > 0 for value in records),
        "neutral_subjects": sum(value["net"] == 0 for value in records),
        "negative_subjects": sum(value["net"] < 0 for value in records),
        "worst_subject": min(records, key=lambda value: (value["net"], value["subject"])),
        "records": records,
    }


def modality_matrix_row(
    modality: str,
    family: str,
    classes: list[int],
    results: list[dict[str, Any]],
    predictions: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    aligned_rows = predictions[(family, "aligned")]
    shuffle_rows = predictions[(family, "shuffle")]
    zero_rows = predictions[(family, "zero")]
    aligned_metrics = classification_metrics(aligned_rows, classes)
    shuffle_correct = classification_metrics(shuffle_rows, classes)["correct"]
    zero_correct = classification_metrics(zero_rows, classes)["correct"]
    a_correct = sum(row["a_prediction"] == row["label"] for row in aligned_rows)
    rescue = sum(
        row["a_prediction"] != row["label"]
        and row["specialist_prediction"] == row["label"]
        for row in aligned_rows
    )
    harm = sum(
        row["a_prediction"] == row["label"]
        and row["specialist_prediction"] != row["label"]
        for row in aligned_rows
    )
    available = sum(
        round(result["held_sample_count"] * result["held_available_fraction"])
        for result in results
    )
    return {
        "family_key": family,
        "classes": classes,
        "modality": modality,
        "selected_folds": sorted(int(result["outer_fold"]) for result in results),
        "held_rows": len(aligned_rows),
        "held_available_rows": int(available),
        "a_correct": int(a_correct),
        "aligned": aligned_metrics,
        "rescue": int(rescue),
        "harm": int(harm),
        "net": int(rescue - harm),
        "aligned_minus_a_pp": float(
            100.0 * (aligned_metrics["correct"] - a_correct) / len(aligned_rows)
        ),
        "aligned_minus_shuffle_correct": int(aligned_metrics["correct"] - shuffle_correct),
        "aligned_minus_zero_correct": int(aligned_metrics["correct"] - zero_correct),
        "deployable_trigger_net": int(
            sum(result["deployable_trigger"]["net"] for result in results)
        ),
        "mean_source_vs_held_balanced_accuracy_gap": float(
            np.mean([result["source_vs_held_balanced_accuracy_gap"] for result in results])
        ),
        "per_subject": subject_stability(aligned_rows),
    }


def aggregate_extended_selection(
    plans: list[dict[str, Any]],
    result_lookup: dict[tuple[int, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for plan in plans:
        fold = int(plan["outer_fold"])
        family = plan["family_key"]
        core_modality = plan["best_single"]["modality"]
        core_result = result_lookup[(fold, family, core_modality)]
        thermal_result = result_lookup[(fold, family, "Thermal")]
        if source_metric_key(thermal_result["source_inner_specialist"]) > source_metric_key(
            core_result["source_inner_specialist"]
        ):
            grouped[family].append(("Thermal", thermal_result))
        else:
            grouped[family].append((core_modality, core_result))
    output = []
    for family, values in sorted(grouped.items()):
        held = sum(result["held_sample_count"] for _, result in values)
        a_correct = sum(result["a_family"]["correct"] for _, result in values)
        aligned = sum(
            result["variants"]["aligned"]["metrics"]["correct"] for _, result in values
        )
        shuffle = sum(
            result["variants"]["shuffle"]["metrics"]["correct"] for _, result in values
        )
        zero = sum(
            result["variants"]["zero"]["metrics"]["correct"] for _, result in values
        )
        choices = {
            str(int(result["outer_fold"])): modality for modality, result in values
        }
        output.append(
            {
                "family_key": family,
                "classes": values[0][1]["classes"],
                "source_selected_modality_by_fold": choices,
                "source_selected_modality_counts": dict(sorted(Counter(choices.values()).items())),
                "held_rows": held,
                "a_correct": a_correct,
                "aligned_correct": aligned,
                "net": aligned - a_correct,
                "aligned_minus_shuffle_correct": aligned - shuffle,
                "aligned_minus_zero_correct": aligned - zero,
                "deployable_trigger_net": sum(
                    result["deployable_trigger"]["net"] for _, result in values
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    paths = {
        "GlobalV": args.globalv_run.resolve(),
        "LocalV": args.localv_run.resolve(),
        "Skeleton": args.skeleton_run.resolve(),
        "IMU": args.imu_run.resolve(),
        "Depth": args.depth_run.resolve(),
        "Thermal": args.thermal_run.resolve(),
    }
    core_final = json.loads(args.core_final.resolve().read_text(encoding="utf-8"))
    stable = {
        value["family_key"]: list(map(int, value["classes"]))
        for value in core_final["single_decision_table"]
    }
    result_lookup: dict[tuple[int, str, str], dict[str, Any]] = {}
    matrix: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for modality, path in paths.items():
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "complete":
            raise RuntimeError(f"P104 modality is incomplete: {modality}")
        if summary["data"]["h3_rows_selected"] != 0 or summary["data"]["h3_users_loaded"]:
            raise RuntimeError(f"H3 reached P104 Thermal extension: {modality}")
        audit[modality] = summary["modality_audit"][modality]
        grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in summary["fold_results"]:
            result_lookup[(int(result["outer_fold"]), result["family_key"], modality)] = result
            if result["family_key"] in stable:
                grouped_results[result["family_key"]].append(result)
        predictions = load_outer_predictions(path / "single_predictions.csv")
        for family, classes in stable.items():
            matrix.append(
                modality_matrix_row(
                    modality,
                    family,
                    classes,
                    grouped_results[family],
                    predictions,
                )
            )
    plan = json.loads(args.core_plan.resolve().read_text(encoding="utf-8"))
    extended = aggregate_extended_selection(plan["plans"], result_lookup)
    thermal = [value for value in matrix if value["modality"] == "Thermal"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete",
        "protocol": "Thermal added after core family/pair freeze; six-way single-modality map, no fusion/pair reselection",
        "modalities": list(paths),
        "stable_family_count": len(stable),
        "confusion_modality_matrix": sorted(
            matrix, key=lambda value: (value["family_key"], list(paths).index(value["modality"]))
        ),
        "thermal_table": sorted(thermal, key=lambda value: value["family_key"]),
        "extended_source_selected_table": extended,
        "knowledge_map": KNOWLEDGE_MAP,
        "thermal_conclusion": {
            "new_specialist_authorized": False,
            "physical_evidence_candidate": "21__22",
            "reason": "Thermal has a positive family/correspondence result for 21__22, but its label-free trigger is net-negative; all other stable families are negative, weak, or counterfactual-inconsistent.",
            "core_shortlist_changed": False,
            "pair_plan_reopened": False,
        },
        "modality_audit": audit,
        "h3_rows_selected": 0,
        "h3_users_loaded": [],
        "student_started": False,
        "final_b_teacher_trained": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "thermal_table": [
                    {
                        "family_key": value["family_key"],
                        "held": value["held_rows"],
                        "a_correct": value["a_correct"],
                        "aligned_correct": value["aligned"]["correct"],
                        "balanced_accuracy": value["aligned"]["balanced_accuracy"],
                        "macro_f1": value["aligned"]["macro_f1"],
                        "net": value["net"],
                        "aligned_minus_shuffle": value["aligned_minus_shuffle_correct"],
                        "aligned_minus_zero": value["aligned_minus_zero_correct"],
                        "trigger_net": value["deployable_trigger_net"],
                    }
                    for value in thermal
                ],
                "extended_source_selected_table": extended,
                "thermal_conclusion": summary["thermal_conclusion"],
                "h3_rows_selected": 0,
                "h3_users_loaded": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
