"""Combine P104 single-modality runs and freeze source-only pair eligibility."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from p104_modality_data import MODALITIES


HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = {
    "GlobalV": HERE / "runs/p104_globalv_specialists_oof_v1",
    "LocalV": HERE / "runs/p104_localv_specialists_oof_v1",
    "Skeleton": HERE / "runs/p104_skeleton_specialists_oof_v1",
    "IMU": HERE / "runs/p104_imu_specialists_oof_v1",
    "Depth": HERE / "runs/p104_depth_specialists_oof_v1",
}
DEFAULT_FAMILIES = HERE / "runs/p104_confusion_atlas_v1/fold_families.json"
DEFAULT_OUTPUT = HERE / "runs/p104_single_modality_audit_v1"
PAIR_CANDIDATES = (
    ("Skeleton", "IMU", "S_PLUS_I"),
    ("LocalV", "Depth", "LOCALV_PLUS_DEPTH"),
    ("LocalV", "Skeleton", "LOCALV_PLUS_S"),
    ("LocalV", "IMU", "LOCALV_PLUS_I"),
)
MODALITY_ORDER = {name: index for index, name in enumerate(MODALITIES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in DEFAULT_RUNS.items():
        parser.add_argument(f"--{name.lower()}-run", type=Path, default=path)
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[tuple[int, str, int], tuple[int, int]]:
    output: dict[tuple[int, str, int], tuple[int, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "source_inner_oof" or row["variant"] != "aligned":
                continue
            key = (int(row["outer_fold"]), row["family_key"], int(row["row_index"]))
            value = (int(row["label"]), int(row["specialist_prediction"]))
            if key in output and output[key] != value:
                raise RuntimeError(f"conflicting P104 source prediction: {key}")
            output[key] = value
    return output


def source_score(result: dict[str, Any]) -> tuple[float, float, float, int]:
    metrics = result["source_inner_specialist"]
    return (
        float(metrics["balanced_accuracy"]),
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
        -MODALITY_ORDER[result["modality"]],
    )


def pair_complementarity(
    first: dict[tuple[int, str, int], tuple[int, int]],
    second: dict[tuple[int, str, int], tuple[int, int]],
    outer_fold: int,
    family_key: str,
    best_single_accuracy: float,
) -> dict[str, Any]:
    first_rows = {
        row: value for (fold, family, row), value in first.items()
        if fold == outer_fold and family == family_key
    }
    second_rows = {
        row: value for (fold, family, row), value in second.items()
        if fold == outer_fold and family == family_key
    }
    if set(first_rows) != set(second_rows) or not first_rows:
        raise RuntimeError(f"P104 pair source coverage differs: fold={outer_fold} family={family_key}")
    rows = sorted(first_rows)
    labels = np.asarray([first_rows[row][0] for row in rows], dtype=np.int64)
    if not np.array_equal(labels, np.asarray([second_rows[row][0] for row in rows])):
        raise RuntimeError("P104 pair source labels differ")
    first_correct = np.asarray(
        [first_rows[row][1] for row in rows], dtype=np.int64
    ) == labels
    second_correct = np.asarray(
        [second_rows[row][1] for row in rows], dtype=np.int64
    ) == labels
    union = first_correct | second_correct
    gain = float(np.mean(union) - best_single_accuracy)
    unique_first = int(np.sum(first_correct & (~second_correct)))
    unique_second = int(np.sum(second_correct & (~first_correct)))
    return {
        "rows": len(rows),
        "first_correct": int(first_correct.sum()),
        "second_correct": int(second_correct.sum()),
        "unique_first": unique_first,
        "unique_second": unique_second,
        "oracle_union_correct": int(union.sum()),
        "oracle_union_accuracy": float(np.mean(union)),
        "oracle_union_gain_over_best_single_pp": float(100.0 * gain),
        "eligible": unique_first >= 2 and unique_second >= 2 and gain >= 0.03,
    }


def aggregate_single_results(
    fold_results: list[dict[str, Any]], stable: set[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in fold_results:
        if result["family_key"] in stable:
            grouped[(result["family_key"], result["modality"])].append(result)
    output: list[dict[str, Any]] = []
    for (family, modality), values in sorted(grouped.items()):
        held = sum(value["held_sample_count"] for value in values)
        a_correct = sum(value["a_family"]["correct"] for value in values)
        aligned = sum(value["variants"]["aligned"]["metrics"]["correct"] for value in values)
        rescue = sum(value["variants"]["aligned"]["intervention"]["rescue"] for value in values)
        harm = sum(value["variants"]["aligned"]["intervention"]["harm"] for value in values)
        variants = {
            name: sum(value["variants"][name]["metrics"]["correct"] for value in values)
            for name in values[0]["variants"]
        }
        output.append(
            {
                "family_key": family,
                "classes": values[0]["classes"],
                "modality": modality,
                "selected_folds": [value["outer_fold"] for value in values],
                "held_rows": held,
                "a_accuracy": float(a_correct / held),
                "specialist_accuracy": float(aligned / held),
                "specialist_minus_a_pp": float(100.0 * (aligned - a_correct) / held),
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "aligned_correct": aligned,
                "shuffle_correct": variants["shuffle"],
                "zero_correct": variants["zero"],
                "aligned_minus_shuffle": aligned - variants["shuffle"],
                "aligned_minus_zero": aligned - variants["zero"],
                "deployable_trigger_net": sum(
                    value["deployable_trigger"]["net"] for value in values
                ),
                "per_fold_net": {
                    str(value["outer_fold"]): value["variants"]["aligned"]["intervention"]["net"]
                    for value in values
                },
            }
        )
    return output


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_paths = {
        "GlobalV": args.globalv_run.resolve(),
        "LocalV": args.localv_run.resolve(),
        "Skeleton": args.skeleton_run.resolve(),
        "IMU": args.imu_run.resolve(),
        "Depth": args.depth_run.resolve(),
    }
    summaries: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[tuple[int, str, int], tuple[int, int]]] = {}
    result_lookup: dict[tuple[int, str, str], dict[str, Any]] = {}
    fold_results: list[dict[str, Any]] = []
    for modality, path in run_paths.items():
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "complete":
            raise RuntimeError(f"P104 {modality} singles are incomplete")
        if summary["data"]["h3_rows_selected"] != 0 or summary["data"]["h3_users_loaded"]:
            raise RuntimeError(f"H3 reached P104 {modality}")
        if summary["data"]["modalities_run"] != [modality]:
            raise RuntimeError(f"P104 modality run identity changed: {modality}")
        summaries[modality] = summary
        predictions[modality] = load_predictions(path / "single_predictions.csv")
        for result in summary["fold_results"]:
            key = (int(result["outer_fold"]), str(result["family_key"]), modality)
            if key in result_lookup:
                raise RuntimeError(f"duplicate P104 result: {key}")
            result_lookup[key] = result
            fold_results.append(result)

    family_archive = json.loads(args.families.resolve().read_text(encoding="utf-8"))
    stable_records = [
        value for value in family_archive["stable_families"]
        if value["formal_verdict_eligible"]
    ]
    stable = {value["family_key"] for value in stable_records}
    plan: list[dict[str, Any]] = []
    selected_single_results: list[dict[str, Any]] = []
    for stable_family in stable_records:
        key = stable_family["family_key"]
        for outer_fold in stable_family["selected_folds"]:
            singles = [
                result_lookup[(int(outer_fold), key, modality)] for modality in MODALITIES
            ]
            ranked = sorted(singles, key=source_score, reverse=True)
            best = ranked[0]
            best_accuracy = float(best["source_inner_specialist"]["accuracy"])
            pair_records: list[dict[str, Any]] = []
            for first, second, name in PAIR_CANDIDATES:
                audit = pair_complementarity(
                    predictions[first],
                    predictions[second],
                    int(outer_fold),
                    key,
                    best_accuracy,
                )
                pair_records.append(
                    {"name": name, "modalities": [first, second], **audit}
                )
            eligible = [value for value in pair_records if value["eligible"]]
            eligible.sort(
                key=lambda value: (
                    -value["oracle_union_gain_over_best_single_pp"],
                    -min(value["unique_first"], value["unique_second"]),
                    value["name"],
                )
            )
            selected_pair = eligible[0] if eligible else None
            plan.append(
                {
                    "outer_fold": int(outer_fold),
                    "family_key": key,
                    "classes": stable_family["classes"],
                    "best_single": {
                        "modality": best["modality"],
                        "source_inner_metrics": best["source_inner_specialist"],
                    },
                    "single_source_ranking": [
                        {
                            "modality": value["modality"],
                            "balanced_accuracy": value["source_inner_specialist"]["balanced_accuracy"],
                            "macro_f1": value["source_inner_specialist"]["macro_f1"],
                            "accuracy": value["source_inner_specialist"]["accuracy"],
                        }
                        for value in ranked
                    ],
                    "pair_candidates": pair_records,
                    "selected_pair": selected_pair,
                }
            )
            selected_single_results.append(best)

    pair_plan = {
        "status": "complete",
        "protocol": "P104 pair eligibility and best-single selection use source inner-subject OOF only",
        "eligibility": {
            "allowed_pairs": [list(value) for value in PAIR_CANDIDATES],
            "unique_corrections_each_at_least": 2,
            "oracle_union_gain_over_best_single_pp_at_least": 3.0,
            "max_pairs_per_outer_family": 1,
        },
        "plans": plan,
        "selected_pair_count": int(sum(value["selected_pair"] is not None for value in plan)),
        "h3_rows_selected": 0,
        "h3_users_loaded": [],
    }
    (output / "pair_plan.json").write_text(
        json.dumps(pair_plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    single_table = aggregate_single_results(fold_results, stable)
    selected_aggregate = aggregate_single_results(selected_single_results, stable)
    summary = {
        "status": "complete",
        "protocol": pair_plan["protocol"],
        "stable_family_count": len(stable),
        "outer_family_evaluations": len(plan),
        "selected_pair_count": pair_plan["selected_pair_count"],
        "single_modality_table": single_table,
        "source_selected_best_single_table": selected_aggregate,
        "pair_plan": "pair_plan.json",
        "h3_rows_selected": 0,
        "h3_users_loaded": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
