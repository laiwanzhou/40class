"""Build the final P104 confusion-to-evidence decision tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from p104_modality_data import MODALITIES


HERE = Path(__file__).resolve().parent
DEFAULT_SINGLE_RUNS = {
    "GlobalV": HERE / "runs/p104_globalv_specialists_oof_v1",
    "LocalV": HERE / "runs/p104_localv_specialists_oof_v1",
    "Skeleton": HERE / "runs/p104_skeleton_specialists_oof_v1",
    "IMU": HERE / "runs/p104_imu_specialists_oof_v1",
    "Depth": HERE / "runs/p104_depth_specialists_oof_v1",
}
DEFAULT_PLAN = HERE / "runs/p104_single_modality_audit_v1/pair_plan.json"
DEFAULT_PAIRS = HERE / "runs/p104_pair_specialists_oof_v1/summary.json"
DEFAULT_OUTPUT = HERE / "runs/p104_final_audit_v1"


FINAL_VERDICTS = {
    "21__22": "NO_CURRENT_MODALITY_EVIDENCE",
    "24__26": "NO_CURRENT_MODALITY_EVIDENCE",
    "24__27": "NO_CURRENT_MODALITY_EVIDENCE",
    "32__34": "NO_CURRENT_MODALITY_EVIDENCE",
    "38__39": "INSUFFICIENT_SAMPLE",
    "3__5": "LOCALV_PLUS_S_PROMISING",
    "6__37": "NO_CURRENT_MODALITY_EVIDENCE",
    "7__37": "LOCAL_VISUAL_SPECIALIST_PROMISING",
    "7__8": "GLOBAL_VISUAL_SPECIALIST_PROMISING",
    "8__10": "NO_CURRENT_MODALITY_EVIDENCE",
    "8__9": "LOCAL_VISUAL_SPECIALIST_PROMISING",
}

VERDICT_RATIONALE = {
    "21__22": "source-selected single and the repeated LocalV+Skeleton pair are both net negative",
    "24__26": "held gains do not survive the zero/shuffle mechanism test; LocalV+IMU is driven by one fold and zero-both is better",
    "24__27": "source-selected GlobalV and both exploratory pairs are held-negative",
    "32__34": "source-selected single is held-negative; pair gain is small, trigger-negative, and IMU correspondence is absent",
    "38__39": "visual gains are positive but only two family folds choose different best modalities and no exact pair repeats",
    "3__5": "repeated LocalV+Skeleton is positive in both folds, aligned beats shuffle/zero, rescue has no harm, and the label-free trigger is positive",
    "6__37": "LocalV evidence is small and fold-unstable; LocalV+IMU is trigger-negative and reverses sign by fold",
    "7__37": "LocalV is source-selected in both folds with positive aligned/shuffle/zero margins and nonnegative trigger; Depth adds no stable paired correspondence",
    "7__8": "GlobalV is source-selected in both folds with positive net, correspondence margin, and label-free trigger",
    "8__10": "source-selected LocalV ties shuffle and has a negative trigger; held-strong Depth was not source-selected",
    "8__9": "LocalV is source-selected in both folds with positive net and the largest correspondence margin; trigger is neutral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in DEFAULT_SINGLE_RUNS.items():
        parser.add_argument(f"--{name.lower()}-run", type=Path, default=path)
    parser.add_argument("--pair-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--pair-summary", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def correct_from_subject(record: dict[str, Any]) -> int:
    return int(round(int(record["rows"]) * float(record["accuracy"])))


def add_subject_interventions(
    target: dict[str, dict[str, int]], result: dict[str, Any]
) -> None:
    a_subjects = result["a_family"]["per_subject"]
    specialist_subjects = result["variants"]["aligned"]["metrics"]["per_subject"]
    if set(a_subjects) != set(specialist_subjects):
        raise RuntimeError("P104 per-subject coverage differs")
    for subject in a_subjects:
        rows = int(a_subjects[subject]["rows"])
        if rows != int(specialist_subjects[subject]["rows"]):
            raise RuntimeError("P104 per-subject row counts differ")
        target[subject]["rows"] += rows
        target[subject]["a_correct"] += correct_from_subject(a_subjects[subject])
        target[subject]["specialist_correct"] += correct_from_subject(
            specialist_subjects[subject]
        )


def finish_subjects(values: dict[str, dict[str, int]]) -> dict[str, Any]:
    records = []
    for subject, record in sorted(values.items()):
        entry = {
            "subject": subject,
            **record,
            "net": record["specialist_correct"] - record["a_correct"],
        }
        records.append(entry)
    return {
        "records": records,
        "positive_subjects": sum(value["net"] > 0 for value in records),
        "neutral_subjects": sum(value["net"] == 0 for value in records),
        "negative_subjects": sum(value["net"] < 0 for value in records),
        "worst_subject": min(records, key=lambda value: (value["net"], value["subject"])),
    }


def load_single_results(paths: dict[str, Path]) -> dict[tuple[int, str, str], dict[str, Any]]:
    lookup: dict[tuple[int, str, str], dict[str, Any]] = {}
    for modality, path in paths.items():
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "complete":
            raise RuntimeError(f"P104 single run incomplete: {modality}")
        if summary["data"]["h3_rows_selected"] != 0 or summary["data"]["h3_users_loaded"]:
            raise RuntimeError(f"H3 reached P104 final single audit: {modality}")
        for result in summary["fold_results"]:
            key = (int(result["outer_fold"]), result["family_key"], modality)
            lookup[key] = result
    return lookup


def selected_single_table(
    plan_archive: dict[str, Any],
    single_lookup: dict[tuple[int, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for plan in plan_archive["plans"]:
        modality = plan["best_single"]["modality"]
        result = single_lookup[(int(plan["outer_fold"]), plan["family_key"], modality)]
        grouped[plan["family_key"]].append((plan, result))
    output: list[dict[str, Any]] = []
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
        rescue = sum(
            result["variants"]["aligned"]["intervention"]["rescue"] for _, result in values
        )
        harm = sum(
            result["variants"]["aligned"]["intervention"]["harm"] for _, result in values
        )
        subjects: dict[str, dict[str, int]] = defaultdict(
            lambda: {"rows": 0, "a_correct": 0, "specialist_correct": 0}
        )
        for _, result in values:
            add_subject_interventions(subjects, result)
        gaps = [result["source_vs_held_balanced_accuracy_gap"] for _, result in values]
        choices = {
            str(int(plan["outer_fold"])): plan["best_single"]["modality"]
            for plan, _ in values
        }
        output.append(
            {
                "family_key": family,
                "classes": values[0][0]["classes"],
                "selected_folds": sorted(int(plan["outer_fold"]) for plan, _ in values),
                "source_selected_modality_by_fold": choices,
                "source_selected_modality_counts": dict(sorted(Counter(choices.values()).items())),
                "held_rows": held,
                "a_correct": a_correct,
                "aligned_correct": aligned,
                "aligned_minus_a_correct": aligned - a_correct,
                "aligned_minus_a_pp": float(100.0 * (aligned - a_correct) / held),
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "aligned_minus_shuffle_correct": aligned - shuffle,
                "aligned_minus_zero_correct": aligned - zero,
                "deployable_trigger_net": sum(
                    result["deployable_trigger"]["net"] for _, result in values
                ),
                "mean_source_vs_held_balanced_accuracy_gap": float(np.mean(gaps)),
                "per_fold_net": {
                    str(int(plan["outer_fold"])): result["variants"]["aligned"]["intervention"]["net"]
                    for plan, result in values
                },
                "per_subject": finish_subjects(subjects),
                "verdict": FINAL_VERDICTS[family],
                "verdict_rationale": VERDICT_RATIONALE[family],
            }
        )
    if set(FINAL_VERDICTS) != {value["family_key"] for value in output}:
        raise RuntimeError("P104 final verdict family coverage changed")
    return output


def pair_table(
    pair_summary: dict[str, Any],
    plan_archive: dict[str, Any],
    single_lookup: dict[tuple[int, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    plan_lookup = {
        (int(value["outer_fold"]), value["family_key"]): value
        for value in plan_archive["plans"]
    }
    result_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in pair_summary["fold_results"]:
        result_groups[(result["family_key"], result["pair_key"])].append(result)
    output: list[dict[str, Any]] = []
    for aggregate in pair_summary["formal_exact_pair_aggregates"]:
        family = aggregate["family_key"]
        key = aggregate["pair_key"]
        results = result_groups[(family, key)]
        subjects: dict[str, dict[str, int]] = defaultdict(
            lambda: {"rows": 0, "a_correct": 0, "specialist_correct": 0}
        )
        selected_single_correct = 0
        for result in results:
            add_subject_interventions(subjects, result)
            plan = plan_lookup[(int(result["outer_fold"]), family)]
            modality = plan["best_single"]["modality"]
            selected_single = single_lookup[(int(result["outer_fold"]), family, modality)]
            selected_single_correct += selected_single["variants"]["aligned"]["metrics"]["correct"]
        variants = aggregate["variant_correct"]
        first, second = aggregate["modalities"]
        promising = family == "3__5" and key == "LocalV+Skeleton"
        output.append(
            {
                **aggregate,
                "selected_best_single_correct_same_folds": selected_single_correct,
                "pair_minus_selected_best_single_correct": aggregate["aligned_correct"]
                - selected_single_correct,
                "first_modality": first,
                "second_modality": second,
                "aligned_minus_shuffle_first_correct": aggregate["aligned_correct"]
                - variants["shuffle_first"],
                "aligned_minus_shuffle_second_correct": aggregate["aligned_correct"]
                - variants["shuffle_second"],
                "aligned_minus_zero_first_correct": aggregate["aligned_correct"]
                - variants["zero_first"],
                "aligned_minus_zero_second_correct": aggregate["aligned_correct"]
                - variants["zero_second"],
                "per_subject": finish_subjects(subjects),
                "verdict": (
                    "LOCALV_PLUS_S_PROMISING"
                    if promising
                    else "NO_CURRENT_MODALITY_EVIDENCE"
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "GlobalV": args.globalv_run.resolve(),
        "LocalV": args.localv_run.resolve(),
        "Skeleton": args.skeleton_run.resolve(),
        "IMU": args.imu_run.resolve(),
        "Depth": args.depth_run.resolve(),
    }
    if set(paths) != set(MODALITIES):
        raise RuntimeError("P104 final modality coverage changed")
    plan = json.loads(args.pair_plan.resolve().read_text(encoding="utf-8"))
    pairs = json.loads(args.pair_summary.resolve().read_text(encoding="utf-8"))
    if pairs["status"] != "complete":
        raise RuntimeError("P104 formal pairs are incomplete")
    if pairs["data"]["h3_rows_selected"] != 0 or pairs["data"]["h3_users_loaded"]:
        raise RuntimeError("H3 reached P104 final pair audit")
    singles = load_single_results(paths)
    single_table = selected_single_table(plan, singles)
    formal_pairs = pair_table(pairs, plan, singles)
    summary = {
        "status": "complete",
        "protocol": "P104 held labels used only after source-only family/modality/pair freeze",
        "single_decision_table": single_table,
        "formal_pair_decision_table": formal_pairs,
        "shortlist": [
            {
                "family_key": value["family_key"],
                "verdict": value["verdict"],
                "rationale": value["verdict_rationale"],
            }
            for value in single_table
            if value["verdict"].endswith("PROMISING")
        ],
        "no_current_modality_evidence": [
            value["family_key"] for value in single_table
            if value["verdict"] == "NO_CURRENT_MODALITY_EVIDENCE"
        ],
        "insufficient_sample": [
            value["family_key"] for value in single_table
            if value["verdict"] == "INSUFFICIENT_SAMPLE"
        ],
        "specialist_bank_sketch": {
            "gate": "A top-1 is a family member and another family member appears in A top-3",
            "entries": [
                {"family_key": "3__5", "evidence": "LocalV+Skeleton", "status": "research shortlist"},
                {"family_key": "7__37", "evidence": "LocalV", "status": "research shortlist"},
                {"family_key": "7__8", "evidence": "GlobalV", "status": "research shortlist"},
                {"family_key": "8__9", "evidence": "LocalV", "status": "research shortlist"},
            ],
            "not_implemented": ["final B teacher", "learned router", "student", "distillation"],
        },
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
                "shortlist": summary["shortlist"],
                "no_current_modality_evidence": summary["no_current_modality_evidence"],
                "insufficient_sample": summary["insufficient_sample"],
                "formal_pairs": [
                    {
                        "family_key": value["family_key"],
                        "pair_key": value["pair_key"],
                        "net": value["net"],
                        "pair_minus_single": value["pair_minus_selected_best_single_correct"],
                        "trigger_net": value["deployable_trigger_net"],
                        "verdict": value["verdict"],
                    }
                    for value in formal_pairs
                ],
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
