"""Merge existing P105/P106 OOF predictions without fitting any new model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_p102_session_closure import load_npz
from p100a_global_teacher_data import H3_USERS
from train_p104_modality_specialists_oof import DEFAULT_SESSION


HERE = Path(__file__).resolve().parent
DEFAULT_P105 = HERE / "runs/p105_specialist_bank_oof_v1"
DEFAULT_P106_ROUTING = HERE / "runs/p106_source_safe_routing_v1"
DEFAULT_P106_FORENSICS = HERE / "runs/p106_hard_confusion_forensics_v1"
DEFAULT_OUTPUT = HERE / "runs/p107_expanded_oracle_bank_audit_v1"

SPECIALISTS: tuple[dict[str, Any], ...] = (
    {
        "family_key": "3__5",
        "classes": [3, 5],
        "provenance": "P105",
        "evidence": "LocalV+Skeleton",
    },
    {
        "family_key": "7__37",
        "classes": [7, 37],
        "provenance": "P105",
        "evidence": "LocalV",
    },
    {
        "family_key": "7__8",
        "classes": [7, 8],
        "provenance": "P105",
        "evidence": "GlobalV",
    },
    {
        "family_key": "8__9",
        "classes": [8, 9],
        "provenance": "P105",
        "evidence": "LocalV",
    },
    {
        "family_key": "24__26",
        "classes": [24, 26],
        "provenance": "P106",
        "evidence": "local_interaction_temporal",
    },
    {
        "family_key": "32__34",
        "classes": [32, 34],
        "provenance": "P106",
        "evidence": "global_visual_phase_delta",
    },
    {
        "family_key": "6__37",
        "classes": [6, 37],
        "provenance": "P106",
        "evidence": "local_interaction_temporal",
    },
    {
        "family_key": "24__27",
        "classes": [24, 27],
        "provenance": "P106",
        "evidence": "skeleton_wrist_arm_temporal",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--p105-root", type=Path, default=DEFAULT_P105)
    parser.add_argument("--p106-routing-root", type=Path, default=DEFAULT_P106_ROUTING)
    parser.add_argument("--p106-forensics-root", type=Path, default=DEFAULT_P106_FORENSICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def family_masks(labels: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.isin(labels, config["classes"]) for config in SPECIALISTS]
    ).astype(bool)


def topk_structure(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(probability), axis=1)[:, ::-1]
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None] + 1
    return order, ranks


def family_triggers(
    top_order: np.ndarray, ranks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    top1 = top_order[:, 0]
    top3 = np.zeros((len(top_order), len(SPECIALISTS)), dtype=bool)
    top5 = np.zeros_like(top3)
    for index, config in enumerate(SPECIALISTS):
        left, right = config["classes"]
        top3[:, index] = (
            np.isin(top1, config["classes"])
            & (ranks[:, left] <= 3)
            & (ranks[:, right] <= 3)
        )
        top5[:, index] = (ranks[:, left] <= 5) & (ranks[:, right] <= 5)
    return top3, top5


def simulate_oracles(
    labels: np.ndarray,
    a_prediction: np.ndarray,
    eligible: np.ndarray,
    specialist_prediction: np.ndarray,
) -> dict[str, Any]:
    rows, experts = eligible.shape
    if specialist_prediction.shape != (rows, experts):
        raise ValueError("specialist prediction shape differs")
    a_correct = a_prediction == labels
    expert_correct = eligible & (specialist_prediction == labels[:, None])

    union_rescue = (~a_correct) & expert_correct.any(axis=1)
    union_selected = np.full(rows, -1, dtype=np.int64)
    union_selected[union_rescue] = np.argmax(expert_correct[union_rescue], axis=1)
    union_prediction = a_prediction.copy()
    union_prediction[union_rescue] = labels[union_rescue]

    eligible_any = eligible.any(axis=1)
    correct_any = expert_correct.any(axis=1)
    forced_selected = np.full(rows, -1, dtype=np.int64)
    forced_selected[eligible_any & correct_any] = np.argmax(
        expert_correct[eligible_any & correct_any], axis=1
    )
    no_correct = eligible_any & (~correct_any)
    forced_selected[no_correct] = np.argmax(eligible[no_correct], axis=1)
    forced_prediction = a_prediction.copy()
    selected_rows = np.flatnonzero(eligible_any)
    forced_prediction[selected_rows] = specialist_prediction[
        selected_rows, forced_selected[selected_rows]
    ]

    exact_selected = np.full(rows, -1, dtype=np.int64)
    for index, config in enumerate(SPECIALISTS):
        left, right = config["classes"]
        exact = (
            (~a_correct)
            & eligible[:, index]
            & (((labels == left) & (a_prediction == right)) | ((labels == right) & (a_prediction == left)))
        )
        if np.any((exact_selected >= 0) & exact):
            raise RuntimeError("expanded exact-edge oracle is ambiguous")
        exact_selected[exact] = index
    exact_prediction = a_prediction.copy()
    exact_rows = np.flatnonzero(exact_selected >= 0)
    exact_prediction[exact_rows] = specialist_prediction[
        exact_rows, exact_selected[exact_rows]
    ]

    return {
        "expert_correct": expert_correct,
        "union": {
            "prediction": union_prediction,
            "selected": union_selected,
        },
        "forced": {
            "prediction": forced_prediction,
            "selected": forced_selected,
        },
        "exact": {
            "prediction": exact_prediction,
            "selected": exact_selected,
        },
    }


def system_metrics(
    labels: np.ndarray, a_prediction: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    a_correct = a_prediction == labels
    correct = prediction == labels
    rescue = int(np.sum((~a_correct) & correct))
    harm = int(np.sum(a_correct & (~correct)))
    return {
        "rows": len(labels),
        "a_correct": int(a_correct.sum()),
        "correct": int(correct.sum()),
        "a_accuracy": float(a_correct.mean()),
        "accuracy": float(correct.mean()),
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
    }


def _ordered_frame(path: Path, rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != rows or set(frame["row_index"].astype(int)) != set(range(rows)):
        raise RuntimeError(f"invalid full-row audit CSV: {path}")
    return frame.sort_values("row_index").reset_index(drop=True)


def load_predictions(
    p105_root: Path,
    p106_root: Path,
    labels: np.ndarray,
    sample_ids: np.ndarray,
) -> np.ndarray:
    rows = len(labels)
    output = np.full((rows, len(SPECIALISTS)), -1, dtype=np.int64)
    p105 = pd.read_csv(p105_root / "specialist_predictions.csv")
    p105 = p105[(p105["split"] == "outer_held_all") & (p105["variant"] == "aligned")]
    p106 = pd.read_csv(p106_root / "predictions.csv")
    p106 = p106[p106["variant"] == "aligned"]
    for index, config in enumerate(SPECIALISTS):
        key = config["family_key"]
        if config["provenance"] == "P105":
            selected = p105[p105["family_key"] == key].sort_values("row_index")
            expected_rows = np.arange(rows)
        else:
            selected = p106[
                (p106["family_key"] == key) & (p106["block"] == config["evidence"])
            ].sort_values("row_index")
            expected_rows = np.flatnonzero(np.isin(labels, config["classes"]))
        observed_rows = selected["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed_rows, expected_rows):
            raise RuntimeError(f"OOF coverage changed for {key}")
        if not np.array_equal(
            selected["sample_id"].astype(str).to_numpy(), sample_ids[observed_rows]
        ):
            raise RuntimeError(f"sample alignment changed for {key}")
        output[observed_rows, index] = selected["specialist_prediction"].to_numpy(
            dtype=np.int64
        )
    eligible = family_masks(labels)
    if np.any(output[eligible] < 0):
        raise RuntimeError("eligible P107 specialist prediction is missing")
    return output


def join_values(values: list[Any]) -> str:
    return "|".join(map(str, values))


def matrix_rows(matrix: np.ndarray, value_name: str) -> list[dict[str, Any]]:
    rows = []
    for left, left_config in enumerate(SPECIALISTS):
        for right, right_config in enumerate(SPECIALISTS):
            rows.append(
                {
                    "left_family": left_config["family_key"],
                    "right_family": right_config["family_key"],
                    value_name: int(matrix[left, right]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    family_dir = output / "families"
    family_dir.mkdir(parents=True, exist_ok=True)
    session = load_npz(args.session.resolve())
    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    fold_ids = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    a_prediction = np.asarray(session["selected_prediction"], dtype=np.int64)
    rows = len(labels)
    if str(np.asarray(session["selected_system"]).item()) != "VS_session":
        raise RuntimeError("P107 requires frozen A = VS + Session")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached P107")
    if not np.array_equal(a_prediction, a_probability.argmax(axis=1)):
        raise RuntimeError("A prediction/probability changed")

    specialist_prediction = load_predictions(
        args.p105_root.resolve(),
        args.p106_forensics_root.resolve(),
        labels,
        sample_ids,
    )
    eligible = family_masks(labels)
    top_order, ranks = topk_structure(a_probability)
    top3_trigger, both_top5 = family_triggers(top_order, ranks)
    simulations = simulate_oracles(labels, a_prediction, eligible, specialist_prediction)
    expert_correct = simulations["expert_correct"]
    a_correct = a_prediction == labels
    rescue_matrix = eligible & (~a_correct[:, None]) & expert_correct
    harm_matrix = eligible & a_correct[:, None] & (~expert_correct)

    p105_routes = _ordered_frame(
        args.p105_root.resolve() / "routing_predictions.csv", rows
    )
    current_route = p105_routes["source_authorized_route"].fillna("").astype(str).to_numpy()
    current_prediction = p105_routes["source_authorized_prediction"].to_numpy(dtype=np.int64)
    rejected = _ordered_frame(
        args.p106_routing_root.resolve() / "routing_predictions.csv", rows
    )
    rejected_route = rejected["route"].fillna("").astype(str).to_numpy()
    rejected_prediction = rejected["system_prediction"].to_numpy(dtype=np.int64)
    if not np.array_equal(p105_routes["sample_id"].astype(str).to_numpy(), sample_ids):
        raise RuntimeError("P105 routing sample alignment changed")
    if not np.array_equal(rejected["sample_id"].astype(str).to_numpy(), sample_ids):
        raise RuntimeError("P106 routing sample alignment changed")

    union = simulations["union"]
    forced = simulations["forced"]
    exact = simulations["exact"]
    union_metrics = system_metrics(labels, a_prediction, union["prediction"])
    forced_metrics = system_metrics(labels, a_prediction, forced["prediction"])
    exact_metrics = system_metrics(labels, a_prediction, exact["prediction"])
    current_metrics = system_metrics(labels, a_prediction, current_prediction)
    rejected_metrics = system_metrics(labels, a_prediction, rejected_prediction)

    correct_expert_count = expert_correct.sum(axis=1)
    eligible_count = eligible.sum(axis=1)
    individual: list[dict[str, Any]] = []
    for index, config in enumerate(SPECIALISTS):
        rescue = rescue_matrix[:, index]
        harm = harm_matrix[:, index]
        other_correct = np.any(
            np.delete(rescue_matrix, index, axis=1), axis=1
        )
        exact_for_expert = exact["selected"] == index
        current_for_expert = current_route == config["family_key"]
        individual.append(
            {
                **config,
                "coverage_rows": int(eligible[:, index].sum()),
                "overlap_coverage_rows": int(np.sum(eligible[:, index] & (eligible_count > 1))),
                "a_error_rows": int(np.sum(eligible[:, index] & (~a_correct))),
                "a_correct_rows": int(np.sum(eligible[:, index] & a_correct)),
                "a_error_truth_in_top5": int(
                    np.sum(eligible[:, index] & (~a_correct) & (ranks[np.arange(rows), labels] <= 5))
                ),
                "both_members_in_top5_rows": int(np.sum(eligible[:, index] & both_top5[:, index])),
                "top3_trigger_rows": int(np.sum(eligible[:, index] & top3_trigger[:, index])),
                "top3_rescue_opportunities_reached": int(
                    np.sum(rescue & top3_trigger[:, index])
                ),
                "top5_rescue_opportunities_reached": int(
                    np.sum(rescue & both_top5[:, index])
                ),
                "rescue": int(rescue.sum()),
                "harm": int(harm.sum()),
                "net": int(rescue.sum() - harm.sum()),
                "unique_rescue": int(np.sum(rescue & (~other_correct))),
                "shared_rescue": int(np.sum(rescue & other_correct)),
                "union_assigned_rescue": int(np.sum(union["selected"] == index)),
                "exact_edge_rows": int(exact_for_expert.sum()),
                "exact_edge_rescue": int(
                    np.sum(exact_for_expert & (exact["prediction"] == labels))
                ),
                "current_corresponding_route_rows": int(current_for_expert.sum()),
                "current_corresponding_rescue": int(
                    np.sum(current_for_expert & (~a_correct) & (current_prediction == labels))
                ),
                "current_corresponding_harm": int(
                    np.sum(current_for_expert & a_correct & (current_prediction != labels))
                ),
                "rescuable_not_reached_current": int(
                    np.sum(rescue & (~current_for_expert))
                ),
            }
        )

    coverage_overlap = eligible.T.astype(np.int64) @ eligible.astype(np.int64)
    rescue_overlap = rescue_matrix.T.astype(np.int64) @ rescue_matrix.astype(np.int64)
    trigger_overlap = top3_trigger.T.astype(np.int64) @ top3_trigger.astype(np.int64)
    disagreement_overlap = np.zeros_like(coverage_overlap)
    for left in range(len(SPECIALISTS)):
        for right in range(len(SPECIALISTS)):
            shared = eligible[:, left] & eligible[:, right]
            disagreement_overlap[left, right] = int(
                np.sum(shared & (specialist_prediction[:, left] != specialist_prediction[:, right]))
            )

    arbitration_rows: list[dict[str, Any]] = []
    for left in range(len(SPECIALISTS)):
        for right in range(left + 1, len(SPECIALISTS)):
            shared = eligible[:, left] & eligible[:, right]
            if not shared.any():
                continue
            left_correct = expert_correct[:, left]
            right_correct = expert_correct[:, right]
            shared_error = shared & (~a_correct)
            arbitration_rows.append(
                {
                    "left_family": SPECIALISTS[left]["family_key"],
                    "right_family": SPECIALISTS[right]["family_key"],
                    "shared_classes": join_values(
                        sorted(set(SPECIALISTS[left]["classes"]) & set(SPECIALISTS[right]["classes"]))
                    ),
                    "coverage_rows": int(shared.sum()),
                    "a_error_rows": int(shared_error.sum()),
                    "truth_in_top5_a_errors": int(
                        np.sum(shared_error & (ranks[np.arange(rows), labels] <= 5))
                    ),
                    "both_correct": int(np.sum(shared & left_correct & right_correct)),
                    "left_only_correct": int(np.sum(shared & left_correct & (~right_correct))),
                    "right_only_correct": int(np.sum(shared & (~left_correct) & right_correct)),
                    "neither_correct": int(np.sum(shared & (~left_correct) & (~right_correct))),
                    "prediction_disagreement": int(
                        np.sum(shared & (specialist_prediction[:, left] != specialist_prediction[:, right]))
                    ),
                    "shared_rescue": int(np.sum(shared_error & left_correct & right_correct)),
                    "left_only_rescue": int(
                        np.sum(shared_error & left_correct & (~right_correct))
                    ),
                    "right_only_rescue": int(
                        np.sum(shared_error & (~left_correct) & right_correct)
                    ),
                    "either_top3_trigger": int(
                        np.sum(shared & (top3_trigger[:, left] | top3_trigger[:, right]))
                    ),
                    "both_top3_trigger": int(
                        np.sum(shared & top3_trigger[:, left] & top3_trigger[:, right])
                    ),
                    "current_left_calls": int(np.sum(shared & (current_route == SPECIALISTS[left]["family_key"]))),
                    "current_right_calls": int(np.sum(shared & (current_route == SPECIALISTS[right]["family_key"]))),
                }
            )

    normalized_entropy = -np.sum(
        a_probability * np.log(np.maximum(a_probability, 1e-12)), axis=1
    ) / np.log(a_probability.shape[1])
    a_margin = a_probability[np.arange(rows), top_order[:, 0]] - a_probability[
        np.arange(rows), top_order[:, 1]
    ]
    current_correct = current_prediction == labels
    current_rescue = (~a_correct) & current_correct
    current_harm = a_correct & (~current_correct)
    rejected_correct = rejected_prediction == labels

    sample_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for row in range(rows):
        eligible_indices = np.flatnonzero(eligible[row]).tolist()
        correct_indices = np.flatnonzero(expert_correct[row]).tolist()
        top3_indices = np.flatnonzero(top3_trigger[row]).tolist()
        top5_indices = np.flatnonzero(both_top5[row]).tolist()
        union_index = int(union["selected"][row])
        forced_index = int(forced["selected"][row])
        exact_index = int(exact["selected"][row])
        top5_classes = top_order[row, :5].tolist()
        top5_probabilities = a_probability[row, top_order[row, :5]].tolist()
        base = {
            "row_index": row,
            "sample_id": sample_ids[row],
            "subject": users[row],
            "outer_fold": int(fold_ids[row]),
            "true_label": int(labels[row]),
            "a_prediction": int(a_prediction[row]),
            "a_correct": bool(a_correct[row]),
            "a_confidence": float(a_probability[row, a_prediction[row]]),
            "a_margin": float(a_margin[row]),
            "a_normalized_entropy": float(normalized_entropy[row]),
            "a_logits_available": False,
            "a_top5_predictions": join_values(top5_classes),
            "a_top5_probabilities": join_values([f"{value:.8f}" for value in top5_probabilities]),
            "truth_rank": int(ranks[row, labels[row]]),
            "truth_in_top5": bool(ranks[row, labels[row]] <= 5),
            "eligible_families": join_values([SPECIALISTS[index]["family_key"] for index in eligible_indices]),
            "eligible_family_count": len(eligible_indices),
            "eligible_specialist_predictions": join_values(
                [
                    f"{SPECIALISTS[index]['family_key']}:{specialist_prediction[row, index]}"
                    for index in eligible_indices
                ]
            ),
            "correct_specialists": join_values([SPECIALISTS[index]["family_key"] for index in correct_indices]),
            "correct_specialist_count": len(correct_indices),
            "expanded_top3_triggered_families": join_values([SPECIALISTS[index]["family_key"] for index in top3_indices]),
            "expanded_top3_trigger_count": len(top3_indices),
            "expanded_both_top5_families": join_values([SPECIALISTS[index]["family_key"] for index in top5_indices]),
            "expanded_both_top5_count": len(top5_indices),
            "current_route": current_route[row],
            "current_prediction": int(current_prediction[row]),
            "current_route_rescue": bool(current_rescue[row]),
            "current_route_harm": bool(current_harm[row]),
            "current_route_matches_true_family": bool(current_route[row] in {SPECIALISTS[index]["family_key"] for index in eligible_indices}),
            "rejected_p106_route": rejected_route[row],
            "rejected_p106_prediction": int(rejected_prediction[row]),
            "rejected_p106_rescue": bool((not a_correct[row]) and rejected_correct[row]),
            "rejected_p106_harm": bool(a_correct[row] and (not rejected_correct[row])),
            "union_selected_specialist": SPECIALISTS[union_index]["family_key"] if union_index >= 0 else "",
            "union_prediction": int(union["prediction"][row]),
            "union_rescue": bool((not a_correct[row]) and union["prediction"][row] == labels[row]),
            "forced_selected_specialist": SPECIALISTS[forced_index]["family_key"] if forced_index >= 0 else "",
            "forced_prediction": int(forced["prediction"][row]),
            "forced_rescue": bool((not a_correct[row]) and forced["prediction"][row] == labels[row]),
            "forced_harm": bool(a_correct[row] and forced["prediction"][row] != labels[row]),
            "exact_edge_specialist": SPECIALISTS[exact_index]["family_key"] if exact_index >= 0 else "",
            "exact_edge_prediction": int(exact["prediction"][row]),
            "exact_edge_rescue": bool((not a_correct[row]) and exact["prediction"][row] == labels[row]),
        }
        sample_rows.append(base)
        for index in eligible_indices:
            config = SPECIALISTS[index]
            current_enters = current_route[row] == config["family_key"]
            family_rows.append(
                {
                    **base,
                    "family_key": config["family_key"],
                    "family_classes": join_values(config["classes"]),
                    "provenance": config["provenance"],
                    "evidence": config["evidence"],
                    "family_left_a_probability": float(a_probability[row, config["classes"][0]]),
                    "family_right_a_probability": float(a_probability[row, config["classes"][1]]),
                    "family_left_a_rank": int(ranks[row, config["classes"][0]]),
                    "family_right_a_rank": int(ranks[row, config["classes"][1]]),
                    "both_family_members_in_top5": bool(both_top5[row, index]),
                    "family_top3_trigger": bool(top3_trigger[row, index]),
                    "exact_edge_for_family": bool(exact_index == index),
                    "enters_corresponding_specialist_current": bool(current_enters),
                    "current_called_specialist": current_route[row],
                    "current_called_specialist_prediction": int(current_prediction[row]),
                    "specialist_prediction": int(specialist_prediction[row, index]),
                    "specialist_correct": bool(expert_correct[row, index]),
                    "specialist_rescue": bool(rescue_matrix[row, index]),
                    "specialist_harm": bool(harm_matrix[row, index]),
                    "correct_expert_not_reached_current": bool(
                        rescue_matrix[row, index] and (not current_enters)
                    ),
                    "a_error_truth_in_top5_last_mile": bool(
                        (not a_correct[row]) and ranks[row, labels[row]] <= 5
                    ),
                    "last_mile_rescuable_by_specialist": bool(
                        (not a_correct[row])
                        and ranks[row, labels[row]] <= 5
                        and expert_correct[row, index]
                    ),
                }
            )

    sample_frame = pd.DataFrame(sample_rows)
    family_frame = pd.DataFrame(family_rows)
    sample_frame.to_csv(output / "sample_to_expert_map.csv", index=False, encoding="utf-8-sig")
    family_frame.to_csv(output / "family_sample_audit_all.csv", index=False, encoding="utf-8-sig")
    for config in SPECIALISTS:
        selected = family_frame[family_frame["family_key"] == config["family_key"]]
        selected.to_csv(
            family_dir / f"{config['family_key']}.csv", index=False, encoding="utf-8-sig"
        )
    family_frame[~family_frame["a_correct"]].to_csv(
        output / "a_error_family_samples.csv", index=False, encoding="utf-8-sig"
    )
    family_frame[family_frame["specialist_harm"]].to_csv(
        output / "a_correct_expert_harm.csv", index=False, encoding="utf-8-sig"
    )
    family_frame[family_frame["correct_expert_not_reached_current"]].to_csv(
        output / "correct_expert_not_reached_current.csv", index=False, encoding="utf-8-sig"
    )
    family_frame[family_frame["a_error_truth_in_top5_last_mile"]].to_csv(
        output / "top5_last_mile_family_samples.csv", index=False, encoding="utf-8-sig"
    )
    sample_frame[sample_frame["eligible_family_count"] > 1].to_csv(
        output / "multi_family_samples.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(individual).to_csv(
        output / "specialist_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(arbitration_rows).to_csv(
        output / "arbitration_pairs.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(matrix_rows(coverage_overlap, "coverage_overlap_rows")).to_csv(
        output / "coverage_overlap_matrix.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(matrix_rows(rescue_overlap, "rescue_overlap_rows")).to_csv(
        output / "rescue_overlap_matrix.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(matrix_rows(trigger_overlap, "top3_trigger_overlap_rows")).to_csv(
        output / "top3_trigger_overlap_matrix.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(matrix_rows(disagreement_overlap, "prediction_disagreement_rows")).to_csv(
        output / "prediction_disagreement_overlap_matrix.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {"system": "a_baseline", **system_metrics(labels, a_prediction, a_prediction)},
            {"system": "abstaining_union_oracle", **union_metrics},
            {"system": "forced_family_oracle", **forced_metrics},
            {"system": "exact_edge_oracle", **exact_metrics},
            {"system": "current_accepted_p105_top3", **current_metrics},
            {"system": "rejected_p106_call_model", **rejected_metrics},
        ]
    ).to_csv(output / "oracle_system_summary.csv", index=False, encoding="utf-8-sig")

    p105_indices = [index for index, value in enumerate(SPECIALISTS) if value["provenance"] == "P105"]
    p106_indices = [index for index, value in enumerate(SPECIALISTS) if value["provenance"] == "P106"]
    p105_rescuable = np.any(rescue_matrix[:, p105_indices], axis=1)
    p106_rescuable = np.any(rescue_matrix[:, p106_indices], axis=1)
    union_rescuable = rescue_matrix.any(axis=1)
    current_reaches_correct = np.asarray(
        [
            any(
                current_route[row] == SPECIALISTS[index]["family_key"]
                for index in np.flatnonzero(rescue_matrix[row])
            )
            for row in range(rows)
        ],
        dtype=bool,
    )
    top3_reaches_correct = np.any(rescue_matrix & top3_trigger, axis=1)
    top5_reaches_correct = np.any(rescue_matrix & both_top5, axis=1)
    routing_coverage = {
        "current_accepted_p105_top3": {
            **current_metrics,
            "routed_rows": int(np.sum(current_route != "")),
            "correct_family_route_rows": int(
                np.sum(
                    [
                        current_route[row]
                        in {
                            SPECIALISTS[index]["family_key"]
                            for index in np.flatnonzero(eligible[row])
                        }
                        for row in range(rows)
                    ]
                )
            ),
            "union_rescue_opportunities_reached": int(
                np.sum(union_rescuable & current_reaches_correct)
            ),
            "union_rescue_opportunities_missed": int(
                np.sum(union_rescuable & (~current_reaches_correct))
            ),
        },
        "rejected_p106_call_model": {
            **rejected_metrics,
            "routed_rows": int(np.sum(rejected_route != "")),
        },
        "expanded_unarbitrated_top3": {
            "trigger_events": int(top3_trigger.sum()),
            "unique_triggered_rows": int(np.sum(top3_trigger.any(axis=1))),
            "multi_trigger_rows": int(np.sum(top3_trigger.sum(axis=1) > 1)),
            "union_rescue_opportunities_reached": int(
                np.sum(union_rescuable & top3_reaches_correct)
            ),
            "union_rescue_opportunities_missed": int(
                np.sum(union_rescuable & (~top3_reaches_correct))
            ),
            "system_net_not_computed": True,
        },
        "expanded_unarbitrated_both_top5": {
            "trigger_events": int(both_top5.sum()),
            "unique_triggered_rows": int(np.sum(both_top5.any(axis=1))),
            "multi_trigger_rows": int(np.sum(both_top5.sum(axis=1) > 1)),
            "union_rescue_opportunities_reached": int(
                np.sum(union_rescuable & top5_reaches_correct)
            ),
            "union_rescue_opportunities_missed": int(
                np.sum(union_rescuable & (~top5_reaches_correct))
            ),
            "system_net_not_computed": True,
        },
    }
    current_route_family = []
    for index, config in enumerate(SPECIALISTS):
        called = current_route == config["family_key"]
        rescue = int(np.sum(called & (~a_correct) & current_correct))
        harm = int(np.sum(called & a_correct & (~current_correct)))
        current_route_family.append(
            {
                "family_key": config["family_key"],
                "provenance": config["provenance"],
                "called_rows": int(called.sum()),
                "true_family_rows": int(np.sum(called & eligible[:, index])),
                "true_family_precision": float(
                    np.mean(eligible[called, index]) if called.any() else 0.0
                ),
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "family_rescue_opportunities": int(rescue_matrix[:, index].sum()),
                "family_rescue_opportunities_reached": int(
                    np.sum(called & rescue_matrix[:, index])
                ),
                "family_rescue_opportunities_missed": int(
                    np.sum((~called) & rescue_matrix[:, index])
                ),
            }
        )
    pd.DataFrame(current_route_family).to_csv(
        output / "current_route_family_coverage.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {
                "route": route,
                "routed_rows": metrics.get("routed_rows"),
                "trigger_events": metrics.get("trigger_events"),
                "unique_triggered_rows": metrics.get("unique_triggered_rows"),
                "multi_trigger_rows": metrics.get("multi_trigger_rows"),
                "rescue": metrics.get("rescue"),
                "harm": metrics.get("harm"),
                "net": metrics.get("net"),
                "union_rescue_opportunities_reached": metrics.get(
                    "union_rescue_opportunities_reached"
                ),
                "union_rescue_opportunities_missed": metrics.get(
                    "union_rescue_opportunities_missed"
                ),
                "system_net_not_computed": metrics.get(
                    "system_net_not_computed", False
                ),
            }
            for route, metrics in routing_coverage.items()
        ]
    ).to_csv(output / "routing_coverage_summary.csv", index=False, encoding="utf-8-sig")

    top5_last_mile = (~a_correct) & (ranks[np.arange(rows), labels] <= 5)
    summary = {
        "status": "complete",
        "protocol": "P107 analysis-only merge of existing P105/P106 source-safe OOF predictions",
        "data": {
            "rows": rows,
            "subjects": sorted(set(users.tolist())),
            "a_errors": int((~a_correct).sum()),
            "a_logits_available": False,
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "specialist_roster": list(SPECIALISTS),
        "oracle_systems": {
            "a_baseline": system_metrics(labels, a_prediction, a_prediction),
            "abstaining_union_oracle": union_metrics,
            "forced_family_oracle": forced_metrics,
            "exact_edge_oracle": exact_metrics,
        },
        "individual_specialists": individual,
        "routing_coverage": routing_coverage,
        "current_route_family_coverage": current_route_family,
        "error_structure": {
            "a_errors": int((~a_correct).sum()),
            "a_errors_truth_in_top5": int(top5_last_mile.sum()),
            "a_errors_truth_in_top5_rate": float(top5_last_mile.sum() / max((~a_correct).sum(), 1)),
            "bank_eligible_a_errors": int(
                np.sum((~a_correct) & eligible.any(axis=1))
            ),
            "bank_eligible_a_errors_truth_in_top5": int(
                np.sum((~a_correct) & eligible.any(axis=1) & top5_last_mile)
            ),
            "expanded_bank_union_rescuable": int(union_rescuable.sum()),
            "rescuable_truth_in_top5": int(np.sum(union_rescuable & top5_last_mile)),
            "rescuable_truth_not_in_top5": int(np.sum(union_rescuable & (~top5_last_mile))),
            "p105_rescuable": int(p105_rescuable.sum()),
            "p106_rescuable": int(p106_rescuable.sum()),
            "p105_only_rescuable": int(np.sum(p105_rescuable & (~p106_rescuable))),
            "p106_only_rescuable": int(np.sum(p106_rescuable & (~p105_rescuable))),
            "both_generations_rescuable": int(np.sum(p105_rescuable & p106_rescuable)),
            "eligible_family_rows": int(np.sum(eligible_count > 0)),
            "multi_family_rows": int(np.sum(eligible_count > 1)),
            "multi_family_a_errors": int(np.sum((eligible_count > 1) & (~a_correct))),
            "multi_family_rescuable": int(np.sum((eligible_count > 1) & union_rescuable)),
            "multiple_correct_expert_rows": int(np.sum(correct_expert_count > 1)),
            "multiple_correct_expert_a_errors": int(
                np.sum((correct_expert_count > 1) & (~a_correct))
            ),
        },
        "coverage_overlap_matrix": coverage_overlap.tolist(),
        "rescue_overlap_matrix": rescue_overlap.tolist(),
        "top3_trigger_overlap_matrix": trigger_overlap.tolist(),
        "prediction_disagreement_overlap_matrix": disagreement_overlap.tolist(),
        "arbitration_pairs": arbitration_rows,
        "outputs": {
            "oracle_system_summary": "oracle_system_summary.csv",
            "specialist_summary": "specialist_summary.csv",
            "routing_coverage_summary": "routing_coverage_summary.csv",
            "current_route_family_coverage": "current_route_family_coverage.csv",
            "all_samples": "sample_to_expert_map.csv",
            "all_family_rows": "family_sample_audit_all.csv",
            "family_tables": [f"families/{config['family_key']}.csv" for config in SPECIALISTS],
            "a_errors": "a_error_family_samples.csv",
            "a_correct_expert_harm": "a_correct_expert_harm.csv",
            "correct_expert_not_reached": "correct_expert_not_reached_current.csv",
            "top5_last_mile": "top5_last_mile_family_samples.csv",
            "multi_family": "multi_family_samples.csv",
        },
        "source_safe": True,
        "models_trained": 0,
        "router_optimized": False,
        "h3_rows_selected": 0,
        "h3_users_loaded": [],
        "unified_b_teacher_trained": False,
        "student_started": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "oracle_systems": summary["oracle_systems"],
                "error_structure": summary["error_structure"],
                "routing_coverage": summary["routing_coverage"],
                "individual_specialists": individual,
                "arbitration_pairs": arbitration_rows,
                "h3_rows_selected": 0,
                "models_trained": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
