"""Trace the frozen P89 -> P90 -> P91 decision chain without tuning on H3.

This audit separates three concepts that were previously easy to conflate:

1. an immediate upstream prediction was correct and a downstream decision
   changed it to a wrong class;
2. at least one independent expert was correct but the deployed fusion was not;
3. the true class only appeared in an expert Top-K candidate union.

All H3 outputs are diagnostic only.  This script does not select a threshold,
fit a model, or write a deployable prediction rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax

import p89_full40_scale_invariant_transfer as full40
from p89_supported_template_gate import h3_protocol, load_grouping, load_imu
from p90_crossuser_visual_router import load_splits
from p90_teacher_fusion_audit import align
from p90_visual_teacher_safe_fusion_audit import load_visual_candidates


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "runs/p93_phase0_decision_provenance_v1"
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalise_probability(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    probability = np.clip(probability, EPSILON, None)
    return probability / probability.sum(axis=1, keepdims=True)


def probability_margin(probability: np.ndarray) -> np.ndarray:
    top = np.partition(probability, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def class_names() -> dict[int, str]:
    output: dict[int, str] = {}
    with (HERE / "data/manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            output[int(row["class_id"])] = row["class_name"]
    return output


def recording_groups() -> dict[str, str]:
    path = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
    output: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            output[row["sample_id"]] = f"{row['user_id']}|{row['recording_date']}"
    return output


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def original_protocols() -> dict[str, tuple[Any, ...]]:
    imu_ids, imu_logits = load_imu()
    grouping = load_grouping()
    return {
        "H1_selection": full40.protocol(full40.H1_RUN, full40.H1_USERS),
        "H2_confirmation": full40.protocol(full40.H2_RUN, full40.H2_USERS),
        "H3_independent_fold0": h3_protocol(imu_ids, imu_logits, grouping)[0],
    }


def align_protocol(
    protocol: tuple[Any, ...], target_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source_ids = np.asarray(protocol[0]).astype(str)
    raw_probability = align(
        source_ids, np.asarray(protocol[2], dtype=np.float64), target_ids.astype(str)
    )
    p87_prediction = align(
        source_ids, np.asarray(protocol[3], dtype=np.int64), target_ids.astype(str)
    )
    return normalise_probability(raw_probability), p87_prediction.astype(np.int64)


def load_expert_pool(
    split_ids: np.ndarray, visual: dict[str, np.ndarray], visual_ids: np.ndarray
) -> tuple[list[str], np.ndarray]:
    skeleton = load_npz(
        HERE / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz"
    )
    motionbert = load_npz(
        PROJECT
        / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front_linear_oof.npz"
    )
    imu = load_npz(
        PROJECT
        / "runs/p90_imu_teacher_blend_v1/imu_p90_sensorwise_plus_deep_crossfit_oof.npz"
    )
    names = [
        "internvideo2_l_early_late_plus_k400",
        "videomaev2_base_plus_internvideo2_l_equal",
        "videomaev2_distilled_base",
        "skeleton_invariant",
        "motionbert_front",
        "imu_sensorwise_deep",
    ]
    probability = [
        normalise_probability(align(visual_ids, visual[name], split_ids))
        for name in names[:3]
    ]
    probability.extend(
        (
            normalise_probability(
                softmax(
                    align(
                        skeleton["sample_ids"].astype(str),
                        skeleton["skeleton_logits"],
                        split_ids,
                    ),
                    axis=1,
                )
            ),
            normalise_probability(
                align(
                    motionbert["sample_ids"].astype(str),
                    motionbert["probabilities"],
                    split_ids,
                )
            ),
            normalise_probability(
                align(
                    imu["sample_ids"].astype(str),
                    imu["probabilities"],
                    split_ids,
                )
            ),
        )
    )
    return names, np.stack(probability, axis=1)


def blended_probability(
    logits: np.ndarray,
    anchor_prediction: np.ndarray,
    neural_weight: float | np.ndarray,
) -> np.ndarray:
    neural = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    anchor = np.full((len(logits), 40), 0.06 / 39.0, dtype=np.float64)
    anchor[np.arange(len(logits)), anchor_prediction.astype(np.int64)] = 0.94
    weight = np.asarray(neural_weight, dtype=np.float64)
    if weight.ndim:
        weight = weight[:, None]
    score = weight * np.log(np.clip(neural, EPSILON, 1.0))
    score += (1.0 - weight) * np.log(np.clip(anchor, EPSILON, 1.0))
    return softmax(score, axis=1)


def load_p91(
    split: str, target_ids: np.ndarray
) -> dict[str, np.ndarray] | None:
    if split == "H2_confirmation":
        source = load_npz(
            PROJECT / "runs/p91_hierarchical_multimodal_h3_v3/inner_predictions.npz"
        )
        rows = align(
            source["sample_ids"].astype(str),
            np.arange(len(source["sample_ids"])),
            target_ids,
        ).astype(np.int64)
        logits = source["direct_logits"][rows]
        anchor = source["teacher_prediction"][rows].astype(np.int64)
        weight = float(source["selected_constant_weight"])
        selected_probability = blended_probability(logits, anchor, weight)
        return {
            "direct_probability": softmax(logits, axis=1),
            "champion_probability": selected_probability,
            "anchor_prediction": anchor,
        }
    if split == "H3_independent_fold0":
        source = load_npz(
            PROJECT / "runs/p91_hierarchical_multimodal_h3_v3/predictions.npz"
        )
        rows = align(
            source["sample_ids"].astype(str),
            np.arange(len(source["sample_ids"])),
            target_ids,
        ).astype(np.int64)
        logits = source["direct_logits"][rows]
        anchor = source["base_prediction"][rows].astype(np.int64)
        weight = float(source["selected_blend_weight"])
        champion_probability = blended_probability(logits, anchor, weight)
        stored_champion = source["blended_prediction"][rows].astype(np.int64)
        if not np.array_equal(champion_probability.argmax(axis=1), stored_champion):
            raise RuntimeError("stored P91 blended prediction differs from frozen blend")
        adaptive_probability = blended_probability(
            logits, anchor, source["adaptive_weights"][rows]
        )
        stored_selected = source["selected_prediction"][rows].astype(np.int64)
        if not np.array_equal(adaptive_probability.argmax(axis=1), stored_selected):
            raise RuntimeError("stored P91 selected prediction differs from adaptive blend")
        return {
            "direct_probability": softmax(logits, axis=1),
            "champion_probability": champion_probability,
            "script_selected_probability": adaptive_probability,
            "anchor_prediction": anchor,
        }
    return None


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    correct = prediction == labels
    return {
        "correct": int(correct.sum()),
        "total": int(len(labels)),
        "accuracy": float(correct.mean()),
    }


def transition_counts(
    labels: np.ndarray, source: np.ndarray, target: np.ndarray
) -> dict[str, int]:
    source_correct = source == labels
    target_correct = target == labels
    return {
        "source_correct": int(source_correct.sum()),
        "target_correct": int(target_correct.sum()),
        "changed": int(np.sum(source != target)),
        "rescue": int(np.sum(~source_correct & target_correct)),
        "harm": int(np.sum(source_correct & ~target_correct)),
        "net": int(target_correct.sum() - source_correct.sum()),
        "unchanged_correct": int(np.sum(source_correct & target_correct)),
        "unchanged_wrong": int(np.sum(~source_correct & ~target_correct)),
    }


def grouped_transition(
    group_values: np.ndarray,
    labels: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for value in sorted(np.unique(group_values).tolist()):
        selected = group_values == value
        output[str(value)] = transition_counts(
            labels[selected], source[selected], target[selected]
        )
    return output


def transition_audit(
    name: str,
    labels: np.ndarray,
    users: np.ndarray,
    groups: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    source_probability: np.ndarray | None,
    pool_prediction: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "transition": name,
        **transition_counts(labels, source, target),
        "per_user": grouped_transition(users, labels, source, target),
        "per_class": grouped_transition(labels.astype(str), labels, source, target),
        "per_recording_group": grouped_transition(groups, labels, source, target),
    }
    harm = (source == labels) & (target != labels)
    pool_agreement = np.mean(pool_prediction == source[:, None], axis=1)
    result["expert_agreement_on_harm"] = {
        "harm_with_pool_majority_supporting_source": int(
            np.sum(harm & (pool_agreement > 0.5))
        ),
        "harm_with_pool_unanimously_supporting_source": int(
            np.sum(harm & (pool_agreement == 1.0))
        ),
    }
    if source_probability is not None:
        margin = probability_margin(source_probability)
        confidence = source_probability.max(axis=1)
        result["confidence_harm"] = {
            f"margin_ge_{threshold:.2f}": int(np.sum(harm & (margin >= threshold)))
            for threshold in (0.05, 0.10, 0.25, 0.50)
        }
        result["confidence_harm"].update(
            {
                f"confidence_ge_{threshold:.2f}": int(
                    np.sum(harm & (confidence >= threshold))
                )
                for threshold in (0.50, 0.75, 0.90)
            }
        )
    return result


def candidate_union_audit(
    labels: np.ndarray, pool_probability: np.ndarray, selected: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": int(selected.sum())}
    if not selected.any():
        return result
    labels = labels[selected]
    probability = pool_probability[selected]
    for k in (2, 3, 5, 10):
        top = np.argsort(-probability, axis=2, kind="stable")[:, :, :k]
        cardinality = np.asarray(
            [len(set(row.reshape(-1).tolist())) for row in top], dtype=np.int64
        )
        covered = np.asarray(
            [int(label) in set(row.reshape(-1).tolist()) for label, row in zip(labels, top)],
            dtype=bool,
        )
        result[f"union_top{k}"] = {
            "covered": int(covered.sum()),
            "coverage": float(covered.mean()),
            "candidate_size_mean": float(cardinality.mean()),
            "candidate_size_median": float(np.median(cardinality)),
            "candidate_size_p90": float(np.percentile(cardinality, 90)),
            "candidate_size_max": int(cardinality.max()),
        }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# P93 Phase 0 decision provenance audit",
        "",
        "> H3 is diagnostic only. No threshold, family, checkpoint, or deployable rule was selected here.",
        "",
        "## Main distinction",
        "",
        "The historical `expert_top1_recoverable` count is an oracle selector ceiling. "
        "It is not the number of rows deterministically damaged by the final decision chain. "
        "The latter is reported as `adjusted_probability_top1 -> p89_safe` harm.",
        "",
    ]
    for split, item in summary["splits"].items():
        tier = item["tier_inventory"]
        lines.extend(
            [
                f"## {split}",
                "",
                f"- rows: {item['rows']}",
                f"- P89 Safe errors: {tier['p89_safe_errors']}",
                f"- immediate adjusted-probability Top-1 correct but P89 Safe wrong: {tier['adjusted_top1_correct_safe_wrong']}",
                f"- P87 sequence correct but P89 Safe wrong: {tier['p87_correct_safe_wrong']}",
                f"- at least one frozen non-redundant expert Top-1 correct: {tier['expert_top1_recoverable']}",
                f"- all frozen non-redundant experts Top-1 wrong: {tier['all_expert_top1_wrong']}",
                f"- all six experts unanimously correct but P89 Safe wrong: {tier['pool_unanimous_true_safe_wrong']}",
                "",
                "| Transition | Source | Target | Rescue | Harm | Net | Changed |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in item["transitions"]:
            lines.append(
                f"| {row['transition']} | {row['source_correct']} | {row['target_correct']} | "
                f"{row['rescue']} | {row['harm']} | {row['net']} | {row['changed']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision rule",
            "",
            "This audit alone cannot promote a change. Any protection or bounded correction must be selected on H1, frozen on H2, and evaluated once on H3. Hard-158 and H3 sample identities remain diagnostic-only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = class_names()
    group_lookup = recording_groups()
    splits = load_splits()
    protocols = original_protocols()
    visual_ids, visual = load_visual_candidates()
    router = load_npz(PROJECT / "runs/p90_crossuser_visual_router_v1/full_predictions.npz")

    summary: dict[str, Any] = {
        "protocol": {
            "purpose": "diagnostic-only decision provenance; no H3 tuning",
            "fixed_champion": {
                "name": "p91_hierarchical_multimodal_h3_v3",
                "correct": 839,
                "total": 973,
                "accuracy": 0.8622816032887975,
            },
            "split_roles": {
                "H1_selection": "development and selection only",
                "H2_confirmation": "frozen confirmation",
                "H3_independent_fold0": "diagnostic outer migration check only",
            },
        },
        "splits": {},
    }
    sample_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    for split_name in SPLITS:
        split = splits[split_name]
        ids = split.sample_ids.astype(str)
        labels = split.labels.astype(np.int64)
        users = split.users.astype(str)
        groups = np.asarray([group_lookup[str(value)] for value in ids], dtype=str)
        raw_probability, p87_prediction = align_protocol(protocols[split_name], ids)
        adjusted_probability = normalise_probability(split.safe_probability)
        p89_safe = split.safe_prediction.astype(np.int64)
        raw_prediction = raw_probability.argmax(axis=1)
        adjusted_prediction = adjusted_probability.argmax(axis=1)

        router_prediction = align(
            router[f"{split_name}_sample_ids"].astype(str),
            router[f"{split_name}_router_prediction"].astype(np.int64),
            ids,
        ).astype(np.int64)
        expert_names, pool_probability = load_expert_pool(ids, visual, visual_ids)
        pool_prediction = pool_probability.argmax(axis=2)
        p91 = load_p91(split_name, ids)

        stages: dict[str, np.ndarray] = {
            "p85_raw_probability_top1": raw_prediction,
            "p87_sequence": p87_prediction,
            "p89_adjusted_probability_top1": adjusted_prediction,
            "p89_safe": p89_safe,
            "p90_router": router_prediction,
        }
        stage_probability: dict[str, np.ndarray] = {
            "p85_raw_probability_top1": raw_probability,
            "p89_adjusted_probability_top1": adjusted_probability,
        }
        if p91 is not None:
            p91_direct_probability = p91["direct_probability"]
            p91_champion_probability = p91["champion_probability"]
            p91_anchor = p91["anchor_prediction"]
            if not np.array_equal(p91_anchor, router_prediction):
                raise RuntimeError(f"{split_name}: P91 anchor differs from P90 router")
            stages["p91_direct"] = p91_direct_probability.argmax(axis=1)
            stages["p91_champion_blend"] = p91_champion_probability.argmax(axis=1)
            stage_probability["p91_direct"] = p91_direct_probability
            stage_probability["p91_champion_blend"] = p91_champion_probability
            if "script_selected_probability" in p91:
                stages["p91_script_selected_adaptive"] = p91[
                    "script_selected_probability"
                ].argmax(axis=1)
                stage_probability["p91_script_selected_adaptive"] = p91[
                    "script_selected_probability"
                ]

        transition_specs = [
            ("p85_raw_probability_top1 -> p87_sequence", "p85_raw_probability_top1", "p87_sequence"),
            ("p85_raw_probability_top1 -> p89_adjusted_probability_top1", "p85_raw_probability_top1", "p89_adjusted_probability_top1"),
            ("p87_sequence -> p89_safe", "p87_sequence", "p89_safe"),
            ("p89_adjusted_probability_top1 -> p89_safe", "p89_adjusted_probability_top1", "p89_safe"),
            ("p89_safe -> p90_router", "p89_safe", "p90_router"),
        ]
        if p91 is not None:
            transition_specs.extend(
                (
                    ("p90_router -> p91_direct", "p90_router", "p91_direct"),
                    ("p90_router -> p91_champion_blend", "p90_router", "p91_champion_blend"),
                    ("p91_direct -> p91_champion_blend", "p91_direct", "p91_champion_blend"),
                    ("p89_adjusted_probability_top1 -> p91_champion_blend", "p89_adjusted_probability_top1", "p91_champion_blend"),
                )
            )
            if "p91_script_selected_adaptive" in stages:
                transition_specs.append(
                    (
                        "p91_champion_blend -> p91_script_selected_adaptive",
                        "p91_champion_blend",
                        "p91_script_selected_adaptive",
                    )
                )

        transitions = []
        for transition_name, source_name, target_name in transition_specs:
            item = transition_audit(
                transition_name,
                labels,
                users,
                groups,
                stages[source_name],
                stages[target_name],
                stage_probability.get(source_name),
                pool_prediction,
            )
            transitions.append(item)
            transition_rows.append(
                {
                    "split": split_name,
                    **{
                        key: item[key]
                        for key in (
                            "transition",
                            "source_correct",
                            "target_correct",
                            "changed",
                            "rescue",
                            "harm",
                            "net",
                            "unchanged_correct",
                            "unchanged_wrong",
                        )
                    },
                }
            )

        safe_errors = p89_safe != labels
        expert_correct = np.any(pool_prediction == labels[:, None], axis=1)
        unanimous_true = np.all(pool_prediction == labels[:, None], axis=1)
        tier_inventory = {
            "p89_safe_errors": int(safe_errors.sum()),
            "adjusted_top1_correct_safe_wrong": int(
                np.sum(safe_errors & (adjusted_prediction == labels))
            ),
            "p87_correct_safe_wrong": int(
                np.sum(safe_errors & (p87_prediction == labels))
            ),
            "expert_top1_recoverable": int(np.sum(safe_errors & expert_correct)),
            "all_expert_top1_wrong": int(np.sum(safe_errors & ~expert_correct)),
            "pool_unanimous_true_safe_wrong": int(
                np.sum(safe_errors & unanimous_true)
            ),
        }
        split_summary = {
            "rows": int(len(ids)),
            "expert_pool": expert_names,
            "stage_metrics": {
                name: metrics(labels, prediction) for name, prediction in stages.items()
            },
            "tier_inventory": tier_inventory,
            "candidate_union": {
                "all_rows": candidate_union_audit(
                    labels, pool_probability, np.ones(len(labels), dtype=bool)
                ),
                "p89_safe_errors": candidate_union_audit(
                    labels, pool_probability, safe_errors
                ),
                "all_expert_top1_wrong": candidate_union_audit(
                    labels, pool_probability, safe_errors & ~expert_correct
                ),
            },
            "transitions": transitions,
        }
        summary["splits"][split_name] = split_summary

        pool_rank = np.stack(
            [
                np.argmax(
                    np.argsort(-pool_probability[:, expert], axis=1, kind="stable")
                    == labels[:, None],
                    axis=1,
                )
                + 1
                for expert in range(pool_probability.shape[1])
            ],
            axis=1,
        )
        raw_margin = probability_margin(raw_probability)
        adjusted_margin = probability_margin(adjusted_probability)
        for row, sample_id in enumerate(ids):
            record: dict[str, Any] = {
                "split": split_name,
                "sample_id": str(sample_id),
                "user": str(users[row]),
                "recording_group": str(groups[row]),
                "true_class_id": int(labels[row]),
                "true_class": names[int(labels[row])],
                "p85_raw_margin": float(raw_margin[row]),
                "p89_adjusted_margin": float(adjusted_margin[row]),
                "p89_safe_error": int(safe_errors[row]),
                "adjusted_top1_correct_safe_wrong": int(
                    safe_errors[row] and adjusted_prediction[row] == labels[row]
                ),
                "expert_top1_recoverable": int(
                    safe_errors[row] and expert_correct[row]
                ),
                "all_expert_top1_wrong": int(
                    safe_errors[row] and not expert_correct[row]
                ),
                "expert_pool_min_true_rank": int(pool_rank[row].min()),
                "expert_pool_top1_agreement": float(
                    Counter(pool_prediction[row].tolist()).most_common(1)[0][1]
                    / pool_prediction.shape[1]
                ),
            }
            for stage_name, prediction in stages.items():
                value = int(prediction[row])
                record[f"{stage_name}_prediction"] = value
                record[f"{stage_name}_correct"] = int(value == labels[row])
            sample_rows.append(record)

    aggregate = {
        "rows": int(sum(item["rows"] for item in summary["splits"].values())),
        "p89_safe_errors": int(
            sum(
                item["tier_inventory"]["p89_safe_errors"]
                for item in summary["splits"].values()
            )
        ),
        "adjusted_top1_correct_safe_wrong": int(
            sum(
                item["tier_inventory"]["adjusted_top1_correct_safe_wrong"]
                for item in summary["splits"].values()
            )
        ),
        "expert_top1_recoverable": int(
            sum(
                item["tier_inventory"]["expert_top1_recoverable"]
                for item in summary["splits"].values()
            )
        ),
        "all_expert_top1_wrong": int(
            sum(
                item["tier_inventory"]["all_expert_top1_wrong"]
                for item in summary["splits"].values()
            )
        ),
    }
    summary["aggregate"] = aggregate
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output / "transitions.csv", transition_rows)
    write_csv(args.output / "samples.csv", sample_rows)
    (args.output / "REPORT.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate, "splits": {
        name: {
            "tier_inventory": item["tier_inventory"],
            "stage_metrics": item["stage_metrics"],
        }
        for name, item in summary["splits"].items()
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
