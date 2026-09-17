"""Build the final paired mechanism audit for P93-v3 cross-attention pooling."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/p93_temporal_cross_attention_v3_h1h2_v1"
DEFAULT_TAXONOMY = (
    HERE / "data/six_modality_audit/small_action_taxonomy_v1.csv"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty audit: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return fmean(items) if items else None


def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    p86 = sum(row["p86_prediction"] == row["label"] for row in rows)
    p93 = sum(row["p93_prediction"] == row["label"] for row in rows)
    return {
        "total": len(rows),
        "p86_correct": p86,
        "p93_correct": p93,
        "delta_correct": p93 - p86,
        "rescue": sum(row["transition"] == "rescue" for row in rows),
        "harm": sum(row["transition"] == "harm" for row in rows),
    }


def build_audit(run_dir: Path, taxonomy_path: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    source_rows = read_rows(run_dir / "h1_paired_predictions.csv")
    taxonomy = {
        int(row["class_id"]): row for row in read_rows(taxonomy_path)
    }
    float_fields = (
        "mean_temporal_effect_rms_ratio",
        "mean_temporal_skeleton_weight",
        "mean_temporal_imu_weight",
        "temporal_skeleton_availability",
        "temporal_imu_availability",
        "temporal_skeleton_cross_attention_entropy",
        "temporal_imu_cross_attention_entropy",
        "temporal_skeleton_mean_abs_offset",
        "temporal_imu_mean_abs_offset",
        "temporal_pool_attention_entropy",
        "temporal_pool_weight_l1_from_uniform",
        "mean_temporal_pool_logit_abs",
    )
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        rows.append(
            {
                **row,
                "label": int(row["label"]),
                "p86_prediction": int(row["p86_prediction"]),
                "p93_prediction": int(row["p93_prediction"]),
                **{field: float(row[field]) for field in float_fields},
            }
        )

    paired = summary["h1"]["paired"]
    counterfactual = summary["h1"]["counterfactual"]
    transitions = Counter(row["transition"] for row in rows)
    p86_correct = sum(row["p86_prediction"] == row["label"] for row in rows)
    p93_correct = sum(row["p93_prediction"] == row["label"] for row in rows)
    users = sorted({row["user_id"] for row in rows})
    h3_users = set(summary["protocol"]["h3_users_excluded"])

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)
    class_rows: list[dict[str, Any]] = []
    for class_id in sorted(by_class):
        class_items = by_class[class_id]
        class_p86 = sum(
            row["p86_prediction"] == class_id for row in class_items
        )
        class_p93 = sum(
            row["p93_prediction"] == class_id for row in class_items
        )
        class_rows.append(
            {
                "class_id": class_id,
                "action_name": taxonomy[class_id]["action_name"],
                "semantic_group": taxonomy[class_id]["semantic_group"],
                "include_fixed_small_action": int(
                    taxonomy[class_id]["include_fixed_small_action"]
                ),
                "total": len(class_items),
                "p86_correct": class_p86,
                "p93_correct": class_p93,
                "delta_correct": class_p93 - class_p86,
                "rescue": sum(
                    row["transition"] == "rescue" for row in class_items
                ),
                "harm": sum(row["transition"] == "harm" for row in class_items),
                "mean_temporal_effect_rms_ratio": mean(
                    row["mean_temporal_effect_rms_ratio"] for row in class_items
                ),
            }
        )

    fixed_small = [
        row
        for row in rows
        if int(taxonomy[row["label"]]["include_fixed_small_action"]) == 1
    ]
    other = [
        row
        for row in rows
        if int(taxonomy[row["label"]]["include_fixed_small_action"]) == 0
    ]
    transition_stats = {}
    for transition in ("rescue", "harm", "stable_correct", "stable_wrong"):
        items = [row for row in rows if row["transition"] == transition]
        transition_stats[transition] = {
            "count": len(items),
            **{
                (field if field.startswith("mean_") else f"mean_{field}"): mean(
                    row[field] for row in items
                )
                for field in (
                    "mean_temporal_effect_rms_ratio",
                    "temporal_pool_weight_l1_from_uniform",
                    "mean_temporal_pool_logit_abs",
                )
            },
        }

    frames = 16
    parts = 5
    radius = int(summary["protocol"]["temporal_radius"])
    local_choices = [
        (min(frames - 1, time + radius) - max(0, time - radius) + 1) * parts
        for time in range(frames)
    ]
    uniform_cross_entropy = fmean(math.log(value) for value in local_choices)
    uniform_absolute_offset = fmean(
        fmean(abs(time - memory) for memory in range(
            max(0, time - radius), min(frames - 1, time + radius) + 1
        ))
        for time in range(frames)
    )
    pool_uniform_entropy = math.log(frames)

    sample_ids = [row["sample_id"] for row in rows]
    checks = {
        "unique_sample_ids": len(sample_ids) == len(set(sample_ids)),
        "source_users_match_protocol": users
        == sorted(summary["protocol"]["source_users"]),
        "no_h3_user_in_h1_rows": not (set(users) & h3_users),
        "p86_count_recomputed": p86_correct == paired["p86"]["correct"],
        "p93_count_recomputed": p93_correct == paired["p93"]["correct"],
        "rescue_count_recomputed": transitions["rescue"] == paired["rescue"],
        "harm_count_recomputed": transitions["harm"] == paired["harm"],
        "zero_counterfactual_exact": counterfactual["zero"][
            "maximum_logit_error_vs_p86_anchor"
        ]
        == 0.0,
        "h2_not_run": summary["h2"] is None,
        "orchestrator_has_no_h3_path": summary["protocol"][
            "h3_code_path_exists"
        ]
        is False,
        "backbone_loss_protocol_unchanged": not any(
            summary["protocol"][key]
            for key in (
                "teacher_pool_changed",
                "backbone_changed",
                "loss_changed",
                "training_protocol_changed",
            )
        ),
    }

    harm_pairs = Counter(
        (row["label"], row["p93_prediction"])
        for row in rows
        if row["transition"] == "harm"
    )
    rescue_pairs = Counter(
        (row["label"], row["p86_prediction"])
        for row in rows
        if row["transition"] == "rescue"
    )

    def named_pairs(counter: Counter[tuple[int, int]]) -> list[dict[str, Any]]:
        return [
            {
                "label": label,
                "label_name": taxonomy[label]["action_name"],
                "prediction": prediction,
                "prediction_name": taxonomy[prediction]["action_name"],
                "count": count,
            }
            for (label, prediction), count in sorted(
                counter.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    return {
        "stage": "P93_temporal_cross_attention_v3_final_mechanism_audit",
        "decision": summary["decision"],
        "sample_universe": {
            "total": len(rows),
            "source_users": users,
            "h3_users_excluded": sorted(h3_users),
        },
        "paired_result": {
            "p86_correct": p86_correct,
            "p93_correct": p93_correct,
            "delta_correct": p93_correct - p86_correct,
            "delta_accuracy_pp": 100.0 * (p93_correct - p86_correct) / len(rows),
            "rescue": transitions["rescue"],
            "harm": transitions["harm"],
            "worst_user_delta_correct": paired["worst_user_delta_correct"],
            "user_delta_correct": paired["user_delta_correct"],
        },
        "counterfactual_specificity": {
            "aligned_correct": p93_correct,
            "zero_correct": counterfactual["zero"]["correct"],
            "reverse_time_correct": counterfactual["reverse_time"]["correct"],
            "sample_roll_correct": counterfactual["sample_roll"]["correct"],
            "aligned_minus_reverse_time": p93_correct
            - counterfactual["reverse_time"]["correct"],
            "aligned_minus_sample_roll": p93_correct
            - counterfactual["sample_roll"]["correct"],
            "zero_maximum_logit_error_vs_p86": counterfactual["zero"][
                "maximum_logit_error_vs_p86_anchor"
            ],
        },
        "attention_alignment": {
            "skeleton_cross_attention_entropy": paired[
                "mean_temporal_skeleton_cross_attention_entropy"
            ],
            "imu_cross_attention_entropy": paired[
                "mean_temporal_imu_cross_attention_entropy"
            ],
            "uniform_local_cross_attention_entropy": uniform_cross_entropy,
            "skeleton_entropy_fraction_of_uniform": paired[
                "mean_temporal_skeleton_cross_attention_entropy"
            ]
            / uniform_cross_entropy,
            "imu_entropy_fraction_of_uniform": paired[
                "mean_temporal_imu_cross_attention_entropy"
            ]
            / uniform_cross_entropy,
            "skeleton_mean_abs_offset": paired[
                "mean_temporal_skeleton_mean_abs_offset"
            ],
            "imu_mean_abs_offset": paired["mean_temporal_imu_mean_abs_offset"],
            "uniform_local_mean_abs_offset": uniform_absolute_offset,
            "pool_attention_entropy": paired[
                "mean_temporal_pool_attention_entropy"
            ],
            "pool_uniform_entropy": pool_uniform_entropy,
            "pool_entropy_fraction_of_uniform": paired[
                "mean_temporal_pool_attention_entropy"
            ]
            / pool_uniform_entropy,
            "pool_weight_l1_from_uniform": paired[
                "mean_temporal_pool_weight_l1_from_uniform"
            ],
            "mean_pool_logit_abs": paired["mean_temporal_pool_logit_abs"],
            "mean_effect_rms_ratio": paired["mean_temporal_effect_rms_ratio"],
            "skeleton_availability": paired[
                "mean_temporal_skeleton_availability"
            ],
            "imu_availability": paired["mean_temporal_imu_availability"],
        },
        "transition_stats": transition_stats,
        "fixed_small_action_subset": subset_summary(fixed_small),
        "other_action_subset": subset_summary(other),
        "class_rows": class_rows,
        "largest_class_losses": sorted(
            (row for row in class_rows if row["delta_correct"] < 0),
            key=lambda row: (row["delta_correct"], row["class_id"]),
        ),
        "largest_class_gains": sorted(
            (row for row in class_rows if row["delta_correct"] > 0),
            key=lambda row: (-row["delta_correct"], row["class_id"]),
        ),
        "harm_pairs": named_pairs(harm_pairs),
        "rescue_pairs": named_pairs(rescue_pairs),
        "integrity_checks": checks,
        "all_integrity_checks_passed": all(checks.values()),
        "mechanism_conclusion": (
            "P93-v3 replaces motion-vector residual injection with genuine local "
            "cross-attention conditioned visual pooling, yet aligned evidence is "
            "worse than reverse-time and sample-rolled evidence. Skeleton attention "
            "offset is indistinguishable from uniform local attention and visual "
            "pooling remains high-entropy. Together with v1/v2, this supports closing "
            "temporal fusion under the current frozen tokens, exact alignment and "
            "small-data protocol; it is not a universal impossibility claim."
        ),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    paired = audit["paired_result"]
    cf = audit["counterfactual_specificity"]
    alignment = audit["attention_alignment"]
    small = audit["fixed_small_action_subset"]
    lines = [
        "# P93-v3 final mechanism audit",
        "",
        "## Outcome",
        "",
        f"- Decision: `{audit['decision']}`.",
        f"- Paired H1: P86 `{paired['p86_correct']}/1107` vs P93-v3 `{paired['p93_correct']}/1107`; net `{paired['delta_correct']}` ({paired['delta_accuracy_pp']:+.3f} pp).",
        f"- Transitions: `{paired['rescue']}` rescue / `{paired['harm']}` harm; worst user `{paired['worst_user_delta_correct']}`.",
        "- H1 failed, so H2 was not run. The orchestrator has no H3 path.",
        "",
        "## Counterfactual specificity",
        "",
        f"- Aligned `{cf['aligned_correct']}`; reverse-time `{cf['reverse_time_correct']}`; sample-roll `{cf['sample_roll_correct']}`.",
        f"- Aligned minus reverse `{cf['aligned_minus_reverse_time']}`; aligned minus sample-roll `{cf['aligned_minus_sample_roll']}`.",
        f"- Zero interaction `{cf['zero_correct']}`; maximum P86 logit error `{cf['zero_maximum_logit_error_vs_p86']}`.",
        "",
        "Correct time and sample correspondence again provide no positive evidence.",
        "",
        "## Attention and alignment",
        "",
        f"- Skeleton cross-attention entropy `{alignment['skeleton_cross_attention_entropy']:.4f}` (`{alignment['skeleton_entropy_fraction_of_uniform']:.4f}` of local-uniform entropy).",
        f"- IMU cross-attention entropy `{alignment['imu_cross_attention_entropy']:.4f}` (`{alignment['imu_entropy_fraction_of_uniform']:.4f}` of local-uniform entropy).",
        f"- Mean absolute offset: Skeleton `{alignment['skeleton_mean_abs_offset']:.4f}`, IMU `{alignment['imu_mean_abs_offset']:.4f}`, local-uniform expectation `{alignment['uniform_local_mean_abs_offset']:.4f}`.",
        f"- Visual pooling entropy `{alignment['pool_attention_entropy']:.4f}` (`{alignment['pool_entropy_fraction_of_uniform']:.4f}` of uniform); L1 from uniform `{alignment['pool_weight_l1_from_uniform']:.4f}`.",
        f"- Mean pooling effect/visual RMS `{alignment['mean_effect_rms_ratio']:.4f}`; mean absolute pooling logit `{alignment['mean_pool_logit_abs']:.4f}`.",
        "",
        "The mechanism changes pooling, but local cross-attention does not learn a transferable aligned-time preference.",
        "",
        "## Class audit",
        "",
        f"- Fixed small-action subset: `{small['p86_correct']}` -> `{small['p93_correct']}` (net `{small['delta_correct']}`, `{small['rescue']}` rescue / `{small['harm']}` harm).",
        "- Largest class losses:",
        "",
    ]
    for row in audit["largest_class_losses"][:5]:
        lines.append(
            f"  - `{row['class_id']} {row['action_name']}`: {row['p86_correct']} -> {row['p93_correct']} (net {row['delta_correct']})."
        )
    lines.extend(
        [
            "",
            "## Integrity and scope",
            "",
            f"All recomputation/protocol checks passed: `{audit['all_integrity_checks_passed']}`.",
            "",
            audit["mechanism_conclusion"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    audit = build_audit(run_dir, args.taxonomy.resolve())
    (run_dir / "mechanism_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_rows(run_dir / "paired_by_class.csv", audit["class_rows"])
    (run_dir / "MECHANISM_AUDIT.md").write_text(
        render_markdown(audit), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "decision",
                    "paired_result",
                    "counterfactual_specificity",
                    "attention_alignment",
                    "fixed_small_action_subset",
                    "all_integrity_checks_passed",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
