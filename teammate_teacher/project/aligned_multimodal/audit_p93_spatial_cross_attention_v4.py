"""Recompute the paired P93-v4 spatial-prepool mechanism audit."""

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
DEFAULT_RUN = HERE / "runs/p93_spatial_cross_attention_v4_h1h2_v1"
DEFAULT_TAXONOMY = HERE / "data/six_modality_audit/small_action_taxonomy_v1.csv"
FLOAT_FIELDS = (
    "mean_spatial_clip_effect_rms_ratio",
    "mean_spatial_correction_rms_ratio",
    "mean_spatial_skeleton_weight",
    "mean_spatial_imu_weight",
    "spatial_skeleton_availability",
    "spatial_imu_availability",
    "spatial_skeleton_part_attention_entropy",
    "spatial_imu_part_attention_entropy",
    "spatial_pool_attention_entropy",
    "spatial_pool_weight_l1_from_uniform",
    "mean_spatial_pool_logit_abs",
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


def subset_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
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
    taxonomy = {int(row["class_id"]): row for row in read_rows(taxonomy_path)}
    rows = [
        {
            **row,
            "label": int(row["label"]),
            "p86_prediction": int(row["p86_prediction"]),
            "p93_prediction": int(row["p93_prediction"]),
            **{field: float(row[field]) for field in FLOAT_FIELDS},
        }
        for row in read_rows(run_dir / "h1_paired_predictions.csv")
    ]
    paired = summary["h1"]["paired"]
    counterfactual = summary["h1"]["counterfactual"]
    gate = summary["h1"]["gate"]
    transitions = Counter(row["transition"] for row in rows)
    p86_correct = sum(row["p86_prediction"] == row["label"] for row in rows)
    p93_correct = sum(row["p93_prediction"] == row["label"] for row in rows)
    users = sorted({row["user_id"] for row in rows})
    h3_users = set(summary["protocol"]["h3_users_excluded"])

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)
    class_rows = []
    for class_id in sorted(by_class):
        items = by_class[class_id]
        old = sum(row["p86_prediction"] == class_id for row in items)
        new = sum(row["p93_prediction"] == class_id for row in items)
        class_rows.append(
            {
                "class_id": class_id,
                "action_name": taxonomy[class_id]["action_name"],
                "semantic_group": taxonomy[class_id]["semantic_group"],
                "include_fixed_small_action": int(
                    taxonomy[class_id]["include_fixed_small_action"]
                ),
                "total": len(items),
                "p86_correct": old,
                "p93_correct": new,
                "delta_correct": new - old,
                "rescue": sum(row["transition"] == "rescue" for row in items),
                "harm": sum(row["transition"] == "harm" for row in items),
                "mean_spatial_clip_effect_rms_ratio": mean(
                    row["mean_spatial_clip_effect_rms_ratio"] for row in items
                ),
            }
        )

    fixed_small = [
        row
        for row in rows
        if int(taxonomy[row["label"]]["include_fixed_small_action"]) == 1
    ]
    other = [row for row in rows if row not in fixed_small]
    transition_stats = {}
    for transition in ("rescue", "harm", "stable_correct", "stable_wrong"):
        items = [row for row in rows if row["transition"] == transition]
        transition_stats[transition] = {
            "count": len(items),
            "mean_spatial_clip_effect_rms_ratio": mean(
                row["mean_spatial_clip_effect_rms_ratio"] for row in items
            ),
            "mean_spatial_pool_weight_l1_from_uniform": mean(
                row["spatial_pool_weight_l1_from_uniform"] for row in items
            ),
            "mean_spatial_pool_logit_abs": mean(
                row["mean_spatial_pool_logit_abs"] for row in items
            ),
        }

    sample_ids = [row["sample_id"] for row in rows]
    zero_error = counterfactual["zero"]["maximum_logit_error_vs_p86_anchor"]
    checks = {
        "unique_sample_ids": len(sample_ids) == len(set(sample_ids)),
        "source_users_match_protocol": users
        == sorted(summary["protocol"]["source_users"]),
        "no_h3_user_in_h1_rows": not (set(users) & h3_users),
        "p86_count_recomputed": p86_correct == paired["p86"]["correct"],
        "p93_count_recomputed": p93_correct == paired["p93"]["correct"],
        "rescue_count_recomputed": transitions["rescue"] == paired["rescue"],
        "harm_count_recomputed": transitions["harm"] == paired["harm"],
        "zero_counterfactual_exact": zero_error <= 1e-4,
        "h2_gate_order_respected": (summary["h2"] is not None) == gate["passed"],
        "orchestrator_has_no_h3_path": summary["protocol"]["h3_code_path_exists"]
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

    aligned_beats_mismatch = p93_correct >= max(
        counterfactual["reverse_time"]["correct"],
        counterfactual["sample_roll"]["correct"],
    )
    if gate["passed"]:
        conclusion = (
            "H1 supports motion-conditioned spatial pooling before global spatial "
            "pooling; H2 determines whether the gain transfers beyond the source users."
        )
    elif paired["delta_correct"] <= 0 and not aligned_beats_mismatch:
        conclusion = (
            "Retaining a 5x5 layer4 grid does not rescue P93: the aligned candidate "
            "fails to improve P86 and does not beat mismatched motion. Early global "
            "spatial pooling is therefore not supported as the main v2/v3 failure cause "
            "under this frozen-backbone, small-data protocol."
        )
    else:
        conclusion = (
            "The spatial path shows partial signal but misses the pre-registered H1 "
            "robustness gate, so there is not enough transferable evidence to advance."
        )

    return {
        "stage": "P93_spatial_cross_attention_v4_final_mechanism_audit",
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
            "zero_maximum_logit_error_vs_p86": zero_error,
        },
        "spatial_attention": {
            "grid": int(summary["protocol"]["spatial_grid"]),
            "part_uniform_entropy": math.log(5),
            "pool_uniform_entropy": math.log(
                int(summary["protocol"]["spatial_grid"]) ** 2
            ),
            **{field: paired[field] for field in paired if "spatial" in field},
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
        "harm_pairs": named_pairs(
            Counter(
                (row["label"], row["p93_prediction"])
                for row in rows
                if row["transition"] == "harm"
            )
        ),
        "rescue_pairs": named_pairs(
            Counter(
                (row["label"], row["p86_prediction"])
                for row in rows
                if row["transition"] == "rescue"
            )
        ),
        "integrity_checks": checks,
        "all_integrity_checks_passed": all(checks.values()),
        "mechanism_conclusion": conclusion,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    paired = audit["paired_result"]
    cf = audit["counterfactual_specificity"]
    spatial = audit["spatial_attention"]
    small = audit["fixed_small_action_subset"]
    lines = [
        "# P93-v4 final mechanism audit",
        "",
        f"- Decision: `{audit['decision']}`.",
        f"- H1: P86 `{paired['p86_correct']}/{audit['sample_universe']['total']}` vs P93-v4 `{paired['p93_correct']}/{audit['sample_universe']['total']}`; net `{paired['delta_correct']}` ({paired['delta_accuracy_pp']:+.3f} pp).",
        f"- Transitions: `{paired['rescue']}` rescue / `{paired['harm']}` harm; worst user `{paired['worst_user_delta_correct']}`.",
        f"- Counterfactuals: aligned `{cf['aligned_correct']}`, reverse-time `{cf['reverse_time_correct']}`, sample-roll `{cf['sample_roll_correct']}`, zero `{cf['zero_correct']}`; zero max logit error `{cf['zero_maximum_logit_error_vs_p86']}`.",
        f"- Spatial pooling: entropy `{spatial['mean_spatial_pool_attention_entropy']:.4f}` vs uniform `{spatial['pool_uniform_entropy']:.4f}`; L1 from uniform `{spatial['mean_spatial_pool_weight_l1_from_uniform']:.4f}`; clip effect/RMS `{spatial['mean_spatial_clip_effect_rms_ratio']:.4f}`.",
        f"- Fixed-small-action subset: `{small['p86_correct']}` -> `{small['p93_correct']}` (net `{small['delta_correct']}`).",
        f"- All recomputation/protocol checks passed: `{audit['all_integrity_checks_passed']}`.",
        "",
        audit["mechanism_conclusion"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    args = parser.parse_args()
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
                    "spatial_attention",
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
