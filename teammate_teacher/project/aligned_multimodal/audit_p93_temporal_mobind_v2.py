"""Build the final mechanism audit for the paired P93-v2 H1 experiment.

The audit is deliberately post-hoc and read-only with respect to model outputs.  It
recomputes the paired counts from the sample-level CSV, joins the frozen taxonomy,
and makes the counterfactual and residual-strength evidence easy to review without
opening any checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "p93_temporal_mobind_v2_h1h2_v1"
DEFAULT_TAXONOMY = (
    Path(__file__).resolve().parent
    / "data"
    / "six_modality_audit"
    / "small_action_taxonomy_v1.csv"
)


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return fmean(items) if items else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_audit(run_dir: Path, taxonomy_path: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    predictions = _read_csv(run_dir / "h1_paired_predictions.csv")
    taxonomy_rows = _read_csv(taxonomy_path)
    taxonomy = {int(row["class_id"]): row for row in taxonomy_rows}

    parsed: list[dict[str, Any]] = []
    for row in predictions:
        parsed.append(
            {
                **row,
                "label": int(row["label"]),
                "p86_prediction": int(row["p86_prediction"]),
                "p93_prediction": int(row["p93_prediction"]),
                "temporal_residual_rms_ratio": float(row["temporal_residual_rms_ratio"]),
                "temporal_skeleton_part_attention_entropy": float(
                    row["temporal_skeleton_part_attention_entropy"]
                ),
                "temporal_imu_part_attention_entropy": float(
                    row["temporal_imu_part_attention_entropy"]
                ),
            }
        )

    p86_correct = sum(row["p86_prediction"] == row["label"] for row in parsed)
    p93_correct = sum(row["p93_prediction"] == row["label"] for row in parsed)
    transitions = Counter(row["transition"] for row in parsed)
    source_users = sorted({row["user_id"] for row in parsed})
    expected_users = sorted(summary["protocol"]["source_users"])
    h3_users = set(summary["protocol"]["h3_users_excluded"])

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        by_class[row["label"]].append(row)

    class_rows: list[dict[str, Any]] = []
    for class_id in sorted(by_class):
        rows = by_class[class_id]
        meta = taxonomy[class_id]
        class_p86 = sum(row["p86_prediction"] == class_id for row in rows)
        class_p93 = sum(row["p93_prediction"] == class_id for row in rows)
        class_rows.append(
            {
                "class_id": class_id,
                "action_name": meta["action_name"],
                "semantic_group": meta["semantic_group"],
                "include_fixed_small_action": int(meta["include_fixed_small_action"]),
                "total": len(rows),
                "p86_correct": class_p86,
                "p93_correct": class_p93,
                "delta_correct": class_p93 - class_p86,
                "rescue": sum(row["transition"] == "rescue" for row in rows),
                "harm": sum(row["transition"] == "harm" for row in rows),
                "mean_residual_rms_ratio": _mean(
                    row["temporal_residual_rms_ratio"] for row in rows
                ),
            }
        )

    def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
        subset_p86 = sum(row["p86_prediction"] == row["label"] for row in rows)
        subset_p93 = sum(row["p93_prediction"] == row["label"] for row in rows)
        return {
            "total": len(rows),
            "p86_correct": subset_p86,
            "p93_correct": subset_p93,
            "delta_correct": subset_p93 - subset_p86,
            "rescue": sum(row["transition"] == "rescue" for row in rows),
            "harm": sum(row["transition"] == "harm" for row in rows),
        }

    fixed_small = [
        row
        for row in parsed
        if int(taxonomy[row["label"]]["include_fixed_small_action"]) == 1
    ]
    not_fixed_small = [
        row
        for row in parsed
        if int(taxonomy[row["label"]]["include_fixed_small_action"]) == 0
    ]

    transition_stats = {}
    for transition in ("rescue", "harm", "stable_correct", "stable_wrong"):
        rows = [row for row in parsed if row["transition"] == transition]
        transition_stats[transition] = {
            "count": len(rows),
            "mean_residual_rms_ratio": _mean(
                row["temporal_residual_rms_ratio"] for row in rows
            ),
            "mean_skeleton_part_attention_entropy": _mean(
                row["temporal_skeleton_part_attention_entropy"] for row in rows
            ),
            "mean_imu_part_attention_entropy": _mean(
                row["temporal_imu_part_attention_entropy"] for row in rows
            ),
        }

    harm_pairs = Counter(
        (row["label"], row["p93_prediction"])
        for row in parsed
        if row["transition"] == "harm"
    )
    rescue_pairs = Counter(
        (row["label"], row["p86_prediction"])
        for row in parsed
        if row["transition"] == "rescue"
    )

    def named_pairs(counter: Counter[tuple[int, int]]) -> list[dict[str, Any]]:
        result = []
        for (label, prediction), count in sorted(
            counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        ):
            result.append(
                {
                    "label": label,
                    "label_name": taxonomy[label]["action_name"],
                    "prediction": prediction,
                    "prediction_name": taxonomy[prediction]["action_name"],
                    "count": count,
                }
            )
        return result

    paired_summary = summary["h1"]["paired"]
    counterfactual = summary["h1"]["counterfactual"]
    aligned_correct = paired_summary["p93"]["correct"]
    normalized_skeleton_entropy = (
        paired_summary["mean_temporal_skeleton_part_attention_entropy"] / math.log(5)
    )
    normalized_imu_entropy = (
        paired_summary["mean_temporal_imu_part_attention_entropy"] / math.log(5)
    )
    sample_ids = [row["sample_id"] for row in parsed]
    checks = {
        "unique_sample_ids": len(set(sample_ids)) == len(sample_ids),
        "source_users_match_protocol": source_users == expected_users,
        "no_h3_user_in_h1_rows": not (set(source_users) & h3_users),
        "p86_count_recomputed": p86_correct == paired_summary["p86"]["correct"],
        "p93_count_recomputed": p93_correct == paired_summary["p93"]["correct"],
        "rescue_count_recomputed": transitions["rescue"] == paired_summary["rescue"],
        "harm_count_recomputed": transitions["harm"] == paired_summary["harm"],
        "zero_counterfactual_exact": counterfactual["zero"]["maximum_logit_error_vs_p86_anchor"]
        == 0.0,
        "h2_not_run": summary["h2"] is None,
        "orchestrator_has_no_h3_path": summary["protocol"]["h3_code_path_exists"] is False,
    }

    return {
        "stage": "P93_temporal_mobind_v2_final_mechanism_audit",
        "decision": summary["decision"],
        "sample_universe": {
            "total": len(parsed),
            "source_users": source_users,
            "h3_users_excluded": sorted(h3_users),
        },
        "paired_result": {
            "p86_correct": p86_correct,
            "p93_correct": p93_correct,
            "delta_correct": p93_correct - p86_correct,
            "delta_accuracy_pp": 100.0 * (p93_correct - p86_correct) / len(parsed),
            "rescue": transitions["rescue"],
            "harm": transitions["harm"],
            "user_delta_correct": paired_summary["user_delta_correct"],
        },
        "counterfactual_specificity": {
            "aligned_correct": aligned_correct,
            "zero_correct": counterfactual["zero"]["correct"],
            "reverse_time_correct": counterfactual["reverse_time"]["correct"],
            "sample_roll_correct": counterfactual["sample_roll"]["correct"],
            "aligned_minus_reverse_time": aligned_correct
            - counterfactual["reverse_time"]["correct"],
            "aligned_minus_sample_roll": aligned_correct
            - counterfactual["sample_roll"]["correct"],
            "zero_maximum_logit_error_vs_p86": counterfactual["zero"][
                "maximum_logit_error_vs_p86_anchor"
            ],
        },
        "attention_and_residual": {
            "mean_temporal_residual_rms_ratio": paired_summary[
                "mean_temporal_residual_rms_ratio"
            ],
            "skeleton_part_attention_entropy": paired_summary[
                "mean_temporal_skeleton_part_attention_entropy"
            ],
            "skeleton_part_attention_normalized_entropy": normalized_skeleton_entropy,
            "imu_part_attention_entropy": paired_summary[
                "mean_temporal_imu_part_attention_entropy"
            ],
            "imu_part_attention_normalized_entropy": normalized_imu_entropy,
            "transition_stats": transition_stats,
        },
        "fixed_small_action_subset": summarize_subset(fixed_small),
        "other_action_subset": summarize_subset(not_fixed_small),
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
            "The v2 implementation preserves the P86 anchor exactly and keeps the temporal "
            "residual bounded, but aligned temporal evidence is worse than both reverse-time "
            "and sample-rolled evidence. The residual is also stronger on harms than rescues. "
            "This is evidence against the tested bounded pre-pooling temporal-residual route, "
            "not a claim that every possible temporal-fusion architecture is impossible."
        ),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    paired = audit["paired_result"]
    cf = audit["counterfactual_specificity"]
    ar = audit["attention_and_residual"]
    ts = ar["transition_stats"]
    small = audit["fixed_small_action_subset"]
    largest_losses = audit["largest_class_losses"][:5]

    lines = [
        "# P93-v2 final mechanism audit",
        "",
        "## Outcome",
        "",
        f"- Decision: `{audit['decision']}`.",
        (
            f"- Paired H1: P86 `{paired['p86_correct']}/{audit['sample_universe']['total']}` "
            f"vs P93-v2 `{paired['p93_correct']}/{audit['sample_universe']['total']}`; "
            f"net `{paired['delta_correct']}` ({paired['delta_accuracy_pp']:+.3f} pp)."
        ),
        f"- Transitions: `{paired['rescue']}` rescue / `{paired['harm']}` harm.",
        "- H1 failed, so H2 was not run. The orchestrator has no H3 evaluation path.",
        "",
        "## Counterfactual attribution",
        "",
        f"- Aligned: `{cf['aligned_correct']}` correct.",
        f"- Reverse time: `{cf['reverse_time_correct']}` correct (aligned minus reverse `{cf['aligned_minus_reverse_time']}`).",
        f"- Sample roll: `{cf['sample_roll_correct']}` correct (aligned minus roll `{cf['aligned_minus_sample_roll']}`).",
        (
            f"- Zero residual: `{cf['zero_correct']}` correct; maximum logit error against "
            f"P86 `{cf['zero_maximum_logit_error_vs_p86']}`."
        ),
        "",
        "Correct temporal order and correct sample correspondence provide no positive evidence: "
        "the aligned residual underperforms both counterfactuals.",
        "",
        "## Residual and attention audit",
        "",
        f"- Mean temporal residual/visual RMS: `{ar['mean_temporal_residual_rms_ratio']:.4f}`.",
        (
            "- Part-attention normalized entropy: Skeleton "
            f"`{ar['skeleton_part_attention_normalized_entropy']:.4f}`, IMU "
            f"`{ar['imu_part_attention_normalized_entropy']:.4f}` (1.0 is uniform)."
        ),
        (
            f"- Mean residual RMS ratio: rescue `{_fmt(ts['rescue']['mean_residual_rms_ratio'])}`, "
            f"harm `{_fmt(ts['harm']['mean_residual_rms_ratio'])}`, stable-correct "
            f"`{_fmt(ts['stable_correct']['mean_residual_rms_ratio'])}`, stable-wrong "
            f"`{_fmt(ts['stable_wrong']['mean_residual_rms_ratio'])}`."
        ),
        (
            "The bounded residual is not numerically exploding; its correspondence is "
            "non-specific, and it is slightly stronger on harms than rescues."
        ),
        "",
        "## Class audit",
        "",
        (
            f"- Fixed small-action subset: `{small['p86_correct']}` -> "
            f"`{small['p93_correct']}` (net `{small['delta_correct']}`, "
            f"`{small['rescue']}` rescue / `{small['harm']}` harm)."
        ),
        "- Largest class losses:",
        "",
    ]
    for row in largest_losses:
        lines.append(
            f"  - `{row['class_id']} {row['action_name']}`: {row['p86_correct']} -> "
            f"{row['p93_correct']} (net {row['delta_correct']})."
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"All recomputation/protocol checks passed: `{audit['all_integrity_checks_passed']}`.",
            "",
            "## Scope of conclusion",
            "",
            audit["mechanism_conclusion"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    audit = build_audit(run_dir, args.taxonomy.resolve())
    (run_dir / "mechanism_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(run_dir / "paired_by_class.csv", audit["class_rows"])
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
                    "fixed_small_action_subset",
                    "all_integrity_checks_passed",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
