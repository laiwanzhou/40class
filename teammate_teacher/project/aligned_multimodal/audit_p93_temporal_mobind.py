from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE / "runs/p86_mobind_fusion_separate_v1_paired"
DEFAULT_CANDIDATE = HERE / "runs/p93_temporal_mobind_separate_h3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired P86 clip vs P93 pre-pooling temporal MoBind audit."
    )
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-run", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def accuracy(rows: list[dict[str, str]]) -> float:
    return float(
        np.mean([int(row["label"]) == int(row["prediction"]) for row in rows])
    )


def main() -> None:
    args = parse_args()
    baseline_run = args.baseline_run.resolve()
    candidate_run = args.candidate_run.resolve()
    output = (args.output_dir or candidate_run).resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = read_rows(baseline_run / "proxy_validation_predictions.csv")
    candidate = read_rows(candidate_run / "proxy_validation_predictions.csv")
    baseline_by_id = {row["sample_id"]: row for row in baseline}
    candidate_by_id = {row["sample_id"]: row for row in candidate}
    if set(baseline_by_id) != set(candidate_by_id):
        raise RuntimeError("paired runs do not contain the same samples")

    transitions: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    rescue = harm = stable_correct = stable_wrong = 0
    for sample_id in sorted(baseline_by_id):
        old = baseline_by_id[sample_id]
        new = candidate_by_id[sample_id]
        if (old["label"], old["user_id"]) != (new["label"], new["user_id"]):
            raise RuntimeError(f"metadata mismatch for {sample_id}")
        label = int(old["label"])
        old_prediction = int(old["prediction"])
        new_prediction = int(new["prediction"])
        old_correct = old_prediction == label
        new_correct = new_prediction == label
        if not old_correct and new_correct:
            transition = "rescue"
            rescue += 1
        elif old_correct and not new_correct:
            transition = "harm"
            harm += 1
        elif old_correct:
            transition = "stable_correct"
            stable_correct += 1
        else:
            transition = "stable_wrong"
            stable_wrong += 1
        row: dict[str, object] = {
            "sample_id": sample_id,
            "user_id": old["user_id"],
            "label": label,
            "p86_prediction": old_prediction,
            "p93_prediction": new_prediction,
            "transition": transition,
            "p86_confidence": float(old["confidence"]),
            "p93_confidence": float(new["confidence"]),
            "p86_motion_reliability": float(old["mean_motion_reliability"]),
            "p93_motion_reliability": float(new["mean_motion_reliability"]),
        }
        transitions.append(row)
        grouped[("user", old["user_id"])].append(row)
        grouped[("class", str(label))].append(row)

    aggregates: list[dict[str, object]] = []
    for (group_type, group), rows in sorted(grouped.items()):
        old_correct = sum(
            int(row["p86_prediction"]) == int(row["label"]) for row in rows
        )
        new_correct = sum(
            int(row["p93_prediction"]) == int(row["label"]) for row in rows
        )
        aggregates.append(
            {
                "group_type": group_type,
                "group": group,
                "total": len(rows),
                "p86_correct": old_correct,
                "p93_correct": new_correct,
                "delta_correct": new_correct - old_correct,
                "p86_accuracy": old_correct / len(rows),
                "p93_accuracy": new_correct / len(rows),
                "rescue": sum(row["transition"] == "rescue" for row in rows),
                "harm": sum(row["transition"] == "harm" for row in rows),
            }
        )

    p86_accuracy = accuracy(baseline)
    p93_accuracy = accuracy(candidate)
    p86_reliability = float(
        np.mean([float(row["mean_motion_reliability"]) for row in baseline])
    )
    p93_reliability = float(
        np.mean([float(row["mean_motion_reliability"]) for row in candidate])
    )
    candidate_reliability = np.asarray(
        [float(row["mean_motion_reliability"]) for row in candidate],
        dtype=np.float64,
    )
    candidate_correct = np.asarray(
        [int(row["label"]) == int(row["prediction"]) for row in candidate],
        dtype=bool,
    )
    candidate_attention_entropy = np.asarray(
        [float(row["mean_part_attention_entropy"]) for row in candidate],
        dtype=np.float64,
    )
    training_history = read_rows(candidate_run / "training_history.csv")
    candidate_summary = json.loads(
        (candidate_run / "summary.json").read_text(encoding="utf-8")
    )
    reliability_weight = float(candidate_summary["config"]["reliability_weight"])
    gate_percentiles = {
        str(percentile): float(np.percentile(candidate_reliability, percentile))
        for percentile in (0, 1, 10, 50, 90, 99, 100)
    }
    user_rows = [row for row in aggregates if row["group_type"] == "user"]
    summary = {
        "status": "complete",
        "protocol": (
            "Frozen paired H3/fold0 audit. No parameter, radius, family or "
            "checkpoint is selected from these labels."
        ),
        "baseline_run": str(baseline_run),
        "candidate_run": str(candidate_run),
        "total": len(transitions),
        "p86_correct": sum(int(row["label"]) == int(row["prediction"]) for row in baseline),
        "p86_accuracy": p86_accuracy,
        "p93_correct": sum(int(row["label"]) == int(row["prediction"]) for row in candidate),
        "p93_accuracy": p93_accuracy,
        "delta_correct": sum(int(row["label"]) == int(row["prediction"]) for row in candidate)
        - sum(int(row["label"]) == int(row["prediction"]) for row in baseline),
        "delta_accuracy_pp": 100.0 * (p93_accuracy - p86_accuracy),
        "rescue": rescue,
        "harm": harm,
        "stable_correct": stable_correct,
        "stable_wrong": stable_wrong,
        "p86_mean_motion_reliability": p86_reliability,
        "p93_mean_motion_reliability": p93_reliability,
        "p93_motion_reliability_percentiles": gate_percentiles,
        "p93_correct_motion_reliability": float(
            candidate_reliability[candidate_correct].mean()
        ),
        "p93_wrong_motion_reliability": float(
            candidate_reliability[~candidate_correct].mean()
        ),
        "p93_mean_part_attention_entropy": float(
            candidate_attention_entropy.mean()
        ),
        "p93_part_attention_maximum_entropy_log10": float(np.log(10.0)),
        "p93_reliability_loss_weight": reliability_weight,
        "training_gate_first_epoch": float(
            training_history[0]["mean_motion_reliability"]
        ),
        "training_gate_last_epoch": float(
            training_history[-1]["mean_motion_reliability"]
        ),
        "training_residual_strength_first_epoch": float(
            training_history[0]["residual_strength"]
        ),
        "training_residual_strength_last_epoch": float(
            training_history[-1]["residual_strength"]
        ),
        "worst_user_delta_correct": min(int(row["delta_correct"]) for row in user_rows),
        "best_user_delta_correct": max(int(row["delta_correct"]) for row in user_rows),
        "decision": "IMPLEMENTATION_REJECTED / HYPOTHESIS_STILL_ACTIVE",
        "reason": (
            "The candidate loses five correct predictions, but it was retrained from "
            "the motion pretrain, replaced rather than preserved P86 clip-local fusion, "
            "shared temporal/global residual strength, and assigned zero loss weight "
            "to a gate that saturated. This rejects v1, not temporal fusion itself."
        ),
    }
    (output / "paired_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_rows(output / "paired_transitions.csv", transitions)
    write_rows(output / "paired_by_group.csv", aggregates)

    worst_classes = sorted(
        (row for row in aggregates if row["group_type"] == "class"),
        key=lambda row: (int(row["delta_correct"]), str(row["group"])),
    )[:8]
    best_classes = sorted(
        (row for row in aggregates if row["group_type"] == "class"),
        key=lambda row: (-int(row["delta_correct"]), str(row["group"])),
    )[:8]
    report = [
        "# P93 temporal MoBind paired audit",
        "",
        f"- P86 clip fusion: `{summary['p86_correct']}/{summary['total']} = {p86_accuracy:.6f}`",
        f"- P93 temporal fusion: `{summary['p93_correct']}/{summary['total']} = {p93_accuracy:.6f}`",
        f"- Transition: `{rescue}` rescue / `{harm}` harm / net `{rescue - harm}`.",
        f"- Mean local reliability: P86 `{p86_reliability:.4f}`, P93 `{p93_reliability:.4f}`.",
        f"- User delta range: `{summary['worst_user_delta_correct']}` to `{summary['best_user_delta_correct']}` correct.",
        f"- Gate percentile 1/50/99: `{gate_percentiles['1']:.4f}` / `{gate_percentiles['50']:.4f}` / `{gate_percentiles['99']:.4f}`.",
        f"- Gate on correct/wrong samples: `{summary['p93_correct_motion_reliability']:.4f}` / `{summary['p93_wrong_motion_reliability']:.4f}`.",
        f"- Gate loss weight: `{reliability_weight:.1f}`; train gate `{summary['training_gate_first_epoch']:.4f}` -> `{summary['training_gate_last_epoch']:.4f}`.",
        f"- Mean 10-part attention entropy: `{summary['p93_mean_part_attention_entropy']:.4f}` / max `{summary['p93_part_attention_maximum_entropy_log10']:.4f}`.",
        "- Decision: reject v1 implementation; temporal-fusion hypothesis remains active and requires a structural revision.",
        "",
        "## Most negative classes",
        "",
        "| class | total | delta | rescue | harm |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {row['group']} | {row['total']} | {row['delta_correct']} | {row['rescue']} | {row['harm']} |"
            for row in worst_classes
        ],
        "",
        "## Most positive classes",
        "",
        "| class | total | delta | rescue | harm |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {row['group']} | {row['total']} | {row['delta_correct']} | {row['rescue']} | {row['harm']} |"
            for row in best_classes
        ],
        "",
        "This is a frozen architecture audit, not a source for H3 retuning.",
    ]
    (output / "PAIRED_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
