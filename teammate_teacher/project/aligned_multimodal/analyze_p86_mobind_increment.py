from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a P86 MoBind increment.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--motion-predictions", type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_summary(
    records: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: item[0]):
        baseline_correct = sum(value["baseline_correct"] for value in values)
        candidate_correct = sum(value["candidate_correct"] for value in values)
        output.append(
            {
                field: key,
                "samples": len(values),
                "baseline_correct": baseline_correct,
                "candidate_correct": candidate_correct,
                "rescues": sum(value["rescued"] for value in values),
                "damages": sum(value["damaged"] for value in values),
                "net_correct": candidate_correct - baseline_correct,
                "accuracy_pp": 100.0
                * (candidate_correct - baseline_correct)
                / max(len(values), 1),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    baseline = {
        row["sample_id"]: row
        for row in read_rows(run / "cached_visual_baseline_predictions.csv")
    }
    candidate = {
        row["sample_id"]: row
        for row in read_rows(run / "proxy_validation_predictions.csv")
    }
    if baseline.keys() != candidate.keys():
        raise RuntimeError("baseline and candidate sample sets differ")
    motion = None
    if args.motion_predictions:
        motion = {
            row["sample_id"]: row
            for row in read_rows(args.motion_predictions.resolve())
        }
        if not baseline.keys() <= motion.keys():
            raise RuntimeError("motion predictions do not cover the fusion run")

    records = []
    for sample_id, base in baseline.items():
        current = candidate[sample_id]
        label = int(base["label"])
        baseline_prediction = int(base["prediction"])
        candidate_prediction = int(current["prediction"])
        baseline_correct = baseline_prediction == label
        candidate_correct = candidate_prediction == label
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "user_id": base["user_id"],
            "class_id": label,
            "baseline_prediction": baseline_prediction,
            "candidate_prediction": candidate_prediction,
            "baseline_correct": int(baseline_correct),
            "candidate_correct": int(candidate_correct),
            "rescued": int(not baseline_correct and candidate_correct),
            "damaged": int(baseline_correct and not candidate_correct),
            "changed": int(baseline_prediction != candidate_prediction),
            "baseline_confidence": float(base["confidence"]),
            "candidate_confidence": float(current["confidence"]),
        }
        for key in (
            "motion_prediction",
            "motion_confidence",
            "mean_motion_reliability",
            "mean_part_attention_entropy",
        ):
            if key in current:
                record[key] = current[key]
        if motion is not None:
            motion_row = motion[sample_id]
            modality_key = (
                "skeleton_prediction"
                if "skeleton_prediction" in motion_row
                else "imu_prediction"
            )
            # The pretraining file contains both predictions. Select the modality
            # matching the run name when possible.
            if "imu" in run.name.lower():
                modality_key = "imu_prediction"
            elif "skeleton" in run.name.lower():
                modality_key = "skeleton_prediction"
            motion_prediction = int(motion_row[modality_key])
            record["pretrained_motion_prediction"] = motion_prediction
            record["pretrained_motion_correct"] = int(motion_prediction == label)
            record["candidate_followed_pretrained_motion"] = int(
                candidate_prediction == motion_prediction
            )
        records.append(record)

    rescues = sum(record["rescued"] for record in records)
    damages = sum(record["damaged"] for record in records)
    summary: dict[str, Any] = {
        "samples": len(records),
        "baseline_correct": sum(record["baseline_correct"] for record in records),
        "candidate_correct": sum(record["candidate_correct"] for record in records),
        "rescues": rescues,
        "damages": damages,
        "net_correct": rescues - damages,
        "changed_predictions": sum(record["changed"] for record in records),
    }
    if motion is not None:
        opportunities = [
            record
            for record in records
            if not record["baseline_correct"] and record["pretrained_motion_correct"]
        ]
        risks = [
            record
            for record in records
            if record["baseline_correct"] and not record["pretrained_motion_correct"]
        ]
        summary["pretrained_motion_complementarity"] = {
            "rescue_opportunities": len(opportunities),
            "realized_rescues": sum(record["rescued"] for record in opportunities),
            "rescue_realization_rate": sum(record["rescued"] for record in opportunities)
            / max(len(opportunities), 1),
            "damage_risks": len(risks),
            "realized_damages": sum(record["damaged"] for record in risks),
            "damage_realization_rate": sum(record["damaged"] for record in risks)
            / max(len(risks), 1),
        }
    write_rows(run / "increment_samples.csv", records)
    write_rows(run / "increment_per_class.csv", grouped_summary(records, "class_id"))
    write_rows(run / "increment_per_user.csv", grouped_summary(records, "user_id"))
    (run / "increment_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
