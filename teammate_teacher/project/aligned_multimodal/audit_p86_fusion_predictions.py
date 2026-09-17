from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from train_p86_visual_student_oof import metric_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit P86 feature-fusion predictions against the immutable cached "
            "Visual anchor on the same samples."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"prediction file is empty: {path}")
    return rows


def index_rows(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    indexed = {row["sample_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"{name} contains duplicate sample IDs")
    return indexed


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped(
    records: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record[field])].append(record)
    rows = []
    sort_key = (lambda item: int(item[0])) if field == "class_id" else None
    for value, members in sorted(buckets.items(), key=sort_key):
        total = len(members)
        visual_correct = sum(row["visual_correct"] for row in members)
        fused_correct = sum(row["fused_correct"] for row in members)
        rescues = sum(row["rescue"] for row in members)
        harms = sum(row["harm"] for row in members)
        rows.append(
            {
                field: value,
                "samples": total,
                "visual_correct": visual_correct,
                "fused_correct": fused_correct,
                "visual_accuracy": visual_correct / total,
                "fused_accuracy": fused_correct / total,
                "delta_accuracy_pp": 100.0 * (fused_correct - visual_correct) / total,
                "rescues": rescues,
                "harms": harms,
                "net_gain": rescues - harms,
                "changed": sum(row["changed"] for row in members),
            }
        )
    return rows


def metric_delta(
    fused: dict[str, Any], visual: dict[str, Any]
) -> dict[str, Any]:
    return {
        "correct": int(fused["correct"] - visual["correct"]),
        "accuracy_pp": 100.0 * (fused["accuracy"] - visual["accuracy"]),
        "balanced_accuracy_pp": 100.0
        * (fused["balanced_accuracy"] - visual["balanced_accuracy"]),
        "macro_f1_pp": 100.0 * (fused["macro_f1"] - visual["macro_f1"]),
        "worst_subject_accuracy_pp": 100.0
        * (fused["worst_subject_accuracy"] - visual["worst_subject_accuracy"]),
    }


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    final_path = run / "final_validation_predictions.csv"
    if final_path.exists():
        fused_path = final_path
        prefix = "final_validation"
        expected = 444
    else:
        fused_path = run / "proxy_validation_predictions.csv"
        prefix = "proxy_validation"
        expected = 973
    visual_path = run / "cached_visual_baseline_predictions.csv"
    fused_rows = read_rows(fused_path)
    visual_rows = index_rows(read_rows(visual_path), "Visual baseline")
    if len(fused_rows) != expected or len(visual_rows) != expected:
        raise RuntimeError(
            f"unexpected {prefix} counts: fused={len(fused_rows)}, "
            f"visual={len(visual_rows)}, expected={expected}"
        )

    records = []
    for fused in fused_rows:
        sample_id = fused["sample_id"]
        if sample_id not in visual_rows:
            raise RuntimeError(f"Visual baseline is missing {sample_id}")
        visual = visual_rows[sample_id]
        if fused["label"] != visual["label"] or fused["user_id"] != visual["user_id"]:
            raise RuntimeError(f"row identity mismatch for {sample_id}")
        label = int(fused["label"])
        fused_prediction = int(fused["prediction"])
        visual_prediction = int(visual["prediction"])
        fused_correct = fused_prediction == label
        visual_correct = visual_prediction == label
        records.append(
            {
                "sample_id": sample_id,
                "user_id": fused["user_id"],
                "class_id": label,
                "visual_prediction": visual_prediction,
                "fused_prediction": fused_prediction,
                "visual_correct": int(visual_correct),
                "fused_correct": int(fused_correct),
                "changed": int(visual_prediction != fused_prediction),
                "rescue": int(fused_correct and not visual_correct),
                "harm": int(visual_correct and not fused_correct),
            }
        )
    if {row["sample_id"] for row in records} != set(visual_rows):
        raise RuntimeError("fused and Visual sample sets differ")

    labels = np.asarray([row["class_id"] for row in records], dtype=np.int64)
    users = [row["user_id"] for row in records]
    fused_predictions = np.asarray(
        [row["fused_prediction"] for row in records], dtype=np.int64
    )
    visual_predictions = np.asarray(
        [row["visual_prediction"] for row in records], dtype=np.int64
    )
    fused_metrics = metric_dict(labels, fused_predictions, users)
    visual_metrics = metric_dict(labels, visual_predictions, users)
    per_user = grouped(records, "user_id")
    per_class = grouped(records, "class_id")
    rescues = sum(row["rescue"] for row in records)
    harms = sum(row["harm"] for row in records)
    audit = {
        "protocol": (
            "Paired sample-level comparison against anchor logits cached directly "
            "from the independently refit Visual checkpoint."
        ),
        "split": prefix,
        "samples": len(records),
        "visual_metrics": visual_metrics,
        "fused_metrics": fused_metrics,
        "delta": metric_delta(fused_metrics, visual_metrics),
        "changed": sum(row["changed"] for row in records),
        "rescues": rescues,
        "harms": harms,
        "net_gain": rescues - harms,
        "positive_subjects": sum(row["net_gain"] > 0 for row in per_user),
        "negative_subjects": sum(row["net_gain"] < 0 for row in per_user),
        "positive_classes": sum(row["net_gain"] > 0 for row in per_class),
        "negative_classes": sum(row["net_gain"] < 0 for row in per_class),
    }
    write_rows(run / f"{prefix}_fusion_audit.csv", records)
    write_rows(run / f"{prefix}_fusion_audit_per_user.csv", per_user)
    write_rows(run / f"{prefix}_fusion_audit_per_class.csv", per_class)
    (run / f"{prefix}_fusion_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
