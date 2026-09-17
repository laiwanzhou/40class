from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较两个无标签 Test 候选的预测变化；不估计准确率")
    parser.add_argument("--baseline-detailed", type=Path, required=True)
    parser.add_argument("--candidate-detailed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    baseline = read_csv(args.baseline_detailed.resolve())
    candidate = read_csv(args.candidate_detailed.resolve())
    baseline_ids = [row["sample_id"] for row in baseline]
    candidate_ids = [row["sample_id"] for row in candidate]
    if baseline_ids != candidate_ids:
        raise ValueError("Candidate sample IDs/order do not match baseline")
    base_pred = np.asarray([int(row["prediction"]) for row in baseline])
    cand_pred = np.asarray([int(row["prediction"]) for row in candidate])
    base_conf = np.asarray([float(row["confidence"]) for row in baseline])
    cand_conf = np.asarray([float(row["confidence"]) for row in candidate])
    base_entropy = np.asarray([float(row["entropy"]) for row in baseline])
    cand_entropy = np.asarray([float(row["entropy"]) for row in candidate])
    transitions = Counter(zip(base_pred.tolist(), cand_pred.tolist()))
    changed = base_pred != cand_pred
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sample_id": sample_id,
            "baseline_prediction": int(base_pred[index]),
            "candidate_prediction": int(cand_pred[index]),
            "prediction_changed": int(changed[index]),
            "baseline_confidence": float(base_conf[index]),
            "candidate_confidence": float(cand_conf[index]),
            "confidence_delta": float(cand_conf[index] - base_conf[index]),
            "baseline_entropy": float(base_entropy[index]),
            "candidate_entropy": float(cand_entropy[index]),
            "entropy_delta": float(cand_entropy[index] - base_entropy[index]),
        }
        for index, sample_id in enumerate(baseline_ids)
    ]
    with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "interpretation": "Unlabelled Test sensitivity only; no accuracy direction is inferred.",
        "samples": len(rows),
        "prediction_change_count": int(changed.sum()),
        "prediction_change_rate": float(changed.mean()),
        "mean_confidence_baseline": float(base_conf.mean()),
        "mean_confidence_candidate": float(cand_conf.mean()),
        "mean_confidence_delta": float((cand_conf - base_conf).mean()),
        "mean_entropy_baseline": float(base_entropy.mean()),
        "mean_entropy_candidate": float(cand_entropy.mean()),
        "mean_entropy_delta": float((cand_entropy - base_entropy).mean()),
        "baseline_predicted_class_count": len(set(base_pred.tolist())),
        "candidate_predicted_class_count": len(set(cand_pred.tolist())),
        "baseline_missing_classes": [class_id for class_id in range(40) if class_id not in set(base_pred.tolist())],
        "candidate_missing_classes": [class_id for class_id in range(40) if class_id not in set(cand_pred.tolist())],
        "largest_changed_transitions": [
            {"from": source, "to": target, "count": count}
            for (source, target), count in transitions.most_common()
            if source != target
        ][:20],
        "details": str(output.with_suffix(".csv").resolve()),
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
