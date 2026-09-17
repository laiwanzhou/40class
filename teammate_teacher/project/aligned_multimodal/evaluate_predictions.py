from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_QUALITY = PROJECT_DIR / "data" / "skeleton_quality.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算整体、subject 和 Skeleton 风险子集指标")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skeleton-quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    return {
        "samples": len(rows),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    predictions_path = args.predictions.resolve()
    output = args.output.resolve() if args.output else predictions_path.with_name("diagnostics.json")
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    quality: dict[str, dict[str, str]] = {}
    if args.skeleton_quality.resolve().is_file():
        with args.skeleton_quality.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
            quality = {row["sample_id"]: row for row in csv.DictReader(handle)}

    enriched: list[dict[str, object]] = []
    for row in predictions:
        sample_id = row["sample_id"]
        if sample_id not in manifest:
            raise KeyError(f"预测样本不在 manifest：{sample_id}")
        item: dict[str, object] = {
            "sample_id": sample_id,
            "label": int(row["label"]),
            "prediction": int(row["prediction"]),
            "user_id": manifest[sample_id]["user_id"],
        }
        item.update(quality.get(sample_id, {}))
        enriched.append(item)

    per_user: dict[str, object] = {}
    by_user: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in enriched:
        by_user[str(row["user_id"])].append(row)
    for user_id in sorted(by_user, key=lambda value: int(value[4:])):
        per_user[user_id] = metrics(by_user[user_id])

    risk_groups: dict[str, object] = {}
    if quality:
        selectors = {
            "no_multi": lambda row: int(row["multi_frames"]) == 0,
            "any_multi": lambda row: int(row["multi_frames"]) > 0,
            "majority_multi": lambda row: int(row["majority_multi"]) == 1,
            "switch_candidate": lambda row: int(row["switch_candidates"]) > 0,
            "tracking_changes_input": lambda row: int(row["tracking_changes_input"]) == 1,
        }
        for name, selector in selectors.items():
            selected = [row for row in enriched if selector(row)]
            if selected:
                risk_groups[name] = metrics(selected)

    confusions = Counter(
        (int(row["label"]), int(row["prediction"]))
        for row in enriched
        if int(row["label"]) != int(row["prediction"])
    )
    summary = {
        "predictions": str(predictions_path),
        "overall": metrics(enriched),
        "per_user": per_user,
        "skeleton_risk_groups": risk_groups,
        "top_confusions": [
            {"label": label, "prediction": prediction, "count": count}
            for (label, prediction), count in confusions.most_common(20)
        ],
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
