from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = PROJECT_DIR / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 Skeleton/Depth/IR 独立专家错误互补性")
    parser.add_argument(
        "--skeleton",
        type=Path,
        default=DEFAULT_RUNS / "skeleton_only" / "val_predictions_best_accuracy.csv",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=DEFAULT_RUNS / "depth_imagenet" / "val_predictions_best_accuracy.csv",
    )
    parser.add_argument(
        "--ir",
        type=Path,
        default=DEFAULT_RUNS / "ir_imagenet" / "val_predictions_best_accuracy.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_RUNS / "expert_complementarity.json"
    )
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, tuple[int, int]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        row["sample_id"]: (int(row["label"]), int(row["prediction"])) for row in rows
    }
    if len(result) != len(rows):
        raise ValueError(f"预测中存在重复 sample_id：{path}")
    return result


def main() -> None:
    args = parse_args()
    predictions = {
        "skeleton": load_predictions(args.skeleton),
        "depth": load_predictions(args.depth),
        "ir": load_predictions(args.ir),
    }
    sample_ids = sorted(set.intersection(*(set(values) for values in predictions.values())))
    if any(len(values) != len(sample_ids) for values in predictions.values()):
        raise ValueError("三个专家的 sample_id 集合不一致")
    labels = {predictions["skeleton"][sample_id][0] for sample_id in sample_ids}
    if not labels.issubset(set(range(40))):
        raise ValueError("标签超出 0..39")

    patterns: dict[str, int] = defaultdict(int)
    per_class: dict[int, dict[str, int]] = {
        class_id: defaultdict(int) for class_id in range(40)
    }
    records: list[dict[str, object]] = []
    for sample_id in sample_ids:
        label = predictions["skeleton"][sample_id][0]
        if any(predictions[name][sample_id][0] != label for name in predictions):
            raise ValueError(f"标签不一致：{sample_id}")
        correct = {
            name: prediction[sample_id][1] == label for name, prediction in predictions.items()
        }
        pattern = "".join(name[0].upper() for name in ("skeleton", "depth", "ir") if correct[name])
        pattern = pattern or "none"
        patterns[pattern] += 1
        per_class[label][pattern] += 1
        per_class[label]["samples"] += 1
        if not correct["skeleton"] and not correct["depth"] and correct["ir"]:
            records.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "skeleton_prediction": predictions["skeleton"][sample_id][1],
                    "depth_prediction": predictions["depth"][sample_id][1],
                    "ir_prediction": predictions["ir"][sample_id][1],
                }
            )

    n = len(sample_ids)
    s_correct = sum(predictions["skeleton"][key][1] == predictions["skeleton"][key][0] for key in sample_ids)
    d_correct = sum(predictions["depth"][key][1] == predictions["depth"][key][0] for key in sample_ids)
    i_correct = sum(predictions["ir"][key][1] == predictions["ir"][key][0] for key in sample_ids)
    sd_oracle = sum(
        predictions["skeleton"][key][1] == predictions["skeleton"][key][0]
        or predictions["depth"][key][1] == predictions["depth"][key][0]
        for key in sample_ids
    )
    sdi_oracle = sum(
        predictions["skeleton"][key][1] == predictions["skeleton"][key][0]
        or predictions["depth"][key][1] == predictions["depth"][key][0]
        or predictions["ir"][key][1] == predictions["ir"][key][0]
        for key in sample_ids
    )
    result = {
        "samples": n,
        "accuracy": {
            "skeleton": s_correct / n,
            "depth": d_correct / n,
            "ir": i_correct / n,
        },
        "conditional_counts": {
            "skeleton_wrong_ir_right": sum(
                predictions["skeleton"][key][1] != predictions["skeleton"][key][0]
                and predictions["ir"][key][1] == predictions["ir"][key][0]
                for key in sample_ids
            ),
            "depth_wrong_ir_right": sum(
                predictions["depth"][key][1] != predictions["depth"][key][0]
                and predictions["ir"][key][1] == predictions["ir"][key][0]
                for key in sample_ids
            ),
            "skeleton_depth_wrong_ir_right": len(records),
        },
        "oracle": {
            "skeleton_depth": sd_oracle / n,
            "skeleton_depth_ir": sdi_oracle / n,
            "ir_unique_upper_bound_gain_pp": 100.0 * (sdi_oracle - sd_oracle) / n,
            "note": "Oracle uses labels and is only an upper bound, never an inference method.",
        },
        "correctness_patterns": dict(sorted(patterns.items())),
        "per_class": {str(key): dict(value) for key, value in per_class.items()},
        "ir_unique_corrections": records,
        "sources": {name: str(path.resolve()) for name, path in vars(args).items() if name != "output"},
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
