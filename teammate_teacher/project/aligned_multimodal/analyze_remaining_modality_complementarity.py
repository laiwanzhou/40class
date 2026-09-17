from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "p0_six_modality_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在相同 subject-disjoint OOF 样本上审计 IR/Radar 的条件纠错上限")
    parser.add_argument("--oof-root", type=Path, default=PROJECT_DIR / "runs" / "p5_oof_fusion")
    parser.add_argument("--ir-root", type=Path, default=PROJECT_DIR / "runs" / "p8_three_expert")
    parser.add_argument("--radar-oof", type=Path, default=DEFAULT_OUTPUT_DIR / "radar_oof_predictions.csv")
    parser.add_argument("--taxonomy", type=Path, default=PROJECT_DIR / "data" / "six_modality_audit" / "small_action_taxonomy_v1.csv")
    parser.add_argument("--depth-weight", type=float, default=0.4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "remaining_modality_complementarity.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_oof(root: Path, depth_weight: float) -> dict[str, dict[str, int]]:
    records = {}
    for fold in range(3):
        data = np.load(root / f"fold_{fold}" / "logits.npz")
        skeleton = data["skeleton_logits"].argmax(1)
        depth = data["depth_logits"].argmax(1)
        fused = ((1.0 - depth_weight) * data["skeleton_logits"] + depth_weight * data["depth_logits"]).argmax(1)
        for sample_id, label, s, d, sd in zip(data["sample_ids"], data["labels"], skeleton, depth, fused):
            records[str(sample_id)] = {
                "label": int(label),
                "skeleton": int(s),
                "depth": int(d),
                "sd": int(sd),
            }
    return records


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def conditional_summary(records: list[dict[str, int]], x_name: str, small_ids: set[int]) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in records])
    predictions = {name: np.asarray([row[name] for row in records]) for name in ["skeleton", "depth", "sd", x_name]}
    correct = {name: values == labels for name, values in predictions.items()}

    def subset(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        sd_wrong = mask & ~correct["sd"]
        return {
            "samples": count,
            "metrics": {name: metrics(labels[mask], values[mask]) for name, values in predictions.items()},
            "skeleton_wrong_x_correct": int(np.sum(mask & ~correct["skeleton"] & correct[x_name])),
            "depth_wrong_x_correct": int(np.sum(mask & ~correct["depth"] & correct[x_name])),
            "sd_wrong_x_correct": int(np.sum(sd_wrong & correct[x_name])),
            "sd_errors": int(sd_wrong.sum()),
            "sd_error_coverage_by_x": float(np.sum(sd_wrong & correct[x_name]) / sd_wrong.sum()) if sd_wrong.sum() else None,
            "x_wrong_sd_correct": int(np.sum(mask & ~correct[x_name] & correct["sd"])),
            "oracle_sd_or_x_accuracy": float(np.mean(correct["sd"][mask] | correct[x_name][mask])),
            "oracle_gain_over_sd_pp": 100.0 * float(np.mean(correct["sd"][mask] | correct[x_name][mask]) - np.mean(correct["sd"][mask])),
        }

    all_mask = np.ones(len(records), dtype=bool)
    small_mask = np.isin(labels, sorted(small_ids))
    per_class = []
    for class_id in sorted(set(labels.tolist())):
        mask = labels == class_id
        sd_wrong_x_correct = int(np.sum(mask & ~correct["sd"] & correct[x_name]))
        per_class.append(
            {
                "class_id": class_id,
                "samples": int(mask.sum()),
                "sd_accuracy": float(np.mean(correct["sd"][mask])),
                f"{x_name}_accuracy": float(np.mean(correct[x_name][mask])),
                "sd_wrong_x_correct": sd_wrong_x_correct,
                "oracle_gain_pp": 100.0 * sd_wrong_x_correct / mask.sum(),
            }
        )
    return {
        "all_common_samples": subset(all_mask),
        "fixed_small_action_samples": subset(small_mask),
        "per_class": per_class,
        "interpretation": (
            "sd_wrong_x_correct and oracle gain are upper bounds, not achievable fusion gains. "
            "A modality advances only after a fixed low-capacity fusion improves labelled OOF without harming folds."
        ),
    }


def main() -> None:
    args = parse_args()
    base = load_oof(args.oof_root.resolve(), float(args.depth_weight))
    taxonomy = read_csv(args.taxonomy.resolve())
    small_ids = {int(row["class_id"]) for row in taxonomy if int(row["include_fixed_small_action"]) == 1}

    ir_predictions = {}
    for fold in range(3):
        for row in read_csv(args.ir_root.resolve() / f"fold_{fold}" / "ir.csv"):
            ir_predictions[row["sample_id"]] = int(row["prediction"])
    ir_records = []
    for sample_id, record in base.items():
        if sample_id in ir_predictions:
            ir_records.append({**record, "ir": ir_predictions[sample_id]})

    radar_rows = read_csv(args.radar_oof.resolve())
    radar_by_key = {
        (int(row["class_id"]), row["user_id"], row["trial_id"]): int(row["random_forest_prediction"])
        for row in radar_rows
    }
    radar_records = []
    for sample_id, record in base.items():
        # train__cXX__userY__trial is stable and also represented by label/user/trial in the raw Radar index.
        parts = sample_id.split("__", 3)
        if len(parts) != 4:
            continue
        user_id = parts[2]
        trial_id = parts[3]
        key = (record["label"], user_id, trial_id)
        if key in radar_by_key:
            radar_records.append({**record, "radar": radar_by_key[key]})

    summary = {
        "protocol": {
            "base": "same three subject-disjoint OOF folds and identical sample IDs",
            "depth_weight": float(args.depth_weight),
            "small_action_class_ids": sorted(small_ids),
        },
        "ir": conditional_summary(ir_records, "ir", small_ids),
        "radar": conditional_summary(radar_records, "radar", small_ids),
        "not_available": {
            "thermal": "No three-fold subject-disjoint OOF predictions yet.",
            "imu": "Owned by teammate; request OOF logits/predictions under the same fold protocol.",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {
        modality: {
            "samples": result["all_common_samples"]["samples"],
            "x_accuracy": result["all_common_samples"]["metrics"][modality]["accuracy"],
            "sd_accuracy_same_subset": result["all_common_samples"]["metrics"]["sd"]["accuracy"],
            "sd_wrong_x_correct": result["all_common_samples"]["sd_wrong_x_correct"],
            "oracle_gain_pp": result["all_common_samples"]["oracle_gain_over_sd_pp"],
            "small_oracle_gain_pp": result["fixed_small_action_samples"]["oracle_gain_over_sd_pp"],
        }
        for modality, result in [("ir", summary["ir"]), ("radar", summary["radar"])]
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
