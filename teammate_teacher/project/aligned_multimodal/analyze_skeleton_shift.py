from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "p0_six_modality_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="量化 Skeleton 多人 Train/Test 差异及其与当前预测的关系")
    parser.add_argument("--train-quality", type=Path, default=PROJECT_DIR / "data" / "skeleton_quality.csv")
    parser.add_argument("--test-quality", type=Path, default=PROJECT_DIR / "data" / "test_skeleton_quality.csv")
    parser.add_argument("--oof-root", type=Path, default=PROJECT_DIR / "runs" / "p5_oof_fusion")
    parser.add_argument("--test-sensitivity", type=Path, default=DEFAULT_OUTPUT_DIR / "sd_test_prediction_sensitivity.csv")
    parser.add_argument("--tracking-oof", type=Path, default=PROJECT_DIR / "runs" / "p0_tracking" / "oof_first_vs_tracked.json")
    parser.add_argument("--depth-weight", type=float, default=0.4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "skeleton_train_test_shift.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def quality_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    frames = np.asarray([int(row["frames"]) for row in rows])
    multi_frames = np.asarray([int(row["multi_frames"]) for row in rows])
    multi_ratios = np.asarray([float(row["multi_ratio"]) for row in rows])
    return {
        "trials": len(rows),
        "frames": int(frames.sum()),
        "multi_frames": int(multi_frames.sum()),
        "multi_frame_fraction_weighted": float(multi_frames.sum() / frames.sum()),
        "trials_ever_multi": int(np.sum(multi_frames > 0)),
        "trials_ever_multi_fraction": float(np.mean(multi_frames > 0)),
        "trials_majority_multi": int(sum(int(row["majority_multi"]) for row in rows)),
        "trials_majority_multi_fraction": float(np.mean([int(row["majority_multi"]) for row in rows])),
        "shorter_than_12_frames": int(np.sum(frames < 12)),
        "shorter_than_12_fraction": float(np.mean(frames < 12)),
        "frames_median": float(np.median(frames)),
        "frames_p10": float(np.quantile(frames, 0.1)),
        "frames_p90": float(np.quantile(frames, 0.9)),
        "multi_ratio_median_among_multi_trials": float(np.median(multi_ratios[multi_ratios > 0])) if np.any(multi_ratios > 0) else None,
        "max_people": int(max(int(row["max_people"]) for row in rows)),
    }


def load_oof(root: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ["sample_ids", "labels", "skeleton_logits", "depth_logits"]}
    for fold in range(3):
        data = np.load(root / f"fold_{fold}" / "logits.npz")
        for key in arrays:
            arrays[key].append(data[key])
    return {key: np.concatenate(values) for key, values in arrays.items()}


def oof_stratum(
    name: str,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    skeleton_predictions: np.ndarray,
    depth_predictions: np.ndarray,
    fused_predictions: np.ndarray,
    quality_by_id: dict[str, dict[str, str]],
    predicate,
) -> dict[str, Any]:
    mask = np.asarray([predicate(quality_by_id[str(sample_id)]) for sample_id in sample_ids])
    if not mask.any():
        return {"name": name, "samples": 0}
    return {
        "name": name,
        "samples": int(mask.sum()),
        "skeleton_accuracy": float(accuracy_score(labels[mask], skeleton_predictions[mask])),
        "depth_accuracy": float(accuracy_score(labels[mask], depth_predictions[mask])),
        "fusion_accuracy": float(accuracy_score(labels[mask], fused_predictions[mask])),
        "fusion_minus_skeleton_pp": 100.0 * float(
            accuracy_score(labels[mask], fused_predictions[mask])
            - accuracy_score(labels[mask], skeleton_predictions[mask])
        ),
    }


def test_stratum(
    name: str,
    rows: list[dict[str, str]],
    quality_by_id: dict[str, dict[str, str]],
    predicate,
) -> dict[str, Any]:
    selected = [row for row in rows if predicate(quality_by_id[row["sample_id"]])]
    if not selected:
        return {"name": name, "samples": 0}
    predictions = [int(row["fusion_prediction"]) for row in selected]
    return {
        "name": name,
        "samples": len(selected),
        "predicted_class_count": len(set(predictions)),
        "mean_fusion_confidence": float(np.mean([float(row["fusion_confidence"]) for row in selected])),
        "mean_fusion_entropy": float(np.mean([float(row["fusion_entropy"]) for row in selected])),
        "drop_depth_prediction_change_rate": float(np.mean([int(row["drop_depth_changes_prediction"]) for row in selected])),
        "drop_skeleton_prediction_change_rate": float(np.mean([int(row["drop_skeleton_changes_prediction"]) for row in selected])),
        "random_depth_shuffle_change_probability_mean": float(
            np.mean([float(row["random_depth_shuffle_change_probability"]) for row in selected])
        ),
    }


def main() -> None:
    args = parse_args()
    train_quality = read_csv(args.train_quality.resolve())
    test_quality = read_csv(args.test_quality.resolve())
    train_quality_by_id = {row["sample_id"]: row for row in train_quality}
    test_quality_by_id = {row["sample_id"]: row for row in test_quality}
    oof = load_oof(args.oof_root.resolve())
    missing_oof = [str(sample_id) for sample_id in oof["sample_ids"] if str(sample_id) not in train_quality_by_id]
    if missing_oof:
        raise KeyError(f"OOF samples missing skeleton quality: {missing_oof[:5]}")
    test_predictions = read_csv(args.test_sensitivity.resolve())
    missing_test = [row["sample_id"] for row in test_predictions if row["sample_id"] not in test_quality_by_id]
    if missing_test:
        raise KeyError(f"Test samples missing skeleton quality: {missing_test[:5]}")

    weight = float(args.depth_weight)
    skeleton_predictions = oof["skeleton_logits"].argmax(1)
    depth_predictions = oof["depth_logits"].argmax(1)
    fused_predictions = ((1.0 - weight) * oof["skeleton_logits"] + weight * oof["depth_logits"]).argmax(1)

    any_multi = lambda row: int(row["multi_frames"]) > 0
    majority_multi = lambda row: int(row["majority_multi"]) == 1
    single_only = lambda row: int(row["multi_frames"]) == 0

    tracking = json.loads(args.tracking_oof.resolve().read_text(encoding="utf-8"))
    summary = {
        "train_oof_quality": quality_summary(train_quality),
        "test_quality": quality_summary(test_quality),
        "test_vs_train_ratios": {
            "weighted_multi_frame_fraction_ratio": quality_summary(test_quality)["multi_frame_fraction_weighted"]
            / quality_summary(train_quality)["multi_frame_fraction_weighted"],
            "ever_multi_trial_fraction_ratio": quality_summary(test_quality)["trials_ever_multi_fraction"]
            / quality_summary(train_quality)["trials_ever_multi_fraction"],
        },
        "oof_accuracy_by_skeleton_quality": [
            oof_stratum("single-person-only", oof["sample_ids"], oof["labels"], skeleton_predictions, depth_predictions, fused_predictions, train_quality_by_id, single_only),
            oof_stratum("ever-multi", oof["sample_ids"], oof["labels"], skeleton_predictions, depth_predictions, fused_predictions, train_quality_by_id, any_multi),
            oof_stratum("majority-multi", oof["sample_ids"], oof["labels"], skeleton_predictions, depth_predictions, fused_predictions, train_quality_by_id, majority_multi),
        ],
        "test_prediction_sensitivity_by_skeleton_quality": {
            "interpretation": "无 Test 标签；仅比较置信度、熵和预测变化率，不计算或推断准确率。",
            "strata": [
                test_stratum("single-person-only", test_predictions, test_quality_by_id, single_only),
                test_stratum("ever-multi", test_predictions, test_quality_by_id, any_multi),
                test_stratum("majority-multi", test_predictions, test_quality_by_id, majority_multi),
            ],
        },
        "existing_tracking_result": {
            "first_oof_accuracy": tracking["methods"]["first"]["pooled_oof"]["accuracy"],
            "tracked_oof_accuracy": tracking["methods"]["tracked"]["pooled_oof"]["accuracy"],
            "first_minus_tracked_pp": tracking["comparisons"]["first_vs_tracked"]["delta_pp"],
            "confidence_interval_pp": tracking["comparisons"]["first_vs_tracked"]["subject_cluster_bootstrap_95_ci_pp"],
            "interpretation": (
                "多人候选风险存在，但现有 nearest temporal tracking 在三折上显著更差；"
                "不能把‘多人更多’直接等同于‘该 tracker 会修复榜单掉点’。"
            ),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
