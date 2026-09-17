from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from aligned_data import frame_map


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "p3_multi_skeleton_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether multi-person Skeleton candidates contain enough identity/quality signal for a new encoder"
    )
    parser.add_argument("--train-manifest", type=Path, default=PROJECT_DIR / "data" / "manifest.csv")
    parser.add_argument("--test-manifest", type=Path, default=PROJECT_DIR / "data" / "test_manifest.csv")
    parser.add_argument("--train-quality", type=Path, default=PROJECT_DIR / "data" / "skeleton_quality.csv")
    parser.add_argument("--test-quality", type=Path, default=PROJECT_DIR / "data" / "test_skeleton_quality.csv")
    parser.add_argument(
        "--shift-summary",
        type=Path,
        default=PROJECT_DIR / "runs" / "p0_six_modality_audit" / "skeleton_train_test_shift.json",
    )
    parser.add_argument(
        "--tracking-summary",
        type=Path,
        default=PROJECT_DIR / "runs" / "p0_tracking" / "oof_first_vs_tracked.json",
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=PROJECT_DIR / "runs" / "p3_sd_imu_rf_full18" / "test_predictions_detailed.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_raw_people(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def keypoints(person: dict[str, Any]) -> np.ndarray | None:
    array = np.asarray(person.get("keypoints", []), dtype=np.float32)
    if array.shape != (17, 3) or not np.isfinite(array).all():
        return None
    return array


def pose_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left[:1]
    right = right - right[:1]
    left_scale = max(float(np.linalg.norm(left, axis=1).mean()), 1e-6)
    right_scale = max(float(np.linalg.norm(right, axis=1).mean()), 1e-6)
    return float(np.linalg.norm(left / left_scale - right / right_scale, axis=1).mean())


def summarize_trial(row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    skeleton_files = frame_map(Path(row["skeleton_dir"]), "skeleton")
    people_counts: list[int] = []
    count_changes = 0
    people_zero_distances: list[float] = []
    alternate_smoother_frames = 0
    score_values: list[float] = []
    root_xy_values: list[float] = []
    candidate_scales: list[float] = []
    slot_frame_counts: Counter[int] = Counter()
    previous_zero: np.ndarray | None = None
    previous_count: int | None = None
    multi_candidate_frames = 0

    for frame_id in sorted(skeleton_files):
        people = load_raw_people(skeleton_files[frame_id])
        people_counts.append(len(people))
        if previous_count is not None and len(people) != previous_count:
            count_changes += 1
        previous_count = len(people)

        poses: list[np.ndarray] = []
        for slot, person in enumerate(people):
            pose = keypoints(person)
            if pose is None:
                continue
            poses.append(pose)
            slot_frame_counts[slot] += 1
            candidate_scales.append(float(np.linalg.norm(pose - pose[:1], axis=1).mean()))
            root_xy_values.extend([abs(float(pose[0, 0])), abs(float(pose[0, 1]))])
            scores = np.asarray(person.get("keypoint_scores", []), dtype=np.float32)
            if scores.shape == (17,) and np.isfinite(scores).all():
                score_values.extend(float(value) for value in scores)

        if len(poses) >= 2:
            multi_candidate_frames += 1
        if poses:
            current_zero = poses[0]
            if previous_zero is not None:
                first_distance = pose_distance(previous_zero, current_zero)
                people_zero_distances.append(first_distance)
                if len(poses) > 1:
                    alternate_distance = min(pose_distance(previous_zero, pose) for pose in poses[1:])
                    if alternate_distance + 0.08 < first_distance and first_distance > 0.20:
                        alternate_smoother_frames += 1
            previous_zero = current_zero

    frame_count = len(people_counts)
    multi_frames = sum(count >= 2 for count in people_counts)
    people_zero_jump_frames = sum(value > 0.20 for value in people_zero_distances)
    trial = {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "class_id": row.get("class_id", ""),
        "user_id": row.get("user_id", ""),
        "frames": frame_count,
        "multi_frames": multi_frames,
        "multi_ratio": multi_frames / frame_count if frame_count else 0.0,
        "max_people": max(people_counts, default=0),
        "people_count_changes": count_changes,
        "people0_jump_frames": people_zero_jump_frames,
        "people0_jump_ratio": people_zero_jump_frames / max(len(people_zero_distances), 1),
        "alternate_smoother_frames": alternate_smoother_frames,
        "candidate_slots_observed": len(slot_frame_counts),
        "slot_frame_counts": json.dumps(dict(sorted(slot_frame_counts.items())), ensure_ascii=False),
        "mean_candidate_scale": float(np.mean(candidate_scales)) if candidate_scales else 0.0,
        "mean_keypoint_score": float(np.mean(score_values)) if score_values else 0.0,
        "score_std": float(np.std(score_values)) if score_values else 0.0,
        "max_abs_root_xy": max(root_xy_values, default=0.0),
    }
    raw = {
        "frames": frame_count,
        "multi_frames": multi_frames,
        "multi_candidate_frames": multi_candidate_frames,
        "candidates": int(sum(people_counts)),
        "score_values": score_values,
        "root_xy_values": root_xy_values,
        "candidate_scales": candidate_scales,
        "people_zero_distances": people_zero_distances,
        "alternate_smoother_frames": alternate_smoother_frames,
    }
    return trial, raw


def select_audit_rows(
    manifest: list[dict[str, str]],
    quality: list[dict[str, str]],
    split: str,
) -> list[dict[str, str]]:
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    selected_ids = [
        row["sample_id"]
        for row in quality
        if int(row["multi_frames"]) > 0
    ]
    missing = [sample_id for sample_id in selected_ids if sample_id not in manifest_by_id]
    if missing:
        raise KeyError(f"{split} quality IDs missing from manifest: {missing[:5]}")
    return [manifest_by_id[sample_id] for sample_id in selected_ids]


def aggregate_raw(parts: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([value for part in parts for value in part["score_values"]], dtype=np.float64)
    roots = np.asarray([value for part in parts for value in part["root_xy_values"]], dtype=np.float64)
    distances = np.asarray(
        [value for part in parts for value in part["people_zero_distances"]],
        dtype=np.float64,
    )
    return {
        "audited_trials": len(parts),
        "frames": int(sum(part["frames"] for part in parts)),
        "multi_frames": int(sum(part["multi_frames"] for part in parts)),
        "candidate_instances": int(sum(part["candidates"] for part in parts)),
        "keypoint_scores": {
            "values": int(scores.size),
            "min": float(scores.min()) if scores.size else None,
            "max": float(scores.max()) if scores.size else None,
            "std": float(scores.std()) if scores.size else None,
            "all_one": bool(scores.size and np.allclose(scores, 1.0)),
        },
        "root_xy": {
            "values": int(roots.size),
            "max_abs": float(np.abs(roots).max()) if roots.size else None,
            "all_zero": bool(roots.size and np.allclose(roots, 0.0)),
        },
        "people0_continuity": {
            "transitions": int(distances.size),
            "jumps_gt_0_20": int(np.sum(distances > 0.20)),
            "jump_fraction": float(np.mean(distances > 0.20)) if distances.size else None,
            "alternate_smoother_frames": int(sum(part["alternate_smoother_frames"] for part in parts)),
        },
    }


def finite_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.std() <= 1e-12 or right_array.std() <= 1e-12:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def prediction_stratum(
    name: str,
    quality_rows: list[dict[str, str]],
    predictions: dict[str, dict[str, str]],
    predicate,
) -> dict[str, Any]:
    selected = [
        predictions[row["sample_id"]]
        for row in quality_rows
        if predicate(row) and row["sample_id"] in predictions
    ]
    return {
        "name": name,
        "samples": len(selected),
        "mean_confidence": float(np.mean([float(row["confidence"]) for row in selected])) if selected else None,
        "mean_entropy": float(np.mean([float(row["entropy"]) for row in selected])) if selected else None,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = read_csv(args.train_manifest)
    test_manifest = read_csv(args.test_manifest)
    train_quality = read_csv(args.train_quality)
    test_quality = read_csv(args.test_quality)
    selected = (
        select_audit_rows(train_manifest, train_quality, "train")
        + select_audit_rows(test_manifest, test_quality, "test")
    )

    trial_rows: list[dict[str, Any]] = []
    raw_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for index, row in enumerate(selected, start=1):
        trial, raw = summarize_trial(row)
        split = str(row["split"])
        split_key = "test" if split == "test" else "train"
        trial["split"] = split_key
        trial_rows.append(trial)
        raw_by_split[split_key].append(raw)
        if index % 50 == 0 or index == len(selected):
            print(f"candidate audit {index}/{len(selected)}", flush=True)

    prediction_by_id = {
        row["sample_id"]: row
        for row in read_csv(args.test_predictions)
    }
    risky_test: list[dict[str, Any]] = []
    for row in trial_rows:
        if row["split"] != "test":
            continue
        prediction = prediction_by_id.get(str(row["sample_id"]), {})
        risky_test.append(
            {
                **row,
                "prediction": prediction.get("prediction", ""),
                "confidence": prediction.get("confidence", ""),
                "entropy": prediction.get("entropy", ""),
            }
        )
    risky_test.sort(
        key=lambda row: (
            -float(row["multi_ratio"]),
            -int(row["alternate_smoother_frames"]),
            float(row["confidence"]) if row["confidence"] != "" else 1.0,
        )
    )
    test_multi_rows = [row for row in risky_test if row["confidence"] != ""]
    confidence = [float(row["confidence"]) for row in test_multi_rows]
    entropy = [float(row["entropy"]) for row in test_multi_rows]
    jump_ratio = [float(row["people0_jump_ratio"]) for row in test_multi_rows]
    alternate_smoother = [float(row["alternate_smoother_frames"]) for row in test_multi_rows]

    shift = json.loads(args.shift_summary.resolve().read_text(encoding="utf-8"))
    tracking = json.loads(args.tracking_summary.resolve().read_text(encoding="utf-8"))
    first_accuracy = float(tracking["methods"]["first"]["pooled_oof"]["accuracy"])
    tracked_accuracy = float(tracking["methods"]["tracked"]["pooled_oof"]["accuracy"])

    summary = {
        "protocol": {
            "train_scope": "all 293 Train/OOF trials that ever contain >=2 candidates",
            "test_scope": "all 86 Test trials that ever contain >=2 candidates",
            "candidate_slot_warning": (
                "The JSON has no persistent person ID. Candidate index is only a per-frame slot, "
                "so slot-level temporal amplitude cannot be interpreted as a tracked person's motion."
            ),
            "test_label_warning": (
                "Test confidence and entropy are sensitivity signals only; no Test accuracy is inferred."
            ),
        },
        "distribution_shift": {
            "train_multi_frame_fraction": shift["train_oof_quality"]["multi_frame_fraction_weighted"],
            "test_multi_frame_fraction": shift["test_quality"]["multi_frame_fraction_weighted"],
            "ratio": shift["test_vs_train_ratios"]["weighted_multi_frame_fraction_ratio"],
            "train_ever_multi_trials": shift["train_oof_quality"]["trials_ever_multi"],
            "test_ever_multi_trials": shift["test_quality"]["trials_ever_multi"],
            "train_majority_multi_trials": shift["train_oof_quality"]["trials_majority_multi"],
            "test_majority_multi_trials": shift["test_quality"]["trials_majority_multi"],
        },
        "raw_candidate_signal": {
            "train": aggregate_raw(raw_by_split["train"]),
            "test": aggregate_raw(raw_by_split["test"]),
            "interpretation": [
                "All keypoint confidence values are constant, so confidence cannot rank candidates.",
                "The root x/y values are zero because poses are root-relative; there is no 2D image position or global 3D translation.",
                "There is no candidate identity field, therefore index changes cannot be distinguished from detector reordering.",
            ],
        },
        "labeled_oof_evidence": {
            "single_person_fusion_accuracy": shift["oof_accuracy_by_skeleton_quality"][0]["fusion_accuracy"],
            "ever_multi_fusion_accuracy": shift["oof_accuracy_by_skeleton_quality"][1]["fusion_accuracy"],
            "majority_multi_fusion_accuracy": shift["oof_accuracy_by_skeleton_quality"][2]["fusion_accuracy"],
            "people0_skeleton_accuracy": first_accuracy,
            "nearest_tracking_skeleton_accuracy": tracked_accuracy,
            "nearest_tracking_delta_pp": 100.0 * (tracked_accuracy - first_accuracy),
            "nearest_tracking_people0_advantage_ci_pp": tracking["comparisons"]["first_vs_tracked"][
                "subject_cluster_bootstrap_95_ci_pp"
            ],
        },
        "test_prediction_sensitivity": {
            "interpretation": (
                "Test has no labels. These are confidence/entropy associations only and cannot establish "
                "whether a multi-person method would improve accuracy."
            ),
            "strata": [
                prediction_stratum(
                    "single-person-only",
                    test_quality,
                    prediction_by_id,
                    lambda row: int(row["multi_frames"]) == 0,
                ),
                prediction_stratum(
                    "ever-multi",
                    test_quality,
                    prediction_by_id,
                    lambda row: int(row["multi_frames"]) > 0,
                ),
                prediction_stratum(
                    "majority-multi",
                    test_quality,
                    prediction_by_id,
                    lambda row: int(row["majority_multi"]) == 1,
                ),
            ],
            "within_ever_multi_pearson": {
                "people0_jump_ratio_vs_confidence": finite_correlation(jump_ratio, confidence),
                "people0_jump_ratio_vs_entropy": finite_correlation(jump_ratio, entropy),
                "alternate_smoother_frames_vs_confidence": finite_correlation(alternate_smoother, confidence),
                "alternate_smoother_frames_vs_entropy": finite_correlation(alternate_smoother, entropy),
            },
        },
        "decision": {
            "systematic_candidate_risk": "distribution risk is real, but mis-selection is not identified strongly enough",
            "new_multi_person_encoder": "stop_before_training",
            "reason": (
                "Test has more multi-candidate frames, but labeled OOF does not show an overall multi-person collapse; "
                "nearest tracking is worse in every fold, and the raw files lack confidence, global position and identity "
                "signals needed to supervise or validate candidate selection. Set pooling would mix subjects without a "
                "reliable way to tell which candidate performs the labeled action."
            ),
            "kept_action": (
                "Keep people[0] for the current submission, preserve the 86-trial Test risk list, and revisit only if "
                "a calibrated 2D projection/detector identity or labeled multi-person audit supplies new evidence."
            ),
        },
        "artifacts": {
            "candidate_trial_audit": str(output_dir / "candidate_trial_audit.csv"),
            "test_risk_cases": str(output_dir / "test_risk_cases.csv"),
        },
    }

    write_csv(output_dir / "candidate_trial_audit.csv", trial_rows)
    write_csv(output_dir / "test_risk_cases.csv", risky_test)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
