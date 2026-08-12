from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_MODULE_DIR = PROJECT_ROOT / "scripts/experiments/pose_skeleton_matching_audit"
sys.path.insert(0, str(AUDIT_MODULE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_core import COMMON_YOLO_INDICES, map_skeleton_to_yolo_order, normalize_pose  # noqa: E402
from audit_multi_person_skeleton_identity import candidate_rmse, load_people, normalize_candidate, normalize_yolo  # noqa: E402
from build_clean_skeleton_frame_index import assign_retained_segments  # noqa: E402


DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build six strict-provenance Skeleton clean views.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--oof-folds", type=Path, default=PROJECT_ROOT / "metadata/splits/train14_oof_3fold.json")
    parser.add_argument("--retained-index", type=Path, default=PROJECT_ROOT / "reports/skeleton_clean_frame_index/skeleton_retained_frame_index.csv")
    parser.add_argument("--ambiguous-index", type=Path, default=PROJECT_ROOT / "reports/skeleton_clean_frame_index/skeleton_ambiguous_frame_index.csv")
    parser.add_argument("--pose-cache", type=Path, default=PROJECT_ROOT / "outputs/depth_ir_pose_roi_40class_probe/pose_tracks.npz")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/skeleton_strict_oof_clean_views")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--ridge", type=float, default=0.001)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_factual_index(retained_path: Path, ambiguous_path: Path) -> pd.DataFrame:
    columns = [
        "sample_id", "class_id", "action_name", "user_id", "trial_id", "fold_split",
        "frame_id", "timestamp", "frame_key", "skeleton_json_path",
        "duplicate_json_files_for_frame", "people_count",
    ]
    frames = [pd.read_csv(path, encoding="utf-8-sig", dtype={"user_id": str})[columns] for path in (retained_path, ambiguous_path)]
    factual = pd.concat(frames, ignore_index=True).sort_values(["sample_id", "frame_id"]).reset_index(drop=True)
    if factual.duplicated(["sample_id", "frame_id"]).any():
        raise ValueError("Master factual index contains duplicate sample_id/frame_id keys")
    return factual


def load_pose_cache(path: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    with np.load(path) as cache:
        sample_ids = cache["sample_ids"].astype(str)
        frame_keys = cache["frame_keys"].astype(str)
        xy = cache["keypoints_xy"].astype(np.float64)
        confidence = cache["keypoints_confidence"].astype(np.float64)
    for index, (sample_id, frame_key) in enumerate(zip(sample_ids, frame_keys, strict=True)):
        lookup[(sample_id, frame_key)] = (xy[index], confidence[index])
    return lookup


def fit_projection(
    factual: pd.DataFrame, fit_users: set[str], pose_lookup: dict, data_root: Path,
    confidence_threshold: float, ridge: float,
) -> tuple[np.ndarray, dict[str, int]]:
    gram = np.zeros((3, 3), dtype=np.float64)
    cross = np.zeros((3, 2), dtype=np.float64)
    frames_used = 0
    correspondences = 0
    candidates = factual[(factual["user_id"].isin(fit_users)) & (factual["people_count"] == 1)]
    for row in candidates.itertuples():
        cached = pose_lookup.get((str(row.sample_id), str(row.frame_key)))
        if cached is None:
            continue
        people = load_people(data_root / str(row.skeleton_json_path))
        if len(people) != 1:
            continue
        candidate = normalize_candidate(people[0])
        if candidate is None:
            continue
        observed, mask, _, _ = normalize_yolo(cached[0], cached[1], confidence_threshold)
        valid = mask & np.isfinite(candidate).all(axis=1)
        if valid.sum() < 6:
            continue
        source = candidate[valid]
        gram += source.T @ source
        cross += source.T @ observed[valid]
        frames_used += 1
        correspondences += int(valid.sum())
    if correspondences < 100:
        raise ValueError(f"Only {correspondences} calibration correspondences for users {sorted(fit_users)}")
    projection = np.linalg.solve(gram + ridge * np.eye(3), cross)
    return projection, {"calibration_frames": frames_used, "calibration_correspondences": correspondences}


def build_view(
    factual: pd.DataFrame, target_users: set[str], fit_users: set[str], validation_users: set[str],
    pose_lookup: dict, data_root: Path, projection: np.ndarray,
    confidence_threshold: float, margin_threshold: float,
) -> pd.DataFrame:
    if fit_users & validation_users:
        raise ValueError("Projection fit users overlap validation users")
    view = factual[factual["user_id"].isin(target_users)].copy()
    output_rows = []
    for row in view.itertuples(index=False):
        people_count = int(row.people_count)
        candidate_index: int | None = None
        margin: float | None = None
        status = "ambiguous"
        reason = "multi_person_visual_decision_unavailable"
        valid_joints: int | None = None
        if people_count == 1:
            candidate_index = 0
            status = "retained"
            reason = "single_person"
        elif people_count > 1:
            cached = pose_lookup.get((str(row.sample_id), str(row.frame_key)))
            if cached is not None:
                observed, observed_mask, _, _ = normalize_yolo(cached[0], cached[1], confidence_threshold)
                people = load_people(data_root / str(row.skeleton_json_path))
                candidates = [normalize_candidate(person) for person in people]
                if len(candidates) == people_count and all(candidate is not None for candidate in candidates):
                    results = [candidate_rmse(observed, observed_mask, candidate, projection) for candidate in candidates]
                    rmses = np.asarray([item[0] for item in results], dtype=np.float64)
                    if np.isfinite(rmses).all():
                        order = np.argsort(rmses)
                        best, second = int(order[0]), int(order[1])
                        margin = float((rmses[second] - rmses[best]) / max(rmses[second], 1e-12))
                        valid_joints = int(results[best][1])
                        if margin >= margin_threshold:
                            candidate_index = best
                            status = "retained"
                            reason = "multi_person_visual_margin_ge_20pct"
                        else:
                            reason = "multi_person_visual_margin_lt_20pct"
        record = row._asdict()
        record.update({
            "candidate_index": candidate_index, "selection_margin": margin,
            "selection_reason": reason, "clean_status": status,
            "use_for_frame_training": status == "retained", "valid_common_joints": valid_joints,
            "projection_fit_user": str(row.user_id) in fit_users,
            "scope_validation_user": str(row.user_id) in validation_users,
        })
        output_rows.append(record)
    result = pd.DataFrame(output_rows)
    result["retained_segment_index"] = assign_retained_segments(result)
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = json.loads(args.oof_folds.read_text(encoding="utf-8"))
    factual = load_factual_index(args.retained_index, args.ambiguous_index)
    pose_lookup = load_pose_cache(args.pose_cache)
    summaries = []
    for fold in oof["folds"]:
        fold_index = int(fold["fold"])
        scope_specs = {
            "inner_selection": (
                set(fold["epoch_selection"]["fit_user_ids"]),
                set(fold["epoch_selection"]["validation_user_ids"]),
            ),
            "formal_outer": (set(fold["train_user_ids"]), set(fold["validation_user_ids"])),
        }
        for scope, (fit_users, validation_users) in scope_specs.items():
            projection, calibration = fit_projection(
                factual, fit_users, pose_lookup, args.data_root,
                args.confidence_threshold, args.ridge,
            )
            view = build_view(
                factual, fit_users | validation_users, fit_users, validation_users,
                pose_lookup, args.data_root, projection,
                args.confidence_threshold, args.margin_threshold,
            )
            scope_dir = output_dir / f"fold_{fold_index}" / scope
            scope_dir.mkdir(parents=True, exist_ok=True)
            index_path = scope_dir / "clean_view.csv"
            view.to_csv(index_path, index=False, encoding="utf-8-sig")
            retained = view["use_for_frame_training"]
            summary = {
                "fold": fold_index, "scope": scope,
                "fit_user_ids": sorted(fit_users), "validation_user_ids": sorted(validation_users),
                "fit_validation_overlap": sorted(fit_users & validation_users),
                "projection_matrix": projection.tolist(), **calibration,
                "unique_frames": len(view), "retained_frames": int(retained.sum()),
                "ambiguous_frames": int((~retained).sum()),
                "single_person_retained": int((view["selection_reason"] == "single_person").sum()),
                "confident_multi_person_retained": int((view["selection_reason"] == "multi_person_visual_margin_ge_20pct").sum()),
                "low_margin_ambiguous": int((view["selection_reason"] == "multi_person_visual_margin_lt_20pct").sum()),
                "unavailable_decision_ambiguous": int((view["selection_reason"] == "multi_person_visual_decision_unavailable").sum()),
                "retained_trials": int(view.loc[retained, "sample_id"].nunique()),
                "empty_trials": int(view["sample_id"].nunique() - view.loc[retained, "sample_id"].nunique()),
                "retained_segments": int(view.loc[retained].groupby(["sample_id", "retained_segment_index"]).ngroups),
                "margin_threshold": args.margin_threshold,
                "confidence_threshold": args.confidence_threshold,
                "ridge": args.ridge,
                "oof_assignment_sha256": sha256(args.oof_folds),
                "master_retained_sha256": sha256(args.retained_index),
                "master_ambiguous_sha256": sha256(args.ambiguous_index),
                "clean_view_sha256": sha256(index_path),
            }
            (scope_dir / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            summaries.append(summary)
            print(json.dumps({key: summary[key] for key in (
                "fold", "scope", "calibration_frames", "retained_frames", "ambiguous_frames",
                "confident_multi_person_retained", "empty_trials",
            )}))
    pd.DataFrame(summaries).drop(columns=["projection_matrix", "fit_user_ids", "validation_user_ids", "fit_validation_overlap"]).to_csv(
        output_dir / "strict_clean_view_summary.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
