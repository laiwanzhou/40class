from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.hierarchical_midfusion_config import (
    load_midfusion_config,
    project_path,
)


COMMON_JOINT_MAP = {
    5: 11, 6: 14, 7: 12, 8: 15, 9: 13, 10: 16,
    11: 4, 12: 1, 13: 5, 14: 2, 15: 6, 16: 3,
}
COMMON_YOLO_INDICES = np.asarray(sorted(COMMON_JOINT_MAP), dtype=np.int64)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def projection_scope_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    scopes = [
        {
            "scope": f"fold_{int(fold['fold'])}",
            "fold": int(fold["fold"]),
            "fold_fit_user_ids": list(fold["fit_user_ids"]),
            "projection_fit_user_ids": list(fold["fit_user_ids"]),
            "scope_validation_user_ids": list(fold["validation_user_ids"]),
        }
        for fold in config.get("grouped_folds", [])
    ]
    train_users = sorted(str(value) for value in config["population"]["train_user_ids"])
    validation_users = [
        str(value) for value in config["population"]["validation_user_ids"]
    ]
    scopes.append(
        {
            "scope": "selected_final",
            "fold": -1,
            "fold_fit_user_ids": train_users,
            "projection_fit_user_ids": train_users,
            "scope_validation_user_ids": validation_users,
        }
    )
    return scopes


def _git_blob(commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _load_factual(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    data = config["data"]
    commit = str(data["skeleton_source_commit"])
    blobs = {
        "retained": _git_blob(commit, str(data["skeleton_retained_blob"])),
        "ambiguous": _git_blob(commit, str(data["skeleton_ambiguous_blob"])),
    }
    columns = [
        "sample_id", "class_id", "action_name", "user_id", "trial_id",
        "fold_split", "frame_id", "timestamp", "frame_key",
        "skeleton_json_path", "duplicate_json_files_for_frame", "people_count",
    ]
    frames = [
        pd.read_csv(BytesIO(value), encoding="utf-8-sig", dtype={"user_id": str})[
            columns
        ]
        for value in blobs.values()
    ]
    factual = pd.concat(frames, ignore_index=True)
    factual = factual.sort_values(["sample_id", "frame_id"]).reset_index(drop=True)
    if factual.duplicated(["sample_id", "frame_id"]).any():
        raise ValueError("Skeleton factual index contains duplicate sample/frame keys")
    return factual, {name: sha256_bytes(value) for name, value in blobs.items()}


def _load_pose_cache(path: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as cache:
        sample_ids = cache["sample_ids"].astype(str)
        frame_keys = cache["frame_keys"].astype(str)
        xy = cache["keypoints_xy"].astype(np.float64)
        confidence = cache["keypoints_confidence"].astype(np.float64)
    for index, key in enumerate(zip(sample_ids, frame_keys, strict=True)):
        lookup[key] = (xy[index], confidence[index])
    return lookup


def _map_skeleton(points: np.ndarray) -> np.ndarray:
    result = np.full((17, 3), np.nan, dtype=np.float64)
    for yolo_index, skeleton_index in COMMON_JOINT_MAP.items():
        result[yolo_index] = points[skeleton_index]
    return result


def _normalize_pose(
    points: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    required = np.asarray([5, 6, 11, 12])
    valid_frame = mask[required].all()
    root = (points[11] + points[12]) * 0.5
    shoulders = (points[5] + points[6]) * 0.5
    scale = np.linalg.norm(shoulders - root) + 0.5 * np.linalg.norm(points[5] - points[6])
    valid_frame = bool(valid_frame and np.isfinite(root).all() and np.isfinite(scale) and scale > 1e-6)
    normalized = (points - root[None]) / max(float(scale), 1e-6)
    normalized_mask = mask & valid_frame & np.isfinite(normalized).all(axis=1)
    normalized[~normalized_mask] = np.nan
    return normalized, normalized_mask


def _load_people(path: Path) -> list[np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: list[np.ndarray] = []
    if not isinstance(payload, list):
        return result
    for person in payload:
        if not isinstance(person, dict):
            continue
        points = np.asarray(person.get("keypoints"), dtype=np.float64)
        if points.shape == (17, 3) and np.isfinite(points).all():
            result.append(_map_skeleton(points))
    return result


def _normalized_candidate(points: np.ndarray) -> np.ndarray | None:
    normalized, mask = _normalize_pose(points, np.isfinite(points).all(axis=1))
    return normalized if mask[COMMON_YOLO_INDICES].sum() >= 6 else None


def _normalized_yolo(
    xy: np.ndarray, confidence: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    common = np.zeros(17, dtype=bool)
    common[COMMON_YOLO_INDICES] = True
    mask = (confidence >= threshold) & np.isfinite(xy).all(axis=1) & common
    return _normalize_pose(xy, mask)


def _fit_projection(
    factual: pd.DataFrame,
    fit_users: set[str],
    pose_lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    data_root: Path,
    *,
    confidence_threshold: float,
    ridge: float,
) -> tuple[np.ndarray, int]:
    gram = np.zeros((3, 3), dtype=np.float64)
    cross = np.zeros((3, 2), dtype=np.float64)
    correspondences = 0
    rows = factual[factual["user_id"].isin(fit_users) & factual["people_count"].eq(1)]
    for row in rows.itertuples():
        cached = pose_lookup.get((str(row.sample_id), str(row.frame_key)))
        if cached is None:
            continue
        people = _load_people(data_root / str(row.skeleton_json_path))
        if len(people) != 1:
            continue
        candidate = _normalized_candidate(people[0])
        if candidate is None:
            continue
        observed, mask = _normalized_yolo(cached[0], cached[1], confidence_threshold)
        valid = mask & np.isfinite(candidate).all(axis=1)
        if valid.sum() < 6:
            continue
        source = candidate[valid]
        gram += source.T @ source
        cross += source.T @ observed[valid]
        correspondences += int(valid.sum())
    if correspondences < 100:
        raise ValueError(f"only {correspondences} Skeleton calibration correspondences")
    return np.linalg.solve(gram + ridge * np.eye(3), cross), correspondences


def _candidate_rmse(
    observed: np.ndarray,
    observed_mask: np.ndarray,
    candidate: np.ndarray,
    projection: np.ndarray,
) -> float:
    projected = candidate @ projection
    valid = observed_mask & np.isfinite(candidate).all(axis=1) & np.isfinite(projected).all(axis=1)
    if valid.sum() < 6:
        return float("nan")
    error = np.linalg.norm(observed[valid] - projected[valid], axis=1)
    return float(np.sqrt(np.mean(np.square(error))))


def _assign_segments(view: pd.DataFrame) -> pd.Series:
    segments = pd.Series(pd.NA, index=view.index, dtype="Int64")
    retained = view[view["use_for_frame_training"].astype(bool)]
    for _, group in retained.groupby("sample_id", sort=False):
        ordered = group.sort_values("frame_id")
        values = ordered["frame_id"].diff().ne(1).cumsum().astype("Int64") - 1
        segments.loc[ordered.index] = values.to_numpy()
    return segments


def _build_view(
    factual: pd.DataFrame,
    target_users: set[str],
    validation_users: set[str],
    pose_lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    data_root: Path,
    projection: np.ndarray,
    *,
    confidence_threshold: float,
    margin_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in factual[factual["user_id"].isin(target_users)].itertuples(index=False):
        candidate_index: int | None = None
        margin: float | None = None
        reason = "multi_person_visual_decision_unavailable"
        if int(row.people_count) == 1:
            candidate_index, reason = 0, "single_person"
        elif int(row.people_count) > 1:
            cached = pose_lookup.get((str(row.sample_id), str(row.frame_key)))
            people = _load_people(data_root / str(row.skeleton_json_path))
            candidates = [_normalized_candidate(person) for person in people]
            if cached is not None and candidates and all(value is not None for value in candidates):
                observed, mask = _normalized_yolo(cached[0], cached[1], confidence_threshold)
                rmses = np.asarray(
                    [_candidate_rmse(observed, mask, value, projection) for value in candidates]
                )
                if np.isfinite(rmses).all() and len(rmses) >= 2:
                    order = np.argsort(rmses)
                    best, second = int(order[0]), int(order[1])
                    margin = float((rmses[second] - rmses[best]) / max(rmses[second], 1e-12))
                    if margin >= margin_threshold:
                        candidate_index = best
                        reason = "multi_person_visual_margin_ge_20pct"
                    else:
                        reason = "multi_person_visual_margin_lt_20pct"
        record = row._asdict()
        record.update(
            {
                "candidate_index": candidate_index,
                "selection_margin": margin,
                "selection_reason": reason,
                "use_for_frame_training": candidate_index is not None,
                "scope_validation_user": str(row.user_id) in validation_users,
            }
        )
        rows.append(record)
    result = pd.DataFrame(rows)
    result["retained_segment_index"] = _assign_segments(result)
    return result


def build_midfusion_clean_views(
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> list[dict[str, Any]]:
    config = load_midfusion_config(config_path)
    data = config["data"]
    formal_output = output_root is None
    output_root = (
        output_root.resolve()
        if output_root is not None
        else project_path(str(data["skeleton_clean_views"])).resolve()
    )
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    factual, source_hashes = _load_factual(config)
    pose_path = Path(str(data["pose_cache"]))
    pose_lookup = _load_pose_cache(pose_path)
    data_root = Path(str(data["root"]))
    reports: list[dict[str, Any]] = []
    for scope in projection_scope_specs(config):
        fit_users = set(scope["projection_fit_user_ids"])
        validation_users = set(scope["scope_validation_user_ids"])
        if fit_users & validation_users:
            raise ValueError("Skeleton projection fit users overlap validation users")
        projection, correspondences = _fit_projection(
            factual,
            fit_users,
            pose_lookup,
            data_root,
            confidence_threshold=0.25,
            ridge=0.001,
        )
        view = _build_view(
            factual,
            fit_users | validation_users,
            validation_users,
            pose_lookup,
            data_root,
            projection,
            confidence_threshold=0.25,
            margin_threshold=0.20,
        )
        fold_dir = output_root / str(scope["scope"])
        fold_dir.mkdir(parents=True)
        view_path = fold_dir / "clean_view.csv"
        view.to_csv(view_path, index=False, encoding="utf-8-sig")
        report = {
            "fold": scope["fold"],
            "scope": scope["scope"],
            "fold_fit_user_ids": scope["fold_fit_user_ids"],
            "projection_fit_user_ids": scope["projection_fit_user_ids"],
            "scope_validation_user_ids": scope["scope_validation_user_ids"],
            "fit_validation_overlap": sorted(fit_users & validation_users),
            "projection_matrix": projection.tolist(),
            "calibration_correspondences": correspondences,
            "retained_trials": int(view.loc[view["use_for_frame_training"], "sample_id"].nunique()),
            "empty_trials": int(
                view["sample_id"].nunique()
                - view.loc[view["use_for_frame_training"], "sample_id"].nunique()
            ),
            "clean_view_sha256": hashlib.sha256(view_path.read_bytes()).hexdigest(),
            "pose_cache_sha256": hashlib.sha256(pose_path.read_bytes()).hexdigest(),
            "source_commit": data["skeleton_source_commit"],
            "source_blob_sha256": source_hashes,
        }
        (fold_dir / "provenance.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(report)
    if formal_output:
        report_path = project_path(str(data["skeleton_clean_view_report"]))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "stage": "P5-HMF0-skeleton-clean-views",
                    "status": "completed",
                    "source_commit": data["skeleton_source_commit"],
                    "output_root": str(data["skeleton_clean_views"]),
                    "scopes": reports,
                    "validation_users_entered_projection_fit": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train12 midfusion Skeleton clean views")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    reports = build_midfusion_clean_views(
        args.config.resolve(), output_root=args.output_root
    )
    print(json.dumps({"status": "completed", "folds": len(reports)}))


if __name__ == "__main__":
    main()
