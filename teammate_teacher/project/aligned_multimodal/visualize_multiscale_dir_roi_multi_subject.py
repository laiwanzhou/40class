from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from audit_yolo11_pose_skeleton import atomic_json, frame_map
from build_multiscale_dir_rois import REGION_INDEX
from visualize_multiscale_dir_rois import HARD_CLASSES, draw_regions, letterbox_crop, representative_frame


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
DEFAULT_POSE_RUN = PROJECT_DIR / "runs" / "p28_adaptive_ir_pose_skeleton_full"
DEFAULT_ROI_RUN = PROJECT_DIR / "runs" / "p29_dir_multiscale_roi_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Step-07 hard classes across subjects.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pose-run", type=Path, default=DEFAULT_POSE_RUN)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--subjects-per-class", type=int, default=3)
    return parser.parse_args()


def user_number(user_id: str) -> int:
    match = re.search(r"(\d+)$", user_id)
    return int(match.group(1)) if match else 10**9


def spread_subject_ids(subject_ids: list[str], count: int) -> list[str]:
    ordered = sorted(subject_ids, key=lambda value: (user_number(value), value))
    if len(ordered) <= count:
        return ordered
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    return [ordered[index] for index in indices]


def cache_path(root: Path, folder: str, sample_id: str) -> Path:
    return root / folder / Path(*sample_id.split("/")).with_suffix(".npz")


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    pose_run = args.pose_run.resolve()
    roi_run = args.roi_run.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else roi_run / "audit_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with (roi_run / "trial_summary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        trial_rows = list(csv.DictReader(handle))

    by_class_user: dict[int, dict[str, list[dict[str, str]]]] = {}
    for row in trial_rows:
        class_id = int(row["class_id"])
        if class_id not in HARD_CLASSES:
            continue
        by_class_user.setdefault(class_id, {}).setdefault(row["user_id"], []).append(row)

    selections: list[dict[str, object]] = []
    rendered_rows: list[np.ndarray] = []
    for class_id in HARD_CLASSES:
        user_trials = by_class_user[class_id]
        selected_users = spread_subject_ids(list(user_trials), args.subjects_per_class)
        panels: list[np.ndarray] = []
        for user_id in selected_users:
            # Longest trial is deterministic and avoids choosing by detector quality.
            trial_row = max(user_trials[user_id], key=lambda row: (int(row["frames"]), row["sample_id"]))
            sample_id = trial_row["sample_id"]
            pose_path = cache_path(pose_run, "trial_cache", sample_id)
            roi_path = cache_path(roi_run, "trial_roi_cache", sample_id)
            with np.load(pose_path, allow_pickle=False) as pose, np.load(
                roi_path, allow_pickle=False
            ) as roi:
                frame_index = representative_frame(pose, roi)
                frame_id = str(roi["frame_ids"][frame_index])
                boxes = roi["roi_boxes_xyxy"][frame_index]
                valid = roi["roi_valid"][frame_index]
                quality = roi["roi_quality"][frame_index]
                ambiguous = bool(roi["left_right_ambiguous"][frame_index])
                workspace_valid = bool(valid[REGION_INDEX["hand_workspace"]])
                workspace = boxes[REGION_INDEX["hand_workspace"]]

            manifest_row = manifest_rows[sample_id]
            ir = cv2.imread(str(frame_map(Path(manifest_row["ir_path"]), "ir")[frame_id]))
            depth = cv2.imread(
                str(frame_map(Path(manifest_row["depth_color_path"]), "depth")[frame_id])
            )
            if ir is None or depth is None:
                raise RuntimeError(f"Image load failed: {sample_id} {frame_id}")
            title = (
                f"C{class_id:02d} {trial_row['class_name']} | {user_id} | "
                f"{frame_index + 1}/{trial_row['frames']}"
            )
            panels.append(draw_regions(ir, boxes, valid, quality, title, ambiguous))
            if workspace_valid:
                panels.append(letterbox_crop(depth, workspace, f"Depth crop | same XYXY | {user_id}"))
            else:
                panels.append(letterbox_crop(depth, np.asarray((0, 0, 639, 479)), f"Depth global fallback | {user_id}"))
            selections.append(
                {
                    "class_id": class_id,
                    "class_name": trial_row["class_name"],
                    "user_id": user_id,
                    "sample_id": sample_id,
                    "frame_index_zero_based": frame_index,
                    "frame_id": frame_id,
                    "workspace_quality": float(quality[REGION_INDEX["hand_workspace"]]),
                    "workspace_valid": workspace_valid,
                    "left_right_ambiguous": ambiguous,
                }
            )
        rendered_rows.append(np.concatenate(panels, axis=1))

    for group_index in range(0, len(rendered_rows), 4):
        class_ids = HARD_CLASSES[group_index : group_index + 4]
        sheet = np.concatenate(rendered_rows[group_index : group_index + 4], axis=0)
        output_path = output_dir / f"hard_classes_multi_subject_{class_ids[0]:02d}_{class_ids[-1]:02d}.jpg"
        if not cv2.imwrite(str(output_path), sheet, (cv2.IMWRITE_JPEG_QUALITY, 92)):
            raise RuntimeError(output_path)
        print(output_path)
    atomic_json(output_dir / "hard_classes_multi_subject_selection.json", {"selections": selections})


if __name__ == "__main__":
    main()
