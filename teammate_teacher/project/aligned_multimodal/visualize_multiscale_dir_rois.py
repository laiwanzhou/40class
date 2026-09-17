from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from audit_yolo11_pose_skeleton import frame_map
from build_multiscale_dir_rois import REGION_INDEX, REGION_NAMES


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
DEFAULT_POSE_RUN = PROJECT_DIR / "runs" / "p28_adaptive_ir_pose_skeleton_full"
DEFAULT_ROI_RUN = PROJECT_DIR / "runs" / "p29_dir_multiscale_roi_40class_pilot"
COLORS = {
    "full_body": (0, 255, 0),
    "left_arm": (255, 255, 0),
    "right_arm": (0, 128, 255),
    "left_hand": (255, 80, 80),
    "right_hand": (80, 80, 255),
    "hand_workspace": (0, 255, 255),
    "global_fallback": (150, 150, 150),
}
HARD_CLASSES = (1, 2, 6, 18, 20, 22, 24, 27, 33, 36, 37, 39)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render D/IR ROI geometry audit sheets.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pose-run", type=Path, default=DEFAULT_POSE_RUN)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def draw_regions(
    image: np.ndarray,
    boxes: np.ndarray,
    valid: np.ndarray,
    quality: np.ndarray,
    title: str,
    ambiguous: bool,
) -> np.ndarray:
    canvas = image.copy()
    for region_name in REGION_NAMES[:-1]:
        region = REGION_INDEX[region_name]
        if not valid[region]:
            continue
        x0, y0, x1, y1 = np.rint(boxes[region]).astype(int)
        thickness = 3 if region_name == "hand_workspace" else 2
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLORS[region_name], thickness)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 50), (16, 16, 16), -1)
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"workspaceQ={quality[REGION_INDEX['hand_workspace']]:.2f} lrAmb={int(ambiguous)}",
        (8, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return canvas


def letterbox_crop(image: np.ndarray, box: np.ndarray, title: str) -> np.ndarray:
    x0, y0, x1, y1 = np.rint(box).astype(int)
    x0 = int(np.clip(x0, 0, image.shape[1] - 1))
    x1 = int(np.clip(x1, x0 + 1, image.shape[1]))
    y0 = int(np.clip(y0, 0, image.shape[0] - 1))
    y1 = int(np.clip(y1, y0 + 1, image.shape[0]))
    crop = image[y0:y1, x0:x1]
    target_width, target_height = 640, 480
    scale = min(target_width / max(crop.shape[1], 1), target_height / max(crop.shape[0], 1))
    resized = cv2.resize(
        crop,
        (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((target_height, target_width, 3), 20, dtype=np.uint8)
    left = (target_width - resized.shape[1]) // 2
    top = (target_height - resized.shape[0]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (target_width, 32), (16, 16, 16), -1)
    cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def representative_frame(
    pose: np.lib.npyio.NpzFile, roi: np.lib.npyio.NpzFile
) -> int:
    keypoints = pose["ir_keypoints_xy_conf"]
    boxes = pose["ir_boxes_xyxy_conf"]
    workspace_quality = roi["roi_quality"][:, REGION_INDEX["hand_workspace"]]
    workspace_valid = roi["roi_valid"][:, REGION_INDEX["hand_workspace"]]
    wrist_xy = keypoints[:, (9, 10), :2]
    wrist_conf = keypoints[:, (9, 10), 2]
    speed = np.zeros(len(keypoints), dtype=np.float32)
    if len(keypoints) > 1:
        displacement = np.linalg.norm(np.diff(wrist_xy, axis=0), axis=2).mean(axis=1)
        person_height = np.maximum(boxes[1:, 3] - boxes[1:, 1], 1.0)
        speed[1:] = displacement / person_height
    speed[~np.isfinite(speed)] = 0.0
    if np.max(speed) > 0:
        speed /= np.max(speed)
    wrists_good = (np.isfinite(wrist_conf) & (wrist_conf >= 0.25)).mean(axis=1)
    score = 0.55 * speed + 0.30 * workspace_quality + 0.15 * wrists_good
    score[~workspace_valid] = -1.0
    # Avoid selecting a warm-up/end frame solely because of one noisy jump.
    if len(score) >= 10:
        score[:2] *= 0.8
        score[-2:] *= 0.8
    return int(np.argmax(score))


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    pose_run = args.pose_run.resolve()
    roi_run = args.roi_run.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else roi_run / "audit_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    summaries: dict[int, dict[str, object]] = {}
    for path in roi_run.glob("trial_summary/*/*/*.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries[int(summary["class_id"])] = summary
    if set(summaries) != set(range(40)):
        raise RuntimeError(f"Expected 40 pilot classes, got {sorted(summaries)}")

    selected_frames: dict[int, int] = {}
    caches: dict[int, tuple[Path, Path]] = {}
    for class_id, summary in summaries.items():
        sample_id = str(summary["sample_id"])
        relative = Path(*sample_id.split("/")).with_suffix(".npz")
        pose_path = pose_run / "trial_cache" / relative
        roi_path = roi_run / "trial_roi_cache" / relative
        with np.load(pose_path) as pose, np.load(roi_path) as roi:
            selected_frames[class_id] = representative_frame(pose, roi)
        caches[class_id] = (pose_path, roi_path)

    for group_start in range(0, 40, 10):
        rows: list[np.ndarray] = []
        for class_id in range(group_start, group_start + 10):
            summary = summaries[class_id]
            sample_id = str(summary["sample_id"])
            manifest_row = manifest_rows[sample_id]
            pose_path, roi_path = caches[class_id]
            frame_index = selected_frames[class_id]
            with np.load(pose_path) as pose, np.load(roi_path) as roi:
                frame_id = str(roi["frame_ids"][frame_index])
                boxes = roi["roi_boxes_xyxy"][frame_index]
                valid = roi["roi_valid"][frame_index]
                quality = roi["roi_quality"][frame_index]
                ambiguous = bool(roi["left_right_ambiguous"][frame_index])
            ir = cv2.imread(str(frame_map(Path(manifest_row["ir_path"]), "ir")[frame_id]))
            depth = cv2.imread(
                str(frame_map(Path(manifest_row["depth_color_path"]), "depth")[frame_id])
            )
            if ir is None or depth is None:
                raise RuntimeError(f"Image load failed: {sample_id} {frame_id}")
            title = f"C{class_id:02d} {summary['class_name']} frame {frame_index + 1}/{summary['frames']}"
            ir_overlay = draw_regions(ir, boxes, valid, quality, "IR | " + title, ambiguous)
            depth_overlay = draw_regions(depth, boxes, valid, quality, "Depth | same coordinates", ambiguous)
            workspace = boxes[REGION_INDEX["hand_workspace"]]
            ir_crop = letterbox_crop(ir, workspace, "IR hand workspace crop")
            depth_crop = letterbox_crop(depth, workspace, "Depth hand workspace crop")
            rows.append(np.concatenate((ir_overlay, depth_overlay, ir_crop, depth_crop), axis=1))
        sheet = np.concatenate(rows, axis=0)
        output_path = output_dir / f"roi_audit_classes_{group_start:02d}_{group_start + 9:02d}.jpg"
        if not cv2.imwrite(str(output_path), sheet, (cv2.IMWRITE_JPEG_QUALITY, 92)):
            raise RuntimeError(output_path)
        print(output_path)

    temporal_rows: list[np.ndarray] = []
    for class_id in HARD_CLASSES:
        summary = summaries[class_id]
        sample_id = str(summary["sample_id"])
        manifest_row = manifest_rows[sample_id]
        _, roi_path = caches[class_id]
        with np.load(roi_path) as roi:
            count = len(roi["frame_ids"])
            indices = np.unique(np.rint(np.linspace(0, count - 1, 4)).astype(int))
            while len(indices) < 4:
                indices = np.unique(np.append(indices, min(count - 1, indices[-1] + 1)))
            panels: list[np.ndarray] = []
            ir_map = frame_map(Path(manifest_row["ir_path"]), "ir")
            for frame_index in indices[:4]:
                frame_id = str(roi["frame_ids"][frame_index])
                ir = cv2.imread(str(ir_map[frame_id]))
                panel = draw_regions(
                    ir,
                    roi["roi_boxes_xyxy"][frame_index],
                    roi["roi_valid"][frame_index],
                    roi["roi_quality"][frame_index],
                    f"C{class_id:02d} {summary['class_name']} {frame_index + 1}/{count}",
                    bool(roi["left_right_ambiguous"][frame_index]),
                )
                panels.append(panel)
        temporal_rows.append(np.concatenate(panels, axis=1))
    temporal_sheet = np.concatenate(temporal_rows, axis=0)
    temporal_path = output_dir / "difficult_classes_four_timepoints.jpg"
    if not cv2.imwrite(str(temporal_path), temporal_sheet, (cv2.IMWRITE_JPEG_QUALITY, 92)):
        raise RuntimeError(temporal_path)
    print(temporal_path)


if __name__ == "__main__":
    main()
