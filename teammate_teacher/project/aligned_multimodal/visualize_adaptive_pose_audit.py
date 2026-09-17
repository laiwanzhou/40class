from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from audit_yolo11_pose_skeleton import draw_h36m, draw_yolo, frame_map


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
DEFAULT_RUN = PROJECT_DIR / "runs" / "p28_adaptive_ir_pose_skeleton_40class_regression"
DEFAULT_OUTPUT = DEFAULT_RUN / "analysis_figures" / "adaptive_pose_before_after_audit.jpg"
DEFAULT_CLASSES = (2, 6, 18, 20, 22, 27, 31, 33, 36)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render primary/fallback/Depth/Skeleton audit rows.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--class-id", action="append", type=int, default=[])
    return parser.parse_args()


def choose_frame(cache: np.lib.npyio.NpzFile) -> int:
    primary = cache["primary_ir_keypoints_xy_conf"]
    final = cache["ir_keypoints_xy_conf"]
    selected = cache["fallback_selected"].astype(bool)
    source = cache["final_pose_source"]
    primary_wrists = (np.isfinite(primary[:, (9, 10), 2]) & (primary[:, (9, 10), 2] >= 0.25)).sum(axis=1)
    final_wrists = (np.isfinite(final[:, (9, 10), 2]) & (final[:, (9, 10), 2] >= 0.25)).sum(axis=1)
    improvement = final_wrists - primary_wrists
    candidates = np.flatnonzero(selected & (improvement > 0))
    if len(candidates):
        return int(candidates[np.argmax(improvement[candidates])])
    candidates = np.flatnonzero(selected)
    if len(candidates):
        return int(candidates[len(candidates) // 2])
    candidates = np.flatnonzero(source == 0)
    if len(candidates):
        return int(candidates[0])
    return int(np.argmin(final_wrists))


def title_panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 52), (18, 18, 18), -1)
    cv2.putText(canvas, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    class_ids = tuple(args.class_id) if args.class_id else DEFAULT_CLASSES
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    summaries: dict[int, dict[str, object]] = {}
    for path in args.run_dir.resolve().glob("trial_summary/*/*/*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        summaries[int(item["class_id"])] = item

    rows: list[np.ndarray] = []
    for class_id in class_ids:
        summary = summaries[class_id]
        sample_id = str(summary["sample_id"])
        manifest = manifest_rows[sample_id]
        cache_path = args.run_dir.resolve() / "trial_cache" / Path(*sample_id.split("/")).with_suffix(".npz")
        with np.load(cache_path) as cache:
            frame_index = choose_frame(cache)
            frame_id = str(cache["frame_ids"][frame_index])
            primary_box = cache["primary_ir_boxes_xyxy_conf"][frame_index]
            primary_keypoints = cache["primary_ir_keypoints_xy_conf"][frame_index]
            final_box = cache["ir_boxes_xyxy_conf"][frame_index]
            final_keypoints = cache["ir_keypoints_xy_conf"][frame_index]
            skeleton = cache["skeleton_h36m_xyz_conf_raw"][frame_index]
            source = int(cache["final_pose_source"][frame_index])
            reason = int(cache["retry_reason_bits"][frame_index])

        ir = cv2.imread(str(frame_map(Path(manifest["ir_path"]), "ir")[frame_id]), cv2.IMREAD_COLOR)
        depth = cv2.imread(
            str(frame_map(Path(manifest["depth_color_path"]), "depth")[frame_id]), cv2.IMREAD_COLOR
        )
        if ir is None or depth is None:
            raise RuntimeError(f"Could not load {sample_id} frame {frame_id}")
        ir = cv2.resize(ir, (640, 480), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_AREA)
        primary_canvas = draw_yolo(ir, primary_box, primary_keypoints, "primary 640/0.10")
        final_canvas = draw_yolo(ir, final_box, final_keypoints, "automatic final")
        depth_canvas = draw_yolo(depth, final_box, final_keypoints, "IR pose copied to D")
        skeleton_canvas = draw_h36m(skeleton)
        label = f"C{class_id:02d} {summary['class_name']} | frame {frame_index + 1}/{summary['common_frames']}"
        primary_canvas = title_panel(primary_canvas, label, f"retry bits={reason}")
        final_canvas = title_panel(final_canvas, "IR final", f"source={source}: 1=primary, 2=fallback")
        depth_canvas = title_panel(depth_canvas, "Depth transfer", "same frame + exact IR image coordinates")
        skeleton_canvas = title_panel(skeleton_canvas, "Dataset Skeleton", "relative 3D; semantic/time pairing only")
        rows.append(np.concatenate((primary_canvas, final_canvas, depth_canvas, skeleton_canvas), axis=1))

    output = np.concatenate(rows, axis=0)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output.resolve()), output, (cv2.IMWRITE_JPEG_QUALITY, 92)):
        raise RuntimeError(f"Could not write {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
