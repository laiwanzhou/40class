from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.roi import PoseDetection, UltralyticsPoseLocator


DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
PROBE_ACTIONS = (
    "Take_medicine",
    "Eat_food",
    "Drink_water",
    "Make_a_phone_call",
    "Take_a_selfie",
    "Take_body_temperature",
    "Watch_TV",
    "Play_games",
    "Turn_pages",
    "Stir_drinks",
    "Wipe_bowls",
    "Walk",
)
KEYPOINT_THRESHOLD = 0.25
SKELETON_EDGES = ((0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12))
DISPLAY_KEYPOINTS = (0, 5, 6, 7, 8, 9, 10, 11, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata/manifest.csv")
    parser.add_argument("--fold", type=Path, default=PROJECT_ROOT / "metadata/splits/fold_0.json")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "yolo11n-pose.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/depth_pose_roi_probe")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/depth_pose_roi_probe.md")
    parser.add_argument("--trials-per-action", type=int, default=4)
    parser.add_argument("--frames-per-trial", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def frame_key(path: Path, modality: str) -> str:
    stem = path.stem
    prefix = "Depth_" if modality == "depth" else "IR_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected {modality} filename: {path.name}")
    key = stem[len(prefix) :]
    return key.removesuffix("_Color")


def paired_frames(depth_dir: Path, ir_dir: Path) -> list[tuple[str, Path, Path]]:
    depth = {frame_key(path, "depth"): path for path in depth_dir.glob("*.png")}
    ir = {frame_key(path, "ir"): path for path in ir_dir.glob("*.png")}
    keys = sorted(depth.keys() & ir.keys())
    if not keys:
        raise ValueError(f"No paired Depth/IR frames: {depth_dir} | {ir_dir}")
    return [(key, depth[key], ir[key]) for key in keys]


def select_trials(manifest: pd.DataFrame, train_users: set[str], count: int) -> pd.DataFrame:
    eligible = manifest[
        manifest["action_name"].isin(PROBE_ACTIONS)
        & manifest["user_id"].isin(train_users)
        & manifest["depth_color_path"].fillna("").ne("")
        & manifest["ir_path"].fillna("").ne("")
    ].copy()
    selected: list[pd.Series] = []
    for action_index, action in enumerate(PROBE_ACTIONS):
        action_rows = eligible[eligible["action_name"] == action].sort_values(
            ["user_id", "trial_id", "sample_id"]
        )
        if action_rows.empty:
            raise ValueError(f"Required probe action is absent: {action}")
        users = sorted(action_rows["user_id"].unique())
        rotated = users[action_index % len(users) :] + users[: action_index % len(users)]
        chosen: list[pd.Series] = []
        for user in rotated:
            chosen.append(action_rows[action_rows["user_id"] == user].iloc[0])
            if len(chosen) == count:
                break
        if len(chosen) < count:
            used = {row["sample_id"] for row in chosen}
            for _, row in action_rows.iterrows():
                if row["sample_id"] not in used:
                    chosen.append(row)
                    if len(chosen) == count:
                        break
        if len(chosen) != count:
            raise ValueError(f"Only selected {len(chosen)} trials for {action}, expected {count}")
        selected.extend(chosen)
    result = pd.DataFrame(selected).reset_index(drop=True)
    if result["user_id"].nunique() < 8:
        raise ValueError(f"Probe only covers {result['user_id'].nunique()} users")
    return result


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detection_row(record: dict[str, object], modality: str, image: np.ndarray, detection: PoseDetection | None) -> dict[str, object]:
    height, width = image.shape[:2]
    row = {**record, "modality": modality, "image_width": width, "image_height": height}
    if detection is None:
        return {
            **row,
            "person_detected": False,
            "bbox_confidence": np.nan,
            "both_shoulders": False,
            "both_wrists": False,
            "at_least_one_wrist": False,
            "head_or_nose": False,
            "bbox_area_ratio": np.nan,
            "bbox_center_x": np.nan,
            "bbox_center_y": np.nan,
            "bbox_width": np.nan,
            "bbox_height": np.nan,
        }
    keypoint_ok = detection.keypoints_confidence >= KEYPOINT_THRESHOLD
    x1, y1, x2, y2 = detection.bbox_xyxy
    return {
        **row,
        "person_detected": True,
        "bbox_confidence": detection.bbox_confidence,
        "both_shoulders": bool(keypoint_ok[5] and keypoint_ok[6]),
        "both_wrists": bool(keypoint_ok[9] and keypoint_ok[10]),
        "at_least_one_wrist": bool(keypoint_ok[9] or keypoint_ok[10]),
        "head_or_nose": bool(keypoint_ok[0] or keypoint_ok[1:5].any()),
        "bbox_area_ratio": float(max(0, x2 - x1) * max(0, y2 - y1) / width / height),
        "bbox_center_x": float((x1 + x2) / 2 / width),
        "bbox_center_y": float((y1 + y2) / 2 / height),
        "bbox_width": float((x2 - x1) / width),
        "bbox_height": float((y2 - y1) / height),
    }


def draw_detection(image: np.ndarray, detection: PoseDetection | None, modality: str) -> Image.Image:
    output = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(output)
    if detection is None:
        draw.rectangle((0, 0, 210, 28), fill="black")
        draw.text((6, 6), f"{modality}: no person >= 0.25", fill="red")
        return output
    x1, y1, x2, y2 = detection.bbox_xyxy.tolist()
    draw.rectangle((x1, y1, x2, y2), outline="lime", width=4)
    confidence = detection.keypoints_confidence >= KEYPOINT_THRESHOLD
    for first, second in SKELETON_EDGES:
        if confidence[first] and confidence[second]:
            draw.line((*detection.keypoints_xy[first], *detection.keypoints_xy[second]), fill="cyan", width=3)
    for index in DISPLAY_KEYPOINTS:
        if confidence[index]:
            x, y = detection.keypoints_xy[index]
            radius = 5
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="yellow", outline="black")
    label = f"{modality} person={detection.bbox_confidence:.3f}"
    draw.rectangle((0, 0, 220, 28), fill="black")
    draw.text((6, 6), label, fill="white")
    return output


def save_visuals(
    output_dir: Path,
    record: dict[str, object],
    depth: np.ndarray,
    ir: np.ndarray,
    depth_detection: PoseDetection | None,
    ir_detection: PoseDetection | None,
) -> Path:
    name = f"{record['sample_id']}__{record['frame_index']:02d}.jpg"
    depth_pose = draw_detection(depth, depth_detection, "Depth")
    ir_pose = draw_detection(ir, ir_detection, "IR")
    (output_dir / "depth_overlays").mkdir(parents=True, exist_ok=True)
    (output_dir / "ir_overlays").mkdir(parents=True, exist_ok=True)
    (output_dir / "side_by_side").mkdir(parents=True, exist_ok=True)
    depth_pose.save(output_dir / "depth_overlays" / name, quality=90)
    ir_pose.save(output_dir / "ir_overlays" / name, quality=90)
    panels = [Image.fromarray(depth), depth_pose, Image.fromarray(ir), ir_pose]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    path = output_dir / "side_by_side" / name
    canvas.save(path, quality=88)
    return path


def add_jitter_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (modality, sample_id), group in frame.groupby(["modality", "sample_id"], sort=False):
        group = group.sort_values("frame_index")
        valid = group.dropna(subset=["bbox_center_x", "bbox_center_y", "bbox_width", "bbox_height"])
        if len(valid) < 2:
            rows.append({"modality": modality, "sample_id": sample_id, "raw_center_jitter": np.nan, "raw_size_jitter": np.nan, "ema_center_jitter": np.nan, "ema_size_jitter": np.nan})
            continue
        values = valid[["bbox_center_x", "bbox_center_y", "bbox_width", "bbox_height"]].to_numpy(float)
        raw_center = np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1).mean()
        raw_size = np.linalg.norm(np.diff(values[:, 2:], axis=0), axis=1).mean()
        smoothed = values.copy()
        for index in range(1, len(smoothed)):
            smoothed[index] = 0.5 * values[index] + 0.5 * smoothed[index - 1]
        ema_center = np.linalg.norm(np.diff(smoothed[:, :2], axis=0), axis=1).mean()
        ema_size = np.linalg.norm(np.diff(smoothed[:, 2:], axis=0), axis=1).mean()
        rows.append({"modality": modality, "sample_id": sample_id, "raw_center_jitter": raw_center, "raw_size_jitter": raw_size, "ema_center_jitter": ema_center, "ema_size_jitter": ema_size})
    return frame.merge(pd.DataFrame(rows), on=["modality", "sample_id"], how="left")


def summarize_group(group: pd.DataFrame) -> dict[str, float | int]:
    detected = group[group["person_detected"]]
    return {
        "frames": len(group),
        "person_success": float(group["person_detected"].mean()),
        "both_shoulders": float(group["both_shoulders"].mean()),
        "both_wrists": float(group["both_wrists"].mean()),
        "one_wrist": float(group["at_least_one_wrist"].mean()),
        "head_or_nose": float(group["head_or_nose"].mean()),
        "mean_confidence": float(detected["bbox_confidence"].mean()) if len(detected) else np.nan,
        "mean_bbox_area_ratio": float(detected["bbox_area_ratio"].mean()) if len(detected) else np.nan,
        "raw_center_jitter": float(group["raw_center_jitter"].drop_duplicates().mean()),
        "raw_size_jitter": float(group["raw_size_jitter"].drop_duplicates().mean()),
        "ema_center_jitter": float(group["ema_center_jitter"].drop_duplicates().mean()),
        "ema_size_jitter": float(group["ema_size_jitter"].drop_duplicates().mean()),
    }


def make_contact_sheet(paths: list[Path], destination: Path) -> None:
    thumbnails: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as opened:
            thumb = opened.convert("RGB")
            thumb.thumbnail((1024, 192))
            thumbnails.append(thumb.copy())
    columns = 2
    rows = int(np.ceil(len(thumbnails) / columns))
    width = max(image.width for image in thumbnails) * columns
    height = max(image.height for image in thumbnails) * rows
    sheet = Image.new("RGB", (width, height), "white")
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % columns) * image.width, (index // columns) * image.height))
    sheet.save(destination, quality=88)


def write_report(frame: pd.DataFrame, trials: pd.DataFrame, report_path: Path) -> None:
    overall = {modality: summarize_group(group) for modality, group in frame.groupby("modality")}
    action_rows = []
    for (modality, action), group in frame.groupby(["modality", "action_name"]):
        action_rows.append({"modality": modality, "name": action, **summarize_group(group)})
    user_rows = []
    for (modality, user), group in frame.groupby(["modality", "user_id"]):
        user_rows.append({"modality": modality, "name": user, **summarize_group(group)})
    def metric_table(rows: list[dict[str, object]]) -> list[str]:
        lines = ["| Input | Group | Person | Both shoulders | Both wrists | >=1 wrist | Head/nose | Mean conf | Bbox area |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in rows:
            lines.append(f"| {row['modality']} | {row['name']} | {row['person_success']:.2%} | {row['both_shoulders']:.2%} | {row['both_wrists']:.2%} | {row['one_wrist']:.2%} | {row['head_or_nose']:.2%} | {row['mean_confidence']:.3f} | {row['mean_bbox_area_ratio']:.3f} |")
        return lines
    lines = [
        "# Depth/IR pose localization feasibility probe",
        "",
        "## Probe design",
        "",
        f"- Training split only: {trials['user_id'].nunique()} users, {trials['action_name'].nunique()} actions, {len(trials)} trials, {len(frame) // 2} paired frames.",
        "- Four trials per action and six uniformly sampled paired frames per trial.",
        "- Model: Ultralytics YOLO11n-pose, detection threshold 0.25, keypoint threshold 0.25, image size 640.",
        "- Inputs tested independently: original Depth_Color pseudo-color and IR copied to three channels by the image loader.",
        "",
        "## Overall metrics",
        "",
        "| Input | Person | Both shoulders | Both wrists | >=1 wrist | Head/nose | Mean conf | Mean bbox area | Raw/EMA center jitter | Raw/EMA size jitter |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for modality in ("depth", "ir"):
        row = overall[modality]
        lines.append(f"| {modality} | {row['person_success']:.2%} | {row['both_shoulders']:.2%} | {row['both_wrists']:.2%} | {row['one_wrist']:.2%} | {row['head_or_nose']:.2%} | {row['mean_confidence']:.3f} | {row['mean_bbox_area_ratio']:.3f} | {row['raw_center_jitter']:.4f}/{row['ema_center_jitter']:.4f} | {row['raw_size_jitter']:.4f}/{row['ema_size_jitter']:.4f} |")
    lines.extend(["", "## Per-action metrics", "", *metric_table(action_rows), "", "## Per-user metrics", "", *metric_table(user_rows), "", "## Gate status", ""])
    passing = [modality for modality, row in overall.items() if row["person_success"] >= 0.8 and row["one_wrist"] >= 0.6]
    if passing:
        best = max(passing, key=lambda modality: (overall[modality]["one_wrist"], overall[modality]["person_success"]))
        lines.append(f"The numeric gate is met by {', '.join(passing)}; `{best}` is the provisional locator input. Training remains blocked until representative overlays are manually checked for true-person coverage, background false detections, small-person behavior, low-motion actions, and unacceptable temporal jumps.")
    else:
        lines.append("Neither input meets both numeric gates (person >=80% and at least one wrist >=60%). The classifier must not be trained; this is a negative route-feasibility result.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", dtype={"user_id": str, "trial_id": str})
    fold = json.loads(args.fold.read_text(encoding="utf-8"))
    trials = select_trials(manifest, set(fold["train_users"]), args.trials_per_action)
    records: list[dict[str, object]] = []
    for _, trial in trials.iterrows():
        pairs = paired_frames(args.data_root / trial["depth_color_path"], args.data_root / trial["ir_path"])
        indices = np.linspace(0, len(pairs) - 1, args.frames_per_trial).round().astype(int)
        for frame_index, pair_index in enumerate(indices):
            key, depth_path, ir_path = pairs[int(pair_index)]
            records.append({
                "sample_id": trial["sample_id"], "class_id": int(trial["class_id"]), "action_name": trial["action_name"],
                "user_id": trial["user_id"], "trial_id": trial["trial_id"], "frame_index": frame_index,
                "frame_key": key, "depth_path": str(depth_path), "ir_path": str(ir_path),
            })
    locator = UltralyticsPoseLocator(args.weights)
    metric_rows: list[dict[str, object]] = []
    representative_paths: list[Path] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        depth_images = [read_rgb(Path(record["depth_path"])) for record in batch]
        ir_images = [read_rgb(Path(record["ir_path"])) for record in batch]
        depth_detections = locator.predict(depth_images)
        ir_detections = locator.predict(ir_images)
        for record, depth, ir, depth_detection, ir_detection in zip(batch, depth_images, ir_images, depth_detections, ir_detections, strict=True):
            metric_rows.append(detection_row(record, "depth", depth, depth_detection))
            metric_rows.append(detection_row(record, "ir", ir, ir_detection))
            path = save_visuals(args.output_dir, record, depth, ir, depth_detection, ir_detection)
            if int(record["frame_index"]) == args.frames_per_trial // 2 and record["sample_id"] in set(trials.groupby("action_name").first()["sample_id"]):
                representative_paths.append(path)
        print(f"processed {min(start + len(batch), len(records))}/{len(records)} paired frames", flush=True)
    metrics = add_jitter_metrics(pd.DataFrame(metric_rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "probe_metrics.csv", index=False, encoding="utf-8-sig")
    trials.to_csv(args.output_dir / "selected_trials.csv", index=False, encoding="utf-8-sig")
    make_contact_sheet(representative_paths, args.output_dir / "side_by_side/contact_sheet.jpg")
    write_report(metrics, trials, args.report)
    print(args.output_dir / "probe_metrics.csv")
    print(args.report)


if __name__ == "__main__":
    main()
