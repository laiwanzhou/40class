from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pose_roi_dataset import paired_frame_paths


DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
EXPORT_ROOT = DATA_ROOT.parent / "roi640_full_inputs_256"
POSE_CACHE = (
    PROJECT_ROOT.parent
    / "40class/outputs/depth_ir_person_crop_40class_fold0/person_crop_pose_tracks.npz"
)
REPORT_DIR = PROJECT_ROOT / "reports/roi640_stage2_temporal_identity"
REVIEW_ROOT = DATA_ROOT.parent / "roi640_stage2_temporal_review"
LOCAL_VIEWS = ("ir_left", "ir_right", "ir_relation")
DISPLAY_VIEWS = ("ir_context", "ir_left", "ir_right", "ir_relation")
SMALL_ACTION_IDS = {
    1, 2, 4, 6, 7, 8, 9, 10, 11, 14, 15, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    parser.add_argument("--pose-cache", type=Path, default=POSE_CACHE)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--review-root", type=Path, default=REVIEW_ROOT)
    parser.add_argument("--jet-workers", type=int, default=8)
    return parser.parse_args()


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(0.0, (second[2] - second[0]) * (second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def false_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if not value and start is None:
            start = index
        elif value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def load_pose_frame(path: Path) -> pd.DataFrame:
    with np.load(path) as cache:
        frame = pd.DataFrame({"sample_id": cache["sample_ids"].astype(str)})
        frame["source_frame_index"] = frame.groupby("sample_id", sort=False).cumcount()
        for name, column in zip(("px1", "py1", "px2", "py2"), range(4), strict=True):
            frame[name] = cache["bbox_xyxy"][:, column]
        for name, keypoint in (
            ("left_shoulder", 5), ("right_shoulder", 6),
            ("left_elbow", 7), ("right_elbow", 8),
            ("left_wrist", 9), ("right_wrist", 10),
        ):
            frame[f"{name}_x"] = cache["keypoints_xy"][:, keypoint, 0]
            frame[f"{name}_y"] = cache["keypoints_xy"][:, keypoint, 1]
            frame[f"{name}_confidence"] = cache["keypoints_confidence"][:, keypoint]
    return frame


def load_effective_audit(export_root: Path, report_dir: Path) -> pd.DataFrame:
    audit = pd.read_csv(export_root / "roi_frame_audit.csv", encoding="utf-8-sig").reset_index(
        names="row_id",
    )
    pixel_path = PROJECT_ROOT / "reports/roi640_quality_audit/roi640_pixel_metrics.csv"
    pixels = pd.read_csv(pixel_path)
    if len(audit) != len(pixels):
        raise ValueError("ROI audit and pixel metrics differ in length")
    audit = audit.merge(pixels, on="row_id", validate="one_to_one")
    low_information_tests = (
        (audit.dynamic_range <= 8).astype(np.int8)
        + (audit["std"] <= 2.0).astype(np.int8)
        + (audit.entropy_32 <= 1.0).astype(np.int8)
    )
    audit["content_invalid"] = (audit.valid == 1) & (low_information_tests >= 2)
    audit["effective_valid"] = (audit.valid == 1) & ~audit.content_invalid
    audit.to_csv(report_dir / "effective_view_audit.csv", index=False, encoding="utf-8-sig")
    return audit


def local_wide(audit: pd.DataFrame, pose: pd.DataFrame) -> pd.DataFrame:
    values = ["valid", "effective_valid", "x1", "y1", "x2", "y2", "confidence", "source"]
    keys = ["split", "class_id", "action_name", "sample_id", "user_id", "source_frame_index"]
    wide = audit[audit.view_name.isin(LOCAL_VIEWS)].pivot(
        index=keys, columns="view_name", values=values,
    ).reset_index()
    wide.columns = ["_".join(str(value) for value in column if str(value)) for column in wide.columns]
    return wide.merge(pose, on=["sample_id", "source_frame_index"], validate="one_to_one")


def person_geometry(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = group.px2.to_numpy(float) - group.px1.to_numpy(float)
    height = group.py2.to_numpy(float) - group.py1.to_numpy(float)
    valid = np.isfinite(np.c_[width, height]).all(axis=1) & (width > 1) & (height > 1)
    context_width = group.x2_ir_relation.to_numpy(float) - group.x1_ir_relation.to_numpy(float)
    context_height = group.y2_ir_relation.to_numpy(float) - group.y1_ir_relation.to_numpy(float)
    width = np.where(valid, width, np.maximum(context_width, 1.0))
    height = np.where(valid, height, np.maximum(context_height, 1.0))
    center_x = np.where(
        valid,
        (group.px1.to_numpy(float) + group.px2.to_numpy(float)) / 2.0,
        (group.x1_ir_relation.to_numpy(float) + group.x2_ir_relation.to_numpy(float)) / 2.0,
    )
    center_y = np.where(
        valid,
        (group.py1.to_numpy(float) + group.py2.to_numpy(float)) / 2.0,
        (group.y1_ir_relation.to_numpy(float) + group.y2_ir_relation.to_numpy(float)) / 2.0,
    )
    return center_x, center_y, width, height


def temporal_metrics(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_id, sample in wide.groupby("sample_id", sort=False):
        sample = sample.sort_values("source_frame_index").reset_index(drop=True)
        person_x, person_y, person_width, person_height = person_geometry(sample)
        person_diagonal = np.hypot(person_width, person_height)
        for view in LOCAL_VIEWS:
            valid = sample[f"effective_valid_{view}"].to_numpy(bool)
            boxes = sample[[f"x1_{view}", f"y1_{view}", f"x2_{view}", f"y2_{view}"]].to_numpy(float)
            centers = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0]
            relative_centers = np.c_[
                (centers[:, 0] - person_x) / person_width,
                (centers[:, 1] - person_y) / person_height,
            ]
            areas = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1.0)
            frame_indices = sample.source_frame_index.to_numpy(int)
            for index in range(1, len(sample)):
                if frame_indices[index] != frame_indices[index - 1] + 1:
                    continue
                if not (valid[index - 1] and valid[index]):
                    continue
                scale = max(float(np.mean(person_diagonal[index - 1 : index + 1])), 1.0)
                center_norm = float(np.linalg.norm(centers[index] - centers[index - 1]) / scale)
                relative_delta = float(np.linalg.norm(relative_centers[index] - relative_centers[index - 1]))
                area_ratio = float(max(areas[index] / areas[index - 1], areas[index - 1] / areas[index]))
                iou = box_iou(boxes[index - 1], boxes[index])
                candidate = bool(iou <= 0.10 and (center_norm >= 0.50 or relative_delta >= 0.50))
                rows.append({
                    "split": sample.split.iloc[0],
                    "class_id": int(sample.class_id.iloc[0]),
                    "action_name": sample.action_name.iloc[0],
                    "sample_id": sample_id,
                    "user_id": sample.user_id.iloc[0],
                    "source_frame_index": int(frame_indices[index]),
                    "view_name": view,
                    "adjacent_iou": iou,
                    "center_displacement_over_person_diagonal": center_norm,
                    "person_relative_center_delta": relative_delta,
                    "area_ratio": area_ratio,
                    "high_risk_jump_candidate": candidate,
                    "severity": max(center_norm, relative_delta) * (1.0 - iou) * min(area_ratio, 10.0),
                })
    return pd.DataFrame(rows)


def roi_swap_score(sample: pd.DataFrame, index: int) -> tuple[float, float]:
    def box(frame: int, view: str) -> np.ndarray:
        return sample.loc[frame, [f"x1_{view}", f"y1_{view}", f"x2_{view}", f"y2_{view}"]].to_numpy(float)

    direct = sum(
        box_iou(box(index, view), box(neighbor, view))
        for neighbor in (index - 1, index + 1)
        for view in ("ir_left", "ir_right")
    )
    crossed = sum(
        box_iou(box(index, current), box(neighbor, previous))
        for neighbor in (index - 1, index + 1)
        for current, previous in (("ir_left", "ir_right"), ("ir_right", "ir_left"))
    )
    return direct, crossed


def swap_metrics(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    for sample_id, sample in wide.groupby("sample_id", sort=False):
        sample = sample.sort_values("source_frame_index").reset_index(drop=True)
        person_diagonal = np.hypot(
            np.maximum(sample.px2.to_numpy(float) - sample.px1.to_numpy(float), 1.0),
            np.maximum(sample.py2.to_numpy(float) - sample.py1.to_numpy(float), 1.0),
        )
        left = sample[["left_wrist_x", "left_wrist_y"]].to_numpy(float)
        right = sample[["right_wrist_x", "right_wrist_y"]].to_numpy(float)
        valid = (
            (sample.left_wrist_confidence.to_numpy(float) >= 0.25)
            & (sample.right_wrist_confidence.to_numpy(float) >= 0.25)
            & sample.effective_valid_ir_left.to_numpy(bool)
            & sample.effective_valid_ir_right.to_numpy(bool)
            & np.isfinite(left).all(axis=1)
            & np.isfinite(right).all(axis=1)
        )
        frame_indices = sample.source_frame_index.to_numpy(int)
        eligible_count = 0
        for index in range(1, len(sample) - 1):
            if not np.array_equal(frame_indices[index - 1 : index + 2], np.arange(frame_indices[index] - 1, frame_indices[index] + 2)):
                continue
            if not valid[index - 1 : index + 2].all():
                continue
            eligible_count += 1
            scale = max(float(np.mean(person_diagonal[index - 1 : index + 2])), 1.0)
            predicted_left = (left[index - 1] + left[index + 1]) / 2.0
            predicted_right = (right[index - 1] + right[index + 1]) / 2.0
            direct = float(
                (np.linalg.norm(left[index] - predicted_left) + np.linalg.norm(right[index] - predicted_right))
                / scale
            )
            crossed = float(
                (np.linalg.norm(right[index] - predicted_left) + np.linalg.norm(left[index] - predicted_right))
                / scale
            )
            endpoint_motion = float(
                (np.linalg.norm(left[index + 1] - left[index - 1]) + np.linalg.norm(right[index + 1] - right[index - 1]))
                / scale
            )
            separation = float(np.linalg.norm(left[index] - right[index]) / scale)
            wrist_candidate = bool(
                crossed + 0.10 < direct
                and crossed <= 0.50 * direct
                and endpoint_motion <= 0.50
                and separation >= 0.05
            )
            if not wrist_candidate:
                continue
            roi_direct, roi_crossed = roi_swap_score(sample, index)
            corroborated = bool(roi_crossed > roi_direct + 0.25)
            candidates.append({
                "split": sample.split.iloc[0],
                "class_id": int(sample.class_id.iloc[0]),
                "action_name": sample.action_name.iloc[0],
                "sample_id": sample_id,
                "user_id": sample.user_id.iloc[0],
                "source_frame_index": int(frame_indices[index]),
                "direct_wrist_cost": direct,
                "crossed_wrist_cost": crossed,
                "endpoint_motion": endpoint_motion,
                "left_right_separation": separation,
                "direct_roi_iou_sum": roi_direct,
                "crossed_roi_iou_sum": roi_crossed,
                "high_confidence_swap_candidate": corroborated,
            })
        eligibility.append({
            "class_id": int(sample.class_id.iloc[0]),
            "action_name": sample.action_name.iloc[0],
            "sample_id": sample_id,
            "eligible_centers": eligible_count,
        })
    return pd.DataFrame(candidates), pd.DataFrame(eligibility)


def long_invalid_runs(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_id, sample in wide.groupby("sample_id", sort=False):
        sample = sample.sort_values("source_frame_index").reset_index(drop=True)
        for view in LOCAL_VIEWS:
            valid = sample[f"effective_valid_{view}"].to_numpy(bool)
            for start, end in false_runs(valid):
                length = end - start + 1
                if length < 8:
                    continue
                rows.append({
                    "split": sample.split.iloc[0],
                    "class_id": int(sample.class_id.iloc[0]),
                    "action_name": sample.action_name.iloc[0],
                    "sample_id": sample_id,
                    "user_id": sample.user_id.iloc[0],
                    "view_name": view,
                    "start_frame": int(sample.source_frame_index.iloc[start]),
                    "end_frame": int(sample.source_frame_index.iloc[end]),
                    "run_length": length,
                    "sample_frames": len(sample),
                })
    return pd.DataFrame(rows)


def jet_lut_lookup() -> np.ndarray:
    values = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(values, cv2.COLORMAP_JET).reshape(256, 3)
    lookup = np.zeros(1 << 24, dtype=bool)
    encoded = (
        bgr[:, 0].astype(np.int64)
        | (bgr[:, 1].astype(np.int64) << 8)
        | (bgr[:, 2].astype(np.int64) << 16)
    )
    lookup[encoded] = True
    return lookup


def check_jet_file(item: tuple[str, Path], lookup: np.ndarray) -> dict[str, Any]:
    sample_id, path = item
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"sample_id": sample_id, "path": str(path), "readable": False, "unexpected_pixels": -1}
    encoded = (
        image[..., 0].astype(np.int64)
        | (image[..., 1].astype(np.int64) << 8)
        | (image[..., 2].astype(np.int64) << 16)
    )
    black = (image == 0).all(axis=2)
    unexpected = (~black) & ~lookup[encoded]
    return {
        "sample_id": sample_id,
        "path": str(path),
        "readable": True,
        "unexpected_pixels": int(unexpected.sum()),
    }


def source_depth_paths(frame_manifest: pd.DataFrame) -> list[tuple[str, Path]]:
    source_manifest = pd.read_csv(PROJECT_ROOT / "metadata/manifest.csv", encoding="utf-8-sig")
    source_manifest = source_manifest.set_index("sample_id")
    items: list[tuple[str, Path]] = []
    for sample_id, group in frame_manifest.groupby("sample_id", sort=False):
        row = source_manifest.loc[sample_id]
        depth_paths, ir_paths = paired_frame_paths(
            DATA_ROOT / str(row.depth_color_path), DATA_ROOT / str(row.ir_path),
        )
        if len(depth_paths) != len(group) or len(depth_paths) != len(ir_paths):
            raise ValueError(f"Source pairing differs for {sample_id}")
        items.extend((sample_id, path) for path in depth_paths)
    return items


def scan_jet_sources(frame_manifest: pd.DataFrame, workers: int) -> pd.DataFrame:
    items = source_depth_paths(frame_manifest)
    lookup = jet_lut_lookup()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(lambda item: check_jet_file(item, lookup), items, chunksize=32))
    return pd.DataFrame(rows)


def choose_positions(center: int, length: int, radius: int = 3) -> list[int]:
    return list(range(max(0, center - radius), min(length, center + radius + 1)))


def make_contact_sheet(
    frame_manifest: pd.DataFrame,
    effective_lookup: dict[tuple[str, int, str], bool],
    export_root: Path,
    sample_id: str,
    positions: Iterable[int],
    output: Path,
    title: str,
) -> None:
    sample = frame_manifest[frame_manifest.sample_id == sample_id].sort_values("source_frame_index")
    by_index = sample.set_index("source_frame_index")
    positions = [position for position in dict.fromkeys(positions) if position in by_index.index]
    tile = 128
    label_height = 22
    timeline_height = 70
    width = max(800, tile * len(positions))
    height = 42 + timeline_height + len(DISPLAY_VIEWS) * (tile + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    timeline_y = 34
    total_frames = len(sample)
    for row_index, view in enumerate(LOCAL_VIEWS):
        y = timeline_y + row_index * 16
        draw.text((4, y), view.replace("ir_", ""), fill="black")
        valid_values = [effective_lookup[(sample_id, int(index), view)] for index in sample.source_frame_index]
        for frame_offset, valid in enumerate(valid_values):
            x1 = 70 + int((width - 80) * frame_offset / max(total_frames, 1))
            x2 = 70 + int((width - 80) * (frame_offset + 1) / max(total_frames, 1))
            draw.rectangle((x1, y, max(x1 + 1, x2), y + 9), fill="green" if valid else "red")
    for column, frame_index in enumerate(positions):
        row = by_index.loc[frame_index]
        for view_index, view in enumerate(DISPLAY_VIEWS):
            path = export_root / str(row[f"{view}_path"])
            with Image.open(path) as image:
                resized = image.convert("L").resize((tile, tile), Image.Resampling.BILINEAR).convert("RGB")
            x = column * tile
            y = 42 + timeline_height + view_index * (tile + label_height)
            sheet.paste(resized, (x, y))
            valid = effective_lookup.get((sample_id, frame_index, view), True)
            color = "green" if valid else "red"
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=color, width=3)
            draw.text((x + 3, y + tile + 2), f"f{frame_index} {view.replace('ir_', '')}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", compress_level=3)


def review_artifacts(
    frames: pd.DataFrame,
    audit: pd.DataFrame,
    jumps: pd.DataFrame,
    swaps: pd.DataFrame,
    runs: pd.DataFrame,
    export_root: Path,
    review_root: Path,
) -> pd.DataFrame:
    effective_lookup = {
        (str(row.sample_id), int(row.source_frame_index), str(row.view_name)): bool(row.effective_valid)
        for row in audit.itertuples(index=False)
        if str(row.view_name).startswith("ir_")
    }
    lengths = frames.groupby("sample_id").size().to_dict()
    rows: list[dict[str, Any]] = []

    high_swaps = swaps[swaps.high_confidence_swap_candidate].copy()
    for row in high_swaps.itertuples(index=False):
        output = review_root / "swap" / f"{row.sample_id}__f{row.source_frame_index:04d}.png"
        make_contact_sheet(
            frames, effective_lookup, export_root, row.sample_id,
            choose_positions(row.source_frame_index, lengths[row.sample_id]), output,
            f"swap candidate | {row.action_name} | {row.sample_id} | frame {row.source_frame_index}",
        )
        rows.append({"event_type": "swap", **row._asdict(), "artifact_path": str(output)})

    selected_jumps = (
        jumps[jumps.high_risk_jump_candidate]
        .sort_values("severity", ascending=False)
        .groupby(["action_name", "view_name"], as_index=False)
        .head(1)
    )
    for row in selected_jumps.itertuples(index=False):
        output = review_root / "jump" / f"{row.sample_id}__{row.view_name}__f{row.source_frame_index:04d}.png"
        make_contact_sheet(
            frames, effective_lookup, export_root, row.sample_id,
            choose_positions(row.source_frame_index, lengths[row.sample_id]), output,
            f"normalized jump | {row.action_name} | {row.view_name} | frame {row.source_frame_index}",
        )
        rows.append({"event_type": "jump", **row._asdict(), "artifact_path": str(output)})

    selected_runs = (
        runs.sort_values("run_length", ascending=False)
        .groupby(["action_name", "view_name"], as_index=False)
        .head(1)
    )
    for row in selected_runs.itertuples(index=False):
        middle = (row.start_frame + row.end_frame) // 2
        positions = []
        for center in (row.start_frame, middle, row.end_frame):
            positions.extend((center - 3, center, center + 3))
        output = review_root / "long_invalid" / f"{row.sample_id}__{row.view_name}__f{row.start_frame:04d}-{row.end_frame:04d}.png"
        make_contact_sheet(
            frames, effective_lookup, export_root, row.sample_id, positions, output,
            f"long invalid | {row.action_name} | {row.view_name} | {row.start_frame}-{row.end_frame}",
        )
        rows.append({"event_type": "long_invalid", **row._asdict(), "artifact_path": str(output)})
    return pd.DataFrame(rows)


def availability_metrics(wide: pd.DataFrame) -> dict[str, Any]:
    local_count = sum(wide[f"effective_valid_{view}"].astype(int) for view in LOCAL_VIEWS)
    none = local_count == 0
    small = wide.class_id.isin(SMALL_ACTION_IDS)
    by_action = wide.assign(all_local_invalid=none).groupby(
        ["class_id", "action_name"], as_index=False,
    ).all_local_invalid.mean()
    small_by_action = by_action[by_action.class_id.isin(SMALL_ACTION_IDS)]

    def conditional_relation(condition: pd.Series, selected: pd.Series) -> float:
        return float(selected[condition].mean()) if condition.any() else 0.0

    left = wide.effective_valid_ir_left.astype(bool)
    right = wide.effective_valid_ir_right.astype(bool)
    relation = wide.effective_valid_ir_relation.astype(bool)
    return {
        "global_all_local_invalid_rate": float(none.mean()),
        "small24_all_local_invalid_rate": float(none[small].mean()),
        "worst_small_action_rate": float(small_by_action.all_local_invalid.max()),
        "worst_small_action": str(small_by_action.loc[small_by_action.all_local_invalid.idxmax(), "action_name"]),
        "local_valid_count_distribution": {
            str(int(key)): int(value)
            for key, value in local_count.value_counts().sort_index().items()
        },
        "relation_valid_given_right_invalid": conditional_relation(~right, relation),
        "relation_valid_given_left_invalid": conditional_relation(~left, relation),
        "relation_valid_given_both_invalid": conditional_relation(~left & ~right, relation),
        "by_action": by_action,
    }


def rate_by_action(events: pd.DataFrame, denominators: pd.DataFrame, event_column: str) -> pd.DataFrame:
    counts = events[events[event_column]].groupby(["class_id", "action_name"]).size().rename("events")
    result = denominators.groupby(["class_id", "action_name"]).size().rename("eligible").to_frame()
    result["events"] = counts
    result["events"] = result.events.fillna(0).astype(int)
    result["rate"] = result.events / result.eligible
    return result.reset_index()


def write_report(
    path: Path,
    audit: pd.DataFrame,
    temporal: pd.DataFrame,
    swaps: pd.DataFrame,
    swap_eligibility: pd.DataFrame,
    runs: pd.DataFrame,
    jet: pd.DataFrame,
    availability: dict[str, Any],
    gate: dict[str, Any],
    artifacts: pd.DataFrame,
) -> None:
    high_jumps = temporal[temporal.high_risk_jump_candidate]
    high_swaps = swaps[swaps.high_confidence_swap_candidate]
    lines = [
        "# ROI640 第二阶段：时序与左右身份审计",
        "",
        "## 1. 结论",
        "",
        f"- 第二阶段验收：**{gate['stage2_status']}**。",
        "- 固定使用现有 YOLO `imgsz=640` 姿态缓存；没有测试 1280。",
        "- 未读取 competition test，未训练模型，未重导 Depth。",
        f"- 有效局部ROI相邻转移：{len(temporal):,}；高风险尺度归一化跳变上界：{len(high_jumps):,}。",
        f"- 左右交换可审计中心帧：{int(swap_eligibility.eligible_centers.sum()):,}；腕轨迹候选：{len(swaps)}；ROI交叉证据同时支持：{len(high_swaps)}。",
        "- 跳变和交换门槛按最坏情况计算：即把全部高置信候选都视为真实错误，因此无需用主观目视结果降低计数才能通过。",
        "",
        "## 2. 输入完整性与JET源色",
        "",
        f"- 有效标记但纯黑且仍未被content mask排除：{gate['values']['known_p0_still_effective']}。",
        f"- 扫描原始Depth_Color：{len(jet):,}帧；不可读：{int((~jet.readable).sum())}；非黑非JET像素：{int(jet.unexpected_pixels.clip(lower=0).sum())}。",
        "",
        "## 3. 三路IR局部冗余",
        "",
        f"- 全局三路均失效：{availability['global_all_local_invalid_rate']:.2%}。",
        f"- 小动作24类三路均失效：{availability['small24_all_local_invalid_rate']:.2%}。",
        f"- 最差小动作：`{availability['worst_small_action']}`，{availability['worst_small_action_rate']:.2%}。",
        f"- `P(relation valid | right invalid)`：{availability['relation_valid_given_right_invalid']:.2%}。",
        f"- `P(relation valid | left invalid)`：{availability['relation_valid_given_left_invalid']:.2%}。",
        f"- `P(relation valid | both invalid)`：{availability['relation_valid_given_both_invalid']:.2%}。",
        "",
        "## 4. 尺度归一化跳变",
        "",
        "高风险候选定义为：相邻帧ROI IoU不高于0.10，并且中心位移/人物框对角线或人物相对中心变化不低于0.50。",
        f"候选率上界：{gate['values']['jump_global_rate']:.4%}；最差类别上界：{gate['values']['jump_worst_class_rate']:.4%}。",
        "该指标同时输出相邻IoU、人物尺度归一化位移、人物相对中心变化和面积倍率；不再使用单一的全图对角线20%作为最终判据。",
        "",
        "## 5. 左右身份交换",
        "",
        "高置信候选要求三帧腕轨迹交叉匹配显著优于直接匹配，并由左右ROI的交叉IoU同时支持。",
        f"候选率上界：{gate['values']['swap_global_rate']:.4%}；最差类别上界：{gate['values']['swap_worst_class_rate']:.4%}。",
        "这些是保守的自动候选，不是对每一帧人体解剖身份的人工真值标注；验收采用全部候选均为真错误的最坏上界。",
        "",
        "## 6. 连续失效和短时序材料",
        "",
        f"- 长度不低于8帧的局部失效区间：{len(runs):,}。",
        f"- 生成审查contact sheet：{len(artifacts):,}，位于 `{REVIEW_ROOT}`。",
        "- swap/jump材料展示事件前后 `t-3..t+3`；长失效材料展示开始、中段和恢复附近，并带完整mask时间轴。",
        "",
        "## 7. 验收门",
        "",
    ]
    for name, value in gate["checks"].items():
        lines.append(f"- {'PASS' if value else 'FAIL'}: `{name}`")
    lines.extend([
        "",
        "## 8. 产物",
        "",
        f"- `{REPORT_DIR / 'normalized_temporal_transitions.csv'}`",
        f"- `{REPORT_DIR / 'normalized_jump_candidates.csv'}`",
        f"- `{REPORT_DIR / 'left_right_swap_candidates.csv'}`",
        f"- `{REPORT_DIR / 'long_invalid_runs.csv'}`",
        f"- `{REPORT_DIR / 'stage2_review_artifacts.csv'}`",
        f"- `{REPORT_DIR / 'stage2_acceptance_gate.json'}`",
        "",
        "第二阶段到此停止；没有执行第三阶段的逆JET导出器实现。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.review_root.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(args.export_root / "all_frame_inputs.csv", encoding="utf-8-sig")
    audit = load_effective_audit(args.export_root, args.report_dir)
    pose = load_pose_frame(args.pose_cache)
    if len(frames) != len(pose):
        raise ValueError("Frame manifest and pose cache differ in length")
    wide = local_wide(audit, pose).sort_values(["sample_id", "source_frame_index"])

    temporal = temporal_metrics(wide)
    temporal.to_csv(args.report_dir / "normalized_temporal_transitions.csv", index=False, encoding="utf-8-sig")
    jumps = temporal[temporal.high_risk_jump_candidate].copy()
    jumps.to_csv(args.report_dir / "normalized_jump_candidates.csv", index=False, encoding="utf-8-sig")
    swaps, swap_eligibility = swap_metrics(wide)
    swaps.to_csv(args.report_dir / "left_right_swap_candidates.csv", index=False, encoding="utf-8-sig")
    swap_eligibility.to_csv(args.report_dir / "left_right_swap_eligibility.csv", index=False, encoding="utf-8-sig")
    runs = long_invalid_runs(wide)
    runs.to_csv(args.report_dir / "long_invalid_runs.csv", index=False, encoding="utf-8-sig")

    jet = scan_jet_sources(frames, args.jet_workers)
    jet.to_csv(args.report_dir / "source_depth_jet_integrity.csv", index=False, encoding="utf-8-sig")
    artifacts = review_artifacts(frames, audit, jumps, swaps, runs, args.export_root, args.review_root)
    artifacts.to_csv(args.report_dir / "stage2_review_artifacts.csv", index=False, encoding="utf-8-sig")

    availability = availability_metrics(wide)
    availability["by_action"].to_csv(
        args.report_dir / "local_availability_by_action.csv", index=False, encoding="utf-8-sig",
    )
    jump_by_action = rate_by_action(temporal, temporal, "high_risk_jump_candidate")
    jump_by_action.to_csv(args.report_dir / "normalized_jump_by_action.csv", index=False, encoding="utf-8-sig")
    high_swaps = swaps[swaps.high_confidence_swap_candidate]
    swap_counts = high_swaps.groupby(["class_id", "action_name"]).size().rename("events")
    swap_by_action = swap_eligibility.groupby(["class_id", "action_name"]).eligible_centers.sum().rename("eligible").to_frame()
    swap_by_action["events"] = swap_counts
    swap_by_action["events"] = swap_by_action.events.fillna(0).astype(int)
    swap_by_action["rate"] = np.divide(
        swap_by_action.events,
        swap_by_action.eligible,
        out=np.zeros(len(swap_by_action), dtype=float),
        where=swap_by_action.eligible.to_numpy() > 0,
    )
    swap_by_action = swap_by_action.reset_index()
    swap_by_action.to_csv(args.report_dir / "left_right_swap_by_action.csv", index=False, encoding="utf-8-sig")

    integrity = json.loads((PROJECT_ROOT / "reports/roi640_quality_audit/roi640_quality_audit_metrics.json").read_text())
    known_p0_still_effective = int((audit.exact_black.fillna(False) & audit.effective_valid).sum())
    jump_global_rate = float(jumps.shape[0] / max(len(temporal), 1))
    jump_worst_class_rate = float(jump_by_action.rate.max())
    swap_global_rate = float(len(high_swaps) / max(int(swap_eligibility.eligible_centers.sum()), 1))
    swap_worst_class_rate = float(swap_by_action.rate.max())
    checks = {
        "missing_files_zero": integrity["integrity"]["missing_files"] == 0,
        "unreadable_files_zero": integrity["integrity"]["unreadable_files"] == 0,
        "wrong_dimensions_zero": integrity["integrity"]["wrong_dimensions"] == 0,
        "known_p0_still_effective_zero": known_p0_still_effective == 0,
        "source_depth_unreadable_zero": int((~jet.readable).sum()) == 0,
        "unexplained_non_jet_pixels_zero": int(jet.unexpected_pixels.clip(lower=0).sum()) == 0,
        "global_all_local_invalid_le_5pct": availability["global_all_local_invalid_rate"] <= 0.05,
        "small24_all_local_invalid_le_3pct": availability["small24_all_local_invalid_rate"] <= 0.03,
        "every_small_action_all_local_invalid_le_10pct": availability["worst_small_action_rate"] <= 0.10,
        "swap_global_upper_bound_le_1pct": swap_global_rate <= 0.01,
        "swap_every_class_upper_bound_le_5pct": swap_worst_class_rate <= 0.05,
        "jump_global_upper_bound_le_1pct": jump_global_rate <= 0.01,
        "jump_every_class_upper_bound_le_5pct": jump_worst_class_rate <= 0.05,
    }
    values = {
        "known_p0_still_effective": known_p0_still_effective,
        "source_depth_frames": len(jet),
        "source_depth_unreadable": int((~jet.readable).sum()),
        "unexpected_non_jet_pixels": int(jet.unexpected_pixels.clip(lower=0).sum()),
        "jump_candidates": len(jumps),
        "jump_global_rate": jump_global_rate,
        "jump_worst_class_rate": jump_worst_class_rate,
        "swap_candidates": len(high_swaps),
        "swap_global_rate": swap_global_rate,
        "swap_worst_class_rate": swap_worst_class_rate,
        **{key: value for key, value in availability.items() if key != "by_action"},
    }
    gate = {
        "stage2_status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "values": values,
        "test_read": False,
        "training_run": False,
        "pose_image_size": 640,
        "pose_1280_tested": False,
    }
    (args.report_dir / "stage2_acceptance_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    write_report(
        PROJECT_ROOT / "reports/roi640_stage2_temporal_identity_audit.md",
        audit, temporal, swaps, swap_eligibility, runs, jet, availability, gate, artifacts,
    )
    print(json.dumps(gate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
