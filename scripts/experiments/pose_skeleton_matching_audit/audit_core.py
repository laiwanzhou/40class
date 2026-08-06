from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


FRAME_RE = re.compile(
    r"(?:Color|Depth|IR)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})_"
    r"(?P<frame>\d+)(?:_Color)?$"
)

JOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
# Official Skeleton topology follows the common H36M-17 order:
# hip, right leg, left leg, spine, thorax, neck/head, left arm, right arm.
# Only anatomically equivalent joints are included; face and spine proxy points are not fabricated.
SKELETON_JOINT_NAMES = (
    "hip_center", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)
COMMON_JOINT_MAP = {
    5: 11, 6: 14, 7: 12, 8: 15, 9: 13, 10: 16,
    11: 4, 12: 1, 13: 5, 14: 2, 15: 6, 16: 3,
}
COMMON_YOLO_INDICES = np.array(sorted(COMMON_JOINT_MAP), dtype=int)
COCO_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)
ANGLE_TRIPLETS = (
    (5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16),
    (5, 11, 13), (6, 12, 14),
)
LEFT_RIGHT_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


@dataclass
class Trial:
    sample_id: str
    class_id: int
    action_name: str
    user_id: str
    trial_id: str
    frame_keys: list[str]
    yolo_xy: np.ndarray
    yolo_confidence: np.ndarray
    skeleton_xyz: np.ndarray
    depth_paths: list[Path]
    ir_paths: list[Path]
    image_size: tuple[int, int]


@dataclass
class NormalizedTrial:
    metadata: Trial
    yolo_2d: np.ndarray
    yolo_2d_mask: np.ndarray
    yolo_25d: np.ndarray
    yolo_25d_mask: np.ndarray
    skeleton: np.ndarray
    yolo_root_2d: np.ndarray
    yolo_scale_2d: np.ndarray


def frame_key(path: Path) -> str:
    match = FRAME_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unparseable frame filename: {path.name}")
    return f"{match.group('timestamp')}_{match.group('frame')}"


def choose_main_person(path: Path) -> np.ndarray | None:
    try:
        people = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    candidates = [item for item in people if isinstance(item, dict) and "keypoints" in item]
    if not candidates:
        return None
    person = max(candidates, key=lambda item: float(np.mean(item.get("keypoint_scores", [0.0]))))
    points = np.asarray(person["keypoints"], dtype=np.float64)
    return points if points.shape == (17, 3) and np.isfinite(points).all() else None


def map_skeleton_to_yolo_order(points: np.ndarray) -> np.ndarray:
    result = np.full((17, 3), np.nan, dtype=np.float64)
    for yolo_index, skeleton_index in COMMON_JOINT_MAP.items():
        result[yolo_index] = points[skeleton_index]
    return result


def decode_jet_relative_depth(rgb: np.ndarray, min_saturation: float = 0.10) -> np.ndarray:
    """Decode a Jet-like RGB visualization to a monotonic, non-metric depth coordinate."""
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    maximum = values.max(axis=-1)
    minimum = values.min(axis=-1)
    chroma = maximum - minimum
    hue = np.zeros_like(maximum)
    valid = (maximum > 0.04) & (chroma / np.maximum(maximum, 1e-12) >= min_saturation)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    red_max = valid & (maximum == red)
    green_max = valid & (maximum == green) & ~red_max
    blue_max = valid & ~(red_max | green_max)
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / chroma[red_max], 6.0)
    hue[green_max] = (blue[green_max] - red[green_max]) / chroma[green_max] + 2.0
    hue[blue_max] = (red[blue_max] - green[blue_max]) / chroma[blue_max] + 4.0
    hue *= 60.0
    result = np.full(maximum.shape, np.nan, dtype=np.float64)
    jet = valid & (hue <= 250.0)
    result[jet] = np.clip(1.0 - hue[jet] / 240.0, 0.0, 1.0)
    return result


def sample_relative_depth(
    path: Path, points: np.ndarray, valid_points: np.ndarray, radius: int, min_saturation: float
) -> np.ndarray:
    with Image.open(path) as image:
        source = image.convert("RGB")
        width, height = source.size
        result = np.full(17, np.nan, dtype=np.float64)
        for joint in np.flatnonzero(valid_points):
            x, y = np.rint(points[joint]).astype(int)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            patch = decode_jet_relative_depth(np.asarray(source.crop((x0, y0, x1, y1))), min_saturation)
            finite = patch[np.isfinite(patch)]
            if finite.size:
                result[joint] = float(np.median(finite))
    return result


def _torso_root_scale(points: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = np.array([5, 6, 11, 12])
    frame_valid = mask[:, required].all(axis=1)
    root = (points[:, 11] + points[:, 12]) * 0.5
    shoulders = (points[:, 5] + points[:, 6]) * 0.5
    shoulder_width = np.linalg.norm(points[:, 5] - points[:, 6], axis=1)
    torso_height = np.linalg.norm(shoulders - root, axis=1)
    scale = torso_height + 0.5 * shoulder_width
    frame_valid &= np.isfinite(root).all(axis=1) & np.isfinite(scale) & (scale > 1e-6)
    return root, scale, frame_valid


def normalize_pose(points: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root, scale, frame_valid = _torso_root_scale(points, mask)
    normalized = (points - root[:, None, :]) / scale[:, None, None]
    normalized_mask = mask & frame_valid[:, None] & np.isfinite(normalized).all(axis=2)
    normalized[~normalized_mask] = np.nan
    return normalized, normalized_mask, root, scale


def normalize_trial(
    trial: Trial, confidence_threshold: float, depth_radius: int, min_saturation: float
) -> NormalizedTrial:
    yolo_valid = (trial.yolo_confidence >= confidence_threshold) & np.isfinite(trial.yolo_xy).all(axis=2)
    common = np.zeros(17, dtype=bool)
    common[COMMON_YOLO_INDICES] = True
    yolo_valid &= common[None]
    yolo_2d, mask_2d, root_2d, scale_2d = normalize_pose(trial.yolo_xy, yolo_valid)
    skeleton_mask = np.isfinite(trial.skeleton_xyz).all(axis=2)
    skeleton, _, _, _ = normalize_pose(trial.skeleton_xyz, skeleton_mask)

    relative_depth = np.stack([
        sample_relative_depth(path, points, valid, depth_radius, min_saturation)
        for path, points, valid in zip(trial.depth_paths, trial.yolo_xy, yolo_valid, strict=True)
    ])
    width, height = trial.image_size
    observed_25d = np.concatenate(
        [trial.yolo_xy / np.array([width, height], dtype=np.float64), relative_depth[..., None]], axis=2
    )
    valid_25d = yolo_valid & np.isfinite(relative_depth)
    yolo_25d, mask_25d, _, _ = normalize_pose(observed_25d, valid_25d)
    return NormalizedTrial(trial, yolo_2d, mask_2d, yolo_25d, mask_25d, skeleton, root_2d, scale_2d)


def fit_linear_projection(trials: Iterable[NormalizedTrial], mode: str, ridge: float) -> np.ndarray:
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for trial in trials:
        target = trial.yolo_2d if mode == "2d" else trial.yolo_25d
        mask = trial.yolo_2d_mask if mode == "2d" else trial.yolo_25d_mask
        valid = mask & np.isfinite(trial.skeleton).all(axis=2)
        source_parts.append(trial.skeleton[valid])
        target_parts.append(target[valid])
    source = np.concatenate(source_parts)
    target = np.concatenate(target_parts)
    if len(source) < 100:
        raise ValueError(f"Insufficient calibration correspondences for {mode}: {len(source)}")
    gram = source.T @ source + ridge * np.eye(source.shape[1])
    return np.linalg.solve(gram, source.T @ target)


def project_skeleton(skeleton: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return skeleton @ matrix


def shift_pair(
    observed: np.ndarray, mask: np.ndarray, projected: np.ndarray, shift: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shift > 0:
        return observed[shift:], mask[shift:], projected[:-shift]
    if shift < 0:
        return observed[:shift], mask[:shift], projected[-shift:]
    return observed, mask, projected


def safe_corr(first: np.ndarray, second: np.ndarray, minimum: int = 4) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < minimum:
        return math.nan
    a, b = first[valid], second[valid]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def joint_angles(points: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = np.full((len(points), len(ANGLE_TRIPLETS)), np.nan)
    valid_result = np.zeros(result.shape, dtype=bool)
    for index, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        valid = mask[:, [a, b, c]].all(axis=1)
        first, second = points[:, a] - points[:, b], points[:, c] - points[:, b]
        denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        valid &= denominator > 1e-9
        cosine = np.sum(first * second, axis=1) / np.maximum(denominator, 1e-9)
        result[valid, index] = np.arccos(np.clip(cosine[valid], -1.0, 1.0))
        valid_result[valid, index] = True
    return result, valid_result


def pair_metrics(
    observed: np.ndarray, mask: np.ndarray, projected: np.ndarray, max_shift: int, pck_threshold: float
) -> tuple[dict[str, float], np.ndarray, int]:
    best: tuple[float, dict[str, float], np.ndarray, int] | None = None
    for shift in range(-max_shift, max_shift + 1):
        target, target_mask, estimate = shift_pair(observed, mask, projected, shift)
        valid = target_mask & np.isfinite(estimate).all(axis=2)
        distances = np.linalg.norm(target - estimate, axis=2)
        common_joints = np.isfinite(projected).any(axis=(0, 2))
        coverage = float(valid[:, common_joints].sum() / max(1, len(valid) * common_joints.sum()))
        rmse = float(np.sqrt(np.mean(np.square(distances[valid])))) if valid.any() else math.nan
        pck = float(np.mean(distances[valid] <= pck_threshold)) if valid.any() else math.nan
        target_angles, target_angle_mask = joint_angles(target, target_mask)
        estimate_angles, estimate_angle_mask = joint_angles(estimate, np.isfinite(estimate).all(axis=2))
        angle_corr = safe_corr(
            np.where(target_angle_mask & estimate_angle_mask, target_angles, np.nan), estimate_angles
        )
        velocity_corrs = []
        for joint in range(17):
            pair_valid = valid[1:, joint] & valid[:-1, joint]
            target_velocity = np.linalg.norm(np.diff(target[:, joint], axis=0), axis=1)
            estimate_velocity = np.linalg.norm(np.diff(estimate[:, joint], axis=0), axis=1)
            velocity_corrs.append(safe_corr(
                np.where(pair_valid, target_velocity, np.nan), estimate_velocity
            ))
        velocity_corr = float(np.nanmean(velocity_corrs)) if np.isfinite(velocity_corrs).any() else math.nan
        score = (
            0.55 * (0.0 if math.isnan(pck) else pck)
            + 0.20 * ((0.0 if math.isnan(angle_corr) else angle_corr) + 1.0) / 2.0
            + 0.15 * ((0.0 if math.isnan(velocity_corr) else velocity_corr) + 1.0) / 2.0
            + 0.10 * coverage
        )
        metrics = {
            "coverage": coverage, "rmse": rmse, "pck": pck,
            "angle_correlation": angle_corr, "velocity_correlation": velocity_corr,
            "matching_score": score,
        }
        if best is None or score > best[0]:
            best = score, metrics, distances, shift
    assert best is not None
    return best[1], best[2], best[3]


def resample(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == length:
        return values.copy()
    positions = np.linspace(0, len(values) - 1, length)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = (positions - lower).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[lower] * (1.0 - weight) + values[upper] * weight


def resample_masked(values: np.ndarray, mask: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    filled = np.nan_to_num(values, nan=0.0)
    weighted = resample(filled * mask[..., None], length)
    support = resample(mask.astype(np.float64), length)
    valid = support >= 0.75
    result = weighted / np.maximum(support[..., None], 1e-9)
    result[~valid] = np.nan
    return result, valid


def retrieval_scores(
    observed: np.ndarray, mask: np.ndarray, gallery: np.ndarray, max_shift: int
) -> np.ndarray:
    scores = np.full(len(gallery), -np.inf)
    for shift in range(-max_shift, max_shift + 1):
        query, query_mask, candidates = shift_pair(observed, mask, gallery.swapaxes(0, 1), shift)
        candidates = candidates.swapaxes(0, 1)
        valid = query_mask[..., None]
        difference = np.where(valid[None], candidates - query[None], 0.0)
        denominator = max(1, int(query_mask.sum()) * observed.shape[2])
        rmse = np.sqrt(np.square(difference).sum(axis=(1, 2, 3)) / denominator)
        scores = np.maximum(scores, np.exp(-rmse))
    return scores


def retrieval_query_is_valid(mask: np.ndarray, minimum_joint_observations: int) -> bool:
    return int(np.asarray(mask, dtype=bool).sum()) >= minimum_joint_observations


def left_right_swap(points: np.ndarray) -> np.ndarray:
    result = points.copy()
    for left, right in LEFT_RIGHT_PAIRS:
        result[:, [left, right]] = result[:, [right, left]]
    return result


def percentile_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney probability P(positive > negative), including half credit for ties."""
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    values, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for group in np.flatnonzero(counts > 1):
            members = inverse == group
            ranks[members] = ranks[members].mean()
    rank_sum = ranks[:len(positive)].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def draw_overlay(
    image_path: Path, observed_xy: np.ndarray, observed_mask: np.ndarray,
    projected_normalized: np.ndarray, root: np.ndarray, scale: float, output: Path, title: str,
) -> None:
    with Image.open(image_path) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    projected = projected_normalized[:, :2] * scale + root
    for a, b in COCO_EDGES:
        if observed_mask[a] and observed_mask[b]:
            draw.line((*observed_xy[a], *observed_xy[b]), fill=(0, 255, 255), width=2)
        if np.isfinite(projected[[a, b]]).all():
            draw.line((*projected[a], *projected[b]), fill=(255, 70, 40), width=2)
    for point, valid in zip(observed_xy, observed_mask, strict=True):
        if valid:
            x, y = point
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 255, 255))
    for point in projected:
        if np.isfinite(point).all():
            x, y = point
            draw.rectangle((x - 2, y - 2, x + 2, y + 2), fill=(255, 70, 40))
    draw.rectangle((0, 0, min(canvas.width, 620), 22), fill="black")
    draw.text((5, 4), title, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
