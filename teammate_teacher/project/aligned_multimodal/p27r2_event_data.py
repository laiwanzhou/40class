from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from p27_data import read_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "runs" / "p27_0_audit" / "p27_train_manifest.csv"
DEFAULT_ALIGNED_CACHE = PROJECT_DIR / "cache" / "aligned_192x144"
DEFAULT_IMU_CACHE = PROJECT_DIR / "cache" / "imu_32"

ROOT = 0
THORAX = 8
HEAD = 10
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 12, 13
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 14, 15, 16
LEFT_HIP, RIGHT_HIP = 4, 1
TORSO_IMU, LEFT_IMU, RIGHT_IMU = 0, 1, 2
EVENT_STEPS = 32
GRID_HEIGHT = 6
GRID_WIDTH = 8

EVENT_NAMES = (
    "left_activity_share",
    "role_stability",
    "role_exchange",
    "bilateral_synchrony",
    "bilateral_alternation",
    "unilateral_head_proximity",
    "head_contact_dwell",
    "approach_phase_strength",
    "hold_phase_strength",
    "return_phase_strength",
    "raise_hold_return",
    "single_tilt_return",
    "repeated_periodicity",
    "sparse_event_density",
    "motion_irregularity",
    "continuous_motion_fraction",
    "visual_imu_peak_alignment",
    "work_center_stability",
)

EVENT_SOURCES = (
    "imu",
    "imu",
    "imu",
    "imu",
    "imu",
    "skeleton",
    "skeleton",
    "skeleton",
    "skeleton",
    "skeleton",
    "skeleton",
    "imu",
    "imu",
    "imu",
    "imu",
    "imu",
    "visual+imu",
    "visual",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe_unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8 or not np.isfinite(norm):
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _body_local(coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    centered = coordinates - coordinates[:, ROOT : ROOT + 1]
    shoulder = np.linalg.norm(
        centered[:, RIGHT_SHOULDER] - centered[:, LEFT_SHOULDER], axis=1
    )
    torso = np.linalg.norm(centered[:, THORAX] - centered[:, ROOT], axis=1)
    scale_values = 0.5 * (shoulder + torso)
    scale_values = scale_values[np.isfinite(scale_values) & (scale_values > 1e-5)]
    scale = float(np.median(scale_values)) if len(scale_values) else 1.0
    scaled = centered / max(scale, 1e-5)
    across = np.median(
        0.5
        * (
            scaled[:, RIGHT_SHOULDER]
            - scaled[:, LEFT_SHOULDER]
            + scaled[:, RIGHT_HIP]
            - scaled[:, LEFT_HIP]
        ),
        axis=0,
    )
    up = np.median(scaled[:, THORAX] - scaled[:, ROOT], axis=0)
    x_axis = _safe_unit(across, np.asarray([1.0, 0.0, 0.0]))
    up = up - float(np.dot(up, x_axis)) * x_axis
    z_axis = _safe_unit(up, np.asarray([0.0, 0.0, 1.0]))
    if z_axis[2] < 0:
        z_axis *= -1
    y_axis = _safe_unit(np.cross(z_axis, x_axis), np.asarray([0.0, 1.0, 0.0]))
    x_axis = _safe_unit(np.cross(y_axis, z_axis), x_axis)
    basis = np.stack([x_axis, y_axis, z_axis])
    return (scaled @ basis.T).astype(np.float32), scale


def _interp_time(values: np.ndarray, count: int = EVENT_STEPS) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((count,) + values.shape[1:], dtype=np.float32)
    if len(values) == 1:
        return np.repeat(values.astype(np.float32), count, axis=0)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, count)
    flat = values.reshape(len(values), -1)
    output = np.stack(
        [np.interp(target, source, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    )
    return output.reshape((count,) + values.shape[1:]).astype(np.float32)


def _segment_reduce(values: np.ndarray, count: int, reducer: str) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((count,) + values.shape[1:], dtype=np.float32)
    edges = np.floor(np.linspace(0, len(values), count + 1)).astype(np.int64)
    output = []
    for index in range(count):
        start = min(int(edges[index]), len(values) - 1)
        stop = min(len(values), max(start + 1, int(edges[index + 1])))
        segment = values[start:stop]
        output.append(
            np.max(segment, axis=0) if reducer == "max" else np.mean(segment, axis=0)
        )
    return np.asarray(output, dtype=np.float32)


def _safe_corr(left: np.ndarray, right: np.ndarray, lag: int = 0) -> float:
    if lag > 0:
        left, right = left[:-lag], right[lag:]
    elif lag < 0:
        left, right = left[-lag:], right[:lag]
    if len(left) < 4:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-8 else 0.0


def _longest_run(mask: np.ndarray) -> float:
    best = current = 0
    for value in mask.astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best / max(len(mask), 1))


def _spectral_entropy(signal: np.ndarray) -> float:
    centered = signal - gaussian_filter1d(signal, sigma=3.0, mode="nearest")
    power = np.abs(np.fft.rfft(centered)) ** 2
    power = power[1:]
    total = float(power.sum())
    if len(power) <= 1 or total <= 1e-8:
        return 0.0
    probability = power / total
    return float(
        -np.sum(probability * np.log(probability + 1e-12)) / math.log(len(power))
    )


def _robust_normalize(signal: np.ndarray) -> tuple[np.ndarray, float]:
    signal = np.asarray(signal, dtype=np.float32)
    q10, q90 = np.quantile(signal, [0.10, 0.90])
    spread = float(q90 - q10)
    if spread <= 1e-6:
        return np.zeros_like(signal), 0.0
    normalized = np.clip((signal - q10) / spread, 0.0, 2.0)
    return normalized.astype(np.float32), spread


def _event_peaks(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    smooth = gaussian_filter1d(signal.astype(np.float32), sigma=0.8, mode="nearest")
    normalized, spread = _robust_normalize(smooth)
    if spread <= 1e-6:
        return np.empty(0, dtype=np.int64), normalized, 0.0
    peaks, properties = find_peaks(normalized, prominence=0.35, distance=3)
    prominence = properties.get("prominences", np.empty(0, dtype=np.float32))
    boundary_peaks: list[int] = []
    boundary_prominence: list[float] = []
    if normalized[0] >= 0.70 and normalized[0] - float(np.min(normalized[:5])) >= 0.35:
        boundary_peaks.append(0)
        boundary_prominence.append(float(normalized[0] - np.min(normalized[:5])))
    if normalized[-1] >= 0.70 and normalized[-1] - float(np.min(normalized[-5:])) >= 0.35:
        boundary_peaks.append(len(normalized) - 1)
        boundary_prominence.append(float(normalized[-1] - np.min(normalized[-5:])))
    if boundary_peaks:
        peaks = np.concatenate([np.asarray(boundary_peaks, dtype=np.int64), peaks])
        prominence = np.concatenate(
            [np.asarray(boundary_prominence, dtype=np.float32), prominence]
        )
        order = np.argsort(peaks)
        peaks = peaks[order]
        prominence = prominence[order]
    quality = float(np.clip(spread / (np.median(np.abs(signal)) + spread + 1e-6), 0, 1))
    if len(prominence):
        quality *= float(np.clip(np.mean(prominence) / 0.75, 0.25, 1.0))
    return peaks.astype(np.int64), normalized, quality


def _peak_set_alignment(left: np.ndarray, right: np.ndarray, tolerance: float = 3.5) -> float:
    if not len(left) or not len(right):
        return 0.0
    distances_left = np.asarray([np.min(np.abs(right - value)) for value in left])
    distances_right = np.asarray([np.min(np.abs(left - value)) for value in right])
    score_left = np.exp(-0.5 * (distances_left / tolerance) ** 2).mean()
    score_right = np.exp(-0.5 * (distances_right / tolerance) ** 2).mean()
    return float(0.5 * (score_left + score_right))


def _quaternion_delta_angle(quaternion: np.ndarray) -> np.ndarray:
    q = quaternion.astype(np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.divide(q, norm, out=np.zeros_like(q), where=norm > 1e-8)
    if len(q) <= 1:
        return np.zeros(len(q), dtype=np.float32)
    dot = np.abs(np.sum(q[1:] * q[:-1], axis=1))
    angle = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
    return np.concatenate([[0.0], angle]).astype(np.float32)


def _single_excursion_score(signal: np.ndarray) -> tuple[float, float]:
    smooth = gaussian_filter1d(signal.astype(np.float32), sigma=1.2, mode="nearest")
    normalized, spread = _robust_normalize(smooth)
    if spread <= 1e-5:
        return 0.0, 0.0
    peaks, properties = find_peaks(normalized, prominence=0.30, distance=5)
    if not len(peaks):
        peak = int(np.argmax(normalized))
        peaks = np.asarray([peak])
        prominences = np.asarray([normalized[peak]])
    else:
        prominences = properties["prominences"]
    main_index = int(np.argmax(prominences))
    peak = int(peaks[main_index])
    centrality = float(np.clip(1.0 - abs(peak / 31.0 - 0.5) / 0.5, 0, 1))
    endpoint_return = float(
        np.exp(-2.5 * (abs(normalized[0] - normalized[-1])))
    )
    unimodal = float(1.0 / max(len(peaks), 1))
    plateau = float(np.mean(normalized >= 0.70 * max(float(normalized.max()), 1e-6)))
    plateau_score = float(np.clip(1.0 - abs(plateau - 0.25) / 0.35, 0, 1))
    score = 0.30 * centrality + 0.30 * endpoint_return + 0.25 * unimodal + 0.15 * plateau_score
    quality = float(np.clip(spread / (np.median(np.abs(signal)) + spread + 1e-6), 0, 1))
    return float(np.clip(score, 0, 1)), quality


def _periodicity(signal: np.ndarray) -> tuple[float, float, float, float]:
    peaks, normalized, quality = _event_peaks(signal)
    detrended = normalized - gaussian_filter1d(normalized, sigma=3.0, mode="nearest")
    autocorrelation = max(
        (_safe_corr(detrended, detrended, lag) for lag in range(3, 11)),
        default=0.0,
    )
    power = np.abs(np.fft.rfft(detrended)) ** 2
    concentration = (
        float(power[1:].max() / max(power[1:].sum(), 1e-8)) if len(power) > 2 else 0.0
    )
    if len(peaks) >= 3:
        intervals = np.diff(peaks).astype(np.float32)
        regularity = float(
            np.clip(1.0 - intervals.std() / max(intervals.mean(), 1e-6), 0, 1)
        )
    else:
        regularity = 0.0
    peak_gate = float(np.clip((len(peaks) - 2) / 3.0, 0, 1))
    periodicity = peak_gate * (
        0.40 * max(autocorrelation, 0.0)
        + 0.35 * concentration
        + 0.25 * regularity
    )
    irregularity = _spectral_entropy(signal) * (
        0.5 + 0.5 * (1.0 - regularity)
    )
    sparse_density = float(np.clip(len(peaks) / 8.0, 0, 1))
    return periodicity, sparse_density, irregularity, quality


def _coarse_grid(frames: np.ndarray) -> np.ndarray:
    height, width = frames.shape[1:]
    if height % GRID_HEIGHT or width % GRID_WIDTH:
        raise ValueError(f"cache resolution {height}x{width} is not grid divisible")
    return frames.reshape(
        len(frames),
        GRID_HEIGHT,
        height // GRID_HEIGHT,
        GRID_WIDTH,
        width // GRID_WIDTH,
    ).mean(axis=(2, 4))


def _motion_box(heat: np.ndarray) -> tuple[int, int, int, int, float]:
    height, width = heat.shape
    positive = heat[heat > 0]
    if len(positive) < 32:
        return 0, 0, width, height, 0.0
    threshold = float(np.quantile(positive, 0.80))
    yy, xx = np.where(heat >= threshold)
    if len(xx) < 32:
        return 0, 0, width, height, 0.0
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    pad_x = max(8, int(0.20 * (x1 - x0)))
    pad_y = max(8, int(0.20 * (y1 - y0)))
    x0, x1 = max(0, x0 - pad_x), min(width, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
    area = (x1 - x0) * (y1 - y0) / float(height * width)
    quality = float(np.clip((1.0 - area) / 0.75, 0, 1))
    return x0, y0, x1, y1, quality


def _motion_statistics(
    diff: np.ndarray, box: tuple[int, int, int, int, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, x1, y1, _ = box
    global_motion = diff.mean(axis=(1, 2))
    crop = diff[:, y0:y1, x0:x1]
    local_motion = crop.mean(axis=(1, 2))
    centers = np.full((len(diff), 2), 0.5, dtype=np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    for index, frame in enumerate(crop):
        mass = float(frame.sum())
        if mass > 1e-8:
            centers[index, 0] = float(((xx - x0) * frame).sum() / mass / max(x1 - x0 - 1, 1))
            centers[index, 1] = float(((yy - y0) * frame).sum() / mass / max(y1 - y0 - 1, 1))
    return global_motion, local_motion, centers


@dataclass
class EventCache:
    sample_ids: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    outer_folds: np.ndarray
    skeleton_tokens: np.ndarray
    imu_tokens: np.ndarray
    visual_tokens: np.ndarray
    modality_mask: np.ndarray
    event_targets: np.ndarray
    event_quality: np.ndarray
    event_names: np.ndarray
    event_sources: np.ndarray
    core_skeleton_features: np.ndarray
    core_depth_features: np.ndarray
    core_imu_features: np.ndarray


class P27R2EventExtractor:
    def __init__(
        self,
        manifest: Path = DEFAULT_MANIFEST,
        aligned_cache: Path = DEFAULT_ALIGNED_CACHE,
        imu_cache: Path = DEFAULT_IMU_CACHE,
    ) -> None:
        self.rows = read_csv(manifest)
        metadata = json.loads((aligned_cache / "metadata.json").read_text(encoding="utf-8"))
        self.locations = {
            sample_id: (int(offset), int(length))
            for sample_id, (offset, length) in zip(
                metadata["sample_ids"], metadata["offsets"], strict=True
            )
        }
        self.depth = np.load(aligned_cache / "depth_uint8.npy", mmap_mode="r", allow_pickle=False)
        self.ir = np.load(aligned_cache / "ir_uint8.npy", mmap_mode="r", allow_pickle=False)
        self.skeleton = np.load(
            aligned_cache / "skeleton_float32.npy", mmap_mode="r", allow_pickle=False
        )
        self.imu = np.load(imu_cache / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
        self.imu_time_mask = np.load(
            imu_cache / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False
        )
        self.imu_device_mask = np.load(
            imu_cache / "device_mask_uint8.npy", mmap_mode="r", allow_pickle=False
        )

    def extract(self) -> EventCache:
        count = len(self.rows)
        skeleton_tokens = np.zeros((count, EVENT_STEPS, 43), dtype=np.float32)
        imu_tokens = np.zeros((count, EVENT_STEPS, 60), dtype=np.float32)
        visual_tokens = np.zeros((count, EVENT_STEPS, 106), dtype=np.float32)
        modality_mask = np.zeros((count, 4), dtype=np.float32)
        event_targets = np.zeros((count, len(EVENT_NAMES)), dtype=np.float32)
        event_quality = np.zeros_like(event_targets)
        core_skeleton = np.zeros((count, EVENT_STEPS * 43 + 43 * 4), dtype=np.float32)
        core_depth = np.zeros(
            (count, EVENT_STEPS * (GRID_HEIGHT * GRID_WIDTH) + 4 * GRID_HEIGHT * GRID_WIDTH),
            dtype=np.float32,
        )
        core_imu = np.zeros((count, EVENT_STEPS * 30 + 30 * 4), dtype=np.float32)

        for index, row in enumerate(self.rows):
            location = self.locations.get(row["sample_id"])
            skeleton_ok = False
            visual_ok = False
            visual_signal = np.zeros(EVENT_STEPS, dtype=np.float32)
            visual_centers = np.full((EVENT_STEPS, 2), 0.5, dtype=np.float32)

            if location is not None and int(row["skeleton_usable"]) == 1:
                offset, length = location
                coordinates = np.asarray(
                    self.skeleton[offset : offset + length, :, :3], dtype=np.float32
                )
                if len(coordinates) >= 2 and np.isfinite(coordinates).all():
                    local, scale = _body_local(coordinates)
                    if 0.05 <= scale <= 2.0:
                        local = _interp_time(local)
                        selected = local[
                            :,
                            [
                                ROOT,
                                THORAX,
                                HEAD,
                                LEFT_SHOULDER,
                                LEFT_ELBOW,
                                LEFT_WRIST,
                                RIGHT_SHOULDER,
                                RIGHT_ELBOW,
                                RIGHT_WRIST,
                            ],
                        ]
                        left, right = local[:, LEFT_WRIST], local[:, RIGHT_WRIST]
                        head, thorax = local[:, HEAD], local[:, THORAX]
                        left_speed = np.linalg.norm(
                            np.diff(left, axis=0, prepend=left[:1]), axis=1
                        )
                        right_speed = np.linalg.norm(
                            np.diff(right, axis=0, prepend=right[:1]), axis=1
                        )
                        whole_speed = np.linalg.norm(
                            np.diff(local, axis=0, prepend=local[:1]), axis=2
                        ).mean(axis=1)
                        relations = np.stack(
                            [
                                np.linalg.norm(left - head, axis=1),
                                np.linalg.norm(right - head, axis=1),
                                np.linalg.norm(left - thorax, axis=1),
                                np.linalg.norm(right - thorax, axis=1),
                                np.linalg.norm(
                                    left - local[:, LEFT_SHOULDER], axis=1
                                ),
                                np.linalg.norm(
                                    right - local[:, RIGHT_SHOULDER], axis=1
                                ),
                                np.linalg.norm(left - right, axis=1),
                                left_speed,
                                right_speed,
                                whole_speed,
                            ],
                            axis=1,
                        )
                        delta_left = np.diff(left, axis=0, prepend=left[:1])
                        delta_right = np.diff(right, axis=0, prepend=right[:1])
                        token = np.concatenate(
                            [selected.reshape(EVENT_STEPS, -1), relations, delta_left, delta_right],
                            axis=1,
                        )
                        if token.shape[1] != 43:
                            raise AssertionError(token.shape)
                        skeleton_tokens[index] = token
                        core_skeleton[index] = np.concatenate(
                            [
                                token.reshape(-1),
                                token.mean(axis=0),
                                token.std(axis=0),
                                np.quantile(token, 0.10, axis=0),
                                np.quantile(token, 0.90, axis=0),
                            ]
                        )
                        self._skeleton_targets(
                            token,
                            event_targets[index],
                            event_quality[index],
                        )
                        modality_mask[index, 0] = 1.0
                        skeleton_ok = True

            imu_index = int(row["imu_cache_index"])
            if imu_index >= 0 and int(row["imu_usable"]) == 1:
                values = np.asarray(self.imu[imu_index], dtype=np.float32)
                time_mask = np.asarray(self.imu_time_mask[imu_index], dtype=np.float32)
                device_mask = np.asarray(self.imu_device_mask[imu_index], dtype=np.float32)
                invariant = np.zeros((5, EVENT_STEPS, 5), dtype=np.float32)
                robust_axes = np.zeros((5, EVENT_STEPS, 6), dtype=np.float32)
                for device in range(5):
                    mask = time_mask[device] > 0
                    if not mask.any():
                        continue
                    signal = values[device]
                    acceleration = signal[:, :3]
                    gyro = signal[:, 3:6]
                    quaternion = signal[:, 6:10]
                    acc_norm = np.linalg.norm(acceleration, axis=1)
                    gyro_norm = np.linalg.norm(gyro, axis=1)
                    jerk = np.linalg.norm(
                        np.diff(acceleration, axis=0, prepend=acceleration[:1]), axis=1
                    )
                    quat_norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
                    unit_quat = np.divide(
                        quaternion,
                        quat_norm,
                        out=np.zeros_like(quaternion),
                        where=quat_norm > 1e-8,
                    )
                    angle = 2.0 * np.arccos(np.clip(np.abs(unit_quat[:, 0]), 0, 1))
                    delta_angle = _quaternion_delta_angle(unit_quat)
                    invariant[device] = np.stack(
                        [acc_norm, gyro_norm, jerk, angle, delta_angle], axis=1
                    )
                    axes = np.concatenate([acceleration, gyro], axis=1)
                    median = np.median(axes[mask], axis=0)
                    q25, q75 = np.quantile(axes[mask], [0.25, 0.75], axis=0)
                    robust_axes[device] = (axes - median) / np.maximum(q75 - q25, 1e-4)
                imu_token = np.concatenate(
                    [
                        invariant.transpose(1, 0, 2).reshape(EVENT_STEPS, -1),
                        robust_axes.transpose(1, 0, 2).reshape(EVENT_STEPS, -1),
                        np.repeat(device_mask[None], EVENT_STEPS, axis=0),
                    ],
                    axis=1,
                )
                imu_tokens[index] = imu_token
                core_imu_signal = np.concatenate(
                    [
                        invariant.transpose(1, 0, 2).reshape(EVENT_STEPS, -1),
                        np.repeat(device_mask[None], EVENT_STEPS, axis=0),
                    ],
                    axis=1,
                )
                core_imu[index] = np.concatenate(
                    [
                        core_imu_signal.reshape(-1),
                        core_imu_signal.mean(axis=0),
                        core_imu_signal.std(axis=0),
                        np.quantile(core_imu_signal, 0.10, axis=0),
                        np.quantile(core_imu_signal, 0.90, axis=0),
                    ]
                )
                self._imu_targets(
                    invariant,
                    time_mask,
                    device_mask,
                    event_targets[index],
                    event_quality[index],
                )
                modality_mask[index, 1] = 1.0

            if location is not None and (
                int(row["depth_usable"]) == 1 or int(row["ir_usable"]) == 1
            ):
                offset, length = location
                depth_gray = (
                    np.asarray(self.depth[offset : offset + length], dtype=np.float32).mean(axis=3)
                    / 255.0
                )
                ir_gray = (
                    np.asarray(self.ir[offset : offset + length], dtype=np.float32) / 255.0
                )
                depth_diff = np.abs(
                    np.diff(depth_gray, axis=0, prepend=depth_gray[:1])
                )
                ir_diff = np.abs(np.diff(ir_gray, axis=0, prepend=ir_gray[:1]))
                combined_diff = 0.5 * (
                    depth_diff / max(float(np.quantile(depth_diff, 0.95)), 1e-5)
                    + ir_diff / max(float(np.quantile(ir_diff, 0.95)), 1e-5)
                )
                box = _motion_box(combined_diff.mean(axis=0))
                depth_global, depth_local, depth_center = _motion_statistics(depth_diff, box)
                ir_global, ir_local, ir_center = _motion_statistics(ir_diff, box)
                center = 0.5 * (depth_center + ir_center)
                depth_grid = _segment_reduce(_coarse_grid(depth_gray), EVENT_STEPS, "mean")
                ir_grid = _segment_reduce(_coarse_grid(ir_gray), EVENT_STEPS, "mean")
                scalars = np.stack(
                    [
                        _segment_reduce(depth_global[:, None], EVENT_STEPS, "mean")[:, 0],
                        _segment_reduce(depth_local[:, None], EVENT_STEPS, "mean")[:, 0],
                        _segment_reduce(depth_local[:, None], EVENT_STEPS, "max")[:, 0],
                        _segment_reduce(ir_global[:, None], EVENT_STEPS, "mean")[:, 0],
                        _segment_reduce(ir_local[:, None], EVENT_STEPS, "mean")[:, 0],
                        _segment_reduce(ir_local[:, None], EVENT_STEPS, "max")[:, 0],
                        _segment_reduce(center, EVENT_STEPS, "mean")[:, 0],
                        _segment_reduce(center, EVENT_STEPS, "mean")[:, 1],
                        np.full(EVENT_STEPS, box[4], dtype=np.float32),
                        np.full(
                            EVENT_STEPS,
                            (box[2] - box[0]) * (box[3] - box[1])
                            / float(depth_gray.shape[1] * depth_gray.shape[2]),
                            dtype=np.float32,
                        ),
                    ],
                    axis=1,
                )
                visual_token = np.concatenate(
                    [
                        depth_grid.reshape(EVENT_STEPS, -1),
                        ir_grid.reshape(EVENT_STEPS, -1),
                        scalars,
                    ],
                    axis=1,
                )
                visual_tokens[index] = visual_token
                depth_feature = depth_grid.reshape(EVENT_STEPS, -1)
                core_depth[index] = np.concatenate(
                    [
                        depth_feature.reshape(-1),
                        depth_feature.mean(axis=0),
                        depth_feature.std(axis=0),
                        np.quantile(depth_feature, 0.10, axis=0),
                        np.quantile(depth_feature, 0.90, axis=0),
                    ]
                )
                visual_signal = 0.5 * (scalars[:, 1] + scalars[:, 4])
                visual_centers = scalars[:, 6:8]
                self._visual_targets(
                    visual_signal,
                    visual_centers,
                    box[4],
                    event_targets[index],
                    event_quality[index],
                )
                modality_mask[index, 2] = float(int(row["depth_usable"]))
                modality_mask[index, 3] = float(int(row["ir_usable"]))
                visual_ok = True

            if visual_ok and modality_mask[index, 1] > 0:
                self._cross_target(
                    visual_signal,
                    imu_tokens[index],
                    event_targets[index],
                    event_quality[index],
                )

            if not skeleton_ok:
                event_quality[index, 5:11] = 0.0

        return EventCache(
            sample_ids=np.asarray([row["sample_id"] for row in self.rows]),
            labels=np.asarray([int(row["class_id"]) for row in self.rows], dtype=np.int64),
            subjects=np.asarray([row["user_id"] for row in self.rows]),
            outer_folds=np.asarray(
                [int(row["subject_fold"]) for row in self.rows], dtype=np.int64
            ),
            skeleton_tokens=skeleton_tokens,
            imu_tokens=imu_tokens,
            visual_tokens=visual_tokens,
            modality_mask=modality_mask,
            event_targets=event_targets,
            event_quality=event_quality,
            event_names=np.asarray(EVENT_NAMES),
            event_sources=np.asarray(EVENT_SOURCES),
            core_skeleton_features=core_skeleton,
            core_depth_features=core_depth,
            core_imu_features=core_imu,
        )

    @staticmethod
    def _skeleton_targets(
        token: np.ndarray, targets: np.ndarray, quality: np.ndarray
    ) -> None:
        left_distance, right_distance = token[:, 27], token[:, 28]
        proximity = np.exp(-((np.stack([left_distance, right_distance], axis=1) / 0.65) ** 2))
        side = int(np.argmax(proximity.mean(axis=0)))
        selected = gaussian_filter1d(proximity[:, side], sigma=1.0, mode="nearest")
        other = proximity[:, 1 - side]
        unilateral = selected * np.clip(selected - other, 0, 1)
        delta = np.diff(selected, prepend=selected[0])
        speed = token[:, 34 + side]
        still_weight = np.exp(-3.0 * speed)
        excursion = float(np.quantile(selected, 0.90) - np.quantile(selected, 0.10))
        phase_quality = float(np.clip(excursion / 0.35, 0, 1))
        targets[5] = float((selected * (1.0 - other)).mean())
        targets[6] = float(np.mean(selected * still_weight))
        targets[7] = float(np.clip(delta[delta > 0].sum() / 0.75, 0, 1))
        targets[8] = targets[6]
        targets[9] = float(np.clip((-delta[delta < 0]).sum() / 0.75, 0, 1))
        target_peak = int(np.argmax(selected))
        order = float(target_peak >= 4 and target_peak <= 27)
        targets[10] = float(
            order
            * min(targets[7], targets[9])
            * (0.4 + 0.6 * targets[8])
        )
        quality[5] = 1.0
        quality[6] = max(0.25, phase_quality)
        quality[7:11] = phase_quality

    @staticmethod
    def _imu_targets(
        invariant: np.ndarray,
        time_mask: np.ndarray,
        device_mask: np.ndarray,
        targets: np.ndarray,
        quality: np.ndarray,
    ) -> None:
        bilateral = bool(
            device_mask[LEFT_IMU] > 0
            and device_mask[RIGHT_IMU] > 0
            and time_mask[LEFT_IMU].mean() >= 0.60
            and time_mask[RIGHT_IMU].mean() >= 0.60
        )
        left = np.log1p(invariant[LEFT_IMU, :, 1]) + 0.25 * np.log1p(
            invariant[LEFT_IMU, :, 2]
        )
        right = np.log1p(invariant[RIGHT_IMU, :, 1]) + 0.25 * np.log1p(
            invariant[RIGHT_IMU, :, 2]
        )
        left_n, left_spread = _robust_normalize(left)
        right_n, right_spread = _robust_normalize(right)
        total_left = float(np.sum(left_n))
        total_right = float(np.sum(right_n))
        share = total_left / max(total_left + total_right, 1e-6)
        denominator = left_n + right_n + 1e-4
        dominance = (left_n - right_n) / denominator
        active = denominator > np.quantile(denominator, 0.35)
        sign = np.sign(dominance[active & (np.abs(dominance) > 0.20)])
        exchanges = float(np.sum(sign[1:] != sign[:-1])) if len(sign) > 1 else 0.0
        synchrony = max(_safe_corr(left_n, right_n, lag) for lag in (-1, 0, 1))
        alternation = max(
            0.0, -min(_safe_corr(left_n, right_n, lag) for lag in (-2, -1, 0, 1, 2))
        )
        targets[0] = share
        targets[1] = float(np.clip(np.mean(np.abs(dominance[active])) if active.any() else 0, 0, 1))
        targets[2] = float(np.clip(exchanges / 6.0, 0, 1))
        targets[3] = float(np.clip((synchrony + 1.0) / 2.0, 0, 1))
        targets[4] = float(np.clip(alternation, 0, 1))
        role_quality = float(
            bilateral
            * np.clip((left_spread + right_spread) / (left_spread + right_spread + 0.5), 0, 1)
        )
        quality[0:5] = role_quality

        dominant_device = LEFT_IMU if total_left >= total_right else RIGHT_IMU
        angle = invariant[dominant_device, :, 3]
        tilt, tilt_quality = _single_excursion_score(angle)
        combined = np.maximum(left_n, right_n)
        periodicity, sparse, irregularity, shape_quality = _periodicity(combined)
        activity_threshold = np.quantile(combined, 0.60)
        continuous = _longest_run(combined >= activity_threshold)
        targets[11] = tilt
        targets[12] = periodicity
        targets[13] = sparse
        targets[14] = irregularity
        targets[15] = continuous
        quality[11] = float(device_mask[dominant_device] > 0) * tilt_quality
        quality[12:16] = shape_quality

    @staticmethod
    def _visual_targets(
        motion: np.ndarray,
        center: np.ndarray,
        roi_quality: float,
        targets: np.ndarray,
        quality: np.ndarray,
    ) -> None:
        _, normalized, motion_quality = _event_peaks(motion)
        active = normalized > 0.30
        if active.sum() >= 3:
            selected = center[active]
            dispersion = float(np.linalg.norm(selected - np.median(selected, axis=0), axis=1).mean())
            targets[17] = float(np.exp(-4.0 * dispersion))
        quality[17] = float(roi_quality * motion_quality * min(1.0, active.sum() / 6.0))

    @staticmethod
    def _cross_target(
        visual_motion: np.ndarray,
        imu_token: np.ndarray,
        targets: np.ndarray,
        quality: np.ndarray,
    ) -> None:
        visual_peaks, _, visual_quality = _event_peaks(visual_motion)
        left_activity = np.log1p(imu_token[:, LEFT_IMU * 5 + 1]) + 0.25 * np.log1p(
            imu_token[:, LEFT_IMU * 5 + 2]
        )
        right_activity = np.log1p(imu_token[:, RIGHT_IMU * 5 + 1]) + 0.25 * np.log1p(
            imu_token[:, RIGHT_IMU * 5 + 2]
        )
        imu_peaks, _, imu_quality = _event_peaks(np.maximum(left_activity, right_activity))
        targets[16] = _peak_set_alignment(visual_peaks, imu_peaks)
        count_balance = (
            min(len(visual_peaks), len(imu_peaks)) / max(len(visual_peaks), len(imu_peaks))
            if len(visual_peaks) and len(imu_peaks)
            else 0.0
        )
        quality[16] = float(min(visual_quality, imu_quality) * count_balance)


def save_event_cache(path: Path, cache: EventCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{field: getattr(cache, field) for field in cache.__dataclass_fields__},
    )


def load_event_cache(path: Path) -> EventCache:
    with np.load(path, allow_pickle=False) as source:
        return EventCache(
            **{field: source[field] for field in EventCache.__dataclass_fields__}
        )
