from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from p27_data import read_csv


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_MANIFEST = PROJECT_DIR / "runs" / "p27_0_audit" / "p27_train_manifest.csv"
DEFAULT_ALIGNED_CACHE = PROJECT_DIR / "cache" / "aligned_192x144"
DEFAULT_IMU_CACHE = PROJECT_DIR / "cache" / "imu_32"
DEFAULT_LOCATORS = PROJECT_DIR / "runs" / "p12_fold_pure_locator_predictions"

LEFT_WRIST = 13
RIGHT_WRIST = 16
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 14
TORSO = 8
PELVIS = 0
HEAD = 10
LEFT_IMU = 1
RIGHT_IMU = 2
TORSO_IMU = 0

GROUP_TARGET_NAMES = {
    "coarse_head_event": [
        "unilateral_head_proximity",
        "head_zone_dwell",
        "approach_phase",
        "hold_phase",
        "return_phase",
    ],
    "hand_roles": [
        "left_activity_share",
        "role_stability",
        "role_exchange",
        "bilateral_synchrony",
        "bilateral_alternation",
    ],
    "motion_shape": [
        "periodicity",
        "spectral_irregularity",
        "sparse_event_count",
        "event_interval_regularity",
    ],
    "visual_imu_soft_alignment": [
        "local_visual_motion",
        "work_center_stability",
        "visual_wrist_peak_alignment",
    ],
    "signed_tilt_diagnostic": [
        "left_relative_quaternion_signed_peak",
        "right_relative_quaternion_signed_peak",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _positions(length: int, count: int = 12) -> np.ndarray:
    if length <= 0:
        return np.zeros(count, dtype=np.int64)
    if length <= count:
        return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
    boundaries = np.floor(np.linspace(0, length, count + 1)).astype(np.int64)
    positions = []
    for index in range(count):
        start = int(boundaries[index])
        stop = max(start + 1, int(boundaries[index + 1]))
        positions.append(min(length - 1, (start + stop - 1) // 2))
    return np.asarray(positions, dtype=np.int64)


def _resample_masked(signal: np.ndarray, mask: np.ndarray, count: int = 32) -> np.ndarray:
    valid = np.flatnonzero(mask > 0)
    if not len(valid):
        return np.zeros(count, dtype=np.float32)
    target = np.linspace(0, len(signal) - 1, count)
    return np.interp(target, valid, signal[valid]).astype(np.float32)


def _safe_corr(left: np.ndarray, right: np.ndarray, lag: int = 0) -> float:
    if lag > 0:
        left, right = left[:-lag], right[lag:]
    elif lag < 0:
        left, right = left[-lag:], right[:lag]
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-8 else 0.0


def _max_autocorrelation(signal: np.ndarray) -> float:
    centered = signal - signal.mean()
    denominator = float(np.mean(centered**2))
    if denominator <= 1e-8:
        return 0.0
    values = [
        float(np.mean(centered[:-lag] * centered[lag:]) / denominator)
        for lag in range(2, min(9, len(signal) - 1))
    ]
    return float(np.clip(max(values, default=0.0), 0.0, 1.0))


def _spectral_entropy(signal: np.ndarray) -> float:
    centered = signal - signal.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    power = power[1:]
    total = float(power.sum())
    if total <= 1e-8 or len(power) <= 1:
        return 0.0
    probabilities = power / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return float(entropy / math.log(len(power)))


def _motion_center(diff: np.ndarray, box: np.ndarray) -> tuple[float, float, float]:
    height, width = diff.shape
    x0 = int(np.clip(round(float(box[0]) * width), 0, width - 1))
    y0 = int(np.clip(round(float(box[1]) * height), 0, height - 1))
    x1 = int(np.clip(round(float(box[2]) * width), x0 + 1, width))
    y1 = int(np.clip(round(float(box[3]) * height), y0 + 1, height))
    crop = diff[y0:y1, x0:x1].astype(np.float64)
    mass = float(crop.sum())
    if mass <= 1e-8:
        return 0.5, 0.5, 0.0
    yy, xx = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
    cx = float((xx * crop).sum() / mass / max(crop.shape[1] - 1, 1))
    cy = float((yy * crop).sum() / mass / max(crop.shape[0] - 1, 1))
    return cx, cy, mass / float(crop.size)


@dataclass
class RawSignals:
    sample_ids: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    folds: np.ndarray
    skeleton_valid: np.ndarray
    imu_valid: np.ndarray
    bilateral_imu_valid: np.ndarray
    visual_valid: np.ndarray
    visual_imu_valid: np.ndarray
    head_distance: np.ndarray
    skeleton_activity: np.ndarray
    skeleton_sync: np.ndarray
    skeleton_features: np.ndarray
    imu_activity: np.ndarray
    imu_features: np.ndarray
    quaternion_signed: np.ndarray
    visual_motion: np.ndarray
    visual_center: np.ndarray
    visual_features: np.ndarray


class P27RSignalExtractor:
    def __init__(
        self,
        manifest: Path = DEFAULT_MANIFEST,
        aligned_cache: Path = DEFAULT_ALIGNED_CACHE,
        imu_cache: Path = DEFAULT_IMU_CACHE,
        locator_dir: Path = DEFAULT_LOCATORS,
        outer_fold: int = 0,
        roi_padding: float = 0.30,
    ) -> None:
        self.rows = read_csv(manifest)
        self.outer_fold = int(outer_fold)
        self.roi_padding = float(roi_padding)
        metadata = json.loads((aligned_cache / "metadata.json").read_text(encoding="utf-8"))
        self.locations = {
            sample_id: (int(offset), int(length))
            for sample_id, (offset, length) in zip(
                metadata["sample_ids"],
                metadata["offsets"],
                strict=True,
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
        locator_path = locator_dir / f"fold_{self.outer_fold}_locator_predictions.csv"
        self.locators = {
            row["sample_id"]: np.asarray(
                [
                    float(row["x0"]) / float(row["raw_width"]),
                    float(row["y0"]) / float(row["raw_height"]),
                    float(row["x1"]) / float(row["raw_width"]),
                    float(row["y1"]) / float(row["raw_height"]),
                ],
                dtype=np.float32,
            )
            for row in read_csv(locator_path)
        }

    def _box(self, sample_id: str) -> np.ndarray:
        box = self.locators.get(
            sample_id, np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        ).copy()
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        box[[0, 2]] += np.asarray([-1.0, 1.0]) * self.roi_padding * width
        box[[1, 3]] += np.asarray([-1.0, 1.0]) * self.roi_padding * height
        return np.clip(box, 0.0, 1.0)

    def extract(self) -> RawSignals:
        n = len(self.rows)
        head_distance = np.zeros((n, 12, 2), dtype=np.float32)
        skeleton_activity = np.zeros((n, 12, 2), dtype=np.float32)
        skeleton_sync = np.zeros((n, 12), dtype=np.float32)
        skeleton_features = np.zeros((n, 12 * 17 * 3), dtype=np.float32)
        imu_activity = np.zeros((n, 32, 3), dtype=np.float32)
        imu_features = np.zeros((n, 32 * 3 + 32 * 2 * 3), dtype=np.float32)
        quaternion_signed = np.zeros((n, 2), dtype=np.float32)
        visual_motion = np.zeros((n, 12), dtype=np.float32)
        visual_center = np.zeros((n, 12, 2), dtype=np.float32)
        visual_features = np.zeros((n, 12 * 4), dtype=np.float32)
        skeleton_valid = np.zeros(n, dtype=bool)
        imu_valid = np.zeros(n, dtype=bool)
        bilateral_imu_valid = np.zeros(n, dtype=bool)
        visual_valid = np.zeros(n, dtype=bool)

        for index, row in enumerate(self.rows):
            sample_id = row["sample_id"]
            location = self.locations.get(sample_id)
            if location is not None and int(row["skeleton_usable"]) == 1:
                offset, length = location
                positions = _positions(length) + offset
                sk = np.asarray(self.skeleton[positions, :, :3], dtype=np.float32)
                shoulder = np.linalg.norm(
                    sk[:, LEFT_SHOULDER] - sk[:, RIGHT_SHOULDER], axis=1
                )
                torso = np.linalg.norm(sk[:, TORSO] - sk[:, PELVIS], axis=1)
                scale = float(np.median(0.5 * (shoulder + torso)))
                if np.isfinite(sk).all() and 0.08 <= scale <= 2.0:
                    centered = (sk - sk[:, TORSO : TORSO + 1]) / scale
                    left = centered[:, LEFT_WRIST]
                    right = centered[:, RIGHT_WRIST]
                    head = centered[:, HEAD]
                    head_distance[index, :, 0] = np.linalg.norm(left - head, axis=1)
                    head_distance[index, :, 1] = np.linalg.norm(right - head, axis=1)
                    left_velocity = np.diff(left, axis=0, prepend=left[:1])
                    right_velocity = np.diff(right, axis=0, prepend=right[:1])
                    skeleton_activity[index, :, 0] = np.linalg.norm(left_velocity, axis=1)
                    skeleton_activity[index, :, 1] = np.linalg.norm(right_velocity, axis=1)
                    denominator = (
                        np.linalg.norm(left_velocity, axis=1)
                        * np.linalg.norm(right_velocity, axis=1)
                    )
                    numerator = np.sum(left_velocity * right_velocity, axis=1)
                    skeleton_sync[index] = np.divide(
                        numerator,
                        denominator,
                        out=np.zeros_like(numerator),
                        where=denominator > 1e-6,
                    )
                    skeleton_features[index] = centered.reshape(-1)
                    skeleton_valid[index] = True

            imu_index = int(row["imu_cache_index"])
            if imu_index >= 0 and int(row["imu_usable"]) == 1:
                array = np.asarray(self.imu[imu_index], dtype=np.float32)
                time_mask = np.asarray(self.imu_time_mask[imu_index], dtype=np.float32)
                device_mask = np.asarray(self.imu_device_mask[imu_index], dtype=np.float32)
                raw_activities: list[np.ndarray] = []
                for device in (LEFT_IMU, RIGHT_IMU, TORSO_IMU):
                    gyro = np.linalg.norm(array[device, :, 3:6], axis=1)
                    acceleration = array[device, :, :3]
                    acceleration_delta = np.linalg.norm(
                        np.diff(acceleration, axis=0, prepend=acceleration[:1]), axis=1
                    )
                    activity = np.log1p(np.maximum(gyro, 0.0)) + 0.25 * np.log1p(
                        np.maximum(acceleration_delta, 0.0)
                    )
                    raw_activities.append(
                        _resample_masked(activity, time_mask[device], count=32)
                    )
                imu_activity[index] = np.stack(raw_activities, axis=1)
                quaternion = []
                for device in (LEFT_IMU, RIGHT_IMU):
                    valid = time_mask[device] > 0
                    vector = array[device, valid, 7:10]
                    if len(vector):
                        component = int(np.argmax(np.ptp(vector, axis=0)))
                        selected = vector[:, component]
                        quaternion.append(
                            float(selected[np.argmax(np.abs(selected))])
                        )
                    else:
                        quaternion.append(0.0)
                quaternion_signed[index] = quaternion
                quaternion_series = []
                for device in (LEFT_IMU, RIGHT_IMU):
                    for channel in range(7, 10):
                        quaternion_series.append(
                            _resample_masked(
                                array[device, :, channel], time_mask[device], count=32
                            )
                        )
                imu_features[index] = np.concatenate(
                    [imu_activity[index].T.reshape(-1), np.concatenate(quaternion_series)]
                )
                imu_valid[index] = bool(
                    device_mask[[LEFT_IMU, RIGHT_IMU, TORSO_IMU]].sum() >= 1
                )
                bilateral_imu_valid[index] = bool(
                    device_mask[LEFT_IMU] > 0
                    and device_mask[RIGHT_IMU] > 0
                    and time_mask[LEFT_IMU].mean() >= 0.6
                    and time_mask[RIGHT_IMU].mean() >= 0.6
                )

            if location is not None and (
                int(row["depth_usable"]) == 1 or int(row["ir_usable"]) == 1
            ):
                offset, length = location
                positions = _positions(length) + offset
                box = self._box(sample_id)
                modality_motion: list[np.ndarray] = []
                modality_centers: list[np.ndarray] = []
                if int(row["depth_usable"]) == 1:
                    frames = np.asarray(self.depth[positions], dtype=np.float32).mean(axis=3)
                    diff = np.abs(np.diff(frames, axis=0, prepend=frames[:1])) / 255.0
                    motion, centers = self._visual_statistics(diff, box)
                    modality_motion.append(motion)
                    modality_centers.append(centers)
                if int(row["ir_usable"]) == 1:
                    frames = np.asarray(self.ir[positions], dtype=np.float32)
                    diff = np.abs(np.diff(frames, axis=0, prepend=frames[:1])) / 255.0
                    motion, centers = self._visual_statistics(diff, box)
                    modality_motion.append(motion)
                    modality_centers.append(centers)
                visual_motion[index] = np.mean(modality_motion, axis=0)
                visual_center[index] = np.mean(modality_centers, axis=0)
                visual_features[index] = np.concatenate(
                    [
                        visual_motion[index],
                        visual_center[index, :, 0],
                        visual_center[index, :, 1],
                        np.linalg.norm(
                            np.diff(
                                visual_center[index],
                                axis=0,
                                prepend=visual_center[index, :1],
                            ),
                            axis=1,
                        ),
                    ]
                )
                visual_valid[index] = True

        return RawSignals(
            sample_ids=np.asarray([row["sample_id"] for row in self.rows]),
            labels=np.asarray([int(row["class_id"]) for row in self.rows], dtype=np.int64),
            subjects=np.asarray([row["user_id"] for row in self.rows]),
            folds=np.asarray([int(row["subject_fold"]) for row in self.rows], dtype=np.int64),
            skeleton_valid=skeleton_valid,
            imu_valid=imu_valid,
            bilateral_imu_valid=bilateral_imu_valid,
            visual_valid=visual_valid,
            visual_imu_valid=visual_valid & bilateral_imu_valid,
            head_distance=head_distance,
            skeleton_activity=skeleton_activity,
            skeleton_sync=skeleton_sync,
            skeleton_features=skeleton_features,
            imu_activity=imu_activity,
            imu_features=imu_features,
            quaternion_signed=quaternion_signed,
            visual_motion=visual_motion,
            visual_center=visual_center,
            visual_features=visual_features,
        )

    @staticmethod
    def _visual_statistics(
        diff: np.ndarray, box: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        motion = np.zeros(len(diff), dtype=np.float32)
        centers = np.full((len(diff), 2), 0.5, dtype=np.float32)
        height, width = diff.shape[1:]
        x0 = int(np.clip(round(float(box[0]) * width), 0, width - 1))
        y0 = int(np.clip(round(float(box[1]) * height), 0, height - 1))
        x1 = int(np.clip(round(float(box[2]) * width), x0 + 1, width))
        y1 = int(np.clip(round(float(box[3]) * height), y0 + 1, height))
        for index, image in enumerate(diff):
            crop = image[y0:y1, x0:x1]
            motion[index] = float(crop.mean())
            cx, cy, _ = _motion_center(image, box)
            centers[index] = [cx, cy]
        return motion, centers


def fit_fold_parameters(
    raw: RawSignals, config: dict[str, Any], outer_fold: int
) -> dict[str, float]:
    train = raw.folds != int(outer_fold)
    fold_fit = config["fold_fit"]
    skeleton = train & raw.skeleton_valid
    imu = train & raw.bilateral_imu_valid
    visual = train & raw.visual_valid
    head_values = raw.head_distance[skeleton].reshape(-1)
    head_threshold = float(
        np.quantile(head_values, float(fold_fit["head_distance_quantile"]))
    )
    proximity = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (raw.head_distance[skeleton] - head_threshold)
                / max(0.15 * head_threshold, 1e-4),
                -30,
                30,
            )
        )
    )
    max_proximity = proximity.max(axis=2)
    phase_velocity = np.abs(
        np.diff(max_proximity, axis=1, prepend=max_proximity[:, :1])
    ).reshape(-1)
    phase_scale = float(
        np.quantile(phase_velocity, float(fold_fit["phase_velocity_quantile"]))
    )
    activity_values = raw.imu_activity[imu, :, :2].reshape(-1)
    activity_scale = float(
        np.quantile(activity_values, float(fold_fit["imu_activity_scale_quantile"]))
    )
    normalized_activity = raw.imu_activity[imu, :, :2] / max(activity_scale, 1e-6)
    prominences = np.abs(np.diff(normalized_activity, axis=1)).reshape(-1)
    peak_prominence = float(
        np.quantile(prominences, float(fold_fit["imu_peak_prominence_quantile"]))
    )
    visual_scale = float(
        np.quantile(
            raw.visual_motion[visual].reshape(-1),
            float(fold_fit["visual_motion_scale_quantile"]),
        )
    )
    return {
        "head_distance_threshold": head_threshold,
        "phase_velocity_scale": max(phase_scale, 1e-3),
        "imu_activity_scale": max(activity_scale, 1e-3),
        "imu_peak_prominence": max(peak_prominence, 1e-3),
        "visual_motion_scale": max(visual_scale, 1e-5),
        "fit_samples_skeleton": int(skeleton.sum()),
        "fit_samples_bilateral_imu": int(imu.sum()),
        "fit_samples_visual": int(visual.sum()),
    }


def build_weak_targets(
    raw: RawSignals, parameters: dict[str, float]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    threshold = parameters["head_distance_threshold"]
    proximity = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (raw.head_distance - threshold) / max(0.15 * threshold, 1e-4),
                -30,
                30,
            )
        )
    )
    chosen = proximity.max(axis=2)
    unilateral = chosen * np.abs(proximity[:, :, 0] - proximity[:, :, 1])
    delta = np.diff(chosen, axis=1, prepend=chosen[:, :1])
    phase_scale = parameters["phase_velocity_scale"]
    approach = np.clip(delta / phase_scale, 0.0, 1.0)
    return_phase = np.clip(-delta / phase_scale, 0.0, 1.0)
    hold = chosen * np.exp(-np.abs(delta) / phase_scale)
    head_sequence = np.stack([unilateral, chosen, approach, hold, return_phase], axis=2)
    head_clip = head_sequence.mean(axis=1)

    activity = np.clip(
        raw.imu_activity[:, :, :2] / parameters["imu_activity_scale"], 0.0, 3.0
    )
    left = activity[:, :, 0]
    right = activity[:, :, 1]
    share = left / np.maximum(left + right, 1e-6)
    dominance = share - 0.5
    role_stability = np.abs(dominance.mean(axis=1)) * 2.0
    sign = np.where(dominance > 0.1, 1, np.where(dominance < -0.1, -1, 0))
    exchange = np.zeros(len(raw.sample_ids), dtype=np.float32)
    for index in range(len(exchange)):
        nonzero = sign[index][sign[index] != 0]
        exchange[index] = (
            float(np.sum(nonzero[1:] != nonzero[:-1]) / 4.0) if len(nonzero) > 1 else 0.0
        )
    synchrony = np.asarray(
        [
            max(_safe_corr(l, r, lag) for lag in (-1, 0, 1))
            for l, r in zip(left, right, strict=True)
        ],
        dtype=np.float32,
    )
    alternation = np.asarray(
        [
            max(0.0, -min(_safe_corr(l, r, lag) for lag in (-2, -1, 0, 1, 2)))
            for l, r in zip(left, right, strict=True)
        ],
        dtype=np.float32,
    )
    hand_roles = np.stack(
        [
            share.mean(axis=1),
            np.clip(role_stability, 0.0, 1.0),
            np.clip(exchange, 0.0, 1.0),
            np.clip((synchrony + 1.0) / 2.0, 0.0, 1.0),
            np.clip(alternation, 0.0, 1.0),
        ],
        axis=1,
    )

    periodicity = np.zeros(len(raw.sample_ids), dtype=np.float32)
    irregularity = np.zeros_like(periodicity)
    event_count = np.zeros_like(periodicity)
    interval_regularity = np.zeros_like(periodicity)
    combined = np.maximum(left, right)
    for index, signal in enumerate(combined):
        periodicity[index] = _max_autocorrelation(signal)
        irregularity[index] = _spectral_entropy(signal)
        peaks, _ = find_peaks(
            signal,
            prominence=parameters["imu_peak_prominence"],
            distance=2,
        )
        event_count[index] = min(float(len(peaks)) / 8.0, 1.0)
        if len(peaks) >= 3:
            intervals = np.diff(peaks).astype(np.float32)
            coefficient = float(intervals.std() / max(intervals.mean(), 1e-6))
            interval_regularity[index] = float(np.clip(1.0 - coefficient, 0.0, 1.0))
    motion_shape = np.stack(
        [periodicity, irregularity, event_count, interval_regularity], axis=1
    )

    visual = np.clip(
        raw.visual_motion / parameters["visual_motion_scale"], 0.0, 3.0
    )
    center_delta = np.linalg.norm(
        np.diff(raw.visual_center, axis=1, prepend=raw.visual_center[:, :1]), axis=2
    )
    work_center_stability = np.exp(-5.0 * center_delta.mean(axis=1))
    visual_alignment = np.zeros(len(raw.sample_ids), dtype=np.float32)
    imu_resampled = np.stack(
        [
            np.interp(
                np.linspace(0, 31, 12),
                np.arange(32),
                combined[index],
            )
            for index in range(len(raw.sample_ids))
        ]
    )
    for index in range(len(raw.sample_ids)):
        visual_alignment[index] = max(
            0.0,
            max(
                _safe_corr(visual[index], imu_resampled[index], lag)
                for lag in (-1, 0, 1)
            ),
        )
    visual_imu = np.stack(
        [
            np.clip(visual.mean(axis=1), 0.0, 1.0),
            np.clip(work_center_stability, 0.0, 1.0),
            np.clip(visual_alignment, 0.0, 1.0),
        ],
        axis=1,
    )
    signed_tilt = np.clip((raw.quaternion_signed + 1.0) / 2.0, 0.0, 1.0)

    targets = {
        "coarse_head_event": head_clip.astype(np.float32),
        "hand_roles": hand_roles.astype(np.float32),
        "motion_shape": motion_shape.astype(np.float32),
        "visual_imu_soft_alignment": visual_imu.astype(np.float32),
        "signed_tilt_diagnostic": signed_tilt.astype(np.float32),
    }
    sequences = {
        "coarse_head_event": head_sequence.astype(np.float32),
        "left_activity_share": share.astype(np.float32),
        "visual_motion": visual.astype(np.float32),
        "imu_wrist_activity": activity.astype(np.float32),
    }
    masks = {
        "coarse_head_event": raw.skeleton_valid.astype(np.float32),
        "hand_roles": raw.bilateral_imu_valid.astype(np.float32),
        "motion_shape": raw.bilateral_imu_valid.astype(np.float32),
        "visual_imu_soft_alignment": raw.visual_imu_valid.astype(np.float32),
        "signed_tilt_diagnostic": raw.bilateral_imu_valid.astype(np.float32),
    }
    return targets, sequences, masks


def feature_groups(raw: RawSignals) -> dict[str, np.ndarray]:
    skeleton_aux = np.concatenate(
        [
            raw.head_distance.reshape(len(raw.sample_ids), -1),
            raw.skeleton_activity.reshape(len(raw.sample_ids), -1),
            raw.skeleton_sync,
        ],
        axis=1,
    )
    roles = np.concatenate(
        [
            raw.skeleton_activity.reshape(len(raw.sample_ids), -1),
            raw.imu_activity[:, :, :2].reshape(len(raw.sample_ids), -1),
        ],
        axis=1,
    )
    shared = np.concatenate(
        [
            raw.visual_features,
            raw.imu_activity[:, :, :2].reshape(len(raw.sample_ids), -1),
        ],
        axis=1,
    )
    return {
        "coarse_head_event": skeleton_aux,
        "hand_roles": roles,
        "motion_shape": raw.imu_features,
        "visual_imu_soft_alignment": shared,
        "signed_tilt_diagnostic": raw.imu_features,
    }


def save_weak_label_npz(
    path: Path,
    raw: RawSignals,
    targets: dict[str, np.ndarray],
    sequences: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    parameters: dict[str, float],
) -> None:
    arrays: dict[str, Any] = {
        "sample_ids": raw.sample_ids,
        "labels": raw.labels,
        "subjects": raw.subjects,
        "folds": raw.folds,
        "fold_parameters_json": np.asarray(
            json.dumps(parameters, sort_keys=True), dtype=np.str_
        ),
    }
    for group, value in targets.items():
        arrays[f"target__{group}"] = value
        arrays[f"mask__{group}"] = masks[group]
        arrays[f"names__{group}"] = np.asarray(GROUP_TARGET_NAMES[group])
    for name, value in sequences.items():
        arrays[f"sequence__{name}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
