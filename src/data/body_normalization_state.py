from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from src.data.clean_skeleton_segments import (
    SkeletonSegments,
    fit_skeleton_normalization,
)
from src.data.raw_imu_segments import IMUSegments, fit_imu_normalization


def _frozen_array(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


@dataclass(frozen=True)
class BodyNormalizationState:
    skeleton_mean: np.ndarray
    skeleton_std: np.ndarray
    imu_mean: np.ndarray
    imu_std: np.ndarray
    fit_sample_ids: tuple[str, ...]
    fit_user_ids: tuple[str, ...]
    skeleton_samples: int
    imu_samples: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "skeleton_mean", _frozen_array(self.skeleton_mean))
        object.__setattr__(self, "skeleton_std", _frozen_array(self.skeleton_std))
        object.__setattr__(self, "imu_mean", _frozen_array(self.imu_mean))
        object.__setattr__(self, "imu_std", _frozen_array(self.imu_std))


def body_normalization_provenance(
    state: BodyNormalizationState,
) -> dict[str, Any]:
    return {
        "fit_sample_ids": list(state.fit_sample_ids),
        "fit_user_ids": list(state.fit_user_ids),
        "skeleton_mean_sha256": _array_sha256(state.skeleton_mean),
        "skeleton_std_sha256": _array_sha256(state.skeleton_std),
        "imu_mean_sha256": _array_sha256(state.imu_mean),
        "imu_std_sha256": _array_sha256(state.imu_std),
        "skeleton_samples": int(state.skeleton_samples),
        "imu_samples": int(state.imu_samples),
    }


def fit_body_normalization_state(
    dataset: Any, fit_indices: np.ndarray
) -> BodyNormalizationState:
    if not hasattr(dataset, "trials"):
        raise TypeError("body normalization requires canonical trial access")
    skeleton_samples: list[SkeletonSegments] = []
    imu_samples: list[IMUSegments] = []
    fit_sample_ids: list[str] = []
    fit_user_ids: set[str] = set()
    for index in np.asarray(fit_indices, dtype=np.int64).tolist():
        trial = dataset.trials[index]
        fit_sample_ids.append(str(trial.sample_id))
        fit_user_ids.add(str(trial.user_id))
        skeleton = dataset.skeleton_loader(trial)
        if bool(skeleton["modality_usable"]):
            skeleton_samples.append(
                SkeletonSegments(
                    features=skeleton["values"],
                    mask=skeleton["mask"],
                    quality=skeleton["quality"],
                )
            )
        imu = dataset.imu_loader(trial)
        if bool(imu["modality_usable"]):
            imu_samples.append(
                IMUSegments(
                    features=imu["values"],
                    role_mask=imu["role_mask"],
                    quality=imu["quality"],
                )
            )
    if not skeleton_samples or not imu_samples:
        raise ValueError("fit scope lacks Skeleton or IMU normalization samples")
    skeleton_mean, skeleton_std = fit_skeleton_normalization(skeleton_samples)
    imu_mean, imu_std = fit_imu_normalization(imu_samples)
    return BodyNormalizationState(
        skeleton_mean=skeleton_mean,
        skeleton_std=skeleton_std,
        imu_mean=imu_mean,
        imu_std=imu_std,
        fit_sample_ids=tuple(fit_sample_ids),
        fit_user_ids=tuple(sorted(fit_user_ids)),
        skeleton_samples=len(skeleton_samples),
        imu_samples=len(imu_samples),
    )


def apply_body_normalization_state(
    dataset: Any, state: BodyNormalizationState
) -> None:
    if not hasattr(dataset.skeleton_loader, "set_normalization"):
        raise TypeError("Skeleton loader cannot accept normalization")
    if not hasattr(dataset.imu_loader, "set_normalization"):
        raise TypeError("IMU loader cannot accept normalization")
    dataset.skeleton_loader.set_normalization(
        state.skeleton_mean, state.skeleton_std
    )
    dataset.imu_loader.set_normalization(state.imu_mean, state.imu_std)
