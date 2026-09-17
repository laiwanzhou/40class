"""Source-only data contract for the configurable four-modality teacher.

This module deliberately has no H3 path or H3 cohort loader.  It constructs the
H1 exploration, E0 source-only, and H2 confirmation cohorts from frozen OOF
features, while grouping every input under one of four physical modalities:
IR, Depth, Skeleton, and IMU.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from p90_teacher_fusion_audit import align


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

VMAE_PATH = PROJECT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz"
IV2_PATH = PROJECT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz"
DEPTH_PATH = PROJECT / "runs/p91_videomaev2_depth_fold0_v1/complete_features.npz"
MOTIONBERT_PATH = PROJECT / "runs/p90_motionbert_teacher_v1/features_pretrain_front_t81.npz"
HDGCN_PATH = PROJECT / "runs/p91_hdgcn_ntu60_xsub_fold0_v1/complete_features.npz"
STATISTICS_PATH = HERE / "runs/p89_crossmodal_statistics_v1/crossmodal_statistics.npz"
MOTION_CACHE = HERE / "runs/p86_motion_window_cache_t16_v1"
ROUTER_PATH = PROJECT / "runs/p90_crossuser_visual_router_v1/full_predictions.npz"
E0_BASE_PATH = HERE / "runs/p87_sequence_decoder_v1/oof_predictions.npz"
STRUCTURED_PATH = HERE / "runs/p87s_holdout1_structured_targets_v1/structured_targets.npz"
VMAE_OOF_PATH = PROJECT / "runs/p90_videomaev2_distilled_teacher_v1/oof_logits.npz"
IV2_OOF_PATH = PROJECT / "runs/p90_internvideo2_l_k400_teacher_v1/oof_logits.npz"
SKELETON_OOF_PATH = HERE / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz"
MOTIONBERT_OOF_PATH = (
    PROJECT / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front_linear_oof.npz"
)
IMU_OOF_PATH = (
    PROJECT / "runs/p90_imu_teacher_blend_v1/imu_p90_sensorwise_plus_deep_crossfit_oof.npz"
)

E0_USERS = frozenset(("user1", "user2", "user21"))
CANONICAL_MODALITIES = ("ir", "depth", "skeleton", "imu")
SAMPLE_PATTERN = re.compile(r"^[^_]+__c(?P<class_id>\d{2})__(?P<user>user\d+)__")


def parse_sample_identity(sample_id: str) -> tuple[int, str]:
    match = SAMPLE_PATTERN.match(str(sample_id))
    if match is None:
        raise ValueError(f"unexpected sample id: {sample_id}")
    return int(match.group("class_id")), match.group("user")


def validate_modalities(modalities: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(value).strip().lower() for value in modalities))
    unknown = sorted(set(selected) - set(CANONICAL_MODALITIES))
    if unknown:
        raise ValueError(f"unknown modalities: {unknown}")
    if not selected:
        raise ValueError("at least one modality must be enabled")
    return tuple(name for name in CANONICAL_MODALITIES if name in selected)


def _read_npz_arrays(path: Path, keys: Iterable[str]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive.files]
        if missing:
            raise KeyError(f"{path} missing arrays: {missing}")
        return {key: np.asarray(archive[key]) for key in keys}


def _aligned_feature(path: Path, key: str, target_ids: np.ndarray) -> np.ndarray:
    source = _read_npz_arrays(path, ("sample_ids", key))
    return align(source["sample_ids"].astype(str), source[key], target_ids.astype(str))


def _normalise_probability(values: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=np.float64), 1e-8, None)
    probability /= probability.sum(axis=-1, keepdims=True)
    return probability.astype(np.float32)


def _softmax_logits(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    return _normalise_probability(np.exp(logits))


def _safe_decoded_probability(prediction: np.ndarray, confidence: float = 0.94) -> np.ndarray:
    output = np.full(
        (len(prediction), 40), (1.0 - confidence) / 39.0, dtype=np.float32
    )
    output[np.arange(len(prediction)), prediction.astype(np.int64)] = confidence
    return output


def _expert_probability(
    sample_ids: np.ndarray,
    base_prediction: np.ndarray,
    active_modalities: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Build only source OOF expert distributions; no H2/H3 split loader is used."""

    names = ["source_safe_decoded", "p87s_emission"]
    groups = ["anchor", "anchor"]
    values = [
        _safe_decoded_probability(base_prediction),
        _normalise_probability(
            _aligned_feature(STRUCTURED_PATH, "emission_probability", sample_ids)
        ),
    ]
    if "ir" in active_modalities:
        vmae = _aligned_feature(VMAE_OOF_PATH, "early_late_logits", sample_ids)
        iv2_early = _aligned_feature(IV2_OOF_PATH, "early_late_logits", sample_ids)
        iv2_joint = _aligned_feature(
            IV2_OOF_PATH, "early_late_plus_k400_logits", sample_ids
        )
        vmae_probability = _softmax_logits(vmae)
        iv2_early_probability = _softmax_logits(iv2_early)
        iv2_joint_probability = _softmax_logits(iv2_joint)
        equal_probability = _softmax_logits(
            0.5 * np.log(vmae_probability.clip(1e-8))
            + 0.5 * np.log(iv2_joint_probability.clip(1e-8))
        )
        names.extend(
            (
                "videomaev2_distilled_base",
                "internvideo2_l_early_late",
                "internvideo2_l_early_late_plus_k400",
                "videomaev2_base_plus_internvideo2_l_equal",
            )
        )
        groups.extend(("ir",) * 4)
        values.extend(
            (
                vmae_probability,
                iv2_early_probability,
                iv2_joint_probability,
                equal_probability,
            )
        )
    if "skeleton" in active_modalities:
        names.extend(("skeleton_invariant", "motionbert_front"))
        groups.extend(("skeleton", "skeleton"))
        values.extend(
            (
                _softmax_logits(
                    _aligned_feature(SKELETON_OOF_PATH, "skeleton_logits", sample_ids)
                ),
                _normalise_probability(
                    _aligned_feature(MOTIONBERT_OOF_PATH, "probabilities", sample_ids)
                ),
            )
        )
    if "imu" in active_modalities:
        names.append("imu_sensorwise_deep")
        groups.append("imu")
        values.append(
            _normalise_probability(
                _aligned_feature(IMU_OOF_PATH, "probabilities", sample_ids)
            )
        )
    probability = np.stack(values, axis=1).astype(np.float32)
    return probability, tuple(names), tuple(groups)


def _source_cohort(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = (
        f"{prefix}_sample_ids",
        f"{prefix}_labels",
        f"{prefix}_safe_prediction",
    )
    source = _read_npz_arrays(ROUTER_PATH, keys)
    sample_ids = source[keys[0]].astype(str)
    labels = source[keys[1]].astype(np.int64)
    prediction = source[keys[2]].astype(np.int64)
    parsed = np.asarray([parse_sample_identity(value)[0] for value in sample_ids], dtype=np.int64)
    if not np.array_equal(labels, parsed):
        raise ValueError(f"{prefix} labels disagree with sample ids")
    return sample_ids, labels, prediction


@dataclass
class FourModalData:
    sample_ids: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    base_prediction: np.ndarray
    expert_probability: np.ndarray
    expert_names: tuple[str, ...]
    expert_groups: tuple[str, ...]
    streams: dict[str, np.ndarray]
    stream_groups: dict[str, str]
    statistics: dict[str, np.ndarray]
    statistic_groups: dict[str, str]
    skeleton_sequence: np.ndarray
    skeleton_mask: np.ndarray
    imu_sequence: np.ndarray
    imu_mask: np.ndarray
    modality_available: dict[str, np.ndarray]
    boundaries: dict[str, np.ndarray]
    active_modalities: tuple[str, ...] = CANONICAL_MODALITIES

    def validate(self) -> None:
        count = len(self.sample_ids)
        one_dimensional = {
            "labels": self.labels,
            "users": self.users,
            "base_prediction": self.base_prediction,
        }
        for name, values in one_dimensional.items():
            if values.shape != (count,):
                raise ValueError(f"{name} has shape {values.shape}, expected {(count,)}")
        for name, values in {**self.streams, **self.statistics}.items():
            if len(values) != count:
                raise ValueError(f"{name} has {len(values)} rows, expected {count}")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
        if set(self.modality_available) != set(self.active_modalities):
            raise ValueError("modality availability keys differ from active modalities")
        for name, values in self.modality_available.items():
            if values.shape != (count,):
                raise ValueError(f"{name} availability has shape {values.shape}")
        if self.expert_probability.shape != (count, len(self.expert_names), 40):
            raise ValueError(
                "expert probability shape differs from rows, names, or 40 classes"
            )
        if len(self.expert_names) != len(self.expert_groups):
            raise ValueError("expert names and groups differ")
        allowed_expert_groups = set(self.active_modalities) | {"anchor"}
        if set(self.expert_groups) - allowed_expert_groups:
            raise ValueError("expert probability has an inactive or invalid group")
        if not np.isfinite(self.expert_probability).all():
            raise ValueError("expert probability contains non-finite values")
        if not np.allclose(self.expert_probability.sum(axis=2), 1.0, atol=1e-4):
            raise ValueError("expert probabilities are not normalized")
        if self.skeleton_sequence.shape[:1] != (count,):
            raise ValueError("skeleton row count mismatch")
        if self.imu_sequence.shape[:1] != (count,):
            raise ValueError("IMU row count mismatch")
        all_rows: list[int] = []
        for rows in self.boundaries.values():
            all_rows.extend(np.asarray(rows, dtype=np.int64).tolist())
        if sorted(all_rows) != list(range(count)):
            raise ValueError("cohort boundaries are not an exact partition")
        if len(np.unique(self.sample_ids)) != count:
            raise ValueError("duplicate sample ids")
        validate_modalities(self.active_modalities)

    def summary(self) -> dict[str, Any]:
        return {
            "samples": int(len(self.sample_ids)),
            "cohorts": {name: int(len(rows)) for name, rows in self.boundaries.items()},
            "users": sorted(np.unique(self.users).tolist()),
            "active_modalities": list(self.active_modalities),
            "availability": {
                name: float(values.mean()) for name, values in self.modality_available.items()
            },
            "experts": [
                {"name": name, "group": group}
                for name, group in zip(self.expert_names, self.expert_groups)
            ],
            "streams": {
                name: {"group": self.stream_groups[name], "shape": list(values.shape[1:])}
                for name, values in self.streams.items()
            },
            "statistics": {
                name: {"group": self.statistic_groups[name], "shape": list(values.shape[1:])}
                for name, values in self.statistics.items()
            },
            "skeleton_sequence": list(self.skeleton_sequence.shape[1:]),
            "imu_sequence": list(self.imu_sequence.shape[1:]),
        }


def build_four_modal_source_data(
    modalities: Iterable[str] = CANONICAL_MODALITIES,
    include_h2: bool = False,
) -> FourModalData:
    """Build H1/E0 and, only when explicit, H2; H3 is never available."""

    active = validate_modalities(modalities)
    # Use the frozen Safe anchor, not the learned P90 router prediction.  The
    # historical router cross-fit used other outer cohorts; retaining it as an
    # H1 training target would silently reintroduce H3-derived supervision.
    h1_ids, _, h1_base = _source_cohort("H1_selection")
    if include_h2:
        h2_ids, _, h2_base = _source_cohort("H2_confirmation")
    else:
        h2_ids = np.asarray([], dtype=str)
        h2_base = np.asarray([], dtype=np.int64)

    vmae_index = _read_npz_arrays(VMAE_PATH, ("sample_ids",))
    all_ids = vmae_index["sample_ids"].astype(str)
    e0_ids = np.asarray(
        [value for value in all_ids if parse_sample_identity(value)[1] in E0_USERS],
        dtype=str,
    )
    e0_base_source = _read_npz_arrays(E0_BASE_PATH, ("sample_ids", "sequence_predictions"))
    e0_base = align(
        e0_base_source["sample_ids"].astype(str),
        e0_base_source["sequence_predictions"].astype(np.int64),
        e0_ids,
    )

    sample_ids = np.concatenate((h1_ids, e0_ids, h2_ids)).astype(str)
    labels = np.asarray([parse_sample_identity(value)[0] for value in sample_ids], dtype=np.int64)
    users = np.asarray([parse_sample_identity(value)[1] for value in sample_ids], dtype=str)
    base_prediction = np.concatenate((h1_base, e0_base, h2_base)).astype(np.int64)
    expert_probability, expert_names, expert_groups = _expert_probability(
        sample_ids, base_prediction, active
    )
    offsets = np.cumsum((0, len(h1_ids), len(e0_ids), len(h2_ids)))
    boundaries = {
        "H1_selection": np.arange(offsets[0], offsets[1], dtype=np.int64),
        "E0_source_only": np.arange(offsets[1], offsets[2], dtype=np.int64),
    }
    if include_h2:
        boundaries["H2_confirmation"] = np.arange(
            offsets[2], offsets[3], dtype=np.int64
        )

    streams: dict[str, np.ndarray] = {}
    stream_groups: dict[str, str] = {}
    if "ir" in active:
        streams["ir_vmae"] = _aligned_feature(VMAE_PATH, "features", sample_ids).reshape(
            len(sample_ids), -1, 768
        ).astype(np.float32)
        streams["ir_iv2"] = _aligned_feature(IV2_PATH, "features", sample_ids).reshape(
            len(sample_ids), -1, 768
        ).astype(np.float32)
        stream_groups.update({"ir_vmae": "ir", "ir_iv2": "ir"})
    if "depth" in active:
        streams["depth"] = _aligned_feature(DEPTH_PATH, "features", sample_ids).reshape(
            len(sample_ids), -1, 768
        ).astype(np.float32)
        stream_groups["depth"] = "depth"
    if "skeleton" in active:
        motionbert = _aligned_feature(MOTIONBERT_PATH, "features", sample_ids)
        if motionbert.shape[1] % 768:
            raise ValueError(f"unexpected MotionBERT shape: {motionbert.shape}")
        streams["skeleton_motionbert"] = motionbert.reshape(
            len(sample_ids), -1, 768
        ).astype(np.float32)
        hdgcn = _aligned_feature(HDGCN_PATH, "features", sample_ids).astype(np.float32)
        if hdgcn.ndim != 3 or hdgcn.shape[-1] != 256:
            raise ValueError(f"unexpected HD-GCN shape: {hdgcn.shape}")
        streams["skeleton_hdgcn"] = np.tile(hdgcn, (1, 1, 3)).astype(np.float32)
        stream_groups.update(
            {"skeleton_motionbert": "skeleton", "skeleton_hdgcn": "skeleton"}
        )

    statistics: dict[str, np.ndarray] = {}
    statistic_groups: dict[str, str] = {}
    if "skeleton" in active or "imu" in active:
        cross = _aligned_feature(STATISTICS_PATH, "features", sample_ids).astype(np.float32)
        if cross.shape[1] != 6195:
            raise ValueError(f"unexpected cross-modal statistic shape: {cross.shape}")
        if "skeleton" in active:
            statistics["skeleton_statistics"] = np.concatenate(
                (cross[:, :2629], cross[:, 5729:5740]), axis=1
            ).astype(np.float32)
            statistic_groups["skeleton_statistics"] = "skeleton"
        if "imu" in active:
            statistics["imu_statistics"] = np.concatenate(
                (cross[:, 2629:5729], cross[:, 5740:5795]), axis=1
            ).astype(np.float32)
            statistic_groups["imu_statistics"] = "imu"
        if "skeleton" in active and "imu" in active:
            statistics["cross_relation_statistics"] = cross[:, 5795:6195].astype(np.float32)
            statistic_groups["cross_relation_statistics"] = "cross"

    with (MOTION_CACHE / "rows.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        motion_ids = np.asarray([row["sample_id"] for row in csv.DictReader(handle)], dtype=str)
    skeleton_sequence = np.asarray(
        np.load(MOTION_CACHE / "skeleton_features.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(len(motion_ids), 32, 17, 13)
    skeleton_mask = np.asarray(
        np.load(MOTION_CACHE / "skeleton_joint_mask.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(len(motion_ids), 32, 17)
    imu_sequence = np.asarray(
        np.load(MOTION_CACHE / "imu_sequences.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(len(motion_ids), 32, 5, 4, 16)
    imu_mask = np.asarray(
        np.load(MOTION_CACHE / "imu_sequence_mask.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(len(motion_ids), 32, 5, 4)

    data = FourModalData(
        sample_ids=sample_ids,
        labels=labels,
        users=users,
        base_prediction=base_prediction,
        expert_probability=expert_probability,
        expert_names=expert_names,
        expert_groups=expert_groups,
        streams=streams,
        stream_groups=stream_groups,
        statistics=statistics,
        statistic_groups=statistic_groups,
        skeleton_sequence=align(motion_ids, skeleton_sequence, sample_ids).astype(np.float32),
        skeleton_mask=align(motion_ids, skeleton_mask, sample_ids).astype(np.float32),
        imu_sequence=align(motion_ids, imu_sequence, sample_ids).astype(np.float32),
        imu_mask=align(motion_ids, imu_mask, sample_ids).astype(np.float32),
        modality_available={
            name: np.ones(len(sample_ids), dtype=np.float32) for name in active
        },
        boundaries=boundaries,
        active_modalities=active,
    )
    data.validate()
    return data


class FourModalPreprocessor:
    def __init__(self, statistics_dim: int = 64, seed: int = 17) -> None:
        self.statistics_dim = int(statistics_dim)
        self.seed = int(seed)
        self.token_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.sequence_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.pcas: dict[str, PCA] = {}

    def fit(self, data: FourModalData, indices: np.ndarray) -> "FourModalPreprocessor":
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) < 2:
            raise ValueError("preprocessor requires at least two training rows")
        for name, stream in data.streams.items():
            values = stream[indices].reshape(-1, stream.shape[-1]).astype(np.float64)
            mean = values.mean(axis=0).astype(np.float32)
            std = np.maximum(values.std(axis=0), 1e-4).astype(np.float32)
            self.token_stats[name] = (mean, std)
        for name, values in data.statistics.items():
            scaler = StandardScaler()
            scaled = scaler.fit_transform(values[indices].astype(np.float64))
            dimensions = min(self.statistics_dim, len(indices) - 1, scaled.shape[1])
            pca = PCA(
                n_components=dimensions,
                whiten=True,
                svd_solver="randomized",
                random_state=self.seed,
            )
            pca.fit(scaled)
            self.scalers[name] = scaler
            self.pcas[name] = pca
        if "skeleton" in data.active_modalities:
            values = data.skeleton_sequence[indices]
            valid = data.skeleton_mask[indices].astype(bool)
            mean = np.zeros(13, dtype=np.float32)
            std = np.ones(13, dtype=np.float32)
            for channel in range(13):
                selected = values[..., channel][valid]
                if len(selected):
                    mean[channel] = float(selected.mean())
                    std[channel] = max(float(selected.std()), 1e-4)
            self.sequence_stats["skeleton"] = (mean, std)
        if "imu" in data.active_modalities:
            values = data.imu_sequence[indices]
            valid = data.imu_mask[indices].astype(bool)
            mean = np.zeros(4, dtype=np.float32)
            std = np.ones(4, dtype=np.float32)
            for channel in range(4):
                selected = values[..., channel, :][valid[..., channel]]
                if len(selected):
                    mean[channel] = float(selected.mean())
                    std[channel] = max(float(selected.std()), 1e-4)
            self.sequence_stats["imu"] = (mean, std)
        return self

    def transform(self, data: FourModalData) -> FourModalData:
        streams = {
            name: ((values - self.token_stats[name][0]) / self.token_stats[name][1]).astype(
                np.float32
            )
            for name, values in data.streams.items()
        }
        statistics = {
            name: self.pcas[name]
            .transform(self.scalers[name].transform(values.astype(np.float64)))
            .astype(np.float32)
            for name, values in data.statistics.items()
        }
        skeleton_sequence = data.skeleton_sequence.copy()
        if "skeleton" in data.active_modalities:
            mean, std = self.sequence_stats["skeleton"]
            skeleton_sequence = ((skeleton_sequence - mean) / std).astype(np.float32)
            skeleton_sequence *= data.skeleton_mask[..., None]
        imu_sequence = data.imu_sequence.copy()
        if "imu" in data.active_modalities:
            mean, std = self.sequence_stats["imu"]
            imu_sequence = (
                (imu_sequence - mean[None, None, None, :, None])
                / std[None, None, None, :, None]
            ).astype(np.float32)
            imu_sequence *= data.imu_mask[..., None]
        transformed = replace(
            data,
            streams=streams,
            statistics=statistics,
            skeleton_sequence=skeleton_sequence,
            imu_sequence=imu_sequence,
        )
        transformed.validate()
        return transformed

    def summary(self) -> dict[str, Any]:
        return {
            "statistics_dim": self.statistics_dim,
            "seed": self.seed,
            "statistics": {
                name: {
                    "input_dim": int(pca.n_features_in_),
                    "output_dim": int(pca.n_components_),
                    "explained_variance": float(pca.explained_variance_ratio_.sum()),
                }
                for name, pca in self.pcas.items()
            },
        }


def counterfactual_data(
    data: FourModalData,
    modality: str,
    mode: str,
    seed: int = 0,
) -> FourModalData:
    """Return a zeroed or sample-shuffled modality without changing labels."""

    modality = str(modality).lower()
    if modality not in data.active_modalities:
        raise ValueError(f"modality {modality!r} is not active")
    if mode not in {"zero", "shuffle"}:
        raise ValueError(f"unsupported counterfactual mode: {mode}")
    permutation = np.random.default_rng(seed).permutation(len(data.sample_ids))

    def alter(values: np.ndarray) -> np.ndarray:
        return np.zeros_like(values) if mode == "zero" else values[permutation].copy()

    streams = {
        name: alter(values) if data.stream_groups[name] == modality else values
        for name, values in data.streams.items()
    }
    statistics = {
        name: alter(values)
        if data.statistic_groups[name] in {modality, "cross"}
        else values
        for name, values in data.statistics.items()
    }
    skeleton_sequence = data.skeleton_sequence
    skeleton_mask = data.skeleton_mask
    imu_sequence = data.imu_sequence
    imu_mask = data.imu_mask
    modality_available = dict(data.modality_available)
    expert_probability = data.expert_probability.copy()
    for expert_index, group in enumerate(data.expert_groups):
        if group != modality:
            continue
        if mode == "zero":
            expert_probability[:, expert_index] = 1.0 / 40.0
        else:
            expert_probability[:, expert_index] = expert_probability[
                permutation, expert_index
            ]
    if modality == "skeleton":
        skeleton_sequence = alter(skeleton_sequence)
        skeleton_mask = alter(skeleton_mask)
    if modality == "imu":
        imu_sequence = alter(imu_sequence)
        imu_mask = alter(imu_mask)
    if mode == "zero":
        modality_available[modality] = np.zeros_like(modality_available[modality])
    else:
        modality_available[modality] = modality_available[modality][permutation].copy()
    result = replace(
        data,
        streams=streams,
        statistics=statistics,
        skeleton_sequence=skeleton_sequence,
        skeleton_mask=skeleton_mask,
        imu_sequence=imu_sequence,
        imu_mask=imu_mask,
        modality_available=modality_available,
        expert_probability=expert_probability,
    )
    result.validate()
    return result
