"""Leakage-safe 12-subject data contract for P100-A.

Only label-free backbone/token arrays are read from the historical caches.  The
40-class target and subject are parsed from the sample id after the development
allow-list has been applied.  No historical 40-class expert probability enters
this data object.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

VMAE_PATH = PROJECT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz"
IV2_PATH = PROJECT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz"
MOTIONBERT_PATH = PROJECT / "runs/p90_motionbert_teacher_v1/features_pretrain_front_t81.npz"
HDGCN_PATH = PROJECT / "runs/p91_hdgcn_ntu60_xsub_fold0_v1/complete_features.npz"
CROSS_STATISTICS_PATH = HERE / "runs/p89_crossmodal_statistics_v1/crossmodal_statistics.npz"
MOTION_CACHE = HERE / "runs/p86_motion_window_cache_t16_v1"

DEV_USERS = (
    "user1",
    "user2",
    "user21",
    "user17",
    "user23",
    "user6",
    "user8",
    "user16",
    "user18",
    "user19",
    "user5",
    "user7",
)
DEV_USER_SET = frozenset(DEV_USERS)
H3_USERS = frozenset(("user3", "user4", "user9", "user20", "user22", "user24"))
FOLD_USERS = (
    ("user1", "user18", "user21"),
    ("user16", "user17", "user23"),
    ("user19", "user2", "user8"),
    ("user5", "user6", "user7"),
)
FOLD_ROW_COUNTS = (460, 483, 513, 485)
CANONICAL_VARIANTS = {
    "V": ("visual",),
    "VS": ("visual", "skeleton"),
    "VI": ("visual", "imu"),
    "VSI": ("visual", "skeleton", "imu"),
}
SAMPLE_PATTERN = re.compile(r"^train__c(?P<class_id>\d{2})__(?P<user>user\d+)__")


def parse_sample_id(sample_id: str) -> tuple[int, str]:
    match = SAMPLE_PATTERN.match(str(sample_id))
    if match is None:
        raise ValueError(f"unexpected sample id: {sample_id}")
    return int(match.group("class_id")), match.group("user")


def _load_npz_fields(path: Path, fields: Iterable[str]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(fields) - set(archive.files))
        if missing:
            raise KeyError(f"{path} missing {missing}")
        return {field: np.asarray(archive[field]) for field in fields}


def _align(
    source_ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray
) -> np.ndarray:
    lookup = {str(sample_id): index for index, sample_id in enumerate(source_ids)}
    missing = [str(sample_id) for sample_id in target_ids if str(sample_id) not in lookup]
    if missing:
        raise KeyError(f"missing {len(missing)} sample ids; first={missing[:3]}")
    rows = np.asarray([lookup[str(sample_id)] for sample_id in target_ids], dtype=np.int64)
    return np.asarray(values[rows])


@dataclass(frozen=True)
class P100AData:
    sample_ids: np.ndarray
    users: np.ndarray
    labels: np.ndarray
    fold_ids: np.ndarray
    visual_vmae: np.ndarray
    visual_iv2: np.ndarray
    visual_vmae_action: np.ndarray
    visual_iv2_action: np.ndarray
    skeleton_motionbert: np.ndarray
    skeleton_hdgcn: np.ndarray
    skeleton_sequence: np.ndarray
    skeleton_mask: np.ndarray
    skeleton_statistics: np.ndarray
    imu_sequence: np.ndarray
    imu_mask: np.ndarray
    imu_statistics: np.ndarray
    cross_statistics: np.ndarray
    skeleton_available: np.ndarray
    imu_available: np.ndarray

    def validate(self) -> None:
        rows = len(self.sample_ids)
        if rows != 1941:
            raise ValueError(f"development pool has {rows} rows, expected 1941")
        if len(np.unique(self.sample_ids)) != rows:
            raise ValueError("duplicate development sample ids")
        if set(np.unique(self.users)) != DEV_USER_SET:
            raise ValueError("development subjects differ from frozen allow-list")
        if set(np.unique(self.users)) & H3_USERS:
            raise ValueError("H3 subject reached P100-A development data")
        if sorted(np.unique(self.fold_ids).tolist()) != [0, 1, 2, 3]:
            raise ValueError("fold ids are not 0..3")
        counts = tuple(int((self.fold_ids == fold).sum()) for fold in range(4))
        if counts != FOLD_ROW_COUNTS:
            raise ValueError(f"fold row counts changed: {counts}")
        for fold, held_users in enumerate(FOLD_USERS):
            actual = set(self.users[self.fold_ids == fold].tolist())
            if actual != set(held_users):
                raise ValueError(f"fold {fold} users changed: {sorted(actual)}")
        expected_shapes = {
            "visual_vmae": (rows, 6, 768),
            "visual_iv2": (rows, 6, 768),
            "visual_vmae_action": (rows, 6, 710),
            "visual_iv2_action": (rows, 6, 400),
            "skeleton_motionbert": (rows, 12, 768),
            "skeleton_hdgcn": (rows, 6, 16, 256),
            "skeleton_sequence": (rows, 32, 17, 13),
            "skeleton_mask": (rows, 32, 17),
            "skeleton_statistics": (rows, 2640),
            "imu_sequence": (rows, 32, 5, 4, 16),
            "imu_mask": (rows, 32, 5, 4),
            "imu_statistics": (rows, 3155),
            "cross_statistics": (rows, 400),
        }
        for name, shape in expected_shapes.items():
            values = getattr(self, name)
            if values.shape != shape:
                raise ValueError(f"{name} has {values.shape}, expected {shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
        parsed = np.asarray([parse_sample_id(value)[0] for value in self.sample_ids])
        if not np.array_equal(parsed.astype(np.int64), self.labels):
            raise ValueError("labels were not derived consistently from sample ids")

    def indices_for_fold(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        if fold not in range(4):
            raise ValueError(f"invalid fold {fold}")
        held = np.flatnonzero(self.fold_ids == fold).astype(np.int64)
        train = np.flatnonzero(self.fold_ids != fold).astype(np.int64)
        if set(self.users[held]) & set(self.users[train]):
            raise RuntimeError("subject leakage across outer fold")
        return train, held

    def summary(self) -> dict[str, object]:
        return {
            "rows": int(len(self.sample_ids)),
            "users": {
                user: int((self.users == user).sum()) for user in sorted(DEV_USER_SET)
            },
            "folds": [
                {
                    "fold": fold,
                    "held_users": list(FOLD_USERS[fold]),
                    "rows": int((self.fold_ids == fold).sum()),
                }
                for fold in range(4)
            ],
            "availability": {
                "visual": 1.0,
                "skeleton": float(self.skeleton_available.mean()),
                "imu": float(self.imu_available.mean()),
            },
            "h3_users_loaded": sorted(set(self.users.tolist()) & H3_USERS),
            "historical_40class_expert_probability_loaded": False,
        }


@lru_cache(maxsize=1)
def load_p100a_data() -> P100AData:
    # Read only ids and label-free representation arrays.  Historical labels and
    # 40-class OOF fields in these archives are intentionally not requested.
    visual_source = _load_npz_fields(
        VMAE_PATH, ("sample_ids", "features", "action_logits")
    )
    all_ids = visual_source["sample_ids"].astype(str)
    parsed = [parse_sample_id(sample_id) for sample_id in all_ids]
    keep = np.asarray([user in DEV_USER_SET for _, user in parsed], dtype=bool)
    sample_ids = all_ids[keep]
    labels = np.asarray([class_id for class_id, user in parsed if user in DEV_USER_SET])
    users = np.asarray([user for _, user in parsed if user in DEV_USER_SET])
    fold_lookup = {
        user: fold for fold, held_users in enumerate(FOLD_USERS) for user in held_users
    }
    fold_ids = np.asarray([fold_lookup[user] for user in users], dtype=np.int64)

    iv2 = _load_npz_fields(IV2_PATH, ("sample_ids", "features", "action_logits"))
    motionbert = _load_npz_fields(MOTIONBERT_PATH, ("sample_ids", "features"))
    hdgcn = _load_npz_fields(HDGCN_PATH, ("sample_ids", "tokens"))
    statistics = _load_npz_fields(
        CROSS_STATISTICS_PATH, ("sample_ids", "features")
    )

    with (MOTION_CACHE / "rows.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        motion_ids = np.asarray(
            [row["sample_id"] for row in csv.DictReader(handle)], dtype=str
        )
    skeleton_sequence_all = np.load(
        MOTION_CACHE / "skeleton_features.npy", mmap_mode="r"
    ).reshape(len(motion_ids), 32, 17, 13)
    skeleton_mask_all = np.load(
        MOTION_CACHE / "skeleton_joint_mask.npy", mmap_mode="r"
    ).reshape(len(motion_ids), 32, 17)
    imu_sequence_all = np.load(
        MOTION_CACHE / "imu_sequences.npy", mmap_mode="r"
    ).reshape(len(motion_ids), 32, 5, 4, 16)
    imu_mask_all = np.load(
        MOTION_CACHE / "imu_sequence_mask.npy", mmap_mode="r"
    ).reshape(len(motion_ids), 32, 5, 4)

    cross = _align(
        statistics["sample_ids"].astype(str), statistics["features"], sample_ids
    ).astype(np.float32)
    skeleton_sequence = _align(
        motion_ids, skeleton_sequence_all, sample_ids
    ).astype(np.float32)
    skeleton_mask = _align(motion_ids, skeleton_mask_all, sample_ids).astype(np.float32)
    imu_sequence = _align(motion_ids, imu_sequence_all, sample_ids).astype(np.float32)
    imu_mask = _align(motion_ids, imu_mask_all, sample_ids).astype(np.float32)
    data = P100AData(
        sample_ids=sample_ids,
        users=users,
        labels=labels.astype(np.int64),
        fold_ids=fold_ids,
        visual_vmae=np.asarray(visual_source["features"][keep]).reshape(-1, 6, 768).astype(np.float32),
        visual_iv2=_align(iv2["sample_ids"].astype(str), iv2["features"], sample_ids).reshape(-1, 6, 768).astype(np.float32),
        visual_vmae_action=np.asarray(visual_source["action_logits"][keep]).reshape(-1, 6, 710).astype(np.float32),
        visual_iv2_action=_align(iv2["sample_ids"].astype(str), iv2["action_logits"], sample_ids).reshape(-1, 6, 400).astype(np.float32),
        skeleton_motionbert=_align(
            motionbert["sample_ids"].astype(str), motionbert["features"], sample_ids
        ).reshape(-1, 12, 768).astype(np.float32),
        skeleton_hdgcn=_align(
            hdgcn["sample_ids"].astype(str), hdgcn["tokens"], sample_ids
        ).astype(np.float32),
        skeleton_sequence=skeleton_sequence,
        skeleton_mask=skeleton_mask,
        skeleton_statistics=np.concatenate(
            (cross[:, :2629], cross[:, 5729:5740]), axis=1
        ).astype(np.float32),
        imu_sequence=imu_sequence,
        imu_mask=imu_mask,
        imu_statistics=np.concatenate(
            (cross[:, 2629:5729], cross[:, 5740:5795]), axis=1
        ).astype(np.float32),
        cross_statistics=cross[:, 5795:6195].astype(np.float32),
        skeleton_available=(skeleton_mask.sum(axis=(1, 2)) > 0).astype(np.float32),
        imu_available=(imu_mask.sum(axis=(1, 2, 3)) > 0).astype(np.float32),
    )
    data.validate()
    return data


@dataclass
class FoldNormalizer:
    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]

    @classmethod
    def fit(cls, data: P100AData, train_indices: np.ndarray) -> "FoldNormalizer":
        means: dict[str, np.ndarray] = {}
        stds: dict[str, np.ndarray] = {}
        token_names = (
            "visual_vmae",
            "visual_iv2",
            "visual_vmae_action",
            "visual_iv2_action",
            "skeleton_motionbert",
            "skeleton_hdgcn",
            "skeleton_statistics",
            "imu_statistics",
            "cross_statistics",
        )
        for name in token_names:
            values = getattr(data, name)[train_indices].reshape(
                -1, getattr(data, name).shape[-1]
            ).astype(np.float64)
            means[name] = values.mean(axis=0).astype(np.float32)
            stds[name] = np.maximum(values.std(axis=0), 1e-4).astype(np.float32)

        skeleton = data.skeleton_sequence[train_indices]
        skeleton_valid = data.skeleton_mask[train_indices].astype(bool)
        skeleton_mean = np.zeros(13, dtype=np.float32)
        skeleton_std = np.ones(13, dtype=np.float32)
        for channel in range(13):
            values = skeleton[..., channel][skeleton_valid]
            if len(values):
                skeleton_mean[channel] = float(values.mean())
                skeleton_std[channel] = max(float(values.std()), 1e-4)
        means["skeleton_sequence"] = skeleton_mean
        stds["skeleton_sequence"] = skeleton_std

        imu = data.imu_sequence[train_indices]
        imu_valid = data.imu_mask[train_indices].astype(bool)
        imu_mean = np.zeros(4, dtype=np.float32)
        imu_std = np.ones(4, dtype=np.float32)
        for channel in range(4):
            values = imu[..., channel, :][imu_valid[..., channel]]
            if len(values):
                imu_mean[channel] = float(values.mean())
                imu_std[channel] = max(float(values.std()), 1e-4)
        means["imu_sequence"] = imu_mean
        stds["imu_sequence"] = imu_std
        return cls(means=means, stds=stds)

    def normalise(self, name: str, values: np.ndarray) -> np.ndarray:
        if name == "skeleton_sequence":
            return (
                (values - self.means[name][None, None, :])
                / self.stds[name][None, None, :]
            ).astype(np.float32)
        if name == "imu_sequence":
            return (
                (values - self.means[name][None, None, :, None])
                / self.stds[name][None, None, :, None]
            ).astype(np.float32)
        return ((values - self.means[name]) / self.stds[name]).astype(np.float32)

    def summary(self) -> dict[str, object]:
        return {
            name: {
                "dimensions": int(len(mean)),
                "mean_abs": float(np.abs(mean).mean()),
                "min_std": float(self.stds[name].min()),
            }
            for name, mean in self.means.items()
        }


class P100ADataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data: P100AData,
        indices: np.ndarray,
        normalizer: FoldNormalizer,
        modalities: tuple[str, ...],
        sample_weights: np.ndarray | None = None,
        skeleton_source: np.ndarray | None = None,
        imu_source: np.ndarray | None = None,
        zero_modalities: Iterable[str] = (),
        cross_available: bool = True,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalizer = normalizer
        self.modalities = modalities
        self.sample_weights = sample_weights
        self.skeleton_source = (
            np.asarray(skeleton_source, dtype=np.int64)
            if skeleton_source is not None
            else np.arange(len(data.sample_ids), dtype=np.int64)
        )
        self.imu_source = (
            np.asarray(imu_source, dtype=np.int64)
            if imu_source is not None
            else np.arange(len(data.sample_ids), dtype=np.int64)
        )
        self.zero_modalities = frozenset(zero_modalities)
        self.cross_available = bool(cross_available)

    def __len__(self) -> int:
        return len(self.indices)

    def _tensor(self, name: str, row: int) -> torch.Tensor:
        return torch.from_numpy(self.normalizer.normalise(name, getattr(self.data, name)[row]))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        item: dict[str, torch.Tensor] = {
            "row": torch.tensor(row, dtype=torch.long),
            "label": torch.tensor(self.data.labels[row], dtype=torch.long),
            "weight": torch.tensor(
                1.0 if self.sample_weights is None else self.sample_weights[row],
                dtype=torch.float32,
            ),
            "visual_available": torch.tensor(1.0, dtype=torch.float32),
            "visual_vmae": self._tensor("visual_vmae", row),
            "visual_iv2": self._tensor("visual_iv2", row),
            "visual_vmae_action": self._tensor("visual_vmae_action", row),
            "visual_iv2_action": self._tensor("visual_iv2_action", row),
        }
        if "skeleton" in self.modalities:
            source = int(self.skeleton_source[row])
            sequence = self._tensor("skeleton_sequence", source)
            mask = torch.from_numpy(self.data.skeleton_mask[source].astype(np.float32))
            item.update(
                {
                    "skeleton_available": torch.tensor(
                        0.0
                        if "skeleton" in self.zero_modalities
                        else self.data.skeleton_available[source],
                        dtype=torch.float32,
                    ),
                    "skeleton_motionbert": self._tensor("skeleton_motionbert", source),
                    "skeleton_hdgcn": self._tensor("skeleton_hdgcn", source),
                    "skeleton_sequence": sequence * mask[..., None],
                    "skeleton_mask": mask,
                    "skeleton_statistics": self._tensor(
                        "skeleton_statistics", source
                    ),
                }
            )
        if "imu" in self.modalities:
            source = int(self.imu_source[row])
            sequence = self._tensor("imu_sequence", source)
            mask = torch.from_numpy(self.data.imu_mask[source].astype(np.float32))
            item.update(
                {
                    "imu_available": torch.tensor(
                        0.0
                        if "imu" in self.zero_modalities
                        else self.data.imu_available[source],
                        dtype=torch.float32,
                    ),
                    "imu_sequence": sequence * mask[..., None],
                    "imu_mask": mask,
                    "imu_statistics": self._tensor("imu_statistics", source),
                }
            )
        if "skeleton" in self.modalities and "imu" in self.modalities:
            # Cross statistics have no well-defined identity after a one-modality
            # shuffle, so they are available only for the correctly paired direct
            # path.  Counterfactual datasets zero them explicitly in the runner.
            item["cross_statistics"] = self._tensor("cross_statistics", row)
            item["cross_available"] = torch.tensor(
                float(
                    self.cross_available
                    and
                    "skeleton" not in self.zero_modalities
                    and "imu" not in self.zero_modalities
                ),
                dtype=torch.float32,
            )
        return item


def class_user_sample_weights(data: P100AData, train_indices: np.ndarray) -> np.ndarray:
    labels = data.labels[train_indices]
    class_counts = np.bincount(labels, minlength=40).astype(np.float64)
    class_weight = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
    users, user_counts = np.unique(data.users[train_indices], return_counts=True)
    user_weight = {
        user: 1.0 / np.sqrt(float(count)) for user, count in zip(users, user_counts)
    }
    output = np.ones(len(data.sample_ids), dtype=np.float32)
    output[train_indices] = np.asarray(
        [
            class_weight[data.labels[row]] * user_weight[data.users[row]]
            for row in train_indices
        ],
        dtype=np.float32,
    )
    output[train_indices] /= output[train_indices].mean()
    return output


def within_subject_permutation(
    data: P100AData,
    indices: np.ndarray,
    modality: str,
    seed: int,
) -> np.ndarray:
    if modality not in ("skeleton", "imu"):
        raise ValueError(modality)
    output = np.arange(len(data.sample_ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    availability = (
        data.skeleton_available if modality == "skeleton" else data.imu_available
    )
    for user in np.unique(data.users[indices]):
        user_rows = indices[data.users[indices] == user]
        for available in (0.0, 1.0):
            group = user_rows[availability[user_rows] == available]
            if len(group) > 1:
                permuted = rng.permutation(group)
                if np.array_equal(permuted, group):
                    permuted = np.roll(permuted, 1)
                output[group] = permuted
    return output


def write_contract(path: Path, data: P100AData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
