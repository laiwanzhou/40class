"""Leakage-safe 18-source-subject data contract for A18 Full Teacher.

The representation contract is identical to P100-A: only label-free Visual and
Skeleton arrays are loaded, while labels and subject IDs are derived from the
canonical sample ID.  All 18 source subjects are used in one full refit; this
module deliberately exposes no validation/held split.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from p100a_global_teacher_data import (
    CROSS_STATISTICS_PATH,
    HDGCN_PATH,
    IV2_PATH,
    MOTIONBERT_PATH,
    MOTION_CACHE,
    VMAE_PATH,
    P100AData,
    _align,
    _load_npz_fields,
    parse_sample_id,
)


A18_SOURCE_USERS = (
    "user1", "user2", "user3", "user4", "user5", "user6", "user7", "user8",
    "user9", "user16", "user17", "user18", "user19", "user20", "user21",
    "user22", "user23", "user24",
)
A18_SOURCE_USER_SET = frozenset(A18_SOURCE_USERS)
A18_ROWS = 2914


class A18Data(P100AData):
    """P100 tensors under the expanded 18-subject OOF contract."""

    def validate(self) -> None:
        rows = len(self.sample_ids)
        if rows != A18_ROWS:
            raise ValueError(f"A18 source pool has {rows} rows, expected {A18_ROWS}")
        if len(np.unique(self.sample_ids)) != rows:
            raise ValueError("duplicate A18 sample ids")
        if set(np.unique(self.users)) != A18_SOURCE_USER_SET:
            raise ValueError("A18 subjects differ from the frozen 18-source allow-list")
        if not np.all(self.fold_ids == -1):
            raise ValueError("A18 full-refit data must not expose an outer fold")
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
            raise ValueError("A18 labels were not derived consistently from sample ids")

    def full_indices(self) -> np.ndarray:
        return np.arange(len(self.sample_ids), dtype=np.int64)

    def summary(self) -> dict[str, object]:
        return {
            "rows": int(len(self.sample_ids)),
            "subjects": len(A18_SOURCE_USERS),
            "users": {
                user: int((self.users == user).sum())
                for user in A18_SOURCE_USERS
            },
            "split": "none_full_refit",
            "train_subjects": 18,
            "train_rows": int(len(self.sample_ids)),
            "validation_subjects": 0,
            "validation_rows": 0,
            "availability": {
                "visual": 1.0,
                "skeleton": float(self.skeleton_available.mean()),
                "imu": float(self.imu_available.mean()),
            },
            "labels_derived_from_sample_id": True,
            "historical_40class_expert_probability_loaded": False,
            "held_rows_loaded_or_used_for_selection": 0,
            "h3_rows_selected": 0,
            "h3_confirmation_run": False,
            "legacy_h3_group_role": "reclassified as source by explicit A18 full-refit authorization",
        }


@lru_cache(maxsize=1)
def load_a18_data() -> A18Data:
    visual_source = _load_npz_fields(
        VMAE_PATH, ("sample_ids", "features", "action_logits")
    )
    all_ids = visual_source["sample_ids"].astype(str)
    parsed = [parse_sample_id(sample_id) for sample_id in all_ids]
    keep = np.asarray([user in A18_SOURCE_USER_SET for _, user in parsed], dtype=bool)
    sample_ids = all_ids[keep]
    labels = np.asarray(
        [class_id for class_id, user in parsed if user in A18_SOURCE_USER_SET],
        dtype=np.int64,
    )
    users = np.asarray(
        [user for _, user in parsed if user in A18_SOURCE_USER_SET], dtype=str
    )
    fold_ids = np.full(len(users), -1, dtype=np.int64)

    iv2 = _load_npz_fields(IV2_PATH, ("sample_ids", "features", "action_logits"))
    motionbert = _load_npz_fields(MOTIONBERT_PATH, ("sample_ids", "features"))
    hdgcn = _load_npz_fields(HDGCN_PATH, ("sample_ids", "tokens"))
    statistics = _load_npz_fields(CROSS_STATISTICS_PATH, ("sample_ids", "features"))
    with MOTION_CACHE.joinpath("rows.csv").open(
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
    skeleton_mask = _align(
        motion_ids, skeleton_mask_all, sample_ids
    ).astype(np.float32)
    imu_sequence = _align(motion_ids, imu_sequence_all, sample_ids).astype(np.float32)
    imu_mask = _align(motion_ids, imu_mask_all, sample_ids).astype(np.float32)

    data = A18Data(
        sample_ids=sample_ids,
        users=users,
        labels=labels,
        fold_ids=fold_ids,
        visual_vmae=np.asarray(visual_source["features"][keep])
        .reshape(-1, 6, 768)
        .astype(np.float32),
        visual_iv2=_align(
            iv2["sample_ids"].astype(str), iv2["features"], sample_ids
        ).reshape(-1, 6, 768).astype(np.float32),
        visual_vmae_action=np.asarray(visual_source["action_logits"][keep])
        .reshape(-1, 6, 710)
        .astype(np.float32),
        visual_iv2_action=_align(
            iv2["sample_ids"].astype(str), iv2["action_logits"], sample_ids
        ).reshape(-1, 6, 400).astype(np.float32),
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


def write_a18_contract(path: Path, data: A18Data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
