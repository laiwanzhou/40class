"""Subject-disjoint feature-capability probes for the P91 Top-5 hard pool.

The script is deliberately a representation audit, not a candidate router.
For each modality it applies one shared linear-probe recipe to eight frozen
confusion pairs.  Feature/view/time-scale and Ridge alpha are selected only by
leave-one-user-out source evaluation on H1 plus the established embargo users.
The globally selected recipe for that modality is then fitted on all source
users and evaluated once on H2.  No pair-specific recipe is selected and H3 is
never read.

Examples:
    python audit_p91_confusion_feature_capability.py --modality visual
    python audit_p91_confusion_feature_capability.py --modality skeleton
    python audit_p91_confusion_feature_capability.py --modality imu
    python audit_p91_confusion_feature_capability.py --modality session
    python audit_p91_confusion_feature_capability.py --assemble
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audit_p91_confidence095_hard_pool import load_names, write_csv
from audit_p91_p87_session_candidate_probe import construct_p87_session_probability
from p90_teacher_common import REPO_ROOT, load_protocol
from p91_unrestricted_fusion_teacher import build_cohorts


HERE = Path(__file__).resolve().parent
OUTPUT = REPO_ROOT / "runs/p91_confusion_feature_capability_v1"
P96_CACHE = REPO_ROOT / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1"
MOTION_CACHE = HERE / "runs/p86_motion_window_cache_t16_v1"
HARD_SAMPLES = (
    REPO_ROOT
    / "runs/p91_p87_session_candidate_probe_v1/candidate_session_samples_h2.csv"
)
ALL_CURRENT_PAIRS = (
    REPO_ROOT
    / "runs/p91_p87_session_candidate_probe_v1/candidate_session_pairs_h2.csv"
)

PAIR_SPECS = (
    (21, 22, "read_documents__turn_pages"),
    (24, 26, "mobile_phone__play_games"),
    (6, 37, "drink_water__take_medicine"),
    (7, 8, "eat_food__tableware"),
    (8, 10, "tableware__stir_drinks"),
    (19, 23, "phone_call__headphones"),
    (0, 4, "wash_face__wipe_hands"),
    (12, 13, "sweep_floor__mop_floor"),
)
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)

JOINT_NAMES = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)
RELATION_NAMES = (
    "left_wrist_head_distance",
    "right_wrist_head_distance",
    "hand_to_hand_distance",
    "left_wrist_pelvis_distance",
    "right_wrist_pelvis_distance",
    "left_elbow_angle_cosine",
    "right_elbow_angle_cosine",
    "left_knee_angle_cosine",
    "right_knee_angle_cosine",
    "shoulder_width",
    "hip_width",
    "pelvis_head_height",
    "torso_direction_x",
    "torso_direction_y",
    "torso_direction_z",
    "left_wrist_speed",
    "right_wrist_speed",
    "whole_body_motion_energy",
)
IMU_DEVICES = ("torso", "left_arm", "right_arm", "left_leg", "right_leg")


@dataclass(frozen=True)
class AuditData:
    analysis_indices: np.ndarray
    source_count: int
    sample_ids: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    source_labels: np.ndarray
    source_users: np.ndarray
    h2_sample_ids: np.ndarray
    h2_labels: np.ndarray
    h2_users: np.ndarray
    h2_primary_target: np.ndarray
    h2_p91_prediction: np.ndarray


@dataclass(frozen=True)
class FeatureBlock:
    values: np.ndarray
    names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality", choices=("visual", "skeleton", "imu", "session")
    )
    parser.add_argument("--assemble", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pair_key(left: int, right: int) -> str:
    a, b = sorted((int(left), int(right)))
    return f"pair_{a:02d}_{b:02d}"


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def load_audit_data() -> AuditData:
    protocol = load_protocol()
    cohorts = build_cohorts()
    h1 = cohorts["H1_selection"]
    embargo = cohorts["E0_p87_sequence_source"]
    h2 = cohorts["H2_confirmation"]
    source_ids = np.concatenate((h1.sample_ids, embargo.sample_ids)).astype(str)
    source_labels = np.concatenate((h1.labels, embargo.labels)).astype(np.int64)
    source_users = np.concatenate((h1.users, embargo.users)).astype(str)
    lookup = {
        str(sample_id): row
        for row, sample_id in enumerate(protocol.sample_ids.astype(str))
    }
    source_index = np.asarray([lookup[value] for value in source_ids], dtype=np.int64)
    h2_ids = h2.sample_ids.astype(str)
    h2_index = np.asarray([lookup[value] for value in h2_ids], dtype=np.int64)
    analysis_indices = np.concatenate((source_index, h2_index))

    hard_rows = {row["sample_id"]: row for row in read_csv(HARD_SAMPLES)}
    primary = np.zeros(len(h2_ids), dtype=bool)
    p91_prediction = np.full(len(h2_ids), -1, dtype=np.int64)
    for row, sample_id in enumerate(h2_ids):
        item = hard_rows.get(str(sample_id))
        if item is None:
            continue
        primary[row] = item["primary_error_target"] == "1"
        p91_prediction[row] = int(item["p91_prediction_id"])
    if int(primary.sum()) != 63:
        raise RuntimeError("current P91 primary hard target is not the frozen 63 rows")
    if set(np.unique(source_users)) != {
        "user1",
        "user2",
        "user6",
        "user8",
        "user17",
        "user21",
        "user23",
    }:
        raise RuntimeError("unexpected H1+embargo source users")
    if set(np.unique(h2.users.astype(str))) != {
        "user5",
        "user7",
        "user16",
        "user18",
        "user19",
    }:
        raise RuntimeError("unexpected H2 users")
    return AuditData(
        analysis_indices=analysis_indices,
        source_count=len(source_ids),
        sample_ids=np.concatenate((source_ids, h2_ids)),
        labels=np.concatenate((source_labels, h2.labels.astype(np.int64))),
        users=np.concatenate((source_users, h2.users.astype(str))),
        source_labels=source_labels,
        source_users=source_users,
        h2_sample_ids=h2_ids,
        h2_labels=h2.labels.astype(np.int64),
        h2_users=h2.users.astype(str),
        h2_primary_target=primary,
        h2_p91_prediction=p91_prediction,
    )


def make_probe(alpha: float) -> Pipeline:
    return Pipeline(
        (
            ("scale", StandardScaler()),
            (
                "ridge",
                RidgeClassifier(
                    alpha=float(alpha),
                    class_weight="balanced",
                    solver="lsqr",
                    tol=1e-5,
                    max_iter=5000,
                ),
            ),
        )
    )


def binary_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return {
        "samples": int(len(labels)),
        "correct": int(np.sum(labels == prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }


def h1_louo_pair_metrics(
    values: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    user_accuracies: list[float] = []
    for left, right, title in PAIR_SPECS:
        selected = np.isin(labels, (left, right))
        indices = np.flatnonzero(selected)
        prediction = np.full(len(indices), -1, dtype=np.int64)
        pair_users = users[indices]
        for user in np.unique(pair_users):
            held_local = pair_users == user
            held = indices[held_local]
            train = indices[~held_local]
            if len(np.unique(labels[train])) != 2:
                raise RuntimeError(f"source fold lacks both classes for {title}/{user}")
            model = make_probe(alpha)
            model.fit(values[train], labels[train])
            prediction[held_local] = model.predict(values[held]).astype(np.int64)
        if np.any(prediction < 0):
            raise RuntimeError(f"incomplete H1 LOUO prediction for {title}")
        metrics = binary_metrics(labels[indices], prediction)
        per_user = {
            str(user): float(
                np.mean(prediction[pair_users == user] == labels[indices][pair_users == user])
            )
            for user in np.unique(pair_users)
        }
        user_accuracies.extend(per_user.values())
        pair_rows.append(
            {
                "pair_id": pair_key(left, right),
                "title": title,
                "left": left,
                "right": right,
                "users": int(len(per_user)),
                "per_user_accuracy": per_user,
                **metrics,
            }
        )
    pair_balanced = [row["balanced_accuracy"] for row in pair_rows]
    pair_accuracy = [row["accuracy"] for row in pair_rows]
    aggregate = {
        "macro_pair_balanced_accuracy": float(np.mean(pair_balanced)),
        "macro_pair_accuracy": float(np.mean(pair_accuracy)),
        "macro_pair_user_accuracy": float(np.mean(user_accuracies)),
        "worst_pair_balanced_accuracy": float(np.min(pair_balanced)),
        "pairs_above_random": int(np.sum(np.asarray(pair_balanced) > 0.5)),
        "pairs_at_or_above_60pct": int(np.sum(np.asarray(pair_balanced) >= 0.60)),
    }
    return aggregate, pair_rows


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, int, int, float]:
    metrics = row["h1"]
    return (
        float(metrics["macro_pair_balanced_accuracy"]),
        float(metrics["macro_pair_user_accuracy"]),
        float(metrics["worst_pair_balanced_accuracy"]),
        int(metrics["pairs_at_or_above_60pct"]),
        -int(row["dimensions"]),
        -float(row["alpha"]),
    )


def temporal_summary(
    signals: np.ndarray,
    mask: np.ndarray,
    signal_names: list[str],
    prefix: str,
    spectral: bool = False,
) -> FeatureBlock:
    values = np.asarray(signals, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
    count = np.maximum(valid.sum(axis=1), 1)
    safe = np.where(valid, values, 0.0)
    mean = safe.sum(axis=1) / count
    centered = np.where(valid, values - mean[:, None, :], 0.0)
    std = np.sqrt(np.sum(centered**2, axis=1) / count)
    rms = np.sqrt(np.sum(safe**2, axis=1) / count)
    minimum = np.min(np.where(valid, values, np.inf), axis=1)
    maximum = np.max(np.where(valid, values, -np.inf), axis=1)
    minimum[~np.isfinite(minimum)] = 0.0
    maximum[~np.isfinite(maximum)] = 0.0
    pair_valid = valid[:, 1:] & valid[:, :-1]
    difference = np.where(pair_valid, values[:, 1:] - values[:, :-1], 0.0)
    pair_count = np.maximum(pair_valid.sum(axis=1), 1)
    mean_abs_difference = np.sum(np.abs(difference), axis=1) / pair_count
    first = np.argmax(valid, axis=1)
    last = values.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    batch = np.arange(len(values))[:, None]
    signal = np.arange(values.shape[2])[None, :]
    endpoint = safe[batch, last, signal] - safe[batch, first, signal]
    endpoint[~valid.any(axis=1)] = 0.0
    blocks = [mean, std, rms, maximum - minimum, mean_abs_difference, endpoint]
    block_names = ["mean", "std", "rms", "range", "mean_abs_diff", "endpoint_delta"]

    if spectral:
        centered_filled = np.where(valid, values - mean[:, None, :], 0.0)
        power = np.abs(np.fft.rfft(centered_filled, axis=1)) ** 2
        non_dc = power[:, 1:, :]
        bins = non_dc.shape[1]
        total_power = np.maximum(non_dc.sum(axis=1), 1e-12)
        first_cut = max(1, bins // 3)
        second_cut = max(first_cut + 1, 2 * bins // 3)
        low = non_dc[:, :first_cut].sum(axis=1) / total_power
        middle = non_dc[:, first_cut:second_cut].sum(axis=1) / total_power
        high = non_dc[:, second_cut:].sum(axis=1) / total_power
        dominant = (np.argmax(non_dc, axis=1) + 1) / max(values.shape[1], 1)
        probability = non_dc / total_power[:, None, :]
        entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)
        entropy /= np.log(max(bins, 2))
        magnitude = np.abs(values)
        max_magnitude = np.max(np.where(valid, magnitude, 0.0), axis=1)
        pause = np.sum(
            valid & (magnitude <= 0.20 * max_magnitude[:, None, :]), axis=1
        ) / count
        if values.shape[1] >= 3:
            peaks = (
                valid[:, 1:-1]
                & (values[:, 1:-1] > values[:, :-2])
                & (values[:, 1:-1] >= values[:, 2:])
            )
            peak_rate = peaks.sum(axis=1) / np.maximum(valid[:, 1:-1].sum(axis=1), 1)
        else:
            peak_rate = np.zeros_like(mean)
        blocks.extend((low, middle, high, dominant, entropy, pause, peak_rate))
        block_names.extend(
            (
                "low_frequency_ratio",
                "middle_frequency_ratio",
                "high_frequency_ratio",
                "dominant_frequency",
                "spectral_entropy",
                "pause_fraction",
                "peak_rate",
            )
        )

    result = np.nan_to_num(
        np.concatenate(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    names = [
        f"{prefix}:{statistic}:{signal_name}"
        for statistic in block_names
        for signal_name in signal_names
    ]
    return FeatureBlock(result, names)


def concatenate_blocks(blocks: Iterable[FeatureBlock]) -> FeatureBlock:
    selected = list(blocks)
    return FeatureBlock(
        np.concatenate([block.values for block in selected], axis=1),
        [name for block in selected for name in block.names],
    )


def l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def row_standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    centered = array - array.mean(axis=-1, keepdims=True)
    return centered / np.maximum(centered.std(axis=-1, keepdims=True), 1e-6)


def visual_feature_builders(data: AuditData) -> dict[str, Callable[[], FeatureBlock]]:
    done = np.asarray(np.load(P96_CACHE / "done.npy", mmap_mode="r"), dtype=bool)
    if not done.all():
        raise RuntimeError("P96 dense24 cache is incomplete")
    dense = np.asarray(
        np.load(P96_CACHE / "features.npy", mmap_mode="r")[data.analysis_indices],
        dtype=np.float32,
    )
    action = np.asarray(
        np.load(P96_CACHE / "ssv2_logits.npy", mmap_mode="r")[data.analysis_indices],
        dtype=np.float32,
    )
    dense = l2_normalize(dense)
    action = row_standardize(action)
    dense_groups = l2_normalize(dense.reshape(len(dense), 8, 3, 1024).mean(axis=2))
    action_groups = row_standardize(
        action.reshape(len(action), 8, 3, 174).mean(axis=2)
    )

    def block(values: np.ndarray, prefix: str) -> FeatureBlock:
        flat = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
        return FeatureBlock(flat, [f"{prefix}:dim_{index}" for index in range(flat.shape[1])])

    return {
        "visual_global_full": lambda: block(dense_groups[:, 0], "global_full"),
        "visual_global_early": lambda: block(dense_groups[:, 1], "global_early"),
        "visual_global_middle": lambda: block(dense_groups[:, 2], "global_middle"),
        "visual_global_late": lambda: block(dense_groups[:, 3], "global_late"),
        "visual_global_temporal": lambda: block(
            dense_groups[:, 1:4], "global_early_middle_late"
        ),
        "visual_workspace_temporal": lambda: block(
            dense[:, (2, 5, 8, 11)], "workspace_full_early_middle_late"
        ),
        "visual_hand_full": lambda: block(dense_groups[:, 4], "hand_full"),
        "visual_hand_early": lambda: block(dense_groups[:, 5], "hand_early"),
        "visual_hand_late": lambda: block(dense_groups[:, 6], "hand_late"),
        "visual_hand_motion_peak": lambda: block(
            dense_groups[:, 7], "hand_motion_peak"
        ),
        "visual_hand_interaction_temporal": lambda: block(
            dense[:, (14, 17, 20, 23)],
            "hand_interaction_full_early_late_motion_peak",
        ),
        "visual_group8": lambda: block(dense_groups, "dense_group8"),
        "visual_group8_ssv2": lambda: block(
            np.concatenate(
                (dense_groups.reshape(len(dense), -1), action_groups.reshape(len(dense), -1)),
                axis=1,
            ),
            "dense_group8_ssv2",
        ),
    }


def skeleton_signal_block(
    skeleton: np.ndarray,
    joint_mask: np.ndarray,
    relations: np.ndarray,
    relation_mask: np.ndarray,
    joints: list[int],
    relation_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = skeleton[..., joints, :]
    selected_mask = joint_mask[..., joints]
    position = selected[..., :3].reshape(*selected.shape[:3], -1)
    position_mask = np.repeat(selected_mask[..., None], 3, axis=-1).reshape(
        *selected_mask.shape[:3], -1
    )
    bone = np.linalg.norm(selected[..., 3:6], axis=-1)
    speed = np.linalg.norm(selected[..., 6:9], axis=-1)
    acceleration = np.linalg.norm(selected[..., 9:12], axis=-1)
    signal_parts = (position, bone, speed, acceleration, relations[..., relation_ids])
    mask_parts = (
        position_mask,
        selected_mask,
        selected_mask,
        selected_mask,
        relation_mask[..., relation_ids],
    )
    names = [
        f"{JOINT_NAMES[joint]}_relative_{axis}"
        for joint in joints
        for axis in ("x", "y", "z")
    ]
    names.extend(f"{JOINT_NAMES[joint]}_bone_length" for joint in joints)
    names.extend(f"{JOINT_NAMES[joint]}_speed" for joint in joints)
    names.extend(f"{JOINT_NAMES[joint]}_acceleration" for joint in joints)
    names.extend(RELATION_NAMES[index] for index in relation_ids)
    return (
        np.concatenate(signal_parts, axis=-1),
        np.concatenate(mask_parts, axis=-1),
        names,
    )


def skeleton_feature_builders(data: AuditData) -> dict[str, Callable[[], FeatureBlock]]:
    rows = read_csv(MOTION_CACHE / "rows.csv")
    cache_lookup = {row["sample_id"]: index for index, row in enumerate(rows)}
    if set(cache_lookup) != set(load_protocol().sample_ids.astype(str).tolist()):
        raise RuntimeError("motion cache sample ids differ from protocol")
    index = np.asarray([cache_lookup[value] for value in data.sample_ids], dtype=np.int64)
    skeleton = np.asarray(
        np.load(MOTION_CACHE / "skeleton_features.npy", mmap_mode="r")[index],
        dtype=np.float32,
    )
    joint_mask = np.asarray(
        np.load(MOTION_CACHE / "skeleton_joint_mask.npy", mmap_mode="r")[index],
        dtype=bool,
    )
    relations = np.asarray(
        np.load(MOTION_CACHE / "skeleton_relations.npy", mmap_mode="r")[index],
        dtype=np.float32,
    )
    relation_mask = np.asarray(
        np.load(MOTION_CACHE / "skeleton_relation_mask.npy", mmap_mode="r")[index],
        dtype=bool,
    )

    def recipe(
        joints: list[int],
        relation_ids: list[int],
        windows: str,
        prefix: str,
    ) -> FeatureBlock:
        signals, mask, names = skeleton_signal_block(
            skeleton, joint_mask, relations, relation_mask, joints, relation_ids
        )
        if windows == "full":
            return temporal_summary(
                signals.reshape(len(signals), -1, signals.shape[-1]),
                mask.reshape(len(mask), -1, mask.shape[-1]),
                names,
                prefix="full",
            )
        if windows == "early":
            return temporal_summary(signals[:, 0], mask[:, 0], names, prefix="early")
        if windows == "late":
            return temporal_summary(signals[:, 1], mask[:, 1], names, prefix="late")
        if windows == "early_late":
            return concatenate_blocks(
                (
                    temporal_summary(signals[:, 0], mask[:, 0], names, prefix="early"),
                    temporal_summary(signals[:, 1], mask[:, 1], names, prefix="late"),
                )
            )
        raise ValueError(windows)

    all_joints = list(range(17))
    hand_head = list(range(9, 17))
    arms = list(range(11, 17))
    all_relations = list(range(18))
    hand_relations = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17]
    arm_relations = [2, 5, 6, 15, 16, 17]
    return {
        "skeleton_all_full": lambda: recipe(
            all_joints, all_relations, "full", "all_full"
        ),
        "skeleton_all_early_late": lambda: recipe(
            all_joints, all_relations, "early_late", "all_early_late"
        ),
        "skeleton_hand_head_full": lambda: recipe(
            hand_head, hand_relations, "full", "hand_head_full"
        ),
        "skeleton_hand_head_early": lambda: recipe(
            hand_head, hand_relations, "early", "hand_head_early"
        ),
        "skeleton_hand_head_late": lambda: recipe(
            hand_head, hand_relations, "late", "hand_head_late"
        ),
        "skeleton_hand_head_early_late": lambda: recipe(
            hand_head, hand_relations, "early_late", "hand_head_early_late"
        ),
        "skeleton_arms_full": lambda: recipe(
            arms, arm_relations, "full", "arms_full"
        ),
        "skeleton_arms_early_late": lambda: recipe(
            arms, arm_relations, "early_late", "arms_early_late"
        ),
    }


def imu_signal_block(
    bin_means: np.ndarray, bin_mask: np.ndarray, devices: list[int]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    signals = []
    masks = []
    names = []
    for device in devices:
        values = bin_means[..., device, :]
        signals.extend(
            (
                np.linalg.norm(values[..., 0:3], axis=-1),
                np.linalg.norm(values[..., 3:6], axis=-1),
                np.linalg.norm(values[..., 6:9], axis=-1),
                np.linalg.norm(values[..., 9:12], axis=-1),
            )
        )
        masks.extend((bin_mask[..., device],) * 4)
        names.extend(
            (
                f"{IMU_DEVICES[device]}_raw_acceleration_magnitude",
                f"{IMU_DEVICES[device]}_raw_angular_velocity_magnitude",
                f"{IMU_DEVICES[device]}_compensated_acceleration_magnitude",
                f"{IMU_DEVICES[device]}_compensated_angular_velocity_magnitude",
            )
        )
    return np.stack(signals, axis=-1), np.stack(masks, axis=-1), names


def imu_feature_builders(data: AuditData) -> dict[str, Callable[[], FeatureBlock]]:
    rows = read_csv(MOTION_CACHE / "rows.csv")
    cache_lookup = {row["sample_id"]: index for index, row in enumerate(rows)}
    if set(cache_lookup) != set(load_protocol().sample_ids.astype(str).tolist()):
        raise RuntimeError("motion cache sample ids differ from protocol")
    index = np.asarray([cache_lookup[value] for value in data.sample_ids], dtype=np.int64)
    statistics = np.asarray(
        np.load(MOTION_CACHE / "imu_bin_statistics.npy", mmap_mode="r")[index],
        dtype=np.float32,
    )
    bin_means = statistics[..., :16]
    bin_mask = np.asarray(
        np.load(MOTION_CACHE / "imu_bin_mask.npy", mmap_mode="r")[index],
        dtype=bool,
    )
    global_statistics = np.asarray(
        np.load(MOTION_CACHE / "imu_global_statistics.npy", mmap_mode="r")[index],
        dtype=np.float32,
    )
    global_mask = np.asarray(
        np.load(MOTION_CACHE / "imu_global_mask.npy", mmap_mode="r")[index],
        dtype=np.float32,
    )

    def temporal(devices: list[int], windows: str) -> FeatureBlock:
        signals, mask, names = imu_signal_block(bin_means, bin_mask, devices)
        if windows == "full":
            return temporal_summary(
                signals.reshape(len(signals), -1, signals.shape[-1]),
                mask.reshape(len(mask), -1, mask.shape[-1]),
                names,
                prefix="full",
                spectral=True,
            )
        if windows == "early":
            return temporal_summary(
                signals[:, 0], mask[:, 0], names, prefix="early", spectral=True
            )
        if windows == "late":
            return temporal_summary(
                signals[:, 1], mask[:, 1], names, prefix="late", spectral=True
            )
        if windows == "early_late":
            return concatenate_blocks(
                (
                    temporal_summary(
                        signals[:, 0], mask[:, 0], names, prefix="early", spectral=True
                    ),
                    temporal_summary(
                        signals[:, 1], mask[:, 1], names, prefix="late", spectral=True
                    ),
                )
            )
        raise ValueError(windows)

    global_stat_names = (
        "mean",
        "std",
        "rms",
        "minimum",
        "maximum",
        "range",
        "mean_abs_difference",
        "mean_squared_difference",
    )
    raw_channels = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")

    def global_block(devices: list[int]) -> FeatureBlock:
        values = global_statistics[:, devices].reshape(len(global_statistics), -1)
        coverage = global_mask[:, devices].reshape(len(global_mask), -1)
        names = [
            f"full_trial:{IMU_DEVICES[device]}:{channel}:{statistic}"
            for device in devices
            for channel in raw_channels
            for statistic in global_stat_names
        ]
        names.extend(
            f"coverage:{IMU_DEVICES[device]}:{kind}"
            for device in devices
            for kind in ("available", "camera_span_fraction")
        )
        return FeatureBlock(
            np.nan_to_num(np.concatenate((values, coverage), axis=1)).astype(np.float32),
            names,
        )

    all_devices = list(range(5))
    arms = [1, 2]
    torso_arms = [0, 1, 2]
    return {
        "imu_all_full_temporal": lambda: temporal(all_devices, "full"),
        "imu_all_early_late": lambda: temporal(all_devices, "early_late"),
        "imu_arms_full_temporal": lambda: temporal(arms, "full"),
        "imu_arms_early": lambda: temporal(arms, "early"),
        "imu_arms_late": lambda: temporal(arms, "late"),
        "imu_arms_early_late": lambda: temporal(arms, "early_late"),
        "imu_torso_arms_early_late": lambda: temporal(torso_arms, "early_late"),
        "imu_all_full_trial_statistics": lambda: global_block(all_devices),
        "imu_arms_full_trial_statistics": lambda: global_block(arms),
    }


def h2_pair_confirmation(
    block: FeatureBlock,
    data: AuditData,
    alpha: float,
    modality: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_values = block.values[: data.source_count]
    h2_values = block.values[data.source_count :]
    pair_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for left, right, title in PAIR_SPECS:
        source_mask = np.isin(data.source_labels, (left, right))
        h2_mask = np.isin(data.h2_labels, (left, right))
        model = make_probe(alpha)
        model.fit(source_values[source_mask], data.source_labels[source_mask])
        h2_indices = np.flatnonzero(h2_mask)
        prediction = model.predict(h2_values[h2_indices]).astype(np.int64)
        pair_labels = data.h2_labels[h2_mask]
        pair_users = data.h2_users[h2_mask]
        metrics = binary_metrics(pair_labels, prediction)
        per_user = {
            str(user): float(
                np.mean(prediction[pair_users == user] == pair_labels[pair_users == user])
            )
            for user in np.unique(pair_users)
        }
        hard = data.h2_primary_target & (
            np.minimum(data.h2_labels, data.h2_p91_prediction) == min(left, right)
        ) & (np.maximum(data.h2_labels, data.h2_p91_prediction) == max(left, right))
        hard_local = hard[h2_indices]
        hard_positions = h2_indices[hard_local]
        hard_prediction = prediction[hard_local]
        pair_rows.append(
            {
                "pair_id": pair_key(left, right),
                "title": title,
                "left": left,
                "right": right,
                "h2_samples": int(h2_mask.sum()),
                "h2_users": int(len(per_user)),
                "h2_per_user_accuracy": per_user,
                "h2_users_above_random": int(
                    np.sum(np.asarray(list(per_user.values())) > 0.5)
                ),
                "hard_target_samples": int(hard.sum()),
                "hard_target_users": int(len(np.unique(data.h2_users[hard]))),
                "hard_rescue": int(
                    np.sum(hard_prediction == data.h2_labels[hard_positions])
                ),
                "hard_accuracy": safe_rate(
                    int(np.sum(hard_prediction == data.h2_labels[hard_positions])),
                    len(hard_positions),
                ),
                **metrics,
            }
        )
        for local, h2_row in enumerate(h2_indices):
            sample_rows.append(
                {
                    "modality": modality,
                    "pair_id": pair_key(left, right),
                    "sample_id": data.h2_sample_ids[h2_row],
                    "user": data.h2_users[h2_row],
                    "true_class_id": int(data.h2_labels[h2_row]),
                    "prediction_id": int(prediction[local]),
                    "correct": int(prediction[local] == data.h2_labels[h2_row]),
                    "primary_hard_target": int(hard[h2_row]),
                }
            )
        coefficients = np.asarray(model.named_steps["ridge"].coef_).reshape(-1)
        top = np.argsort(-np.abs(coefficients))[: min(12, len(coefficients))]
        for rank, feature in enumerate(top, 1):
            importance_rows.append(
                {
                    "modality": modality,
                    "pair_id": pair_key(left, right),
                    "title": title,
                    "rank": rank,
                    "feature": block.names[int(feature)],
                    "coefficient": float(coefficients[int(feature)]),
                    "absolute_coefficient": float(abs(coefficients[int(feature)])),
                }
            )
    return pair_rows, importance_rows, sample_rows


def run_learned_modality(modality: str) -> None:
    data = load_audit_data()
    if modality == "visual":
        builders = visual_feature_builders(data)
    elif modality == "skeleton":
        builders = skeleton_feature_builders(data)
    elif modality == "imu":
        builders = imu_feature_builders(data)
    else:
        raise ValueError(modality)

    group_results: list[dict[str, Any]] = []
    for group_name, builder in builders.items():
        block = builder()
        if block.values.shape[0] != len(data.labels) or block.values.shape[1] != len(block.names):
            raise RuntimeError(f"feature contract failed for {group_name}")
        alpha_rows = []
        for alpha in ALPHAS:
            aggregate, pair_rows = h1_louo_pair_metrics(
                block.values[: data.source_count],
                data.source_labels,
                data.source_users,
                alpha,
            )
            alpha_rows.append(
                {"alpha": alpha, "h1": aggregate, "pair_metrics": pair_rows}
            )
        best_alpha = max(
            alpha_rows,
            key=lambda row: (
                row["h1"]["macro_pair_balanced_accuracy"],
                row["h1"]["macro_pair_user_accuracy"],
                row["h1"]["worst_pair_balanced_accuracy"],
                -row["alpha"],
            ),
        )
        group_results.append(
            {
                "feature_group": group_name,
                "dimensions": int(block.values.shape[1]),
                "alpha": float(best_alpha["alpha"]),
                "h1": best_alpha["h1"],
                "pair_metrics": best_alpha["pair_metrics"],
                "alpha_grid": alpha_rows,
            }
        )
        print(
            f"{modality} H1 {group_name} dim={block.values.shape[1]} "
            f"alpha={best_alpha['alpha']:g} "
            f"macro_pair_BA={best_alpha['h1']['macro_pair_balanced_accuracy']:.4f}",
            flush=True,
        )

    selected = max(group_results, key=selection_key)
    selected_block = builders[selected["feature_group"]]()
    h2_pairs, importance, h2_samples = h2_pair_confirmation(
        selected_block, data, float(selected["alpha"]), modality
    )
    result = {
        "modality": modality,
        "status": "complete_h1_selected_h2_frozen_confirmation",
        "protocol": {
            "source_split": "H1_selection + E0_p87_sequence_source",
            "source_users": sorted(np.unique(data.source_users).tolist()),
            "source_samples": int(data.source_count),
            "selection": "leave-one-source-user-out; one global recipe per modality",
            "pair_specific_feature_selection": False,
            "pair_specific_alpha_selection": False,
            "h2_users": sorted(np.unique(data.h2_users).tolist()),
            "h2_samples": int(len(data.h2_labels)),
            "h3_read": False,
            "final_model_trained": False,
        },
        "selected_recipe": {
            "feature_group": selected["feature_group"],
            "dimensions": selected["dimensions"],
            "alpha": selected["alpha"],
            "h1": selected["h1"],
        },
        "h1_feature_groups": group_results,
        "h2_pair_confirmation": h2_pairs,
        "selected_feature_importance": importance,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{modality}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(OUTPUT / f"{modality}_feature_importance.csv", importance)
    write_csv(OUTPUT / f"{modality}_h2_pair_predictions.csv", h2_samples)
    print(
        f"selected {modality}: {selected['feature_group']} alpha={selected['alpha']:g}",
        flush=True,
    )


def run_session() -> None:
    data = load_audit_data()
    emission, posterior, metadata, source_audit = construct_p87_session_probability(
        data.h2_sample_ids
    )
    variants = {"p87_emission": emission, "p87_session": posterior}
    results: dict[str, list[dict[str, Any]]] = {}
    sample_rows: list[dict[str, Any]] = []
    for variant, probability in variants.items():
        pair_rows = []
        for left, right, title in PAIR_SPECS:
            pair_mask = np.isin(data.h2_labels, (left, right))
            h2_indices = np.flatnonzero(pair_mask)
            pair_probability = probability[h2_indices][:, (left, right)]
            prediction = np.asarray((left, right))[np.argmax(pair_probability, axis=1)]
            pair_labels = data.h2_labels[pair_mask]
            pair_users = data.h2_users[pair_mask]
            per_user = {
                str(user): float(
                    np.mean(
                        prediction[pair_users == user]
                        == pair_labels[pair_users == user]
                    )
                )
                for user in np.unique(pair_users)
            }
            hard = data.h2_primary_target & (
                np.minimum(data.h2_labels, data.h2_p91_prediction)
                == min(left, right)
            ) & (
                np.maximum(data.h2_labels, data.h2_p91_prediction)
                == max(left, right)
            )
            hard_prediction = np.asarray((left, right))[
                np.argmax(probability[hard][:, (left, right)], axis=1)
            ] if int(hard.sum()) else np.empty(0, dtype=np.int64)
            pair_rows.append(
                {
                    "pair_id": pair_key(left, right),
                    "title": title,
                    "left": left,
                    "right": right,
                    "h2_samples": int(pair_mask.sum()),
                    "h2_users": int(len(per_user)),
                    "h2_per_user_accuracy": per_user,
                    "h2_users_above_random": int(
                        np.sum(np.asarray(list(per_user.values())) > 0.5)
                    ),
                    "hard_target_samples": int(hard.sum()),
                    "hard_target_users": int(len(np.unique(data.h2_users[hard]))),
                    "hard_rescue": int(
                        np.sum(hard_prediction == data.h2_labels[hard])
                    ),
                    "hard_accuracy": safe_rate(
                        int(np.sum(hard_prediction == data.h2_labels[hard])),
                        int(hard.sum()),
                    ),
                    **binary_metrics(pair_labels, prediction),
                }
            )
            for local, h2_row in enumerate(h2_indices):
                sample_rows.append(
                    {
                        "variant": variant,
                        "pair_id": pair_key(left, right),
                        "sample_id": data.h2_sample_ids[h2_row],
                        "user": data.h2_users[h2_row],
                        "true_class_id": int(data.h2_labels[h2_row]),
                        "prediction_id": int(prediction[local]),
                        "correct": int(prediction[local] == data.h2_labels[h2_row]),
                        "primary_hard_target": int(hard[h2_row]),
                    }
                )
        results[variant] = pair_rows
    output = {
        "modality": "session",
        "status": "complete_fixed_p87_pair_restriction",
        "protocol": {
            "feature": "P87 structured posterior marginal restricted to each pair",
            "configuration_selected_in_this_audit": False,
            "h2_users": sorted(np.unique(data.h2_users).tolist()),
            "h3_read": False,
        },
        "session_source_audit": source_audit,
        "session_metadata_coverage": int(np.sum(metadata[:, 0] >= 0)),
        "variants": results,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "session_result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(OUTPUT / "session_h2_pair_predictions.csv", sample_rows)
    print("session fixed pair audit complete", flush=True)


def current_confusion_rows(data: AuditData, names: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    for left, right, title in PAIR_SPECS:
        selected = data.h2_primary_target & (
            np.minimum(data.h2_labels, data.h2_p91_prediction) == min(left, right)
        ) & (np.maximum(data.h2_labels, data.h2_p91_prediction) == max(left, right))
        direction_parts = []
        for truth, prediction in ((left, right), (right, left)):
            count = int(
                np.sum(
                    selected
                    & (data.h2_labels == truth)
                    & (data.h2_p91_prediction == prediction)
                )
            )
            if count:
                direction_parts.append(
                    f"{names[truth]}->{names[prediction]}={count}"
                )
        rows.append(
            {
                "pair_id": pair_key(left, right),
                "title": title,
                "class_a": names[left],
                "class_b": names[right],
                "hard_target_samples": int(selected.sum()),
                "hard_target_users": int(len(np.unique(data.h2_users[selected]))),
                "hard_target_user_ids": ";".join(
                    sorted(np.unique(data.h2_users[selected]).tolist())
                ),
                "directions": ";".join(direction_parts),
            }
        )
    return rows


def assemble() -> None:
    data = load_audit_data()
    names = load_names()
    learned = {
        modality: json.loads(
            (OUTPUT / f"{modality}_result.json").read_text(encoding="utf-8")
        )
        for modality in ("visual", "skeleton", "imu")
    }
    session = json.loads((OUTPUT / "session_result.json").read_text(encoding="utf-8"))
    confusion = current_confusion_rows(data, names)
    pair_rows = []
    for base in confusion:
        item = dict(base)
        pair_id = base["pair_id"]
        for modality, result in learned.items():
            pair = next(
                row for row in result["h2_pair_confirmation"] if row["pair_id"] == pair_id
            )
            item[f"{modality}_h2_balanced_accuracy"] = pair["balanced_accuracy"]
            item[f"{modality}_h2_accuracy"] = pair["accuracy"]
            item[f"{modality}_hard_rescue"] = pair["hard_rescue"]
            item[f"{modality}_hard_accuracy"] = pair["hard_accuracy"]
            item[f"{modality}_users_above_random"] = pair["h2_users_above_random"]
        session_pair = next(
            row
            for row in session["variants"]["p87_session"]
            if row["pair_id"] == pair_id
        )
        item["session_h2_balanced_accuracy"] = session_pair["balanced_accuracy"]
        item["session_h2_accuracy"] = session_pair["accuracy"]
        item["session_hard_rescue"] = session_pair["hard_rescue"]
        item["session_hard_accuracy"] = session_pair["hard_accuracy"]
        item["session_users_above_random"] = session_pair["h2_users_above_random"]
        pair_rows.append(item)

    h1_rows = []
    h1_pair_winners = []
    for modality, result in learned.items():
        for group in result["h1_feature_groups"]:
            h1_rows.append(
                {
                    "modality": modality,
                    "feature_group": group["feature_group"],
                    "dimensions": group["dimensions"],
                    "selected_alpha": group["alpha"],
                    **group["h1"],
                    "globally_selected": int(
                        group["feature_group"]
                        == result["selected_recipe"]["feature_group"]
                    ),
                }
            )
        global_group = result["selected_recipe"]["feature_group"]
        for left, right, title in PAIR_SPECS:
            pair_id = pair_key(left, right)
            choices = []
            for group in result["h1_feature_groups"]:
                pair = next(
                    row for row in group["pair_metrics"] if row["pair_id"] == pair_id
                )
                choices.append(
                    {
                        "feature_group": group["feature_group"],
                        "balanced_accuracy": pair["balanced_accuracy"],
                        "accuracy": pair["accuracy"],
                    }
                )
            winner = max(
                choices,
                key=lambda row: (row["balanced_accuracy"], row["accuracy"]),
            )
            global_pair = next(
                row for row in choices if row["feature_group"] == global_group
            )
            h1_pair_winners.append(
                {
                    "modality": modality,
                    "pair_id": pair_id,
                    "title": title,
                    "h1_pair_best_feature_group": winner["feature_group"],
                    "h1_pair_best_balanced_accuracy": winner["balanced_accuracy"],
                    "global_selected_feature_group": global_group,
                    "global_recipe_pair_balanced_accuracy": global_pair[
                        "balanced_accuracy"
                    ],
                }
            )

    requested_ids = {pair_key(left, right) for left, right, _ in PAIR_SPECS}
    all_current = []
    for row in read_csv(ALL_CURRENT_PAIRS):
        users = int(row["users"])
        errors = int(row["target_errors"])
        if users >= 2:
            tier = "cross_user"
        elif errors >= 3:
            tier = "single_user_recurrent"
        else:
            tier = "single_user_sparse"
        all_current.append(
            {
                "pair_id": row["pair_id"],
                "class_a_name": row["class_a_name"],
                "class_b_name": row["class_b_name"],
                "target_errors": errors,
                "users": users,
                "user_ids": row["user_ids"],
                "directions": row["directions"],
                "priority_tier": tier,
                "included_in_probe": int(row["pair_id"] in requested_ids),
            }
        )
    all_current.sort(
        key=lambda row: (
            0 if row["priority_tier"] == "cross_user" else 1,
            -row["target_errors"],
            row["pair_id"],
        )
    )

    prediction_maps: dict[str, dict[tuple[str, str], int]] = {}
    for modality in ("visual", "skeleton", "imu"):
        rows = read_csv(OUTPUT / f"{modality}_h2_pair_predictions.csv")
        prediction_maps[modality] = {
            (row["pair_id"], row["sample_id"]): int(row["correct"])
            for row in rows
            if row["primary_hard_target"] == "1"
        }
    session_rows = read_csv(OUTPUT / "session_h2_pair_predictions.csv")
    prediction_maps["session"] = {
        (row["pair_id"], row["sample_id"]): int(row["correct"])
        for row in session_rows
        if row["variant"] == "p87_session" and row["primary_hard_target"] == "1"
    }
    hard_keys = sorted(set().union(*(set(values) for values in prediction_maps.values())))
    if len(hard_keys) != 19:
        raise RuntimeError("specified confusion pairs no longer cover 19 hard targets")
    overlap_rows = []
    hard_lookup = {row["sample_id"]: row for row in read_csv(HARD_SAMPLES)}
    for pair_id, sample_id in hard_keys:
        source = hard_lookup[sample_id]
        item: dict[str, Any] = {
            "pair_id": pair_id,
            "sample_id": sample_id,
            "user": source["user"],
            "true_class_name": source["true_class_name"],
            "p91_prediction_name": source["p91_prediction_name"],
        }
        for modality in ("visual", "skeleton", "imu", "session"):
            item[f"{modality}_correct"] = prediction_maps[modality][
                (pair_id, sample_id)
            ]
        item["any_modality_correct"] = int(
            any(item[f"{modality}_correct"] for modality in prediction_maps)
        )
        item["correct_modality_count"] = int(
            sum(item[f"{modality}_correct"] for modality in prediction_maps)
        )
        overlap_rows.append(item)
    overlap = {
        "hard_targets_in_eight_pairs": len(overlap_rows),
        "visual_rescue": sum(row["visual_correct"] for row in overlap_rows),
        "skeleton_rescue": sum(row["skeleton_correct"] for row in overlap_rows),
        "imu_rescue": sum(row["imu_correct"] for row in overlap_rows),
        "session_rescue": sum(row["session_correct"] for row in overlap_rows),
        "oracle_union_rescue": sum(row["any_modality_correct"] for row in overlap_rows),
        "unresolved_by_all": sum(not row["any_modality_correct"] for row in overlap_rows),
        "visual_unique": sum(
            row["visual_correct"]
            and not row["skeleton_correct"]
            and not row["imu_correct"]
            and not row["session_correct"]
            for row in overlap_rows
        ),
        "skeleton_unique": sum(
            row["skeleton_correct"]
            and not row["visual_correct"]
            and not row["imu_correct"]
            and not row["session_correct"]
            for row in overlap_rows
        ),
        "imu_unique": sum(
            row["imu_correct"]
            and not row["visual_correct"]
            and not row["skeleton_correct"]
            and not row["session_correct"]
            for row in overlap_rows
        ),
        "session_unique": sum(
            row["session_correct"]
            and not row["visual_correct"]
            and not row["skeleton_correct"]
            and not row["imu_correct"]
            for row in overlap_rows
        ),
    }

    aggregate = {}
    for modality, result in learned.items():
        h2_rows = result["h2_pair_confirmation"]
        aggregate[modality] = {
            "h1_macro_pair_balanced_accuracy": result["selected_recipe"]["h1"][
                "macro_pair_balanced_accuracy"
            ],
            "h2_macro_pair_balanced_accuracy": float(
                np.mean([row["balanced_accuracy"] for row in h2_rows])
            ),
            "h2_worst_pair_balanced_accuracy": float(
                np.min([row["balanced_accuracy"] for row in h2_rows])
            ),
            "hard_rescue": int(sum(row["hard_rescue"] for row in h2_rows)),
            "hard_targets": int(sum(row["hard_target_samples"] for row in h2_rows)),
        }
    session_h2 = session["variants"]["p87_session"]
    aggregate["session"] = {
        "h1_macro_pair_balanced_accuracy": None,
        "h2_macro_pair_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in session_h2])
        ),
        "h2_worst_pair_balanced_accuracy": float(
            np.min([row["balanced_accuracy"] for row in session_h2])
        ),
        "hard_rescue": int(sum(row["hard_rescue"] for row in session_h2)),
        "hard_targets": int(sum(row["hard_target_samples"] for row in session_h2)),
    }
    write_csv(OUTPUT / "current_confusion_audit.csv", confusion)
    write_csv(OUTPUT / "all_current_confusions.csv", all_current)
    write_csv(OUTPUT / "modality_capability_h2.csv", pair_rows)
    write_csv(OUTPUT / "h1_feature_discovery.csv", h1_rows)
    write_csv(OUTPUT / "h1_pair_feature_winners.csv", h1_pair_winners)
    write_csv(OUTPUT / "hard_target_modality_overlap.csv", overlap_rows)
    summary = {
        "experiment_id": "p91_confusion_feature_capability_v1",
        "status": "complete_h1_selected_h2_frozen_representation_audit",
        "protocol": {
            "primary_target": (
                "P91 confidence < 0.95, P91 error, truth inside dynamic Top-5"
            ),
            "primary_target_samples": int(data.h2_primary_target.sum()),
            "source": "H1 + established embargo users, leave-one-user-out",
            "confirmation": "one frozen H2 evaluation per modality recipe",
            "probe": "standardized class-balanced Ridge binary probe; diagnostic only",
            "pair_specific_recipe_selection": False,
            "h3_read": False,
            "p91_modified": False,
            "final_model_trained": False,
        },
        "selected_recipes": {
            modality: result["selected_recipe"] for modality, result in learned.items()
        },
        "aggregate_capability": aggregate,
        "hard_target_overlap": overlap,
        "all_current_confusions": all_current,
        "confusions": confusion,
        "modality_capability": pair_rows,
        "session": session,
        "decision": {
            "three_modality_large_teacher": "not_supported_by_current_probe",
            "shared_candidate_conditioned_teacher": "supported_for_next_H1_capability_test",
            "motion_encoder_redesign": "only_narrow_hand_head_and_arm_statistics_followup_supported",
            "keep_visual_direction": "workspace_object_interaction_with_multiple_time_positions",
            "close_or_deprioritize": [
                "full-body Skeleton as a general hard-pool reranker",
                "all-device IMU temporal/frequency encoder",
                "pair-specific specialist proliferation",
                "single late-window Visual representation",
            ],
            "reason": (
                "Visual workspace features transfer best; Skeleton has no unique rescue in the 19 specified hard targets; simple arm IMU and session each add two unique oracle rescues, which favors a shared candidate-conditioned audit rather than a large unconditional fusion teacher."
            ),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["selected_recipes"], ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.assemble:
        assemble()
        return
    if args.modality == "session":
        run_session()
        return
    if args.modality in {"visual", "skeleton", "imu"}:
        run_learned_modality(args.modality)
        return
    raise SystemExit("choose --modality or --assemble")


if __name__ == "__main__":
    main()
