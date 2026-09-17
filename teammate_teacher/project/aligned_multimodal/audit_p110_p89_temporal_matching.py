"""P89-error-conditioned few-shot temporal matching audit.

This is not a 40-class B model and does not use A9.  It evaluates frozen
family-local support matching with source-subject-only method selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from a18_full_teacher_data import A18_SOURCE_USERS, load_a18_data


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_P89 = HERE / "runs/p89_verified_micro_union_audit_v1/validation_predictions.npz"
DEFAULT_MANIFEST = HERE / "data/six_modality_audit/train_union_manifest.csv"
DEFAULT_P30 = HERE / "runs/p30_shared_dir_roi_features_full/trial_feature_cache"
DEFAULT_P29 = HERE / "runs/p29_dir_multiscale_roi_full/trial_roi_cache"
DEFAULT_CLASSES = PROJECT / "class_mapping.csv"
DEFAULT_OUTPUT = HERE / "runs/p110_p89_temporal_matching_v1"

SEED = 20260824
STEPS = 12
PROJECTION_DIM = 24
NEAREST_SUPPORT = 3

P89_FOLDS = {
    "H1": ("user6", "user8", "user17", "user23"),
    "H2": ("user5", "user7", "user16", "user18", "user19"),
    "H3": ("user20", "user22", "user24", "user3", "user4", "user9"),
}
FAMILIES = {
    "TABLE_ORAL": (6, 7, 8, 9, 10, 11, 14, 37),
    "DEVICE_HEAD_BODY": (17, 19, 20, 23, 24, 26, 38, 39),
    "DOCUMENT": (18, 21, 22),
    "POSTURE_LOCOMOTION": (28, 29, 32, 34, 35, 36),
    "FACE_WIPE": (0, 4, 14),
}
HARD_TRUE_CLASSES = (8, 38, 19, 37, 26, 7, 24, 10, 18, 4, 20, 39, 23, 34)

BASE_METHODS = {
    "V_OTAM": ("visual", "ordered"),
    "V_SET": ("visual", "set"),
    "V_GEOM": ("geometry", "ordered"),
    "S_DTW": ("skeleton", "ordered"),
    "I_DTW": ("imu", "ordered"),
}
METHOD_COMPONENTS = {
    "V_OTAM": ("V_OTAM",),
    "V_SET": ("V_SET",),
    "V_GEOM": ("V_GEOM",),
    "S_DTW": ("S_DTW",),
    "I_DTW": ("I_DTW",),
    "V_OTAM+S_DTW": ("V_OTAM", "S_DTW"),
    "V_SET+S_DTW": ("V_SET", "S_DTW"),
    "V_OTAM+I_DTW": ("V_OTAM", "I_DTW"),
    "V_OTAM+V_GEOM": ("V_OTAM", "V_GEOM"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p89", type=Path, default=DEFAULT_P89)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p30-root", type=Path, default=DEFAULT_P30)
    parser.add_argument("--p29-root", type=Path, default=DEFAULT_P29)
    parser.add_argument("--class-mapping", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild-sequence-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise RuntimeError(f"refusing to write empty P110 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def class_names(path: Path) -> dict[int, str]:
    result = {
        int(row["action_id"]): str(row["action_name"])
        for row in read_csv(path.resolve())
    }
    if set(result) != set(range(40)):
        raise RuntimeError("P110 class mapping differs from 0..39")
    return result


def parse_canonical_id(sample_id: str) -> tuple[int, str, str]:
    parts = str(sample_id).split("__")
    if len(parts) != 4 or not parts[1].startswith("c") or not parts[2].startswith("user"):
        raise ValueError(f"invalid canonical sample id: {sample_id}")
    return int(parts[1][1:]), parts[2], parts[3]


def load_p89(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as source:
        ids = np.concatenate([source[f"h{fold}_sample_ids"].astype(str) for fold in (1, 2, 3)])
        prediction = np.concatenate([source[f"h{fold}_safe"].astype(np.int64) for fold in (1, 2, 3)])
        fold_names = np.concatenate(
            [np.repeat(f"H{fold}", len(source[f"h{fold}_sample_ids"])) for fold in (1, 2, 3)]
        ).astype(str)
    parsed = [parse_canonical_id(value) for value in ids]
    labels = np.asarray([value[0] for value in parsed], dtype=np.int64)
    users = np.asarray([value[1] for value in parsed], dtype=str)
    if len(ids) != 2470 or len(set(ids.tolist())) != 2470:
        raise RuntimeError("P110 P89 canonical rows differ from 2470 unique samples")
    if int(np.sum(labels == prediction)) != 2117:
        raise RuntimeError("P110 P89 Safe baseline differs from 2117/2470")
    for fold_name, held_users in P89_FOLDS.items():
        if set(users[fold_names == fold_name].tolist()) != set(held_users):
            raise RuntimeError(f"P110 {fold_name} users differ")
    return {
        "sample_ids": ids,
        "labels": labels,
        "users": users,
        "fold_names": fold_names,
        "prediction": prediction,
    }


def uniform_indices(length: int, steps: int = STEPS) -> np.ndarray:
    if length <= 0:
        raise ValueError("cannot resample an empty sequence")
    return np.rint(np.linspace(0, length - 1, steps)).astype(np.int64)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-6)


def manifest_lookup(path: Path) -> dict[tuple[int, str, str], str]:
    result: dict[tuple[int, str, str], str] = {}
    for row in read_csv(path.resolve()):
        if row["split"] != "train":
            continue
        key = (int(row["class_id"]), row["user_id"], row["trial_id"])
        if key in result:
            raise RuntimeError(f"duplicate P110 manifest key: {key}")
        result[key] = row["sample_id"]
    return result


def visual_sequence(
    feature_path: Path,
    roi_path: Path,
    projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(feature_path, allow_pickle=False) as source:
        frame_ids = source["frame_ids"].astype(str)
        modality_names = source["modality_names"].astype(str).tolist()
        region_names = source["region_names"].astype(str).tolist()
        raw = source["features"].astype(np.float32)
        quality = source["roi_quality"].astype(np.float32)
        valid = source["roi_valid"].astype(np.float32)
    with np.load(roi_path, allow_pickle=False) as source:
        if not np.array_equal(frame_ids, source["frame_ids"].astype(str)):
            raise RuntimeError(f"P110 P29/P30 frame alignment differs for {feature_path}")
        boxes = source["roi_boxes_xyxy"].astype(np.float32)
        arm = source["arm_joint_xy_conf_for_roi"].astype(np.float32)
        ambiguous = source["left_right_ambiguous"].astype(np.float32)
        pose_quality = source["pose_quality_factor"].astype(np.float32)
        width = float(np.asarray(source["image_width"]).item())
        height = float(np.asarray(source["image_height"]).item())

    modality_ids = [modality_names.index(value) for value in ("depth", "ir")]
    region_ids = [region_names.index(value) for value in ("left_hand", "right_hand", "hand_workspace")]
    chosen = raw[:, modality_ids][:, :, region_ids]
    chosen = l2_normalize(chosen)
    projected = np.einsum("tmrd,dp->tmrp", chosen, projection, optimize=True)
    projected = l2_normalize(projected)
    blocks: list[np.ndarray] = []
    for modality in range(2):
        left, right, workspace = [projected[:, modality, index] for index in range(3)]
        blocks.extend((left, right, workspace, left - right, 0.5 * (left + right) - workspace))
    appearance = l2_normalize(np.concatenate(blocks, axis=-1))

    selected_boxes = boxes[:, region_ids].copy()
    selected_boxes[..., (0, 2)] /= max(width, 1.0)
    selected_boxes[..., (1, 3)] /= max(height, 1.0)
    center = 0.5 * (selected_boxes[..., :2] + selected_boxes[..., 2:])
    size = np.maximum(selected_boxes[..., 2:] - selected_boxes[..., :2], 0.0)
    box_base = np.concatenate((center, size), axis=-1).reshape(len(boxes), -1)
    left_center, right_center, workspace_center = [center[:, index] for index in range(3)]
    relative = np.concatenate(
        (left_center - right_center, left_center - workspace_center, right_center - workspace_center),
        axis=-1,
    )
    distance = np.stack(
        (
            np.linalg.norm(left_center - right_center, axis=-1),
            np.linalg.norm(left_center - workspace_center, axis=-1),
            np.linalg.norm(right_center - workspace_center, axis=-1),
        ),
        axis=-1,
    )
    area = (size[..., 0] * size[..., 1]).reshape(len(boxes), -1)
    arm_scaled = arm.copy()
    arm_scaled[..., 0] /= max(width, 1.0)
    arm_scaled[..., 1] /= max(height, 1.0)
    geometry_base = np.concatenate(
        (
            box_base,
            relative,
            distance,
            area,
            arm_scaled.reshape(len(arm_scaled), -1),
            quality[:, region_ids],
            valid[:, region_ids],
            ambiguous[:, None],
            pose_quality[:, None],
        ),
        axis=-1,
    )
    # P29 intentionally stores NaN joint coordinates when pose/ROI geometry is
    # unavailable.  Keep its explicit quality/validity channels as the missing
    # evidence signal, but never allow the coordinate sentinel into a metric.
    geometry_base = np.nan_to_num(geometry_base, nan=0.0, posinf=0.0, neginf=0.0)
    delta = np.diff(geometry_base, axis=0, prepend=geometry_base[:1])
    geometry = np.concatenate((geometry_base, delta), axis=-1).astype(np.float32)
    take = uniform_indices(len(frame_ids))
    return appearance[take].astype(np.float32), geometry[take].astype(np.float32)


def skeleton_sequences(data: Any) -> np.ndarray:
    sequence = data.skeleton_sequence.astype(np.float32)
    mask = data.skeleton_mask.astype(bool)
    joints = np.asarray([0, 1, 4, 8, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64)
    selected = sequence[:, :, joints]
    selected_mask = mask[:, :, joints]
    position = selected[..., :3].reshape(len(sequence), 32, -1)
    bone = np.linalg.norm(selected[..., 3:6], axis=-1)
    speed = np.linalg.norm(selected[..., 6:9], axis=-1)
    acceleration = np.linalg.norm(selected[..., 9:12], axis=-1)
    left_wrist = sequence[:, :, 13, :3]
    right_wrist = sequence[:, :, 16, :3]
    head = sequence[:, :, 10, :3]
    pelvis = sequence[:, :, 0, :3]
    relations = np.stack(
        (
            np.linalg.norm(left_wrist - right_wrist, axis=-1),
            np.linalg.norm(left_wrist - head, axis=-1),
            np.linalg.norm(right_wrist - head, axis=-1),
            np.linalg.norm(left_wrist - pelvis, axis=-1),
            np.linalg.norm(right_wrist - pelvis, axis=-1),
            head[..., 1] - pelvis[..., 1],
        ),
        axis=-1,
    )
    values = np.concatenate(
        (position, bone, speed, acceleration, relations, selected_mask.astype(np.float32)),
        axis=-1,
    )
    values[~np.isfinite(values)] = 0.0
    return values[:, uniform_indices(32)].astype(np.float32)


def imu_sequences(data: Any) -> np.ndarray:
    values = data.imu_sequence.astype(np.float32)
    mask = data.imu_mask.astype(bool)
    count = np.maximum(mask.sum(axis=3), 1)
    mean = np.sum(values * mask[..., None], axis=3) / count[..., None]
    torso = mean[:, :, 0]
    left = mean[:, :, 1]
    right = mean[:, :, 2]

    def magnitude(block: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.linalg.norm(block[..., start : start + 3], axis=-1) for start in (0, 3, 6, 9)],
            axis=-1,
        )

    result = np.concatenate(
        (
            torso,
            left,
            right,
            left - right,
            magnitude(torso),
            magnitude(left),
            magnitude(right),
            mask[:, :, :3].any(axis=3).astype(np.float32),
        ),
        axis=-1,
    )
    result[~np.isfinite(result)] = 0.0
    return result[:, uniform_indices(32)].astype(np.float32)


def build_sequence_cache(
    path: Path,
    data: Any,
    manifest: Path,
    p30_root: Path,
    p29_root: Path,
) -> None:
    lookup = manifest_lookup(manifest)
    rng = np.random.default_rng(SEED)
    projection = rng.choice((-1.0, 1.0), size=(896, PROJECTION_DIM)).astype(np.float32)
    projection /= math.sqrt(PROJECTION_DIM)
    visual: list[np.ndarray] = []
    geometry: list[np.ndarray] = []
    for index, sample_id in enumerate(data.sample_ids.astype(str), start=1):
        key = parse_canonical_id(sample_id)
        cache_id = lookup[key]
        relative = Path(*cache_id.split("/")).with_suffix(".npz")
        visual_value, geometry_value = visual_sequence(
            p30_root.resolve() / relative,
            p29_root.resolve() / relative,
            projection,
        )
        visual.append(visual_value)
        geometry.append(geometry_value)
        if index % 250 == 0 or index == len(data.sample_ids):
            print(f"P110 sequence cache {index}/{len(data.sample_ids)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=data.sample_ids.astype(str),
        users=data.users.astype(str),
        labels=data.labels.astype(np.int64),
        visual=np.stack(visual).astype(np.float16),
        geometry=np.stack(geometry).astype(np.float16),
        skeleton=skeleton_sequences(data).astype(np.float16),
        imu=imu_sequences(data).astype(np.float16),
        skeleton_available=data.skeleton_available.astype(bool),
        imu_available=data.imu_available.astype(bool),
    )


def load_sequence_cache(path: Path, data: Any) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as source:
        result = {key: np.asarray(source[key]) for key in source.files}
    if not np.array_equal(result["sample_ids"].astype(str), data.sample_ids.astype(str)):
        raise RuntimeError("P110 sequence cache sample alignment differs")
    for key in ("visual", "geometry", "skeleton", "imu"):
        if result[key].shape[:2] != (len(data.sample_ids), STEPS):
            raise RuntimeError(f"P110 {key} sequence shape differs: {result[key].shape}")
        if not np.isfinite(result[key]).all():
            raise RuntimeError(f"P110 {key} contains non-finite values")
        result[key] = result[key].astype(np.float32)
    return result


def source_standardize(
    values: np.ndarray,
    source: np.ndarray,
    available: np.ndarray | None = None,
) -> np.ndarray:
    fit = source if available is None else source[available[source]]
    flattened = values[fit].reshape(-1, values.shape[-1])
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    transformed = (values - mean) / np.maximum(std, 1e-4)
    if available is not None:
        transformed[~available] = 0.0
    return l2_normalize(transformed).astype(np.float32)


@torch.inference_mode()
def ordered_distance(
    query: np.ndarray,
    reference: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    result = np.empty((len(query), len(reference)), dtype=np.float32)
    reference_tensor = torch.from_numpy(np.ascontiguousarray(reference)).to(device)
    steps_query, steps_reference = query.shape[1], reference.shape[1]
    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        query_tensor = torch.from_numpy(np.ascontiguousarray(query[start:stop])).to(device)
        cost = 1.0 - torch.einsum("btd,rsd->brts", query_tensor, reference_tensor)
        previous = torch.full(
            (stop - start, len(reference), steps_reference + 1),
            torch.inf,
            dtype=torch.float32,
            device=device,
        )
        previous[:, :, 0] = 0.0
        for time_query in range(steps_query):
            current = torch.full_like(previous, torch.inf)
            for time_reference in range(1, steps_reference + 1):
                best = torch.minimum(
                    torch.minimum(previous[:, :, time_reference], current[:, :, time_reference - 1]),
                    previous[:, :, time_reference - 1],
                )
                current[:, :, time_reference] = cost[:, :, time_query, time_reference - 1] + best
            previous = current
        result[start:stop] = (
            previous[:, :, steps_reference] / float(steps_query + steps_reference)
        ).cpu().numpy()
    return result


@torch.inference_mode()
def set_distance(
    query: np.ndarray,
    reference: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    result = np.empty((len(query), len(reference)), dtype=np.float32)
    reference_tensor = torch.from_numpy(np.ascontiguousarray(reference)).to(device)
    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        query_tensor = torch.from_numpy(np.ascontiguousarray(query[start:stop])).to(device)
        cost = 1.0 - torch.einsum("btd,rsd->brts", query_tensor, reference_tensor)
        forward = cost.min(dim=3).values.mean(dim=2)
        backward = cost.min(dim=2).values.mean(dim=2)
        result[start:stop] = (0.5 * (forward + backward)).cpu().numpy()
    return result


def base_distance(
    method: str,
    sequences: dict[str, np.ndarray],
    query: np.ndarray,
    reference: np.ndarray,
    device: str,
    query_override: np.ndarray | None = None,
    zero_query: bool = False,
) -> np.ndarray:
    sequence_name, metric = BASE_METHODS[method]
    query_rows = query if query_override is None else query_override
    query_values = sequences[sequence_name][query_rows]
    if zero_query:
        query_values = np.zeros_like(query_values)
    reference_values = sequences[sequence_name][reference]
    if metric == "ordered":
        return ordered_distance(query_values, reference_values, device)
    return set_distance(query_values, reference_values, device)


def distance_scale(distance: np.ndarray, source_local: np.ndarray) -> float:
    selected = distance[np.ix_(source_local, source_local)]
    values = selected[np.triu_indices(len(source_local), k=1)]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if len(values) else 1.0


def combine_distances(
    components: tuple[str, ...],
    base: dict[str, np.ndarray],
    scales: dict[str, float],
) -> np.ndarray:
    return np.mean(
        [base[component] / max(scales[component], 1e-6) for component in components],
        axis=0,
    ).astype(np.float32)


def predict_from_distance(
    distance: np.ndarray,
    reference_labels: np.ndarray,
    classes: tuple[int, ...],
    reference_masks: list[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.empty(len(distance), dtype=np.int64)
    margins = np.empty(len(distance), dtype=np.float32)
    class_scores = np.empty((len(distance), len(classes)), dtype=np.float32)
    shared = isinstance(reference_masks, np.ndarray) and reference_masks.dtype == bool
    for row in range(len(distance)):
        allowed = reference_masks if shared else reference_masks[row]
        scores: list[float] = []
        for class_id in classes:
            values = distance[row, allowed & (reference_labels == class_id)]
            if not len(values):
                scores.append(float("inf"))
                continue
            count = min(NEAREST_SUPPORT, len(values))
            scores.append(float(np.mean(np.partition(values, count - 1)[:count])))
        scores_array = np.asarray(scores, dtype=np.float32)
        order = np.argsort(scores_array)
        predictions[row] = classes[int(order[0])]
        margins[row] = float(
            (scores_array[order[1]] - scores_array[order[0]])
            / max(abs(float(scores_array[order[1]])), 1e-6)
        )
        class_scores[row] = scores_array
    return predictions, margins, class_scores


def metrics(labels: np.ndarray, prediction: np.ndarray, classes: tuple[int, ...]) -> dict[str, Any]:
    if not len(labels):
        return {"rows": 0, "correct": 0, "accuracy": None, "balanced_accuracy": None, "macro_f1": None}
    return {
        "rows": int(len(labels)),
        "correct": int(np.sum(labels == prediction)),
        "accuracy": float(np.mean(labels == prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=list(classes), average="macro", zero_division=0)),
    }


def shuffled_rows(sample_ids: np.ndarray, users: np.ndarray, rows: np.ndarray) -> np.ndarray:
    result = rows.copy()
    for user in sorted(set(users[rows].tolist())):
        local = np.flatnonzero(users[rows] == user)
        if len(local) > 1:
            order = local[np.argsort(sample_ids[rows[local]])]
            result[order] = rows[np.roll(order, 1)]
    return result


def method_cross_distance(
    method: str,
    sequences: dict[str, np.ndarray],
    query: np.ndarray,
    reference: np.ndarray,
    scales: dict[str, float],
    device: str,
    query_override: np.ndarray | None = None,
    zero_query: bool = False,
) -> np.ndarray:
    base = {
        component: base_distance(
            component,
            sequences,
            query,
            reference,
            device,
            query_override=query_override,
            zero_query=zero_query,
        )
        for component in METHOD_COMPONENTS[method]
    }
    return combine_distances(METHOD_COMPONENTS[method], base, scales)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = class_names(args.class_mapping)
    p89 = load_p89(args.p89)
    data = load_a18_data()
    full_lookup = {value: index for index, value in enumerate(data.sample_ids.astype(str))}
    if not set(p89["sample_ids"].tolist()) <= set(full_lookup):
        raise RuntimeError("P110 P89 rows are not contained in A18 data")
    p89_full_rows = np.asarray([full_lookup[value] for value in p89["sample_ids"]], dtype=np.int64)
    if not np.array_equal(data.labels[p89_full_rows], p89["labels"]):
        raise RuntimeError("P110 P89/A18 label alignment differs")

    cache_path = output / "sequence_cache.npz"
    if args.rebuild_sequence_cache or not cache_path.is_file():
        build_sequence_cache(
            cache_path,
            data,
            args.manifest,
            args.p30_root,
            args.p29_root,
        )
    cached = load_sequence_cache(cache_path, data)
    smoke_families = ("DOCUMENT",) if args.smoke else tuple(FAMILIES)

    selected_predictions: dict[str, np.ndarray] = {
        family: np.full(len(p89["sample_ids"]), -1, dtype=np.int64) for family in smoke_families
    }
    selected_margins: dict[str, np.ndarray] = {
        family: np.full(len(p89["sample_ids"]), np.nan, dtype=np.float32) for family in smoke_families
    }
    family_true_predictions: dict[str, np.ndarray] = {
        family: np.full(len(p89["sample_ids"]), -1, dtype=np.int64) for family in smoke_families
    }
    fold_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    method_held_rows: list[dict[str, Any]] = []
    error_method_rows: list[dict[str, Any]] = []
    fold_details: list[dict[str, Any]] = []

    for fold_name, held_users_tuple in P89_FOLDS.items():
        if args.smoke and fold_name != "H1":
            continue
        held_users = set(held_users_tuple)
        source_full = np.flatnonzero(~np.isin(data.users, list(held_users)))
        held_full = np.flatnonzero(np.isin(data.users, list(held_users)))
        fold_p89 = np.flatnonzero(p89["fold_names"] == fold_name)
        if set(p89["users"][fold_p89].tolist()) != held_users:
            raise RuntimeError(f"P110 {fold_name} P89 fold mismatch")
        if not set(p89_full_rows[fold_p89].tolist()) <= set(held_full.tolist()):
            raise RuntimeError(f"P110 {fold_name} full-row alignment mismatch")

        sequences = {
            "visual": l2_normalize(cached["visual"]).astype(np.float32),
            "geometry": source_standardize(cached["geometry"], source_full),
            "skeleton": source_standardize(
                cached["skeleton"], source_full, cached["skeleton_available"].astype(bool)
            ),
            "imu": source_standardize(cached["imu"], source_full, cached["imu_available"].astype(bool)),
        }

        for family_name in smoke_families:
            classes = FAMILIES[family_name]
            family_full = np.flatnonzero(np.isin(data.labels, classes))
            family_users = data.users[family_full]
            source_local = np.flatnonzero(~np.isin(family_users, list(held_users)))
            held_local = np.flatnonzero(np.isin(family_users, list(held_users)))
            base_square: dict[str, np.ndarray] = {}
            scales: dict[str, float] = {}
            for method in BASE_METHODS:
                base_square[method] = base_distance(
                    method, sequences, family_full, family_full, args.device
                )
                scales[method] = distance_scale(base_square[method], source_local)

            method_square = {
                method: combine_distances(components, base_square, scales)
                for method, components in METHOD_COMPONENTS.items()
            }
            best_method: str | None = None
            best_key: tuple[float, float, float, int] | None = None
            source_labels = data.labels[family_full[source_local]]
            source_users = data.users[family_full[source_local]]
            reference_masks = [source_users != source_users[row] for row in range(len(source_local))]
            for method_index, method in enumerate(METHOD_COMPONENTS):
                distance = method_square[method][np.ix_(source_local, source_local)]
                prediction, _, _ = predict_from_distance(
                    distance, source_labels, classes, reference_masks
                )
                result = metrics(source_labels, prediction, classes)
                inner_rows.append(
                    {
                        "outer_fold": fold_name,
                        "family": family_name,
                        "method": method,
                        **result,
                    }
                )
                key = (
                    float(result["macro_f1"]),
                    float(result["balanced_accuracy"]),
                    float(result["accuracy"]),
                    -method_index,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_method = method
            assert best_method is not None

            held_family_full = family_full[held_local]
            family_p89_local = np.flatnonzero(
                (p89["fold_names"] == fold_name) & np.isin(p89["labels"], classes)
            )
            p89_by_full_row = {
                int(p89_full_rows[row]): int(row) for row in family_p89_local.tolist()
            }
            held_method_predictions: dict[str, np.ndarray] = {}
            held_method_margins: dict[str, np.ndarray] = {}
            for method in METHOD_COMPONENTS:
                held_distance = method_square[method][np.ix_(held_local, source_local)]
                method_prediction, method_margin, _ = predict_from_distance(
                    held_distance,
                    data.labels[family_full[source_local]],
                    classes,
                    np.ones(len(source_local), dtype=bool),
                )
                held_method_predictions[method] = method_prediction
                held_method_margins[method] = method_margin
                method_result = metrics(data.labels[held_family_full], method_prediction, classes)
                p89_method_prediction = np.asarray(
                    [
                        method_prediction[index]
                        for index, full_row in enumerate(held_family_full.tolist())
                        if full_row in p89_by_full_row
                    ],
                    dtype=np.int64,
                )
                p89_method_rows = np.asarray(
                    [
                        p89_by_full_row[full_row]
                        for full_row in held_family_full.tolist()
                        if full_row in p89_by_full_row
                    ],
                    dtype=np.int64,
                )
                p89_method_wrong = (
                    p89["prediction"][p89_method_rows] != p89["labels"][p89_method_rows]
                )
                method_held_rows.append(
                    {
                        "outer_fold": fold_name,
                        "family": family_name,
                        "method": method,
                        "source_selected": int(method == best_method),
                        **method_result,
                        "p89_family_rows": len(p89_method_rows),
                        "p89_errors": int(np.sum(p89_method_wrong)),
                        "p89_error_rescue": int(
                            np.sum(
                                p89_method_wrong
                                & (p89_method_prediction == p89["labels"][p89_method_rows])
                            )
                        ),
                        "p89_correct_harm": int(
                            np.sum(
                                ~p89_method_wrong
                                & (p89_method_prediction != p89["labels"][p89_method_rows])
                            )
                        ),
                    }
                )
                for local_index, full_row in enumerate(held_family_full.tolist()):
                    p89_row = p89_by_full_row.get(full_row)
                    if p89_row is None or p89["prediction"][p89_row] == p89["labels"][p89_row]:
                        continue
                    error_method_rows.append(
                        {
                            "sample_id": p89["sample_ids"][p89_row],
                            "subject": p89["users"][p89_row],
                            "outer_fold": fold_name,
                            "true_class": int(p89["labels"][p89_row]),
                            "true_name": names[int(p89["labels"][p89_row])],
                            "p89_prediction": int(p89["prediction"][p89_row]),
                            "p89_prediction_name": names[int(p89["prediction"][p89_row])],
                            "family": family_name,
                            "method": method,
                            "source_selected": int(method == best_method),
                            "specialist_prediction": int(method_prediction[local_index]),
                            "specialist_prediction_name": names[int(method_prediction[local_index])],
                            "specialist_margin": float(method_margin[local_index]),
                            "rescued": int(method_prediction[local_index] == p89["labels"][p89_row]),
                        }
                    )

            held_prediction = held_method_predictions[best_method]
            held_margin = held_method_margins[best_method]
            held_metrics = metrics(data.labels[held_family_full], held_prediction, classes)

            held_prediction_lookup = {
                int(row): int(prediction)
                for row, prediction in zip(held_family_full.tolist(), held_prediction.tolist())
            }
            for row in family_p89_local:
                family_true_predictions[family_name][row] = held_prediction_lookup[int(p89_full_rows[row])]
            base_wrong = p89["prediction"][family_p89_local] != p89["labels"][family_p89_local]
            candidate = family_true_predictions[family_name][family_p89_local]
            oracle_rescue = int(np.sum(base_wrong & (candidate == p89["labels"][family_p89_local])))
            oracle_harm = int(np.sum(~base_wrong & (candidate != p89["labels"][family_p89_local])))

            shuffled_full = shuffled_rows(
                data.sample_ids.astype(str), data.users.astype(str), held_family_full
            )
            shuffled_distance = method_cross_distance(
                best_method,
                sequences,
                held_family_full,
                family_full[source_local],
                scales,
                args.device,
                query_override=shuffled_full,
            )
            shuffled_prediction, _, _ = predict_from_distance(
                shuffled_distance,
                data.labels[family_full[source_local]],
                classes,
                np.ones(len(source_local), dtype=bool),
            )
            zero_distance = method_cross_distance(
                best_method,
                sequences,
                held_family_full,
                family_full[source_local],
                scales,
                args.device,
                zero_query=True,
            )
            zero_prediction, _, _ = predict_from_distance(
                zero_distance,
                data.labels[family_full[source_local]],
                classes,
                np.ones(len(source_local), dtype=bool),
            )

            all_held_full = p89_full_rows[fold_p89]
            route_distance = method_cross_distance(
                best_method,
                sequences,
                all_held_full,
                family_full[source_local],
                scales,
                args.device,
            )
            route_prediction, route_margin, _ = predict_from_distance(
                route_distance,
                data.labels[family_full[source_local]],
                classes,
                np.ones(len(source_local), dtype=bool),
            )
            selected_predictions[family_name][fold_p89] = route_prediction
            selected_margins[family_name][fold_p89] = route_margin

            fold_rows.append(
                {
                    "outer_fold": fold_name,
                    "family": family_name,
                    "classes": "|".join(map(str, classes)),
                    "source_rows": len(source_local),
                    "held_rows": len(held_local),
                    "selected_method": best_method,
                    "inner_macro_f1": best_key[0],
                    "held_accuracy": held_metrics["accuracy"],
                    "held_balanced_accuracy": held_metrics["balanced_accuracy"],
                    "held_macro_f1": held_metrics["macro_f1"],
                    "shuffle_accuracy": float(np.mean(shuffled_prediction == data.labels[held_family_full])),
                    "zero_accuracy": float(np.mean(zero_prediction == data.labels[held_family_full])),
                    "p89_family_rows": len(family_p89_local),
                    "p89_errors": int(np.sum(base_wrong)),
                    "oracle_rescue": oracle_rescue,
                    "oracle_harm": oracle_harm,
                    "oracle_net": oracle_rescue - oracle_harm,
                }
            )
            fold_details.append(
                {
                    "outer_fold": fold_name,
                    "family": family_name,
                    "held_users": sorted(held_users),
                    "selected_method": best_method,
                    "distance_scales": scales,
                    "inner_selection_key": best_key,
                    "held_metrics": held_metrics,
                }
            )
            print(
                f"P110 {fold_name} {family_name}: {best_method} "
                f"held={held_metrics['accuracy']:.4f} rescue/harm={oracle_rescue}/{oracle_harm}",
                flush=True,
            )

    if args.smoke:
        write_csv(output / "smoke_family_fold_metrics.csv", fold_rows)
        write_csv(output / "smoke_method_inner_metrics.csv", inner_rows)
        write_csv(output / "smoke_method_held_metrics.csv", method_held_rows)
        write_csv(output / "smoke_error_method_predictions.csv", error_method_rows)
        print("P110 smoke complete", flush=True)
        return

    memberships = defaultdict(list)
    for family_name, classes in FAMILIES.items():
        for class_id in classes:
            memberships[class_id].append(family_name)

    system = p89["prediction"].copy()
    entry_family = np.full(len(system), "", dtype="<U32")
    entry = np.zeros(len(system), dtype=bool)
    entry_specialist = np.full(len(system), -1, dtype=np.int64)
    entry_margin = np.full(len(system), np.nan, dtype=np.float32)
    for row, prediction in enumerate(p89["prediction"].tolist()):
        candidates = [value for value in memberships[int(prediction)] if value in selected_predictions]
        if len(candidates) != 1:
            continue
        family_name = candidates[0]
        specialist = int(selected_predictions[family_name][row])
        if specialist < 0:
            raise RuntimeError(f"P110 missing routed prediction for {family_name}/{row}")
        entry[row] = True
        entry_family[row] = family_name
        entry_specialist[row] = specialist
        entry_margin[row] = selected_margins[family_name][row]
        system[row] = specialist

    base_correct = p89["prediction"] == p89["labels"]
    system_correct = system == p89["labels"]
    rescue = (~base_correct) & system_correct
    harm = base_correct & (~system_correct)

    family_summary_rows: list[dict[str, Any]] = []
    for family_name, classes in FAMILIES.items():
        selected = np.isin(p89["labels"], classes)
        prediction = family_true_predictions[family_name][selected]
        if (prediction < 0).any():
            raise RuntimeError(f"P110 missing oracle-family prediction for {family_name}")
        labels = p89["labels"][selected]
        family_metrics = metrics(labels, prediction, classes)
        wrong = p89["prediction"][selected] != labels
        family_rescue = int(np.sum(wrong & (prediction == labels)))
        family_harm = int(np.sum(~wrong & (prediction != labels)))
        subject_accuracy = {
            user: float(np.mean(prediction[p89["users"][selected] == user] == labels[p89["users"][selected] == user]))
            for user in sorted(set(p89["users"][selected].tolist()))
            if np.any(p89["users"][selected] == user)
        }
        family_summary_rows.append(
            {
                "family": family_name,
                "classes": "|".join(map(str, classes)),
                "rows": family_metrics["rows"],
                "accuracy": family_metrics["accuracy"],
                "balanced_accuracy": family_metrics["balanced_accuracy"],
                "macro_f1": family_metrics["macro_f1"],
                "p89_errors": int(np.sum(wrong)),
                "oracle_rescue": family_rescue,
                "oracle_harm": family_harm,
                "oracle_net": family_rescue - family_harm,
                "worst_subject": min(subject_accuracy, key=subject_accuracy.get),
                "worst_subject_accuracy": min(subject_accuracy.values()),
                "selected_methods": "|".join(
                    f"{row['outer_fold']}:{row['selected_method']}"
                    for row in fold_rows
                    if row["family"] == family_name
                ),
            }
        )

    sample_family_rows: list[dict[str, Any]] = []
    for row in range(len(p89["sample_ids"])):
        candidate_families = sorted(
            set(memberships[int(p89["labels"][row])]) | set(memberships[int(p89["prediction"][row])])
        )
        for family_name in candidate_families:
            if family_name not in selected_predictions:
                continue
            sample_family_rows.append(
                {
                    "sample_id": p89["sample_ids"][row],
                    "subject": p89["users"][row],
                    "outer_fold": p89["fold_names"][row],
                    "true_class": int(p89["labels"][row]),
                    "p89_prediction": int(p89["prediction"][row]),
                    "p89_correct": int(base_correct[row]),
                    "family": family_name,
                    "true_in_family": int(int(p89["labels"][row]) in FAMILIES[family_name]),
                    "p89_prediction_in_family": int(int(p89["prediction"][row]) in FAMILIES[family_name]),
                    "specialist_prediction": int(selected_predictions[family_name][row]),
                    "specialist_margin": float(selected_margins[family_name][row]),
                    "specialist_correct": int(selected_predictions[family_name][row] == p89["labels"][row]),
                }
            )

    error_rows: list[dict[str, Any]] = []
    error_count = Counter(p89["labels"][~base_correct].tolist())
    for row in np.flatnonzero(~base_correct):
        true_families = memberships[int(p89["labels"][row])]
        predicted_families = memberships[int(p89["prediction"][row])]
        error_rows.append(
            {
                "sample_id": p89["sample_ids"][row],
                "subject": p89["users"][row],
                "outer_fold": p89["fold_names"][row],
                "true_class": int(p89["labels"][row]),
                "true_name": names[int(p89["labels"][row])],
                "p89_prediction": int(p89["prediction"][row]),
                "p89_prediction_name": names[int(p89["prediction"][row])],
                "true_class_error_count": error_count[int(p89["labels"][row])],
                "true_families": "|".join(true_families),
                "prediction_families": "|".join(predicted_families),
                "unique_top1_entry": int(entry[row]),
                "entry_family": entry_family[row],
                "specialist_prediction": int(entry_specialist[row]),
                "specialist_prediction_name": names[int(entry_specialist[row])] if entry_specialist[row] >= 0 else "",
                "rescued": int(rescue[row]),
                "still_wrong": int(not system_correct[row]),
            }
        )

    method_by_error: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in error_method_rows:
        method_by_error[str(row["sample_id"])].append(row)
    error_method_summary_rows: list[dict[str, Any]] = []
    for row in np.flatnonzero(~base_correct):
        sample_id = str(p89["sample_ids"][row])
        attempts = method_by_error[sample_id]
        rescued_methods = sorted(
            {f"{value['family']}:{value['method']}" for value in attempts if value["rescued"]}
        )
        selected_attempts = [value for value in attempts if value["source_selected"]]
        selected_rescues = sorted(
            {
                f"{value['family']}:{value['method']}"
                for value in selected_attempts
                if value["rescued"]
            }
        )
        error_method_summary_rows.append(
            {
                "sample_id": sample_id,
                "subject": p89["users"][row],
                "outer_fold": p89["fold_names"][row],
                "true_class": int(p89["labels"][row]),
                "true_name": names[int(p89["labels"][row])],
                "p89_prediction": int(p89["prediction"][row]),
                "p89_prediction_name": names[int(p89["prediction"][row])],
                "covered_by_frozen_family": int(bool(attempts)),
                "attempted_families": "|".join(sorted({value["family"] for value in attempts})),
                "source_selected_rescued": int(bool(selected_rescues)),
                "source_selected_rescuers": "|".join(selected_rescues),
                "posthoc_any_method_rescued": int(bool(rescued_methods)),
                "posthoc_rescuers": "|".join(rescued_methods),
            }
        )

    class_method_summary_rows: list[dict[str, Any]] = []
    for class_id, error_total in Counter(p89["labels"][~base_correct].tolist()).most_common():
        class_samples = {
            str(p89["sample_ids"][row])
            for row in np.flatnonzero((~base_correct) & (p89["labels"] == class_id))
        }
        class_attempts = [
            value for sample_id in class_samples for value in method_by_error[sample_id]
        ]
        per_method: list[tuple[int, str]] = []
        for method in METHOD_COMPONENTS:
            rescued_samples = {
                str(value["sample_id"])
                for value in class_attempts
                if value["method"] == method and value["rescued"]
            }
            per_method.append((len(rescued_samples), method))
        best_rescue, best_method = max(per_method, key=lambda value: (value[0], value[1]))
        selected_rescue = {
            str(value["sample_id"])
            for value in class_attempts
            if value["source_selected"] and value["rescued"]
        }
        any_rescue = {
            str(value["sample_id"]) for value in class_attempts if value["rescued"]
        }
        covered = {str(value["sample_id"]) for value in class_attempts}
        class_method_summary_rows.append(
            {
                "true_class": int(class_id),
                "true_name": names[int(class_id)],
                "p89_errors": int(error_total),
                "covered_errors": len(covered),
                "source_selected_rescue": len(selected_rescue),
                "best_single_method_posthoc": best_method if covered else "",
                "best_single_method_rescue": best_rescue,
                "posthoc_any_method_rescue": len(any_rescue),
                "unresolved_after_any_method": int(error_total) - len(any_rescue),
            }
        )

    covered_error_count = int(
        sum(value["covered_by_frozen_family"] for value in error_method_summary_rows)
    )
    selected_error_rescue = int(
        sum(value["source_selected_rescued"] for value in error_method_summary_rows)
    )
    any_method_error_rescue = int(
        sum(value["posthoc_any_method_rescued"] for value in error_method_summary_rows)
    )

    summary = {
        "status": "complete",
        "protocol": "P89 Top-1 error conditioned source-subject few-shot temporal matching",
        "uses_a9": False,
        "p89_baseline": {
            "rows": len(p89["labels"]),
            "correct": int(np.sum(base_correct)),
            "accuracy": float(np.mean(base_correct)),
            "errors": int(np.sum(~base_correct)),
            "top_true_error_classes": [
                {"class_id": int(class_id), "name": names[int(class_id)], "errors": int(count)}
                for class_id, count in Counter(p89["labels"][~base_correct].tolist()).most_common(20)
            ],
        },
        "families": FAMILIES,
        "hard_true_classes": HARD_TRUE_CLASSES,
        "matching": {
            "steps": STEPS,
            "nearest_support": NEAREST_SUPPORT,
            "methods": METHOD_COMPONENTS,
            "method_selected_source_only": True,
            "old_trial_aggregate_roi_probe_reused": False,
        },
        "top1_family_entry": {
            "entry_rows": int(np.sum(entry)),
            "abstain_rows": int(np.sum(~entry)),
            "correct": int(np.sum(system_correct)),
            "accuracy": float(np.mean(system_correct)),
            "rescue": int(np.sum(rescue)),
            "harm": int(np.sum(harm)),
            "net": int(np.sum(rescue) - np.sum(harm)),
        },
        "p89_error_only_audit": {
            "errors": int(np.sum(~base_correct)),
            "covered_by_frozen_family": covered_error_count,
            "uncovered": int(np.sum(~base_correct)) - covered_error_count,
            "source_selected_any_family_rescue": selected_error_rescue,
            "posthoc_any_frozen_method_rescue": any_method_error_rescue,
            "posthoc_oracle_warning": (
                "any-method rescue uses held labels after prediction and is a ceiling, not a router"
            ),
        },
        "constraints": {
            "source_subject_disjoint": True,
            "held_label_used_for_method_selection": False,
            "flat_40_class_b": False,
            "candidate_reranker": False,
            "threshold_sweep": False,
            "negative_results_preserved": True,
        },
        "artifacts": {
            "error_atlas": "p89_top1_error_atlas.csv",
            "family_fold_metrics": "family_fold_metrics.csv",
            "family_summary": "family_summary.csv",
            "method_inner_metrics": "method_inner_metrics.csv",
            "method_held_metrics": "method_held_metrics.csv",
            "error_method_predictions": "p89_error_method_predictions.csv",
            "error_method_summary": "p89_error_method_summary.csv",
            "class_method_summary": "p89_error_class_method_summary.csv",
            "sample_family_predictions": "sample_family_predictions.csv",
            "fold_details": "fold_details.json",
            "sequence_cache": "sequence_cache.npz",
        },
    }
    write_csv(output / "p89_top1_error_atlas.csv", error_rows)
    write_csv(output / "family_fold_metrics.csv", fold_rows)
    write_csv(output / "family_summary.csv", family_summary_rows)
    write_csv(output / "method_inner_metrics.csv", inner_rows)
    write_csv(output / "method_held_metrics.csv", method_held_rows)
    write_csv(output / "p89_error_method_predictions.csv", error_method_rows)
    write_csv(output / "p89_error_method_summary.csv", error_method_summary_rows)
    write_csv(output / "p89_error_class_method_summary.csv", class_method_summary_rows)
    write_csv(output / "sample_family_predictions.csv", sample_family_rows)
    (output / "fold_details.json").write_text(
        json.dumps(fold_details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
