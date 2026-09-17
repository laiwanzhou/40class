from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from audit_yolo11_pose_skeleton import frame_map
from p31_skeleton_imu_preprocessing import safe_trial_path


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_ROWS = PROJECT_DIR / "runs/p86_motion_window_cache_t16_v1/rows.csv"
TEST_ROWS = PROJECT_DIR / "runs/p87s_test_motion_window_t16_v1/rows.csv"
TRAIN_MANIFEST = PROJECT_DIR / "data/six_modality_audit/train_union_manifest.csv"
TEST_MANIFEST = PROJECT_DIR / "data/p46_test_union_manifest.csv"
TRAIN_METADATA = PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv"
TEST_METADATA = PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv"
TRAIN_VISUAL = PROJECT_DIR / "runs/p28_adaptive_ir_pose_skeleton_full/trial_cache"
TEST_VISUAL = PROJECT_DIR / "runs/p28_adaptive_ir_pose_skeleton_test/trial_cache"
SKELETON_CACHE = PROJECT_DIR / "runs/p89_hidden_subject_identity_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p89_ir_actor_identity_clustering_v1"
SEED = 20260816

# These are collection waves, not class labels.  Pairwise validation hides the
# identity of two users at a time, exactly matching each half of Public Test.
TRAIN_COHORTS = {
    "users1_5": [f"user{i}" for i in range(1, 6)],
    "users6_9": [f"user{i}" for i in range(6, 10)],
    "users16_20": [f"user{i}" for i in range(16, 21)],
    "users21_24": [f"user{i}" for i in range(21, 25)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover the two anonymous subjects in each Public-Test collection wave "
            "from visually localized IR appearance, body proportions, and recording blocks."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--frames-per-trial", type=int, default=7)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resize_values(patch: np.ndarray, width: int, height: int, local: bool) -> np.ndarray:
    if patch.size == 0:
        return np.zeros(width * height, dtype=np.float32)
    values = cv2.resize(patch, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    if local:
        lower, upper = np.percentile(values, (3.0, 97.0))
        values = (values - lower) / max(float(upper - lower), 8.0)
    else:
        values /= 255.0
    return np.clip(values, 0.0, 1.0).reshape(-1)


def clipped_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    x1 = int(np.clip(np.floor(x1), 0, width - 1))
    y1 = int(np.clip(np.floor(y1), 0, height - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, width))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, height))
    return image[y1:y2, x1:x2]


def frame_descriptor(image: np.ndarray, box: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box[:4], dtype=np.float64)
    width = max(x2 - x1, 4.0)
    height = max(y2 - y1, 4.0)
    full = clipped_crop(
        image,
        np.asarray((x1 - 0.05 * width, y1 - 0.03 * height, x2 + 0.05 * width, y2)),
    )

    valid = np.isfinite(keypoints[:, :2]).all(axis=1) & (keypoints[:, 2] >= 0.25)
    torso_joints = np.asarray((5, 6, 11, 12), dtype=np.int64)
    if np.sum(valid[torso_joints]) >= 3:
        points = keypoints[torso_joints[valid[torso_joints]], :2]
        tx1, ty1 = points.min(axis=0)
        tx2, ty2 = points.max(axis=0)
        torso_width = max(float(tx2 - tx1), 0.25 * width)
        torso_height = max(float(ty2 - ty1), 0.20 * height)
        torso_box = np.asarray(
            (
                tx1 - 0.35 * torso_width,
                ty1 - 0.25 * torso_height,
                tx2 + 0.35 * torso_width,
                ty2 + 0.35 * torso_height,
            )
        )
        shoulder_y = float(np.nanmedian(keypoints[[5, 6], 1])) if np.any(valid[[5, 6]]) else y1 + 0.25 * height
    else:
        torso_box = np.asarray((x1 + 0.10 * width, y1 + 0.16 * height, x2 - 0.10 * width, y1 + 0.62 * height))
        shoulder_y = y1 + 0.25 * height
    torso = clipped_crop(image, torso_box)
    head = clipped_crop(
        image,
        np.asarray((x1 + 0.10 * width, y1, x2 - 0.10 * width, min(shoulder_y + 0.08 * height, y1 + 0.42 * height))),
    )

    full_small = cv2.resize(full, (12, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(full_small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(full_small, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    gradient_profiles = np.concatenate((magnitude.mean(axis=0), magnitude.mean(axis=1))) / 255.0

    # Absolute IR intensity retains clothing reflectance; local contrast retains
    # clothing shape/texture across automatic exposure changes.
    return np.concatenate(
        (
            resize_values(full, 8, 16, False),
            resize_values(full, 8, 16, True),
            resize_values(torso, 8, 12, False),
            resize_values(torso, 8, 12, True),
            resize_values(head, 8, 8, False),
            resize_values(head, 8, 8, True),
            gradient_profiles.astype(np.float32),
        )
    ).astype(np.float32)


def trial_descriptor(
    row: dict[str, str],
    manifest: dict[str, dict[str, str]],
    visual_root: Path,
    frames_per_trial: int,
) -> tuple[np.ndarray, bool]:
    source_id = row["source_id"]
    trial = manifest.get(source_id)
    if trial is None and row["sample_id"] in manifest:
        trial = manifest[row["sample_id"]]
    visual_path = visual_root / safe_trial_path(source_id).with_suffix(".npz")
    if trial is None or not visual_path.is_file() or trial.get("ir_usable") != "1":
        return np.zeros(612, dtype=np.float32), False
    with np.load(visual_path, allow_pickle=False) as visual:
        frame_ids = np.asarray(visual["frame_ids"]).astype(str)
        boxes = np.asarray(visual["ir_boxes_xyxy_conf"], dtype=np.float32)
        keypoints = np.asarray(visual["ir_keypoints_xy_conf"], dtype=np.float32)
    usable = np.flatnonzero(
        np.isfinite(boxes[:, :4]).all(axis=1)
        & (boxes[:, 4] >= 0.10)
        & (np.sum(keypoints[:, :, 2] >= 0.25, axis=1) >= 6)
    )
    if not len(usable):
        return np.zeros(612, dtype=np.float32), False
    chosen_positions = np.linspace(0, len(usable) - 1, min(frames_per_trial, len(usable))).round().astype(np.int64)
    chosen = usable[np.unique(chosen_positions)]
    images = frame_map(Path(trial["ir_path"]), "ir")
    descriptors = []
    for index in chosen:
        image_path = images.get(frame_ids[index])
        if image_path is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        descriptors.append(frame_descriptor(image, boxes[index], keypoints[index]))
    if not descriptors:
        return np.zeros(612, dtype=np.float32), False
    values = np.stack(descriptors)
    # Median suppresses action-specific limb crossings and occasional bystanders.
    return np.median(values, axis=0).astype(np.float32), True


def build_appearance(
    rows: list[dict[str, str]],
    manifest_path: Path,
    visual_root: Path,
    cache_path: Path,
    frames_per_trial: int,
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sample_ids = np.asarray([row["sample_id"] for row in rows]).astype(str)
    if cache_path.is_file() and not rebuild:
        with np.load(cache_path) as data:
            if np.array_equal(data["sample_ids"].astype(str), sample_ids):
                return np.asarray(data["features"], dtype=np.float32), np.asarray(data["present"], dtype=bool)
    manifest_rows = read_csv(manifest_path)
    manifest: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        manifest[row["sample_id"]] = row
        official_id = row.get("official_sample_id", "")
        if official_id:
            manifest[official_id] = row
    features = []
    present = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        values, available = trial_descriptor(row, manifest, visual_root, frames_per_trial)
        features.append(values)
        present.append(available)
        if (index + 1) % 200 == 0 or index + 1 == len(rows):
            print(json.dumps({"feature_rows": index + 1, "total": len(rows), "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    matrix = np.stack(features)
    available = np.asarray(present, dtype=bool)
    np.savez_compressed(cache_path, sample_ids=sample_ids, features=matrix, present=available)
    return matrix, available


def aligned_skeleton(rows: list[dict[str, str]], path: Path, train: bool) -> np.ndarray:
    with np.load(path) as data:
        source_ids = data["sample_ids"].astype(str)
        source_values = np.asarray(data["features"], dtype=np.float32)
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
    wanted = [row["source_id"] if train else row["sample_id"] for row in rows]
    return np.stack([source_values[lookup[sample_id]] for sample_id in wanted])


def metadata_for_rows(rows: list[dict[str, str]], path: Path) -> list[dict[str, str]]:
    lookup = {row["sample_id"]: row for row in read_csv(path)}
    return [lookup[row["sample_id"]] for row in rows]


def temporal_blocks(indices: np.ndarray, metadata: list[dict[str, str]], gap_seconds: float) -> list[np.ndarray]:
    by_date: dict[str, list[int]] = defaultdict(list)
    missing = []
    for index in map(int, indices):
        row = metadata[index]
        if row.get("recording_date") and row.get("start_seconds"):
            by_date[row["recording_date"]].append(index)
        else:
            missing.append(np.asarray([index], dtype=np.int64))
    blocks = []
    for date_indices in by_date.values():
        ordered = sorted(date_indices, key=lambda index: float(metadata[index]["start_seconds"]))
        current = [ordered[0]]
        end = float(metadata[ordered[0]]["start_seconds"]) + float(metadata[ordered[0]].get("duration_seconds") or 0.0)
        for index in ordered[1:]:
            start = float(metadata[index]["start_seconds"])
            current_end = start + float(metadata[index].get("duration_seconds") or 0.0)
            if start - end > gap_seconds:
                blocks.append(np.asarray(current, dtype=np.int64))
                current = []
            current.append(index)
            end = max(end, current_end)
        blocks.append(np.asarray(current, dtype=np.int64))
    return blocks + missing


def assignment_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_values = sorted(set(truth.tolist()))
    prediction_values = sorted(set(map(int, prediction.tolist())))
    matrix = np.zeros((len(truth_values), len(prediction_values)), dtype=np.float64)
    for i, truth_value in enumerate(truth_values):
        for j, prediction_value in enumerate(prediction_values):
            matrix[i, j] = np.sum((truth == truth_value) & (prediction == prediction_value))
    row, column = linear_sum_assignment(-matrix)
    return float(matrix[row, column].sum() / len(truth))


def cluster_values(
    values: np.ndarray,
    indices: np.ndarray,
    blocks: list[np.ndarray],
    mode: str,
) -> np.ndarray:
    local_lookup = {int(index): position for position, index in enumerate(indices)}
    local_blocks = [
        np.asarray([local_lookup[int(index)] for index in block if int(index) in local_lookup], dtype=np.int64)
        for block in blocks
    ]
    local_blocks = [block for block in local_blocks if len(block)]
    estimator = KMeans(n_clusters=2, n_init=5, random_state=SEED)
    if mode == "trial":
        return estimator.fit_predict(values)
    if mode == "trial_majority":
        prediction = estimator.fit_predict(values)
        for block in local_blocks:
            winner = Counter(map(int, prediction[block])).most_common(1)[0][0]
            prediction[block] = winner
        return prediction
    if mode == "block_mean":
        block_values = np.stack([np.median(values[block], axis=0) for block in local_blocks])
        block_sizes = np.asarray([len(block) for block in local_blocks], dtype=np.float64)
        estimator.fit(block_values, sample_weight=block_sizes)
        prediction = np.zeros(len(indices), dtype=np.int64)
        for block, label in zip(local_blocks, estimator.labels_):
            prediction[block] = int(label)
        return prediction
    raise ValueError(mode)


def projected_features(train_appearance: np.ndarray, test_appearance: np.ndarray, train_skeleton: np.ndarray, test_skeleton: np.ndarray):
    appearance_scaler = StandardScaler().fit(train_appearance)
    train_scaled = appearance_scaler.transform(train_appearance)
    test_scaled = appearance_scaler.transform(test_appearance)
    pca = PCA(n_components=32, whiten=True, svd_solver="randomized", random_state=SEED)
    train_appearance_pca = pca.fit_transform(train_scaled)
    test_appearance_pca = pca.transform(test_scaled)
    skeleton_scaler = StandardScaler().fit(train_skeleton)
    return {
        "appearance": (train_appearance_pca, test_appearance_pca),
        "skeleton": (skeleton_scaler.transform(train_skeleton), skeleton_scaler.transform(test_skeleton)),
        "combined": (
            np.concatenate((train_appearance_pca, 0.5 * skeleton_scaler.transform(train_skeleton)), axis=1),
            np.concatenate((test_appearance_pca, 0.5 * skeleton_scaler.transform(test_skeleton)), axis=1),
        ),
    }


def configurations() -> list[dict[str, object]]:
    result = []
    for feature in ("appearance", "skeleton", "combined"):
        for dimensions in ((8, 16, 32) if feature == "appearance" else (0,)):
            for normalization in ("standard", "l2"):
                # Gap is immaterial for unsmoothed trial clustering; evaluate it once.
                result.append(
                    {"feature": feature, "dimensions": dimensions, "normalization": normalization, "gap_seconds": 300.0, "mode": "trial"}
                )
                for mode in ("trial_majority", "block_mean"):
                    result.append(
                        {"feature": feature, "dimensions": dimensions, "normalization": normalization, "gap_seconds": 300.0, "mode": mode}
                    )
    return result


def prepare_local(values: np.ndarray, configuration: dict[str, object]) -> np.ndarray:
    dimensions = int(configuration["dimensions"])
    if dimensions:
        values = values[:, :dimensions]
    if configuration["normalization"] == "l2":
        return normalize(values)
    return StandardScaler().fit_transform(values)


def evaluate_configuration(configuration, feature_values, users, metadata):
    records = []
    for cohort, cohort_users in TRAIN_COHORTS.items():
        for left, right in itertools.combinations(cohort_users, 2):
            indices = np.flatnonzero(np.isin(users, (left, right)))
            truth = users[indices]
            values = prepare_local(feature_values[indices], configuration)
            blocks = temporal_blocks(indices, metadata, float(configuration["gap_seconds"]))
            prediction = cluster_values(values, indices, blocks, str(configuration["mode"]))
            records.append(
                {
                    "cohort": cohort,
                    "pair": f"{left}/{right}",
                    "rows": int(len(indices)),
                    "blocks": int(len(blocks)),
                    "accuracy": assignment_accuracy(truth, prediction),
                }
            )
    scores = np.asarray([record["accuracy"] for record in records])
    return {
        "configuration": configuration,
        "mean_pair_accuracy": float(scores.mean()),
        "median_pair_accuracy": float(np.median(scores)),
        "p10_pair_accuracy": float(np.percentile(scores, 10)),
        "minimum_pair_accuracy": float(scores.min()),
        "pairs_above_80": int(np.sum(scores >= 0.80)),
        "pairs_above_90": int(np.sum(scores >= 0.90)),
        "pair_results": records,
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_rows = read_csv(TRAIN_ROWS)
    test_rows = read_csv(TEST_ROWS)
    train_appearance, train_present = build_appearance(
        train_rows, TRAIN_MANIFEST, TRAIN_VISUAL, output / "train_ir_actor_appearance.npz", args.frames_per_trial, args.rebuild_features
    )
    test_appearance, test_present = build_appearance(
        test_rows, TEST_MANIFEST, TEST_VISUAL, output / "test_ir_actor_appearance.npz", args.frames_per_trial, args.rebuild_features
    )
    train_skeleton = aligned_skeleton(train_rows, SKELETON_CACHE / "train_skeleton_identity.npz", True)
    test_skeleton = aligned_skeleton(test_rows, SKELETON_CACHE / "test_skeleton_identity.npz", False)
    feature_sets = projected_features(train_appearance, test_appearance, train_skeleton, test_skeleton)
    users = np.asarray([row["user_id"] for row in train_rows]).astype(str)
    train_metadata = metadata_for_rows(train_rows, TRAIN_METADATA)
    test_metadata = metadata_for_rows(test_rows, TEST_METADATA)

    evaluations = []
    configs = configurations()
    for index, configuration in enumerate(configs, start=1):
        train_values = feature_sets[str(configuration["feature"])][0]
        evaluations.append(evaluate_configuration(configuration, train_values, users, train_metadata))
        if index % 3 == 0 or index == len(configs):
            print(json.dumps({"evaluated_configurations": index, "total": len(configs)}), flush=True)
    evaluations.sort(
        key=lambda item: (
            item["p10_pair_accuracy"], item["mean_pair_accuracy"], item["minimum_pair_accuracy"]
        ),
        reverse=True,
    )
    selected = evaluations[0]

    configuration = selected["configuration"]
    test_values = feature_sets[str(configuration["feature"])][1]
    test_dates = np.asarray([row.get("recording_date", "") for row in test_metadata]).astype(str)
    waves = {
        "users10_11_wave": np.flatnonzero((test_dates >= "2025-05-31") & (test_dates <= "2025-06-02")),
        "users25_26_wave": np.flatnonzero((test_dates >= "2025-06-12") & (test_dates <= "2025-06-17")),
    }
    cluster_ids = np.full(len(test_rows), -1, dtype=np.int64)
    wave_summary = {}
    for wave_index, (wave, indices) in enumerate(waves.items()):
        values = prepare_local(test_values[indices], configuration)
        blocks = temporal_blocks(indices, test_metadata, float(configuration["gap_seconds"]))
        prediction = cluster_values(values, indices, blocks, str(configuration["mode"]))
        cluster_ids[indices] = wave_index * 2 + prediction
        wave_summary[wave] = {
            "rows": int(len(indices)),
            "blocks": int(len(blocks)),
            "cluster_sizes": {str(value): int(np.sum(prediction == value)) for value in sorted(set(prediction.tolist()))},
        }
    # The one row lacking timestamp remains explicitly unassigned; it must not be
    # silently attached to a subject by sample-id order.
    with (output / "test_anonymous_subject_clusters.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "recording_date", "cluster_id"))
        writer.writeheader()
        for row, metadata_row, cluster_id in zip(test_rows, test_metadata, cluster_ids):
            writer.writerow({"sample_id": row["sample_id"], "recording_date": metadata_row.get("recording_date", ""), "cluster_id": int(cluster_id)})

    summary = {
        "stage": "P89_ir_actor_identity_clustering_v1",
        "protocol": (
            "No Test labels or leaderboard feedback. Visual target boxes are fixed before identity clustering. "
            "Configuration is selected over every same-wave pair of known Train users, then frozen for the two official Public-Test user waves."
        ),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_appearance_present": int(train_present.sum()),
        "test_appearance_present": int(test_present.sum()),
        "selected": selected,
        "test_waves": wave_summary,
        "top_configurations": evaluations[:20],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("selected", "top_configurations")}, indent=2), flush=True)
    print(json.dumps({"selected": {key: value for key, value in selected.items() if key != "pair_results"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
