from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from audit_yolo11_pose_skeleton import frame_map
from p31_skeleton_imu_preprocessing import build_skeleton_features, safe_trial_path
from p46_event_preprocessing import rotate_skeleton_cache
from p89_skeleton_active_actor_expert import normalized_pose, slot_diagnostics
from p89_skeleton_identity_tracking_expert import (
    FOLDS,
    TEST_MANIFEST,
    TEST_PIXEL,
    TEST_ROWS,
    TEST_SOURCE,
    TRAIN_MANIFEST,
    TRAIN_PIXEL,
    TRAIN_ROWS,
    TRAIN_SOURCE,
    load_people,
    metrics,
    model,
    summarize_window,
)


PROJECT_DIR = Path(__file__).resolve().parent
BASELINE = PROJECT_DIR / "runs/p89_skeleton_invariant_expert_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p89_skeleton_imu_actor_matching_v1"
H1_USERS = ["user6", "user8", "user17", "user23"]
H2_USERS = ["user5", "user7", "user16", "user18", "user19"]

# These are label-free gates.  The development split selects one on H1 and the
# held-out H2 users are reported only after that selection is frozen.
STRATEGIES = {
    "imu_match_loose": {
        "minimum_coverage": 0.70,
        "maximum_identity_mad": 0.18,
        "minimum_score": 0.05,
        "minimum_margin": 0.05,
    },
    "imu_match_guarded": {
        "minimum_coverage": 0.75,
        "maximum_identity_mad": 0.12,
        "minimum_score": 0.10,
        "minimum_margin": 0.10,
    },
    "imu_match_tight": {
        "minimum_coverage": 0.80,
        "maximum_identity_mad": 0.08,
        "minimum_score": 0.15,
        "minimum_margin": 0.15,
    },
}

# WTC, WTLA, WTRA, WTLL, WTRL mapped to H36M joints.
BODY_GROUPS = (
    (7, 8, 9, 10),
    (11, 12, 13),
    (14, 15, 16),
    (4, 5, 6),
    (1, 2, 3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match the Skeleton target actor to the wearable IMU motion trace."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def interpolate(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    valid = np.isfinite(result)
    if valid.sum() < 2:
        return np.full_like(result, np.nan)
    index = np.arange(len(result))
    result[~valid] = np.interp(index[~valid], index[valid], result[valid])
    return result


def imu_motion(source: np.lib.npyio.NpzFile, frames: int) -> np.ndarray:
    values = np.asarray(source["imu_values"], dtype=np.float64)
    frame_index = np.asarray(source["imu_frame_index"], dtype=np.int64)
    offsets = np.asarray(source["imu_device_offsets"], dtype=np.int64)
    result = np.full((frames, 5), np.nan, dtype=np.float64)
    for device in range(min(5, len(offsets) - 1)):
        start, stop = int(offsets[device]), int(offsets[device + 1])
        if stop <= start:
            continue
        local_values = values[start:stop]
        local_frames = frame_index[start:stop]
        # Gyroscope magnitude is orientation-invariant and most directly tracks
        # articulated angular motion.  Acceleration change supplies translation.
        gyro = np.linalg.norm(local_values[:, 3:6], axis=1)
        accel = local_values[:, :3]
        accel -= np.nanmedian(accel, axis=0, keepdims=True)
        strength = np.log1p(np.maximum(gyro, 0.0)) + 0.35 * np.log1p(
            np.linalg.norm(accel, axis=1)
        )
        for frame in np.unique(local_frames):
            if 0 <= frame < frames:
                result[frame, device] = float(np.nanmedian(strength[local_frames == frame]))
        result[:, device] = interpolate(result[:, device])
    return result


def skeleton_motion(people_by_frame: list[list[np.ndarray]], slot: int) -> np.ndarray:
    poses = []
    for people in people_by_frame:
        poses.append(normalized_pose(people[slot]) if slot < len(people) else None)
    result = np.full((len(poses), 5), np.nan, dtype=np.float64)
    for frame in range(1, len(poses)):
        if poses[frame - 1] is None or poses[frame] is None:
            continue
        delta = np.linalg.norm(poses[frame] - poses[frame - 1], axis=1)
        for device, joints in enumerate(BODY_GROUPS):
            result[frame, device] = float(np.nanmedian(delta[list(joints)]))
    for device in range(5):
        result[:, device] = interpolate(result[:, device])
    return result


def standardized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 12:
        return float("nan")
    x = np.log1p(np.maximum(left[valid], 0.0))
    y = np.maximum(right[valid], 0.0)
    x -= np.median(x)
    y -= np.median(y)
    sx = float(np.sqrt(np.mean(x * x)))
    sy = float(np.sqrt(np.mean(y * y)))
    if sx < 1e-6 or sy < 1e-6:
        return float("nan")
    return float(np.mean((x / sx) * (y / sy)))


def event_overlap(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 12:
        return float("nan")
    x = left[valid]
    y = right[valid]
    x_event = x >= np.quantile(x, 0.70)
    y_event = y >= np.quantile(y, 0.70)
    union = int(np.sum(x_event | y_event))
    return float(np.sum(x_event & y_event) / max(union, 1))


def matching_score(skeleton: np.ndarray, imu: np.ndarray) -> float:
    device_scores = []
    for device in range(5):
        best_correlation = -1.0
        best_overlap = 0.0
        for lag in (-1, 0, 1):
            if lag < 0:
                left, right = skeleton[-lag:, device], imu[:lag, device]
            elif lag > 0:
                left, right = skeleton[:-lag, device], imu[lag:, device]
            else:
                left, right = skeleton[:, device], imu[:, device]
            correlation = standardized_correlation(left, right)
            overlap = event_overlap(left, right)
            if np.isfinite(correlation) and correlation > best_correlation:
                best_correlation = correlation
                best_overlap = overlap if np.isfinite(overlap) else 0.0
        if best_correlation > -1.0:
            device_scores.append(best_correlation + 0.35 * best_overlap)
    return float(np.mean(device_scores)) if len(device_scores) >= 3 else float("-inf")


def build_alternate_feature(
    people_by_frame: list[list[np.ndarray]],
    slot: int,
    frame_times: np.ndarray,
    chosen: np.ndarray,
) -> np.ndarray:
    empty = np.full((17, 4), np.nan, dtype=np.float32)
    empty[:, 3] = 0.0
    raw = np.stack(
        [people[slot] if slot < len(people) else empty for people in people_by_frame]
    ).astype(np.float32)
    built = build_skeleton_features(raw, frame_times)
    rotated = rotate_skeleton_cache(
        built["features"],
        built["feature_mask"],
        built["joint_mask"],
        built["relations"],
        built["relation_mask"],
        frame_times,
    )
    quality = np.asarray(built["frame_quality"], dtype=np.float32)
    quality *= np.asarray([slot < len(people) for people in people_by_frame], dtype=np.float32)
    return summarize_window(
        rotated["features"][chosen].astype(np.float16).astype(np.float32),
        np.asarray(built["joint_mask"])[chosen],
        rotated["relations"][chosen].astype(np.float16).astype(np.float32),
        rotated["relation_mask"][chosen],
        quality[chosen].astype(np.float16).astype(np.float32),
    )


def build_split(
    *,
    rows_path: Path,
    pixel_cache: Path,
    source_cache: Path,
    manifest_path: Path,
    baseline_path: Path,
    output: Path,
    split: str,
    rebuild: bool,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, list[dict[str, str]]]:
    rows = read_csv(rows_path)
    baseline = np.load(baseline_path, mmap_mode="r")
    if len(rows) != len(baseline):
        raise RuntimeError(f"{split} baseline alignment changed")
    feature_path = output / f"{split}_best_alternate_features.npy"
    audit_path = output / f"{split}_matching_audit.csv"
    if not rebuild and feature_path.is_file() and audit_path.is_file():
        return rows, baseline, np.load(feature_path, mmap_mode="r"), read_csv(audit_path)

    output.mkdir(parents=True, exist_ok=True)
    alternate = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=baseline.shape
    )
    manifests = {row["sample_id"]: row for row in read_csv(manifest_path)}
    indices = np.asarray(np.load(pixel_cache / "source_frame_indices.npy", mmap_mode="r"))
    audit_rows: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        alternate[index] = baseline[index]
        source_id = row["source_id"]
        source_path = source_cache / "trial_motion_cache" / safe_trial_path(source_id).with_suffix(".npz")
        with np.load(source_path, allow_pickle=False) as source:
            frame_ids = np.asarray(source["frame_ids"]).astype(str)
            frame_times = np.asarray(source["frame_time_seconds"], dtype=np.float64)
            imu = imu_motion(source, len(frame_ids))
        skeleton_paths = frame_map(Path(manifests[source_id]["skeleton_path"]), "skeleton")
        people = [load_people(skeleton_paths[frame_id]) for frame_id in frame_ids]
        maximum_people = max((len(item) for item in people), default=0)
        diagnostics = [slot_diagnostics(people, slot) for slot in range(maximum_people)]
        scores = [matching_score(skeleton_motion(people, slot), imu) for slot in range(maximum_people)]
        best_alternate_slot = -1
        if maximum_people > 1:
            best_alternate_slot = 1 + int(np.argmax(scores[1:]))
            alternate[index] = build_alternate_feature(
                people, best_alternate_slot, frame_times, np.asarray(indices[index], dtype=np.int64)
            )
        audit_rows.append(
            {
                "sample_id": row["sample_id"],
                "source_id": source_id,
                "class_id": row["class_id"],
                "user_id": row["user_id"],
                "frames": str(len(frame_ids)),
                "multi_frames": str(sum(len(item) > 1 for item in people)),
                "best_alternate_slot": str(best_alternate_slot),
                "slot0_score": str(scores[0] if scores else float("-inf")),
                "alternate_score": str(scores[best_alternate_slot] if best_alternate_slot > 0 else float("-inf")),
                "alternate_coverage": str(diagnostics[best_alternate_slot]["coverage"] if best_alternate_slot > 0 else 0.0),
                "alternate_identity_mad": str(diagnostics[best_alternate_slot]["identity_mad"] if best_alternate_slot > 0 else float("inf")),
                "all_scores": json.dumps(scores),
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            alternate.flush()
            print(json.dumps({"split": split, "processed": index + 1, "total": len(rows), "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    return rows, baseline, alternate, audit_rows


def selected_mask(audit: list[dict[str, str]], parameters: dict[str, float]) -> np.ndarray:
    result = []
    for row in audit:
        score = float(row["alternate_score"])
        baseline = float(row["slot0_score"])
        result.append(
            int(row["best_alternate_slot"]) > 0
            and float(row["alternate_coverage"]) >= parameters["minimum_coverage"]
            and float(row["alternate_identity_mad"]) <= parameters["maximum_identity_mad"]
            and score >= parameters["minimum_score"]
            and score - baseline >= parameters["minimum_margin"]
        )
    return np.asarray(result, dtype=bool)


def subset_report(labels: np.ndarray, prediction: np.ndarray, baseline: np.ndarray, users: np.ndarray, selected: np.ndarray, wanted_users: list[str]) -> dict[str, object]:
    held = np.isin(users, wanted_users)
    return {
        "metrics": metrics(labels[held], prediction[held]),
        "baseline_correct": int(np.sum(labels[held] == baseline[held])),
        "net": int(np.sum(labels[held] == prediction[held]) - np.sum(labels[held] == baseline[held])),
        "selected_trials": int(np.sum(selected & held)),
        "per_user_net": {
            user: int(np.sum(labels[users == user] == prediction[users == user]) - np.sum(labels[users == user] == baseline[users == user]))
            for user in wanted_users
        },
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    train_rows, train_base, train_alt, train_audit = build_split(
        rows_path=TRAIN_ROWS,
        pixel_cache=TRAIN_PIXEL,
        source_cache=TRAIN_SOURCE,
        manifest_path=TRAIN_MANIFEST,
        baseline_path=BASELINE / "train_features.npy",
        output=output,
        split="train",
        rebuild=args.rebuild_features,
    )
    test_rows, test_base, test_alt, test_audit = build_split(
        rows_path=TEST_ROWS,
        pixel_cache=TEST_PIXEL,
        source_cache=TEST_SOURCE,
        manifest_path=TEST_MANIFEST,
        baseline_path=BASELINE / "test_features.npy",
        output=output,
        split="test",
        rebuild=args.rebuild_features,
    )
    labels = np.asarray([int(row["class_id"]) for row in train_rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in train_rows]).astype(str)
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    base_oof = np.zeros((len(labels), 40), dtype=np.float64)
    alt_oof = np.zeros_like(base_oof)
    for fold in folds:
        fit = np.isin(users, fold["train_users"])
        held = np.isin(users, fold["val_users"])
        estimator = model()
        estimator.fit(train_base[fit], labels[fit])
        target = np.flatnonzero(held)
        for features, destination in ((train_base, base_oof), (train_alt, alt_oof)):
            partial = estimator.predict_proba(features[held])
            destination[np.ix_(target, estimator.classes_.astype(np.int64))] = partial
        print({"fold": fold["fold"], **metrics(labels[held], base_oof[target].argmax(1))}, flush=True)

    base_prediction = base_oof.argmax(1)
    candidates = {}
    for name, parameters in STRATEGIES.items():
        chosen = selected_mask(train_audit, parameters)
        probability = base_oof.copy()
        probability[chosen] = alt_oof[chosen]
        prediction = probability.argmax(1)
        candidates[name] = {
            "parameters": parameters,
            "overall": metrics(labels, prediction),
            "overall_net": int(np.sum(labels == prediction) - np.sum(labels == base_prediction)),
            "H1": subset_report(labels, prediction, base_prediction, users, chosen, H1_USERS),
            "H2": subset_report(labels, prediction, base_prediction, users, chosen, H2_USERS),
        }
    # Select using H1 only: maximize net, then minimum per-user gain, then fewer changes.
    selected_name = max(
        STRATEGIES,
        key=lambda name: (
            candidates[name]["H1"]["net"],
            min(candidates[name]["H1"]["per_user_net"].values()),
            -candidates[name]["H1"]["selected_trials"],
        ),
    )
    train_chosen = selected_mask(train_audit, STRATEGIES[selected_name])
    test_chosen = selected_mask(test_audit, STRATEGIES[selected_name])
    selected_oof = base_oof.copy()
    selected_oof[train_chosen] = alt_oof[train_chosen]

    final = model()
    final.fit(train_base, labels)
    test_base_probability = np.zeros((len(test_rows), 40), dtype=np.float64)
    test_alt_probability = np.zeros_like(test_base_probability)
    for features, destination in ((test_base, test_base_probability), (test_alt, test_alt_probability)):
        partial = final.predict_proba(features)
        destination[:, final.classes_.astype(np.int64)] = partial
    selected_test = test_base_probability.copy()
    selected_test[test_chosen] = test_alt_probability[test_chosen]
    np.savez_compressed(
        output / "oof_logits.npz",
        sample_ids=np.asarray([row["sample_id"] for row in train_rows]),
        labels=labels,
        skeleton_logits=np.log(np.maximum(selected_oof, 1e-12)),
        selected_actor=train_chosen,
    )
    np.savez_compressed(
        output / "test_logits.npz",
        sample_ids=np.asarray([row["sample_id"] for row in test_rows]),
        skeleton_logits=np.log(np.maximum(selected_test, 1e-12)),
        selected_actor=test_chosen,
    )
    report = {
        "stage": "P89_IMU_synchronized_multi_person_actor_matching_v1",
        "protocol": "The target actor is selected without labels by synchronizing per-limb Skeleton motion magnitude to wearable IMU motion. Gates are selected on H1 subject-disjoint users and transferred unchanged to H2 and Test.",
        "baseline": metrics(labels, base_prediction),
        "candidates": candidates,
        "selected_on_H1": selected_name,
        "H2_confirmation": candidates[selected_name]["H2"],
        "train_selected_trials": int(train_chosen.sum()),
        "test_selected_trials": int(test_chosen.sum()),
        "test_prediction_changes": int(np.sum(selected_test.argmax(1) != test_base_probability.argmax(1))),
        "test_selected_ids": [test_rows[index]["sample_id"] for index in np.flatnonzero(test_chosen)],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
