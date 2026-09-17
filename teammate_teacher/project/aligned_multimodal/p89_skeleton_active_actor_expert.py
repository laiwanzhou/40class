from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from audit_yolo11_pose_skeleton import frame_map
from p31_skeleton_imu_preprocessing import build_skeleton_features, safe_trial_path
from p46_event_preprocessing import rotate_skeleton_cache
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
    bone_signature,
    load_people,
    metrics,
    model,
    summarize_window,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p89_skeleton_active_actor_v1"
STRATEGIES: dict[str, dict[str, float]] = {
    "active_guarded_2x": {
        "minimum_coverage": 0.75,
        "maximum_identity_mad": 0.08,
        "motion_ratio": 2.0,
        "minimum_motion": 0.04,
    },
    "active_guarded_3x": {
        "minimum_coverage": 0.75,
        "maximum_identity_mad": 0.08,
        "motion_ratio": 3.0,
        "minimum_motion": 0.04,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a persistent active Skeleton actor in multi-person trials and "
            "evaluate only with fixed subject-disjoint folds."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_pose(person: np.ndarray) -> np.ndarray | None:
    signature = bone_signature(person)
    if signature is None:
        return None
    xyz = np.asarray(person[:, :3], dtype=np.float64)
    scale = float(np.median(np.exp(signature)))
    if not np.isfinite(scale) or scale < 1e-5:
        return None
    # The supplied 3-D skeletons are root-centred.  This normalization removes
    # body size while preserving articulation and its temporal change.
    return ((xyz - xyz[[0]]) / scale).astype(np.float32)


def slot_diagnostics(
    people_by_frame: list[list[np.ndarray]], slot: int
) -> dict[str, float]:
    poses: list[np.ndarray | None] = []
    signatures = []
    for people in people_by_frame:
        if slot >= len(people):
            poses.append(None)
            continue
        pose = normalized_pose(people[slot])
        signature = bone_signature(people[slot])
        poses.append(pose)
        if signature is not None:
            signatures.append(signature)
    coverage = float(sum(pose is not None for pose in poses) / len(poses))
    motions = [
        float(np.median(np.linalg.norm(right - left, axis=1)))
        for left, right in zip(poses[:-1], poses[1:])
        if left is not None and right is not None
    ]
    motion = float(np.quantile(motions, 0.75)) if motions else 0.0
    if signatures:
        stacked = np.stack(signatures)
        reference = np.median(stacked, axis=0)
        identity_mad = float(np.median(np.abs(stacked - reference)))
    else:
        identity_mad = float("inf")
    return {
        "coverage": coverage,
        "motion": motion,
        "identity_mad": identity_mad,
    }


def select_active_sequence(
    people_by_frame: list[list[np.ndarray]], parameters: dict[str, float]
) -> tuple[np.ndarray, dict[str, Any]]:
    empty = np.full((17, 4), np.nan, dtype=np.float32)
    empty[:, 3] = 0.0
    maximum_people = max((len(people) for people in people_by_frame), default=0)
    if maximum_people == 0:
        return np.stack([empty for _ in people_by_frame]), {
            "selected_slot": -1,
            "changed_frames": 0,
            "slot_diagnostics": [],
        }
    diagnostics = [
        slot_diagnostics(people_by_frame, slot) for slot in range(maximum_people)
    ]
    selected_slot = 0
    baseline_motion = max(diagnostics[0]["motion"], 1e-6)
    eligible = []
    for slot in range(1, maximum_people):
        item = diagnostics[slot]
        if (
            item["coverage"] >= parameters["minimum_coverage"]
            and item["identity_mad"] <= parameters["maximum_identity_mad"]
            and item["motion"] >= parameters["minimum_motion"]
            and item["motion"] >= parameters["motion_ratio"] * baseline_motion
        ):
            eligible.append(slot)
    if eligible:
        selected_slot = max(eligible, key=lambda slot: diagnostics[slot]["motion"])
    selected = [
        people[selected_slot] if selected_slot < len(people) else empty
        for people in people_by_frame
    ]
    changed_frames = sum(
        selected_slot > 0 and selected_slot < len(people) for people in people_by_frame
    )
    return np.stack(selected).astype(np.float32), {
        "selected_slot": selected_slot,
        "changed_frames": int(changed_frames),
        "slot_diagnostics": diagnostics,
    }


def build_split_features(
    *,
    rows_path: Path,
    pixel_cache: Path,
    source_cache: Path,
    manifest_path: Path,
    output: Path,
    split: str,
    rebuild: bool,
) -> tuple[list[dict[str, str]], dict[str, np.ndarray], list[dict[str, str]]]:
    rows = read_csv(rows_path)
    manifests = {row["sample_id"]: row for row in read_csv(manifest_path)}
    indices = np.asarray(np.load(pixel_cache / "source_frame_indices.npy", mmap_mode="r"))
    feature_paths = {name: output / f"{split}_{name}_features.npy" for name in STRATEGIES}
    audit_path = output / f"{split}_active_actor_audit.csv"
    if (
        not rebuild
        and all(path.is_file() for path in feature_paths.values())
        and audit_path.is_file()
    ):
        return (
            rows,
            {name: np.load(path, mmap_mode="r") for name, path in feature_paths.items()},
            read_csv(audit_path),
        )

    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(rows), 9103)
        )
        for name, path in feature_paths.items()
    }
    audit_rows: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        source_id = row["source_id"]
        manifest = manifests[source_id]
        skeleton_paths = frame_map(Path(manifest["skeleton_path"]), "skeleton")
        source_path = (
            source_cache
            / "trial_motion_cache"
            / safe_trial_path(source_id).with_suffix(".npz")
        )
        with np.load(source_path, allow_pickle=False) as source:
            frame_ids = np.asarray(source["frame_ids"]).astype(str)
            frame_times = np.asarray(source["frame_time_seconds"], dtype=np.float64)
        people_by_frame = [load_people(skeleton_paths[frame_id]) for frame_id in frame_ids]
        chosen = np.asarray(indices[index], dtype=np.int64)
        audit: dict[str, str] = {
            "sample_id": row["sample_id"],
            "source_id": source_id,
            "class_id": row["class_id"],
            "user_id": row["user_id"],
            "frames": str(len(frame_ids)),
            "multi_frames": str(sum(len(people) > 1 for people in people_by_frame)),
        }
        for name, parameters in STRATEGIES.items():
            raw, selection = select_active_sequence(people_by_frame, parameters)
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
            quality *= np.asarray(
                [
                    selection["selected_slot"] >= 0
                    and selection["selected_slot"] < len(people)
                    for people in people_by_frame
                ],
                dtype=np.float32,
            )
            arrays[name][index] = summarize_window(
                rotated["features"][chosen].astype(np.float16).astype(np.float32),
                np.asarray(built["joint_mask"])[chosen],
                rotated["relations"][chosen].astype(np.float16).astype(np.float32),
                rotated["relation_mask"][chosen],
                quality[chosen].astype(np.float16).astype(np.float32),
            )
            audit[f"{name}_selected_slot"] = str(selection["selected_slot"])
            audit[f"{name}_changed_frames"] = str(selection["changed_frames"])
            audit[f"{name}_diagnostics"] = json.dumps(selection["slot_diagnostics"])
        audit_rows.append(audit)
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            for array in arrays.values():
                array.flush()
            print(
                json.dumps(
                    {
                        "split": split,
                        "processed": index + 1,
                        "total": len(rows),
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    return rows, arrays, audit_rows


def stratum_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    multi_frames: np.ndarray,
    total_frames: np.ndarray,
) -> dict[str, object]:
    strata = {
        "single": multi_frames == 0,
        "ever_multi": multi_frames > 0,
        "majority_multi": multi_frames > 0.5 * total_frames,
    }
    return {
        name: metrics(labels[rows], prediction[rows])
        for name, rows in strata.items()
        if np.any(rows)
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_rows, train_features, train_audit = build_split_features(
        rows_path=TRAIN_ROWS,
        pixel_cache=TRAIN_PIXEL,
        source_cache=TRAIN_SOURCE,
        manifest_path=TRAIN_MANIFEST,
        output=output,
        split="train",
        rebuild=args.rebuild_features,
    )
    test_rows, test_features, test_audit = build_split_features(
        rows_path=TEST_ROWS,
        pixel_cache=TEST_PIXEL,
        source_cache=TEST_SOURCE,
        manifest_path=TEST_MANIFEST,
        output=output,
        split="test",
        rebuild=args.rebuild_features,
    )
    labels = np.asarray([int(row["class_id"]) for row in train_rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in train_rows]).astype(str)
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    multi_frames = np.asarray([int(row["multi_frames"]) for row in train_audit])
    total_frames = np.asarray([int(row["frames"]) for row in train_audit])
    results = {}
    probabilities = {}
    test_probabilities = {}
    for name in STRATEGIES:
        oof = np.zeros((len(labels), 40), dtype=np.float64)
        per_fold = []
        for fold in folds:
            fit = np.isin(users, fold["train_users"])
            held = np.isin(users, fold["val_users"])
            estimator = model()
            estimator.fit(train_features[name][fit], labels[fit])
            partial = estimator.predict_proba(train_features[name][held])
            target = np.flatnonzero(held)
            oof[np.ix_(target, estimator.classes_.astype(np.int64))] = partial
            per_fold.append(
                {"fold": int(fold["fold"]), **metrics(labels[held], oof[target].argmax(1))}
            )
        final = model()
        final.fit(train_features[name], labels)
        partial = final.predict_proba(test_features[name])
        test_probability = np.zeros((len(test_rows), 40), dtype=np.float64)
        test_probability[:, final.classes_.astype(np.int64)] = partial
        probabilities[name] = oof
        test_probabilities[name] = test_probability
        prediction = oof.argmax(1)
        results[name] = {
            "overall": metrics(labels, prediction),
            "folds": per_fold,
            "strata": stratum_metrics(labels, prediction, multi_frames, total_frames),
            "train_selected_alternate_trials": int(
                sum(int(row[f"{name}_selected_slot"]) > 0 for row in train_audit)
            ),
            "test_selected_alternate_trials": int(
                sum(int(row[f"{name}_selected_slot"]) > 0 for row in test_audit)
            ),
        }
        print(name, results[name], flush=True)

    selected = max(
        STRATEGIES,
        key=lambda name: (
            results[name]["overall"]["correct"],
            results[name]["strata"]["ever_multi"]["correct"],
        ),
    )
    np.savez_compressed(
        output / "oof_logits.npz",
        sample_ids=np.asarray([row["sample_id"] for row in train_rows]),
        labels=labels,
        skeleton_logits=np.log(np.maximum(probabilities[selected], 1e-12)),
    )
    np.savez_compressed(
        output / "test_logits.npz",
        sample_ids=np.asarray([row["sample_id"] for row in test_rows]),
        skeleton_logits=np.log(np.maximum(test_probabilities[selected], 1e-12)),
    )
    summary = {
        "stage": "P89_active_actor_multi_person_Skeleton_v1",
        "protocol": (
            "The actor rule is label-free: choose a non-first slot only when it is "
            "persistent, bone-identity stable, and at least 2x/3x more articulated. "
            "All accuracy comparisons use fixed subject-disjoint folds."
        ),
        "strategies": STRATEGIES,
        "results": results,
        "selected": selected,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
