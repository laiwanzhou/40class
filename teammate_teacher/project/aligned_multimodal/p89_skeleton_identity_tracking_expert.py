from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from audit_yolo11_pose_skeleton import frame_map
from p31_skeleton_imu_preprocessing import (
    H36M_PARENTS,
    build_skeleton_features,
    safe_trial_path,
)
from p46_event_preprocessing import rotate_skeleton_cache
from p89_skeleton_invariant_expert import (
    bilateral_signals,
    masked_fill,
    phase_statistics,
    robust_statistics,
)


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_ROWS = PROJECT_DIR / "runs/p86_motion_window_cache_t16_v1/rows.csv"
TEST_ROWS = PROJECT_DIR / "runs/p87s_test_motion_window_t16_v1/rows.csv"
TRAIN_PIXEL = PROJECT_DIR / "runs/p86_visual_pixel_cache_t16_r160_v12"
TEST_PIXEL = PROJECT_DIR / "runs/p87s_test_pixel_cache_t16_r160_v1"
TRAIN_SOURCE = PROJECT_DIR / "runs/p31_skeleton_imu_full"
TEST_SOURCE = PROJECT_DIR / "runs/p87s_test_motion_source_v1"
TRAIN_MANIFEST = PROJECT_DIR / "data/six_modality_audit/train_union_manifest.csv"
TEST_MANIFEST = PROJECT_DIR / "data/p46_test_union_manifest.csv"
FOLDS = PROJECT_DIR / "data/subject_folds/folds_summary.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p89_skeleton_identity_tracking_v1"
SEED = 20260816


STRATEGIES: dict[str, dict[str, float]] = {
    # The reference is the temporal median of people[0]'s action-invariant bone
    # lengths.  A non-first candidate is accepted only for a clear people[0]
    # identity outlier.  These two settings intentionally change very few frames.
    "identity_tight": {"mad_multiplier": 8.0, "ratio": 0.40, "margin": 0.015},
    "identity_medium": {"mad_multiplier": 5.0, "ratio": 0.50, "margin": 0.015},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "P89 multi-person Skeleton audit using bone-length identity signatures. "
            "All model selection uses fixed subject-disjoint OOF folds."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_person(person: object) -> np.ndarray | None:
    if not isinstance(person, dict):
        return None
    keypoints = np.asarray(person.get("keypoints", []), dtype=np.float32)
    scores = np.asarray(person.get("keypoint_scores", []), dtype=np.float32)
    if keypoints.shape != (17, 3) or scores.shape != (17,):
        return None
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((keypoints, scores[:, None]), axis=1).astype(np.float32)


def load_people(path: Path) -> list[np.ndarray]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        payload = []
    if not isinstance(payload, list):
        return []
    result = [load_person(person) for person in payload]
    return [person for person in result if person is not None]


def bone_signature(person: np.ndarray) -> np.ndarray | None:
    xyz = np.asarray(person[:, :3], dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=1) & (person[:, 3] > 0.0)
    bone_valid = valid & valid[H36M_PARENTS]
    lengths = np.linalg.norm(xyz - xyz[H36M_PARENTS], axis=1)[1:]
    if not np.all(bone_valid[1:]) or np.any(lengths < 1e-5):
        return None
    # Log bone lengths preserve body scale and proportions while removing pose
    # angles and camera translation, which are what defeated nearest-pose tracking.
    return np.log(lengths).astype(np.float32)


def select_identity_sequence(
    people_by_frame: list[list[np.ndarray]],
    parameters: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    empty = np.full((17, 4), np.nan, dtype=np.float32)
    empty[:, 3] = 0.0
    first_signatures = []
    for people in people_by_frame:
        if people:
            signature = bone_signature(people[0])
            if signature is not None:
                first_signatures.append(signature)
    if not first_signatures:
        return np.stack([people[0] if people else empty for people in people_by_frame]), {
            "changed_frames": 0,
            "eligible_frames": 0,
            "reference_available": 0,
        }

    reference = np.median(np.stack(first_signatures), axis=0)
    first_distances = []
    for people in people_by_frame:
        signature = bone_signature(people[0]) if people else None
        first_distances.append(
            float(np.median(np.abs(signature - reference)))
            if signature is not None
            else np.nan
        )
    first_distances_array = np.asarray(first_distances, dtype=np.float64)
    center = float(np.nanmedian(first_distances_array))
    mad = float(np.nanmedian(np.abs(first_distances_array - center)))
    robust_scale = max(1.4826 * mad, 0.005)
    cutoff = center + parameters["mad_multiplier"] * robust_scale

    selected = []
    selected_slots = []
    eligible = 0
    for people, first_distance in zip(people_by_frame, first_distances_array):
        if not people:
            selected.append(empty)
            selected_slots.append(-1)
            continue
        best_slot = 0
        if len(people) > 1 and np.isfinite(first_distance):
            eligible += 1
            alternate_distances = []
            for person in people[1:]:
                signature = bone_signature(person)
                alternate_distances.append(
                    float(np.median(np.abs(signature - reference)))
                    if signature is not None
                    else np.inf
                )
            alternate_slot = 1 + int(np.argmin(alternate_distances))
            alternate_distance = alternate_distances[alternate_slot - 1]
            if (
                first_distance > cutoff
                and alternate_distance < parameters["ratio"] * first_distance
                and alternate_distance + parameters["margin"] < first_distance
            ):
                best_slot = alternate_slot
        selected.append(people[best_slot])
        selected_slots.append(best_slot)
    slots = np.asarray(selected_slots, dtype=np.int16)
    return np.stack(selected).astype(np.float32), {
        "changed_frames": int(np.sum(slots > 0)),
        "eligible_frames": int(eligible),
        "reference_available": 1,
        "distance_center": center,
        "distance_mad": mad,
        "cutoff": cutoff,
    }


def summarize_window(
    skeleton: np.ndarray,
    joint_mask: np.ndarray,
    relations: np.ndarray,
    relation_mask: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    # Match p89_skeleton_invariant_expert.py exactly, but for one trial.
    skeleton = skeleton.reshape(1, 32, 17, 13)
    joint_mask = joint_mask.reshape(1, 32, 17).astype(bool)
    relations = relations.reshape(1, 32, 18)
    relation_mask = relation_mask.reshape(1, 32, 18).astype(bool)
    quality = quality.reshape(1, 32, 1)
    vector = masked_fill(
        skeleton[..., :12].reshape(1, 32, -1),
        np.repeat(joint_mask[..., None], 12, axis=3).reshape(1, 32, -1),
    )
    stream = skeleton[..., :12].reshape(1, 32, 17, 4, 3)
    norms = np.linalg.norm(stream, axis=-1)
    norm_mask = np.repeat(joint_mask[..., None], 4, axis=3)
    norms = masked_fill(norms.reshape(1, 32, -1), norm_mask.reshape(1, 32, -1))
    bone_norm = np.linalg.norm(skeleton[..., 3:6], axis=-1)
    bilateral = masked_fill(
        bilateral_signals(bone_norm), np.ones((1, 32, 12), dtype=bool)
    )
    invariant = np.concatenate(
        (norms, masked_fill(relations, relation_mask), bilateral, quality), axis=2
    )
    blocks = [
        robust_statistics(vector),
        robust_statistics(invariant),
        phase_statistics(invariant, 8),
        robust_statistics(invariant[:, :16]),
        robust_statistics(invariant[:, 16:]),
    ]
    coverage = np.concatenate(
        (
            joint_mask.mean(axis=(1, 2))[:, None],
            relation_mask.mean(axis=(1, 2))[:, None],
            quality.mean(axis=1),
            quality.std(axis=1),
        ),
        axis=1,
    )
    blocks.append(coverage.astype(np.float32))
    return np.nan_to_num(
        np.concatenate(blocks, axis=1)[0], nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)


def build_split_features(
    *,
    rows_path: Path,
    pixel_cache: Path,
    source_cache: Path,
    manifest_path: Path,
    output: Path,
    split: str,
    rebuild: bool,
) -> tuple[list[dict[str, str]], dict[str, np.ndarray], dict[str, Any]]:
    rows = read_csv(rows_path)
    manifests = {row["sample_id"]: row for row in read_csv(manifest_path)}
    indices = np.asarray(np.load(pixel_cache / "source_frame_indices.npy", mmap_mode="r"))
    if len(indices) != len(rows):
        raise RuntimeError(f"{split} visual index rows changed")
    feature_paths = {
        name: output / f"{split}_{name}_features.npy" for name in STRATEGIES
    }
    audit_path = output / f"{split}_tracking_audit.csv"
    if not rebuild and all(path.is_file() for path in feature_paths.values()) and audit_path.is_file():
        features = {name: np.load(path, mmap_mode="r") for name, path in feature_paths.items()}
        audit_rows = read_csv(audit_path)
        return rows, features, {
            name: {
                "changed_frames": int(sum(int(row[f"{name}_changed_frames"]) for row in audit_rows)),
                "changed_trials": int(sum(int(row[f"{name}_changed_frames"]) > 0 for row in audit_rows)),
            }
            for name in STRATEGIES
        }

    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(rows), 9103)
        )
        for name, path in feature_paths.items()
    }
    audit_rows = []
    totals = {name: {"changed_frames": 0, "changed_trials": 0} for name in STRATEGIES}
    started = time.perf_counter()
    for index, row in enumerate(rows):
        source_id = row["source_id"]
        if source_id not in manifests:
            raise RuntimeError(f"{split} source missing from manifest: {source_id}")
        manifest = manifests[source_id]
        skeleton_paths = frame_map(Path(manifest["skeleton_path"]), "skeleton")
        source_path = (
            source_cache / "trial_motion_cache" / safe_trial_path(source_id).with_suffix(".npz")
        )
        with np.load(source_path, allow_pickle=False) as source:
            frame_ids = np.asarray(source["frame_ids"]).astype(str)
            frame_times = np.asarray(source["frame_time_seconds"], dtype=np.float64)
        missing = [frame_id for frame_id in frame_ids if frame_id not in skeleton_paths]
        if missing:
            raise RuntimeError(f"{split} raw Skeleton frame missing: {source_id}/{missing[0]}")
        people_by_frame = [load_people(skeleton_paths[frame_id]) for frame_id in frame_ids]
        audit: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "source_id": source_id,
            "frames": len(frame_ids),
            "multi_frames": sum(len(people) > 1 for people in people_by_frame),
        }
        chosen = np.asarray(indices[index], dtype=np.int64)
        for name, parameters in STRATEGIES.items():
            raw, selection = select_identity_sequence(people_by_frame, parameters)
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
            quality *= np.asarray([bool(people) for people in people_by_frame], dtype=np.float32)
            # Reproduce the immutable P86/P87 motion-window cache boundary exactly.
            # Without this round-trip, unchanged trials differ merely because this
            # audit summarizes FP32 values while the deployed cache stores FP16.
            window_features = rotated["features"][chosen].astype(np.float16).astype(np.float32)
            window_relations = rotated["relations"][chosen].astype(np.float16).astype(np.float32)
            window_quality = quality[chosen].astype(np.float16).astype(np.float32)
            arrays[name][index] = summarize_window(
                window_features,
                np.asarray(built["joint_mask"])[chosen],
                window_relations,
                rotated["relation_mask"][chosen],
                window_quality,
            )
            audit[f"{name}_changed_frames"] = selection["changed_frames"]
            audit[f"{name}_eligible_frames"] = selection["eligible_frames"]
            totals[name]["changed_frames"] += selection["changed_frames"]
            totals[name]["changed_trials"] += int(selection["changed_frames"] > 0)
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
    return rows, arrays, totals


def model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=700,
        max_depth=28,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
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
            item = {"fold": int(fold["fold"]), **metrics(labels[held], oof[target].argmax(1))}
            per_fold.append(item)
            print(name, item, flush=True)
        final = model()
        final.fit(train_features[name], labels)
        partial = final.predict_proba(test_features[name])
        test_probability = np.zeros((len(test_rows), 40), dtype=np.float64)
        test_probability[:, final.classes_.astype(np.int64)] = partial
        probabilities[name] = oof
        test_probabilities[name] = test_probability
        results[name] = {"overall": metrics(labels, oof.argmax(1)), "folds": per_fold}
        print(name, results[name]["overall"], flush=True)

    selected = max(
        STRATEGIES,
        key=lambda name: (
            results[name]["overall"]["accuracy"],
            results[name]["overall"]["balanced_accuracy"],
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
        "stage": "P89_bone_identity_multi_person_tracking_v1",
        "protocol": (
            "Fixed subject-disjoint folds. Candidate selection uses only within-trial "
            "log bone lengths and a people[0] median anchor; labels, predictions and "
            "leaderboard feedback never enter tracking."
        ),
        "strategies": STRATEGIES,
        "train_tracking": train_audit,
        "test_tracking": test_audit,
        "results": results,
        "selected": selected,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
