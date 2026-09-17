from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from audit_yolo11_pose_skeleton import frame_map
from p31_skeleton_imu_preprocessing import build_skeleton_features
from p46_event_preprocessing import rotate_skeleton_cache
from p86_visual_student_data import WINDOW_BOUNDS
from p89_skeleton_identity_tracking_expert import load_people, summarize_window
from p89_skeleton_invariant_expert import make_model, metrics


PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST = PROJECT_DIR / "data/six_modality_audit/train_union_manifest.csv"
P28 = PROJECT_DIR / "runs/p28_adaptive_ir_pose_skeleton_full/trial_summary.csv"
BASE = PROJECT_DIR / "runs/p89_skeleton_invariant_expert_v1"
BASE_ROWS = PROJECT_DIR / "runs/p86_motion_window_cache_t16_v1/rows.csv"
TEST_ROWS = PROJECT_DIR / "runs/p87s_test_motion_window_t16_v1/rows.csv"
FOLDS = PROJECT_DIR / "data/subject_folds/folds_summary.json"
OUTPUT = PROJECT_DIR / "runs/p89_expanded_2931_skeleton_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def window_indices(frame_count: int, low: float, high: float) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("empty synchronized sequence")
    return np.rint(np.linspace(low * (frame_count - 1), high * (frame_count - 1), 16)).astype(
        np.int64
    )


def counter(frame_id: str) -> str:
    return str(frame_id).rsplit("_", 1)[-1]


def extra_rows() -> list[dict[str, str]]:
    completed = {row["sample_id"] for row in read_csv(P28)}
    rows = []
    for row in read_csv(MANIFEST):
        eligible = all(
            row[f"{modality}_usable"] == "1"
            for modality in ("depth_color", "ir", "skeleton")
        )
        if eligible and row["sample_id"] not in completed:
            rows.append(row)
    rows.sort(key=lambda row: row["sample_id"])
    if len(rows) != 17:
        raise RuntimeError(f"repaired training set changed: expected 17, got {len(rows)}")
    return rows


def build_extra_feature(row: dict[str, str]) -> tuple[np.ndarray, dict[str, object]]:
    depth = frame_map(Path(row["depth_color_path"]), "depth")
    ir = frame_map(Path(row["ir_path"]), "ir")
    skeleton = frame_map(Path(row["skeleton_path"]), "skeleton")
    # These 17 trials were rejected only because D/IR use a bare frame counter
    # while Skeleton prefixes the same counter with an absolute timestamp.
    skeleton_by_counter = {counter(frame_id): path for frame_id, path in skeleton.items()}
    common = sorted(set(depth) & set(ir) & set(skeleton_by_counter), key=lambda value: int(value))
    if len(common) != len(depth) or len(common) != len(ir) or len(common) != len(skeleton):
        raise RuntimeError(
            f"counter repair is not one-to-one: {row['sample_id']} "
            f"D={len(depth)} IR={len(ir)} S={len(skeleton)} common={len(common)}"
        )
    people_by_frame = [load_people(skeleton_by_counter[frame_id]) for frame_id in common]
    empty = np.full((17, 4), np.nan, dtype=np.float32)
    empty[:, 3] = 0.0
    raw = np.stack([people[0] if people else empty for people in people_by_frame])
    # Numeric camera counters are 10 Hz in the same contract used by P31.
    frame_times = (np.asarray(common, dtype=np.float64) - float(common[0])) / 10.0
    built = build_skeleton_features(raw, frame_times)
    rotated = rotate_skeleton_cache(
        built["features"],
        built["feature_mask"],
        built["joint_mask"],
        built["relations"],
        built["relation_mask"],
        frame_times,
    )
    chosen = np.stack(
        [window_indices(len(common), low, high) for low, high in WINDOW_BOUNDS]
    )
    quality = np.asarray(built["frame_quality"], dtype=np.float32)
    quality *= np.asarray([bool(people) for people in people_by_frame], dtype=np.float32)
    feature = summarize_window(
        rotated["features"][chosen].astype(np.float16).astype(np.float32),
        np.asarray(built["joint_mask"])[chosen],
        rotated["relations"][chosen].astype(np.float16).astype(np.float32),
        rotated["relation_mask"][chosen],
        quality[chosen].astype(np.float16).astype(np.float32),
    )
    return feature, {
        "sample_id": row["sample_id"],
        "class_id": int(row["class_id"]),
        "user_id": row["user_id"],
        "frames": len(common),
        "multi_frames": int(sum(len(people) > 1 for people in people_by_frame)),
        "imu_usable": int(row["imu_usable"]),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    extra_path = OUTPUT / "extra17_features.npy"
    audit_path = OUTPUT / "extra17_audit.json"
    repaired_rows = extra_rows()
    if extra_path.is_file() and audit_path.is_file():
        extra_features = np.load(extra_path, mmap_mode="r")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        built = [build_extra_feature(row) for row in repaired_rows]
        extra_features = np.stack([item[0] for item in built]).astype(np.float32)
        audit = [item[1] for item in built]
        np.save(extra_path, extra_features)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    base_rows = read_csv(BASE_ROWS)
    test_rows = read_csv(TEST_ROWS)
    base_features = np.asarray(np.load(BASE / "train_features.npy", mmap_mode="r"))
    test_features = np.asarray(np.load(BASE / "test_features.npy", mmap_mode="r"))
    features = np.concatenate((base_features, np.asarray(extra_features)), axis=0)
    labels = np.asarray(
        [int(row["class_id"]) for row in base_rows]
        + [int(row["class_id"]) for row in repaired_rows],
        dtype=np.int64,
    )
    users = np.asarray(
        [row["user_id"] for row in base_rows]
        + [row["user_id"] for row in repaired_rows]
    ).astype(str)
    sample_ids = np.asarray(
        [row["sample_id"] for row in base_rows]
        + [
            f"train__c{int(row['class_id']):02d}__{row['user_id']}__{row['trial_id']}"
            for row in repaired_rows
        ]
    )
    if len(set(sample_ids.tolist())) != 2931:
        raise RuntimeError("expanded sample ids are not unique")
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    oof = np.zeros((len(features), 40), dtype=np.float64)
    per_fold = []
    base_count = len(base_rows)
    for fold in folds:
        fit = np.isin(users, fold["train_users"])
        held = np.isin(users, fold["val_users"])
        estimator = make_model("extra_trees")
        estimator.fit(features[fit], labels[fit])
        partial = estimator.predict_proba(features[held])
        target = np.flatnonzero(held)
        oof[np.ix_(target, estimator.classes_.astype(np.int64))] = partial
        common = held.copy()
        common[base_count:] = False
        extra = held.copy()
        extra[:base_count] = False
        item = {
            "fold": int(fold["fold"]),
            "common_2914": metrics(labels[common], oof[common].argmax(axis=1)),
            "extra17": metrics(labels[extra], oof[extra].argmax(axis=1))
            if np.any(extra)
            else None,
            "expanded_2931": metrics(labels[held], oof[held].argmax(axis=1)),
        }
        per_fold.append(item)
        print(json.dumps(item), flush=True)

    final = make_model("extra_trees")
    final.fit(features, labels)
    partial = final.predict_proba(test_features)
    test_probability = np.zeros((len(test_rows), 40), dtype=np.float64)
    test_probability[:, final.classes_.astype(np.int64)] = partial
    np.savez_compressed(
        OUTPUT / "oof_logits.npz",
        sample_ids=sample_ids,
        labels=labels,
        skeleton_logits=np.log(np.maximum(oof, 1e-12)),
    )
    np.savez_compressed(
        OUTPUT / "test_logits.npz",
        sample_ids=np.asarray([row["sample_id"] for row in test_rows]),
        skeleton_logits=np.log(np.maximum(test_probability, 1e-12)),
    )
    common_metrics = metrics(labels[:base_count], oof[:base_count].argmax(axis=1))
    extra_metrics = metrics(labels[base_count:], oof[base_count:].argmax(axis=1))
    report = {
        "stage": "P89_repair_counter_aligned_2931_skeleton_training_v1",
        "protocol": (
            "Recover exactly 17 D/IR/Skeleton trials rejected by literal frame-id "
            "intersection. D/IR bare counters and Skeleton timestamp-prefixed counters "
            "are joined one-to-one by their final counter. Fixed subject-disjoint folds."
        ),
        "base_rows": base_count,
        "repaired_rows": len(repaired_rows),
        "expanded_rows": len(features),
        "common_2914_metrics": common_metrics,
        "extra17_metrics": extra_metrics,
        "per_fold": per_fold,
        "extra_audit": audit,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
