"""Build the label-free Skeleton feature contract required by A18 on Kaggle Test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from aligned_data import frame_map
from build_raw_skeleton_cache import load_first_person
from p31_skeleton_imu_preprocessing import safe_trial_path
from p89_build_crossmodal_statistics import (
    synchronization_features,
    temporal_statistics,
)
from p90_motionbert_teacher import (
    INIT_SPECS,
    VIEW_AXES,
    extract_features as extract_motionbert_features,
    interpolate_pose,
    normalize_projection,
)
from p91_hdgcn_pretrained_teacher import (
    H36M_TO_NTU,
    extract_features as extract_hdgcn_features,
    load_npz,
)


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "data/p46_test_union_manifest.csv"
DEFAULT_MOTION_SOURCE = HERE / "runs/p87s_test_motion_source_v1"
DEFAULT_MOTION_WINDOW = HERE / "runs/p87s_test_motion_window_t16_v1"
DEFAULT_OUTPUT = HERE / "runs/a18_test_features_v1"
TEST_ROWS = 405


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--motion-source", type=Path, default=DEFAULT_MOTION_SOURCE)
    parser.add_argument("--motion-window", type=Path, default=DEFAULT_MOTION_WINDOW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--motionbert-batch-size", type=int, default=16)
    parser.add_argument("--hdgcn-batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def official_rows(manifest_path: Path, motion_window: Path) -> list[dict[str, str]]:
    manifest = read_csv(manifest_path)
    if len(manifest) != TEST_ROWS:
        raise RuntimeError(f"official Test universe changed: {len(manifest)}")
    by_official = {row["official_sample_id"]: row for row in manifest}
    if len(by_official) != TEST_ROWS:
        raise RuntimeError("duplicate official Test sample id")
    motion_rows = read_csv(motion_window / "rows.csv")
    sample_ids = [row["sample_id"] for row in motion_rows]
    if len(sample_ids) != TEST_ROWS or len(set(sample_ids)) != TEST_ROWS:
        raise RuntimeError("Test motion row contract changed")
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_official]
    if missing:
        raise RuntimeError(f"motion rows are absent from Test manifest: {missing[:3]}")
    return [by_official[sample_id] for sample_id in sample_ids]


def load_trial_raw_pose(row: dict[str, str], motion_source: Path) -> np.ndarray:
    trial_path = (
        motion_source
        / "trial_motion_cache"
        / safe_trial_path(row["sample_id"]).with_suffix(".npz")
    )
    with np.load(trial_path, allow_pickle=False) as trial:
        frame_ids = np.asarray(trial["frame_ids"]).astype(str)
    skeleton_paths = frame_map(Path(row["skeleton_path"]), "skeleton")
    missing = [frame_id for frame_id in frame_ids if frame_id not in skeleton_paths]
    if missing:
        raise RuntimeError(
            f"Test Skeleton timeline is incomplete: {row['official_sample_id']}/{missing[0]}"
        )
    return np.stack(
        [load_first_person(skeleton_paths[frame_id])[0] for frame_id in frame_ids]
    ).astype(np.float32)


def build_pose_caches(
    rows: list[dict[str, str]],
    motion_source: Path,
    motionbert_path: Path,
    hdgcn_path: Path,
) -> None:
    sample_ids = np.asarray([row["official_sample_id"] for row in rows])
    motionbert = np.zeros((TEST_ROWS, 81, 17, 3), dtype=np.float32)
    hdgcn = np.zeros((TEST_ROWS, 3, 64, 25, 2), dtype=np.float32)
    for index, row in enumerate(rows):
        raw = load_trial_raw_pose(row, motion_source)
        xyz81, confidence81 = interpolate_pose(raw, 81)
        motionbert[index] = normalize_projection(
            xyz81, confidence81, VIEW_AXES["front"]
        )

        xyz64, confidence64 = interpolate_pose(raw, 64)
        mapped = xyz64[:, H36M_TO_NTU].copy()
        mapped -= mapped[0, 1][None, None, :]
        mapped[confidence64[:, H36M_TO_NTU] <= 0] = 0
        hdgcn[index, :, :, :, 0] = mapped.transpose(2, 0, 1)
        if (index + 1) % 100 == 0 or index + 1 == TEST_ROWS:
            print(f"A18 Test pose={index + 1}/{TEST_ROWS}", flush=True)

    labels = np.full(TEST_ROWS, -1, dtype=np.int64)
    np.savez_compressed(
        motionbert_path,
        sample_ids=sample_ids,
        labels=labels,
        target_frames=np.asarray(81, dtype=np.int32),
        front=motionbert,
    )
    np.savez_compressed(
        hdgcn_path,
        sample_ids=sample_ids,
        labels=labels,
        pose=hdgcn,
        mapping=H36M_TO_NTU,
    )


def build_cross_statistics(motion_window: Path, output_path: Path) -> None:
    skeleton = np.asarray(
        np.load(motion_window / "skeleton_features.npy", mmap_mode="r"),
        dtype=np.float32,
    ).reshape(-1, 32, 17, 13)
    relations = np.asarray(
        np.load(motion_window / "skeleton_relations.npy", mmap_mode="r"),
        dtype=np.float32,
    ).reshape(-1, 32, 18)
    imu = np.asarray(
        np.load(motion_window / "imu_bin_statistics.npy", mmap_mode="r"),
        dtype=np.float32,
    ).reshape(-1, 32, 5, 52)
    imu_global = np.asarray(
        np.load(motion_window / "imu_global_statistics.npy", mmap_mode="r"),
        dtype=np.float32,
    ).reshape(len(skeleton), -1)
    skeleton_quality = np.asarray(
        np.load(motion_window / "skeleton_frame_quality.npy", mmap_mode="r"),
        dtype=np.float32,
    ).reshape(-1, 32, 1)
    imu_mask = np.asarray(
        np.load(motion_window / "imu_bin_mask.npy", mmap_mode="r"),
        dtype=np.float32,
    ).reshape(-1, 32, 5)
    blocks = [
        temporal_statistics(skeleton.reshape(len(skeleton), 32, -1)),
        temporal_statistics(relations),
        temporal_statistics(imu.reshape(len(imu), 32, -1)),
        np.nan_to_num(imu_global),
        temporal_statistics(skeleton_quality),
        temporal_statistics(imu_mask),
        synchronization_features(skeleton, imu),
    ]
    features = np.concatenate(blocks, axis=1).astype(np.float32)
    rows = read_csv(motion_window / "rows.csv")
    if features.shape != (TEST_ROWS, 6195):
        raise RuntimeError(f"Test cross-statistics shape changed: {features.shape}")
    np.savez_compressed(
        output_path,
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        features=features,
    )


def feature_contract(path: Path, field: str, shape: tuple[int, ...]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        sample_ids = np.asarray(archive["sample_ids"]).astype(str)
        values = np.asarray(archive[field])
    if len(sample_ids) != TEST_ROWS or len(np.unique(sample_ids)) != TEST_ROWS:
        raise RuntimeError(f"{path.name} sample-id contract changed")
    if values.shape != shape or not np.isfinite(values).all():
        raise RuntimeError(f"{path.name}/{field} has invalid shape or values: {values.shape}")
    return {"path": str(path), "sha256": sha256(path), "shape": list(values.shape)}


def main() -> None:
    args = parse_args()
    if args.motionbert_batch_size < 1 or args.hdgcn_batch_size < 1:
        raise ValueError("feature extraction batch sizes must be positive")
    manifest = args.manifest.resolve()
    motion_source = args.motion_source.resolve()
    motion_window = args.motion_window.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = official_rows(manifest, motion_window)
    motionbert_pose = output / "motionbert_pose_t81.npz"
    hdgcn_pose = output / "hdgcn_ntu25_pose_t64.npz"
    motionbert_features = output / "motionbert_pretrain_front_t81.npz"
    hdgcn_features = output / "hdgcn_six_stream_tokens.npz"
    cross_statistics = output / "crossmodal_statistics.npz"
    started = time.perf_counter()

    if args.overwrite or not (motionbert_pose.is_file() and hdgcn_pose.is_file()):
        build_pose_caches(rows, motion_source, motionbert_pose, hdgcn_pose)
    if args.overwrite or not motionbert_features.is_file():
        extract_motionbert_features(
            INIT_SPECS["pretrain"],
            motionbert_pose,
            ["front"],
            motionbert_features,
            args.motionbert_batch_size,
        )
    if args.overwrite or not hdgcn_features.is_file():
        extract_hdgcn_features(
            load_npz(hdgcn_pose), hdgcn_features, args.hdgcn_batch_size
        )
    if args.overwrite or not cross_statistics.is_file():
        build_cross_statistics(motion_window, cross_statistics)

    artifacts = {
        "motionbert": feature_contract(
            motionbert_features, "features", (TEST_ROWS, 12 * 768)
        ),
        "hdgcn": feature_contract(
            hdgcn_features, "tokens", (TEST_ROWS, 6, 16, 256)
        ),
        "cross_statistics": feature_contract(
            cross_statistics, "features", (TEST_ROWS, 6195)
        ),
    }
    summary = {
        "status": "complete",
        "stage": "A18_label_free_test_skeleton_features",
        "test_rows": TEST_ROWS,
        "labels_read": False,
        "source_timeline": (
            "P87S Test motion-source frame IDs; visual-aligned for 401 readable rows "
            "and full Skeleton timeline for four unreadable-IR rows"
        ),
        "artifacts": artifacts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
