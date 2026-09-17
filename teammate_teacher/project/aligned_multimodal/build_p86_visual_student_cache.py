from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from p30_shared_dir_roi_data import P30SharedDIRFeatureDataset
from p30_shared_dir_roi_model import MODALITY_NAMES, REGION_NAMES, PYRAMID_FEATURE_DIM
from p86_visual_student_data import FRAME_COUNT, VIEW_NAMES, WINDOW_BOUNDS, window_indices


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_P30 = PROJECT_DIR / "runs/p30_shared_dir_roi_features_full"
DEFAULT_TEACHER = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
)
DEFAULT_LOGITS = (
    PROJECT_DIR
    / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p86_visual_student_cache_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a value-equivalent compact P86 memmap from verified P30 features."
    )
    parser.add_argument("--p30-run", type=Path, default=DEFAULT_P30)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_LOGITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_replace(building: Path, final: Path) -> None:
    if final.exists():
        final.unlink()
    os.replace(building, final)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_files = {
        "features": output / "features.npy",
        "view_mask": output / "view_mask.npy",
        "view_quality": output / "view_quality.npy",
        "time_position": output / "time_position.npy",
        "rows": output / "rows.csv",
        "summary": output / "summary.json",
    }
    if all(path.exists() for path in final_files.values()) and not args.overwrite:
        print(f"P86 compact cache already complete: {output}", flush=True)
        return

    with np.load(args.teacher_features.resolve(), allow_pickle=False) as data:
        canonical_ids = np.asarray(data["sample_ids"]).astype(str)
        source_ids = np.asarray(data["source_ids"]).astype(str)
        teacher_labels = np.asarray(data["labels"], dtype=np.int64)
        teacher_users = np.asarray(data["users"]).astype(str)
    with np.load(args.teacher_logits.resolve(), allow_pickle=False) as data:
        logit_ids = np.asarray(data["sample_ids"]).astype(str)
        folds = np.asarray(data["folds"], dtype=np.int64)
    if not np.array_equal(canonical_ids, logit_ids):
        raise RuntimeError("P86 teacher feature/logit order mismatch")
    source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
    if len(source_lookup) != 2914:
        raise RuntimeError("P86 teacher source IDs are not unique")

    dataset = P30SharedDIRFeatureDataset(args.p30_run)
    if len(dataset) != 2914:
        raise RuntimeError(f"P30 universe changed: {len(dataset)}")
    count = len(dataset)
    feature_shape = (count, 2, FRAME_COUNT, len(VIEW_NAMES), PYRAMID_FEATURE_DIM)
    scalar_shape = feature_shape[:-1]
    building = {
        key: path.with_name(path.name + ".building")
        for key, path in final_files.items()
        if key not in {"summary"}
    }
    for path in building.values():
        if path.exists():
            path.unlink()
    features = np.lib.format.open_memmap(
        building["features"], mode="w+", dtype=np.float16, shape=feature_shape
    )
    view_mask = np.lib.format.open_memmap(
        building["view_mask"], mode="w+", dtype=np.bool_, shape=scalar_shape
    )
    view_quality = np.lib.format.open_memmap(
        building["view_quality"], mode="w+", dtype=np.float16, shape=scalar_shape
    )
    time_position = np.lib.format.open_memmap(
        building["time_position"], mode="w+", dtype=np.float32, shape=(count, 2, FRAME_COUNT)
    )
    ir_index = MODALITY_NAMES.index("ir")
    region_indices = tuple(
        REGION_NAMES.index(name)
        for name in ("global_fallback", "full_body", "hand_workspace")
    )
    rows: list[dict[str, str | int]] = []
    start = time.perf_counter()
    for row_index in range(count):
        item = dataset[row_index]
        source_id = str(item["sample_id"])
        if source_id not in source_lookup:
            raise RuntimeError(f"P30 source missing from teacher universe: {source_id}")
        teacher_index = source_lookup[source_id]
        if int(item["class_id"]) != int(teacher_labels[teacher_index]):
            raise RuntimeError(f"label mismatch: {source_id}")
        if str(item["user_id"]) != teacher_users[teacher_index]:
            raise RuntimeError(f"user mismatch: {source_id}")
        length = len(item["frame_ids"])
        chosen = np.stack(
            [window_indices(length, low, high) for low, high in WINDOW_BOUNDS]
        )
        chosen_tensor = np.asarray(chosen, dtype=np.int64)
        item_features = item["features"].numpy()
        item_mask = item["roi_valid"].numpy()
        item_quality = item["roi_quality"].numpy()
        features[row_index] = item_features[chosen_tensor, ir_index][
            :, :, region_indices
        ].astype(np.float16)
        view_mask[row_index] = item_mask[chosen_tensor][:, :, region_indices]
        view_quality[row_index] = item_quality[chosen_tensor][:, :, region_indices].astype(
            np.float16
        )
        view_mask[row_index, :, :, 0] = True
        view_quality[row_index, :, :, 0] = np.maximum(
            view_quality[row_index, :, :, 0], np.float16(0.5)
        )
        time_position[row_index] = chosen_tensor.astype(np.float32) / float(max(length - 1, 1))
        rows.append(
            {
                "row_index": row_index,
                "sample_id": canonical_ids[teacher_index],
                "source_id": source_id,
                "user_id": teacher_users[teacher_index],
                "class_id": int(teacher_labels[teacher_index]),
                "fold": int(folds[teacher_index]),
                "frames": length,
            }
        )
        if (row_index + 1) % 250 == 0 or row_index + 1 == count:
            print(
                f"compact {row_index + 1}/{count} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    features.flush()
    view_mask.flush()
    view_quality.flush()
    time_position.flush()
    del features, view_mask, view_quality, time_position
    with building["rows"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key in ("features", "view_mask", "view_quality", "time_position", "rows"):
        atomic_replace(building[key], final_files[key])

    check_features = np.load(final_files["features"], mmap_mode="r", allow_pickle=False)
    check_mask = np.load(final_files["view_mask"], mmap_mode="r", allow_pickle=False)
    check_quality = np.load(final_files["view_quality"], mmap_mode="r", allow_pickle=False)
    check_time = np.load(final_files["time_position"], mmap_mode="r", allow_pickle=False)
    if check_features.shape != feature_shape or check_mask.shape != scalar_shape:
        raise RuntimeError("compact cache shape verification failed")
    if not np.isfinite(check_features).all() or not np.isfinite(check_quality).all():
        raise RuntimeError("compact cache contains non-finite visual values")
    if not np.isfinite(check_time).all() or np.any(np.diff(check_time, axis=2) < 0):
        raise RuntimeError("compact cache time positions are invalid")
    if not check_mask[:, :, :, 0].all():
        raise RuntimeError("compact scene view must always be available")
    summary = {
        "stage": "P86_compact_visual_student_cache",
        "protocol": (
            "Value-equivalent selection from verified P30 IR pyramid features: "
            "early/late x 16 frames x scene/person/workspace. No labels, logits or "
            "Large features are used as student inputs."
        ),
        "rows": count,
        "shape": list(feature_shape),
        "dtype": "float16",
        "windows": [list(bounds) for bounds in WINDOW_BOUNDS],
        "views": list(VIEW_NAMES),
        "source_p30": str(args.p30_run.resolve()),
        "elapsed_seconds": time.perf_counter() - start,
        "files": {
            key: {"path": str(path), "bytes": path.stat().st_size}
            for key, path in final_files.items()
            if key != "summary"
        },
        "verification": {
            "all_finite": True,
            "time_monotonic": True,
            "scene_always_available": True,
        },
    }
    final_files["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
