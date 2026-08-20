from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch

from scripts.audit_thermal_v2_inputs import (
    DEFAULT_CONTEXT,
    DEFAULT_DATA_ROOT,
    DEFAULT_WEIGHTS,
    YoloPosePredictor,
    load_jsonl,
    sha256_file,
    trial_path,
)
from src.data.thermal_v2_features import encode_pose_step
from src.data.thermal_v2_inventory import image_files
from src.data.thermal_v2_sampling import normalized_window_indices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "metadata/thermal/thermal_v2_pose_cache.npz"
ALLOWED_SPLITS = {"train12", "val_user6_user7"}


def pose_key(sample_id: str, frame_index: int) -> str:
    return f"{sample_id}|{frame_index}"


def build_pose_cache_arrays(
    records: Sequence[dict[str, Any]],
    *,
    data_root: Path,
    predictor: Callable[
        [list[Path]], list[tuple[torch.Tensor | None, torch.Tensor | None]]
    ],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(record.get("development_split") not in ALLOWED_SPLITS for record in records):
        raise ValueError("record outside fixed development population")
    jobs: list[tuple[str, int, Path, tuple[int, int]]] = []
    usable_trials = 0
    for record in records:
        if not bool(record.get("usable", False)):
            continue
        usable_trials += 1
        files = image_files(trial_path(data_root, record))
        if not files:
            raise ValueError(f"usable trial has no Thermal frames: {record['sample_id']}")
        indices = sorted(
            {index for window in normalized_window_indices(len(files)) for index in window}
        )
        frame_size = tuple(int(value) for value in record.get("frame_size", (0, 0)))
        if frame_size[0] < 1 or frame_size[1] < 1:
            image = cv2.imdecode(np.fromfile(files[0], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"undecodable Thermal frame: {files[0]}")
            frame_size = (image.shape[1], image.shape[0])
        jobs.extend(
            (str(record["sample_id"]), index, files[index], frame_size)
            for index in indices
        )

    keys: list[str] = []
    pose: list[np.ndarray] = []
    valid: list[bool] = []
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        predictions = predictor([path for _, _, path, _ in batch])
        if len(predictions) != len(batch):
            raise ValueError("pose predictor returned the wrong batch length")
        for (sample_id, index, _, frame_size), (keypoints, bbox) in zip(
            batch, predictions, strict=True
        ):
            is_valid = keypoints is not None and bbox is not None
            keys.append(pose_key(sample_id, index))
            pose.append(encode_pose_step(keypoints, bbox, frame_size).numpy())
            valid.append(is_valid)
    if len(keys) != len(set(keys)):
        raise ValueError("pose cache keys must be unique")
    arrays = {
        "keys": np.asarray(keys, dtype=str),
        "pose": np.asarray(pose, dtype=np.float32).reshape(-1, 56),
        "valid": np.asarray(valid, dtype=np.bool_),
    }
    summary = {
        "canonical_trials": len(records),
        "usable_trials": usable_trials,
        "cached_unique_frames": len(keys),
        "valid_pose_frames": int(arrays["valid"].sum()),
        "failed_pose_frames": int((~arrays["valid"]).sum()),
    }
    return arrays, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen Thermal v2 pose cache.")
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = args.context.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    records = load_jsonl(context)
    arrays, summary = build_pose_cache_arrays(
        records,
        data_root=args.data_root.resolve(),
        predictor=YoloPosePredictor(weights, device=args.device),
        batch_size=args.batch_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        **summary,
        "cache_path": str(output),
        "cache_bytes": output.stat().st_size,
        "cache_sha256": sha256_file(output),
        "context_sha256": sha256_file(context),
        "yolo_weights_sha256": sha256_file(weights),
        "splits": sorted(ALLOWED_SPLITS),
        "sample_selection": "three_normalized_time_windows_16_frames_unique_keys",
        "heldout4_labels_read": False,
        "competition_test_read": False,
        "ir_depth_inputs_read": False,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
