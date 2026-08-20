from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data.thermal_v2_inventory import (
    image_files,
    load_canonical_thermal_records,
    resolve_thermal_trial_path,
)
from src.data.thermal_v2_sampling import normalized_window_indices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
DEFAULT_T0_REPORT = PROJECT_ROOT / "reports" / "thermal_stage0_data_alignment_audit.json"
DEFAULT_SPLIT = (
    PROJECT_ROOT / "metadata" / "splits" / "train12_val2_user6_user7_development.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "metadata" / "thermal" / "thermal_v2_train12_normalization.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChannelMoments:
    def __init__(self) -> None:
        self.pixel_count = 0
        self.channel_sum = np.zeros(3, dtype=np.float64)
        self.channel_square_sum = np.zeros(3, dtype=np.float64)

    def update(self, rgb: np.ndarray) -> None:
        values = np.asarray(rgb, dtype=np.float64)
        if values.ndim < 2 or values.shape[-1] != 3:
            raise ValueError("rgb must end with three channels")
        flat = values.reshape(-1, 3)
        if not np.isfinite(flat).all():
            raise ValueError("rgb statistics input must be finite")
        self.pixel_count += len(flat)
        self.channel_sum += flat.sum(axis=0)
        self.channel_square_sum += np.square(flat).sum(axis=0)

    def finalize(self):
        if self.pixel_count < 1:
            raise ValueError("cannot finalize empty channel moments")
        mean = self.channel_sum / self.pixel_count
        variance = np.maximum(
            self.channel_square_sum / self.pixel_count - np.square(mean), 0.0
        )
        return mean, np.sqrt(variance)


def normalization_frame_indices(frame_count: int) -> tuple[int, ...]:
    windows = ((0.25, 0.25), (0.5, 0.5), (0.75, 0.75))
    values = normalized_window_indices(
        frame_count, windows=windows, frames_per_window=1
    )
    return tuple(dict.fromkeys(window[0] for window in values))


def decode_preprocessed_rgb(
    path: Path, *, resize_short_side: int = 176, crop_size: int = 160
) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"undecodable image: {path}")
    height, width = image.shape[:2]
    scale = resize_short_side / min(height, width)
    resized_width = max(crop_size, int(round(width * scale)))
    resized_height = max(crop_size, int(round(height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    top = (resized_height - crop_size) // 2
    left = (resized_width - crop_size) // 2
    crop_bgr = resized[top : top + crop_size, left : left + crop_size]
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def fit_train12_normalization(
    *,
    data_root: Path,
    t0_report: Path,
    split_path: Path,
) -> dict[str, Any]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_users = set(str(value) for value in split["train_user_ids"])
    records = load_canonical_thermal_records(t0_report)
    train_records = [record for record in records if record["development_split"] == "train12"]
    if {str(record["user_id"]) for record in train_records} != train_users:
        raise ValueError("T0 train12 users do not match the fixed development split")

    moments = ChannelMoments()
    decoded_frames = 0
    skipped_trials = 0
    sampled_ids: list[str] = []
    for record in train_records:
        if not record.get("usable", False):
            skipped_trials += 1
            continue
        files = image_files(resolve_thermal_trial_path(data_root, record))
        if not files:
            skipped_trials += 1
            continue
        for index in normalization_frame_indices(len(files)):
            moments.update(decode_preprocessed_rgb(files[index]))
            decoded_frames += 1
        sampled_ids.append(str(record["sample_id"]))

    mean, std = moments.finalize()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "thermal_v2_train12_rgb_normalization",
        "fit_population": "train12_only",
        "fit_user_ids": sorted(train_users),
        "canonical_train12_trials": len(train_records),
        "sampled_usable_trials": len(sampled_ids),
        "skipped_unavailable_trials": skipped_trials,
        "decoded_sample_frames": decoded_frames,
        "pixel_count": moments.pixel_count,
        "rgb_mean": mean.tolist(),
        "rgb_std": std.tolist(),
        "sample_policy": "unique nearest frames at normalized times 0.25, 0.50, 0.75 per usable train12 trial",
        "resize_short_side": 176,
        "center_crop_size": 160,
        "source_t0_report_sha256": sha256_file(t0_report),
        "development_split_sha256": sha256_file(split_path),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sampled_ids).encode("utf-8")
        ).hexdigest(),
        "script_sha256": sha256_file(Path(__file__)),
        "forbidden_access": {
            "heldout4_labels": False,
            "competition_test": False,
            "quarantined_evidence": False,
            "ir_or_depth_inputs": False,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["artifact_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Thermal v2 RGB normalization on the fixed train12 users."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--t0-report", type=Path, default=DEFAULT_T0_REPORT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = fit_train12_normalization(
        data_root=args.data_root.resolve(),
        t0_report=args.t0_report.resolve(),
        split_path=args.split.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sampled_trials": payload["sampled_usable_trials"],
                "sampled_frames": payload["decoded_sample_frames"],
                "artifact_sha256": payload["artifact_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
