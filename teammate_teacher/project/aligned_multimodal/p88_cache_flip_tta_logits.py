from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from adapt_p87s_structured_student import (
    model_build_args,
    resolve_config_path,
)
from p86_cached_motion_data import (
    P86CachedSequenceMotionDataset,
    collate_p86_cached_motion,
)
from train_p86_mobind_fusion_proxy import build_model, metric_dict, model_forward


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache original and horizontal-flip P88 logits from a frozen P87-S run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


class PixelMotionDataset(Dataset[dict[str, Any]]):
    def __init__(self, base: P86CachedSequenceMotionDataset, pixel_cache: Path) -> None:
        self.base = base
        self.images = np.load(pixel_cache / "images.npy", mmap_mode="r")
        if len(self.images) != len(base.rows):
            raise RuntimeError("pixel and motion cache row counts differ")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        cache_index = int(item["cache_index"])
        item["images"] = torch.from_numpy(
            np.asarray(self.images[cache_index]).copy()
        )
        return item


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    base_checkpoint_path = Path(run_summary["base_checkpoint"]).resolve()
    base_summary = json.loads(
        (base_checkpoint_path.parent / "summary.json").read_text(encoding="utf-8")
    )
    model, _visual_config, _pretrain_config = build_model(
        model_build_args(base_checkpoint_path, base_summary)
    )
    checkpoint = torch.load(
        run_dir / "unified_student.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)

    config = base_summary["config"]
    repository_root = PROJECT_DIR.parent
    sequence_cache = resolve_config_path(config["sequence_cache"], repository_root)
    motion_cache = resolve_config_path(config["motion_cache"], repository_root)
    pixel_cache = resolve_config_path(config["pixel_cache"], repository_root)
    full = P86CachedSequenceMotionDataset(
        sequence_cache,
        motion_cache,
        pixel_cache,
        resolve_config_path(config["teacher_features"], repository_root),
        resolve_config_path(config["teacher_logits"], repository_root),
        imu_teacher_logits=(
            resolve_config_path(config["imu_teacher_logits"], repository_root)
            if config.get("imu_teacher_logits")
            else None
        ),
        imu_event_features=(
            resolve_config_path(config["imu_event_features"], repository_root)
            if config.get("imu_event_features")
            else None
        ),
    )
    import csv

    with (run_dir / "subject_holdout_predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reference_rows = list(csv.DictReader(handle))
    sample_ids = [row["sample_id"] for row in reference_rows]
    indices = np.asarray([full.index_lookup[sample_id] for sample_id in sample_ids])
    selected = P86CachedSequenceMotionDataset(
        sequence_cache,
        motion_cache,
        pixel_cache,
        resolve_config_path(config["teacher_features"], repository_root),
        resolve_config_path(config["teacher_logits"], repository_root),
        indices=indices,
        temporal_augment=False,
        imu_teacher_logits=(
            resolve_config_path(config["imu_teacher_logits"], repository_root)
            if config.get("imu_teacher_logits")
            else None
        ),
        imu_event_features=(
            resolve_config_path(config["imu_event_features"], repository_root)
            if config.get("imu_event_features")
            else None
        ),
    )
    loader = DataLoader(
        PixelMotionDataset(selected, pixel_cache),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate_p86_cached_motion,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    original_rows = []
    flipped_rows = []
    seen_ids: list[str] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch = to_device(batch, device)
            images = batch["images"]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                original_sequence = model.visual.encode_backbone_sequence(images)
                original_batch = {**batch, "backbone_sequence": original_sequence}
                original = model_forward(model, original_batch)["logits"]
                flipped_sequence = model.visual.encode_backbone_sequence(
                    torch.flip(images, dims=(-1,))
                )
                flipped_batch = {**batch, "backbone_sequence": flipped_sequence}
                flipped = model_forward(model, flipped_batch)["logits"]
            original_rows.append(original.float().cpu().numpy())
            flipped_rows.append(flipped.float().cpu().numpy())
            seen_ids.extend(map(str, batch["sample_id"]))
            labels.extend(map(int, batch["label"].cpu().numpy()))
            print(
                json.dumps(
                    {
                        "batch": batch_index,
                        "processed": len(seen_ids),
                        "total": len(selected),
                    }
                ),
                flush=True,
            )
    if seen_ids != sample_ids:
        raise RuntimeError("TTA inference row order changed")
    original_logits = np.concatenate(original_rows)
    flipped_logits = np.concatenate(flipped_rows)
    labels_array = np.asarray(labels, dtype=np.int64)
    saved_logits = np.asarray(
        np.load(run_dir / "subject_holdout_logits.npy"), dtype=np.float32
    )
    original_prediction = original_logits.argmax(axis=1)
    flipped_prediction = flipped_logits.argmax(axis=1)
    averaged_prediction = (0.5 * original_logits + 0.5 * flipped_logits).argmax(axis=1)
    users = [row["user_id"] for row in reference_rows]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "flip_tta_logits.npz",
        sample_ids=np.asarray(sample_ids),
        labels=labels_array,
        original_logits=original_logits.astype(np.float32),
        flipped_logits=flipped_logits.astype(np.float32),
    )
    summary = {
        "stage": "P88_horizontal_flip_TTA_cache",
        "status": "complete",
        "run_dir": str(run_dir),
        "rows": len(sample_ids),
        "original_reconstruction": metric_dict(labels_array, original_prediction, users),
        "flipped": metric_dict(labels_array, flipped_prediction, users),
        "equal_average": metric_dict(labels_array, averaged_prediction, users),
        "original_vs_saved_top1_agreement": float(
            np.mean(original_prediction == saved_logits.argmax(axis=1))
        ),
        "original_vs_saved_max_abs_logit_difference": float(
            np.max(np.abs(original_logits - saved_logits))
        ),
        "output": str(output / "flip_tta_logits.npz"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
