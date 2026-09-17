from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from p86_mc3_visual_model import P86MC3VisualStudent
from p86_visual_pixel_data import P86VisualPixelDataset, collate_p86_pixels


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PIXELS = PROJECT_DIR / "runs/p86_visual_pixel_cache_t16_r160_v12"
DEFAULT_FEATURES = PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
DEFAULT_LOGITS = (
    PROJECT_DIR
    / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache compact MC3 layer4 time sequences for fast P86 mechanism audits."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_LOGITS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--spatial-grid",
        type=int,
        choices=(1, 2, 5),
        default=1,
        help="Use 2 or 5 to retain ordered layer4 regions per frame.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[P86MC3VisualStudent, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    if config.get("backbone") != "mc3_18_temporal":
        raise ValueError("sequence caching requires a temporal MC3 checkpoint")
    model = P86MC3VisualStudent(
        classes=int(config.get("classes", 40)),
        width=int(config.get("width", 512)),
        dropout=float(config.get("dropout", 0.18)),
        fusion_mode=str(config.get("fusion_mode", "gated")),
        enable_distillation_projection=bool(
            config.get("enable_distillation_projection", False)
        ),
        frames=int(config["frames"]),
        kinetics_pretrained=False,
        temporal_modeling=True,
        exact_time_modeling=bool(config.get("exact_time_modeling", False)),
        cross_view_time_modeling=bool(config.get("cross_view_time_modeling", False)),
        spatial_region_modeling=bool(config.get("spatial_region_modeling", False)),
        region_temporal_modeling=bool(config.get("region_temporal_modeling", False)),
        structured_region_modeling=bool(config.get("structured_region_modeling", False)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().to(device)
    return model, config


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint_path, device)
    frames = int(config["frames"])

    reference = P86VisualPixelDataset(
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        augment=False,
    )
    sequence_name = (
        "backbone_region_sequence_fp16.npy"
        if args.spatial_grid > 1
        else "backbone_sequence_fp16.npy"
    )
    sequence_path = output / sequence_name
    anchor_logits_path = output / "anchor_logits_fp16.npy"
    completed_path = output / "completed.npy"
    expected_shape = (
        (len(reference), 2, 3, frames, args.spatial_grid**2, 512)
        if args.spatial_grid > 1
        else (len(reference), 2, 3, frames, 512)
    )
    if sequence_path.exists() or anchor_logits_path.exists() or completed_path.exists():
        if (
            not sequence_path.exists()
            or not anchor_logits_path.exists()
            or not completed_path.exists()
        ):
            raise RuntimeError(
                "partial sequence cache metadata; remove output or restore all cache files"
            )
        sequence = np.lib.format.open_memmap(sequence_path, mode="r+")
        anchor_logits = np.lib.format.open_memmap(anchor_logits_path, mode="r+")
        completed = np.lib.format.open_memmap(completed_path, mode="r+")
        if (
            sequence.shape != expected_shape
            or anchor_logits.shape != (len(reference), 40)
            or completed.shape != (len(reference),)
        ):
            raise RuntimeError("existing sequence cache shape differs")
    else:
        sequence = np.lib.format.open_memmap(
            sequence_path, mode="w+", dtype=np.float16, shape=expected_shape
        )
        completed = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=np.uint8, shape=(len(reference),)
        )
        anchor_logits = np.lib.format.open_memmap(
            anchor_logits_path, mode="w+", dtype=np.float16, shape=(len(reference), 40)
        )
        sequence[:] = 0
        anchor_logits[:] = 0
        completed[:] = 0
        sequence.flush()
        anchor_logits.flush()
        completed.flush()

    pending = np.flatnonzero(~np.asarray(completed, dtype=bool))
    loader = None
    if len(pending):
        dataset = P86VisualPixelDataset(
            args.pixel_cache,
            args.teacher_features,
            args.teacher_logits,
            indices=pending,
            augment=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=True,
            collate_fn=collate_p86_pixels,
        )
    started = time.perf_counter()
    emitted = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader or ()):
            if args.max_batches and batch_index >= args.max_batches:
                break
            images = batch["images"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                if args.spatial_grid > 1:
                    compact, values = model.encode_backbone_spatial_sequences(
                        images, args.spatial_grid
                    )
                else:
                    values = model.encode_backbone_sequence(images)
                    compact = values
                valid = batch["view_valid"].to(device, non_blocking=True)
                quality = batch["view_quality"].to(device, non_blocking=True)
                global_time = batch["global_time_position"].to(device, non_blocking=True)
                if args.spatial_grid == 2 and model.spatial_region_modeling:
                    anchor = model.forward_from_backbone_region_sequence(
                        values, valid, quality, global_time
                    )["logits"]
                else:
                    anchor = model.forward_from_backbone_sequence(
                        compact, valid, quality, global_time
                    )["logits"]
            values_np = values.to(dtype=torch.float16).cpu().numpy()
            anchor_np = anchor.to(dtype=torch.float16).cpu().numpy()
            indices = np.asarray(batch["cache_index"], dtype=np.int64)
            sequence[indices] = values_np
            anchor_logits[indices] = anchor_np
            completed[indices] = 1
            emitted += len(indices)
            if emitted % 64 < len(indices):
                sequence.flush()
                anchor_logits.flush()
                completed.flush()
                print(
                    json.dumps(
                        {
                            "cached": int(np.asarray(completed).sum()),
                            "total": len(reference),
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
    sequence.flush()
    anchor_logits.flush()
    completed.flush()

    with (output / "rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ("sample_id", "source_id", "user_id", "class_id")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in reference.rows:
            writer.writerow({key: row[key] for key in fieldnames})
    complete_count = int(np.asarray(completed).sum())
    summary = {
        "stage": (
            "P86_MC3_layer4_region_sequence_cache"
            if args.spatial_grid > 1
            else "P86_MC3_layer4_sequence_cache"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "shape": list(expected_shape),
        "dtype": "float16",
        "spatial_grid": args.spatial_grid,
        "completed": complete_count,
        "total": len(reference),
        "cache_bytes": int(
            sequence_path.stat().st_size
            + anchor_logits_path.stat().st_size
            + completed_path.stat().st_size
        ),
        "purpose": (
            "Training-only acceleration for temporal, distillation and multimodal fusion "
            "mechanism audits. Final inference still reads raw modalities and runs MC3."
        ),
        "large_videomae_required": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
