from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapt_p87s_structured_student import model_build_args
from p86_cached_motion_data import (
    P86CachedSequenceMotionDataset,
    collate_p86_cached_motion,
)
from p87s_deploy_model import load_p87s_deploy_checkpoint
from train_p86_mobind_fusion_proxy import (
    DEFAULT_MOTION,
    DEFAULT_PIXELS,
    DEFAULT_TEACHER_FEATURES,
    DEFAULT_TEACHER_LOGITS,
    build_model,
    model_forward,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "runs/p87s_fusion_holdout1_c7_structured12_v1/unified_student.pt"
DEFAULT_SEQUENCE = PROJECT_DIR / "runs/p87s_mc3_sequence_holdout1_v1"
DEFAULT_DEPTH = PROJECT_DIR / "runs/p88_depth_sequence_holdout1_v1"
DEFAULT_REFERENCE = PROJECT_DIR / "runs/p87s_fusion_holdout1_c7_structured12_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_depth_features_holdout1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen P87 anchor and registered-Depth embeddings for P88."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sequence-cache", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--depth-cache", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "source_id", "user_id", "class_id")
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def validity_statistics(values: torch.Tensor) -> torch.Tensor:
    # [B,2,T,3] -> early/late x view mean and std = 12 deployable scalars.
    if values.ndim != 4 or values.shape[1] != 2 or values.shape[3] != 3:
        raise ValueError("unexpected P88 Depth validity geometry")
    mean = values.mean(dim=2).flatten(1)
    std = values.std(dim=2, unbiased=False).flatten(1)
    return torch.cat((mean, std), dim=1)


def load_frozen_p87(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "deployment_model_config" in checkpoint:
        return load_p87s_deploy_checkpoint(path)[0]
    if checkpoint.get("stage") != "P87S_label_free_adaptation":
        raise ValueError("unsupported frozen P87 checkpoint format")
    base_path = Path(str(checkpoint["base_checkpoint"])).resolve()
    base_summary = json.loads(
        (base_path.parent / "summary.json").read_text(encoding="utf-8")
    )
    build_args = model_build_args(base_path, base_summary)
    model, _, _ = build_model(build_args)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = P86CachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        temporal_augment=False,
    )
    depth_rows = read_rows(args.depth_cache.resolve() / "rows.csv")
    if [row["sample_id"] for row in dataset.rows] != [row["sample_id"] for row in depth_rows]:
        raise RuntimeError("P88 Depth and P87 row orders differ")
    depth_sequence = np.load(
        args.depth_cache.resolve() / "backbone_sequence_fp16.npy", mmap_mode="r"
    )
    depth_valid = np.load(
        args.depth_cache.resolve() / "depth_valid_fraction_fp16.npy", mmap_mode="r"
    )
    depth_completed = np.load(
        args.depth_cache.resolve() / "completed.npy", mmap_mode="r"
    )
    if not np.asarray(depth_completed).all():
        raise RuntimeError("P88 Depth sequence cache is incomplete")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate_p86_cached_motion,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_p87(checkpoint)
    model.eval().to(device)
    anchor_logits = np.zeros((len(dataset), 40), dtype=np.float32)
    anchor_embedding = np.zeros((len(dataset), 512), dtype=np.float16)
    depth_embedding = np.zeros((len(dataset), 512), dtype=np.float16)
    depth_logits = np.zeros((len(dataset), 40), dtype=np.float16)
    valid_statistics = np.zeros((len(dataset), 12), dtype=np.float16)
    started = time.perf_counter()
    emitted = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            indices = np.asarray(batch["cache_index"], dtype=np.int64)
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            depth_value = torch.from_numpy(
                np.asarray(depth_sequence[indices], dtype=np.float32).copy()
            ).to(device, non_blocking=True)
            valid_fraction = torch.from_numpy(
                np.asarray(depth_valid[indices], dtype=np.float32).copy()
            ).to(device, non_blocking=True)
            depth_view_valid = batch["view_valid"] & (valid_fraction >= 0.02)
            depth_quality = batch["view_quality"] * torch.sqrt(
                valid_fraction.clamp_min(0.0)
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                anchor = model_forward(model, batch)
                depth_output = model.visual.forward_from_backbone_sequence(
                    depth_value,
                    depth_view_valid,
                    depth_quality,
                    batch["global_time_position"],
                )
            anchor_logits[indices] = anchor["logits"].float().cpu().numpy()
            anchor_embedding[indices] = (
                anchor["visual_embedding"].to(dtype=torch.float16).cpu().numpy()
            )
            depth_embedding[indices] = (
                depth_output["visual_embedding"].to(dtype=torch.float16).cpu().numpy()
            )
            depth_logits[indices] = depth_output["logits"].to(dtype=torch.float16).cpu().numpy()
            valid_statistics[indices] = (
                validity_statistics(valid_fraction).to(dtype=torch.float16).cpu().numpy()
            )
            emitted += len(indices)
            if emitted % 512 < len(indices):
                print(
                    json.dumps(
                        {
                            "features": emitted,
                            "total": len(dataset),
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
    if emitted != len(dataset):
        raise RuntimeError("P88 feature extraction stopped before the full universe")
    np.save(output / "anchor_logits.npy", anchor_logits)
    np.save(output / "anchor_embedding_fp16.npy", anchor_embedding)
    np.save(output / "depth_embedding_fp16.npy", depth_embedding)
    np.save(output / "depth_logits_fp16.npy", depth_logits)
    np.save(output / "depth_valid_statistics_fp16.npy", valid_statistics)
    write_rows(output / "rows.csv", dataset.rows)

    reference_rows = read_rows(args.reference_run.resolve() / "subject_holdout_predictions.csv")
    reference_logits = np.load(
        args.reference_run.resolve() / "subject_holdout_logits.npy", allow_pickle=False
    )
    lookup = {row["sample_id"]: index for index, row in enumerate(dataset.rows)}
    aligned_indices = np.asarray([lookup[row["sample_id"]] for row in reference_rows])
    regenerated = anchor_logits[aligned_indices]
    maximum_difference = float(np.max(np.abs(regenerated - reference_logits)))
    top1_agreement = float(
        np.mean(regenerated.argmax(axis=1) == reference_logits.argmax(axis=1))
    )
    if top1_agreement != 1.0 or maximum_difference > 0.03:
        raise RuntimeError(
            "P88 regenerated anchor differs from frozen P87 evidence: "
            f"top1={top1_agreement}, max={maximum_difference}"
        )
    summary = {
        "stage": "P88_frozen_anchor_registered_depth_feature_cache",
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "p87_checkpoint_modified": False,
        "rows": len(dataset),
        "features": {
            "anchor_logits": list(anchor_logits.shape),
            "anchor_embedding": list(anchor_embedding.shape),
            "depth_embedding": list(depth_embedding.shape),
            "depth_logits": list(depth_logits.shape),
            "depth_valid_statistics": list(valid_statistics.shape),
        },
        "anchor_equivalence": {
            "reference_run": str(args.reference_run.resolve()),
            "rows": len(reference_rows),
            "maximum_absolute_logit_difference": maximum_difference,
            "top1_agreement": top1_agreement,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
