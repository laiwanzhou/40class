from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from adapt_p87s_structured_student import model_build_args, resolve_config_path
from p86_cached_motion_data import (
    P86CachedSequenceMotionDataset,
    TEMPORAL_MOTION_FIELDS,
    collate_p86_cached_motion,
)
from train_p86_mobind_fusion_proxy import build_model, metric_dict, model_forward


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache deterministic temporal-TTA logits from a frozen P87-S run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def temporal_indices(steps: int, scale: float, shift: float) -> np.ndarray:
    base = np.arange(steps, dtype=np.float32)
    center = 0.5 * (steps - 1)
    positions = np.clip((base - center) * scale + center + shift, 0.0, steps - 1.0)
    return np.rint(positions).astype(np.int64)


class TemporalView(Dataset[dict[str, Any]]):
    def __init__(self, base: P86CachedSequenceMotionDataset, indices: np.ndarray) -> None:
        self.base = base
        self.indices = torch.from_numpy(indices.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        temporal_index = self.indices
        sequence_key = (
            "backbone_region_sequence"
            if "backbone_region_sequence" in item
            else "backbone_sequence"
        )
        item[sequence_key] = torch.index_select(item[sequence_key], 2, temporal_index)
        for field in ("view_valid", "view_quality", "global_time_position"):
            item[field] = torch.index_select(item[field], 1, temporal_index)
        for field in TEMPORAL_MOTION_FIELDS:
            item[field] = torch.index_select(item[field], 1, temporal_index)
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
    dataset_kwargs = {
        "sequence_cache": sequence_cache,
        "motion_cache": motion_cache,
        "pixel_cache": pixel_cache,
        "teacher_features": resolve_config_path(config["teacher_features"], repository_root),
        "teacher_logits": resolve_config_path(config["teacher_logits"], repository_root),
        "temporal_augment": False,
        "imu_teacher_logits": (
            resolve_config_path(config["imu_teacher_logits"], repository_root)
            if config.get("imu_teacher_logits") else None
        ),
        "imu_event_features": (
            resolve_config_path(config["imu_event_features"], repository_root)
            if config.get("imu_event_features") else None
        ),
    }
    full = P86CachedSequenceMotionDataset(**dataset_kwargs)
    import csv

    with (run_dir / "subject_holdout_predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reference_rows = list(csv.DictReader(handle))
    sample_ids = [row["sample_id"] for row in reference_rows]
    selected_indices = np.asarray([full.index_lookup[value] for value in sample_ids])
    selected = P86CachedSequenceMotionDataset(
        **dataset_kwargs, indices=selected_indices
    )
    steps = int(selected.backbone_sequence.shape[3]) if selected.uses_spatial_regions else int(selected.backbone_sequence.shape[3])
    variants = {
        "original": (1.0, 0.0),
        "scale_086": (0.86, 0.0),
        "scale_114": (1.14, 0.0),
        "shift_m1": (1.0, -1.0),
        "shift_p1": (1.0, 1.0),
        "scale_093": (0.93, 0.0),
        "scale_107": (1.07, 0.0),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    cached: dict[str, np.ndarray] = {}
    labels: np.ndarray | None = None
    users = [row["user_id"] for row in reference_rows]
    variant_metrics = {}
    with torch.inference_mode():
        for name, (scale, shift) in variants.items():
            indices = temporal_indices(steps, scale, shift)
            loader = DataLoader(
                TemporalView(selected, indices), batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers,
                persistent_workers=args.workers > 0, pin_memory=True,
                collate_fn=collate_p86_cached_motion,
            )
            rows = []
            label_rows = []
            seen_ids: list[str] = []
            for batch in loader:
                batch = to_device(batch, device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = model_forward(model, batch)["logits"]
                rows.append(logits.float().cpu().numpy())
                label_rows.append(batch["label"].cpu().numpy())
                seen_ids.extend(map(str, batch["sample_id"]))
            if seen_ids != sample_ids:
                raise RuntimeError("temporal TTA row order changed")
            logits_array = np.concatenate(rows).astype(np.float32)
            current_labels = np.concatenate(label_rows).astype(np.int64)
            if labels is None:
                labels = current_labels
            elif not np.array_equal(labels, current_labels):
                raise RuntimeError("temporal TTA labels changed")
            cached[name] = logits_array
            variant_metrics[name] = {
                "indices": indices.tolist(),
                "metrics": metric_dict(labels, logits_array.argmax(axis=1), users),
            }
            print(json.dumps({"variant": name, **variant_metrics[name]["metrics"]}), flush=True)

    assert labels is not None
    saved_logits = np.asarray(np.load(run_dir / "subject_holdout_logits.npy"), dtype=np.float32)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "temporal_tta_logits.npz",
        sample_ids=np.asarray(sample_ids), labels=labels, **cached,
    )
    summary = {
        "stage": "P88_temporal_TTA_cache", "status": "complete",
        "run_dir": str(run_dir), "rows": len(sample_ids), "steps": steps,
        "variants": variant_metrics,
        "original_vs_saved_top1_agreement": float(
            np.mean(cached["original"].argmax(axis=1) == saved_logits.argmax(axis=1))
        ),
        "original_vs_saved_max_abs_logit_difference": float(
            np.max(np.abs(cached["original"] - saved_logits))
        ),
        "output": str(output / "temporal_tta_logits.npz"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
