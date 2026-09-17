from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p86_cached_motion_data import MOTION_FIELDS
from p87s_deploy_model import load_p87s_deploy_checkpoint
from p87s_test_data import (
    P87STestCachedSequenceMotionDataset,
    P87STestPixelDataset,
    collate_p87s_test,
)
from train_p86_mobind_fusion_proxy import model_forward


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "runs/p87s_test_adapt_structured12_v1/unified_student.pt"
DEFAULT_SEQUENCE = PROJECT_DIR / "runs/p87s_test_mc3_sequence_v1"
DEFAULT_MOTION = PROJECT_DIR / "runs/p87s_test_motion_window_t16_v1"
DEFAULT_PIXELS = PROJECT_DIR / "runs/p87s_test_pixel_cache_t16_r160_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p87s_deployment_equivalence_v1"
DEFAULT_INDICES = (0, 11, 13, 53, 153, 193, 267, 404)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that raw V+S+I inference and the optional Student sequence-cache "
            "acceleration produce equivalent final logits."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sequence-cache", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--indices", type=int, nargs="*", default=list(DEFAULT_INDICES))
    parser.add_argument("--max-logit-absolute-error", type=float, default=0.05)
    return parser.parse_args()


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    indices = np.asarray(args.indices, dtype=np.int64)
    if not len(indices) or np.any(indices < 0) or np.any(indices >= 405):
        raise ValueError("deployment-equivalence indices must be within [0, 404]")
    raw = P87STestPixelDataset(args.pixel_cache, indices=indices)
    cached = P87STestCachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        indices=indices,
        temporal_augment=False,
    )
    raw_batch = collate_p87s_test([raw[index] for index in range(len(raw))])
    cached_batch = collate_p87s_test(
        [cached[index] for index in range(len(cached))]
    )
    if raw_batch["sample_id"] != cached_batch["sample_id"]:
        raise RuntimeError("raw and cached Student row orders differ")

    model, checkpoint = load_p87s_deploy_checkpoint(args.checkpoint.resolve())
    if checkpoint.get("stage") not in {
        "P87S_label_free_test_adaptation",
        "P162_P150_distilled_deployment",
    }:
        raise ValueError("equivalence audit requires a self-contained adapted Student")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    raw_batch = to_device(raw_batch, device)
    cached_batch = to_device(cached_batch, device)
    motion = {field: cached_batch[field] for field in MOTION_FIELDS}
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        fresh_sequence = model.visual.encode_backbone_sequence(raw_batch["images"])
        raw_output = model.forward_from_backbone_sequence(
            fresh_sequence,
            raw_batch["view_valid"],
            raw_batch["view_quality"],
            raw_batch["global_time_position"],
            motion,
        )["logits"]
        cached_output = model_forward(model, cached_batch)["logits"]
    fresh_sequence = fresh_sequence.float().cpu().numpy()
    cached_sequence = cached_batch["backbone_sequence"].float().cpu().numpy()
    raw_logits = raw_output.float().cpu().numpy()
    cached_logits = cached_output.float().cpu().numpy()
    sequence_error = np.abs(fresh_sequence - cached_sequence)
    logit_error = np.abs(raw_logits - cached_logits)
    raw_prediction = raw_logits.argmax(axis=1)
    cached_prediction = cached_logits.argmax(axis=1)
    visual_missing = ~raw_batch["view_valid"].cpu().numpy().any(axis=(1, 2, 3))
    summary = {
        "stage": "P87S_raw_vs_cached_deployment_equivalence",
        "sample_rows": len(indices),
        "indices": indices.tolist(),
        "sample_ids": list(raw_batch["sample_id"]),
        "visual_missing_rows_in_sample": int(visual_missing.sum()),
        "sequence_max_absolute_error": float(sequence_error.max()),
        "sequence_mean_absolute_error": float(sequence_error.mean()),
        "logit_max_absolute_error": float(logit_error.max()),
        "logit_mean_absolute_error": float(logit_error.mean()),
        "prediction_agreement": float(np.mean(raw_prediction == cached_prediction)),
        "max_logit_absolute_error_contract": args.max_logit_absolute_error,
        "status": "passed",
        "interpretation": (
            "The sequence cache is a deterministic, rebuildable acceleration of the "
            "deployed Student backbone. It contains no labels or teacher features and "
            "is not required by the self-contained checkpoint."
        ),
    }
    if summary["logit_max_absolute_error"] > args.max_logit_absolute_error:
        raise RuntimeError(json.dumps(summary, ensure_ascii=False))
    if summary["prediction_agreement"] != 1.0:
        raise RuntimeError(json.dumps(summary, ensure_ascii=False))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
