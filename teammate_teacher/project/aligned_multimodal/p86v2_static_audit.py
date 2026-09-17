from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from p86v2_metrics import emission_metrics
from p86v2_protocol import (
    DEFAULT_PROTOCOL,
    assert_no_forbidden_path,
    build_split,
    load_protocol,
    read_rows,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PIXEL_CACHE = PROJECT_DIR / "runs/p86_visual_pixel_cache_t16_r160_v12"
DEFAULT_TEACHER = (
    PROJECT_DIR
    / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p86v2_static_audit_v1"
VIEW_NAMES = ("scene", "person", "workspace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P86-v2 Train-only static bottleneck audit")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXEL_CACHE)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pixel-samples", type=int, default=128)
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(values.max()),
    }


def parameter_group(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "visual" and len(parts) > 2:
        if parts[1] in {"stem", "layer1", "layer2", "layer3", "layer4"}:
            return ".".join(parts[:2])
        return ".".join(parts[:2])
    if parts[0] in {"stem", "layer1", "layer2", "layer3", "layer4"}:
        return parts[0]
    if parts[0] == "motion_residual" and len(parts) > 1:
        return ".".join(parts[:2])
    return parts[0]


def checkpoint_audit(path: Path) -> dict:
    checkpoint = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    groups: dict[str, int] = {}
    total = 0
    for name, value in state.items():
        count = int(value.numel())
        total += count
        key = parameter_group(name)
        groups[key] = groups.get(key, 0) + count
    return {
        "path": str(path.resolve()),
        "checkpoint_bytes": path.stat().st_size,
        "parameters": total,
        "fp32_parameter_bytes": total * 4,
        "groups": {
            key: {
                "parameters": count,
                "fraction": count / total,
                "fp32_bytes": count * 4,
            }
            for key, count in sorted(groups.items(), key=lambda item: -item[1])
        },
    }


def temporal_sampling_audit(source_indices: np.ndarray) -> dict:
    unique_count = np.asarray(
        [len(np.unique(sample.reshape(-1))) for sample in source_indices], dtype=np.int64
    )
    overlap = np.asarray(
        [
            len(set(sample[0].tolist()) & set(sample[1].tolist()))
            for sample in source_indices
        ],
        dtype=np.int64,
    )
    repeated_slots = 32 - unique_count
    frame_count = source_indices[:, 1, -1] + 1
    normalized_max_gap = []
    for sample, count in zip(source_indices, frame_count, strict=True):
        chosen = np.unique(sample.reshape(-1))
        gaps = np.diff(chosen) / max(int(count) - 1, 1)
        normalized_max_gap.append(float(gaps.max()) if len(gaps) else 0.0)
    return {
        "source_frames": distribution(frame_count),
        "unique_source_frames_of_32_slots": distribution(unique_count),
        "repeated_slots_of_32": distribution(repeated_slots),
        "early_late_exact_overlap": distribution(overlap),
        "normalized_largest_unsampled_gap": distribution(np.asarray(normalized_max_gap)),
        "samples_with_any_repeated_slot": int((repeated_slots > 0).sum()),
        "samples_with_at_least_four_repeated_slots": int((repeated_slots >= 4).sum()),
    }


def pixel_information_audit(
    images: np.ndarray,
    allowed_indices: np.ndarray,
    maximum_samples: int,
) -> dict:
    if maximum_samples < 1:
        raise ValueError("pixel_samples must be positive")
    positions = np.linspace(
        0, len(allowed_indices) - 1, min(maximum_samples, len(allowed_indices)), dtype=np.int64
    )
    chosen = allowed_indices[positions]
    temporal_energy = {name: [] for name in VIEW_NAMES}
    contrast = {name: [] for name in VIEW_NAMES}
    view_correlation = {"scene_person": [], "scene_workspace": [], "person_workspace": []}
    pairs = ((0, 1, "scene_person"), (0, 2, "scene_workspace"), (1, 2, "person_workspace"))
    for index in chosen:
        # Striding before conversion keeps each read small while preserving coarse motion/layout.
        value = np.asarray(images[int(index), :, :, :, ::4, ::4], dtype=np.float32) / 255.0
        for view_index, view_name in enumerate(VIEW_NAMES):
            frames = value[:, :, view_index]
            temporal_energy[view_name].append(float(np.abs(np.diff(frames, axis=1)).mean()))
            contrast[view_name].append(float(frames.std()))
        for left, right, name in pairs:
            a = value[:, :, left].reshape(-1)
            b = value[:, :, right].reshape(-1)
            a = a - a.mean()
            b = b - b.mean()
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            view_correlation[name].append(float(np.dot(a, b) / denominator) if denominator else 0.0)
    return {
        "sample_count": int(len(chosen)),
        "coarse_temporal_absolute_difference": {
            key: distribution(np.asarray(value)) for key, value in temporal_energy.items()
        },
        "coarse_pixel_contrast": {
            key: distribution(np.asarray(value)) for key, value in contrast.items()
        },
        "same_slot_view_correlation": {
            key: distribution(np.asarray(value)) for key, value in view_correlation.items()
        },
    }


def teacher_development_audit(path: Path, holdout_ids: set[str]) -> dict:
    assert_no_forbidden_path(path)
    with np.load(path.resolve(), allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        selected = np.flatnonzero(np.isin(sample_ids, sorted(holdout_ids)))
        if len(selected) != len(holdout_ids):
            raise RuntimeError("teacher logits do not cover the development holdout exactly")
        users = np.asarray(data["users"])[selected].astype(str)
        labels = np.asarray(data["labels"], dtype=np.int64)[selected]
        result = {}
        for key in ("early_logits", "late_logits", "window_mean_logits", "early_late_logits"):
            result[key] = emission_metrics(np.asarray(data[key])[selected], labels, users)
        return result


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    cache = args.pixel_cache.resolve()
    assert_no_forbidden_path(cache)
    rows = read_rows(cache / "rows.csv")
    development = build_split(rows, "development", protocol)
    confirmation = build_split(rows, "confirmation", protocol)
    allowed_indices = np.asarray(
        sorted(development.training_indices + development.holdout_indices), dtype=np.int64
    )

    source = np.load(cache / "source_frame_indices.npy", mmap_mode="r")
    valid = np.load(cache / "view_valid.npy", mmap_mode="r")
    quality = np.load(cache / "view_quality.npy", mmap_mode="r")
    images = np.load(cache / "images.npy", mmap_mode="r")
    if len(rows) != len(source) or len(rows) != len(images):
        raise RuntimeError("pixel cache arrays and rows differ")
    allowed_source = np.asarray(source[allowed_indices])
    allowed_valid = np.asarray(valid[allowed_indices], dtype=bool)
    allowed_quality = np.asarray(quality[allowed_indices], dtype=np.float32)
    view_audit = {}
    for view_index, name in enumerate(VIEW_NAMES):
        mask = allowed_valid[..., view_index]
        view_audit[name] = {
            "valid_fraction": float(mask.mean()),
            "quality_when_valid": distribution(allowed_quality[..., view_index][mask]),
        }

    development_ids = {rows[index]["sample_id"] for index in development.holdout_indices}
    checkpoints = {
        "visual_proxy_anchor": checkpoint_audit(
            PROJECT_DIR
            / "runs/p86_visual_mc3_temporal_t16_r160_layer2_proxy_v13/visual_student.pt"
        ),
        "unified_proxy_anchor": checkpoint_audit(
            PROJECT_DIR / "runs/p86_mobind_fusion_separate_v1_paired/unified_student.pt"
        ),
    }
    result = {
        "stage": "P86-v2",
        "status": "static_train_only_audit",
        "protocol": {
            "path": str(args.protocol.resolve()),
            "development_name": development.name,
            "development_train_samples": len(development.training_indices),
            "development_holdout_samples": len(development.holdout_indices),
            "confirmation_name": confirmation.name,
            "confirmation_holdout_samples": len(confirmation.holdout_indices),
            "embargo_samples": len(development.embargo_indices),
            "sample_fingerprint": development.sample_fingerprint,
        },
        "temporal_sampling": temporal_sampling_audit(allowed_source),
        "views": view_audit,
        "pixels": pixel_information_audit(images, allowed_indices, args.pixel_samples),
        "teacher_development_oof": teacher_development_audit(
            args.teacher_logits, development_ids
        ),
        "parameter_allocation": checkpoints,
        "confirmation_predictions_read": False,
        "embargo_predictions_or_metrics_read": False,
        "kaggle_test_used": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
