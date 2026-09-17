"""Frozen EgoVLP EgoClip/EgoNCE video teacher on P86 grayscale clips."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PIXELS = HERE / "runs/p86_visual_pixel_cache_t16_r160_v12"
EXTERNAL = PROJECT.parent / "external_data/EgoVLP/EgoVLP-main"
CHECKPOINT = PROJECT.parent / "external_data/EgoVLP/EgoVLP_PT_BEST.pth"
OUTPUT = HERE / "runs/p131_egovlp_teacher_v1"
A18 = PROJECT / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
VIEWS = ("scene", "person", "workspace")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        output[key] = value
    return output


def load_video_model(checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(EXTERNAL))
    from model.video_transformer import SpaceTimeTransformer

    # The official artifact pickles a ConfigParser containing PosixPath values.
    # Keep the compatibility shim local so Windows can deserialize it unchanged.
    original_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original_posix_path
    state = clean_state_dict(checkpoint["state_dict"])
    temporal_key = "video_model.temporal_embed"
    if temporal_key not in state:
        raise RuntimeError("EgoVLP checkpoint has no temporal embedding")
    num_frames = int(state[temporal_key].shape[1])
    model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init="zeros",
        attention_style="frozen-in-time",
    )
    model.head = nn.Identity()
    model.pre_logits = nn.Identity()
    video_state = {
        key[len("video_model.") :]: value
        for key, value in state.items()
        if key.startswith("video_model.")
    }
    missing, unexpected = model.load_state_dict(video_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"EgoVLP video state mismatch missing={missing[:6]} unexpected={unexpected[:6]}"
        )
    projection = nn.Linear(768, 256)
    projection.load_state_dict(
        {
            "weight": state["vid_proj.0.weight"],
            "bias": state["vid_proj.0.bias"],
        },
        strict=True,
    )
    model.to(device).eval()
    projection.to(device).eval()
    for parameter in list(model.parameters()) + list(projection.parameters()):
        parameter.requires_grad_(False)
    return model, projection, checkpoint, num_frames


def prepare_video(
    values: np.ndarray, device: torch.device, num_frames: int
) -> torch.Tensor:
    # P86: B, 2 windows, 16 frames, H, W. Uniformly sample checkpoint frame count.
    values = values.reshape(len(values), 32, values.shape[-2], values.shape[-1])
    indices = np.rint(np.linspace(0, 31, num_frames)).astype(np.int64)
    values = values[:, indices]
    video = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    video = video[:, :, None].repeat(1, 1, 3, 1, 1) / 255.0
    batch, frames, channels, height, width = video.shape
    video = F.interpolate(
        video.reshape(batch * frames, channels, height, width),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, frames, channels, 224, 224)
    mean = torch.tensor(MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 1, 3, 1, 1)
    return (video - mean) / std


def build_cache(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.pixel_cache / "rows.csv")
    images = np.load(args.pixel_cache / "images.npy", mmap_mode="r")
    raw = np.lib.format.open_memmap(
        args.output_dir / "raw_features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 3, 768),
    )
    projected = np.lib.format.open_memmap(
        args.output_dir / "projected_features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 3, 256),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, projection, checkpoint, num_frames = load_video_model(
        args.checkpoint, device
    )
    records = [(row, view) for row in range(len(rows)) for view in range(3)]
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            values = np.stack([images[row, :, :, view] for row, view in batch_records])
            feature = model(prepare_video(values, device, num_frames)).float()
            projected_value = projection(feature)
            feature = feature.cpu().numpy().astype(np.float16)
            projected_value = projected_value.cpu().numpy().astype(np.float16)
            for index, (row, view) in enumerate(batch_records):
                raw[row, view] = feature[index]
                projected[row, view] = projected_value[index]
            if start % (args.batch_size * 20) == 0:
                print(
                    json.dumps(
                        {
                            "stage": "P131_cache",
                            "encoded": min(start + len(batch_records), len(records)),
                            "total": len(records),
                        }
                    ),
                    flush=True,
                )
    raw.flush()
    projected.flush()
    report = {
        "stage": "P131_EgoVLP_frozen_cache",
        "rows": len(rows),
        "views": list(VIEWS),
        "raw_shape": list(raw.shape),
        "projected_shape": list(projected.shape),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_num_frames": num_frames,
        "pretraining": "EgoClip with EgoNCE",
        "labels_used": False,
        "test_rows_loaded": 0,
    }
    (args.output_dir / "cache_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def l2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(labels)),
        "correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def build_oof(args: argparse.Namespace) -> dict:
    rows = read_rows(args.pixel_cache / "rows.csv")
    sample_ids = np.asarray([row["sample_id"] for row in rows], dtype=str)
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    reference = np.load(A18)
    lookup = {value: index for index, value in enumerate(reference["sample_ids"].astype(str))}
    positions = np.asarray([lookup[value] for value in sample_ids], dtype=np.int64)
    fold_ids = reference["fold_ids"][positions].astype(np.int64)
    raw = np.asarray(np.load(args.output_dir / "raw_features.npy", mmap_mode="r"), dtype=np.float32)
    projected = np.asarray(
        np.load(args.output_dir / "projected_features.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    variants = {
        "workspace_raw": l2(raw[:, 2]),
        "all_raw": l2(raw.reshape(len(raw), -1)),
        "all_projected": l2(projected.reshape(len(projected), -1)),
        "all_raw_projected": np.concatenate(
            (l2(raw.reshape(len(raw), -1)), l2(projected.reshape(len(projected), -1))),
            axis=1,
        ).astype(np.float32),
    }
    saved: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "labels": labels,
        "fold_ids": fold_ids,
    }
    result = {}
    for name, values in variants.items():
        logits = np.zeros((len(labels), 40), dtype=np.float64)
        folds = []
        for fold in range(3):
            held = fold_ids == fold
            model = make_pipeline(
                StandardScaler(),
                RidgeClassifier(
                    alpha=args.alpha,
                    class_weight="balanced",
                    solver="lsqr",
                    tol=1e-4,
                ),
            )
            model.fit(values[~held], labels[~held])
            scores = np.asarray(model.decision_function(values[held]), dtype=np.float64)
            classes = model.named_steps["ridgeclassifier"].classes_.astype(np.int64)
            held_rows = np.flatnonzero(held)
            logits[held_rows[:, None], classes[None, :]] = scores
            folds.append({"fold": fold, **metrics(labels[held], logits[held].argmax(1))})
        prediction = logits.argmax(1)
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        result[name] = {
            "feature_dim": int(values.shape[1]),
            "alpha": args.alpha,
            "metrics": metrics(labels, prediction),
            "folds": folds,
        }
        saved[f"{name}_logits"] = logits.astype(np.float32)
        saved[f"{name}_probability"] = probability.astype(np.float32)
    report = {
        "stage": "P131_EgoVLP_complete_subject_safe_OOF",
        "status": "complete",
        "protocol": {
            "backbone_frozen": True,
            "primary_variant": "all_raw_projected",
            "alpha_fixed": args.alpha,
            "held_fold_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "variants": result,
    }
    np.savez_compressed(args.output_dir / "oof_predictions.npz", **saved)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "cache", "oof"), default="all")
    parser.add_argument("--pixel-cache", type=Path, default=PIXELS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=3000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pixel_cache = args.pixel_cache.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.stage in ("all", "cache"):
        build_cache(args)
    if args.stage in ("all", "oof"):
        build_oof(args)


if __name__ == "__main__":
    main()
