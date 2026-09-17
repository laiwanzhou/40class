"""Frozen official LaViLa TimeSformer-B video teacher and strict OOF heads."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
EXTERNAL = PROJECT.parent / "external_data/LaViLa/LaViLa-main"
CHECKPOINT = (
    PROJECT.parent
    / "external_data/LaViLa/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth"
)
OUTPUT = HERE / "runs/p155_lavila_timesformer_teacher_v1"
A18 = PROJECT / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
VIEWS = ("scene", "person", "workspace")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path):
    original = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return checkpoint, cleaned


def resize_temporal_embedding(value: torch.Tensor, frames: int) -> torch.Tensor:
    if value.shape[1] == frames:
        return value
    return F.interpolate(
        value.permute(0, 2, 1),
        size=frames,
        mode="linear",
        align_corners=False,
    ).permute(0, 2, 1)


def load_visual(path: Path, device: torch.device, frames: int):
    sys.path.insert(0, str(EXTERNAL))
    from lavila.models.openai_model import QuickGELU
    from lavila.models.timesformer import SpaceTimeTransformer

    checkpoint, state = load_checkpoint(path)
    temporal_key = "visual.temporal_embed"
    if temporal_key not in state:
        matching = [key for key in state if key.endswith("visual.temporal_embed")]
        raise RuntimeError(f"checkpoint lacks visual temporal embedding: {matching}")
    checkpoint_frames = int(state[temporal_key].shape[1])
    visual = SpaceTimeTransformer(
        num_frames=frames,
        time_init="zeros",
        attention_style="frozen-in-time",
        ln_pre=True,
        act_layer=QuickGELU,
    )
    visual.head = nn.Identity()
    visual.pre_logits = nn.Identity()
    visual_state = {
        key[len("visual.") :]: value
        for key, value in state.items()
        if key.startswith("visual.")
    }
    visual_state["temporal_embed"] = resize_temporal_embedding(
        visual_state["temporal_embed"], frames
    )
    missing, unexpected = visual.load_state_dict(visual_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"LaViLa visual mismatch missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    projection = None
    if "image_projection" in state:
        projection = nn.Linear(768, int(state["image_projection"].shape[1]), bias=False)
        projection.weight.data.copy_(state["image_projection"].T)
    visual.to(device).eval()
    if projection is not None:
        projection.to(device).eval()
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    if projection is not None:
        for parameter in projection.parameters():
            parameter.requires_grad_(False)
    return visual, projection, checkpoint, checkpoint_frames


def prepare_video(values: np.ndarray, device: torch.device, frames: int) -> torch.Tensor:
    values = values.reshape(len(values), 32, values.shape[-2], values.shape[-1])
    indices = np.rint(np.linspace(0, 31, frames)).astype(np.int64)
    values = values[:, indices]
    video = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    video = video[:, None].repeat(1, 3, 1, 1, 1) / 255.0
    batch, channels, frame_count, height, width = video.shape
    video = F.interpolate(
        video.permute(0, 2, 1, 3, 4).reshape(
            batch * frame_count, channels, height, width
        ),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, frame_count, channels, 224, 224).permute(0, 2, 1, 3, 4)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1, 1)
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
    visual, projection, checkpoint, checkpoint_frames = load_visual(
        args.checkpoint, device, args.frames
    )
    records = [(row, view) for row in range(len(rows)) for view in range(3)]
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            values = np.stack([images[row, :, :, view] for row, view in batch_records])
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.float16
            ):
                feature = visual(
                    prepare_video(values, device, args.frames)
                ).float()
                projected_value = (
                    projection(feature) if projection is not None else feature[:, :256]
                )
            feature_numpy = feature.cpu().numpy().astype(np.float16)
            projected_numpy = projected_value.cpu().numpy().astype(np.float16)
            for index, (row, view) in enumerate(batch_records):
                raw[row, view] = feature_numpy[index]
                projected[row, view] = projected_numpy[index]
            if start % (args.batch_size * 20) == 0:
                print(
                    json.dumps(
                        {
                            "stage": "P155_cache",
                            "encoded": min(start + len(batch_records), len(records)),
                            "total": len(records),
                        }
                    ),
                    flush=True,
                )
    raw.flush()
    projected.flush()
    report = {
        "stage": "P155_LaViLa_frozen_cache",
        "rows": len(rows),
        "views": list(VIEWS),
        "frames": args.frames,
        "checkpoint_frames": checkpoint_frames,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "projection_loaded": projection is not None,
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
    raw = np.asarray(
        np.load(args.output_dir / "raw_features.npy", mmap_mode="r"), dtype=np.float32
    )
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
    payload = {"sample_ids": sample_ids, "labels": labels, "fold_ids": fold_ids}
    results = {}
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
            rows_held = np.flatnonzero(held)
            logits[rows_held[:, None], classes[None, :]] = scores
            folds.append({"fold": fold, **metrics(labels[held], logits[held].argmax(1))})
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        prediction = probability.argmax(axis=1)
        results[name] = {
            "feature_dim": int(values.shape[1]),
            "alpha": args.alpha,
            "metrics": metrics(labels, prediction),
            "folds": folds,
        }
        payload[f"{name}_logits"] = logits.astype(np.float32)
        payload[f"{name}_probability"] = probability.astype(np.float32)
    report = {
        "stage": "P155_LaViLa_complete_subject_safe_OOF",
        "status": "complete",
        "protocol": {
            "backbone_frozen": True,
            "frames_fixed": args.frames,
            "alpha_fixed": args.alpha,
            "held_fold_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "variants": results,
    }
    np.savez_compressed(args.output_dir / "oof_predictions.npz", **payload)
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
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
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
