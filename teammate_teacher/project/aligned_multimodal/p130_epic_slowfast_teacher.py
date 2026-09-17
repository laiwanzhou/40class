"""Frozen EPIC-KITCHENS-100 SlowFast R50 feature teacher on P86 clips."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PIXELS = HERE / "runs/p86_visual_pixel_cache_t16_r160_v12"
EXTERNAL = PROJECT.parent / "external_data/epic-kitchens-slowfast-master"
CHECKPOINT = EXTERNAL / "SlowFast.pyth"
CONFIG = EXTERNAL / "configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml"
OUTPUT = HERE / "runs/p130_epic_slowfast_teacher_v1"
A18 = PROJECT / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)
VIEWS = ("scene", "person", "workspace")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_model(checkpoint_path: Path, config_path: Path, device: torch.device):
    detectron2 = types.ModuleType("detectron2")
    layers = types.ModuleType("detectron2.layers")
    layers.ROIAlign = torch.nn.Identity
    detectron2.layers = layers
    sys.modules.setdefault("detectron2", detectron2)
    sys.modules.setdefault("detectron2.layers", layers)
    sys.path.insert(0, str(EXTERNAL))
    from slowfast.config.defaults import get_cfg
    from slowfast.models import build_model

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.NUM_GPUS = 0
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"EPIC checkpoint mismatch missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg, checkpoint


def prepare_video(values: np.ndarray, device: torch.device) -> list[torch.Tensor]:
    # values: B, 2 windows, 16 frames, H, W -> B, C, 32, 224, 224
    video = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    video = video.reshape(len(video), 32, 1, video.shape[-2], video.shape[-1])
    video = video.permute(0, 2, 1, 3, 4).repeat(1, 3, 1, 1, 1) / 255.0
    batch, channel, frames, height, width = video.shape
    video = F.interpolate(
        video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channel, height, width),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, frames, channel, 224, 224).permute(0, 2, 1, 3, 4)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1, 1)
    fast = (video - mean) / std
    slow = fast[:, :, ::4]
    return [slow, fast]


def build_cache(args: argparse.Namespace) -> dict:
    if args.checkpoint.stat().st_size != 276_771_081:
        raise RuntimeError(
            f"EPIC checkpoint incomplete: {args.checkpoint.stat().st_size}/276771081"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.pixel_cache / "rows.csv")
    images = np.load(args.pixel_cache / "images.npy", mmap_mode="r")
    features = np.lib.format.open_memmap(
        args.output_dir / "features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 3, 2304),
    )
    verb = np.lib.format.open_memmap(
        args.output_dir / "verb_probability.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 3, 97),
    )
    noun = np.lib.format.open_memmap(
        args.output_dir / "noun_probability.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 3, 300),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, checkpoint = load_model(args.checkpoint, args.config, device)
    captured: list[torch.Tensor] = []

    def hook(_module, inputs):
        pathways = inputs[0]
        pooled = [F.adaptive_avg_pool3d(value, 1).flatten(1) for value in pathways]
        captured.append(torch.cat(pooled, dim=1).detach())

    handle = model.head.register_forward_pre_hook(hook)
    records = [(row, view) for row in range(len(rows)) for view in range(3)]
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            values = np.stack(
                [images[row, :, :, view] for row, view in batch_records]
            )
            captured.clear()
            output = model(prepare_video(values, device))
            if len(captured) != 1 or captured[0].shape[1] != 2304:
                raise RuntimeError(
                    f"EPIC feature hook changed: {[tuple(value.shape) for value in captured]}"
                )
            verb_value, noun_value = output
            feature_value = captured[0].cpu().numpy().astype(np.float16)
            verb_value = verb_value.cpu().numpy().astype(np.float16)
            noun_value = noun_value.cpu().numpy().astype(np.float16)
            for index, (row, view) in enumerate(batch_records):
                features[row, view] = feature_value[index]
                verb[row, view] = verb_value[index]
                noun[row, view] = noun_value[index]
            if start % (args.batch_size * 20) == 0:
                print(
                    json.dumps(
                        {
                            "stage": "P130_cache",
                            "encoded": min(start + len(batch_records), len(records)),
                            "total": len(records),
                        }
                    ),
                    flush=True,
                )
    handle.remove()
    features.flush()
    verb.flush()
    noun.flush()
    report = {
        "stage": "P130_EPIC_SlowFast_frozen_cache",
        "rows": len(rows),
        "views": list(VIEWS),
        "features_shape": list(features.shape),
        "verb_shape": list(verb.shape),
        "noun_shape": list(noun.shape),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "official_pretraining": "EPIC-KITCHENS-100 verb+noun multitask",
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


def aligned_scores(model, values: np.ndarray) -> np.ndarray:
    score = np.asarray(model.decision_function(values), dtype=np.float64)
    output = np.full((len(values), 40), score.min() - 1.0, dtype=np.float64)
    output[:, model.classes_.astype(np.int64)] = score
    return output


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
    feature = np.asarray(np.load(args.output_dir / "features.npy", mmap_mode="r"), dtype=np.float32)
    verb = np.asarray(
        np.load(args.output_dir / "verb_probability.npy", mmap_mode="r"), dtype=np.float32
    )
    noun = np.asarray(
        np.load(args.output_dir / "noun_probability.npy", mmap_mode="r"), dtype=np.float32
    )
    variants = {
        "workspace": l2(feature[:, 2]),
        "all_views": l2(feature.reshape(len(feature), -1)),
        "all_views_epic_logits": np.concatenate(
            (l2(feature.reshape(len(feature), -1)), verb.reshape(len(verb), -1), noun.reshape(len(noun), -1)),
            axis=1,
        ).astype(np.float32),
    }
    saved: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "labels": labels,
        "fold_ids": fold_ids,
    }
    report_variants = {}
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
            prediction = logits[held].argmax(axis=1)
            folds.append({"fold": fold, **metrics(labels[held], prediction)})
        prediction = logits.argmax(axis=1)
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        report_variants[name] = {
            "feature_dim": int(values.shape[1]),
            "alpha": args.alpha,
            "metrics": metrics(labels, prediction),
            "folds": folds,
        }
        saved[f"{name}_logits"] = logits.astype(np.float32)
        saved[f"{name}_probability"] = probability.astype(np.float32)
    report = {
        "stage": "P130_EPIC_SlowFast_complete_subject_safe_OOF",
        "status": "complete",
        "protocol": {
            "backbone_frozen": True,
            "primary_variant": "all_views_epic_logits",
            "alpha_fixed": args.alpha,
            "held_fold_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "variants": report_variants,
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
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=3000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pixel_cache = args.pixel_cache.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.stage in ("all", "cache"):
        build_cache(args)
    if args.stage in ("all", "oof"):
        build_oof(args)


if __name__ == "__main__":
    main()
