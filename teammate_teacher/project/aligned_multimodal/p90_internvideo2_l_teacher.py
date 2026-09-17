"""Fold-0 gate for the official InternVideo2 distilled ViT-L K400 teacher."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import log_softmax
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import VideoMAEImageProcessor

from p90_teacher_common import REPO_ROOT, classification_metrics, load_protocol
from p90_videomae_lora_teacher import EvalTrialDataset, read_aligned_rows
from p90_videomaev2_distilled_teacher import (
    P85_OOF,
    P85_STRONG,
    accuracy,
    aligned,
    class_sample_weights,
)
from train_p46_videomae_head import l2_normalize, make_model, row_standardize


MODEL_REPO = "OpenGVLab/InternVideo2_distillation_models"
MODEL_FILE = (
    Path(r"C:\Users\ncy\.cache\huggingface\hub")
    / "models--OpenGVLab--InternVideo2_distillation_models"
    / "snapshots/449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac"
    / "stage1/L14/L14_ft_k710_ft_k400_f8/pytorch_model.bin"
)
OFFICIAL_ROOT = (
    REPO_ROOT.parent
    / "external_data/InternVideo/InternVideo2/single_modality/models"
)
PROCESSOR_SNAPSHOT = Path(
    r"C:\Users\ncy\.cache\huggingface\hub"
    r"\models--MCG-NJU--videomae-large-finetuned-kinetics"
    r"\snapshots\0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "p90_internvideo2_l_k400_teacher_v1"
P90_BASE_FOLD0 = (
    REPO_ROOT / "runs/p90_videomaev2_distilled_teacher_v1/fold0_logits.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-batch", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument(
        "--reuse-cache", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def import_official_model() -> Any:
    package = types.ModuleType("p90_iv2_models")
    package.__path__ = [str(OFFICIAL_ROOT)]
    sys.modules["p90_iv2_models"] = package
    from p90_iv2_models.internvideo2 import internvideo2_large_patch14_224

    return internvideo2_large_patch14_224


def build_model(device: torch.device) -> tuple[nn.Module, nn.Module]:
    constructor = import_official_model()
    model = constructor(
        num_classes=400,
        num_frames=8,
        tubelet_size=1,
        drop_path_rate=0.1,
        head_drop_path_rate=0.1,
        init_values=1e-5,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
    )
    state = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    head = model.head
    model.head = nn.Identity()
    model = model.eval().to(device=device, dtype=torch.bfloat16)
    head = head.eval().to(device=device, dtype=torch.bfloat16)
    return model, head


class EightFrameVideoCollator:
    def __init__(self, processor: VideoMAEImageProcessor) -> None:
        self.processor = processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        views = len(items[0]["videos"])
        videos = []
        for item in items:
            if len(item["videos"]) != views:
                raise ValueError("mixed clip count")
            for video in item["videos"]:
                positions = np.linspace(0, len(video) - 1, 8).round().astype(int)
                videos.append([video[position] for position in positions])
        pixels = self.processor(videos, return_tensors="pt").pixel_values
        return {
            "pixel_values": pixels,
            "sample_ids": [item["sample_id"] for item in items],
        }


@torch.inference_mode()
def extract(args: argparse.Namespace, cache: Path) -> dict[str, np.ndarray]:
    protocol = load_protocol()
    rows = read_aligned_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("InternVideo2-L fold-0 gate requires CUDA")
    model, head = build_model(device)
    processor = VideoMAEImageProcessor.from_pretrained(
        PROCESSOR_SNAPSHOT, local_files_only=True
    )
    dataset = EvalTrialDataset(rows, protocol.labels, "ir")
    partial = cache.parent / "partial_features.npz"
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    sample_ids: list[str] = []
    if partial.exists():
        with np.load(partial, allow_pickle=False) as source:
            sample_ids = source["sample_ids"].astype(str).tolist()
            feature_parts.append(
                np.asarray(source["features"], dtype=np.float16).reshape(-1, 6, 768)
            )
            action_parts.append(
                np.asarray(source["action_logits"], dtype=np.float16).reshape(-1, 6, 400)
            )
        expected = protocol.sample_ids[: len(sample_ids)].tolist()
        if sample_ids != expected:
            raise ValueError("InternVideo2 partial cache order changed")
        print(f"resuming partial trials={len(sample_ids)}/{len(dataset)}", flush=True)
    remaining_dataset = Subset(dataset, range(len(sample_ids), len(dataset)))
    loader = DataLoader(
        remaining_dataset,
        batch_size=args.trial_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=EightFrameVideoCollator(processor),
        pin_memory=True,
    )
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for batch_number, batch in enumerate(loader):
        pixels = batch["pixel_values"].permute(0, 2, 1, 3, 4)
        pixels = pixels.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        features = model(pixels)
        action_logits = head(features)
        trials = len(batch["sample_ids"])
        if features.shape != (trials * 6, 768):
            raise ValueError(f"Unexpected InternVideo2 feature shape: {features.shape}")
        feature_parts.append(features.reshape(trials, 6, 768).half().cpu().numpy())
        action_parts.append(action_logits.reshape(trials, 6, 400).half().cpu().numpy())
        sample_ids.extend(batch["sample_ids"])
        if (batch_number + 1) % 20 == 0:
            print(
                f"  extracted trials={len(sample_ids)}/{len(dataset)} "
                f"elapsed_min={(time.time() - started) / 60:.1f}",
                flush=True,
            )
        if (batch_number + 1) % args.checkpoint_every == 0:
            temporary = cache.parent / "partial_features.tmp.npz"
            np.savez(
                temporary,
                sample_ids=np.asarray(sample_ids),
                features=np.concatenate(feature_parts).reshape(-1, 2, 3, 768),
                action_logits=np.concatenate(action_parts).reshape(-1, 2, 3, 400),
            )
            temporary.replace(partial)
            with np.load(partial, allow_pickle=False) as source:
                feature_parts = [np.asarray(source["features"]).reshape(-1, 6, 768)]
                action_parts = [np.asarray(source["action_logits"]).reshape(-1, 6, 400)]
    if sample_ids != protocol.sample_ids.tolist():
        raise ValueError("InternVideo2 extraction order changed")
    payload = {
        "sample_ids": protocol.sample_ids,
        "labels": protocol.labels,
        "users": protocol.users,
        "fold_id": protocol.fold_id,
        "features": np.concatenate(feature_parts).reshape(-1, 2, 3, 768),
        "action_logits": np.concatenate(action_parts).reshape(-1, 2, 3, 400),
    }
    np.savez_compressed(cache, **payload)
    metadata = {
        "model_repo": MODEL_REPO,
        "model_file": str(MODEL_FILE),
        "official_repo": "https://github.com/OpenGVLab/InternVideo",
        "official_commit": "3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3",
        "license": "official GitHub repository Apache-2.0; HF weight repo lacks metadata",
        "pretraining": "K-Mash, distilled from InternVideo2 Stage2 1B, K710 then K400 fine-tuning",
        "official_k400_top1": 90.4,
        "parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
            + sum(parameter.numel() for parameter in head.parameters())
        ),
        "frames_per_clip": 8,
        "clips": "early/late x scene/person/workspace",
        "elapsed_seconds": time.time() - started,
        "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 2**30),
    }
    (cache.parent / "cache_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def feature_sets(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    features = l2_normalize(np.asarray(data["features"], dtype=np.float32))
    action = row_standardize(
        np.asarray(data["action_logits"], dtype=np.float32).reshape(len(features), -1)
    )
    mean = l2_normalize(features.mean(axis=1))
    delta = features[:, 1] - features[:, 0]
    return {
        "early_late": features.reshape(len(features), -1),
        "window_mean": mean.reshape(len(features), -1),
        "temporal_delta": np.concatenate((mean, delta), axis=1).reshape(
            len(features), -1
        ),
        "k400_logits": action,
        "early_late_plus_k400": np.concatenate(
            (features.reshape(len(features), -1), action), axis=1
        ),
    }


def evaluate_fold0(data: dict[str, np.ndarray], output: Path) -> dict[str, Any]:
    protocol = load_protocol()
    train_indices = protocol.train_indices(0)
    val_indices = protocol.val_indices(0)
    labels = protocol.labels
    recipes = {
        "early_late": (3000.0, 0.75),
        "window_mean": (1000.0, 0.75),
        "temporal_delta": (3000.0, 0.75),
        "k400_logits": (1000.0, 0.50),
        "early_late_plus_k400": (3000.0, 0.75),
    }
    matrices = feature_sets(data)
    candidate_logits: dict[str, np.ndarray] = {}
    candidates: dict[str, Any] = {}
    for name, values in matrices.items():
        alpha, power = recipes[name]
        model = make_model(alpha)
        model.fit(
            values[train_indices],
            labels[train_indices],
            ridge__sample_weight=class_sample_weights(labels[train_indices], power),
        )
        logits = np.asarray(model.decision_function(values[val_indices]), dtype=np.float64)
        candidate_logits[name] = logits
        candidates[name] = {
            "recipe": {"alpha": alpha, "class_weight_power": power},
            "metrics": classification_metrics(logits, labels[val_indices]),
        }

    with np.load(P85_OOF, allow_pickle=False) as source:
        p85_visual = aligned(
            source["sample_ids"], source["early_late_logits"], protocol.sample_ids[val_indices]
        ).astype(np.float64)
    with np.load(P85_STRONG, allow_pickle=False) as source:
        p85_strong = np.log(
            np.clip(
                aligned(
                    source["oof_sample_ids"],
                    source["oof_teacher_probability"],
                    protocol.sample_ids[val_indices],
                ),
                1e-8,
                1.0,
            )
        )
    with np.load(P90_BASE_FOLD0, allow_pickle=False) as source:
        p90_base = aligned(
            source["sample_ids"], source["early_late_logits"], protocol.sample_ids[val_indices]
        ).astype(np.float64)
    bases = {
        "p85_visual_ridge": log_softmax(p85_visual, axis=1),
        "p85_fullwindow_multimodal": p85_strong,
        "p90_videomaev2_distilled_base": log_softmax(p90_base, axis=1),
    }
    baselines = {
        name: classification_metrics(scores, labels[val_indices])
        for name, scores in bases.items()
    }
    blends: dict[str, Any] = {}
    for name, logits in candidate_logits.items():
        candidate_logp = log_softmax(logits, axis=1)
        item: dict[str, Any] = {}
        for base_name, base_scores in bases.items():
            fixed = {
                str(weight): accuracy(
                    (1.0 - weight) * base_scores + weight * candidate_logp,
                    labels[val_indices],
                )
                for weight in (0.10, 0.25, 0.50)
            }
            scan = [
                (
                    float(weight),
                    accuracy(
                        (1.0 - weight) * base_scores + weight * candidate_logp,
                        labels[val_indices],
                    ),
                )
                for weight in np.arange(0.0, 1.0001, 0.025)
            ]
            best = max(scan, key=lambda row: row[1])
            item[base_name] = {
                "fixed_weights": fixed,
                "diagnostic_best_weight": best[0],
                "diagnostic_best_accuracy": best[1],
            }
        blends[name] = item
    np.savez_compressed(
        output / "fold0_logits.npz",
        sample_ids=protocol.sample_ids[val_indices],
        labels=labels[val_indices],
        **{
            f"{name}_logits": values.astype(np.float32)
            for name, values in candidate_logits.items()
        },
    )
    best_name = max(candidates, key=lambda name: candidates[name]["metrics"]["accuracy"])
    return {
        "protocol": "P90 subject fold0 only; fixed P85 Ridge recipes",
        "baselines": baselines,
        "candidates": candidates,
        "best_fold0_candidate": best_name,
        "blends": blends,
    }


def main(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "complete_features.npz"
    if args.reuse_cache and cache.exists():
        with np.load(cache, allow_pickle=False) as source:
            data = {key: np.asarray(source[key]) for key in source.files}
    else:
        data = extract(args, cache)
    report = evaluate_fold0(data, output)
    (output / "fold0_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main(parse_args())
