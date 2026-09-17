"""Fold-0 gate for the raw self-supervised VideoMAE V2 Large teacher.

The official Hugging Face remote code predates the installed Transformers
version, so this script instantiates from the official config and then loads
the official safetensors checkpoint with ``strict=True``.  It evaluates only
fold 0 unless a later, separate OOF stage is explicitly justified.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

from p90_teacher_common import REPO_ROOT, classification_metrics, load_protocol
from p90_videomae_lora_teacher import EvalTrialDataset, VideoCollator, read_aligned_rows
from p90_videomaev2_distilled_teacher import (
    P85_OOF,
    P85_STRONG,
    accuracy,
    aligned,
    class_sample_weights,
)
from train_p46_videomae_head import l2_normalize, make_model


MODEL_REPO = "OpenGVLab/VideoMAEv2-Large"
MODEL_SNAPSHOT = Path(
    r"C:\Users\ncy\.cache\huggingface\hub\models--OpenGVLab--VideoMAEv2-Large"
    r"\snapshots\9981a9c8f77118c421e5228e1b219468a4b0238d"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "p90_videomaev2_large_teacher_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-batch", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--reuse-cache", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def build_model(device: torch.device) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(
        MODEL_SNAPSHOT, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_config(config, trust_remote_code=True)
    state = load_file(MODEL_SNAPSHOT / "model.safetensors", device="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device=device, dtype=torch.float16)


@torch.inference_mode()
def extract(args: argparse.Namespace, cache: Path) -> dict[str, np.ndarray]:
    protocol = load_protocol()
    rows = read_aligned_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The fold-0 Large teacher gate requires CUDA")
    model = build_model(device)
    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_SNAPSHOT, local_files_only=True
    )
    dataset = EvalTrialDataset(rows, protocol.labels, "ir")
    loader = DataLoader(
        dataset,
        batch_size=args.trial_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=VideoCollator(processor),
        pin_memory=True,
    )
    feature_parts: list[np.ndarray] = []
    sample_ids: list[str] = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for batch_number, batch in enumerate(loader):
        pixels = batch["pixel_values"].permute(0, 2, 1, 3, 4)
        pixels = pixels.to(device=device, dtype=torch.float16, non_blocking=True)
        features = model(pixel_values=pixels)
        trials = len(batch["sample_ids"])
        if features.shape != (trials * 6, 1024):
            raise ValueError(f"Unexpected Large feature shape: {features.shape}")
        feature_parts.append(features.reshape(trials, 6, 1024).cpu().numpy())
        sample_ids.extend(batch["sample_ids"])
        if (batch_number + 1) % 20 == 0:
            print(
                f"  extracted trials={len(sample_ids)}/{len(dataset)} "
                f"elapsed_min={(time.time() - started) / 60:.1f}",
                flush=True,
            )
    if sample_ids != protocol.sample_ids.tolist():
        raise ValueError("VideoMAE V2 Large extraction order changed")
    payload = {
        "sample_ids": protocol.sample_ids,
        "labels": protocol.labels,
        "users": protocol.users,
        "fold_id": protocol.fold_id,
        "features": np.concatenate(feature_parts).reshape(-1, 2, 3, 1024),
    }
    np.savez_compressed(cache, **payload)
    summary = {
        "model_repo": MODEL_REPO,
        "snapshot": str(MODEL_SNAPSHOT),
        "official_repo": "https://github.com/OpenGVLab/VideoMAEv2",
        "license": "cc-by-nc-4.0",
        "pretraining": "self-supervised UnlabeledHybrid-1M",
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "clips": "early/late x scene/person/workspace",
        "elapsed_seconds": time.time() - started,
        "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "cache": str(cache),
    }
    (cache.parent / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def feature_sets(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    features = l2_normalize(np.asarray(data["features"], dtype=np.float32))
    mean = l2_normalize(features.mean(axis=1))
    delta = features[:, 1] - features[:, 0]
    return {
        "early_late": features.reshape(len(features), -1),
        "window_mean": mean.reshape(len(features), -1),
        "temporal_delta": np.concatenate((mean, delta), axis=1).reshape(
            len(features), -1
        ),
    }


def evaluate_fold0(data: dict[str, np.ndarray], output: Path) -> dict[str, object]:
    protocol = load_protocol()
    train_indices = protocol.train_indices(0)
    val_indices = protocol.val_indices(0)
    labels = protocol.labels
    recipes = {
        "early_late": (3000.0, 0.75),
        "window_mean": (1000.0, 0.75),
        "temporal_delta": (3000.0, 0.75),
    }
    matrices = feature_sets(data)
    logits_by_name: dict[str, np.ndarray] = {}
    candidates: dict[str, object] = {}
    for name, values in matrices.items():
        alpha, power = recipes[name]
        model = make_model(alpha)
        model.fit(
            values[train_indices],
            labels[train_indices],
            ridge__sample_weight=class_sample_weights(labels[train_indices], power),
        )
        logits = np.asarray(model.decision_function(values[val_indices]), dtype=np.float64)
        logits_by_name[name] = logits
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
    baselines = {
        "p85_visual_ridge": classification_metrics(p85_visual, labels[val_indices]),
        "p85_fullwindow_multimodal": classification_metrics(
            p85_strong, labels[val_indices]
        ),
        "p90_distilled_base_early_late": {"accuracy": 0.7810894141829393},
    }
    blends: dict[str, object] = {}
    from scipy.special import log_softmax

    for name, logits in logits_by_name.items():
        candidate_logp = log_softmax(logits, axis=1)
        item: dict[str, object] = {}
        for base_name, base_scores in (
            ("p85_visual_ridge", log_softmax(p85_visual, axis=1)),
            ("p85_fullwindow_multimodal", p85_strong),
        ):
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
            for name, values in logits_by_name.items()
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
