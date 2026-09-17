"""Fold-0 screen for the official VideoMAE V2 giant-distilled K710 teacher.

The checkpoint is a ViT-Base distilled from the official VideoMAE V2 giant
teacher on K710.  It is materially different from the P85 K400 VideoMAE-L
backbone.  Frozen features are extracted for early/late x
scene/person/workspace IR clips, then evaluated with the same Ridge recipes
that were already selected for P85.  Only fold 0 is evaluated here; this file
does not authorize the remaining folds unless the gate improves.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from scipy.special import log_softmax
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor

from p90_teacher_common import REPO_ROOT, classification_metrics, load_protocol
from p90_videomae_lora_teacher import EvalTrialDataset, VideoCollator, read_aligned_rows
from train_p46_videomae_head import l2_normalize, make_model, row_standardize


MODEL_REPO = "OpenGVLab/VideoMAE2"
MODEL_FILE = "distill/vit_b_k710_dl_from_giant.pth"
PROCESSOR_REPO = "MCG-NJU/videomae-large-finetuned-kinetics"
EXTERNAL_REPO = REPO_ROOT.parent / "external_data" / "VideoMAEv2"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "p90_videomaev2_distilled_teacher_v1"
P85_OOF = (
    REPO_ROOT
    / "aligned_multimodal/runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)
P85_STRONG = (
    REPO_ROOT
    / "aligned_multimodal/runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-batch", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_model(device: torch.device) -> tuple[torch.nn.Module, Path]:
    if not (EXTERNAL_REPO / "models/modeling_finetune.py").exists():
        raise FileNotFoundError(
            f"Official VideoMAE V2 repository is missing: {EXTERNAL_REPO}"
        )
    sys.path.insert(0, str(EXTERNAL_REPO.resolve()))
    from models.modeling_finetune import vit_base_patch16_224

    checkpoint = Path(
        hf_hub_download(MODEL_REPO, MODEL_FILE, local_files_only=True)
    )
    model = vit_base_patch16_224(
        num_classes=710,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["module"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device=device, dtype=torch.float16)
    return model, checkpoint


@torch.inference_mode()
def extract(args: argparse.Namespace, cache: Path) -> dict[str, np.ndarray]:
    protocol = load_protocol()
    rows = read_aligned_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = build_model(device)
    processor_snapshot = Path(
        snapshot_download(PROCESSOR_REPO, local_files_only=True)
    )
    processor = VideoMAEImageProcessor.from_pretrained(
        processor_snapshot, local_files_only=True
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
    feature_parts = []
    action_parts = []
    sample_ids: list[str] = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    for batch_number, batch in enumerate(loader):
        pixels = batch["pixel_values"].permute(0, 2, 1, 3, 4)
        pixels = pixels.to(device=device, dtype=torch.float16, non_blocking=True)
        features = model.forward_features(pixels)
        action_logits = model.head(features)
        trials = len(batch["sample_ids"])
        if features.shape[0] != trials * 6:
            raise ValueError("IR batch did not contain six clips per trial")
        feature_parts.append(features.reshape(trials, 6, 768).cpu().numpy())
        action_parts.append(action_logits.reshape(trials, 6, 710).cpu().numpy())
        sample_ids.extend(batch["sample_ids"])
        if (batch_number + 1) % 20 == 0:
            print(
                f"  extracted trials={len(sample_ids)}/{len(dataset)} "
                f"elapsed_min={(time.time() - started) / 60:.1f}",
                flush=True,
            )
    if sample_ids != protocol.sample_ids.tolist():
        raise ValueError("VideoMAE V2 extraction order changed")
    payload = {
        "sample_ids": protocol.sample_ids,
        "labels": protocol.labels,
        "users": protocol.users,
        "fold_id": protocol.fold_id,
        "features": np.concatenate(feature_parts).reshape(-1, 2, 3, 768),
        "action_logits": np.concatenate(action_parts).reshape(-1, 2, 3, 710),
    }
    np.savez_compressed(cache, **payload)
    extraction = {
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "checkpoint": str(checkpoint),
        "official_repo": "https://github.com/OpenGVLab/VideoMAEv2",
        "checkpoint_license": "Hugging Face repository metadata: cc-by-nc-4.0",
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "clips": "early/late x scene/person/workspace",
        "processor": PROCESSOR_REPO,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_gib": (
            float(torch.cuda.max_memory_allocated() / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "cache": str(cache),
    }
    (cache.parent / "cache_summary.json").write_text(
        json.dumps(extraction, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def class_sample_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    reference = counts[counts > 0].mean()
    weights = np.zeros(40, dtype=np.float64)
    present = counts > 0
    weights[present] = np.power(reference / counts[present], power)
    values = weights[labels]
    return values / values.mean()


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
        "k710_logits": action,
        "early_late_plus_k710": np.concatenate(
            (features.reshape(len(features), -1), action), axis=1
        ),
    }


def aligned(source_ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids.astype(str))}
    return np.asarray(values)[[lookup[value] for value in target_ids.astype(str)]]


def accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(scores.argmax(axis=1) == labels))


def evaluate_fold0(data: dict[str, np.ndarray], output: Path) -> dict[str, Any]:
    protocol = load_protocol()
    if not np.array_equal(data["sample_ids"].astype(str), protocol.sample_ids):
        raise ValueError("VideoMAE V2 cache and P90 protocol differ")
    train_indices = protocol.train_indices(0)
    val_indices = protocol.val_indices(0)
    labels = protocol.labels
    recipes = {
        "early_late": (3000.0, 0.75),
        "window_mean": (1000.0, 0.75),
        "temporal_delta": (3000.0, 0.75),
        "k710_logits": (1000.0, 0.50),
        "early_late_plus_k710": (3000.0, 0.75),
    }
    matrices = feature_sets(data)
    candidate_logits: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {}
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
        results[name] = {
            "recipe_source": "fixed transfer from the corresponding P85 feature family",
            "alpha": alpha,
            "class_weight_power": power,
            "dimensions": int(values.shape[1]),
            "metrics": classification_metrics(logits, labels[val_indices]),
        }

    with np.load(P85_OOF, allow_pickle=False) as source:
        p85_ridge = aligned(
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
    baseline = {
        "p85_visual_ridge": classification_metrics(p85_ridge, labels[val_indices]),
        "p85_fullwindow_multimodal": classification_metrics(
            p85_strong, labels[val_indices]
        ),
    }
    blends: dict[str, Any] = {}
    for name, logits in candidate_logits.items():
        candidate_logp = log_softmax(logits, axis=1)
        item: dict[str, Any] = {}
        for base_name, base_scores in (
            ("p85_visual_ridge", log_softmax(p85_ridge, axis=1)),
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
        **{f"{name}_logits": values.astype(np.float32) for name, values in candidate_logits.items()},
    )
    best_name = max(results, key=lambda name: results[name]["metrics"]["accuracy"])
    return {
        "protocol": (
            "P90 subject fold0 only; all Ridge recipes fixed from P85; feature-family "
            "choice is a development-fold diagnostic and is not a three-fold claim"
        ),
        "fold": 0,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(val_indices)),
        "baselines": baseline,
        "candidates": results,
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
        print(f"reusing {cache}", flush=True)
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
