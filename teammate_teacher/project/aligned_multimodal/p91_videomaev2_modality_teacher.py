"""One-fold screen of the VideoMAEv2 distilled teacher on Depth/Thermal.

This is intentionally a representation screen rather than a deployment model.
It extracts the official K710 distilled features for scene/person/workspace clips,
then evaluates both the new modality alone and direct IR+new-modality feature
fusion on the untouched P90 fold 0.  No other folds are run by this script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from scipy.special import log_softmax
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEImageProcessor

from build_p46_videomae_modality_cache import (
    prepare_trial as prepare_modality_trial,
    thermal_dir,
)
from p90_crossuser_visual_router import load_splits
from p90_teacher_common import REPO_ROOT, classification_metrics, load_protocol
from p90_videomae_lora_teacher import P29_RUN, VideoCollator, read_aligned_rows
from p90_videomaev2_distilled_teacher import (
    PROCESSOR_REPO,
    build_model,
    class_sample_weights,
)
from train_p46_videomae_head import l2_normalize, make_model, row_standardize


IR_VMAE = REPO_ROOT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz"
IR_IV2 = REPO_ROOT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz"
DEFAULT_OUTPUT = REPO_ROOT / "runs/p91_videomaev2_depth_fold0_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("depth", "thermal"), default="depth")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-batch", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class ModalityTrialDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        labels: np.ndarray,
        modality: str,
    ) -> None:
        self.rows = rows
        self.labels = labels
        self.modality = modality

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        available = self.modality != "thermal" or any(thermal_dir(row).glob("*.jpg"))
        if available:
            clips, _ = prepare_modality_trial(row, P29_RUN, self.modality)
        else:
            black = np.zeros((224, 224, 3), dtype=np.uint8)
            clips = [[black] * 16 for _ in range(3)]
        if len(clips) != 3:
            raise ValueError(f"{self.modality} did not produce three ROI clips")
        return {
            "videos": clips,
            "clip_indices": [0, 1, 2],
            "label": int(self.labels[index]),
            "sample_id": row["sample_id"],
        }


@torch.inference_mode()
def extract(args: argparse.Namespace, cache: Path) -> dict[str, np.ndarray]:
    protocol = load_protocol()
    rows = read_aligned_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = build_model(device)
    processor_snapshot = Path(snapshot_download(PROCESSOR_REPO, local_files_only=True))
    processor = VideoMAEImageProcessor.from_pretrained(
        processor_snapshot, local_files_only=True
    )
    dataset = ModalityTrialDataset(rows, protocol.labels, args.modality)
    loader = DataLoader(
        dataset,
        batch_size=args.trial_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=VideoCollator(processor),
        pin_memory=True,
    )
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    sample_ids: list[str] = []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for batch_number, batch in enumerate(loader):
        pixels = batch["pixel_values"].permute(0, 2, 1, 3, 4)
        pixels = pixels.to(device=device, dtype=torch.float16, non_blocking=True)
        features = model.forward_features(pixels)
        action_logits = model.head(features)
        trials = len(batch["sample_ids"])
        if features.shape[0] != trials * 3:
            raise ValueError(f"{args.modality} batch did not contain three clips per trial")
        feature_parts.append(features.reshape(trials, 3, 768).cpu().numpy())
        action_parts.append(action_logits.reshape(trials, 3, 710).cpu().numpy())
        sample_ids.extend(batch["sample_ids"])
        if (batch_number + 1) % 20 == 0:
            print(
                f"  {args.modality} extracted={len(sample_ids)}/{len(dataset)} "
                f"elapsed_min={(time.time() - started) / 60:.1f}",
                flush=True,
            )
    if sample_ids != protocol.sample_ids.tolist():
        raise ValueError("extraction order changed")
    features = np.concatenate(feature_parts)
    action_logits = np.concatenate(action_parts)
    modality_available = np.asarray(
        [
            args.modality != "thermal" or any(thermal_dir(row).glob("*.jpg"))
            for row in rows
        ],
        dtype=np.uint8,
    )
    features[modality_available == 0] = 0
    action_logits[modality_available == 0] = 0
    payload = {
        "sample_ids": protocol.sample_ids,
        "labels": protocol.labels,
        "users": protocol.users,
        "fold_id": protocol.fold_id,
        "features": features,
        "action_logits": action_logits,
        "modality_available": modality_available,
        "modality": np.asarray(args.modality),
    }
    np.savez_compressed(cache, **payload)
    summary = {
        "modality": args.modality,
        "checkpoint": str(checkpoint),
        "clips": "scene/person/workspace",
        "samples": len(sample_ids),
        "available_samples": int(modality_available.sum()),
        "elapsed_seconds": time.time() - started,
        "peak_cuda_gib": (
            float(torch.cuda.max_memory_allocated() / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    (cache.parent / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def check_alignment(data: dict[str, np.ndarray], sample_ids: np.ndarray, name: str) -> None:
    if not np.array_equal(data["sample_ids"].astype(str), sample_ids.astype(str)):
        raise ValueError(f"{name} sample order differs from P90 protocol")


def modality_features(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    features = l2_normalize(np.asarray(data["features"], dtype=np.float32))
    action = row_standardize(
        np.asarray(data["action_logits"], dtype=np.float32).reshape(len(features), -1)
    )
    mean = l2_normalize(features.mean(axis=1))
    return {
        "roi_all": features.reshape(len(features), -1),
        "roi_mean": mean,
        "k710_logits": action,
        "roi_all_plus_k710": np.concatenate(
            (features.reshape(len(features), -1), action), axis=1
        ),
    }


def ir_features(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    features = l2_normalize(np.asarray(data["features"], dtype=np.float32))
    mean = l2_normalize(features.mean(axis=1)).reshape(len(features), -1)
    return features.reshape(len(features), -1), mean


def aligned_safe_probability(sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    splits = load_splits()
    probability_by_id: dict[str, np.ndarray] = {}
    prediction_by_id: dict[str, int] = {}
    for split in splits.values():
        for sample_id, probability, prediction in zip(
            split.sample_ids, split.safe_probability, split.safe_prediction
        ):
            probability_by_id[str(sample_id)] = probability
            prediction_by_id[str(sample_id)] = int(prediction)
    missing = [str(value) for value in sample_ids if str(value) not in probability_by_id]
    if missing:
        raise ValueError(f"safe baseline is missing {len(missing)} requested rows")
    return (
        np.stack([probability_by_id[str(value)] for value in sample_ids]),
        np.asarray([prediction_by_id[str(value)] for value in sample_ids], dtype=np.int64),
    )


def accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(scores).argmax(axis=1) == labels))


def change_audit(
    base_prediction: np.ndarray, candidate_prediction: np.ndarray, labels: np.ndarray
) -> dict[str, int]:
    changed = base_prediction != candidate_prediction
    return {
        "changed": int(changed.sum()),
        "rescued": int(((base_prediction != labels) & (candidate_prediction == labels)).sum()),
        "harmed": int(((base_prediction == labels) & (candidate_prediction != labels)).sum()),
    }


def evaluate_fold0(data: dict[str, np.ndarray], output: Path) -> dict[str, Any]:
    protocol = load_protocol()
    check_alignment(data, protocol.sample_ids, "new modality")
    ir_vmae = load_npz(IR_VMAE)
    ir_iv2 = load_npz(IR_IV2)
    check_alignment(ir_vmae, protocol.sample_ids, "IR VideoMAEv2")
    check_alignment(ir_iv2, protocol.sample_ids, "IR InternVideo2")

    train = protocol.train_indices(0)
    val = protocol.val_indices(0)
    labels = protocol.labels
    safe_probability, safe_prediction = aligned_safe_probability(protocol.sample_ids[val])
    new = modality_features(data)
    vmae_all, vmae_mean = ir_features(ir_vmae)
    iv2_all, iv2_mean = ir_features(ir_iv2)
    new_all = new["roi_all"]
    new_mean = new["roi_mean"]
    candidates = {
        **new,
        "ir_vmae_plus_new": np.concatenate((vmae_all, new_all), axis=1),
        "ir_two_teacher_plus_new": np.concatenate((vmae_all, iv2_all, new_all), axis=1),
        "ir_two_teacher_mean_plus_new_mean": np.concatenate(
            (vmae_mean, iv2_mean, new_mean), axis=1
        ),
    }
    recipes = {
        "roi_all": (3000.0, 0.75),
        "roi_mean": (1000.0, 0.75),
        "k710_logits": (1000.0, 0.50),
        "roi_all_plus_k710": (3000.0, 0.75),
        "ir_vmae_plus_new": (3000.0, 0.75),
        "ir_two_teacher_plus_new": (3000.0, 0.75),
        "ir_two_teacher_mean_plus_new_mean": (1000.0, 0.75),
    }
    results: dict[str, Any] = {}
    logits_by_name: dict[str, np.ndarray] = {}
    safe_logp = np.log(np.clip(safe_probability, 1e-8, 1.0))
    base_prediction = safe_prediction
    for name, values in candidates.items():
        alpha, power = recipes[name]
        model = make_model(alpha)
        model.fit(
            values[train],
            labels[train],
            ridge__sample_weight=class_sample_weights(labels[train], power),
        )
        logits = np.asarray(model.decision_function(values[val]), dtype=np.float64)
        logits_by_name[name] = logits.astype(np.float32)
        prediction = logits.argmax(axis=1)
        blends: dict[str, Any] = {}
        candidate_logp = log_softmax(logits, axis=1)
        for weight in (0.05, 0.10, 0.20, 0.30, 0.50):
            blended_prediction = (
                (1.0 - weight) * safe_logp + weight * candidate_logp
            ).argmax(axis=1)
            blends[str(weight)] = {
                "accuracy": float(np.mean(blended_prediction == labels[val])),
                **change_audit(base_prediction, blended_prediction, labels[val]),
            }
        results[name] = {
            "alpha": alpha,
            "class_weight_power": power,
            "dimensions": int(values.shape[1]),
            "metrics": classification_metrics(logits, labels[val]),
            "safe_blends_fold0_diagnostic": blends,
        }
        print(
            f"  {name}: accuracy={accuracy(logits, labels[val]):.6f} dim={values.shape[1]}",
            flush=True,
        )

    np.savez_compressed(
        output / "fold0_logits.npz",
        sample_ids=protocol.sample_ids[val],
        labels=labels[val],
        safe_prediction=base_prediction,
        safe_probability=safe_probability.astype(np.float32),
        **{f"{name}_logits": values for name, values in logits_by_name.items()},
    )
    router_path = REPO_ROOT / "runs/p90_crossuser_visual_router_v1/gate_predictions.npz"
    router = load_npz(router_path)
    router_lookup = {
        str(sample_id): int(prediction)
        for sample_id, prediction in zip(router["sample_ids"], router["router_prediction"])
    }
    router_prediction = np.asarray(
        [router_lookup[str(sample_id)] for sample_id in protocol.sample_ids[val]]
    )
    best_standalone = max(results, key=lambda key: results[key]["metrics"]["accuracy"])
    return {
        "protocol": "P90 subject fold0 only; Ridge recipes fixed before this screen",
        "modality": str(np.asarray(data["modality"]).item()),
        "train_samples": int(len(train)),
        "validation_samples": int(len(val)),
        "baselines": {
            "p89_safe": {
                "accuracy": float(np.mean(base_prediction == labels[val])),
                "correct": int(np.sum(base_prediction == labels[val])),
            },
            "p90_crossuser_visual_router": {
                "accuracy": float(np.mean(router_prediction == labels[val])),
                "correct": int(np.sum(router_prediction == labels[val])),
            },
        },
        "candidates": results,
        "best_standalone": best_standalone,
        "selection_warning": (
            "Blend weights are diagnostic on fold0 and are not authorized for deployment. "
            "A positive screen must be repeated with source-only OOF weight selection."
        ),
    }


def main(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "complete_features.npz"
    if args.reuse_cache and cache.exists():
        data = load_npz(cache)
        if str(np.asarray(data["modality"]).item()) != args.modality:
            raise ValueError("cache modality and --modality differ")
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
