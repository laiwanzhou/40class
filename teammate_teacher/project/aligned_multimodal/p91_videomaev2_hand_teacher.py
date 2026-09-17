"""Dynamic hand-centric VideoMAEv2 teacher, screened on P90 fold 0 only.

Six clips are produced per trial: full-duration and peak-hand-motion windows,
each with left-hand, right-hand and two-hand interaction crops.  The crops are
driven by the cached per-frame pose ROIs rather than fixed image coordinates.
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
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEImageProcessor

from audit_yolo11_pose_skeleton import frame_map
from build_p30_shared_dir_roi_feature_cache import read_ir
from build_p46_videomae_cache import safe_relative, square_crop, uniform_indices
from p90_crossuser_visual_router import load_splits
from p90_teacher_common import REPO_ROOT, classification_metrics, load_protocol
from p90_videomae_lora_teacher import P29_RUN, VideoCollator, read_aligned_rows
from p90_videomaev2_distilled_teacher import (
    PROCESSOR_REPO,
    build_model,
    class_sample_weights,
)
from train_p46_videomae_head import l2_normalize, make_model, row_standardize


DEFAULT_OUTPUT = REPO_ROOT / "runs/p91_videomaev2_hand_fold0_v1"
VMAE = REPO_ROOT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz"
IV2 = REPO_ROOT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz"
DEPTH = REPO_ROOT / "runs/p91_videomaev2_depth_fold0_v1/complete_features.npz"
THERMAL = REPO_ROOT / "runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-batch", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def bounded_uniform(low: int, high: int, count: int = 16) -> np.ndarray:
    if high <= low:
        return np.full(count, max(low, 0), dtype=np.int64)
    return np.rint(np.linspace(low, high - 1, count)).astype(np.int64)


def centers(boxes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.full((len(boxes), 2), np.nan, dtype=np.float32)
    output[valid] = (boxes[valid, :2] + boxes[valid, 2:]) * 0.5
    for coordinate in range(2):
        values = output[:, coordinate]
        known = np.flatnonzero(np.isfinite(values))
        if len(known):
            output[:, coordinate] = np.interp(np.arange(len(values)), known, values[known])
        else:
            output[:, coordinate] = 0
    return output


def motion_window(
    boxes: np.ndarray, valid: np.ndarray, left: int, right: int
) -> np.ndarray:
    left_center = centers(boxes[:, left], valid[:, left])
    right_center = centers(boxes[:, right], valid[:, right])
    velocity = np.zeros(len(boxes), dtype=np.float32)
    if len(boxes) > 1:
        velocity[1:] = np.linalg.norm(np.diff(left_center, axis=0), axis=1)
        velocity[1:] += np.linalg.norm(np.diff(right_center, axis=0), axis=1)
    if len(velocity) >= 5:
        velocity = np.convolve(velocity, np.ones(5, dtype=np.float32) / 5, mode="same")
    center = int(np.argmax(velocity))
    span = min(len(boxes), max(16, int(round(len(boxes) * 0.45))))
    low = min(max(center - span // 2, 0), max(len(boxes) - span, 0))
    return bounded_uniform(low, low + span)


def fallback_box(
    boxes: np.ndarray,
    valid: np.ndarray,
    frame: int,
    primary: int,
    workspace: int,
    person: int,
) -> np.ndarray:
    for index in (primary, workspace, person):
        if valid[frame, index]:
            return boxes[frame, index]
    return np.full(4, np.nan, dtype=np.float32)


def interaction_box(
    boxes: np.ndarray,
    valid: np.ndarray,
    frame: int,
    left: int,
    right: int,
    workspace: int,
    person: int,
) -> np.ndarray:
    hands = [boxes[frame, index] for index in (left, right) if valid[frame, index]]
    if hands:
        values = np.stack(hands)
        return np.asarray(
            [values[:, 0].min(), values[:, 1].min(), values[:, 2].max(), values[:, 3].max()],
            dtype=np.float32,
        )
    return fallback_box(boxes, valid, frame, workspace, workspace, person)


def prepare_hand_clips(row: dict[str, str]) -> list[list[np.ndarray]]:
    cache = P29_RUN / "trial_roi_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
    with np.load(cache, allow_pickle=False) as data:
        frame_ids = np.asarray(data["frame_ids"]).astype(str)
        names = np.asarray(data["region_names"]).astype(str).tolist()
        boxes = np.asarray(data["roi_boxes_xyxy"], dtype=np.float32)
        valid = np.asarray(data["roi_valid"], dtype=bool)
    left = names.index("left_hand")
    right = names.index("right_hand")
    workspace = names.index("hand_workspace")
    person = names.index("full_body")
    windows = [uniform_indices(len(frame_ids)), motion_window(boxes, valid, left, right)]
    paths = frame_map(Path(row["ir_dir"]), "ir")
    clips: list[list[np.ndarray]] = []
    for chosen in windows:
        left_video: list[np.ndarray] = []
        right_video: list[np.ndarray] = []
        interaction_video: list[np.ndarray] = []
        for frame in chosen:
            image = read_ir(paths[frame_ids[frame]])
            left_box = fallback_box(boxes, valid, frame, left, workspace, person)
            right_box = fallback_box(boxes, valid, frame, right, workspace, person)
            both_box = interaction_box(
                boxes, valid, frame, left, right, workspace, person
            )
            left_video.append(square_crop(image, left_box, scale=1.85))
            right_video.append(square_crop(image, right_box, scale=1.85))
            interaction_video.append(square_crop(image, both_box, scale=1.55))
        clips.extend((left_video, right_video, interaction_video))
    return clips


class HandDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], labels: np.ndarray) -> None:
        self.rows = rows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clips = prepare_hand_clips(self.rows[index])
        return {
            "videos": clips,
            "clip_indices": list(range(6)),
            "label": int(self.labels[index]),
            "sample_id": self.rows[index]["sample_id"],
        }


@torch.inference_mode()
def extract(args: argparse.Namespace, cache: Path) -> dict[str, np.ndarray]:
    protocol = load_protocol()
    rows = read_aligned_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = build_model(device)
    snapshot = Path(snapshot_download(PROCESSOR_REPO, local_files_only=True))
    processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
    dataset = HandDataset(rows, protocol.labels)
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
        action = model.head(features)
        trials = len(batch["sample_ids"])
        feature_parts.append(features.reshape(trials, 2, 3, 768).cpu().numpy())
        action_parts.append(action.reshape(trials, 2, 3, 710).cpu().numpy())
        sample_ids.extend(batch["sample_ids"])
        if (batch_number + 1) % 20 == 0:
            print(
                f"  hand extracted={len(sample_ids)}/{len(dataset)} "
                f"elapsed_min={(time.time() - started) / 60:.1f}",
                flush=True,
            )
    if sample_ids != protocol.sample_ids.tolist():
        raise ValueError("hand extraction order changed")
    payload = {
        "sample_ids": protocol.sample_ids,
        "labels": protocol.labels,
        "users": protocol.users,
        "fold_id": protocol.fold_id,
        "features": np.concatenate(feature_parts),
        "action_logits": np.concatenate(action_parts),
    }
    np.savez_compressed(cache, **payload)
    (cache.parent / "cache_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "clips": "full/peak-motion x left-hand/right-hand/interaction",
                "elapsed_seconds": time.time() - started,
                "peak_cuda_gib": (
                    float(torch.cuda.max_memory_allocated() / 2**30)
                    if device.type == "cuda"
                    else 0.0
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def flattened(data: dict[str, np.ndarray]) -> np.ndarray:
    values = l2_normalize(np.asarray(data["features"], dtype=np.float32))
    return values.reshape(len(values), -1)


def mean_feature(data: dict[str, np.ndarray]) -> np.ndarray:
    values = l2_normalize(np.asarray(data["features"], dtype=np.float32))
    return l2_normalize(values.mean(axis=1)).reshape(len(values), -1)


def safe_h3(sample_ids: np.ndarray) -> np.ndarray:
    lookup: dict[str, int] = {}
    for split in load_splits().values():
        lookup.update(
            {str(key): int(value) for key, value in zip(split.sample_ids, split.safe_prediction)}
        )
    return np.asarray([lookup[str(value)] for value in sample_ids], dtype=np.int64)


def evaluate(data: dict[str, np.ndarray], output: Path) -> dict[str, Any]:
    protocol = load_protocol()
    sources = {
        "vmae": load_npz(VMAE),
        "iv2": load_npz(IV2),
        "depth": load_npz(DEPTH),
        "thermal": load_npz(THERMAL),
        "hand": data,
    }
    for name, source in sources.items():
        if not np.array_equal(source["sample_ids"].astype(str), protocol.sample_ids.astype(str)):
            raise ValueError(f"{name} order mismatch")
    all_features = {name: flattened(source) for name, source in sources.items()}
    means = {name: mean_feature(source) for name, source in sources.items()}
    hand_action = row_standardize(
        np.asarray(data["action_logits"], dtype=np.float32).reshape(len(protocol.labels), -1)
    )
    candidates = {
        "hand_all": all_features["hand"],
        "hand_mean": means["hand"],
        "hand_plus_k710": np.concatenate((all_features["hand"], hand_action), axis=1),
        "ir_two_plus_hand": np.concatenate(
            (all_features["vmae"], all_features["iv2"], all_features["hand"]), axis=1
        ),
        "all_visual_tokens": np.concatenate(tuple(all_features.values()), axis=1),
        "all_visual_means": np.concatenate(tuple(means.values()), axis=1),
    }
    recipes = {
        "hand_all": (3000.0, 0.75),
        "hand_mean": (1000.0, 0.75),
        "hand_plus_k710": (3000.0, 0.75),
        "ir_two_plus_hand": (3000.0, 0.75),
        "all_visual_tokens": (4000.0, 0.75),
        "all_visual_means": (1000.0, 0.75),
    }
    train = protocol.train_indices(0)
    val = protocol.val_indices(0)
    labels = protocol.labels
    with np.load(
        REPO_ROOT / "runs/p90_crossuser_visual_router_v1/gate_predictions.npz",
        allow_pickle=False,
    ) as router:
        router_lookup = {
            str(key): int(value)
            for key, value in zip(router["sample_ids"], router["router_prediction"])
        }
    base = np.asarray([router_lookup[str(value)] for value in protocol.sample_ids[val]])
    safe = safe_h3(protocol.sample_ids[val])
    results: dict[str, Any] = {}
    logits_out: dict[str, np.ndarray] = {}
    union = base == labels[val]
    for name, values in candidates.items():
        alpha, power = recipes[name]
        model = make_model(alpha)
        model.fit(
            values[train],
            labels[train],
            ridge__sample_weight=class_sample_weights(labels[train], power),
        )
        logits = np.asarray(model.decision_function(values[val]), dtype=np.float32)
        prediction = logits.argmax(axis=1)
        candidate_correct = prediction == labels[val]
        results[name] = {
            "dimensions": int(values.shape[1]),
            "metrics": classification_metrics(logits, labels[val]),
            "rescue_over_p90": int(np.sum(~(base == labels[val]) & candidate_correct)),
            "harm_if_replaced": int(np.sum((base == labels[val]) & ~candidate_correct)),
            "union_oracle_accuracy": float(np.mean((base == labels[val]) | candidate_correct)),
        }
        union |= candidate_correct
        logits_out[name] = logits
        print(f"  {name}: {results[name]}", flush=True)
    np.savez_compressed(
        output / "fold0_logits.npz",
        sample_ids=protocol.sample_ids[val],
        labels=labels[val],
        p90_prediction=base,
        p89_safe_prediction=safe,
        **{f"{name}_logits": value for name, value in logits_out.items()},
    )
    return {
        "protocol": "P90 fold0 only; fixed Ridge recipes",
        "baselines": {
            "p89_safe": float(np.mean(safe == labels[val])),
            "p90_visual_router": float(np.mean(base == labels[val])),
        },
        "candidates": results,
        "all_candidate_union_oracle_accuracy": float(union.mean()),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "complete_features.npz"
    data = load_npz(cache) if args.reuse_cache and cache.exists() else extract(args, cache)
    report = evaluate(data, output)
    (output / "fold0_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
