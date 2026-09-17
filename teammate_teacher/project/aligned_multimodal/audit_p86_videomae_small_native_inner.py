from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from audit_p86_videomae_small_frozen_inner import metrics, selection_score
from build_p46_videomae_cache import encode, restore_legacy_attention_biases
from build_p46_videomae_multiclip_cache import prepare_trial
from build_p85_videomae_full40_multiclip_cache import read_rows
from p86_videomae_small_visual_model import DEFAULT_VIDEOMAE_SMALL, resolve_snapshot
from train_p46_videomae_head import l2_normalize, make_model, row_standardize
from train_p85_videomae_full40_head import sample_weights
from train_p86_visual_student_oof import split_universe


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P29 = PROJECT_DIR / "runs/p29_dir_multiscale_roi_full"
DEFAULT_TEACHER_LOGITS = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p86_videomae_small_native_probe_fold0_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official 16x224 VideoMAE-Small frozen probe on fold0 inner subjects."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p29-run", type=Path, default=DEFAULT_P29)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrained-model", default=DEFAULT_VIDEOMAE_SMALL)
    parser.add_argument("--outer-fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--trial-batch", type=int, default=8)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = load_npz(args.teacher_logits)
    split = split_universe(teacher, args.outer_fold, args.seed)
    sample_ids = np.asarray(split["sample_ids"]).astype(str)
    selected_ids = np.concatenate(
        (sample_ids[split["inner_train"]], sample_ids[split["inner_dev"]])
    )
    row_lookup = {row["sample_id"]: row for row in read_rows(args.manifest.resolve())}
    rows = [row_lookup[value] for value in selected_ids]

    snapshot = resolve_snapshot(args.pretrained_model)
    processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = VideoMAEForVideoClassification.from_pretrained(snapshot, local_files_only=True)
    load_audit = restore_legacy_attention_biases(model, snapshot)
    if (
        int(model.config.hidden_size) != 384
        or int(model.config.num_frames) != 16
        or int(model.config.image_size) != 224
        or int(model.config.num_labels) != 400
    ):
        raise RuntimeError("official VideoMAE-Small geometry changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    feature_rows: list[np.ndarray] = []
    kinetics_rows: list[np.ndarray] = []
    started = time.perf_counter()
    peak_cuda_gib = 0.0
    for batch_start in range(0, len(rows), args.trial_batch):
        batch_rows = rows[batch_start : batch_start + args.trial_batch]
        videos: list[list[np.ndarray]] = []
        for row in batch_rows:
            row_videos, _ = prepare_trial(row, args.p29_run.resolve())
            videos.extend(row_videos)
        feature, kinetics, peak = encode(model, processor, videos, device)
        peak_cuda_gib = max(peak_cuda_gib, peak)
        feature_rows.append(feature.reshape(len(batch_rows), 2, 3, 384))
        kinetics_rows.append(kinetics.reshape(len(batch_rows), 2, 3, 400))
        processed = min(batch_start + len(batch_rows), len(rows))
        if processed == len(rows) or processed % 80 == 0:
            print(
                json.dumps(
                    {
                        "extracted": processed,
                        "total": len(rows),
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                        "peak_cuda_gib": round(peak_cuda_gib, 3),
                    }
                ),
                flush=True,
            )

    features = np.concatenate(feature_rows).astype(np.float32)
    kinetics = np.concatenate(kinetics_rows).astype(np.float32)
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in rows]).astype(str)
    train_count = len(split["inner_train"])
    train_index = np.arange(train_count)
    dev_index = np.arange(train_count, len(rows))
    normalized = l2_normalize(features)
    standardized_kinetics = row_standardize(kinetics.reshape(len(rows), -1))
    matrices = {
        "six_clip_features": normalized.reshape(len(rows), -1),
        "kinetics_logits": standardized_kinetics,
        "features_plus_kinetics": np.concatenate(
            (normalized.reshape(len(rows), -1), standardized_kinetics), axis=1
        ),
    }
    candidates: list[dict[str, Any]] = []
    for feature_name, values in matrices.items():
        for power in (0.35, 0.75):
            weights = sample_weights(labels[train_index], power)
            for alpha in (300.0, 1000.0, 3000.0):
                head = make_model(alpha)
                head.fit(
                    values[train_index],
                    labels[train_index],
                    ridge__sample_weight=weights,
                )
                prediction = head.predict(values[dev_index]).astype(np.int64)
                value_metrics = metrics(labels[dev_index], prediction, users[dev_index])
                candidates.append(
                    {
                        "feature_set": feature_name,
                        "class_weight_power": power,
                        "alpha": alpha,
                        "selection_score": selection_score(value_metrics),
                        **value_metrics,
                    }
                )
    candidates.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    np.savez_compressed(
        output / "inner_native_features.npz",
        sample_ids=selected_ids,
        users=users,
        labels=labels,
        split=np.asarray(["inner_train"] * train_count + ["inner_dev"] * len(dev_index)),
        features=features.astype(np.float16),
        kinetics_logits=kinetics.astype(np.float16),
    )
    summary = {
        "stage": "P86_VideoMAE_Small_native_geometry_inner_probe",
        "protocol": (
            "Stream raw IR into the official frozen 16x224 Kinetics checkpoint. Only fold0 "
            "inner-train fits Ridge probes and only inner-dev evaluates them; outer-held is not read."
        ),
        "counts": {"inner_train": int(train_count), "inner_dev": int(len(dev_index))},
        "geometry": {"frames": 16, "resolution": 224, "clips": 6},
        "pretrained_model": args.pretrained_model,
        "pretrained_load_audit": load_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_gib": peak_cuda_gib,
        "best": candidates[0],
        "candidates": candidates,
        "outer_held_evaluated": False,
        "test_used_for_selection": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
