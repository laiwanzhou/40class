from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from audit_yolo11_pose_skeleton import frame_map
from build_p30_shared_dir_roi_feature_cache import read_ir
from build_p46_videomae_cache import (
    VIEW_NAMES,
    atomic_npz,
    encode,
    restore_legacy_attention_biases,
    safe_relative,
    square_crop,
)
from p46_protocol import DEFAULT_MANIFEST


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_P29 = PROJECT_DIR / "runs/p29_dir_multiscale_roi_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p46_videomae_large_multiclip_v1"
MODEL_NAME = "MCG-NJU/videomae-large-finetuned-kinetics"
WINDOW_NAMES = ("early", "late")
WINDOW_BOUNDS = ((0.0, 0.70), (0.30, 1.0))
FRAME_COUNT = 16
HIDDEN_SIZE = 1024
CLASSIFIER_CLASSES = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract early/late Large VideoMAE features at 3 synchronized spatial views."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p29-run", type=Path, default=DEFAULT_P29)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trial-batch", type=int, default=1)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["detail_selected"] == "1"]
    rows.sort(key=lambda row: row["sample_id"])
    if len(rows) != 1384:
        raise RuntimeError(f"Frozen Detail21 count changed: {len(rows)}")
    return rows


def window_indices(frame_count: int, low: float, high: float) -> np.ndarray:
    if frame_count < 1 or not 0.0 <= low < high <= 1.0:
        raise ValueError("Invalid temporal window")
    end = frame_count - 1
    return np.rint(np.linspace(low * end, high * end, FRAME_COUNT)).astype(np.int64)


def prepare_trial(
    row: dict[str, str], p29_run: Path
) -> tuple[list[list[np.ndarray]], dict[str, np.ndarray]]:
    p29_path = p29_run / "trial_roi_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
    with np.load(p29_path, allow_pickle=False) as data:
        frame_ids = np.asarray(data["frame_ids"]).astype(str)
        region_names = tuple(np.asarray(data["region_names"]).astype(str))
        boxes = np.asarray(data["roi_boxes_xyxy"], dtype=np.float32)
        valid = np.asarray(data["roi_valid"], dtype=bool)
        quality = np.asarray(data["roi_quality"], dtype=np.float32)
    person_index = region_names.index("full_body")
    workspace_index = region_names.index("hand_workspace")
    ir_paths = frame_map(Path(row["ir_dir"]), "ir")
    clips: list[list[np.ndarray]] = []
    all_indices: list[np.ndarray] = []
    person_valid: list[np.ndarray] = []
    workspace_valid: list[np.ndarray] = []
    person_quality: list[np.ndarray] = []
    workspace_quality: list[np.ndarray] = []
    for low, high in WINDOW_BOUNDS:
        chosen = window_indices(len(frame_ids), low, high)
        all_indices.append(chosen)
        scene: list[np.ndarray] = []
        person: list[np.ndarray] = []
        workspace: list[np.ndarray] = []
        for index in chosen:
            frame_id = frame_ids[index]
            if frame_id not in ir_paths:
                raise RuntimeError(f"Missing IR frame {frame_id}: {row['source_id']}")
            image = read_ir(ir_paths[frame_id])
            person_box = boxes[index, person_index] if valid[index, person_index] else np.full(4, np.nan)
            workspace_box = boxes[index, workspace_index] if valid[index, workspace_index] else person_box
            scene.append(image)
            person.append(square_crop(image, person_box, scale=1.15))
            workspace.append(square_crop(image, workspace_box, scale=1.40))
        clips.extend((scene, person, workspace))
        person_valid.append(valid[chosen, person_index].astype(np.uint8))
        workspace_valid.append(valid[chosen, workspace_index].astype(np.uint8))
        person_quality.append(quality[chosen, person_index].astype(np.float32))
        workspace_quality.append(quality[chosen, workspace_index].astype(np.float32))
    return clips, {
        "frame_indices": np.stack(all_indices).astype(np.int16),
        "person_valid": np.stack(person_valid),
        "workspace_valid": np.stack(workspace_valid),
        "person_quality": np.stack(person_quality),
        "workspace_quality": np.stack(workspace_quality),
    }


def valid_cache(path: Path, row: dict[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["sample_id"].item()) == row["sample_id"]
                and int(data["label"].item()) == int(row["class_id"])
                and data["features"].shape == (2, 3, HIDDEN_SIZE)
                and data["kinetics_logits"].shape == (2, 3, CLASSIFIER_CLASSES)
                and np.isfinite(data["features"]).all()
                and np.isfinite(data["kinetics_logits"]).all()
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def aggregate(rows: list[dict[str, str]], output: Path) -> Path:
    sample_ids: list[str] = []
    source_ids: list[str] = []
    users: list[str] = []
    labels: list[int] = []
    features: list[np.ndarray] = []
    kinetics: list[np.ndarray] = []
    for row in rows:
        path = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
        if not valid_cache(path, row):
            raise RuntimeError(f"Missing or invalid multi-clip cache: {row['source_id']}")
        with np.load(path, allow_pickle=False) as data:
            sample_ids.append(row["sample_id"])
            source_ids.append(row["source_id"])
            users.append(row["user_id"])
            labels.append(int(row["class_id"]))
            features.append(np.asarray(data["features"], dtype=np.float16))
            kinetics.append(np.asarray(data["kinetics_logits"], dtype=np.float16))
    path = output / "complete_features.npz"
    atomic_npz(
        path,
        sample_ids=np.asarray(sample_ids),
        source_ids=np.asarray(source_ids),
        users=np.asarray(users),
        labels=np.asarray(labels, dtype=np.int64),
        window_names=np.asarray(WINDOW_NAMES),
        view_names=np.asarray(VIEW_NAMES),
        features=np.stack(features),
        kinetics_logits=np.stack(kinetics),
    )
    return path


def main() -> None:
    args = parse_args()
    rows = read_rows(args.manifest.resolve())
    if args.max_trials > 0:
        rows = rows[: args.max_trials]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            args.model,
            allow_patterns=("*.json", "*.safetensors", "*.txt"),
            local_files_only=True,
        )
    )
    processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = VideoMAEForVideoClassification.from_pretrained(snapshot, local_files_only=True)
    bias_report = restore_legacy_attention_biases(model, snapshot)
    if int(model.config.hidden_size) != HIDDEN_SIZE or int(model.config.num_labels) != CLASSIFIER_CLASSES:
        raise RuntimeError("Large VideoMAE architecture changed")
    device = torch.device(args.device)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    completed = skipped = 0
    peak_cuda_gib = 0.0
    for batch_start in range(0, len(rows), args.trial_batch):
        batch_rows = rows[batch_start : batch_start + args.trial_batch]
        work_rows: list[dict[str, str]] = []
        videos: list[list[np.ndarray]] = []
        metadata: list[dict[str, np.ndarray]] = []
        for row in batch_rows:
            cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
            if not args.overwrite and valid_cache(cache, row):
                skipped += 1
                continue
            row_videos, row_metadata = prepare_trial(row, args.p29_run.resolve())
            work_rows.append(row)
            videos.extend(row_videos)
            metadata.append(row_metadata)
        if work_rows:
            feature, logits, peak = encode(model, processor, videos, device)
            peak_cuda_gib = max(peak_cuda_gib, peak)
            feature = feature.reshape(len(work_rows), 2, 3, HIDDEN_SIZE)
            logits = logits.reshape(len(work_rows), 2, 3, CLASSIFIER_CLASSES)
            for index, row in enumerate(work_rows):
                cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
                atomic_npz(
                    cache,
                    sample_id=np.asarray(row["sample_id"]),
                    source_id=np.asarray(row["source_id"]),
                    user=np.asarray(row["user_id"]),
                    label=np.asarray(int(row["class_id"]), dtype=np.int64),
                    window_names=np.asarray(WINDOW_NAMES),
                    view_names=np.asarray(VIEW_NAMES),
                    features=feature[index].astype(np.float16),
                    kinetics_logits=logits[index].astype(np.float16),
                    **metadata[index],
                )
                completed += 1
        processed = min(batch_start + len(batch_rows), len(rows))
        if processed == len(rows) or processed % 20 == 0:
            print(
                json.dumps(
                    {
                        "stage": "cache_progress",
                        "processed": processed,
                        "total": len(rows),
                        "built": completed,
                        "skipped": skipped,
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                        "peak_cuda_gib": round(peak_cuda_gib, 3),
                    }
                ),
                flush=True,
            )
    complete = aggregate(rows, output) if len(rows) == 1384 else None
    summary = {
        "protocol": "frozen Large VideoMAE early/late x scene/person/workspace IR features",
        "model": args.model,
        "snapshot": str(snapshot),
        "attention_bias_compatibility": bias_report,
        "windows": list(WINDOW_NAMES),
        "window_bounds": WINDOW_BOUNDS,
        "views": list(VIEW_NAMES),
        "trials": len(rows),
        "built": completed,
        "skipped": skipped,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_gib": peak_cuda_gib,
        "complete_features": str(complete) if complete is not None else None,
    }
    (output / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
