from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from audit_yolo11_pose_skeleton import frame_map
from build_p30_shared_dir_roi_feature_cache import read_depth
from build_p46_videomae_cache import (
    DEFAULT_P29,
    MODEL_NAME,
    PROJECT_DIR,
    VIEW_NAMES,
    atomic_npz,
    encode,
    read_rows,
    restore_legacy_attention_biases,
    safe_relative,
    square_crop,
    uniform_indices,
)
from p46_protocol import DEFAULT_MANIFEST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a frozen VideoMAE expert from Depth_Color or Thermal frames."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p29-run", type=Path, default=DEFAULT_P29)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=("depth", "thermal"), required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trial-batch", type=int, default=4)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def thermal_dir(row: dict[str, str]) -> Path:
    source = str(Path(row["ir_dir"]).resolve())
    marker = f"{Path().anchor}IR{Path().anchor}" if Path().anchor else ""
    # Windows paths are stored in the manifest. Replace exactly one directory
    # component without relying on the current platform's path separator rules.
    candidates = ("\\IR\\", "/IR/")
    for candidate in candidates:
        if candidate in source:
            return Path(source.replace(candidate, candidate[0] + "Thermal" + candidate[-1], 1))
    source_path = Path(source)
    if source_path.name.casefold() == "ir":
        return source_path.parent / "Thermal"
    raise RuntimeError(f"could not derive Thermal path from {row['ir_dir']}")


def read_thermal(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read Thermal image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_trial(
    row: dict[str, str], p29_run: Path, modality: str
) -> tuple[list[list[np.ndarray]], dict[str, np.ndarray]]:
    p29_path = p29_run / "trial_roi_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
    with np.load(p29_path, allow_pickle=False) as data:
        ir_frame_ids = np.asarray(data["frame_ids"]).astype(str)
        region_names = tuple(np.asarray(data["region_names"]).astype(str))
        boxes = np.asarray(data["roi_boxes_xyxy"], dtype=np.float32)
        valid = np.asarray(data["roi_valid"], dtype=bool)
    person_index = region_names.index("full_body")
    workspace_index = region_names.index("hand_workspace")
    if modality == "depth":
        paths = frame_map(Path(row["depth_dir"]), "depth")
        chosen_roi = uniform_indices(len(ir_frame_ids))
        selected_paths = [paths[ir_frame_ids[index]] for index in chosen_roi]
        reader = read_depth
    else:
        files = sorted(thermal_dir(row).glob("*.jpg"))
        if not files:
            raise RuntimeError(f"missing Thermal frames: {row['source_id']}")
        chosen_thermal = uniform_indices(len(files))
        selected_paths = [files[index] for index in chosen_thermal]
        if len(files) == 1:
            chosen_roi = np.zeros(16, dtype=np.int64)
        else:
            chosen_roi = np.rint(
                chosen_thermal / (len(files) - 1) * (len(ir_frame_ids) - 1)
            ).astype(np.int64)
        reader = read_thermal
    scene: list[np.ndarray] = []
    person: list[np.ndarray] = []
    workspace: list[np.ndarray] = []
    for path, roi_index in zip(selected_paths, chosen_roi):
        image = reader(path)
        height, width = image.shape[:2]
        scale = np.asarray([width / 640.0, height / 480.0] * 2, dtype=np.float32)
        person_box = (
            boxes[roi_index, person_index] * scale
            if valid[roi_index, person_index]
            else np.asarray([np.nan] * 4)
        )
        workspace_box = (
            boxes[roi_index, workspace_index] * scale
            if valid[roi_index, workspace_index]
            else person_box
        )
        scene.append(image)
        person.append(square_crop(image, person_box, scale=1.15))
        workspace.append(square_crop(image, workspace_box, scale=1.40))
    return [scene, person, workspace], {
        "roi_frame_indices": chosen_roi.astype(np.int16),
        "person_valid": valid[chosen_roi, person_index].astype(np.uint8),
        "workspace_valid": valid[chosen_roi, workspace_index].astype(np.uint8),
    }


def valid_cache(path: Path, row: dict[str, str], modality: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["sample_id"].item()) == row["sample_id"]
                and str(data["modality"].item()) == modality
                and int(data["label"].item()) == int(row["class_id"])
                and int(data["modality_available"].item()) in {0, 1}
                and data["features"].shape == (3, 768)
                and data["kinetics_logits"].shape == (3, 400)
                and np.isfinite(data["features"]).all()
                and np.isfinite(data["kinetics_logits"]).all()
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def aggregate(rows: list[dict[str, str]], output: Path, modality: str) -> Path:
    sample_ids: list[str] = []
    source_ids: list[str] = []
    users: list[str] = []
    labels: list[int] = []
    features: list[np.ndarray] = []
    kinetics_logits: list[np.ndarray] = []
    modality_available: list[int] = []
    for row in rows:
        path = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
        if not valid_cache(path, row, modality):
            raise RuntimeError(f"missing or invalid {modality} cache: {row['source_id']}")
        with np.load(path, allow_pickle=False) as data:
            sample_ids.append(row["sample_id"])
            source_ids.append(row["source_id"])
            users.append(row["user_id"])
            labels.append(int(row["class_id"]))
            features.append(np.asarray(data["features"], dtype=np.float16))
            kinetics_logits.append(np.asarray(data["kinetics_logits"], dtype=np.float16))
            modality_available.append(int(data["modality_available"].item()))
    path = output / "complete_features.npz"
    atomic_npz(
        path,
        sample_ids=np.asarray(sample_ids),
        source_ids=np.asarray(source_ids),
        users=np.asarray(users),
        labels=np.asarray(labels, dtype=np.int64),
        view_names=np.asarray(VIEW_NAMES),
        modality=np.asarray(modality),
        features=np.stack(features),
        kinetics_logits=np.stack(kinetics_logits),
        modality_available=np.asarray(modality_available, dtype=np.uint8),
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
    device = torch.device(args.device)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    completed = 0
    skipped = 0
    missing_modality = 0
    peak_cuda_gib = 0.0
    for batch_start in range(0, len(rows), args.trial_batch):
        batch_rows = rows[batch_start : batch_start + args.trial_batch]
        work_rows: list[dict[str, str]] = []
        videos: list[list[np.ndarray]] = []
        metadata: list[dict[str, np.ndarray]] = []
        for row in batch_rows:
            cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
            if not args.overwrite and valid_cache(cache, row, args.modality):
                skipped += 1
                continue
            if args.modality == "thermal" and not any(thermal_dir(row).glob("*.jpg")):
                atomic_npz(
                    cache,
                    sample_id=np.asarray(row["sample_id"]),
                    source_id=np.asarray(row["source_id"]),
                    user=np.asarray(row["user_id"]),
                    label=np.asarray(int(row["class_id"]), dtype=np.int64),
                    modality=np.asarray(args.modality),
                    modality_available=np.asarray(0, dtype=np.uint8),
                    view_names=np.asarray(VIEW_NAMES),
                    features=np.zeros((3, 768), dtype=np.float16),
                    kinetics_logits=np.zeros((3, 400), dtype=np.float16),
                    roi_frame_indices=np.zeros(16, dtype=np.int16),
                    person_valid=np.zeros(16, dtype=np.uint8),
                    workspace_valid=np.zeros(16, dtype=np.uint8),
                )
                completed += 1
                missing_modality += 1
                continue
            trial_videos, trial_metadata = prepare_trial(
                row, args.p29_run.resolve(), args.modality
            )
            work_rows.append(row)
            videos.extend(trial_videos)
            metadata.append(trial_metadata)
        if work_rows:
            feature, logits, peak = encode(model, processor, videos, device)
            peak_cuda_gib = max(peak_cuda_gib, peak)
            feature = feature.reshape(len(work_rows), 3, 768)
            logits = logits.reshape(len(work_rows), 3, 400)
            for index, row in enumerate(work_rows):
                cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
                atomic_npz(
                    cache,
                    sample_id=np.asarray(row["sample_id"]),
                    source_id=np.asarray(row["source_id"]),
                    user=np.asarray(row["user_id"]),
                    label=np.asarray(int(row["class_id"]), dtype=np.int64),
                    modality=np.asarray(args.modality),
                    modality_available=np.asarray(1, dtype=np.uint8),
                    view_names=np.asarray(VIEW_NAMES),
                    features=feature[index].astype(np.float16),
                    kinetics_logits=logits[index].astype(np.float16),
                    **metadata[index],
                )
                completed += 1
        processed = min(batch_start + len(batch_rows), len(rows))
        if processed == len(rows) or processed % max(20, args.trial_batch) == 0:
            print(
                json.dumps(
                    {
                        "stage": f"{args.modality}_cache_progress",
                        "processed": processed,
                        "total": len(rows),
                        "built": completed,
                        "skipped": skipped,
                        "missing_modality": missing_modality,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    complete = aggregate(rows, output, args.modality)
    summary: dict[str, Any] = {
        "protocol": f"frozen Kinetics VideoMAE {args.modality} scene/person/workspace features",
        "model": args.model,
        "modality": args.modality,
        "attention_bias_compatibility": bias_report,
        "trials": len(rows),
        "built": completed,
        "skipped": skipped,
        "missing_modality": missing_modality,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_gib": peak_cuda_gib,
        "complete_features": str(complete),
    }
    (output / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
