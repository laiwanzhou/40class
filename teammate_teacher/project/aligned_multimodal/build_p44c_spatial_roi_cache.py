from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_yolo11_pose_skeleton import atomic_json, frame_map, safe_name, write_csv
from build_p30_shared_dir_roi_feature_cache import (
    atomic_feature_npz,
    pixel_roi_crops,
    read_depth,
    read_ir,
)
from p30_shared_dir_roi_model import (
    MODALITY_NAMES,
    REGION_NAMES,
    SharedResNet18Pyramid,
    imagenet_normalize,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "p27_strong_inner" / "fold_0.csv"
DEFAULT_ROI_RUN = PROJECT_DIR / "runs" / "p29_dir_multiscale_roi_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p44c_spatial_roi_fold0"
LOCAL_REGIONS = ("left_hand", "right_hand", "hand_workspace")
SPATIAL_GRID = 3
SPATIAL_DIM = 128
GROUP_A = {6, 7, 8, 9, 10, 11, 14, 37}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build P44-C layer2 3x3 spatial tokens for fold0 train and Group-A val."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--frame-batch", type=int, default=16)
    parser.add_argument("--backbone-batch", type=int, default=192)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_id(row: dict[str, str]) -> str:
    return f"{row['class_name']}/{row['user_id']}/{row['trial_id']}"


def roi_path(roi_run: Path, row: dict[str, str]) -> Path:
    return roi_run / "trial_roi_cache" / safe_name(source_id(row)).with_suffix(".npz")


def output_path(output: Path, row: dict[str, str]) -> Path:
    return output / "trial_spatial_cache" / safe_name(source_id(row)).with_suffix(".npz")


def selected_rows(path: Path, maximum: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if row["split"] == "train"
        or (row["split"] == "val" and int(row["class_id"]) in GROUP_A)
    ]
    rows.sort(key=lambda row: (row["split"], int(row["class_id"]), row["user_id"], row["trial_id"]))
    return rows[:maximum] if maximum > 0 else rows


@torch.inference_mode()
def extract_trial(
    row: dict[str, str],
    cached_roi: Path,
    backbone: SharedResNet18Pyramid,
    device: torch.device,
    crop_size: int,
    frame_batch: int,
    backbone_batch: int,
) -> dict[str, np.ndarray]:
    with np.load(cached_roi, allow_pickle=False) as roi:
        arrays = {key: np.asarray(roi[key]) for key in roi.files}
    if tuple(str(value) for value in arrays["region_names"]) != REGION_NAMES:
        raise RuntimeError(f"P29 region mismatch: {source_id(row)}")
    frame_ids = [str(value) for value in arrays["frame_ids"]]
    depth_paths = frame_map(Path(row["depth_dir"]), "depth")
    ir_paths = frame_map(Path(row["ir_dir"]), "ir")
    missing = [fid for fid in frame_ids if fid not in depth_paths or fid not in ir_paths]
    if missing:
        raise RuntimeError(f"missing synchronized frame {missing[0]}: {source_id(row)}")

    local_indices = [REGION_NAMES.index(name) for name in LOCAL_REGIONS]
    output = np.zeros(
        (
            len(frame_ids),
            len(MODALITY_NAMES),
            len(LOCAL_REGIONS),
            SPATIAL_GRID,
            SPATIAL_GRID,
            SPATIAL_DIM,
        ),
        dtype=np.float16,
    )
    use_amp = device.type == "cuda"
    for start in range(0, len(frame_ids), frame_batch):
        stop = min(start + frame_batch, len(frame_ids))
        chosen = frame_ids[start:stop]
        depth = np.stack([read_depth(depth_paths[fid]) for fid in chosen])
        ir = np.stack([read_ir(ir_paths[fid]) for fid in chosen])
        pixels = torch.from_numpy(np.concatenate((depth, ir), axis=0)).to(
            device=device, dtype=torch.float32
        )
        pixels = pixels.permute(0, 3, 1, 2).div_(255.0)
        boxes = torch.from_numpy(arrays["roi_boxes_xyxy"][start:stop, local_indices]).to(
            device=device, dtype=torch.float32
        )
        valid = torch.from_numpy(arrays["roi_valid"][start:stop, local_indices]).to(
            device=device, dtype=torch.bool
        )

        # pixel_roi_crops expects the seven-region layout.  ROIAlign directly is
        # cheaper here, so create a temporary 3-region implementation inline.
        n = stop - start
        safe_boxes = boxes.clone()
        fallback = torch.from_numpy(
            arrays["roi_boxes_xyxy"][start:stop, REGION_NAMES.index("global_fallback")]
        ).to(device=device, dtype=torch.float32)
        safe_boxes[~valid] = fallback[:, None, :].expand_as(safe_boxes)[~valid]
        repeated = safe_boxes.unsqueeze(0).expand(len(MODALITY_NAMES), -1, -1, -1)
        image_indices = torch.arange(
            len(MODALITY_NAMES) * n, device=device, dtype=boxes.dtype
        ).reshape(len(MODALITY_NAMES), n, 1).expand(-1, -1, len(LOCAL_REGIONS))
        rois = torch.cat((image_indices.reshape(-1, 1), repeated.reshape(-1, 4)), dim=1)
        from torchvision.ops import roi_align

        crops = roi_align(
            pixels,
            rois,
            output_size=(crop_size, crop_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        maps: list[torch.Tensor] = []
        for crop_start in range(0, len(crops), backbone_batch):
            normalized = imagenet_normalize(crops[crop_start : crop_start + backbone_batch])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                layer2, _, _ = backbone.forward_feature_maps(normalized)
                maps.append(torch.nn.functional.adaptive_avg_pool2d(layer2, SPATIAL_GRID).float().cpu())
        grid = torch.cat(maps).reshape(
            len(MODALITY_NAMES), n, len(LOCAL_REGIONS), SPATIAL_DIM, SPATIAL_GRID, SPATIAL_GRID
        )
        grid = grid.permute(1, 0, 2, 4, 5, 3).numpy().astype(np.float16)
        grid *= arrays["roi_valid"][start:stop, None, local_indices, None, None, None]
        output[start:stop] = grid

    return {
        "frame_ids": arrays["frame_ids"],
        "modality_names": np.asarray(MODALITY_NAMES),
        "region_names": np.asarray(LOCAL_REGIONS),
        "spatial_features": output,
        "roi_valid": arrays["roi_valid"][:, local_indices],
        "roi_quality": arrays["roi_quality"][:, local_indices],
        "roi_clipped_ratio": arrays["roi_clipped_ratio"][:, local_indices],
        "pose_quality_factor": arrays["pose_quality_factor"],
    }


def main() -> None:
    args = parse_args()
    rows = selected_rows(args.manifest.resolve(), args.max_trials)
    if not rows:
        raise RuntimeError("no fold0 rows selected")
    missing = [source_id(row) for row in rows if not roi_path(args.roi_run, row).is_file()]
    if missing:
        raise RuntimeError(f"missing P29 ROI cache: {missing[:3]}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    backbone = SharedResNet18Pyramid(imagenet_pretrained=True).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    config = {
        "stage": "P44-C spatial ROI cache",
        "manifest": str(args.manifest.resolve()),
        "selected_trials": len(rows),
        "selection": "all 1348 fold0-train trials plus 201 Group-A fold0-val trials",
        "regions": list(LOCAL_REGIONS),
        "modalities": list(MODALITY_NAMES),
        "feature": "shared ImageNet ResNet18 layer2, adaptive 3x3 grid, no GAP",
        "shape_per_trial": "[T,2,3,3,3,128] fp16",
        "all_frame_policy": "all synchronized frames; adjacent difference is computed by the model",
        "device": str(device),
    }
    atomic_json(output / "config.json", config)
    summaries: list[dict[str, Any]] = []
    begun = time.perf_counter()
    for index, row in enumerate(rows, 1):
        target = output_path(output, row)
        started = time.perf_counter()
        if target.is_file() and not args.overwrite:
            with np.load(target, allow_pickle=False) as data:
                frames = len(data["frame_ids"])
            status = "cached"
        else:
            arrays = extract_trial(
                row,
                roi_path(args.roi_run, row),
                backbone,
                device,
                args.crop_size,
                args.frame_batch,
                args.backbone_batch,
            )
            atomic_feature_npz(target, arrays, compressed=True)
            frames = len(arrays["frame_ids"])
            status = "built"
        seconds = time.perf_counter() - started
        summaries.append(
            {
                "source_id": source_id(row),
                "split": row["split"],
                "class_id": int(row["class_id"]),
                "user_id": row["user_id"],
                "frames": frames,
                "status": status,
                "seconds": seconds,
                "bytes": target.stat().st_size,
            }
        )
        print(
            f"[{index}/{len(rows)}] {source_id(row)} frames={frames} {status} "
            f"{seconds:.2f}s {target.stat().st_size / 1024**2:.2f}MiB",
            flush=True,
        )
    write_csv(output / "trial_summary.csv", summaries)
    built = [row for row in summaries if row["status"] == "built"]
    summary = {
        **config,
        "completed_trials": len(summaries),
        "completed_frames": sum(int(row["frames"]) for row in summaries),
        "cache_bytes": sum(int(row["bytes"]) for row in summaries),
        "built_trials": len(built),
        "elapsed_seconds": time.perf_counter() - begun,
        "new_frames_per_second": sum(int(row["frames"]) for row in built)
        / max(sum(float(row["seconds"]) for row in built), 1e-6),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
