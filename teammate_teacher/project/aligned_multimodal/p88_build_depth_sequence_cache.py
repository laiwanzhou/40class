from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from audit_yolo11_pose_skeleton import frame_map
from build_p46_videomae_cache import safe_relative, square_crop
from build_p86_mc3_sequence_cache import load_model
from depth_encoding import decode_jet_rgb, resize_decoded_depth


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_PIXELS = PROJECT_DIR / "runs/p86_visual_pixel_cache_t16_r160_v12"
DEFAULT_P29 = PROJECT_DIR / "runs/p29_dir_multiscale_roi_full"
DEFAULT_CHECKPOINT = PROJECT_DIR / "runs/p87s_visual_holdout1_v1/visual_student.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_depth_sequence_holdout1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode registered JET Depth and run it through the frozen P87 MC3 "
            "backbone. Only P88 caches are written; the P87 checkpoint is read-only."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--p29-run", type=Path, default=DEFAULT_P29)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-trials", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_decoded_depth(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read Depth frame: {path}")
    if bgr.shape[:2] != (480, 640):
        bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return decode_jet_rgb(rgb)


def resize_crop(
    depth: np.ndarray,
    valid: np.ndarray,
    box: np.ndarray,
    scale: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth_crop = square_crop(depth, box, scale)
    valid_crop = square_crop(valid, box, scale)
    return resize_decoded_depth(
        depth_crop, valid_crop, height=resolution, width=resolution
    )


def prepare_trial(
    row: dict[str, str],
    p29_run: Path,
    chosen_windows: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    roi_path = (
        p29_run / "trial_roi_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
    )
    with np.load(roi_path, allow_pickle=False) as data:
        frame_ids = np.asarray(data["frame_ids"]).astype(str)
        region_names = tuple(np.asarray(data["region_names"]).astype(str))
        boxes = np.asarray(data["roi_boxes_xyxy"], dtype=np.float32)
        roi_valid = np.asarray(data["roi_valid"], dtype=bool)
    person_index = region_names.index("full_body")
    workspace_index = region_names.index("hand_workspace")
    depth_paths = frame_map(Path(row["depth_dir"]), "depth")
    images = np.zeros(
        (chosen_windows.shape[0], chosen_windows.shape[1], 3, resolution, resolution),
        dtype=np.uint8,
    )
    valid_fraction = np.zeros(images.shape[:3], dtype=np.float32)
    repaired_total = 0
    decoded_frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame_index in sorted(set(chosen_windows.reshape(-1).tolist())):
        frame_id = frame_ids[int(frame_index)]
        path = depth_paths.get(frame_id)
        if path is None:
            raise RuntimeError(f"missing registered Depth frame {frame_id}: {row['source_id']}")
        decoded, valid, repaired = read_decoded_depth(path)
        decoded_frames[int(frame_index)] = (decoded, valid)
        repaired_total += repaired
    for window_index, chosen in enumerate(chosen_windows):
        for time_index, frame_index_value in enumerate(chosen):
            frame_index = int(frame_index_value)
            decoded, valid = decoded_frames[frame_index]
            person_box = (
                boxes[frame_index, person_index]
                if roi_valid[frame_index, person_index]
                else np.full(4, np.nan, dtype=np.float32)
            )
            workspace_box = (
                boxes[frame_index, workspace_index]
                if roi_valid[frame_index, workspace_index]
                else person_box
            )
            scene_depth, scene_valid = resize_decoded_depth(
                decoded, valid, height=resolution, width=resolution
            )
            person_depth, person_valid = resize_crop(
                decoded, valid, person_box, 1.15, resolution
            )
            workspace_depth, workspace_valid = resize_crop(
                decoded, valid, workspace_box, 1.40, resolution
            )
            for view_index, (depth_view, valid_view) in enumerate(
                (
                    (scene_depth, scene_valid),
                    (person_depth, person_valid),
                    (workspace_depth, workspace_valid),
                )
            ):
                images[window_index, time_index, view_index] = depth_view
                valid_fraction[window_index, time_index, view_index] = float(
                    np.mean(valid_view.astype(bool))
                )
    return images, valid_fraction, repaired_total


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "source_id", "user_id", "class_id")
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    pixel_cache = args.pixel_cache.resolve()
    rows = read_csv(pixel_cache / "rows.csv")
    manifest_lookup = {row["sample_id"]: row for row in read_csv(args.manifest)}
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in manifest_lookup]
    if missing:
        raise RuntimeError(f"manifest misses P88 rows: {missing[:3]}")
    manifest_rows = [manifest_lookup[row["sample_id"]] for row in rows]
    if args.max_trials > 0:
        manifest_rows = manifest_rows[: args.max_trials]
        rows = rows[: args.max_trials]
    source_indices = np.load(
        pixel_cache / "source_frame_indices.npy", mmap_mode="r", allow_pickle=False
    )[: len(rows)]
    with (pixel_cache / "summary.json").open("r", encoding="utf-8") as handle:
        pixel_summary = json.load(handle)
    resolution = int(pixel_summary["resolution"])
    frames = int(pixel_summary["frames_per_window"])
    checkpoint = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model(checkpoint, device)
    if int(model_config["frames"]) != frames:
        raise RuntimeError("P87 checkpoint and pixel-cache frame counts differ")
    sequence_path = output / "backbone_sequence_fp16.npy"
    valid_path = output / "depth_valid_fraction_fp16.npy"
    completed_path = output / "completed.npy"
    expected_sequence = (len(rows), 2, 3, frames, 512)
    expected_valid = (len(rows), 2, frames, 3)
    if sequence_path.exists() or valid_path.exists() or completed_path.exists():
        if not all(path.exists() for path in (sequence_path, valid_path, completed_path)):
            raise RuntimeError("partial P88 Depth cache; restore all cache files")
        sequence = np.lib.format.open_memmap(sequence_path, mode="r+")
        valid_fraction = np.lib.format.open_memmap(valid_path, mode="r+")
        completed = np.lib.format.open_memmap(completed_path, mode="r+")
        if sequence.shape != expected_sequence or valid_fraction.shape != expected_valid:
            raise RuntimeError("existing P88 Depth cache geometry differs")
    else:
        sequence = np.lib.format.open_memmap(
            sequence_path, mode="w+", dtype=np.float16, shape=expected_sequence
        )
        valid_fraction = np.lib.format.open_memmap(
            valid_path, mode="w+", dtype=np.float16, shape=expected_valid
        )
        completed = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=np.uint8, shape=(len(rows),)
        )
        sequence[:] = 0
        valid_fraction[:] = 0
        completed[:] = 0
        sequence.flush()
        valid_fraction.flush()
        completed.flush()
        write_rows(output / "rows.csv", rows)
    pending = np.flatnonzero(~np.asarray(completed, dtype=bool))
    started = time.perf_counter()
    repaired_total = 0
    for offset in range(0, len(pending), args.batch_size):
        batch_indices = pending[offset : offset + args.batch_size]
        batch_images: list[np.ndarray] = []
        batch_valid: list[np.ndarray] = []
        for index in batch_indices:
            images, valid, repaired = prepare_trial(
                manifest_rows[int(index)],
                args.p29_run.resolve(),
                np.asarray(source_indices[int(index)], dtype=np.int64),
                resolution,
            )
            batch_images.append(images)
            batch_valid.append(valid)
            repaired_total += repaired
        image_tensor = torch.from_numpy(np.stack(batch_images)).to(
            device, non_blocking=True
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            values = model.encode_backbone_sequence(image_tensor)
        sequence[batch_indices] = values.to(dtype=torch.float16).cpu().numpy()
        valid_fraction[batch_indices] = np.stack(batch_valid).astype(np.float16)
        completed[batch_indices] = 1
        if offset % max(args.batch_size * 32, 1) == 0 or offset + len(batch_indices) == len(pending):
            sequence.flush()
            valid_fraction.flush()
            completed.flush()
            print(
                json.dumps(
                    {
                        "completed": int(np.asarray(completed).sum()),
                        "total": len(rows),
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    summary = {
        "stage": "P88_registered_depth_through_frozen_P87_MC3",
        "status": "complete" if int(np.asarray(completed).sum()) == len(rows) else "partial",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "p87_checkpoint_modified": False,
        "rows": len(rows),
        "shape": list(sequence.shape),
        "dtype": "float16",
        "decoded_depth": "OpenCV JET index with invalid pixels kept at zero",
        "views": ["scene", "person", "workspace"],
        "repaired_off_palette_pixels_current_run": repaired_total,
        "mean_valid_fraction": float(np.asarray(valid_fraction, dtype=np.float32).mean()),
        "elapsed_seconds_current_run": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
