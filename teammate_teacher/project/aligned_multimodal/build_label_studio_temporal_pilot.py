from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from aligned_data import AlignedMultimodalDataset, frame_map
from audit_motion_crop import analyse_trial


PROJECT_DIR = Path(__file__).resolve().parent
AUDIT_DIR = PROJECT_DIR / "data" / "local_action_audit_v1"
DEFAULT_SELECTION = AUDIT_DIR / "label_studio_pilot30" / "selection.csv"
DEFAULT_OUTPUT = AUDIT_DIR / "label_studio_temporal_pilot30_v2"
DEFAULT_CACHE = PROJECT_DIR / "cache" / "aligned_192x144"
NUM_FRAMES = 12
MODEL_WIDTH = 192
MODEL_HEIGHT = 144
AUDIT_WIDTH = 320
AUDIT_HEIGHT = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a temporal Label Studio pilot using the exact 12 inference frames"
    )
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifest.csv")
    parser.add_argument("--annotations", type=Path, default=AUDIT_DIR / "annotations.csv")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def inference_positions(length: int, count: int = NUM_FRAMES) -> list[int]:
    """Mirror AlignedMultimodalDataset._sample_positions for augment=False."""
    if length <= 0:
        raise ValueError("Frame sequence must not be empty")
    if length <= count:
        return (
            np.rint(np.linspace(0, length - 1, count, dtype=np.float64))
            .astype(np.int64)
            .tolist()
        )
    boundaries = np.floor(np.linspace(0, length, count + 1, dtype=np.float64)).astype(np.int64)
    positions: list[int] = []
    for index in range(count):
        start = int(boundaries[index])
        end = max(start + 1, int(boundaries[index + 1]))
        positions.append(min(length - 1, (start + end - 1) // 2))
    return positions


def assert_sampling_matches_training_code(lengths: set[int]) -> None:
    dataset = object.__new__(AlignedMultimodalDataset)
    dataset.num_frames = NUM_FRAMES
    dataset.augment = False
    for length in sorted(lengths):
        expected = dataset._sample_positions(length)
        actual = inference_positions(length)
        if actual != expected:
            raise AssertionError(
                f"Sampling mismatch for length={length}: generated={actual}, dataset={expected}"
            )


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def model_bbox(box_320: list[int]) -> list[int]:
    x0, y0, x1, y1 = box_320
    left = math.floor(x0 * MODEL_WIDTH / AUDIT_WIDTH)
    top = math.floor(y0 * MODEL_HEIGHT / AUDIT_HEIGHT)
    right_exclusive = math.ceil((x1 + 1) * MODEL_WIDTH / AUDIT_WIDTH)
    bottom_exclusive = math.ceil((y1 + 1) * MODEL_HEIGHT / AUDIT_HEIGHT)
    left = min(max(0, left), MODEL_WIDTH - 1)
    top = min(max(0, top), MODEL_HEIGHT - 1)
    right_exclusive = min(max(left + 1, right_exclusive), MODEL_WIDTH)
    bottom_exclusive = min(max(top + 1, bottom_exclusive), MODEL_HEIGHT)
    return [left, top, right_exclusive - 1, bottom_exclusive - 1]


def crop_and_resize(frame: np.ndarray, box: list[int]) -> np.ndarray:
    left, top, right, bottom = box
    crop = frame[top : bottom + 1, left : right + 1]
    return np.asarray(
        Image.fromarray(crop).resize(
            (MODEL_WIDTH, MODEL_HEIGHT),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )


def contact_sheet(
    frames: list[np.ndarray],
    positions: list[int],
    title: str,
    border: tuple[int, int, int],
) -> Image.Image:
    if len(frames) != NUM_FRAMES or len(positions) != NUM_FRAMES:
        raise ValueError("A temporal contact sheet must contain exactly 12 frames")
    columns = 4
    rows = 3
    label_height = 20
    title_height = 30
    gap = 5
    tile_width = MODEL_WIDTH
    tile_height = label_height + MODEL_HEIGHT
    canvas_width = columns * tile_width + (columns + 1) * gap
    canvas_height = title_height + rows * tile_height + (rows + 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), title, fill=(245, 245, 245))
    for frame_index, (frame, position) in enumerate(zip(frames, positions)):
        row, column = divmod(frame_index, columns)
        left = gap + column * (tile_width + gap)
        top = title_height + gap + row * (tile_height + gap)
        draw.rectangle(
            (left, top, left + tile_width - 1, top + tile_height - 1),
            outline=border,
            width=2,
        )
        draw.text(
            (left + 5, top + 4),
            f"t{frame_index + 1:02d}  common-index={position}",
            fill=(230, 230, 230),
        )
        image = Image.fromarray(frame)
        if image.mode != "RGB":
            image = image.convert("RGB")
        canvas.paste(image, (left, top + label_height))
    return canvas


def save_jpeg(image: Image.Image, path: Path, quality: int = 86) -> None:
    image.convert("RGB").save(path, quality=quality, optimize=True)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    annotations_path = args.annotations.resolve()
    selection_path = args.selection.resolve()
    cache_dir = args.cache_dir.resolve()
    output = args.output_dir.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_by_id = {row["sample_id"]: row for row in read_csv(manifest_path)}
    annotation_by_id = {row["sample_id"]: row for row in read_csv(annotations_path)}
    selected = sorted(read_csv(selection_path), key=lambda row: int(row["pilot_index"]))
    if len(selected) != 30 or len({row["sample_id"] for row in selected}) != 30:
        raise RuntimeError("Temporal pilot must reuse the 30 unique v1 pilot samples")

    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if (int(metadata["image_width"]), int(metadata["image_height"])) != (
        MODEL_WIDTH,
        MODEL_HEIGHT,
    ):
        raise ValueError("Cache resolution does not match the audited model input")
    cache_locations = {
        sample_id: (int(offset), int(length))
        for sample_id, (offset, length) in zip(metadata["sample_ids"], metadata["offsets"])
    }
    selected_lengths = {cache_locations[row["sample_id"]][1] for row in selected}
    assert_sampling_matches_training_code(selected_lengths)

    depth_cache = np.load(cache_dir / "depth_uint8.npy", mmap_mode="r", allow_pickle=False)
    ir_cache = np.load(cache_dir / "ir_uint8.npy", mmap_mode="r", allow_pickle=False)
    tasks: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    total_image_bytes = 0
    task_image_bytes: list[int] = []

    for pilot_index, selected_row in enumerate(selected, start=1):
        sample_id = selected_row["sample_id"]
        row = manifest_by_id[sample_id]
        annotation = annotation_by_id[sample_id]
        offset, length = cache_locations[sample_id]
        positions = inference_positions(length)
        absolute_positions = np.asarray(positions, dtype=np.int64) + offset

        maps = {
            "depth": frame_map(Path(row["depth_dir"]), "depth"),
            "ir": frame_map(Path(row["ir_dir"]), "ir"),
            "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
        }
        common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
        if len(common_ids) != length:
            raise RuntimeError(
                f"{sample_id}: source common frames={len(common_ids)} but cache length={length}"
            )
        frame_ids = [common_ids[position] for position in positions]

        depth_frames = [
            np.asarray(depth_cache[index], dtype=np.uint8) for index in absolute_positions
        ]
        ir_frames = [
            np.repeat(
                np.asarray(ir_cache[index], dtype=np.uint8)[:, :, None],
                3,
                axis=2,
            )
            for index in absolute_positions
        ]

        crop_record, _ = analyse_trial(row, AUDIT_WIDTH, AUDIT_HEIGHT)
        bbox_320 = [int(value) for value in crop_record["bbox"]]
        bbox_model = model_bbox(bbox_320)
        local_frames = [crop_and_resize(frame, bbox_model) for frame in depth_frames]

        old_case = AUDIT_DIR / annotation["case_image"]
        with Image.open(old_case) as old_montage:
            keyframes = old_montage.convert("RGB").crop((0, 0, 960, 240))

        stem = f"{pilot_index:02d}_{sample_id}"
        paths = {
            "keyframes": images_dir / f"{stem}_keyframes.jpg",
            "depth12": images_dir / f"{stem}_depth12.jpg",
            "local12": images_dir / f"{stem}_local12.jpg",
            "ir12": images_dir / f"{stem}_ir12.jpg",
        }
        save_jpeg(keyframes, paths["keyframes"], quality=88)
        save_jpeg(
            contact_sheet(
                depth_frames,
                positions,
                "DEPTH 12 | exact cached frames used by validation/test inference",
                (0, 190, 255),
            ),
            paths["depth12"],
        )
        save_jpeg(
            contact_sheet(
                local_frames,
                positions,
                f"LOCAL CROP 12 | one trial bbox at model resolution: {bbox_model}",
                (255, 0, 190),
            ),
            paths["local12"],
        )
        save_jpeg(
            contact_sheet(
                ir_frames,
                positions,
                "IR 12 | same cached temporal positions as Depth",
                (210, 210, 210),
            ),
            paths["ir12"],
        )
        sample_image_bytes = sum(path.stat().st_size for path in paths.values())
        total_image_bytes += sample_image_bytes
        task_image_bytes.append(sample_image_bytes)

        tasks.append(
            {
                "data": {
                    "sample_id": sample_id,
                    "sample_info": (
                        f"Temporal Pilot {pilot_index}/30｜真实类别：{row['class_name']}｜"
                        f"Subject：{row['user_id']}｜Trial：{row['trial_id']}｜"
                        f"motion fallback：{bool(crop_record['fallback'])}"
                    ),
                    "keyframes_image": image_data_url(paths["keyframes"]),
                    "depth12_image": image_data_url(paths["depth12"]),
                    "local12_image": image_data_url(paths["local12"]),
                    "ir12_image": image_data_url(paths["ir12"]),
                }
            }
        )
        index_rows.append(
            {
                "pilot_index": pilot_index,
                "sample_id": sample_id,
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "user_id": row["user_id"],
                "trial_id": row["trial_id"],
                "common_frame_count": length,
                "inference_positions": json.dumps(positions),
                "inference_frame_ids": json.dumps(frame_ids, ensure_ascii=False),
                "motion_fallback": int(bool(crop_record["fallback"])),
                "bbox_320x240": json.dumps(bbox_320),
                "bbox_192x144": json.dumps(bbox_model),
                "keyframes_image": str(paths["keyframes"].relative_to(output)),
                "depth12_image": str(paths["depth12"].relative_to(output)),
                "local12_image": str(paths["local12"].relative_to(output)),
                "ir12_image": str(paths["ir12"].relative_to(output)),
            }
        )
        print(f"temporal pilot {pilot_index}/30: {sample_id}", flush=True)

    tasks_path = output / "tasks.json"
    tasks_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    with (output / "selection.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    summary = {
        "version": 2,
        "tasks": len(tasks),
        "old_project_preserved": True,
        "reuses_v1_sample_ids": True,
        "model_input_resolution": [MODEL_WIDTH, MODEL_HEIGHT],
        "inference_frames": NUM_FRAMES,
        "sampling_contract": (
            "Exact augment=False AlignedMultimodalDataset sampling: common "
            "Depth/IR/Skeleton frame sequence split into 12 bins, using each bin midpoint."
        ),
        "training_sampling_note": (
            "Training with augment=True samples randomly inside each temporal bin; "
            "there is no single fixed set of 12 training frames."
        ),
        "depth_contract": "Raw uint8 frames are read from the exact aligned_192x144 model cache.",
        "ir_contract": "IR uses the same cache offsets and temporal positions as Depth.",
        "local_contract": (
            "Candidate local view only; the current classifier does not consume it. "
            "All 12 crops use one motion-derived trial-level bbox."
        ),
        "classification_predictions_in_tasks": False,
        "embedded_image_bytes": total_image_bytes,
        "tasks_json_bytes": tasks_path.stat().st_size,
        "image_dimensions": {
            "keyframes": [960, 240],
            "depth12": [793, 542],
            "local12": [793, 542],
            "ir12": [793, 542],
        },
        "unscaled_image_vertical_pixels_per_task": 1866,
        "image_bytes_per_task": {
            "minimum": min(task_image_bytes),
            "median": int(np.median(task_image_bytes)),
            "maximum": max(task_image_bytes),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
