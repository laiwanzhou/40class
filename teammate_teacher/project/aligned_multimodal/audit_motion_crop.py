from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from aligned_data import frame_map
from depth_encoding import decode_jet_rgb, resize_decoded_depth


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "motion_crop_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 320x240 有序 Depth 上审计 motion crop")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def expanded_box(
    box: tuple[int, int, int, int], width: int, height: int, expansion: float = 0.25
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    box_width = max(1.0, (x1 - x0 + 1) * (1.0 + expansion))
    box_height = max(1.0, (y1 - y0 + 1) * (1.0 + expansion))
    target_ratio = 4.0 / 3.0
    if box_width / box_height < target_ratio:
        box_width = box_height * target_ratio
    else:
        box_height = box_width / target_ratio
    box_width = max(box_width, width * 0.35)
    box_height = max(box_height, height * 0.35)
    box_width, box_height = min(box_width, width), min(box_height, height)
    left = int(round(center_x - box_width / 2))
    top = int(round(center_y - box_height / 2))
    left = min(max(0, left), width - int(round(box_width)))
    top = min(max(0, top), height - int(round(box_height)))
    return left, top, left + int(round(box_width)) - 1, top + int(round(box_height)) - 1


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative[-1] <= 0:
        return float(np.quantile(values, quantile))
    target = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)
    return float(sorted_values[index])


def motion_mask(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(depth) < 2:
        return np.zeros(depth.shape[1:], dtype=np.float32), np.zeros(depth.shape[1:], dtype=np.uint8)
    both_valid = valid[1:] & valid[:-1]
    differences = np.abs(depth[1:].astype(np.int16) - depth[:-1].astype(np.int16))
    differences = np.where(both_valid, differences, 0)
    heat = np.percentile(differences, 90, axis=0).astype(np.float32)
    stable = both_valid.sum(axis=0) >= max(1, int(np.ceil(0.7 * (len(depth) - 1))))
    active = (stable & (heat >= 2.0)).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    active = cv2.morphologyEx(active, cv2.MORPH_OPEN, kernel)
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    filtered = np.zeros_like(active)
    minimum_component = max(20, int(round(active.size * 0.0005)))
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_component:
            filtered[labels == component] = 1
    return heat, filtered


def analyse_trial(row: dict[str, str], width: int, height: int) -> tuple[dict[str, object], np.ndarray]:
    depth_map = frame_map(Path(row["depth_dir"]), "depth")
    frame_ids = sorted(depth_map)
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rgb_mid: np.ndarray | None = None
    for frame_index, frame_id in enumerate(frame_ids):
        with Image.open(depth_map[frame_id]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if frame_index == len(frame_ids) // 2:
            rgb_mid = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        decoded, valid, _ = decode_jet_rgb(rgb)
        decoded, valid = resize_decoded_depth(decoded, valid, height, width)
        frames.append(decoded)
        masks.append(valid.astype(bool))
    depth = np.stack(frames)
    valid = np.stack(masks)
    heat, active = motion_mask(depth, valid)
    both = valid[1:] & valid[:-1]
    pair_diff = np.abs(depth[1:].astype(np.int16) - depth[:-1].astype(np.int16))
    pair_values = pair_diff[both]
    invalid_flip = float(np.mean(valid[1:] != valid[:-1])) if len(valid) > 1 else 0.0
    ys, xs = np.where(active > 0)
    fallback = len(xs) < max(20, int(round(active.size * 0.001)))
    if fallback:
        box = (0, 0, width - 1, height - 1)
        raw_box_fraction = 1.0
    else:
        # Tiny isolated responses and a long movement trail should not stretch the
        # crop across the full room. Retain the central 90% of motion energy and
        # let the minimum crop size provide the surrounding body context.
        weights = np.maximum(heat[ys, xs], 1e-6).astype(np.float64)
        raw_box = (
            int(np.floor(weighted_quantile(xs, weights, 0.05))),
            int(np.floor(weighted_quantile(ys, weights, 0.05))),
            int(np.ceil(weighted_quantile(xs, weights, 0.95))),
            int(np.ceil(weighted_quantile(ys, weights, 0.95))),
        )
        raw_box_fraction = (
            (raw_box[2] - raw_box[0] + 1) * (raw_box[3] - raw_box[1] + 1) / active.size
        )
        box = expanded_box(raw_box, width, height)
    box_fraction = (box[2] - box[0] + 1) * (box[3] - box[1] + 1) / active.size

    assert rgb_mid is not None
    heat_uint8 = np.clip(heat / max(float(np.percentile(heat, 99)), 1e-6) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    overlay = (0.65 * rgb_mid + 0.35 * colored).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=(0, 255, 0), width=3)
    draw.text((5, 5), f"c{int(row['class_id']):02d} {row['trial_id']}", fill=(255, 255, 255))
    record = {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "class_id": int(row["class_id"]),
        "class_name": row["class_name"],
        "user_id": row["user_id"],
        "trial_id": row["trial_id"],
        "frames": len(depth),
        "pair_absdiff_median": float(np.median(pair_values)) if pair_values.size else 0.0,
        "pair_absdiff_p90": float(np.percentile(pair_values, 90)) if pair_values.size else 0.0,
        "invalid_flip_fraction": invalid_flip,
        "active_pixel_fraction": float(active.mean()),
        "raw_bbox_fraction": raw_box_fraction,
        "expanded_bbox_fraction": box_fraction,
        "fallback": bool(fallback),
        "bbox": list(box),
    }
    return record, np.asarray(image)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)
    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for class_id in range(40):
        candidates = sorted(by_class[class_id], key=lambda row: row["sample_id"])
        rng.shuffle(candidates)
        selected.extend(candidates[: args.samples_per_class])

    records: list[dict[str, object]] = []
    montage_examples: dict[int, np.ndarray] = {}
    for index, row in enumerate(selected, 1):
        record, image = analyse_trial(row, args.width, args.height)
        records.append(record)
        montage_examples.setdefault(int(row["class_id"]), image)
        if index % 40 == 0 or index == len(selected):
            print(f"motion audit {index}/{len(selected)}", flush=True)

    metrics = [
        "pair_absdiff_median",
        "pair_absdiff_p90",
        "invalid_flip_fraction",
        "active_pixel_fraction",
        "raw_bbox_fraction",
        "expanded_bbox_fraction",
    ]
    summary = {
        "sampled_trials": len(records),
        "samples_per_class": args.samples_per_class,
        "analysis_resolution": [args.width, args.height],
        "uses_all_original_frames": True,
        "fallback_trials": sum(bool(record["fallback"]) for record in records),
        "fallback_fraction": float(np.mean([record["fallback"] for record in records])),
        "statistics": {
            metric: describe([float(record[metric]) for record in records]) for metric in metrics
        },
        "per_class": {},
        "method": {
            "depth": "exact OpenCV JET inversion at source resolution, then mask-aware resize",
            "motion": "90th percentile adjacent-frame absolute difference on pixels valid in >=70% frame pairs",
            "threshold": "difference >= 2 JET indices, morphology, components >=0.05% image",
            "crop": "central 90% motion-energy box, 25% expansion, 4:3, minimum 35% width/height, full-frame fallback",
        },
    }
    for class_id in range(40):
        subset = [record for record in records if int(record["class_id"]) == class_id]
        summary["per_class"][str(class_id)] = {
            "class_name": subset[0]["class_name"],
            "fallback_fraction": float(np.mean([record["fallback"] for record in subset])),
            "active_pixel_fraction_median": float(
                np.median([record["active_pixel_fraction"] for record in subset])
            ),
            "expanded_bbox_fraction_median": float(
                np.median([record["expanded_bbox_fraction"] for record in subset])
            ),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    tile_height, tile_width = args.height, args.width
    montage = np.zeros((5 * tile_height, 8 * tile_width, 3), dtype=np.uint8)
    for class_id, image in montage_examples.items():
        row, column = divmod(class_id, 8)
        montage[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = image
    Image.fromarray(montage).save(output.with_name("motion_crop_montage.jpg"), quality=92)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
