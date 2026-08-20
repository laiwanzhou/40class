from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
DEFAULT_YOLO_WEIGHTS = Path(r"D:\work\2026.7.14_kaggle\40class\yolo11n-pose.pt")
MODALITIES = ("Depth_Color", "IMU", "IR", "Radar", "Skeleton", "Thermal")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SEALED_HELDOUT_USERS = {"user4", "user17", "user23", "user24"}
CLASS_DIRECTORY_PATTERN = re.compile(r"^(?P<class_id>\d+)_(?P<action_name>.+)$")
FRAME_NUMBER_PATTERN = re.compile(r"(\d+)(?!.*\d)")
IR_TIMESTAMP_PATTERN = re.compile(
    r"(?:IR|Depth)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})_"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Thermal Stage T0 audit")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--development-split",
        type=Path,
        default=PROJECT_ROOT / "metadata/splits/train12_val2_user6_user7_development.json",
    )
    parser.add_argument(
        "--oof-split",
        type=Path,
        default=PROJECT_ROOT / "metadata/splits/train14_oof_3fold.json",
    )
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "reports/thermal_stage0_data_alignment_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "reports/thermal_stage0_data_alignment_audit.md",
    )
    parser.add_argument(
        "--montage-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/thermal_stage0_montages",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--representative-trials", type=int, default=56)
    parser.add_argument("--localization-frames", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-localization", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_train14_users(users: Iterable[str]) -> list[str]:
    normalized = sorted({str(user) for user in users})
    forbidden = SEALED_HELDOUT_USERS & set(normalized)
    if forbidden:
        raise ValueError(f"sealed heldout users are forbidden: {sorted(forbidden)}")
    if not normalized:
        raise ValueError("train-14 user set is empty")
    return normalized


def sample_thermal_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count < 1 or sample_count < 1:
        raise ValueError("frame_count and sample_count must be positive")
    positions = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    return np.rint(positions * (frame_count - 1)).astype(int).tolist()


def normalized_time_candidate_pairs(
    reference_count: int, candidate_count: int
) -> list[tuple[int, int]]:
    if reference_count < 1 or candidate_count < 1:
        raise ValueError("reference_count and candidate_count must be positive")
    candidate_times = (
        np.zeros(1, dtype=np.float64)
        if candidate_count == 1
        else np.linspace(0.0, 1.0, candidate_count, dtype=np.float64)
    )
    reference_indices = np.rint(candidate_times * (reference_count - 1)).astype(int)
    return [(int(reference), int(candidate)) for candidate, reference in enumerate(reference_indices)]


def classify_thermal_rendering(frames_rgb: Sequence[np.ndarray]) -> dict[str, Any]:
    if not frames_rgb:
        return {
            "rendering": "unknown_no_decodable_sample",
            "grayscale_pixel_fraction": None,
            "color_pixel_fraction": None,
            "palette_curve_stability": None,
            "auto_scale_identifiable": False,
            "auto_scale_assessment": "not_identifiable_from_rendered_rgb_jpeg",
        }
    gray_fractions: list[float] = []
    color_fractions: list[float] = []
    palette_curves: list[np.ndarray] = []
    for frame in frames_rgb:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame, got {frame.shape}")
        values = frame.astype(np.int16)
        spread = values.max(axis=2) - values.min(axis=2)
        gray_fractions.append(float(np.mean(spread <= 2)))
        color_fractions.append(float(np.mean(spread >= 8)))
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        luminance = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        curve = np.full(8, np.nan, dtype=np.float64)
        for bin_index in range(8):
            lower = bin_index * 32
            upper = 256 if bin_index == 7 else (bin_index + 1) * 32
            mask = (luminance >= lower) & (luminance < upper) & (hsv[:, :, 1] >= 20)
            if int(mask.sum()) >= 32:
                curve[bin_index] = float(np.median(hsv[:, :, 0][mask]))
        palette_curves.append(curve)
    grayscale_fraction = float(np.mean(gray_fractions))
    color_fraction = float(np.mean(color_fractions))
    stacked = np.stack(palette_curves)
    valid_stds = np.asarray(
        [
            float(np.std(stacked[np.isfinite(stacked[:, index]), index]))
            for index in range(stacked.shape[1])
            if np.isfinite(stacked[:, index]).any()
        ],
        dtype=np.float64,
    )
    stability = None if not len(valid_stds) else float(max(0.0, 1.0 - np.median(valid_stds) / 90.0))
    if grayscale_fraction >= 0.98:
        rendering = "grayscale_copy"
    elif color_fraction >= 0.10:
        rendering = "stable_pseudocolor"
    else:
        rendering = "weak_or_mixed_color"
    return {
        "rendering": rendering,
        "grayscale_pixel_fraction": grayscale_fraction,
        "color_pixel_fraction": color_fraction,
        "palette_curve_stability": stability,
        "auto_scale_identifiable": False,
        "auto_scale_assessment": "not_identifiable_from_rendered_rgb_jpeg",
    }


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
        if part
    )


def _image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_key,
    )


def _last_number(path: Path) -> int | None:
    match = FRAME_NUMBER_PATTERN.search(path.stem)
    return int(match.group(1)) if match else None


def audit_trial_images(trial_path: Path) -> dict[str, Any]:
    present = trial_path.is_dir()
    files = _image_files(trial_path)
    selected = set(sample_thermal_indices(len(files), min(7, len(files)))) if files else set()
    decoded_count = 0
    corrupt_jpegs = 0
    content_hashes: list[str] = []
    sampled_rgb: list[np.ndarray] = []
    resolutions: set[tuple[int, int]] = set()
    consecutive_duplicates = 0
    previous_hash: str | None = None
    for index, path in enumerate(files):
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if len(encoded) else None
        if image is None:
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                corrupt_jpegs += 1
            continue
        decoded_count += 1
        height, width = image.shape[:2]
        resolutions.add((width, height))
        content_hash = hashlib.sha256(image.tobytes()).hexdigest()
        content_hashes.append(content_hash)
        if previous_hash == content_hash:
            consecutive_duplicates += 1
        previous_hash = content_hash
        if index in selected:
            sampled_rgb.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    unique_count = len(set(content_hashes))
    duplicate_count = max(0, decoded_count - unique_count)
    rendering = classify_thermal_rendering(sampled_rgb)
    first_number = _last_number(files[0]) if files else None
    last_number = _last_number(files[-1]) if files else None
    return {
        "directory_present": present,
        "file_count": len(files),
        "decodable": decoded_count > 0,
        "decodable_frame_count": decoded_count,
        "decode_failure_count": len(files) - decoded_count,
        "corrupt_jpeg_count": corrupt_jpegs,
        "usable": decoded_count > 0,
        "unique_decoded_frame_count": unique_count,
        "duplicate_frame_count": duplicate_count,
        "consecutive_duplicate_frame_count": consecutive_duplicates,
        "distinct_frame_ratio": (float(unique_count / decoded_count) if decoded_count else None),
        "single_frame": decoded_count == 1,
        "extremely_short_le4": 0 < decoded_count <= 4,
        "shorter_than_13": 0 < decoded_count < 13,
        "resolutions": [f"{width}x{height}" for width, height in sorted(resolutions)],
        "first_frame_number": first_number,
        "last_frame_number": last_number,
        **rendering,
    }


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(np.std(values))
    if std < 1e-10:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = _standardize(left)
    right = _standardize(right)
    rows, columns = len(left) + 1, len(right) + 1
    costs = np.full((rows, columns), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0
    for row in range(1, rows):
        for column in range(1, columns):
            local = abs(left[row - 1] - right[column - 1])
            costs[row, column] = local + min(
                costs[row - 1, column],
                costs[row, column - 1],
                costs[row - 1, column - 1],
            )
    return float(costs[-1, -1] / max(len(left), len(right)))


def estimate_motion_alignment(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 1 or candidate.ndim != 1 or min(len(reference), len(candidate)) < 4:
        return {"offset": None, "scale": None, "correlation": None, "dtw_distance": None}
    reference_grid = np.linspace(0.0, 1.0, len(reference), dtype=np.float64)
    candidate_grid = np.linspace(0.0, 1.0, len(candidate), dtype=np.float64)
    best: tuple[float, float, float, np.ndarray] | None = None
    for scale in np.arange(0.80, 1.201, 0.02):
        for offset in np.arange(-0.25, 0.251, 0.01):
            source_times = (reference_grid - offset) / scale
            valid = (source_times >= 0.0) & (source_times <= 1.0)
            if int(valid.sum()) < max(12, int(0.6 * len(reference_grid))):
                continue
            aligned = np.interp(source_times[valid], candidate_grid, candidate)
            ref = reference[valid]
            if float(np.std(ref)) < 1e-10 or float(np.std(aligned)) < 1e-10:
                correlation = -1.0
            else:
                correlation = float(np.corrcoef(ref, aligned)[0, 1])
            score = correlation - 0.05 * abs(math.log(scale)) - 0.02 * abs(offset)
            if best is None or score > best[0]:
                best = (score, float(offset), float(scale), aligned)
    if best is None:
        return {"offset": None, "scale": None, "correlation": None, "dtw_distance": None}
    _, offset, scale, _ = best
    source_times = (reference_grid - offset) / scale
    valid = (source_times >= 0.0) & (source_times <= 1.0)
    aligned = np.interp(source_times[valid], candidate_grid, candidate)
    ref = reference[valid]
    correlation = (
        None
        if float(np.std(ref)) < 1e-10 or float(np.std(aligned)) < 1e-10
        else float(np.corrcoef(ref, aligned)[0, 1])
    )
    return {
        "offset": offset,
        "scale": scale,
        "correlation": correlation,
        "dtw_distance": _dtw_distance(ref, aligned),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _oof_train_users(oof: dict[str, Any]) -> list[str]:
    users: set[str] = set()
    for fold in oof["folds"]:
        users.update(str(user) for user in fold["train_user_ids"])
        users.update(str(user) for user in fold["validation_user_ids"])
    return validate_train14_users(users)


def inventory_train14_trials(data_root: Path, train14_users: Sequence[str]) -> list[dict[str, Any]]:
    allowed = set(validate_train14_users(train14_users))
    rows: dict[tuple[int, str, str], dict[str, Any]] = {}
    for modality in MODALITIES:
        modality_root = data_root / modality
        if not modality_root.is_dir():
            raise FileNotFoundError(f"Missing modality root: {modality_root}")
        for class_dir in sorted((path for path in modality_root.iterdir() if path.is_dir()), key=_natural_key):
            match = CLASS_DIRECTORY_PATTERN.match(class_dir.name)
            if not match:
                continue
            class_id = int(match.group("class_id"))
            action_name = match.group("action_name")
            for user_id in sorted(allowed):
                user_dir = class_dir / user_id
                if not user_dir.is_dir():
                    continue
                for trial_dir in sorted((path for path in user_dir.iterdir() if path.is_dir()), key=_natural_key):
                    key = (class_id, user_id, trial_dir.name)
                    row = rows.setdefault(
                        key,
                        {
                            "sample_id": f"train__c{class_id:02d}__{user_id}__{trial_dir.name}",
                            "class_id": class_id,
                            "action_name": action_name,
                            "user_id": user_id,
                            "trial_id": trial_dir.name,
                            "paths": {},
                        },
                    )
                    if row["action_name"] != action_name:
                        raise ValueError(f"Action-name conflict for {key}")
                    row["paths"][modality] = str(trial_dir)
    output = sorted(rows.values(), key=lambda row: (row["class_id"], row["user_id"], row["trial_id"]))
    if set(row["user_id"] for row in output) != allowed:
        raise ValueError("Filesystem inventory does not cover the exact train-14 user set")
    return output


def _duration_bucket(frame_count: int) -> str:
    if frame_count <= 0:
        return "missing_or_unusable"
    if frame_count == 1:
        return "single_frame"
    if frame_count <= 8:
        return "2_to_8"
    if frame_count <= 32:
        return "9_to_32"
    if frame_count <= 96:
        return "33_to_96"
    return "over_96"


def _quantiles(values: Sequence[float | int]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {key: None for key in ("min", "p25", "median", "p75", "p95", "max", "mean")}
    return {
        "min": float(np.min(finite)),
        "p25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "p75": float(np.quantile(finite, 0.75)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def serializable_trial_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"paths", "thermal_path"}
    }


def summarize_localization_groups(
    records: Sequence[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record[field]), []).append(record)
    rows: list[dict[str, Any]] = []
    for name, group in sorted(grouped.items()):
        detected = [record for record in group if record["detected"]]
        continuity = [
            record["previous_bbox_iou"]
            for record in group
            if record.get("previous_bbox_iou") is not None
        ]
        rows.append(
            {
                field: name,
                "frames": len(group),
                "detection_rate": float(np.mean([record["detected"] for record in group])),
                "detected_confidence": _quantiles(
                    [record["confidence"] for record in detected]
                ),
                "bbox_continuity_iou": _quantiles(continuity),
            }
        )
    return rows


def _group_summary(rows: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[field]), []).append(row)
    output: list[dict[str, Any]] = []
    for name, group in sorted(grouped.items()):
        output.append(
            {
                field: name,
                "canonical_trials": len(group),
                "directory_present": sum(bool(row["directory_present"]) for row in group),
                "decodable_trials": sum(bool(row["decodable"]) for row in group),
                "usable_trials": sum(bool(row["usable"]) for row in group),
                "missing_directories": sum(not bool(row["directory_present"]) for row in group),
                "frame_count": _quantiles([row["decodable_frame_count"] for row in group if row["decodable"]]),
            }
        )
    return output


def _frame_boundary(path: Path, modality: str) -> dict[str, Any]:
    files = _image_files(path)
    first = files[0] if files else None
    last = files[-1] if files else None
    first_timestamp = None
    last_timestamp = None
    if modality in {"IR", "Depth_Color"}:
        for target, name in (("first", first), ("last", last)):
            if name is None:
                continue
            match = IR_TIMESTAMP_PATTERN.search(name.stem)
            if match and target == "first":
                first_timestamp = match.group("timestamp")
            elif match:
                last_timestamp = match.group("timestamp")
    return {
        "frame_count": len(files),
        "first_name": first.name if first else None,
        "last_name": last.name if last else None,
        "first_frame_number": _last_number(first) if first else None,
        "last_frame_number": _last_number(last) if last else None,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def _motion_energy(path: Path, size: tuple[int, int] = (64, 48)) -> np.ndarray:
    files = _image_files(path)
    energies: list[float] = []
    previous: np.ndarray | None = None
    for frame_path in files:
        image = cv2.imdecode(np.fromfile(frame_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if previous is not None:
            energies.append(float(np.mean(np.abs(image - previous))))
        previous = image
    return np.asarray(energies, dtype=np.float64)


def _representative_rows(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["usable"] and "IR" in row["paths"] and row["decodable_frame_count"] > 0
    ]
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    seen_users: set[str] = set()
    seen_buckets: set[str] = set()
    for class_id in range(40):
        candidates = [row for row in eligible if int(row["class_id"]) == class_id]
        candidates.sort(
            key=lambda row: (
                row["user_id"] in seen_users,
                row["duration_bucket"] in seen_buckets,
                hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest(),
            )
        )
        if not candidates:
            continue
        row = candidates[0]
        chosen.append(row)
        chosen_ids.add(row["sample_id"])
        seen_users.add(row["user_id"])
        seen_buckets.add(row["duration_bucket"])
    coverage_targets = [
        ("user_id", value) for value in sorted({row["user_id"] for row in eligible})
    ] + [
        ("duration_bucket", value)
        for value in sorted({row["duration_bucket"] for row in eligible})
    ]
    for field, value in coverage_targets:
        if any(row[field] == value for row in chosen):
            continue
        candidates = [row for row in eligible if row[field] == value and row["sample_id"] not in chosen_ids]
        if candidates:
            row = sorted(candidates, key=lambda item: item["sample_id"])[0]
            chosen.append(row)
            chosen_ids.add(row["sample_id"])
    for row in sorted(eligible, key=lambda item: hashlib.sha256(item["sample_id"].encode()).hexdigest()):
        if len(chosen) >= count:
            break
        if row["sample_id"] not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(row["sample_id"])
    return chosen[:count]


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    x1, y1 = max(lx1, rx1), max(ly1, ry1)
    x2, y2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _draw_yolo_overlay(image_rgb: np.ndarray, row: dict[str, Any]) -> Image.Image:
    image = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    bbox = row.get("bbox_xyxy")
    detected = bool(row.get("detected"))
    if bbox is not None:
        draw.rectangle(tuple(float(value) for value in bbox), outline=(40, 255, 80) if detected else (255, 180, 20), width=3)
    label = f"t={row['normalized_time']:.2f} conf={row.get('confidence') or 0:.2f}"
    draw.rectangle((0, 0, min(image.width, 220), 24), fill="black")
    draw.text((5, 5), label, fill="white" if detected else "orange")
    return image


def _save_montage_pages(
    panels: Sequence[tuple[dict[str, Any], Image.Image]], montage_dir: Path, frames_per_trial: int
) -> list[str]:
    montage_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[dict[str, Any], Image.Image]]] = {}
    for row, panel in panels:
        grouped.setdefault(row["sample_id"], []).append((row, panel))
    paths: list[str] = []
    trial_groups = list(grouped.values())
    for page_index in range(0, len(trial_groups), 10):
        page = trial_groups[page_index : page_index + 10]
        cell_width, cell_height = 320, 275
        canvas = Image.new("RGB", (frames_per_trial * cell_width, len(page) * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, group in enumerate(page):
            group = sorted(group, key=lambda item: item[0]["frame_order"])
            for column, (record, image) in enumerate(group):
                thumb = image.copy()
                thumb.thumbnail((cell_width, 240))
                canvas.paste(thumb, (column * cell_width, row_index * cell_height + 28))
                if column == 0:
                    header = (
                        f"c{record['class_id']:02d} {record['action_name']} | "
                        f"{record['user_id']} | {record['duration_bucket']}"
                    )
                    draw.text((4, row_index * cell_height + 5), header, fill="black")
        path = montage_dir / f"thermal_yolo_pose_montage_{page_index // 10 + 1:02d}.jpg"
        canvas.save(path, quality=90)
        paths.append(str(path.relative_to(PROJECT_ROOT).as_posix()))
    return paths


def run_localization_audit(
    rows: Sequence[dict[str, Any]],
    weights: Path,
    montage_dir: Path,
    frames_per_trial: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    frame_records: list[dict[str, Any]] = []
    for row in rows:
        files = _image_files(Path(row["paths"]["Thermal"]))
        indices = list(dict.fromkeys(sample_thermal_indices(len(files), frames_per_trial)))
        for order, index in enumerate(indices):
            frame_records.append(
                {
                    "sample_id": row["sample_id"],
                    "class_id": row["class_id"],
                    "action_name": row["action_name"],
                    "user_id": row["user_id"],
                    "duration_bucket": row["duration_bucket"],
                    "frame_order": order,
                    "frame_index": index,
                    "normalized_time": 0.0 if len(files) == 1 else index / (len(files) - 1),
                    "path": str(files[index]),
                }
            )
    panels: list[tuple[dict[str, Any], Image.Image]] = []
    for start in range(0, len(frame_records), batch_size):
        batch = frame_records[start : start + batch_size]
        images_bgr = [cv2.imdecode(np.fromfile(row["path"], dtype=np.uint8), cv2.IMREAD_COLOR) for row in batch]
        if any(image is None for image in images_bgr):
            raise RuntimeError("Representative localization frame became undecodable")
        results = model.predict(images_bgr, device=device, imgsz=640, conf=0.01, verbose=False)
        for record, image_bgr, result in zip(batch, images_bgr, results, strict=True):
            height, width = image_bgr.shape[:2]
            confidences = (
                result.boxes.conf.detach().float().cpu().numpy()
                if result.boxes is not None and len(result.boxes)
                else np.asarray([], dtype=np.float32)
            )
            best = int(np.argmax(confidences)) if len(confidences) else None
            confidence = float(confidences[best]) if best is not None else None
            bbox = (
                result.boxes.xyxy[best].detach().float().cpu().numpy().tolist()
                if best is not None
                else None
            )
            detected = confidence is not None and confidence >= 0.25
            area_ratio = None
            edge_clipped = False
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / width / height
                edge_clipped = x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2
            failure_types: list[str] = []
            if confidence is None:
                failure_types.append("no_candidate_at_0.01")
            elif not detected:
                failure_types.append("low_confidence_below_0.25")
            if sum(float(value) >= 0.25 for value in confidences) > 1:
                failure_types.append("multiple_person_candidates")
            if detected and area_ratio is not None and area_ratio < 0.05:
                failure_types.append("tiny_bbox")
            if detected and edge_clipped:
                failure_types.append("edge_clipped_bbox")
            record.update(
                {
                    "detected": detected,
                    "confidence": confidence,
                    "bbox_xyxy": bbox,
                    "bbox_area_ratio": area_ratio,
                    "candidate_count_at_0.25": int(sum(float(value) >= 0.25 for value in confidences)),
                    "failure_types": failure_types,
                }
            )
            panels.append((record, _draw_yolo_overlay(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), record)))
    transition_ious: list[float] = []
    by_trial: dict[str, list[dict[str, Any]]] = {}
    for row in frame_records:
        by_trial.setdefault(row["sample_id"], []).append(row)
    for group in by_trial.values():
        group.sort(key=lambda row: row["frame_order"])
        for left, right in zip(group, group[1:]):
            if left["detected"] and right["detected"]:
                iou = _bbox_iou(left["bbox_xyxy"], right["bbox_xyxy"])
                transition_ious.append(iou)
                right["previous_bbox_iou"] = iou
                if iou < 0.25:
                    right["failure_types"].append("bbox_discontinuity_iou_below_0.25")
            else:
                right["previous_bbox_iou"] = None
    failure_counts: dict[str, int] = {}
    for row in frame_records:
        for failure in row["failure_types"]:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
    detected_confidences = [row["confidence"] for row in frame_records if row["detected"]]
    montage_paths = _save_montage_pages(panels, montage_dir, frames_per_trial)
    return {
        "model": "YOLO11n-pose",
        "weights_path": str(weights),
        "weights_sha256": sha256_file(weights),
        "weights_bytes": weights.stat().st_size,
        "threshold": 0.25,
        "representative_trials": len(by_trial),
        "representative_frames": len(frame_records),
        "detection_rate": float(np.mean([row["detected"] for row in frame_records])),
        "confidence": _quantiles(detected_confidences),
        "bbox_continuity_iou": _quantiles(transition_ious),
        "bbox_continuity_rate_iou_ge_0_5": (
            float(np.mean(np.asarray(transition_ious) >= 0.5)) if transition_ious else None
        ),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "montage_paths": montage_paths,
        "coverage": {
            "users": len({row["user_id"] for row in frame_records}),
            "classes": len({row["class_id"] for row in frame_records}),
            "duration_buckets": sorted({row["duration_bucket"] for row in frame_records}),
        },
        "by_user": summarize_localization_groups(frame_records, "user_id"),
        "by_class": summarize_localization_groups(frame_records, "class_id"),
        "by_duration_bucket": summarize_localization_groups(
            frame_records, "duration_bucket"
        ),
        "frame_records": [
            {key: value for key, value in row.items() if key != "path"}
            for row in frame_records
        ],
    }


def _markdown_table(rows: Sequence[dict[str, Any]], key: str) -> list[str]:
    lines = [
        f"| {key} | Canonical | Present | Decodable | Usable | Missing | Frames min/median/p95/max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        quantiles = row["frame_count"]
        frame_text = "/".join(
            "n/a" if quantiles[name] is None else f"{quantiles[name]:.0f}"
            for name in ("min", "median", "p95", "max")
        )
        lines.append(
            f"| {row[key]} | {row['canonical_trials']} | {row['directory_present']} | "
            f"{row['decodable_trials']} | {row['usable_trials']} | {row['missing_directories']} | {frame_text} |"
        )
    return lines


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["thermal_data_audit"]["summary"]
    alignment = report["temporal_alignment_audit"]
    localization = report["localization_audit"]
    sampling = report["thermal_sampling_contract"]
    lines = [
        "# Thermal Stage T0 Data and Alignment Audit",
        "",
        "## Scope and hard boundaries",
        "",
        f"- Baseline commit: `{report['provenance']['baseline_commit_sha']}`.",
        f"- Branch: `{report['provenance']['branch']}`; worktree: `{report['provenance']['worktree']}`.",
        "- Population: official train-14 users only. The sealed users `user4,user17,user23,user24`, heldout labels, competition test, and quarantined evidence were not enumerated or opened.",
        "- IR/X3D is frozen. This audit performed no training, tuning, checkpoint selection, or ExpertEvidence write.",
        "- Raw modality data and YOLO weights were opened read-only; outputs are reports and montages in this worktree.",
        "- Deployment ledger at T0: frozen IR X3D + shared YOLO known subtotal `20,644,200` bytes; new Thermal learned artifacts `0` bytes; hard complete-package ceiling `<95,000,000` bytes. A complete-package pass is not claimed before the other retained experts and a trained Thermal candidate are inventoried.",
        "",
        "## Executive findings",
        "",
        f"- Canonical train-14 trials: **{summary['canonical_trials']}**. Thermal directory present: **{summary['directory_present_trials']}**; decodable: **{summary['decodable_trials']}**; usable: **{summary['usable_trials']}**. These are separate states.",
        f"- Thermal image files: **{summary['frame_files']}**; decoded: **{summary['decoded_frames']}**; corrupt JPEG: **{summary['corrupt_jpegs']}**; exact duplicate decoded frames: **{summary['duplicate_frames']}**.",
        f"- Single-frame trials: **{summary['single_frame_trials']}**; 2-4-frame-or-single trials: **{summary['extremely_short_trials']}**; below 13 frames: **{summary['shorter_than_13_trials']}**. None were removed from the canonical population.",
        f"- Rendering verdict: `{summary['rendering_verdict']}`. Channel evidence rejects grayscale replication, but rendered RGB JPEGs cannot establish or exclude per-frame automatic temperature scaling because raw radiometric values and scale metadata are absent.",
        f"- IR/Thermal common trials: **{alignment['ir_thermal']['common_trials']}**; median Thermal/IR frame-count ratio: **{alignment['ir_thermal']['frame_count_ratio']['median']:.3f}**. Thermal has independent frame numbering and no shared filename timestamps with IR.",
        f"- Motion alignment on {alignment['motion_diagnostic']['trial_count']} stratified trials produced median correlation `{alignment['motion_diagnostic']['correlation']['median']:.3f}` with materially varying offsets/scales. Motion peaks are diagnostic only and are not part of the sampling contract.",
        f"- Thermal-native YOLO detection rate: **{localization['detection_rate']:.1%}** on {localization['representative_frames']} representative frames; median detected confidence: **{localization['confidence']['median']:.3f}**; median adjacent-sample bbox IoU: **{localization['bbox_continuity_iou']['median']:.3f}**.",
        "- Current evidence supports trial-level fusion. It does not support frame-level IR/Thermal registration or scaled reuse of IR boxes.",
        "",
        "## 1. Thermal data audit",
        "",
        "`directory_present` means the canonical trial directory exists. `decodable` means at least one image decodes. `usable` means at least one Thermal frame can enter the simple full-frame/native-localization path; defects and short duration remain label-free quality fields.",
        "",
        "### Overall",
        "",
        f"- Frame-count distribution over decodable trials: `{json.dumps(summary['frame_count'], ensure_ascii=False)}`.",
        f"- Trial-level distinct-frame-ratio distribution: `{json.dumps(summary['distinct_frame_ratio'], ensure_ascii=False)}`.",
        f"- Rendered palette-curve stability distribution: `{json.dumps(summary['palette_curve_stability'], ensure_ascii=False)}`.",
        f"- Resolutions: `{json.dumps(summary['resolution_counts'], ensure_ascii=False)}`.",
        f"- Rendering categories: `{json.dumps(summary['rendering_counts'], ensure_ascii=False)}`.",
        "- Pseudocolor stability is assessed from channel spread and hue-versus-luminance palette curves on uniformly located frames. This demonstrates a consistent rendered palette family, not calibrated temperature equivalence across frames.",
        "",
        "### By development split",
        "",
        *_markdown_table(report["thermal_data_audit"]["by_development_split"], "development_split"),
        "",
        "### By OOF validation owner",
        "",
        *_markdown_table(report["thermal_data_audit"]["by_oof_fold"], "oof_fold"),
        "",
        "### By user",
        "",
        *_markdown_table(report["thermal_data_audit"]["by_user"], "user_id"),
        "",
        "Class-level details are preserved in the JSON report to keep this document scannable.",
        "",
        "## 2. IR/Depth temporal relationship",
        "",
        "- Pairing by directory sort position is forbidden. Each modality is independently naturally ordered, assigned `t=i/(N-1)`, and only then mapped by nearest normalized time as a candidate correspondence.",
        f"- IR/Thermal frame-count ratio: `{json.dumps(alignment['ir_thermal']['frame_count_ratio'], ensure_ascii=False)}`.",
        f"- Depth/Thermal frame-count ratio: `{json.dumps(alignment['depth_thermal']['frame_count_ratio'], ensure_ascii=False)}`.",
        f"- IR/Thermal Tukey outliers: **{alignment['ir_thermal']['outlier_count']}**; representative extremes are listed in the JSON report.",
        "- IR/Depth filenames carry matching wall-clock timestamps and frame numbers in the audited export. Thermal filenames carry only independent `frame_N` indices; no common timestamp boundary is available.",
        f"- Motion diagnostic offset: `{json.dumps(alignment['motion_diagnostic']['offset'], ensure_ascii=False)}`; scale: `{json.dumps(alignment['motion_diagnostic']['scale'], ensure_ascii=False)}`; DTW: `{json.dumps(alignment['motion_diagnostic']['dtw_distance'], ensure_ascii=False)}`.",
        "- Correlation/DTW can diagnose trial-relative lag or rate mismatch. Without synchronized timestamps, calibration targets, or stable cross-trial offset/scale, they do not prove frame identity.",
        "",
        "### Evidence boundary",
        "",
        "| Claim | Stage T0 evidence | Decision |",
        "| --- | --- | --- |",
        "| Same trial identity | Canonical class/user/trial key and common directories | Sufficient for trial-level late fusion |",
        "| Candidate relative-time correspondence | Independent normalized timelines | Diagnostic only |",
        "| Temporal frame registration | No shared Thermal timestamps; variable count ratio and motion offset/scale | Not established |",
        "| Spatial registration / IR bbox transfer | Different resolution plus no camera calibration or pixel correspondence | Not established; prohibited |",
        "",
        "## 3. Thermal-native localization",
        "",
        f"- YOLO weights: `{localization['weights_sha256']}` ({localization['weights_bytes']} bytes), threshold `0.25`, `imgsz=640`.",
        f"- Confidence distribution: `{json.dumps(localization['confidence'], ensure_ascii=False)}`.",
        f"- Bbox continuity distribution: `{json.dumps(localization['bbox_continuity_iou'], ensure_ascii=False)}`; IoU >= 0.5 rate: `{localization['bbox_continuity_rate_iou_ge_0_5']}`.",
        f"- Failure types: `{json.dumps(localization['failure_type_counts'], ensure_ascii=False)}`.",
        f"- Representative coverage: **{localization['coverage']['users']} users**, **{localization['coverage']['classes']} classes**, duration buckets `{', '.join(localization['coverage']['duration_buckets'])}`. Per-user, per-class, and per-duration results are in the JSON report.",
        "- Numeric detections are not accepted as ground truth. The montage is the required human check for false people, missed limbs, clipped subjects, furniture/background detections, and context loss.",
        f"- Human montage verdict: `{localization['human_montage_review']['verdict']}`. Observed modes: {'; '.join(localization['human_montage_review']['observed_failure_modes'])}.",
        "- IR bboxes were neither read nor scaled. If Thermal-native YOLO proves unreliable after human review, retain a heat/motion context crop when label-free confidence is adequate and a full-frame fallback otherwise; encode the route and reliability in availability/quality. Do not train a detector in T0.",
        "",
        "Montages:",
        "",
        *[f"- `{montage}`" for montage in localization["montage_paths"]],
        "",
        "## 4. Frozen Thermal sampling contract",
        "",
        f"- Timeline: `{sampling['timeline']}`.",
        f"- Targets: `{sampling['targets']}`.",
        f"- Short trials: `{sampling['short_trial_policy']}`.",
        f"- Quality: `{', '.join(sampling['quality_fields'])}`.",
        "- The sampler never imports IR indices, never uses IR's 13-frame contract, and never selects motion peaks. Variable duration stays explicit; singleton/short trials remain canonical and use repeated nearest-source indices plus a mask/quality signal when a fixed tensor is required.",
        "",
        "## 5. Frozen candidate order after T0",
        "",
        "1. **iFormer-T + TSM**: primary budget-friendly Thermal-native candidate.",
        "2. **iFormer-S + TSM**: upgrade only after pretrained-loading provenance, serialized bytes, and complete-package headroom pass.",
        "3. **Pretrained MobileNetV3 + TSM**: matched control using the same Thermal sampler, localization routes, training protocol, and evaluation surface.",
        "4. **Deferred**: VideoMamba, DART, and IR+Thermal early fusion. They are outside the first controlled Thermal generation.",
        "",
        "## 6. Approval gate and later evidence contract",
        "",
        "After human approval, development starts only on the fixed train12 / user6-user7 validation split. The recipe is then frozen before generating formal train-14 OOF evidence on the shared persisted folds.",
        "Every later Thermal expert must emit 40-class logits, availability, label-free quality/quality-mask, fusion-quality score, hashes, deployed bytes, preprocessing dependencies, and fold lineage through `ExpertEvidence`. Reports must include Accuracy, Macro-F1, worst-user Accuracy, IR unique-correct/oracle-pair metrics, exact deployment bytes, and latency.",
        "",
        "**Stage T0 stop:** no formal model was trained. Await human review of this report and montages before Stage T1.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    development = _load_json(args.development_split)
    oof = _load_json(args.oof_split)
    train14_users = _oof_train_users(oof)
    development_users = validate_train14_users(
        list(development["train_user_ids"]) + list(development["validation_user_ids"])
    )
    if set(development_users) != set(train14_users):
        raise ValueError("Development split does not cover the exact formal train-14 users")
    trials = inventory_train14_trials(args.data_root, train14_users)
    if len(trials) != int(oof["trial_count"]):
        raise ValueError(f"Expected {oof['trial_count']} train-14 trials, found {len(trials)}")
    development_train = set(str(user) for user in development["train_user_ids"])
    development_val = set(str(user) for user in development["validation_user_ids"])
    fold_by_user: dict[str, int] = {}
    for fold in oof["folds"]:
        for user in fold["validation_user_ids"]:
            if str(user) in fold_by_user:
                raise ValueError(f"Duplicate OOF validation owner: {user}")
            fold_by_user[str(user)] = int(fold["fold"])
    for row in trials:
        row["development_split"] = (
            "train12" if row["user_id"] in development_train else "val_user6_user7"
        )
        row["oof_fold"] = f"fold_{fold_by_user[row['user_id']]}"
        row["paths"] = dict(row["paths"])
        row["thermal_path"] = row["paths"].get(
            "Thermal",
            str(args.data_root / "Thermal" / f"{row['class_id']}_{row['action_name']}" / row["user_id"] / row["trial_id"]),
        )

    def audit_row(row: dict[str, Any]) -> dict[str, Any]:
        return {**row, **audit_trial_images(Path(row["thermal_path"]))}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        audited = list(pool.map(audit_row, trials))
    for row in audited:
        row["duration_bucket"] = _duration_bucket(int(row["decodable_frame_count"]))
    resolution_counts: dict[str, int] = {}
    rendering_counts: dict[str, int] = {}
    for row in audited:
        for resolution in row["resolutions"]:
            resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1
        rendering_counts[row["rendering"]] = rendering_counts.get(row["rendering"], 0) + 1
    rendering_verdict = (
        "stable_pseudocolor_rendering; grayscale replication rejected; per-frame auto-scale unresolved"
        if rendering_counts.get("stable_pseudocolor", 0) > rendering_counts.get("grayscale_copy", 0)
        else "mixed rendering; inspect trial-level categories"
    )
    thermal_summary = {
        "canonical_trials": len(audited),
        "directory_present_trials": sum(row["directory_present"] for row in audited),
        "decodable_trials": sum(row["decodable"] for row in audited),
        "usable_trials": sum(row["usable"] for row in audited),
        "frame_files": sum(row["file_count"] for row in audited),
        "decoded_frames": sum(row["decodable_frame_count"] for row in audited),
        "corrupt_jpegs": sum(row["corrupt_jpeg_count"] for row in audited),
        "duplicate_frames": sum(row["duplicate_frame_count"] for row in audited),
        "consecutive_duplicate_frames": sum(row["consecutive_duplicate_frame_count"] for row in audited),
        "single_frame_trials": sum(row["single_frame"] for row in audited),
        "extremely_short_trials": sum(row["extremely_short_le4"] for row in audited),
        "shorter_than_13_trials": sum(row["shorter_than_13"] for row in audited),
        "frame_count": _quantiles([row["decodable_frame_count"] for row in audited if row["decodable"]]),
        "distinct_frame_ratio": _quantiles([row["distinct_frame_ratio"] for row in audited if row["distinct_frame_ratio"] is not None]),
        "palette_curve_stability": _quantiles(
            [
                row["palette_curve_stability"]
                for row in audited
                if row["palette_curve_stability"] is not None
            ]
        ),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "rendering_counts": dict(sorted(rendering_counts.items())),
        "rendering_verdict": rendering_verdict,
    }

    relation_rows: list[dict[str, Any]] = []
    for row in audited:
        if not row["directory_present"]:
            continue
        relation: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "class_id": row["class_id"],
            "user_id": row["user_id"],
            "thermal": {
                "frame_count": row["file_count"],
                "first_frame_number": row["first_frame_number"],
                "last_frame_number": row["last_frame_number"],
            },
        }
        for modality, key in (("IR", "ir"), ("Depth_Color", "depth")):
            if modality in row["paths"]:
                relation[key] = _frame_boundary(Path(row["paths"][modality]), modality)
                relation[f"thermal_to_{key}_frame_ratio"] = (
                    row["file_count"] / relation[key]["frame_count"]
                    if relation[key]["frame_count"]
                    else None
                )
        relation_rows.append(relation)

    def relation_summary(key: str) -> dict[str, Any]:
        ratio_key = f"thermal_to_{key}_frame_ratio"
        applicable = [row for row in relation_rows if key in row and row.get(ratio_key) is not None]
        ratios = np.asarray([row[ratio_key] for row in applicable], dtype=np.float64)
        q1, q3 = np.quantile(ratios, [0.25, 0.75]) if len(ratios) else (np.nan, np.nan)
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outliers = [row for row in applicable if row[ratio_key] < lower or row[ratio_key] > upper]
        extremes = sorted(applicable, key=lambda row: row[ratio_key])[:5] + sorted(
            applicable, key=lambda row: row[ratio_key], reverse=True
        )[:5]
        return {
            "common_trials": len(applicable),
            "frame_count_ratio": _quantiles(ratios.tolist()),
            "tukey_lower": float(lower) if np.isfinite(lower) else None,
            "tukey_upper": float(upper) if np.isfinite(upper) else None,
            "outlier_count": len(outliers),
            "extreme_trials": [
                {"sample_id": row["sample_id"], "ratio": row[ratio_key]} for row in extremes
            ],
        }

    representative = _representative_rows(audited, args.representative_trials)
    candidate_examples: list[dict[str, Any]] = []
    for row in representative[:12]:
        thermal_count = len(_image_files(Path(row["paths"]["Thermal"])))
        ir_count = len(_image_files(Path(row["paths"]["IR"])))
        pairs = normalized_time_candidate_pairs(ir_count, thermal_count)
        positions = sorted({0, len(pairs) // 2, len(pairs) - 1})
        candidate_examples.append(
            {
                "sample_id": row["sample_id"],
                "ir_frame_count": ir_count,
                "thermal_frame_count": thermal_count,
                "pairs": [
                    {
                        "ir_index": pairs[index][0],
                        "thermal_index": pairs[index][1],
                        "normalized_time": (
                            0.0 if thermal_count == 1 else index / (thermal_count - 1)
                        ),
                    }
                    for index in positions
                ],
            }
        )
    motion_rows: list[dict[str, Any]] = []
    for row in representative:
        ir_energy = _motion_energy(Path(row["paths"]["IR"]))
        thermal_energy = _motion_energy(Path(row["paths"]["Thermal"]))
        alignment = estimate_motion_alignment(ir_energy, thermal_energy)
        motion_rows.append(
            {
                "sample_id": row["sample_id"],
                "class_id": row["class_id"],
                "action_name": row["action_name"],
                "user_id": row["user_id"],
                "duration_bucket": row["duration_bucket"],
                "ir_motion_steps": len(ir_energy),
                "thermal_motion_steps": len(thermal_energy),
                "ir_motion_peak_normalized_time": (
                    (int(np.argmax(ir_energy)) + 0.5) / len(ir_energy) if len(ir_energy) else None
                ),
                "thermal_motion_peak_normalized_time": (
                    (int(np.argmax(thermal_energy)) + 0.5) / len(thermal_energy)
                    if len(thermal_energy)
                    else None
                ),
                **alignment,
            }
        )
    valid_motion = [row for row in motion_rows if row["correlation"] is not None]
    motion_summary = {
        "trial_count": len(motion_rows),
        "valid_alignment_trials": len(valid_motion),
        "user_count": len({row["user_id"] for row in motion_rows}),
        "class_count": len({row["class_id"] for row in motion_rows}),
        "duration_buckets": sorted({row["duration_bucket"] for row in motion_rows}),
        "correlation": _quantiles([row["correlation"] for row in valid_motion]),
        "offset": _quantiles([row["offset"] for row in valid_motion]),
        "scale": _quantiles([row["scale"] for row in valid_motion]),
        "dtw_distance": _quantiles([row["dtw_distance"] for row in valid_motion]),
        "stable_high_correlation_fraction": (
            float(
                np.mean(
                    [
                        row["correlation"] >= 0.5
                        and abs(row["offset"]) <= 0.10
                        and 0.90 <= row["scale"] <= 1.10
                        for row in valid_motion
                    ]
                )
            )
            if valid_motion
            else None
        ),
        "motion_peaks_use": "alignment_diagnostic_only_not_training_frame_selection",
        "trials": motion_rows,
    }

    if args.skip_localization:
        localization = {
            "skipped": True,
            "weights_path": str(args.yolo_weights),
            "weights_sha256": sha256_file(args.yolo_weights),
            "weights_bytes": args.yolo_weights.stat().st_size,
            "representative_frames": 0,
            "detection_rate": 0.0,
            "confidence": _quantiles([]),
            "bbox_continuity_iou": _quantiles([]),
            "bbox_continuity_rate_iou_ge_0_5": None,
            "failure_type_counts": {"localization_skipped": 1},
            "montage_paths": [],
            "coverage": {"users": 0, "classes": 0, "duration_buckets": []},
            "by_user": [],
            "by_class": [],
            "by_duration_bucket": [],
            "frame_records": [],
            "human_montage_review": {
                "pages_reviewed": [],
                "verdict": "not_reviewed_localization_skipped",
                "observed_failure_modes": [],
            },
        }
    else:
        localization = run_localization_audit(
            representative,
            args.yolo_weights,
            args.montage_dir,
            args.localization_frames,
            args.batch_size,
            args.device,
        )
        localization["human_montage_review"] = {
            "pages_reviewed": localization["montage_paths"],
            "verdict": "conditional_locator_only_full_frame_fallback_required",
            "observed_failure_modes": [
                "near-field subjects frequently touch image boundaries",
                "far seated or crouched subjects produce small or low-confidence boxes",
                "person scale and pose changes can cause low-confidence gaps and box discontinuity",
                "when a subject leaves the frame, a small heated object or background region can be selected as a false person",
                "tight person-only crops risk discarding action-defining table, screen, cup, phone, and room context",
            ],
        }

    baseline_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
    ).strip()
    authority_paths = [
        PROJECT_ROOT / "docs/superpowers/plans/2026-08-10-x3d-s-adaptive-multiclip.md",
        PROJECT_ROOT / "docs/superpowers/plans/2026-08-11-six-modal-program-charter.md",
        PROJECT_ROOT / "docs/superpowers/specs/2026-08-11-six-modal-sparse-evidence-fusion-design.md",
        PROJECT_ROOT / "reports/task03_schema_notes.md",
        PROJECT_ROOT / "reports/task03_baseline_fold0_report.md",
        args.development_split,
        args.oof_split,
    ]
    report = {
        "schema_version": 1,
        "stage": "thermal_stage_t0",
        "status": "audit_complete_training_blocked_pending_human_approval",
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_commit_sha": baseline_sha,
            "branch": branch,
            "worktree": str(PROJECT_ROOT),
            "data_root": str(args.data_root),
            "population": "official_train14_only",
            "train14_users": train14_users,
            "sealed_heldout_users_not_enumerated": sorted(SEALED_HELDOUT_USERS),
            "competition_test_accessed": False,
            "quarantined_evidence_accessed": False,
            "ir_x3d_training_or_tuning_performed": False,
            "authority_file_sha256": {
                str(path.relative_to(PROJECT_ROOT).as_posix()): sha256_file(path)
                for path in authority_paths
            },
        },
        "thermal_data_audit": {
            "summary": thermal_summary,
            "anomalies": {
                "missing_directory_sample_ids": [
                    row["sample_id"] for row in audited if not row["directory_present"]
                ],
                "decode_failure_trials": [
                    {
                        "sample_id": row["sample_id"],
                        "decode_failure_count": row["decode_failure_count"],
                    }
                    for row in audited
                    if row["decode_failure_count"]
                ],
                "duplicate_frame_trials": [
                    {
                        "sample_id": row["sample_id"],
                        "duplicate_frame_count": row["duplicate_frame_count"],
                        "distinct_frame_ratio": row["distinct_frame_ratio"],
                    }
                    for row in audited
                    if row["duplicate_frame_count"]
                ],
                "single_frame_sample_ids": [
                    row["sample_id"] for row in audited if row["single_frame"]
                ],
                "extremely_short_le4_sample_ids": [
                    row["sample_id"] for row in audited if row["extremely_short_le4"]
                ],
            },
            "by_user": _group_summary(audited, "user_id"),
            "by_class": _group_summary(audited, "class_id"),
            "by_development_split": _group_summary(audited, "development_split"),
            "by_oof_fold": _group_summary(audited, "oof_fold"),
            "canonical_trial_records": [
                serializable_trial_record(row) for row in audited
            ],
        },
        "temporal_alignment_audit": {
            "pairing_prohibition": "never_pair_frames_by_raw_sorted_position",
            "candidate_rule": "independently_sort_each_modality_then_nearest_neighbor_on_t=i/(N-1)",
            "ir_thermal": relation_summary("ir"),
            "depth_thermal": relation_summary("depth"),
            "motion_diagnostic": motion_summary,
            "normalized_candidate_pair_examples": candidate_examples,
            "trial_relations": relation_rows,
            "evidence_decision": {
                "trial_level_fusion_supported": True,
                "frame_level_temporal_registration_supported": False,
                "frame_level_spatial_registration_supported": False,
                "ir_bbox_reuse_supported": False,
                "reason": "common trial identity but no Thermal timestamps/calibration; variable temporal scale/offset and unproven camera geometry",
            },
        },
        "localization_audit": localization,
        "deployment_budget_audit": {
            "hard_complete_package_ceiling_exclusive_bytes": 95000000,
            "known_frozen_ir_x3d_plus_shared_yolo_subtotal_bytes": 20644200,
            "stage_t0_new_learned_artifact_bytes": 0,
            "bytes_before_ceiling_excluding_other_retained_experts": 74355800,
            "complete_package_gate_status": "not_evaluable_at_t0_no_thermal_model_or_complete_six_expert_inventory",
            "future_requirement": "inventory_every_inference_artifact_once_and_require_total_below_95000000",
        },
        "thermal_sampling_contract": {
            "timeline": "Thermal natural frame order with t=i/(N-1); singleton t=0",
            "targets": "uniform normalized-time targets mapped to nearest Thermal source index",
            "short_trial_policy": "retain canonical row; repeat nearest source index only when a fixed tensor is required; emit source-uniqueness mask and quality",
            "variable_duration_supported": True,
            "single_frame_supported": True,
            "imports_ir_indices": False,
            "copies_ir_13_frame_contract": False,
            "uses_motion_peak_sampling": False,
            "quality_fields": [
                "directory_present",
                "decodable_frame_fraction",
                "distinct_frame_ratio",
                "unique_sampled_source_ratio",
                "duration_bucket",
                "localizer_confidence",
                "bbox_continuity",
                "fallback_route",
            ],
        },
        "candidate_freeze": [
            {"rank": 1, "candidate": "iFormer-T + TSM", "status": "primary"},
            {
                "rank": 2,
                "candidate": "iFormer-S + TSM",
                "status": "conditional_on_pretrained_loading_and_byte_audit",
            },
            {"rank": 3, "candidate": "pretrained MobileNetV3 + TSM", "status": "matched_control"},
            {"candidate": "VideoMamba", "status": "deferred"},
            {"candidate": "DART", "status": "deferred"},
            {"candidate": "IR+Thermal early fusion", "status": "deferred"},
        ],
        "future_experiment_contract": {
            "development_split": str(args.development_split.relative_to(PROJECT_ROOT).as_posix()),
            "formal_oof_split": str(args.oof_split.relative_to(PROJECT_ROOT).as_posix()),
            "output_classes": 40,
            "required_evidence": ["logits", "availability", "quality", "quality_mask", "fusion_quality_score", "ExpertEvidence"],
            "required_metrics": [
                "Accuracy",
                "Macro-F1",
                "worst-user Accuracy",
                "IR unique-correct",
                "IR oracle-pair Accuracy",
                "deployed serialized bytes",
                "latency",
            ],
            "complete_package_byte_ceiling_exclusive": 95000000,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown_output)
    print(json.dumps({"json": str(args.json_output), "markdown": str(args.markdown_output)}))


if __name__ == "__main__":
    main()
