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
from audit_motion_crop import analyse_trial


PROJECT_DIR = Path(__file__).resolve().parent
SMALL_ACTION_IDS = (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blinded local-view audit pack for fixed small actions")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifest.csv")
    parser.add_argument("--existing-audit", type=Path, default=PROJECT_DIR / "data" / "motion_crop_audit.csv")
    parser.add_argument("--oof-root", type=Path, default=PROJECT_DIR / "runs" / "p5_oof_fusion")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "data" / "local_action_audit_v1")
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_oof(root: Path) -> dict[str, dict[str, object]]:
    result = {}
    for fold in range(3):
        data = np.load(root / f"fold_{fold}" / "logits.npz")
        fused = 0.6 * data["skeleton_logits"] + 0.4 * data["depth_logits"]
        probabilities = np.exp(fused - fused.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        for index, sample_id in enumerate(data["sample_ids"].astype(str)):
            result[sample_id] = {
                "fold": fold,
                "label": int(data["labels"][index]),
                "prediction": int(fused[index].argmax()),
                "confidence": float(probabilities[index].max()),
            }
    return result


def select_rows(
    rows: list[dict[str, str]],
    existing_ids: set[str],
    oof: dict[str, dict[str, object]],
    samples_per_class: int,
    seed: int,
) -> list[dict[str, str]]:
    by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["class_id"]) in SMALL_ACTION_IDS and row["sample_id"] in oof:
            by_class[int(row["class_id"])].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for class_id in SMALL_ACTION_IDS:
        candidates = sorted(by_class[class_id], key=lambda row: row["sample_id"])
        rng.shuffle(candidates)
        existing = [row for row in candidates if row["sample_id"] in existing_ids]
        chosen = existing[: min(len(existing), samples_per_class)]
        users = {row["user_id"] for row in chosen}
        remaining = [row for row in candidates if row not in chosen]
        remaining.sort(
            key=lambda row: (
                oof[row["sample_id"]]["prediction"] == class_id,
                row["user_id"] in users,
                row["sample_id"],
            )
        )
        for row in remaining:
            if len(chosen) >= samples_per_class:
                break
            chosen.append(row)
            users.add(row["user_id"])
        selected.extend(chosen)
    return selected


def load_resized(path: Path, mode: str = "RGB") -> Image.Image:
    with Image.open(path) as image:
        return image.convert(mode).resize((320, 240), Image.Resampling.BILINEAR).convert("RGB")


def draw_box(image: Image.Image, box: list[int]) -> Image.Image:
    result = image.copy()
    ImageDraw.Draw(result).rectangle(tuple(box), outline=(0, 255, 0), width=3)
    return result


def trial_montage(row: dict[str, str], record: dict[str, object], overlay: np.ndarray) -> Image.Image:
    depth = frame_map(Path(row["depth_dir"]), "depth")
    ir = frame_map(Path(row["ir_dir"]), "ir")
    common = sorted(set(depth) & set(ir))
    positions = [int(round(value)) for value in np.linspace(0, len(common) - 1, 3)]
    frame_ids = [common[index] for index in positions]
    box = [int(value) for value in record["bbox"]]
    tiles: list[list[Image.Image]] = [[], [], []]
    for frame_id in frame_ids:
        depth_image = load_resized(depth[frame_id])
        ir_image = load_resized(ir[frame_id], "L")
        crop = depth_image.crop((box[0], box[1], box[2] + 1, box[3] + 1)).resize(
            (320, 240), Image.Resampling.BILINEAR
        )
        tiles[0].append(draw_box(depth_image, box))
        tiles[1].append(crop)
        tiles[2].append(draw_box(ir_image, box))
    tiles[0].append(Image.fromarray(overlay))
    tiles[1].append(Image.new("RGB", (320, 240), (20, 20, 20)))
    tiles[2].append(Image.new("RGB", (320, 240), (20, 20, 20)))
    canvas = Image.new("RGB", (1280, 720), (0, 0, 0))
    for row_index, image_row in enumerate(tiles):
        for column, image in enumerate(image_row):
            canvas.paste(image, (column * 320, row_index * 240))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1279, 24), fill=(0, 0, 0))
    draw.text(
        (5, 5),
        f"GT c{int(row['class_id']):02d} {row['class_name']} | {row['user_id']} {row['trial_id']} | "
        f"fallback={record['fallback']} box={float(record['expanded_bbox_fraction']):.3f}",
        fill=(255, 255, 255),
    )
    return canvas


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    cases = output / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(args.manifest.resolve())
    existing = read_csv(args.existing_audit.resolve())
    existing_ids = {row["sample_id"] for row in existing}
    oof = load_oof(args.oof_root.resolve())
    selected = select_rows(
        manifest, existing_ids, oof, args.samples_per_class, args.seed
    )
    if len(selected) != len(SMALL_ACTION_IDS) * args.samples_per_class:
        raise RuntimeError(f"Unexpected audit selection size: {len(selected)}")

    annotations = []
    overlays_by_class: dict[int, list[tuple[dict[str, object], np.ndarray]]] = defaultdict(list)
    for index, row in enumerate(selected, 1):
        record, overlay = analyse_trial(row, 320, 240)
        case = trial_montage(row, record, overlay)
        case_name = f"c{int(row['class_id']):02d}_{row['user_id']}_{row['trial_id']}.jpg"
        case.save(cases / case_name, quality=84, optimize=True)
        prediction = oof[row["sample_id"]]
        annotations.append(
            {
                "sample_id": row["sample_id"],
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "user_id": row["user_id"],
                "trial_id": row["trial_id"],
                "fold": prediction["fold"],
                "selected_by_existing_audit": int(row["sample_id"] in existing_ids),
                "motion_fallback": int(bool(record["fallback"])),
                "active_pixel_fraction": record["active_pixel_fraction"],
                "expanded_bbox_fraction": record["expanded_bbox_fraction"],
                "case_image": f"cases/{case_name}",
                "localization_coverage": "",
                "raw_observability": "",
                "recommended_view": "",
                "issue_flags": "",
                "reviewer_note": "",
                "oof_prediction_hidden_during_annotation": prediction["prediction"],
                "oof_correct_hidden_during_annotation": int(
                    prediction["prediction"] == int(row["class_id"])
                ),
                "oof_confidence_hidden_during_annotation": prediction["confidence"],
            }
        )
        overlays_by_class[int(row["class_id"])].append((record, overlay))
        if index % 21 == 0 or index == len(selected):
            print(f"local audit {index}/{len(selected)}", flush=True)

    with (output / "annotations.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotations[0]))
        writer.writeheader()
        writer.writerows(annotations)

    overview = Image.new("RGB", (7 * 320, 3 * 240), (0, 0, 0))
    for slot, class_id in enumerate(SMALL_ACTION_IDS):
        examples = overlays_by_class[class_id]
        record, image = max(
            examples,
            key=lambda item: (
                bool(item[0]["fallback"]),
                float(item[0]["expanded_bbox_fraction"]),
            ),
        )
        tile = Image.fromarray(image)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 216, 319, 239), fill=(0, 0, 0))
        draw.text(
            (4, 219),
            f"c{class_id:02d} fallback={int(bool(record['fallback']))}",
            fill=(255, 255, 255),
        )
        row_index, column = divmod(slot, 7)
        overview.paste(tile, (column * 320, row_index * 240))
    overview.save(output / "class_overview.jpg", quality=88, optimize=True)

    small_existing = [row for row in existing if int(row["class_id"]) in SMALL_ACTION_IDS]
    summary = {
        "taxonomy": "fixed small-action taxonomy v1",
        "selected_trials": len(annotations),
        "samples_per_class": args.samples_per_class,
        "classes": list(SMALL_ACTION_IDS),
        "existing_small_action_trials_reused": sum(
            int(row["selected_by_existing_audit"]) for row in annotations
        ),
        "previous_objective_motion_audit": {
            "trials": len(small_existing),
            "fallback_trials": sum(row["fallback"].lower() == "true" for row in small_existing),
            "fallback_fraction": float(
                np.mean([row["fallback"].lower() == "true" for row in small_existing])
            ),
        },
        "new_selection_objective_motion": {
            "fallback_trials": sum(int(row["motion_fallback"]) for row in annotations),
            "fallback_fraction": float(
                np.mean([int(row["motion_fallback"]) for row in annotations])
            ),
        },
        "annotation_axes": {
            "localization_coverage": ["complete", "missing_object_or_context", "motion_only", "background_error"],
            "raw_observability": ["clear", "ambiguous", "not_visible"],
            "recommended_view": ["local", "global", "global_plus_local", "other_modality"],
            "issue_flags": "multi-select",
        },
        "blinding": "OOF prediction columns are placed at the end and must remain hidden during the first annotation pass.",
        "training_gate": "No Local-only training until manual labels are complete and the pass criterion is frozen.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
