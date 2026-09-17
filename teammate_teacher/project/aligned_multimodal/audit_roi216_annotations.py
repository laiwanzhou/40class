from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from audit_motion_crop import analyse_trial
from import_label_studio_roi216 import (
    RAW_HEIGHT,
    RAW_WIDTH,
    read_csv,
    select_final_rectangle,
    strip_explanation,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "local_roi_annotation_v2"
PROJECT_ASSETS = DATA_DIR / "label_studio_roi216"
DEFAULT_EXPORT = (
    PROJECT_ASSETS
    / "exports"
    / "project-3-at-2026-07-27-16-10-11e43ac1.json"
)
DEFAULT_PRIVATE_INDEX = PROJECT_ASSETS / "task_index_private.csv"
DEFAULT_SELECTION = DATA_DIR / "selection.csv"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_OUTPUT = DATA_DIR / "roi_annotation_audit.csv"
EDGE_CHANGE_RATIO = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ROI216 human boxes with the original motion boxes and apply "
            "the annotator's interpretation contract."
        )
    )
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--private-index", type=Path, default=DEFAULT_PRIVATE_INDEX)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def label_studio_box(rectangle: dict[str, Any]) -> tuple[float, float, float, float]:
    value = rectangle["value"]
    x0 = RAW_WIDTH * float(value["x"]) / 100.0
    y0 = RAW_HEIGHT * float(value["y"]) / 100.0
    width = RAW_WIDTH * float(value["width"]) / 100.0
    height = RAW_HEIGHT * float(value["height"]) / 100.0
    return (
        x0,
        y0,
        min(RAW_WIDTH - 1.0, x0 + width - 1.0),
        min(RAW_HEIGHT - 1.0, y0 + height - 1.0),
    )


def map_box(
    box: list[float],
    source_width: int = 320,
    source_height: int = 240,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (
        x0 * RAW_WIDTH / source_width,
        y0 * RAW_HEIGHT / source_height,
        (x1 + 1) * RAW_WIDTH / source_width - 1,
        (y1 + 1) * RAW_HEIGHT / source_height - 1,
    )


def box_area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]) + 1.0)
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]) + 1.0)
    intersection = width * height
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def extract_annotation(task: dict[str, Any]) -> dict[str, Any]:
    annotations = [
        annotation
        for annotation in task.get("annotations", [])
        if not annotation.get("was_cancelled", False)
    ]
    if len(annotations) != 1:
        raise ValueError(
            f"{task.get('data', {}).get('task_key')}: expected one annotation"
        )
    results = annotations[0].get("result", [])
    choices: dict[str, str] = {}
    notes: list[str] = []
    rectangles: list[dict[str, Any]] = []
    for result in results:
        if result.get("type") == "choices":
            values = result.get("value", {}).get("choices", [])
            if len(values) != 1:
                raise ValueError("Every choice result must contain one value")
            choices[str(result["from_name"])] = strip_explanation(str(values[0]))
        elif result.get("type") == "textarea":
            notes.extend(
                str(value).strip()
                for value in result.get("value", {}).get("text", [])
                if str(value).strip()
            )
        elif (
            result.get("type") == "rectanglelabels"
            and result.get("from_name") == "roi_box"
        ):
            rectangles.append(result)
    rectangle, ignored = select_final_rectangle(
        rectangles,
        str(task.get("data", {}).get("task_key", "")),
    )
    return {
        "region_status": choices["region_status"],
        "temporal_status": choices["single_box_temporally_valid"],
        "note": " | ".join(notes),
        "box": label_studio_box(rectangle) if rectangle is not None else None,
        "ignored_small_rectangles": ignored,
    }


def interpretation(region_status: str, changed_edges: int) -> tuple[str, int]:
    if region_status != "suitable":
        return "serious_problem_by_region_choice", 0
    if changed_edges == 4:
        return "serious_problem_four_edge_reframe", 0
    if changed_edges == 3:
        return "substantial_adjustment_three_edges", 0
    if changed_edges in (1, 2):
        return "minor_tuning_original_usable", 1
    return "accepted_original", 1


def numeric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [float(row["auto_human_iou"]) for row in rows]
    decided_usable = [
        int(row["original_box_usable"])
        for row in rows
        if str(row["original_box_usable"]) != ""
    ]
    return {
        "count": len(rows),
        "mean_iou": mean(ious),
        "median_iou": median(ious),
        "iou_ge_0_3": sum(value >= 0.3 for value in ious) / len(ious),
        "iou_ge_0_5": sum(value >= 0.5 for value in ious) / len(ious),
        "iou_ge_0_7": sum(value >= 0.7 for value in ious) / len(ious),
        "original_usable": (
            sum(decided_usable) / len(decided_usable) if decided_usable else None
        ),
        "quality_decisions": len(decided_usable),
        "fallback_count": sum(int(row["original_fallback"]) for row in rows),
    }


def main() -> None:
    args = parse_args()
    tasks = json.loads(args.export.resolve().read_text(encoding="utf-8-sig"))
    if not isinstance(tasks, list) or len(tasks) != 186:
        raise ValueError("ROI216 export must contain all 186 newly annotated tasks")
    private = {
        row["task_key"]: row for row in read_csv(args.private_index.resolve())
    }
    selection = {
        row["sample_id"]: row for row in read_csv(args.selection.resolve())
    }
    manifest = {row["sample_id"]: row for row in read_csv(args.manifest.resolve())}
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        task_key = str(task.get("data", {}).get("task_key", ""))
        private_row = private[task_key]
        sample_id = private_row["sample_id"]
        selected = selection[sample_id]
        parsed = extract_annotation(task)
        original_source = "private_original_correction"
        if private_row["auto_bbox_raw_private"]:
            original = tuple(
                float(value)
                for value in json.loads(private_row["auto_bbox_raw_private"])
            )
            original_fallback = int(private_row["auto_fallback_private"])
        else:
            original_record, _ = analyse_trial(manifest[sample_id], 320, 240)
            original = map_box([float(value) for value in original_record["bbox"]])
            original_fallback = int(bool(original_record["fallback"]))
            original_source = "post_annotation_recomputed_blind"
        human = parsed["box"]
        if human is None:
            if (
                private_row["annotation_mode"] == "correction"
                and parsed["region_status"] == "suitable"
            ):
                human = original
                human_box_source = "accepted_original_without_edit"
            else:
                raise ValueError(f"{sample_id}: missing final human ROI")
        else:
            human_box_source = "human_final"
        edge_shifts = [abs(human[index] - original[index]) for index in range(4)]
        thresholds = [
            RAW_WIDTH * EDGE_CHANGE_RATIO,
            RAW_HEIGHT * EDGE_CHANGE_RATIO,
            RAW_WIDTH * EDGE_CHANGE_RATIO,
            RAW_HEIGHT * EDGE_CHANGE_RATIO,
        ]
        edge_changed = [
            int(shift >= threshold)
            for shift, threshold in zip(edge_shifts, thresholds, strict=True)
        ]
        if private_row["annotation_mode"] == "blind":
            quality = "pending_machine_box_review"
            usable: int | str = ""
        else:
            quality, usable = interpretation(
                parsed["region_status"],
                sum(edge_changed),
            )
        row: dict[str, Any] = {
            "task_key": task_key,
            "sample_id": sample_id,
            "selection_index": int(selected["selection_index"]),
            "annotation_mode": private_row["annotation_mode"],
            "fold": int(selected["fold"]),
            "class_id": int(selected["class_id"]),
            "class_name": selected["class_name"],
            "user_id": selected["user_id"],
            "trial_id": selected["trial_id"],
            "region_status": parsed["region_status"],
            "human_box_source": human_box_source,
            "original_box_source": original_source,
            "original_fallback": original_fallback,
            "human_x0": human[0],
            "human_y0": human[1],
            "human_x1": human[2],
            "human_y1": human[3],
            "original_x0": original[0],
            "original_y0": original[1],
            "original_x1": original[2],
            "original_y1": original[3],
            "auto_human_iou": box_iou(original, human),
            "auto_area_ratio": box_area(original) / (RAW_WIDTH * RAW_HEIGHT),
            "human_area_ratio": box_area(human) / (RAW_WIDTH * RAW_HEIGHT),
            "x0_shift": edge_shifts[0],
            "y0_shift": edge_shifts[1],
            "x1_shift": edge_shifts[2],
            "y1_shift": edge_shifts[3],
            "x0_changed": edge_changed[0],
            "y0_changed": edge_changed[1],
            "x1_changed": edge_changed[2],
            "y1_changed": edge_changed[3],
            "changed_edge_count": sum(edge_changed),
            "original_quality_interpretation": quality,
            "original_box_usable": usable,
            "ignored_small_rectangles": parsed["ignored_small_rectangles"],
            "mirror_reflection_note": int(bool(parsed["note"])),
            "free_note": parsed["note"],
        }
        rows.append(row)
        if index % 10 == 0 or index == len(tasks):
            print(f"ROI annotation audit {index}/{len(tasks)}", flush=True)
    rows.sort(key=lambda row: int(row["selection_index"]))
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    blind = [row for row in rows if row["annotation_mode"] == "blind"]
    correction = [row for row in rows if row["annotation_mode"] == "correction"]
    summary = {
        "contract": {
            "edge_changed_threshold": "2% of the corresponding image dimension",
            "non_suitable": "original box has a serious problem",
            "suitable_four_changed_edges": "original box has a serious problem",
            "suitable_one_or_two_changed_edges": (
                "original box is usable; human fine-tuning is preferred"
            ),
            "second_choice": "stored but excluded from the quality decision",
            "non_empty_note": "mirror/reflection issue",
            "blind_geometry": (
                "IoU/edge shifts are descriptive only; the machine box was hidden "
                "and requires a second review before any quality decision"
            ),
        },
        "tasks": len(rows),
        "blind": numeric_summary(blind),
        "correction": numeric_summary(correction),
        "region_status": dict(Counter(row["region_status"] for row in rows)),
        "quality_interpretation": dict(
            Counter(row["original_quality_interpretation"] for row in rows)
        ),
        "changed_edge_count": dict(
            sorted(Counter(int(row["changed_edge_count"]) for row in rows).items())
        ),
        "mirror_reflection_notes": sum(
            int(row["mirror_reflection_note"]) for row in rows
        ),
        "ignored_small_rectangles": sum(
            int(row["ignored_small_rectangles"]) for row in rows
        ),
    }
    summary_path = args.output.resolve().with_name("roi_annotation_audit_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
