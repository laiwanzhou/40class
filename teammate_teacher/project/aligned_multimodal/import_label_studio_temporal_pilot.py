from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PILOT_DIR = (
    PROJECT_DIR
    / "data"
    / "local_action_audit_v1"
    / "label_studio_temporal_pilot30_v2"
)
OUTPUT_DIR = PROJECT_DIR / "data" / "local_action_audit_v1"
ALLOWED = {
    "bbox_quality": {
        "suitable",
        "missing_key_region",
        "over_wide_or_full_frame",
        "wrong_region_or_subject",
    },
    "local_judgability": {"yes", "barely", "no"},
    "recommended_view": {"local", "global", "global_plus_local", "other_modality"},
}
KEYFRAME_WIDTH = 960.0
KEYFRAME_HEIGHT = 240.0
MIDDLE_LEFT = 320.0
MIDDLE_RIGHT = 640.0
MODEL_WIDTH = 192.0
MODEL_HEIGHT = 144.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a Label Studio temporal-pilot JSON export without touching v1 labels"
    )
    parser.add_argument("export_json", type=Path)
    parser.add_argument(
        "--annotations-output",
        type=Path,
        default=OUTPUT_DIR / "annotations_temporal_pilot30.csv",
    )
    parser.add_argument(
        "--boxes-output",
        type=Path,
        default=OUTPUT_DIR / "temporal_pilot30_manual_boxes.csv",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_choice(value: str) -> str:
    return value.split(" （", 1)[0].strip()


def parse_annotation(task: dict[str, object]) -> dict[str, object]:
    annotations = task.get("annotations") or task.get("completions") or []
    completed = [
        annotation
        for annotation in annotations
        if not bool(annotation.get("was_cancelled", False))
    ]
    if len(completed) != 1:
        raise ValueError(
            f"{task.get('data', {}).get('sample_id')}: expected one completed annotation, "
            f"found {len(completed)}"
        )
    annotation = completed[0]
    parsed: dict[str, object] = {
        "annotation_id": annotation.get("id", ""),
        "lead_time": annotation.get("lead_time", ""),
        "created_at": annotation.get("created_at", ""),
        "updated_at": annotation.get("updated_at", ""),
        "free_note": "",
        "manual_bbox": None,
    }
    for item in annotation.get("result", []):
        field = str(item.get("from_name", ""))
        value = item.get("value", {})
        if field in ALLOWED:
            choices = value.get("choices") or []
            if len(choices) != 1:
                raise ValueError(f"{field}: expected one choice")
            choice = canonical_choice(str(choices[0]))
            if choice not in ALLOWED[field]:
                raise ValueError(f"{field}: unexpected value {choice!r}")
            parsed[field] = choice
        elif field == "free_note":
            texts = value.get("text") or []
            parsed[field] = str(texts[0]).strip() if texts else ""
        elif field == "manual_bbox":
            if parsed["manual_bbox"] is not None:
                raise ValueError("Only one manual_bbox is allowed per task")
            parsed["manual_bbox"] = {
                "x": float(value["x"]),
                "y": float(value["y"]),
                "width": float(value["width"]),
                "height": float(value["height"]),
                "rotation": float(value.get("rotation", 0.0)),
            }
    missing = [field for field in ALLOWED if field not in parsed]
    if missing:
        raise ValueError(f"Missing required choices: {missing}")
    return parsed


def convert_manual_box(sample_id: str, box: dict[str, float]) -> dict[str, object]:
    left = box["x"] / 100.0 * KEYFRAME_WIDTH
    top = box["y"] / 100.0 * KEYFRAME_HEIGHT
    right = left + box["width"] / 100.0 * KEYFRAME_WIDTH
    bottom = top + box["height"] / 100.0 * KEYFRAME_HEIGHT
    clipped_left = max(left, MIDDLE_LEFT)
    clipped_top = max(top, 0.0)
    clipped_right = min(right, MIDDLE_RIGHT)
    clipped_bottom = min(bottom, KEYFRAME_HEIGHT)
    valid = clipped_right > clipped_left and clipped_bottom > clipped_top
    if not valid:
        raise ValueError(
            f"{sample_id}: manual bbox must overlap the middle keyframe panel"
        )
    clipped_area = (clipped_right - clipped_left) * (clipped_bottom - clipped_top)
    original_area = max(1e-9, (right - left) * (bottom - top))
    if clipped_area / original_area < 0.95:
        raise ValueError(
            f"{sample_id}: manual bbox crosses outside the middle keyframe panel"
        )
    return {
        "sample_id": sample_id,
        "keyframe_x_percent": box["x"],
        "keyframe_y_percent": box["y"],
        "keyframe_width_percent": box["width"],
        "keyframe_height_percent": box["height"],
        "local_x_px": clipped_left - MIDDLE_LEFT,
        "local_y_px": clipped_top,
        "local_width_px": clipped_right - clipped_left,
        "local_height_px": clipped_bottom - clipped_top,
        "panel_width_px": 320,
        "panel_height_px": 240,
        "model_x_px": (clipped_left - MIDDLE_LEFT) * MODEL_WIDTH / 320.0,
        "model_y_px": clipped_top * MODEL_HEIGHT / 240.0,
        "model_width_px": (clipped_right - clipped_left) * MODEL_WIDTH / 320.0,
        "model_height_px": (clipped_bottom - clipped_top) * MODEL_HEIGHT / 240.0,
        "model_frame_width_px": int(MODEL_WIDTH),
        "model_frame_height_px": int(MODEL_HEIGHT),
    }


def main() -> None:
    args = parse_args()
    selection = read_csv(PILOT_DIR / "selection.csv")
    selection_by_id = {row["sample_id"]: row for row in selection}
    export = json.loads(args.export_json.resolve().read_text(encoding="utf-8"))
    if not isinstance(export, list):
        raise ValueError("Label Studio JSON export must be a list")

    annotation_rows: list[dict[str, object]] = []
    box_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for task in export:
        sample_id = str(task.get("data", {}).get("sample_id", ""))
        if sample_id not in selection_by_id:
            continue
        if sample_id in seen:
            raise ValueError(f"Duplicate exported task: {sample_id}")
        seen.add(sample_id)
        parsed = parse_annotation(task)
        source = selection_by_id[sample_id]
        annotation_rows.append(
            {
                "pilot_index": int(source["pilot_index"]),
                "sample_id": sample_id,
                "class_id": int(source["class_id"]),
                "class_name": source["class_name"],
                "user_id": source["user_id"],
                "trial_id": source["trial_id"],
                "bbox_quality": parsed["bbox_quality"],
                "local_judgability": parsed["local_judgability"],
                "recommended_view": parsed["recommended_view"],
                "free_note": parsed["free_note"],
                "has_manual_bbox": int(parsed["manual_bbox"] is not None),
                "annotation_id": parsed["annotation_id"],
                "lead_time": parsed["lead_time"],
                "created_at": parsed["created_at"],
                "updated_at": parsed["updated_at"],
            }
        )
        if parsed["manual_bbox"] is not None:
            box_rows.append(convert_manual_box(sample_id, parsed["manual_bbox"]))

    if not annotation_rows:
        raise ValueError("No temporal-pilot annotations were found in the export")
    annotation_rows.sort(key=lambda row: int(row["pilot_index"]))
    args.annotations_output.parent.mkdir(parents=True, exist_ok=True)
    with args.annotations_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotation_rows[0]))
        writer.writeheader()
        writer.writerows(annotation_rows)

    box_fieldnames = [
        "sample_id",
        "keyframe_x_percent",
        "keyframe_y_percent",
        "keyframe_width_percent",
        "keyframe_height_percent",
        "local_x_px",
        "local_y_px",
        "local_width_px",
        "local_height_px",
        "panel_width_px",
        "panel_height_px",
        "model_x_px",
        "model_y_px",
        "model_width_px",
        "model_height_px",
        "model_frame_width_px",
        "model_frame_height_px",
    ]
    with args.boxes_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=box_fieldnames)
        writer.writeheader()
        writer.writerows(box_rows)
    print(
        json.dumps(
            {
                "imported_annotations": len(annotation_rows),
                "manual_boxes": len(box_rows),
                "annotations_output": str(args.annotations_output.resolve()),
                "boxes_output": str(args.boxes_output.resolve()),
                "v1_files_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
