from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
AUDIT_DIR = PROJECT_DIR / "data" / "local_action_audit_v1"
PILOT_DIR = AUDIT_DIR / "label_studio_pilot30"
CHOICE_FIELDS = {
    "localization_coverage": {
        "complete",
        "missing_object_or_context",
        "motion_only",
        "background_error",
    },
    "raw_observability": {"clear", "ambiguous", "not_visible"},
    "recommended_view": {"local", "global", "global_plus_local", "other_modality"},
}
MANUAL_BOX_FIELDS = [
    "sample_id",
    "label",
    "montage_x_pct",
    "montage_y_pct",
    "montage_width_pct",
    "montage_height_pct",
    "local_x_px",
    "local_y_px",
    "local_width_px",
    "local_height_px",
    "inside_middle_depth_panel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a Label Studio pilot export back into the audit CSV")
    parser.add_argument("export_json", type=Path)
    parser.add_argument("--source-csv", type=Path, default=AUDIT_DIR / "annotations.csv")
    parser.add_argument("--selection", type=Path, default=PILOT_DIR / "selection.csv")
    parser.add_argument("--output-csv", type=Path, default=AUDIT_DIR / "annotations_with_pilot30.csv")
    parser.add_argument(
        "--manual-boxes-csv",
        type=Path,
        default=AUDIT_DIR / "pilot30_manual_boxes.csv",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    annotations = [
        annotation
        for annotation in task.get("annotations", [])
        if not annotation.get("was_cancelled", False)
    ]
    if not annotations:
        return None
    return max(annotations, key=lambda annotation: int(annotation.get("id", 0)))


def english_token(label: str) -> str:
    return label.strip().split(maxsplit=1)[0]


def parse_result(annotation: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in annotation.get("result", []):
        field = item.get("from_name")
        value = item.get("value", {})
        if field in CHOICE_FIELDS:
            choices = value.get("choices", [])
            if not choices:
                continue
            token = english_token(str(choices[0]))
            if token not in CHOICE_FIELDS[field]:
                raise ValueError(f"Unexpected {field} choice: {choices[0]}")
            parsed[field] = token
        elif field == "free_note":
            text = value.get("text", [])
            parsed["reviewer_note"] = str(text[0]).strip() if text else ""
        elif field == "manual_bbox":
            labels = value.get("rectanglelabels", [])
            if not labels:
                continue
            if "manual_bbox" in parsed:
                raise ValueError("Only one manual_bbox is allowed per task")
            parsed["manual_bbox"] = {
                "label": str(labels[0]),
                "x_pct": float(value["x"]),
                "y_pct": float(value["y"]),
                "width_pct": float(value["width"]),
                "height_pct": float(value["height"]),
                "rotation": float(value.get("rotation", 0.0)),
                "original_width": int(item.get("original_width", 1280)),
                "original_height": int(item.get("original_height", 720)),
            }
    return parsed


def convert_manual_box(sample_id: str, box: dict[str, Any]) -> dict[str, Any]:
    montage_width = int(box["original_width"])
    montage_height = int(box["original_height"])
    x = float(box["x_pct"]) * montage_width / 100.0
    y = float(box["y_pct"]) * montage_height / 100.0
    width = float(box["width_pct"]) * montage_width / 100.0
    height = float(box["height_pct"]) * montage_height / 100.0
    panel_left, panel_top = 320.0, 0.0
    panel_right, panel_bottom = 640.0, 240.0
    inside = (
        x >= panel_left - 1.0
        and y >= panel_top - 1.0
        and x + width <= panel_right + 1.0
        and y + height <= panel_bottom + 1.0
    )
    clipped_left = min(max(x, panel_left), panel_right)
    clipped_top = min(max(y, panel_top), panel_bottom)
    clipped_right = min(max(x + width, panel_left), panel_right)
    clipped_bottom = min(max(y + height, panel_top), panel_bottom)
    return {
        "sample_id": sample_id,
        "label": box["label"],
        "montage_x_pct": box["x_pct"],
        "montage_y_pct": box["y_pct"],
        "montage_width_pct": box["width_pct"],
        "montage_height_pct": box["height_pct"],
        "local_x_px": clipped_left - panel_left,
        "local_y_px": clipped_top - panel_top,
        "local_width_px": max(0.0, clipped_right - clipped_left),
        "local_height_px": max(0.0, clipped_bottom - clipped_top),
        "inside_middle_depth_panel": int(inside),
    }


def main() -> None:
    args = parse_args()
    tasks = json.loads(args.export_json.resolve().read_text(encoding="utf-8-sig"))
    if not isinstance(tasks, list):
        raise TypeError("Label Studio export must be a JSON list")

    expected = {row["sample_id"] for row in read_csv(args.selection.resolve())}
    imported: dict[str, dict[str, Any]] = {}
    incomplete: list[str] = []
    for task in tasks:
        sample_id = str(task.get("data", {}).get("sample_id", ""))
        if sample_id not in expected:
            continue
        annotation = latest_annotation(task)
        if annotation is None:
            incomplete.append(sample_id)
            continue
        result = parse_result(annotation)
        missing = [field for field in CHOICE_FIELDS if field not in result]
        if "manual_bbox" not in result:
            missing.append("manual_bbox")
        if missing:
            incomplete.append(f"{sample_id}: missing {','.join(missing)}")
            continue
        imported[sample_id] = result

    rows = read_csv(args.source_csv.resolve())
    manual_boxes = [
        convert_manual_box(sample_id, result["manual_bbox"])
        for sample_id, result in sorted(imported.items())
    ]
    for row in rows:
        result = imported.get(row["sample_id"])
        if result is None:
            continue
        for field in CHOICE_FIELDS:
            row[field] = result[field]
        row["issue_flags"] = ""
        row["reviewer_note"] = result.get("reviewer_note", "")

    output = args.output_csv.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    boxes_output = args.manual_boxes_csv.resolve()
    boxes_output.parent.mkdir(parents=True, exist_ok=True)
    with boxes_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_BOX_FIELDS)
        writer.writeheader()
        writer.writerows(manual_boxes)
    print(
        json.dumps(
            {
                "expected_pilot_tasks": len(expected),
                "merged_annotations": len(imported),
                "incomplete": incomplete,
                "output_csv": str(output),
                "manual_boxes_csv": str(boxes_output),
                "boxes_inside_target_panel": sum(
                    int(row["inside_middle_depth_panel"]) for row in manual_boxes
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
