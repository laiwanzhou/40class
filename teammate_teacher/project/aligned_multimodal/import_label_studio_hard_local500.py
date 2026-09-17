from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_roi_locator_models import box_iou


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX = (
    PROJECT_DIR
    / "data"
    / "hard_local_v1"
    / "label_studio_annotation500"
    / "task_index_private.csv"
)
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "hard_local_v1" / "annotation500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import completed Hard Local 500 Label Studio JSON"
    )
    parser.add_argument("--export-json", type=Path, required=True)
    parser.add_argument("--task-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a non-empty subset of the 500 tasks to be imported.",
    )
    parser.add_argument(
        "--output-csv-name",
        default="annotations500_final.csv",
    )
    parser.add_argument(
        "--summary-name",
        default="annotation500_audit_summary.json",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def choice_prefix(value: str) -> str:
    for separator in ("（", "("):
        if separator in value:
            return value.split(separator, 1)[0]
    return value


def result_by_name(
    annotation: dict[str, object],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in annotation["result"]:
        name = str(item["from_name"])
        if name in result:
            previous = result[name]
            if (
                item.get("type") == "rectanglelabels"
                and previous.get("type") == "rectanglelabels"
            ):
                # Label Studio can retain an accidental click-drag as a second,
                # tiny ROI. Keep the larger action-region candidate; duplicate
                # choices or text remain hard errors.
                previous_value = previous["value"]
                item_value = item["value"]
                previous_area = float(previous_value["width"]) * float(
                    previous_value["height"]
                )
                item_area = float(item_value["width"]) * float(
                    item_value["height"]
                )
                if item_area > previous_area:
                    result[name] = item
                continue
            raise ValueError(f"Duplicate non-rectangle result from_name={name}")
        result[name] = item
    return result


def rectangle_box(
    item: dict[str, object],
    width: int,
    height: int,
) -> np.ndarray:
    value = item["value"]
    x0 = float(value["x"]) * width / 100.0
    y0 = float(value["y"]) * height / 100.0
    box_width = float(value["width"]) * width / 100.0
    box_height = float(value["height"]) * height / 100.0
    x1 = x0 + box_width - 1.0
    y1 = y0 + box_height - 1.0
    box = np.asarray([x0, y0, x1, y1], dtype=np.float64)
    if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
        raise ValueError(f"Invalid rectangle {box.tolist()} for {width}x{height}")
    return box


def get_choice(results: dict[str, dict[str, object]], name: str) -> str:
    item = results.get(name)
    if item is None or item["type"] != "choices":
        raise ValueError(f"Missing required choice {name}")
    choices = item["value"].get("choices", [])
    if len(choices) != 1:
        raise ValueError(f"{name} must contain one choice")
    return choice_prefix(str(choices[0]))


def main() -> None:
    args = parse_args()
    index_rows = read_csv(args.task_index.resolve())
    if len(index_rows) != 500:
        raise ValueError(f"Expected 500 task-index rows, got {len(index_rows)}")
    index_by_key = {row["task_key"]: row for row in index_rows}
    exported = json.loads(args.export_json.resolve().read_text(encoding="utf-8"))
    if not isinstance(exported, list):
        raise ValueError("Label Studio export must be a JSON list")
    if not args.allow_partial and len(exported) != 500:
        raise ValueError(f"Expected a 500-task JSON export, got {len(exported)}")
    if args.allow_partial and not (1 <= len(exported) <= 500):
        raise ValueError(
            f"Partial export must contain 1..500 tasks, got {len(exported)}"
        )
    output_rows: list[dict[str, object]] = []
    for task in exported:
        task_key = str(task["data"]["task_key"])
        private = index_by_key.get(task_key)
        if private is None:
            raise KeyError(f"Unknown task_key {task_key}")
        completed = [
            annotation
            for annotation in task.get("annotations", [])
            if not annotation.get("was_cancelled", False)
        ]
        if not completed and args.allow_partial:
            continue
        if len(completed) != 1:
            raise ValueError(
                f"{task_key}: expected one completed annotation, got {len(completed)}"
            )
        annotation = completed[0]
        results = result_by_name(annotation)
        if "depth_roi_box" not in results or "thermal_roi_box" not in results:
            raise ValueError(f"{task_key}: Depth/Thermal rectangle missing")
        depth_width = int(private["depth_width"])
        depth_height = int(private["depth_height"])
        thermal_width = int(private["thermal_width"])
        thermal_height = int(private["thermal_height"])
        depth_box = rectangle_box(
            results["depth_roi_box"], depth_width, depth_height
        )
        thermal_box = rectangle_box(
            results["thermal_roi_box"], thermal_width, thermal_height
        )
        machine_depth = np.asarray(
            json.loads(private["depth_machine_bbox"]), dtype=np.float64
        )
        mapped_thermal = np.asarray(
            json.loads(private["thermal_mapped_bbox"]), dtype=np.float64
        )
        depth_assessment = get_choice(results, "depth_box_assessment")
        thermal_assessment = get_choice(results, "thermal_mapping_assessment")
        temporal_valid = get_choice(results, "single_box_temporally_valid")
        free_note = ""
        if "free_note" in results:
            texts = results["free_note"]["value"].get("text", [])
            if texts:
                free_note = str(texts[0]).strip()
        thermal_manual = thermal_assessment == "needs_manual_adjustment"
        output_rows.append(
            {
                "annotation_index": int(private["annotation_index"]),
                "task_key": task_key,
                "sample_id": private["sample_id"],
                "fold": int(private["fold"]),
                "class_id": int(private["class_id"]),
                "class_name": private["class_name"],
                "user_id": private["user_id"],
                "trial_id": private["trial_id"],
                "selection_category": private["selection_category"],
                "selection_reasons": private["selection_reasons"],
                "fallback": int(private["motion_fallback"]),
                "locator_v1_machine_source": private[
                    "locator_v1_machine_source"
                ],
                "locator_v1_uncertainty": float(
                    private["locator_v1_uncertainty"]
                ),
                "depth_box_assessment": depth_assessment,
                "depth_machine_final_iou": box_iou(machine_depth, depth_box),
                "depth_width": depth_width,
                "depth_height": depth_height,
                "depth_x0": float(depth_box[0]),
                "depth_y0": float(depth_box[1]),
                "depth_x1": float(depth_box[2]),
                "depth_y1": float(depth_box[3]),
                "thermal_mapping_assessment": thermal_assessment,
                "thermal_width": thermal_width,
                "thermal_height": thermal_height,
                "thermal_mapped_final_iou": box_iou(
                    mapped_thermal, thermal_box
                ),
                "thermal_final_x0": float(thermal_box[0]),
                "thermal_final_y0": float(thermal_box[1]),
                "thermal_final_x1": float(thermal_box[2]),
                "thermal_final_y1": float(thermal_box[3]),
                "thermal_manual_bbox_present": int(thermal_manual),
                "thermal_manual_x0": (
                    float(thermal_box[0]) if thermal_manual else ""
                ),
                "thermal_manual_y0": (
                    float(thermal_box[1]) if thermal_manual else ""
                ),
                "thermal_manual_x1": (
                    float(thermal_box[2]) if thermal_manual else ""
                ),
                "thermal_manual_y1": (
                    float(thermal_box[3]) if thermal_manual else ""
                ),
                "single_box_temporally_valid": temporal_valid,
                "free_note": free_note,
                "annotation_id": int(annotation["id"]),
                "lead_time": float(annotation.get("lead_time") or 0.0),
                "created_at": annotation.get("created_at", ""),
                "updated_at": annotation.get("updated_at", ""),
            }
        )
    output_rows.sort(key=lambda row: int(row["annotation_index"]))
    if not output_rows:
        raise ValueError("No completed annotations found in the export")
    if not args.allow_partial and len(output_rows) != 500:
        raise ValueError(f"Expected 500 completed annotations, got {len(output_rows)}")
    if len({row["sample_id"] for row in output_rows}) != len(output_rows):
        raise ValueError("Imported output contains duplicate samples")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.output_csv_name
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    depth_counts = Counter(str(row["depth_box_assessment"]) for row in output_rows)
    thermal_counts = Counter(
        str(row["thermal_mapping_assessment"]) for row in output_rows
    )
    fallback_rows = [row for row in output_rows if int(row["fallback"]) == 1]
    denominator = len(output_rows)
    report = {
        "annotations": len(output_rows),
        "expected_full_annotations": 500,
        "is_complete_500": len(output_rows) == 500,
        "partial_import_allowed": bool(args.allow_partial),
        "source_json": str(args.export_json.resolve()),
        "output_csv": str(csv_path),
        "depth": {
            "counts": dict(depth_counts),
            "direct_accept_rate": depth_counts["direct_accept"] / denominator,
            "minor_adjustment_rate": (
                depth_counts["minor_adjustment"] / denominator
            ),
            "severe_error_rate": (
                depth_counts["severe_error_redraw"] / denominator
            ),
            "fallback_samples": len(fallback_rows),
            "fallback_severe_error_rate": (
                sum(
                    row["depth_box_assessment"] == "severe_error_redraw"
                    for row in fallback_rows
                )
                / max(1, len(fallback_rows))
            ),
        },
        "thermal": {
            "counts": dict(thermal_counts),
            "direct_mapping_usable_rate": (
                thermal_counts["direct_usable"] / denominator
            ),
            "padding_usable_rate": (
                thermal_counts["usable_with_padding"] / denominator
            ),
            "manual_adjustment_rate": (
                thermal_counts["needs_manual_adjustment"] / denominator
            ),
            "unusable_rate": thermal_counts["unusable"] / denominator,
        },
        "single_box_temporally_invalid": int(
            sum(
                row["single_box_temporally_valid"] == "invalid"
                for row in output_rows
            )
        ),
        "notes": int(sum(bool(str(row["free_note"]).strip()) for row in output_rows)),
    }
    (output_dir / args.summary_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
