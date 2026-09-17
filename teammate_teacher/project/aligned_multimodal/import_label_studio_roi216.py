from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "local_roi_annotation_v2"
DEFAULT_PROJECT_DIR = DEFAULT_DATA_DIR / "label_studio_roi216"
DEFAULT_SELECTION = DEFAULT_DATA_DIR / "selection.csv"
DEFAULT_PRIVATE_INDEX = DEFAULT_PROJECT_DIR / "task_index_private.csv"
DEFAULT_PRIOR = DEFAULT_PROJECT_DIR / "prior_pilot30_annotations.csv"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "roi_annotations.csv"
RAW_WIDTH = 640
RAW_HEIGHT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import the independent ROI216 Label Studio export, merge Pilot30, "
            "validate one-box semantics, and emit fold-pure locator manifests."
        )
    )
    parser.add_argument("export_json", type=Path)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--private-index", type=Path, default=DEFAULT_PRIVATE_INDEX)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context", type=float, default=0.15)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def strip_explanation(value: str) -> str:
    return value.split("（", 1)[0].split("(", 1)[0].strip()


def rectangle_area_percent(rectangle: dict[str, Any]) -> float:
    value = rectangle.get("value", {})
    return float(value["width"]) * float(value["height"])


def select_final_rectangle(
    rectangles: list[dict[str, Any]],
    sample_id: str,
) -> tuple[dict[str, Any] | None, int]:
    """Keep the intended ROI and reject ambiguous multi-box annotations.

    Two exported tasks contain a second box created by an accidental click. Its
    area is less than 1% of the intended box, so keeping the largest rectangle is
    unambiguous. If a future export contains two comparably sized boxes, stop
    instead of guessing which one is the final ROI.
    """
    if not rectangles:
        return None, 0
    ordered = sorted(rectangles, key=rectangle_area_percent, reverse=True)
    if len(ordered) > 1:
        largest = rectangle_area_percent(ordered[0])
        second = rectangle_area_percent(ordered[1])
        if largest <= 0 or second / largest >= 0.20:
            raise ValueError(
                f"{sample_id}: multiple comparable ROI boxes require manual review"
            )
    return ordered[0], len(ordered) - 1


def standardize_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    context: float,
    target_ratio: float = 4.0 / 3.0,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
        raise ValueError(f"Box outside {width}x{height}: {box}")
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    box_width = (x1 - x0 + 1) * (1.0 + 2.0 * context)
    box_height = (y1 - y0 + 1) * (1.0 + 2.0 * context)
    if box_width / box_height < target_ratio:
        box_width = box_height * target_ratio
    else:
        box_height = box_width / target_ratio
    box_width = min(float(width), box_width)
    box_height = min(float(height), box_height)
    left = min(max(0.0, center_x - box_width / 2.0), width - box_width)
    top = min(max(0.0, center_y - box_height / 2.0), height - box_height)
    return left, top, left + box_width - 1.0, top + box_height - 1.0


def parse_annotation(
    task: dict[str, Any],
    private: dict[str, str],
) -> dict[str, Any]:
    annotations = [
        row
        for row in task.get("annotations", [])
        if not row.get("was_cancelled", False)
    ]
    if not annotations:
        raise ValueError(f"{private['sample_id']}: no completed annotation")
    annotation = annotations[-1]
    results = annotation.get("result", [])
    choices: dict[str, str] = {}
    notes: list[str] = []
    rectangles: list[dict[str, Any]] = []
    for result in results:
        from_name = result.get("from_name")
        value = result.get("value", {})
        if result.get("type") == "choices":
            values = value.get("choices", [])
            if len(values) != 1:
                raise ValueError(
                    f"{private['sample_id']}: {from_name} must contain one choice"
                )
            choices[str(from_name)] = strip_explanation(str(values[0]))
        elif result.get("type") == "textarea" and from_name == "free_note":
            notes.extend(str(item) for item in value.get("text", []) if str(item).strip())
        elif result.get("type") == "rectanglelabels" and from_name == "roi_box":
            rectangles.append(result)

    region_status = choices.get("region_status")
    temporal = choices.get("single_box_temporally_valid")
    valid_region = {"suitable", "missing_key_region", "wrong_region"}
    valid_temporal = {"stable", "needs_wider_context", "invalid"}
    if region_status not in valid_region:
        raise ValueError(
            f"{private['sample_id']}: invalid/missing region_status={region_status}"
        )
    if temporal not in valid_temporal:
        raise ValueError(
            f"{private['sample_id']}: invalid/missing temporal status={temporal}"
        )

    bbox_source = ""
    box: tuple[float, float, float, float] | None = None
    rectangle, ignored_small_rectangles = select_final_rectangle(
        rectangles,
        private["sample_id"],
    )
    if rectangle is not None:
        value = rectangle["value"]
        x0 = RAW_WIDTH * float(value["x"]) / 100.0
        y0 = RAW_HEIGHT * float(value["y"]) / 100.0
        box_width = RAW_WIDTH * float(value["width"]) / 100.0
        box_height = RAW_HEIGHT * float(value["height"]) / 100.0
        box = (
            x0,
            y0,
            min(RAW_WIDTH - 1.0, x0 + box_width - 1.0),
            min(RAW_HEIGHT - 1.0, y0 + box_height - 1.0),
        )
        bbox_source = "human_drawn"
    elif (
        private["annotation_mode"] == "correction"
        and region_status == "suitable"
        and private["auto_bbox_raw_private"]
    ):
        values = json.loads(private["auto_bbox_raw_private"])
        box = tuple(float(value) for value in values)
        bbox_source = "human_accepted_auto"
    elif temporal != "invalid":
        raise ValueError(
            f"{private['sample_id']}: a final box is required unless temporal=invalid"
        )

    return {
        "region_status": region_status,
        "single_box_temporally_valid": temporal,
        "free_note": " | ".join(notes),
        "bbox_source": bbox_source,
        "box": box,
        "ignored_small_rectangles": ignored_small_rectangles,
        "annotation_id": annotation.get("id", ""),
        "lead_time": annotation.get("lead_time", ""),
        "created_at": annotation.get("created_at", ""),
        "updated_at": annotation.get("updated_at", ""),
    }


def output_row(
    selection: dict[str, str],
    parsed: dict[str, Any],
    context: float,
) -> dict[str, Any]:
    box = parsed["box"]
    standardized = (
        standardize_box(box, RAW_WIDTH, RAW_HEIGHT, context)
        if box is not None
        else (math.nan, math.nan, math.nan, math.nan)
    )
    raw_values = box if box is not None else (math.nan, math.nan, math.nan, math.nan)
    training_eligible = (
        selection["annotation_mode"] == "correction" and box is not None
    )
    return {
        "sample_id": selection["sample_id"],
        "fold": int(selection["fold"]),
        "class_id": int(selection["class_id"]),
        "class_name": selection["class_name"],
        "user_id": selection["user_id"],
        "trial_id": selection["trial_id"],
        "annotation_mode": selection["annotation_mode"],
        "training_eligible": int(training_eligible),
        "bbox_source": parsed["bbox_source"],
        "ignored_small_rectangles": parsed.get("ignored_small_rectangles", 0),
        "region_status": parsed["region_status"],
        "single_box_temporally_valid": parsed["single_box_temporally_valid"],
        "raw_width": RAW_WIDTH,
        "raw_height": RAW_HEIGHT,
        "x0": raw_values[0],
        "y0": raw_values[1],
        "x1": raw_values[2],
        "y1": raw_values[3],
        "standard_x0": standardized[0],
        "standard_y0": standardized[1],
        "standard_x1": standardized[2],
        "standard_y1": standardized[3],
        "standard_context": context,
        "free_note": parsed["free_note"],
        "annotation_id": parsed.get("annotation_id", ""),
        "lead_time": parsed.get("lead_time", ""),
        "created_at": parsed.get("created_at", ""),
        "updated_at": parsed.get("updated_at", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.context <= 0.5:
        raise ValueError("--context must be in [0, 0.5]")
    selection_rows = read_csv(args.selection.resolve())
    selection = {row["sample_id"]: row for row in selection_rows}
    private_rows = read_csv(args.private_index.resolve())
    private = {row["task_key"]: row for row in private_rows}
    prior_rows = read_csv(args.prior.resolve())

    tasks = json.loads(args.export_json.resolve().read_text(encoding="utf-8-sig"))
    if not isinstance(tasks, list):
        raise ValueError("Label Studio export must be a JSON list")
    exported: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_key = str(task.get("data", {}).get("task_key", ""))
        if not task_key:
            raise ValueError("Exported task is missing data.task_key")
        if task_key in exported:
            raise ValueError(f"Duplicate exported task: {task_key}")
        exported[task_key] = task
    if set(exported) != set(private):
        missing = sorted(set(private) - set(exported))
        extra = sorted(set(exported) - set(private))
        raise ValueError(
            f"Export must contain all 186 new tasks: missing={missing[:3]}, extra={extra[:3]}"
        )

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for task_key in sorted(exported):
        try:
            private_row = private[task_key]
            sample_id = private_row["sample_id"]
            parsed = parse_annotation(exported[task_key], private_row)
            rows.append(output_row(selection[sample_id], parsed, args.context))
        except ValueError as error:
            errors.append(str(error))
    if errors:
        report = args.output.resolve().with_name("roi_import_errors.txt")
        report.write_text("\n".join(errors) + "\n", encoding="utf-8")
        raise ValueError(
            f"{len(errors)} annotation errors; see {report}. First: {errors[0]}"
        )

    for prior in prior_rows:
        prior_box = (
            float(prior["x0"]),
            float(prior["y0"]),
            float(prior["x1"]),
            float(prior["y1"]),
        )
        parsed = {
            "box": prior_box,
            "bbox_source": prior["bbox_source"],
            "ignored_small_rectangles": 0,
            "region_status": prior["region_status"],
            "single_box_temporally_valid": (
                prior["single_box_temporally_valid"] or "unknown_prior_pilot30"
            ),
            "free_note": prior["free_note"],
        }
        rows.append(output_row(selection[prior["sample_id"]], parsed, args.context))
    rows.sort(key=lambda row: int(selection[row["sample_id"]]["selection_index"]))
    if len(rows) != 216 or len({row["sample_id"] for row in rows}) != 216:
        raise AssertionError("Merged ROI annotations must contain 216 unique samples")
    write_csv(args.output.resolve(), rows)

    fold_dir = args.output.resolve().parent / "fold_pure_manifests"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for held_fold in range(3):
        train = [
            row
            for row in rows
            if int(row["training_eligible"]) == 1 and int(row["fold"]) != held_fold
        ]
        blind_eval = [
            row
            for row in rows
            if row["annotation_mode"] == "blind" and int(row["fold"]) == held_fold
        ]
        write_csv(fold_dir / f"fold_{held_fold}_locator_train.csv", train)
        write_csv(fold_dir / f"fold_{held_fold}_blind_eval.csv", blind_eval)
        if any(int(row["fold"]) == held_fold for row in train):
            raise AssertionError(f"Fold leakage in fold_{held_fold}_locator_train.csv")
        if any(row["annotation_mode"] != "blind" for row in blind_eval):
            raise AssertionError(f"Non-blind row in fold_{held_fold}_blind_eval.csv")

    summary = {
        "annotations": len(rows),
        "blind_eval_only": sum(row["annotation_mode"] == "blind" for row in rows),
        "correction_training_eligible": sum(
            int(row["training_eligible"]) for row in rows
        ),
        "single_box_invalid": sum(
            row["single_box_temporally_valid"] == "invalid" for row in rows
        ),
        "bbox_sources": {
            source: sum(row["bbox_source"] == source for row in rows)
            for source in sorted(set(row["bbox_source"] for row in rows))
        },
        "fold_contract": {
            str(fold): {
                "locator_train": sum(
                    int(row["training_eligible"]) == 1 and int(row["fold"]) != fold
                    for row in rows
                ),
                "blind_eval": sum(
                    row["annotation_mode"] == "blind" and int(row["fold"]) == fold
                    for row in rows
                ),
            }
            for fold in range(3)
        },
    }
    args.output.resolve().with_name("roi_import_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
