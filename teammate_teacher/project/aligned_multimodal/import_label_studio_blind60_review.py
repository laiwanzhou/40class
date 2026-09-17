from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from import_label_studio_roi216 import (
    RAW_HEIGHT,
    RAW_WIDTH,
    select_final_rectangle,
    strip_explanation,
)


PROJECT_DIR = Path(__file__).resolve().parent
REVIEW_DIR = (
    PROJECT_DIR
    / "data"
    / "local_roi_annotation_v2"
    / "label_studio_blind60_review_v2"
)
DEFAULT_PRIVATE_INDEX = REVIEW_DIR / "task_index_private.csv"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "data"
    / "local_roi_annotation_v2"
    / "blind60_review_annotations.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the second-pass machine-box review for the 60 blind tasks."
    )
    parser.add_argument("export_json", type=Path)
    parser.add_argument("--private-index", type=Path, default=DEFAULT_PRIVATE_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_box(rectangle: dict[str, Any]) -> tuple[float, float, float, float]:
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


def stored_box(row: dict[str, str], field: str) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in json.loads(row[field]))
    if len(values) != 4:
        raise ValueError(f"{row['sample_id']}: invalid stored box in {field}")
    return values


def main() -> None:
    args = parse_args()
    private = {
        row["task_key"]: row for row in read_csv(args.private_index.resolve())
    }
    tasks = json.loads(args.export_json.resolve().read_text(encoding="utf-8-sig"))
    if not isinstance(tasks, list) or len(tasks) != 60:
        raise ValueError("ROI60 review export must contain exactly 60 tasks")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in tasks:
        task_key = str(task.get("data", {}).get("task_key", ""))
        if task_key not in private or task_key in seen:
            raise ValueError(f"Unexpected or duplicate task_key={task_key}")
        seen.add(task_key)
        annotations = [
            annotation
            for annotation in task.get("annotations", [])
            if not annotation.get("was_cancelled", False)
        ]
        if not annotations:
            raise ValueError(f"{task_key}: missing completed annotation")
        annotation = annotations[-1]
        choices: dict[str, str] = {}
        notes: list[str] = []
        rectangles: list[dict[str, Any]] = []
        for result in annotation.get("result", []):
            result_type = result.get("type")
            if result_type == "choices":
                values = result.get("value", {}).get("choices", [])
                if len(values) != 1:
                    raise ValueError(f"{task_key}: choice must contain one value")
                choices[str(result["from_name"])] = strip_explanation(str(values[0]))
            elif result_type == "textarea":
                notes.extend(
                    str(value).strip()
                    for value in result.get("value", {}).get("text", [])
                    if str(value).strip()
                )
            elif (
                result_type == "rectanglelabels"
                and result.get("from_name") == "roi_box"
            ):
                rectangles.append(result)
        assessment = choices.get("machine_box_assessment", "")
        final_source = choices.get("final_box_source", "")
        if assessment not in {
            "machine_good",
            "machine_usable_human_better",
            "machine_bad",
        }:
            raise ValueError(f"{task_key}: invalid machine_box_assessment")
        if final_source not in {
            "keep_machine",
            "use_first_blind_human",
            "use_current_adjustment",
        }:
            raise ValueError(f"{task_key}: invalid final_box_source")
        private_row = private[task_key]
        machine = stored_box(private_row, "machine_bbox_raw_private")
        first_human = stored_box(
            private_row,
            "first_blind_human_bbox_raw_private",
        )
        rectangle, ignored = select_final_rectangle(
            rectangles,
            private_row["sample_id"],
        )
        current = parse_box(rectangle) if rectangle is not None else None
        if final_source == "keep_machine":
            final = machine
        elif final_source == "use_first_blind_human":
            final = first_human
        else:
            if current is None:
                raise ValueError(f"{task_key}: current adjustment has no ROI box")
            final = current
        rows.append(
            {
                "review_index": int(private_row["review_index"]),
                "task_key": task_key,
                "sample_id": private_row["sample_id"],
                "fold": int(private_row["fold"]),
                "class_id": int(private_row["class_id_private"]),
                "class_name": private_row["class_name_private"],
                "user_id": private_row["user_id"],
                "trial_id": private_row["trial_id"],
                "original_fallback": int(private_row["original_fallback"]),
                "machine_box_assessment": assessment,
                "final_box_source": final_source,
                "x0": final[0],
                "y0": final[1],
                "x1": final[2],
                "y1": final[3],
                "ignored_small_rectangles": ignored,
                "free_note": " | ".join(notes),
                "annotation_id": annotation.get("id", ""),
                "lead_time": annotation.get("lead_time", ""),
                "created_at": annotation.get("created_at", ""),
                "updated_at": annotation.get("updated_at", ""),
            }
        )
    if seen != set(private):
        raise ValueError("ROI60 review export is incomplete")
    rows.sort(key=lambda row: int(row["review_index"]))
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "annotations": len(rows),
        "machine_box_assessment": {
            value: sum(row["machine_box_assessment"] == value for row in rows)
            for value in (
                "machine_good",
                "machine_usable_human_better",
                "machine_bad",
            )
        },
        "final_box_source": {
            value: sum(row["final_box_source"] == value for row in rows)
            for value in (
                "keep_machine",
                "use_first_blind_human",
                "use_current_adjustment",
            )
        },
        "notes": sum(bool(row["free_note"]) for row in rows),
    }
    args.output.resolve().with_name("blind60_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
