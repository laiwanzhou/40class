from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from import_label_studio_roi216 import standardize_box


PROJECT_DIR = Path(__file__).resolve().parent
ROI_DIR = PROJECT_DIR / "data" / "local_roi_annotation_v2"
DEFAULT_BASE = ROI_DIR / "roi_annotations.csv"
DEFAULT_REVIEW = ROI_DIR / "blind60_review_annotations.csv"
DEFAULT_OUTPUT = ROI_DIR / "roi_annotations_final.csv"
RAW_WIDTH = 640
RAW_HEIGHT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the ROI60 second review into the 216 annotations and rebuild "
            "fold-pure locator manifests."
        )
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context", type=float, default=0.15)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base = read_csv(args.base.resolve())
    review = {row["sample_id"]: row for row in read_csv(args.review.resolve())}
    if len(base) != 216 or len(review) != 60:
        raise ValueError("Expected 216 base annotations and 60 reviewed blind tasks")
    output: list[dict[str, object]] = []
    updated: set[str] = set()
    source_names = {
        "keep_machine": "blind_review_kept_machine",
        "use_first_blind_human": "blind_review_first_human",
        "use_current_adjustment": "blind_review_current_adjustment",
    }
    for row in base:
        result: dict[str, object] = dict(row)
        result["machine_box_assessment"] = ""
        result["blind_review_final_source"] = ""
        if row["annotation_mode"] == "blind":
            reviewed = review[row["sample_id"]]
            updated.add(row["sample_id"])
            box = tuple(
                float(reviewed[field]) for field in ("x0", "y0", "x1", "y1")
            )
            standardized = standardize_box(
                box,
                RAW_WIDTH,
                RAW_HEIGHT,
                args.context,
            )
            result.update(
                {
                    "bbox_source": source_names[reviewed["final_box_source"]],
                    "x0": box[0],
                    "y0": box[1],
                    "x1": box[2],
                    "y1": box[3],
                    "standard_x0": standardized[0],
                    "standard_y0": standardized[1],
                    "standard_x1": standardized[2],
                    "standard_y1": standardized[3],
                    "machine_box_assessment": reviewed[
                        "machine_box_assessment"
                    ],
                    "blind_review_final_source": reviewed["final_box_source"],
                    "free_note": reviewed["free_note"],
                    "annotation_id": reviewed["annotation_id"],
                    "lead_time": reviewed["lead_time"],
                    "created_at": reviewed["created_at"],
                    "updated_at": reviewed["updated_at"],
                }
            )
        output.append(result)
    if updated != set(review):
        raise ValueError("Reviewed blind sample IDs do not match the base annotations")
    if len(output) != 216 or len({row["sample_id"] for row in output}) != 216:
        raise AssertionError("Final ROI table must contain 216 unique samples")
    write_csv(args.output.resolve(), output)

    fold_dir = args.output.resolve().parent / "fold_pure_manifests_final"
    for held_fold in range(3):
        train = [
            row
            for row in output
            if int(row["training_eligible"]) == 1 and int(row["fold"]) != held_fold
        ]
        reviewed_eval = [
            row
            for row in output
            if row["annotation_mode"] == "blind" and int(row["fold"]) == held_fold
        ]
        if len(train) != 104 or len(reviewed_eval) != 20:
            raise AssertionError(
                f"Fold {held_fold}: train={len(train)} eval={len(reviewed_eval)}"
            )
        write_csv(fold_dir / f"fold_{held_fold}_locator_train.csv", train)
        write_csv(fold_dir / f"fold_{held_fold}_reviewed_eval.csv", reviewed_eval)

    review_rows = list(review.values())
    summary = {
        "annotations": len(output),
        "correction_locator_training": sum(
            int(row["training_eligible"]) for row in output
        ),
        "reviewed_eval_only": sum(
            row["annotation_mode"] == "blind" for row in output
        ),
        "machine_box_assessment": dict(
            Counter(row["machine_box_assessment"] for row in review_rows)
        ),
        "final_box_source": dict(
            Counter(row["final_box_source"] for row in review_rows)
        ),
        "fallback": {
            "count": sum(int(row["original_fallback"]) for row in review_rows),
            "bad": sum(
                int(row["original_fallback"])
                and row["machine_box_assessment"] == "machine_bad"
                for row in review_rows
            ),
        },
        "non_fallback": {
            "count": sum(not int(row["original_fallback"]) for row in review_rows),
            "bad": sum(
                not int(row["original_fallback"])
                and row["machine_box_assessment"] == "machine_bad"
                for row in review_rows
            ),
        },
        "fold_contract": {
            str(fold): {
                "locator_train": 104,
                "reviewed_eval": 20,
            }
            for fold in range(3)
        },
    }
    args.output.resolve().with_name("roi_annotations_final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
