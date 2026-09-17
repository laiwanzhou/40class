from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
AUDIT_DIR = PROJECT_DIR / "data" / "local_action_audit_v1"
DEFAULT_OUTPUT = AUDIT_DIR / "label_studio_pilot30"
SMALL_ACTION_IDS = (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39)
BASE_FALLBACK_CLASS_IDS = {6, 7, 9, 17, 18, 23, 24, 27}
EXTRA_HARD_CLASS_IDS = (8, 10, 18, 19, 21, 22, 25, 26, 39)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable 30-task Label Studio local-view pilot")
    parser.add_argument("--annotations", type=Path, default=AUDIT_DIR / "annotations.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_one(
    candidates: list[dict[str, str]],
    prefer_fallback: bool,
) -> dict[str, str]:
    preferred = [
        row for row in candidates
        if bool(int(row["motion_fallback"])) == prefer_fallback
    ]
    pool = preferred or candidates
    return min(
        pool,
        key=lambda row: (
            int(row["oof_correct_hidden_during_annotation"]),
            float(row["oof_confidence_hidden_during_annotation"]),
            row["sample_id"],
        ),
    )


def select_rows(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], str]]:
    by_class: dict[int, list[dict[str, str]]] = {}
    for class_id in SMALL_ACTION_IDS:
        by_class[class_id] = [
            row for row in rows if int(row["class_id"]) == class_id
        ]

    selected: list[tuple[dict[str, str], str]] = []
    used: set[str] = set()
    for class_id in SMALL_ACTION_IDS:
        row = choose_one(
            by_class[class_id],
            prefer_fallback=class_id in BASE_FALLBACK_CLASS_IDS,
        )
        selected.append((row, "one_per_class"))
        used.add(row["sample_id"])

    for class_id in EXTRA_HARD_CLASS_IDS:
        remaining = [
            row for row in by_class[class_id]
            if row["sample_id"] not in used
        ]
        first = next(row for row, _ in selected if int(row["class_id"]) == class_id)
        row = choose_one(
            remaining,
            prefer_fallback=not bool(int(first["motion_fallback"])),
        )
        selected.append((row, "extra_hard_class_contrast"))
        used.add(row["sample_id"])

    if len(selected) != 30 or len(used) != 30:
        raise RuntimeError("Pilot selection must contain exactly 30 unique trials")
    return selected


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def main() -> None:
    args = parse_args()
    annotations_path = args.annotations.resolve()
    audit_dir = annotations_path.parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = select_rows(read_csv(annotations_path))

    tasks = []
    index_rows = []
    for pilot_index, (row, reason) in enumerate(selected, start=1):
        image_path = audit_dir / row["case_image"]
        tasks.append(
            {
                "data": {
                    "image": image_data_url(image_path),
                    "sample_id": row["sample_id"],
                    "sample_info": (
                        f"Pilot {pilot_index}/30｜真实类别：{row['class_name']}｜"
                        f"Subject：{row['user_id']}｜Trial：{row['trial_id']}"
                    ),
                }
            }
        )
        index_rows.append(
            {
                "pilot_index": pilot_index,
                "sample_id": row["sample_id"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "user_id": row["user_id"],
                "trial_id": row["trial_id"],
                "motion_fallback": row["motion_fallback"],
                "case_image": row["case_image"],
                "selection_reason": reason,
            }
        )

    (output / "tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output / "selection.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    summary = {
        "tasks": len(tasks),
        "classes": len(set(row["class_id"] for row in index_rows)),
        "fallback_tasks": sum(int(row["motion_fallback"]) for row in index_rows),
        "selection": (
            "risk-enriched workflow pilot: one task from every fixed small-action class, "
            "plus a contrasting task from nine predeclared hard classes"
        ),
        "warning": "This 30-task pilot is not an unbiased estimate of the full 168-task label distribution.",
        "images": "embedded as base64 data URLs; no local-files server setup is required",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
