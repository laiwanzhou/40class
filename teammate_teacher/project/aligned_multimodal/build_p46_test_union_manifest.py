from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "data/six_modality_audit/test_union_manifest.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data/p46_test_union_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a three-part cache-safe proxy ID for anonymous official Test trials."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 405:
        raise RuntimeError(f"Official Test count changed: {len(rows)}")
    output_rows = []
    seen = set()
    unreadable_ir_trials = []
    for row in rows:
        official_sample_id = row["sample_id"]
        if official_sample_id in seen or "/" in official_sample_id:
            raise RuntimeError(f"Unsafe or duplicate official sample ID: {official_sample_id}")
        seen.add(official_sample_id)
        enriched = dict(row)
        enriched["official_sample_id"] = official_sample_id
        enriched["sample_id"] = f"test/anonymous/{official_sample_id}"
        # This is a cache path placeholder only.  It must never be used as a
        # genuine subject identity for P46 subject-batch calibration.
        enriched["user_id"] = "anonymous"
        enriched["trial_id"] = official_sample_id
        ir_directory = Path(row["ir_path"])
        ir_files = sorted(ir_directory.glob("*.png"))
        unreadable = [path for path in ir_files if cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) is None]
        p46_ir_readable = bool(ir_files) and not unreadable
        enriched["p46_ir_readable"] = "1" if p46_ir_readable else "0"
        if not p46_ir_readable:
            # The general six-modality manifest treats a non-empty file as
            # usable.  P46 needs decodable pixels, so exclude this trial from
            # the visual cache while retaining it in the 405-row master list.
            enriched["ir_usable"] = "0"
            unreadable_ir_trials.append(official_sample_id)
        output_rows.append(enriched)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(output)
    print(
        f"Wrote {len(output_rows)} cache-safe Test rows: {output}; "
        f"P46-unreadable IR trials={unreadable_ir_trials}"
    )


if __name__ == "__main__":
    main()
