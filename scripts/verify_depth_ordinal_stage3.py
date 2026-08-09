from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ordinal_depth import (  # noqa: E402
    JET_LUT_BGR,
    load_depth_color_ordinal,
    mask_aware_resize_ordinal,
)


DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "reports/roi640_stage2_temporal_identity/source_depth_jet_integrity.csv"
)
DEFAULT_JSON = PROJECT_ROOT / "reports/depth_ordinal_stage3_verification.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports/depth_ordinal_stage3_verification.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify inverse-JET ordinal decoding on one real train frame per trial.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def selected_rows(path: Path) -> list[dict[str, str]]:
    first_by_sample: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            first_by_sample.setdefault(row["sample_id"], row)
    return list(first_by_sample.values())


def main() -> None:
    args = parse_args()
    rows = selected_rows(args.source)
    started = time.perf_counter()
    valid_pixels = 0
    black_pixels = 0
    roundtrip_mismatches = 0
    resize_invalid_nonzero = 0
    resize_valid_pixels = 0

    for row in rows:
        path = Path(row["path"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        decoded = load_depth_color_ordinal(path)
        reconstructed = JET_LUT_BGR[decoded.values]
        roundtrip_mismatches += int(
            np.count_nonzero(np.any(reconstructed[decoded.pixel_valid] != image[decoded.pixel_valid], axis=1))
        )
        valid_pixels += int(np.count_nonzero(decoded.pixel_valid))
        black_pixels += int(np.count_nonzero(~decoded.pixel_valid))
        resized, resized_valid = mask_aware_resize_ordinal(
            decoded.values,
            decoded.pixel_valid,
            (256, 256),
        )
        resize_invalid_nonzero += int(np.count_nonzero(resized[~resized_valid]))
        resize_valid_pixels += int(np.count_nonzero(resized_valid))

    result = {
        "stage": 3,
        "status": "passed" if roundtrip_mismatches == 0 and resize_invalid_nonzero == 0 else "failed",
        "selection": "first source Depth_Color frame from each train/val trial in the Stage 2 manifest",
        "trials_checked": len(rows),
        "valid_source_pixels": valid_pixels,
        "black_invalid_source_pixels": black_pixels,
        "roundtrip_mismatched_pixels": roundtrip_mismatches,
        "resized_valid_pixels": resize_valid_pixels,
        "resized_invalid_nonzero_pixels": resize_invalid_nonzero,
        "output_size": [256, 256],
        "competition_test_read": False,
        "depth_assets_exported": False,
        "training_run": False,
        "seconds": round(time.perf_counter() - started, 3),
    }
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Stage 3: inverse-JET ordinal Depth verification",
        "",
        f"- Status: **{result['status']}**",
        f"- Real train/val trials checked: {result['trials_checked']:,}",
        f"- Valid source pixels decoded: {result['valid_source_pixels']:,}",
        f"- Black invalid source pixels: {result['black_invalid_source_pixels']:,}",
        f"- Exact JET round-trip mismatches: {result['roundtrip_mismatched_pixels']:,}",
        f"- Nonzero pixels behind the resized invalid mask: {result['resized_invalid_nonzero_pixels']:,}",
        "- Output-size smoke check: `256x256`",
        "- Competition test read: no",
        "- Depth assets exported: no",
        "- Training run: no",
        "",
        "This stage implements and verifies the codec and mask-aware resize only. "
        "The Depth-only exporter and combined manifest belong to the next stage.",
    ]
    args.report_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
