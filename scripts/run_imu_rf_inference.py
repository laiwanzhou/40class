from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.inference.imu_rf_inference import run_imu_rf_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final IMU Random Forest")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_imu_rf_inference(
        model_dir=args.model_dir,
        input_index=args.input_index,
        stage2_root=args.stage2_root,
        output_dir=args.output_dir,
        split=args.split,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
