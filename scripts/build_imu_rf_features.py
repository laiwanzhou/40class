from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.features.imu_rf_features import build_rf_feature_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed IMU RF summary features")
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--training-index-dir", type=Path, required=True)
    parser.add_argument("--normalization-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_rf_feature_artifacts(
        stage2_root=args.stage2_root,
        training_index_dir=args.training_index_dir,
        normalization_dir=args.normalization_dir,
        output_dir=args.output_dir,
        preflight_only=args.preflight_only,
        repository_root=REPOSITORY_ROOT,
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
