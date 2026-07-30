from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.training.imu_rf_finalization import run_finalization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize the compact IMU Random Forest")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "imu_rf_final_v1.json"
    )
    parser.add_argument(
        "--compact-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "imu_rf_compact_fold0.json",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_finalization(
        feature_root=args.feature_root,
        compact_root=args.compact_root,
        config_path=args.config,
        compact_config_path=args.compact_config,
        output_root=args.output_root,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if result.get("status") in {"preflight_ok", "exact_match"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
