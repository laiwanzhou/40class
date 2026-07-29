from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.training.imu_rf_trainer import train_random_forest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the fixed IMU random forest screen")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("plain", "balanced"), required=True)
    parser.add_argument("--random-state", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_random_forest(
        feature_root=args.feature_root,
        config_path=args.config,
        variant=args.variant,
        random_state=args.random_state,
        output_dir=args.output_dir,
        preflight_only=args.preflight_only,
    )
    serializable = {
        key: value
        for key, value in result.items()
        if key not in {"predictions", "probabilities"}
    }
    print(json.dumps(serializable, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

