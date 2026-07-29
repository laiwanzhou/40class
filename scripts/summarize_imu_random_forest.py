from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.training.imu_rf_trainer import summarize_rf_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize the fixed IMU RF screen")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--baseline-validation-outputs", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = summarize_rf_experiment(
        experiment_root=args.experiment_root,
        baseline_validation_outputs=args.baseline_validation_outputs,
        config_path=args.config,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

