from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.train_thermal_generation2 import (
    load_generation2_config,
    require_training_authorization,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strictly authorized Thermal generation-2 job.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorize-training")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_generation2_config(args.config.resolve())
    if args.validate_only:
        print(
            json.dumps(
                {
                    "experiment_id": config["experiment_id"],
                    "status": "validated_not_started",
                    "training_authorized": config["training_authorized"],
                },
                sort_keys=True,
            )
        )
        return
    require_training_authorization(config, token=args.authorize_training)
    raise RuntimeError(
        "formal generation-2 execution remains disabled until the A2 runtime gate passes"
    )


if __name__ == "__main__":
    main()
