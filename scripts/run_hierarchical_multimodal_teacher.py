from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_hierarchical_multimodal_teacher import (
    run_fixed_validation,
    run_grouped_cv,
    run_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hierarchical multimodal teacher")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--mode", choices=("smoke", "fixed-validation", "grouped-cv"), default="smoke"
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    mode = "smoke" if args.smoke_test else args.mode
    if mode == "grouped-cv":
        report = run_grouped_cv(args.config.resolve())
        print(json.dumps({"status": report["status"], "selected_candidate": report["selected_candidate"]}))
        return
    if mode == "fixed-validation":
        report = run_fixed_validation(
            args.config.resolve(), output_root=args.output_root
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "selected_candidate": report["selected_candidate"],
                }
            )
        )
        return
    output = args.output_root or (
        PROJECT_ROOT / "outputs/hierarchical_multimodal_midfusion_stage1/smoke"
    )
    report = run_smoke(args.config.resolve(), output_root=output.resolve())
    print(json.dumps(report))


if __name__ == "__main__":
    main()
