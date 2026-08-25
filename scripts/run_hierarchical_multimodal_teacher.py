from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_hierarchical_multimodal_teacher import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hierarchical multimodal teacher")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if not args.smoke_test:
        raise ValueError("only --smoke-test is implemented before grouped-CV qualification")
    output = args.output_root or (
        PROJECT_ROOT / "outputs/hierarchical_multimodal_midfusion_stage1/smoke"
    )
    report = run_smoke(args.config.resolve(), output_root=output.resolve())
    print(json.dumps(report))


if __name__ == "__main__":
    main()
