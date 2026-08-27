from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.motionbert_p6b_config import (
    load_motionbert_p6b_config,
    project_path,
)
from src.train_motionbert_lite_skeleton_expert import run_motionbert_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MotionBERT-Lite Skeleton expert P6-B")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml",
    )
    parser.add_argument("--mode", choices=("smoke",), default="smoke")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_motionbert_p6b_config(args.config.resolve())
    output = args.output_root or project_path(str(config["outputs"]["root"])) / "smoke"
    report = run_motionbert_smoke(args.config.resolve(), output_root=output)
    formal_report = project_path(str(config["outputs"]["smoke_report"]))
    if formal_report.exists():
        raise FileExistsError(formal_report)
    formal_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = formal_report.with_suffix(formal_report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(formal_report)
    print(json.dumps({"status": report["status"], "peak_cuda_mib": report["peak_cuda_mib"]}))


if __name__ == "__main__":
    main()
