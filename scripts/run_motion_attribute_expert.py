from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.motion_attribute_config import (
    load_motion_attribute_config,
    project_path,
)
from src.train_motion_attribute_expert import run_motion_attribute_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Motion Attribute Expert")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/motion_attribute_expert.yaml",
    )
    parser.add_argument("--mode", choices=("smoke",), default="smoke")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_motion_attribute_config(args.config.resolve())
    root = args.output_root or project_path(str(config["outputs"]["root"])) / "smoke"
    cache = project_path(str(config["outputs"]["root"])) / "input_cache.npz"
    report = run_motion_attribute_smoke(
        args.config.resolve(), cache_path=cache, output_root=root
    )
    formal = project_path(str(config["outputs"]["smoke_report"]))
    if formal.exists():
        raise FileExistsError(formal)
    formal.parent.mkdir(parents=True, exist_ok=True)
    temporary = formal.with_suffix(formal.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(formal)
    print(json.dumps({"status": report["status"], "peak_cuda_mib": report["peak_cuda_mib"]}))


if __name__ == "__main__":
    main()
