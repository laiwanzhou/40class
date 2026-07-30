from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.training.imu_rf_production import build_production_dataset, train_production_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the final full-data IMU Random Forest")
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--training-index-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "imu_rf_final_v1.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = build_production_dataset(
        stage2_root=args.stage2_root,
        training_index_dir=args.training_index_dir,
        feature_root=args.feature_root,
    )
    if args.preflight_only:
        result = {
            "status": "preflight_ok",
            "sample_count": len(dataset.sample_ids),
            "feature_shape": list(dataset.features.shape),
            "features_finite": bool(__import__("numpy").isfinite(dataset.features).all()),
            "training_data_manifest": dataset.training_data_manifest,
        }
    else:
        if args.output_dir is None:
            raise ValueError("--output-dir is required unless --preflight-only is used")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
        result = train_production_package(
            dataset=dataset,
            config_path=args.config,
            output_dir=args.output_dir,
            creation_commit=commit,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
