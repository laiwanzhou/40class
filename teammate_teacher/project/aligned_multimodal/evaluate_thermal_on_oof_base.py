from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_thermal_oof_fusion import (
    cross_fitted_thermal_residual,
    load_thermal,
    serialize_result,
    write_predictions,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_THERMAL_ROOT = (
    PROJECT_DIR.parent / "thermal_baseline" / "runs" / "p11_thermal_imagenet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add cross-fitted Thermal residual fusion to an arbitrary OOF base"
    )
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--base-key", default="fused_logits")
    parser.add_argument("--thermal-root", type=Path, default=DEFAULT_THERMAL_ROOT)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.base_oof.resolve(), allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str)
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int64)
        base_logits = data[args.base_key].astype(np.float32)
    thermal = load_thermal(args.thermal_root.resolve())
    result = cross_fitted_thermal_residual(
        args.name,
        sample_ids,
        labels,
        folds,
        base_logits,
        thermal,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.csv", result)
    np.savez_compressed(
        output_dir / "logits.npz",
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        base_logits=base_logits,
        candidate_logits=result["logits"],
        expert_present=result["present"].astype(np.int64),
    )
    summary = {
        "name": args.name,
        "protocol": (
            "Thermal temperatures and weights are selected on the other two OOF "
            "folds. Thermal-missing samples fall back exactly to the supplied base."
        ),
        **serialize_result(
            result,
            int(args.bootstrap_repeats),
            int(args.seed),
        ),
        "sources": {
            "base_oof": str(args.base_oof.resolve()),
            "thermal_root": str(args.thermal_root.resolve()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
