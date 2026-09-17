from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from adapt_p87s_structured_student import (
    PROJECT_DIR,
    make_loader,
    model_build_args,
    resolve_config_path,
)
from p86_cached_motion_data import P86CachedSequenceMotionDataset
from train_p86_mobind_fusion_proxy import build_model, evaluate


DEFAULT_TARGETS = (
    PROJECT_DIR / "runs/p87s_holdout1_structured_targets_v1/structured_targets.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one derived P87-S checkpoint on the frozen subject holdout."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--structured-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / "unified_student.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reference_dir = args.reference_run.resolve()
    reference_summary = json.loads(
        (reference_dir / "summary.json").read_text(encoding="utf-8")
    )
    build_args = model_build_args(checkpoint_path, reference_summary)
    model, _, _ = build_model(build_args)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    config = reference_summary["config"]
    repository_root = PROJECT_DIR.parent
    common = {
        "sequence_cache": resolve_config_path(config["sequence_cache"], repository_root),
        "motion_cache": resolve_config_path(config["motion_cache"], repository_root),
        "pixel_cache": resolve_config_path(config["pixel_cache"], repository_root),
        "teacher_features": resolve_config_path(
            config["teacher_features"], repository_root
        ),
        "teacher_logits": resolve_config_path(config["teacher_logits"], repository_root),
        "imu_teacher_logits": (
            resolve_config_path(config["imu_teacher_logits"], repository_root)
            if config.get("imu_teacher_logits")
            else None
        ),
        "imu_event_features": (
            resolve_config_path(config["imu_event_features"], repository_root)
            if config.get("imu_event_features")
            else None
        ),
    }
    full = P86CachedSequenceMotionDataset(**common)
    targets = np.load(args.structured_targets.resolve(), allow_pickle=False)
    sample_ids = targets["sample_ids"].astype(str)
    selected_ids = sample_ids[targets["target_mask"].astype(bool)]
    indices = np.asarray(
        [full.index_lookup[sample_id] for sample_id in selected_ids], dtype=np.int64
    )
    evaluation = P86CachedSequenceMotionDataset(
        **common, indices=indices, temporal_augment=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics, rows, logits = evaluate(
        model,
        make_loader(evaluation, args.batch_size, args.workers, shuffle=False),
        device,
        max_batches=0,
    )
    if [row["sample_id"] for row in rows] != selected_ids.tolist():
        raise RuntimeError("Derived checkpoint evaluation row order differs from targets")
    write_rows(run_dir / "subject_holdout_predictions.csv", rows)
    np.save(run_dir / "subject_holdout_logits.npy", logits)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "status": "formal_evaluated",
            "subject_holdout_metrics": metrics,
            "reference_run": str(reference_dir),
            "evaluation_rows": len(rows),
        }
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
