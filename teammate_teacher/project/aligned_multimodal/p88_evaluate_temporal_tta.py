from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import classification_metrics, decode_sessions
from p88_aligned_repeat_holdout import decode_aligned_repeat
from p88_oof_candidate_ensemble import load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic P88 temporal TTA.")
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--tta-cache", type=Path, required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=Path("runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"))
    parser.add_argument("--train-metadata", type=Path, default=Path("data/p85_recording_metadata/train_recording_metadata.csv"))
    parser.add_argument("--repeat-config-summary", type=Path, default=Path("runs/p88_aligned_repeat_h1_v1/summary.json"))
    parser.add_argument("--fixed-summary", type=Path)
    parser.add_argument("--weights", type=float, nargs="+", default=(0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        sample_ids, labels, _base_probability, base_decoded, metadata, indices,
        sessions, transition, decoder, repeat,
    ) = load_protocol(args)
    with np.load(args.tta_cache.resolve(), allow_pickle=False) as cache:
        if not np.array_equal(cache["sample_ids"].astype(str), sample_ids):
            raise RuntimeError("temporal TTA cache row order differs")
        logits = {key: cache[key].astype(np.float64) for key in cache.files if key not in {"sample_ids", "labels"}}
    means = {
        "wide_scale": np.mean([logits["scale_086"], logits["scale_114"]], axis=0),
        "mild_scale": np.mean([logits["scale_093"], logits["scale_107"]], axis=0),
        "shift": np.mean([logits["shift_m1"], logits["shift_p1"]], axis=0),
        "all": np.mean([
            logits["scale_086"], logits["scale_114"],
            logits["shift_m1"], logits["shift_p1"],
        ], axis=0),
    }
    selected_here = args.fixed_summary is None
    if args.fixed_summary:
        source = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8"))
        configurations = [source["best"]["configuration"]]
    else:
        configurations = [
            {"family": family, "weight": float(weight)}
            for family in means for weight in args.weights
        ]
    results = []
    original = logits["original"]
    for configuration in configurations:
        family = str(configuration["family"])
        weight = float(configuration["weight"])
        blended = (1.0 - weight) * original + weight * means[family]
        logp = log_softmax_numpy(blended)
        raw = logp.argmax(axis=1)
        decoded = decode_sessions(logp, sessions, transition, decoder)
        aligned, grouping = decode_aligned_repeat(
            logp, indices, metadata, transition, decoder, repeat
        )
        results.append({
            "configuration": configuration,
            "raw": classification_metrics(labels, raw),
            "decoded": classification_metrics(labels, decoded),
            "aligned": classification_metrics(labels, aligned),
            "aligned_rescue_harm_vs_base": rescue_harm(labels, base_decoded, aligned),
            "grouping": grouping,
        })
    results.sort(key=lambda result: (
        result["aligned"]["correct"], result["aligned"]["balanced_accuracy"],
        result["decoded"]["correct"], -result["configuration"]["weight"],
    ), reverse=True)
    summary = {
        "stage": "P88_temporal_TTA_evaluation", "status": "complete",
        "selected_on_current_holdout": selected_here,
        "holdout_users": sorted(args.holdout_users),
        "base_decoded": classification_metrics(labels, base_decoded),
        "best": results[0], "repeat_configuration": asdict(repeat),
        "decoder_configuration": asdict(decoder), "grid_size": len(configurations),
        "all_candidates": results,
    }
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
