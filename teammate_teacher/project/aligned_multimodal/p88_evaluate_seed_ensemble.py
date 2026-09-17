from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import classification_metrics, decode_sessions
from p88_aligned_repeat_holdout import decode_aligned_repeat
from p88_oof_candidate_ensemble import load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stochastic P88 adaptation ensembles.")
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--alternate-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=Path("runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"))
    parser.add_argument("--train-metadata", type=Path, default=Path("data/p85_recording_metadata/train_recording_metadata.csv"))
    parser.add_argument("--repeat-config-summary", type=Path, default=Path("runs/p88_aligned_repeat_h1_v1/summary.json"))
    parser.add_argument("--fixed-summary", type=Path)
    return parser.parse_args()


def load_run(run: Path, sample_ids: np.ndarray) -> np.ndarray:
    run = run.resolve()
    with (run / "subject_holdout_predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    current = np.asarray([row["sample_id"] for row in rows])
    if not np.array_equal(current, sample_ids):
        raise RuntimeError(f"row order differs for {run}")
    return np.asarray(np.load(run / "subject_holdout_logits.npy"), dtype=np.float64)


def main() -> None:
    args = parse_args()
    (
        sample_ids, labels, _base_probability, base_decoded, metadata, indices,
        sessions, transition, decoder, repeat,
    ) = load_protocol(args)
    logits = [
        np.asarray(np.load(args.base_run.resolve() / "subject_holdout_logits.npy"), dtype=np.float64)
    ] + [load_run(run, sample_ids) for run in args.alternate_runs]
    names = [args.base_run.resolve().name] + [run.resolve().name for run in args.alternate_runs]
    if args.fixed_summary:
        source = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8"))
        configurations = [source["best"]["configuration"]]
        selected_here = False
    else:
        configurations = []
        for index in range(len(logits)):
            weights = [0.0] * len(logits); weights[index] = 1.0
            configurations.append({"name": f"single_{index}", "weights": weights})
        for second in range(1, len(logits)):
            for weight in (0.25, 0.5, 0.75):
                weights = [0.0] * len(logits)
                weights[0] = 1.0 - weight; weights[second] = weight
                configurations.append({
                    "name": f"base_alt{second}_{weight:g}", "weights": weights,
                })
        if len(logits) >= 3:
            configurations.append({
                "name": "all_equal", "weights": [1.0 / len(logits)] * len(logits),
            })
            weights = [0.0] * len(logits); weights[1] = weights[2] = 0.5
            configurations.append({"name": "alternate_pair_equal", "weights": weights})
        selected_here = True
    results = []
    for configuration in configurations:
        weights = np.asarray(configuration["weights"], dtype=np.float64)
        if len(weights) != len(logits) or not np.isclose(weights.sum(), 1.0):
            raise RuntimeError("invalid seed ensemble weights")
        blended = np.sum(
            np.stack(logits, axis=0) * weights[:, None, None], axis=0
        )
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
        result["decoded"]["correct"], result["raw"]["correct"],
    ), reverse=True)
    summary = {
        "stage": "P88_stochastic_adaptation_ensemble", "status": "complete",
        "selected_on_current_holdout": selected_here,
        "holdout_users": sorted(args.holdout_users), "run_names": names,
        "base_decoded": classification_metrics(labels, base_decoded),
        "best": results[0], "repeat_configuration": asdict(repeat),
        "decoder_configuration": asdict(decoder), "grid_size": len(results),
        "all_candidates": results,
    }
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
