from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from p86v2_metrics import emission_metrics, rescue_harm, softmax
from p86v2_protocol import build_split, load_protocol, read_rows


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = PROJECT_DIR / "runs/p86v2_visual_baseline_dev_a_v1"
DEFAULT_CANDIDATE = PROJECT_DIR / "runs/p86v2_visual_temporal_dedup_dev_a_v1"
DEFAULT_PIXELS = PROJECT_DIR / "runs/p86_visual_pixel_cache_t16_r160_v12"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p86v2_visual_h01_analysis_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired P86-v2 H0/H1 development analysis")
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-run", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_run(path: Path) -> tuple[dict, dict[str, np.ndarray], float]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != "P86-v2" or summary.get("split") != "development":
        raise RuntimeError(f"not a formal P86-v2 development run: {path}")
    if summary.get("status") != "formal":
        raise RuntimeError(f"cannot compare a smoke run: {path}")
    with np.load(path / "holdout_logits.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    history = np.genfromtxt(path / "training_history.csv", delimiter=",", names=True)
    seconds = float(np.atleast_1d(history["seconds"]).sum())
    return summary, arrays, seconds


def exact_sign_pvalue(rescued: int, harmed: int) -> float:
    discordant = rescued + harmed
    if discordant == 0:
        return 1.0
    lower = min(rescued, harmed)
    probability = sum(math.comb(discordant, index) for index in range(lower + 1))
    probability /= 2**discordant
    return min(1.0, 2.0 * probability)


def metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    keys = (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "worst_subject_accuracy",
        "negative_log_likelihood",
        "brier_score",
        "ece_15",
        "mean_confidence",
        "mean_entropy",
    )
    return {key: float(candidate[key]) - float(baseline[key]) for key in keys}


def stratum_metrics(
    name: str,
    mask: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    baseline_logits: np.ndarray,
    candidate_logits: np.ndarray,
    repeats: np.ndarray,
) -> dict:
    baseline = emission_metrics(baseline_logits[mask], labels[mask], users[mask])
    candidate = emission_metrics(candidate_logits[mask], labels[mask], users[mask])
    paired = rescue_harm(candidate_logits[mask], baseline_logits[mask], labels[mask], users[mask])
    return {
        "name": name,
        "samples": int(mask.sum()),
        "repeat_slots_mean": float(repeats[mask].mean()),
        "baseline_accuracy": baseline["accuracy"],
        "candidate_accuracy": candidate["accuracy"],
        "accuracy_delta": candidate["accuracy"] - baseline["accuracy"],
        "baseline_nll": baseline["negative_log_likelihood"],
        "candidate_nll": candidate["negative_log_likelihood"],
        "nll_delta": candidate["negative_log_likelihood"]
        - baseline["negative_log_likelihood"],
        "rescued": paired["rescued"],
        "harmed": paired["harmed"],
        "net": paired["net"],
    }


def main() -> None:
    args = parse_args()
    baseline_summary, baseline, baseline_seconds = load_run(args.baseline_run.resolve())
    candidate_summary, candidate, candidate_seconds = load_run(args.candidate_run.resolve())
    for key in ("sample_ids", "users", "labels"):
        if not np.array_equal(baseline[key], candidate[key]):
            raise RuntimeError(f"paired run mismatch: {key}")
    sample_ids = baseline["sample_ids"].astype(str)
    users = baseline["users"].astype(str)
    labels = baseline["labels"].astype(np.int64)
    baseline_logits = baseline["logits"].astype(np.float64)
    candidate_logits = candidate["logits"].astype(np.float64)

    rows = read_rows(args.pixel_cache / "rows.csv")
    development = build_split(rows, "development", load_protocol())
    expected_ids = {rows[index]["sample_id"] for index in development.holdout_indices}
    if set(sample_ids.tolist()) != expected_ids:
        raise RuntimeError("run predictions are not exactly the frozen development holdout")
    source = np.load(args.pixel_cache / "source_frame_indices.npy", mmap_mode="r")
    lookup = {row["sample_id"]: index for index, row in enumerate(rows)}
    repeats = np.asarray(
        [32 - len(np.unique(source[lookup[sample_id]].reshape(-1))) for sample_id in sample_ids],
        dtype=np.int64,
    )
    strata = []
    for name, low, high in (
        ("repeat_0_3", 0, 3),
        ("repeat_4_7", 4, 7),
        ("repeat_8_15", 8, 15),
        ("repeat_16_plus", 16, 31),
    ):
        mask = (repeats >= low) & (repeats <= high)
        if mask.any():
            strata.append(
                stratum_metrics(
                    name,
                    mask,
                    labels,
                    users,
                    baseline_logits,
                    candidate_logits,
                    repeats,
                )
            )

    baseline_metrics = emission_metrics(baseline_logits, labels, users)
    candidate_metrics = emission_metrics(candidate_logits, labels, users)
    paired = rescue_harm(candidate_logits, baseline_logits, labels, users)
    baseline_probability = softmax(baseline_logits)
    candidate_probability = softmax(candidate_logits)
    result = {
        "stage": "P86-v2",
        "experiment": "H1_temporal_source_dedup",
        "baseline_architecture": baseline_summary["architecture"],
        "candidate_architecture": candidate_summary["architecture"],
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_delta_candidate_minus_baseline": metric_delta(
            candidate_metrics, baseline_metrics
        ),
        "rescue_harm": paired,
        "paired_exact_sign_pvalue": exact_sign_pvalue(paired["rescued"], paired["harmed"]),
        "repeat_strata": strata,
        "true_probability_delta_mean": float(
            (
                candidate_probability[np.arange(len(labels)), labels]
                - baseline_probability[np.arange(len(labels)), labels]
            ).mean()
        ),
        "cost": {
            "baseline_parameters": baseline_summary["student_parameters"],
            "candidate_parameters": candidate_summary["student_parameters"],
            "parameter_delta": candidate_summary["student_parameters"]
            - baseline_summary["student_parameters"],
            "baseline_checkpoint_bytes": baseline_summary["checkpoint_bytes"],
            "candidate_checkpoint_bytes": candidate_summary["checkpoint_bytes"],
            "baseline_train_seconds": baseline_seconds,
            "candidate_train_seconds": candidate_seconds,
        },
        "confirmation_predictions_read": False,
        "embargo_predictions_or_metrics_read": False,
        "kaggle_test_used": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
