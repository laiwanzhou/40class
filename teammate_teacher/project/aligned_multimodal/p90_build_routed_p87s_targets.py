"""Build leakage-safe P87-S soft targets from the P90 visual router.

The held cohort is never used to fit either the router (done upstream) or the
visual probability temperature.  On rows where the strict outer router elects
to replace the P89-safe decision, its score is used as the visual-teacher
mixture mass.  All other P87-S structured targets remain bit-for-bit unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from p90_crossuser_visual_router import CANDIDATE_NAME, load_splits
from p90_teacher_fusion_audit import align
from p90_visual_teacher_safe_fusion_audit import load_visual_candidates


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_ROUTER_DIR = REPO_ROOT / "runs/p90_crossuser_visual_router_v1"
DEFAULT_TARGETS = HERE / "runs/p87s_holdout1_structured_targets_v1/structured_targets.npz"
DEFAULT_OUTPUT = HERE / "runs/p90_p87s_routed_targets_h1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--held-split",
        choices=("H1_selection", "H2_confirmation", "H3_independent_fold0"),
        default="H1_selection",
    )
    parser.add_argument("--base-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--router-dir", type=Path, default=DEFAULT_ROUTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    values = np.clip(probability, 1e-12, 1.0)
    return -np.sum(values * np.log(values), axis=1) / np.log(values.shape[1])


def calibrated_probability(
    train_probability: np.ndarray,
    train_labels: np.ndarray,
    held_probability: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    train_logp = np.log(np.clip(train_probability, 1e-12, 1.0))
    held_logp = np.log(np.clip(held_probability, 1e-12, 1.0))

    def nll(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probability = softmax(train_logp / temperature, axis=1)
        correct_probability = probability[np.arange(len(train_labels)), train_labels]
        return float(-np.log(np.clip(correct_probability, 1e-12, 1.0)).mean())

    result = minimize_scalar(nll, bounds=(-3.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Visual temperature calibration failed: {result.message}")
    temperature = float(np.exp(result.x))
    calibrated = softmax(held_logp / temperature, axis=1)
    return calibrated, {
        "temperature": temperature,
        "train_nll_before": nll(0.0),
        "train_nll_after": nll(float(result.x)),
    }


def metrics(prediction: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    correct = int(np.sum(prediction == labels))
    return {
        "correct": correct,
        "total": int(len(labels)),
        "accuracy": float(correct / len(labels)),
    }


def main() -> None:
    args = parse_args()
    router_dir = args.router_dir.resolve()
    base_path = args.base_targets.resolve()
    output = args.output_dir.resolve()

    router_summary = json.loads(
        (router_dir / "full_summary.json").read_text(encoding="utf-8")
    )
    train_splits = router_summary["cohorts"][args.held_split]["train_splits"]
    threshold = float(
        router_summary["cohorts"][args.held_split]["nested_threshold_selection"]["threshold"]
    )
    if args.held_split in train_splits:
        raise RuntimeError("Held split leaked into router training cohorts")

    splits = load_splits()
    held = splits[args.held_split]
    reference_ids, candidates = load_visual_candidates()
    full_candidate = candidates[CANDIDATE_NAME]
    train_probability = np.concatenate(
        [align(reference_ids, full_candidate, splits[name].sample_ids) for name in train_splits]
    )
    train_labels = np.concatenate([splits[name].labels for name in train_splits])
    held_probability = align(reference_ids, full_candidate, held.sample_ids)
    held_visual, calibration = calibrated_probability(
        train_probability, train_labels, held_probability
    )

    prefix = args.held_split
    with np.load(router_dir / "full_predictions.npz", allow_pickle=False) as source:
        router_ids = source[f"{prefix}_sample_ids"].astype(str)
        router_labels = source[f"{prefix}_labels"].astype(np.int64)
        safe_prediction = source[f"{prefix}_safe_prediction"].astype(np.int64)
        router_prediction = source[f"{prefix}_router_prediction"].astype(np.int64)
        route_score = source[f"{prefix}_route_score"].astype(np.float64)
    if not np.array_equal(router_ids, held.sample_ids):
        raise RuntimeError("Router prediction order differs from held protocol")
    if not np.array_equal(router_labels, held.labels):
        raise RuntimeError("Router labels differ from held protocol")
    route = router_prediction != safe_prediction
    if threshold > 1.0 and np.any(route):
        raise RuntimeError("No-route threshold produced routed rows")
    if threshold <= 1.0 and np.any(route & (route_score + 1e-12 < threshold)):
        raise RuntimeError("Router output contains a row below the frozen threshold")
    visual_prediction = held_visual.argmax(axis=1)
    if np.any(route & (visual_prediction != router_prediction)):
        raise RuntimeError("Calibrated visual argmax differs from routed prediction")

    with np.load(base_path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    target_ids = arrays["sample_ids"].astype(str)
    index = {sample_id: row for row, sample_id in enumerate(target_ids)}
    try:
        held_target_rows = np.asarray([index[sample_id] for sample_id in router_ids], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError(f"Base target is missing routed sample {error.args[0]}") from error
    target_mask = arrays["target_mask"].astype(bool)
    if not np.all(target_mask[held_target_rows]):
        raise RuntimeError("Base target mask does not cover every held router row")
    if int(target_mask.sum()) != len(router_ids):
        raise RuntimeError(
            "Base target mask contains rows outside the held cohort; use the matching fold targets"
        )

    base_probability = arrays["structured_distillation_probability"].astype(np.float64)
    original_held_probability = base_probability[held_target_rows].copy()
    routed_held_probability = original_held_probability.copy()
    # The router score estimates P(visual is preferable to safe) on disagreements,
    # hence it is the natural mixture posterior.  No held labels tune this weight.
    routed_held_probability[route] = (
        (1.0 - route_score[route, None]) * original_held_probability[route]
        + route_score[route, None] * held_visual[route]
    )
    routed_held_probability /= routed_held_probability.sum(axis=1, keepdims=True)
    base_probability[held_target_rows] = routed_held_probability
    arrays["structured_distillation_probability"] = base_probability.astype(np.float32)
    arrays["structured_distillation_prediction"] = base_probability.argmax(axis=1).astype(
        np.int64
    )
    confidence = arrays["structured_confidence"].astype(np.float64)
    confidence[held_target_rows] = 1.0 - normalized_entropy(routed_held_probability)
    arrays["structured_confidence"] = confidence.astype(np.float32)
    score_by_id = dict(zip(router_ids.tolist(), route_score.tolist()))
    arrays.update(
        p90_router_route=np.isin(target_ids, router_ids[route]),
        p90_router_score=np.asarray(
            [score_by_id.get(sample_id, 0.0) for sample_id in target_ids],
            dtype=np.float32,
        ),
    )

    # Labels appear only below this point, after the target tensor is finalized.
    old_prediction = original_held_probability.argmax(axis=1)
    new_prediction = routed_held_probability.argmax(axis=1)
    old_correct = old_prediction == router_labels
    new_correct = new_prediction == router_labels
    summary = {
        "stage": "P90 routed P87-S soft-target construction",
        "protocol": (
            "The outer router and threshold were fit only on the two non-held cohorts. "
            "Visual temperature is fit on those same cohorts. Held labels are accessed "
            "only after the output probabilities have been finalized, for audit."
        ),
        "held_split": args.held_split,
        "router_training_splits": train_splits,
        "router_threshold": threshold,
        "candidate": CANDIDATE_NAME,
        "visual_calibration": calibration,
        "mixture_rule": "alpha=router_score on routed disagreements; alpha=0 otherwise",
        "held_rows": int(len(router_ids)),
        "routed_rows": int(route.sum()),
        "mean_routed_alpha": float(route_score[route].mean()) if np.any(route) else 0.0,
        "changed_target_argmax": int(np.sum(new_prediction != old_prediction)),
        "base_target_metrics": metrics(old_prediction, router_labels),
        "routed_target_metrics": metrics(new_prediction, router_labels),
        "router_hard_metrics": metrics(router_prediction, router_labels),
        "safe_hard_metrics": metrics(safe_prediction, router_labels),
        "target_rescue": int(np.sum(~old_correct & new_correct)),
        "target_harm": int(np.sum(old_correct & ~new_correct)),
        "target_net_gain": int(np.sum(new_correct) - np.sum(old_correct)),
        "base_targets": str(base_path),
        "base_targets_sha256": sha256(base_path),
        "router_predictions": str(router_dir / "full_predictions.npz"),
        "router_predictions_sha256": sha256(router_dir / "full_predictions.npz"),
        "label_use_assertion": "held labels are audit-only and do not influence saved targets",
    }

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "structured_targets.npz", **arrays)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
