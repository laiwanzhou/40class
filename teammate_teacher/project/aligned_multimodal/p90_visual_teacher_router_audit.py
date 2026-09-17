"""Confidence router screen between P89 safe and P90 visual teachers.

All scalar thresholds are selected on H1 with no H1-user regression and are
then transferred unchanged to H2 and H3.  This is a deliberately small rule
family to test whether the large visual oracle gap is trivially routable.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p89_supported_template_gate import (
    h3_protocol,
    load_grouping,
    load_imu,
    safe_probability_and_prediction,
)
from p90_teacher_fusion_audit import align, temper, users_audit
from p90_visual_teacher_safe_fusion_audit import load_visual_candidates


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "runs/p90_visual_teacher_router_audit_v1"


def margin(probability: np.ndarray) -> np.ndarray:
    top = np.partition(probability, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def route(
    safe_probability: np.ndarray,
    safe_prediction: np.ndarray,
    candidate_probability: np.ndarray,
    configuration: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    candidate = temper(candidate_probability, configuration["temperature"])
    candidate_prediction = candidate.argmax(axis=1)
    candidate_confidence = candidate.max(axis=1)
    safe_confidence = safe_probability[
        np.arange(len(safe_prediction)), safe_prediction
    ]
    selected = (
        (candidate_prediction != safe_prediction)
        & (candidate_confidence >= configuration["candidate_min"])
        & (safe_confidence <= configuration["safe_max"])
        & (
            candidate_confidence - safe_confidence
            >= configuration["confidence_gap_min"]
        )
        & (margin(candidate) >= configuration["candidate_margin_min"])
    )
    prediction = safe_prediction.copy()
    prediction[selected] = candidate_prediction[selected]
    return prediction, selected


def report_rule(
    labels: np.ndarray,
    safe_probability: np.ndarray,
    safe_prediction: np.ndarray,
    candidate_probability: np.ndarray,
    users: np.ndarray,
    configuration: dict[str, float],
) -> dict[str, Any]:
    prediction, selected = route(
        safe_probability, safe_prediction, candidate_probability, configuration
    )
    safe_correct = safe_prediction == labels
    prediction_correct = prediction == labels
    by_user = users_audit(labels, safe_prediction, prediction, users)
    return {
        "configuration": configuration,
        "metrics": classification_metrics(labels, prediction),
        "route_count": int(selected.sum()),
        "rescue": int(np.sum(~safe_correct & prediction_correct)),
        "harm": int(np.sum(safe_correct & ~prediction_correct)),
        "minimum_user_gain": min(row["gain"] for row in by_user.values()),
        "per_user": by_user,
    }


def select_h1(
    labels: np.ndarray,
    safe_probability: np.ndarray,
    safe_prediction: np.ndarray,
    candidate_probability: np.ndarray,
    users: np.ndarray,
) -> dict[str, Any]:
    results = []
    for values in itertools.product(
        (0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
        (0.0, 0.2, 0.4, 0.6, 0.8, 0.9),
        (0.2, 0.4, 0.6, 0.8, 1.0),
        (-0.5, -0.25, 0.0, 0.1, 0.2, 0.3),
        (0.0, 0.05, 0.1, 0.2),
    ):
        configuration = dict(
            zip(
                (
                    "temperature",
                    "candidate_min",
                    "safe_max",
                    "confidence_gap_min",
                    "candidate_margin_min",
                ),
                values,
            )
        )
        results.append(
            report_rule(
                labels,
                safe_probability,
                safe_prediction,
                candidate_probability,
                users,
                configuration,
            )
        )
    baseline_correct = int(np.sum(safe_prediction == labels))
    valid = [
        row
        for row in results
        if row["minimum_user_gain"] >= 0
        and row["metrics"]["correct"] >= baseline_correct
    ]
    if not valid:
        return {
            "configuration": None,
            "metrics": classification_metrics(labels, safe_prediction),
            "route_count": 0,
            "rescue": 0,
            "harm": 0,
            "minimum_user_gain": 0,
            "per_user": users_audit(labels, safe_prediction, safe_prediction, users),
        }
    valid.sort(
        key=lambda row: (
            row["metrics"]["correct"],
            row["metrics"]["balanced_accuracy"],
            row["rescue"] - row["harm"],
            -row["harm"],
            -row["route_count"],
        ),
        reverse=True,
    )
    return valid[0]


def main() -> None:
    grouping = load_grouping()
    old_imu_ids, old_imu_logits = load_imu()
    h1_raw = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2_raw = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_probability, h1_safe = safe_probability_and_prediction(
        h1_raw, old_imu_ids, old_imu_logits, grouping
    )
    h2_probability, h2_safe = safe_probability_and_prediction(
        h2_raw, old_imu_ids, old_imu_logits, grouping
    )
    h3_raw, h3_probability, h3_safe = h3_protocol(
        old_imu_ids, old_imu_logits, grouping
    )[:3]
    splits = {
        "H1_selection": (h1_raw, h1_probability, h1_safe),
        "H2_confirmation": (h2_raw, h2_probability, h2_safe),
        "H3_independent_fold0": (h3_raw, h3_probability, h3_safe),
    }
    reference_ids, candidates = load_visual_candidates()
    report: dict[str, Any] = {
        "protocol": (
            "Small scalar-confidence router family selected on H1 with no H1 "
            "user regression; configuration transferred unchanged to H2/H3."
        ),
        "safe": {
            split: classification_metrics(protocol[1], safe)
            for split, (protocol, _, safe) in splits.items()
        },
        "candidates": {},
    }
    for name, full_probability in candidates.items():
        probabilities = {
            split: align(reference_ids, full_probability, protocol[0])
            for split, (protocol, _, _) in splits.items()
        }
        h1_protocol, h1_safe_probability, h1_prediction = splits["H1_selection"]
        selected = select_h1(
            h1_protocol[1],
            h1_safe_probability,
            h1_prediction,
            probabilities["H1_selection"],
            h1_protocol[4].users,
        )
        item: dict[str, Any] = {"H1_selection": selected}
        if selected["configuration"] is not None:
            for split in ("H2_confirmation", "H3_independent_fold0"):
                protocol, safe_probability, safe_prediction = splits[split]
                item[split] = report_rule(
                    protocol[1],
                    safe_probability,
                    safe_prediction,
                    probabilities[split],
                    protocol[4].users,
                    selected["configuration"],
                )
        report["candidates"][name] = item
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
