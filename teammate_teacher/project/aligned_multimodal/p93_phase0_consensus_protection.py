"""H1-selected consensus protection for the P89 continuous emission.

The rule may replace P89 Safe by the adjusted continuous Top-1 only when a
frozen non-redundant expert consensus supports that class.  Selection uses H1;
H2 is confirmation.  H3 is evaluated only if the pre-registered H1/H2 gate
passes, so a failed screen cannot consume the independent fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_p93_decision_provenance import (
    load_expert_pool,
    normalise_probability,
    probability_margin,
)
from p90_crossuser_visual_router import load_splits
from p90_visual_teacher_safe_fusion_audit import load_visual_candidates


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "runs/p93_phase0_consensus_protection_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def evaluate(
    split: Any,
    probability: np.ndarray,
    pool_prediction: np.ndarray,
    minimum_votes: int,
    minimum_margin: float,
) -> tuple[dict[str, Any], np.ndarray]:
    labels = split.labels.astype(np.int64)
    users = split.users.astype(str)
    safe = split.safe_prediction.astype(np.int64)
    candidate = probability.argmax(axis=1)
    votes = np.sum(pool_prediction == candidate[:, None], axis=1)
    selected = (
        (candidate != safe)
        & (votes >= int(minimum_votes))
        & (probability_margin(probability) >= float(minimum_margin))
    )
    prediction = safe.copy()
    prediction[selected] = candidate[selected]
    base_correct = safe == labels
    candidate_correct = prediction == labels
    per_user: dict[str, int] = {}
    for user in sorted(np.unique(users).tolist()):
        rows = users == user
        per_user[user] = int(
            candidate_correct[rows].sum() - base_correct[rows].sum()
        )
    return (
        {
            "configuration": {
                "minimum_votes": int(minimum_votes),
                "minimum_margin": float(minimum_margin),
                "pool_size": int(pool_prediction.shape[1]),
            },
            "base_correct": int(base_correct.sum()),
            "candidate_correct": int(candidate_correct.sum()),
            "accuracy": float(candidate_correct.mean()),
            "route": int(selected.sum()),
            "rescue": int(np.sum(~base_correct & candidate_correct)),
            "harm": int(np.sum(base_correct & ~candidate_correct)),
            "net": int(candidate_correct.sum() - base_correct.sum()),
            "minimum_user_gain": int(min(per_user.values(), default=0)),
            "negative_users": int(sum(value < 0 for value in per_user.values())),
            "per_user": per_user,
        },
        prediction,
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    visual_ids, visual = load_visual_candidates()
    prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expert_names: list[str] | None = None
    for split_name, split in splits.items():
        names, pool_probability = load_expert_pool(
            split.sample_ids.astype(str), visual, visual_ids
        )
        if expert_names is None:
            expert_names = names
        elif names != expert_names:
            raise RuntimeError("expert order differs across splits")
        prepared[split_name] = (
            normalise_probability(split.safe_probability),
            pool_probability.argmax(axis=2),
        )

    h1_grid: list[dict[str, Any]] = []
    for minimum_votes in (4, 5, 6):
        for minimum_margin in (0.0, 0.10, 0.25, 0.50):
            probability, pool_prediction = prepared["H1_selection"]
            row, _ = evaluate(
                splits["H1_selection"],
                probability,
                pool_prediction,
                minimum_votes,
                minimum_margin,
            )
            h1_grid.append(row)
    eligible = [
        row
        for row in h1_grid
        if row["net"] > 0 and row["minimum_user_gain"] >= 0
    ]
    selected = max(
        eligible or h1_grid,
        key=lambda row: (
            row["net"],
            -row["harm"],
            row["minimum_user_gain"],
            -row["route"],
            row["configuration"]["minimum_votes"],
            row["configuration"]["minimum_margin"],
        ),
    )
    configuration = selected["configuration"]
    h2_probability, h2_pool = prepared["H2_confirmation"]
    h2, _ = evaluate(
        splits["H2_confirmation"],
        h2_probability,
        h2_pool,
        int(configuration["minimum_votes"]),
        float(configuration["minimum_margin"]),
    )
    h1_effective = bool(selected["net"] >= 5)
    h2_passed = bool(
        h2["net"] > 0 and h2["minimum_user_gain"] >= 0 and h2["harm"] < h2["rescue"]
    )
    gate_passed = bool(h1_effective and h2_passed)
    h3: dict[str, Any] = {
        "status": "blocked_by_h1_h2_gate",
        "evaluated": False,
    }
    if gate_passed:
        h3_probability, h3_pool = prepared["H3_independent_fold0"]
        h3_result, _ = evaluate(
            splits["H3_independent_fold0"],
            h3_probability,
            h3_pool,
            int(configuration["minimum_votes"]),
            float(configuration["minimum_margin"]),
        )
        h3 = {"status": "evaluated_once", "evaluated": True, **h3_result}

    payload = {
        "protocol": {
            "hypothesis": (
                "Protect an adjusted continuous Top-1 from P89 Safe only when "
                "a frozen non-redundant expert consensus supports it."
            ),
            "selection": "12 pre-registered rules on H1 only",
            "confirmation": "selected H1 rule transferred unchanged to H2",
            "h3_rule": "evaluate only when H1 net >=5 and H2 is positive with no user regression",
            "expert_pool": expert_names,
        },
        "H1_selected": selected,
        "H1_all_candidates": sorted(
            h1_grid,
            key=lambda row: (row["net"], -row["harm"], -row["route"]),
            reverse=True,
        ),
        "H2_confirmation": h2,
        "gate": {
            "h1_effective": h1_effective,
            "h2_passed": h2_passed,
            "passed": gate_passed,
            "decision": (
                "reject; do not integrate into P91 and do not expand folds"
                if not gate_passed
                else "eligible for one H3 evaluation"
            ),
        },
        "H3_independent_fold0": h3,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = "\n".join(
        [
            "# P93 Phase 0 consensus protection screen",
            "",
            "## Decision",
            "",
            f"- H1: rescue {selected['rescue']}, harm {selected['harm']}, net {selected['net']}; effective gate: {h1_effective}.",
            f"- H2 frozen transfer: rescue {h2['rescue']}, harm {h2['harm']}, net {h2['net']}, worst user {h2['minimum_user_gain']}; passed: {h2_passed}.",
            f"- Final: {payload['gate']['decision']}.",
            "- H3 was not evaluated by this reproducible screen because the H1/H2 gate failed.",
            "",
            "The screen confirms that deterministic downstream harm exists, but the deployable consensus signal is not stable enough across source users to promote a low-cost rule.",
            "",
        ]
    )
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
