"""Post-hoc, label-audit-only decomposition of frozen P117 outer routes.

No model or threshold is refit.  Each candidate is applied alone using the exact
outer score and threshold already produced by the no-session P117 run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import load_candidate_splits


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/p117_multicandidate_no_session_ablation_v1"
OUTPUT = HERE / "runs/p117_candidate_route_ablation_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--full-visual-bank", action="store_true")
    parser.add_argument("--structured-bank", action="store_true")
    parser.add_argument("--legacy-visual-bank", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    output_dir = args.output.resolve()
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    predictions = np.load(source / "predictions.npz")
    data = load_candidate_splits(
        full_visual_bank=args.full_visual_bank,
        structured_bank=args.structured_bank,
        legacy_visual_bank=args.legacy_visual_bank,
    )
    report: dict[str, object] = {
        "stage": "P117_frozen_candidate_route_ablation_v1",
        "protocol": (
            "No refit and no new threshold selection. Apply each candidate alone with "
            "the frozen outer score/threshold from p117_multicandidate_no_session_ablation_v1."
        ),
        "test_rows_loaded": 0,
        "test_labels_loaded": 0,
        "submission_generated": False,
        "candidates": {},
    }
    first_cohort = next(iter(summary["cohorts"].values()))
    for candidate_name in first_cohort["thresholds"]:
        cohorts = {}
        total_correct = total_safe = total_rescue = total_harm = total_changed = 0
        for split_name, value in data.items():
            split = value.split
            threshold = float(
                summary["cohorts"][split_name]["thresholds"][candidate_name]["threshold"]
            )
            score = predictions[f"{split_name}_{candidate_name}_route_score"]
            candidate = value.candidates[candidate_name].argmax(axis=1)
            route = (candidate != split.safe_prediction) & (score >= threshold)
            output = split.safe_prediction.copy()
            output[route] = candidate[route]
            rescue = int(
                np.sum((output == split.labels) & (split.safe_prediction != split.labels))
            )
            harm = int(
                np.sum((output != split.labels) & (split.safe_prediction == split.labels))
            )
            per_user = {}
            for user in sorted(set(split.users.astype(str).tolist())):
                selected = split.users.astype(str) == user
                per_user[user] = int(
                    np.sum(output[selected] == split.labels[selected])
                    - np.sum(split.safe_prediction[selected] == split.labels[selected])
                )
            metrics = classification_metrics(split.labels, output)
            safe_correct = int(np.sum(split.safe_prediction == split.labels))
            cohorts[split_name] = {
                "threshold": threshold,
                "metrics": metrics,
                "safe_correct": safe_correct,
                "net": metrics["correct"] - safe_correct,
                "route_count": int(route.sum()),
                "rescue": rescue,
                "harm": harm,
                "minimum_user_gain": int(min(per_user.values())),
                "per_user_gain": per_user,
            }
            total_correct += metrics["correct"]
            total_safe += safe_correct
            total_rescue += rescue
            total_harm += harm
            total_changed += int(route.sum())
        report["candidates"][candidate_name] = {
            "cohorts": cohorts,
            "aggregate": {
                "rows": 2470,
                "correct": total_correct,
                "accuracy": total_correct / 2470,
                "safe_correct": total_safe,
                "net": total_correct - total_safe,
                "route_count": total_changed,
                "rescue": total_rescue,
                "harm": total_harm,
            },
        }
    candidates = report["candidates"]
    champion = max(
        candidates,
        key=lambda name: candidates[name]["aggregate"]["correct"],
    )
    report["decision"] = {
        "champion": champion,
        "correct": candidates[champion]["aggregate"]["correct"],
        "accuracy": candidates[champion]["aggregate"]["accuracy"],
        "note": (
            "A18 remains in the shared router feature matrix, but only the P90 visual "
            "label is authorized to replace P89 safe."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
