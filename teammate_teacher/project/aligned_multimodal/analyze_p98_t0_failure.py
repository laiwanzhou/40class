"""Mechanism audit for the source-only P98-T0 H1 OOF result.

The audit is intentionally limited to the saved H1 OOF predictions.  It does
not expose or load H2/H3 and it does not select a deployment threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_RUN = PROJECT / "runs/p98_four_modal_teacher_h1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    with np.load(run / "h1_oof_predictions.npz", allow_pickle=False) as source:
        labels = source["labels"].astype(np.int64)
        users = source["users"].astype(str)
        base = source["base_prediction"].astype(np.int64)
        teacher = source["direct_logits"].argmax(axis=1).astype(np.int64)
        reliability = source["reliability_logits"].astype(np.float64)
        modality = {
            key.removesuffix("_logits"): source[key].argmax(axis=1).astype(np.int64)
            for key in source.files
            if key.endswith("_logits")
            and key not in {"direct_logits", "reliability_logits"}
        }
    base_correct = base == labels
    teacher_correct = teacher == labels
    base_wrong = ~base_correct
    reliability_auc = float(roc_auc_score(base_wrong, reliability))
    reliability_ap = float(average_precision_score(base_wrong, reliability))
    result: dict[str, Any] = {
        "stage": "P98-T0 H1 failure attribution",
        "scope": "saved H1 OOF only; H2/H3 unread",
        "samples": int(len(labels)),
        "base": {
            "correct": int(base_correct.sum()),
            "accuracy": float(base_correct.mean()),
        },
        "teacher": {
            "correct": int(teacher_correct.sum()),
            "accuracy": float(teacher_correct.mean()),
            "rescue": int((base_wrong & teacher_correct).sum()),
            "harm": int((base_correct & ~teacher_correct).sum()),
            "net": int(teacher_correct.sum() - base_correct.sum()),
            "accuracy_inside_base_error_pool": float(teacher_correct[base_wrong].mean()),
            "accuracy_inside_base_correct_pool": float(teacher_correct[base_correct].mean()),
        },
        "label_oracle": {
            "correct": int((base_correct | teacher_correct).sum()),
            "accuracy": float((base_correct | teacher_correct).mean()),
        },
        "reliability": {
            "target": "base_prediction_is_wrong",
            "prevalence": float(base_wrong.mean()),
            "roc_auc": reliability_auc,
            "average_precision": reliability_ap,
            "mean_logit_base_wrong": float(reliability[base_wrong].mean()),
            "mean_logit_base_correct": float(reliability[base_correct].mean()),
        },
        "modality_heads": {
            name: {
                "correct": int((prediction == labels).sum()),
                "accuracy": float((prediction == labels).mean()),
            }
            for name, prediction in modality.items()
        },
        "per_user": {},
        "seed_range_by_user": {},
        "attribution": {
            "positive_evidence": (
                "Fusion rescues non-zero anchor errors and exceeds every standalone "
                "modality head, so multimodal complementarity is present."
            ),
            "primary_failure": (
                "The unconstrained classifier relearns all 40 decisions and destroys "
                "far more correct anchor decisions than it rescues."
            ),
            "reliability_boundary": (
                "The OOF reliability head is near random and therefore cannot justify "
                "a threshold router."
            ),
            "structural_gap": (
                "T0 consumes frozen representation tokens but omits the OOF 40-class "
                "expert-distribution pathway used by P91."
            ),
            "required_revision": (
                "Restore modality-owned expert-distribution tokens and learn a bounded "
                "residual around an exact source-safe fallback."
            ),
        },
    }
    for user in sorted(np.unique(users).tolist()):
        selected = users == user
        result["per_user"][user] = {
            "samples": int(selected.sum()),
            "base_correct": int(base_correct[selected].sum()),
            "teacher_correct": int(teacher_correct[selected].sum()),
            "net": int(teacher_correct[selected].sum() - base_correct[selected].sum()),
            "rescue": int((base_wrong[selected] & teacher_correct[selected]).sum()),
            "harm": int((base_correct[selected] & ~teacher_correct[selected]).sum()),
        }
    for user in sorted(np.unique(users).tolist()):
        values = [
            float(item["best_accuracy"])
            for item in summary["seed_audits"]
            if item["held_user"] == user
        ]
        result["seed_range_by_user"][user] = {
            "minimum": float(min(values)),
            "maximum": float(max(values)),
            "range": float(max(values) - min(values)),
        }
    return result


def main() -> None:
    args = parse_args()
    result = audit(args)
    output = args.run.resolve() / "mechanism_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {
        "base": result["base"],
        "teacher": result["teacher"],
        "label_oracle": result["label_oracle"],
        "reliability": result["reliability"],
        "modality_heads": result["modality_heads"],
        "per_user": result["per_user"],
        "seed_range_by_user": result["seed_range_by_user"],
        "attribution": result["attribution"],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
