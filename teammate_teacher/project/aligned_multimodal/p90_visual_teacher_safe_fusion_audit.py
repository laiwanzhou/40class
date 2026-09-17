"""Strict P90 visual-teacher residual audit on the frozen P89 safe pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import log_softmax, softmax

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p89_supported_template_gate import (
    h3_protocol,
    load_grouping,
    load_imu,
    safe_probability_and_prediction,
)
from p90_teacher_fusion_audit import (
    align,
    candidate_report,
    oracle,
    select_h1,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUTPUT = REPO_ROOT / "runs/p90_visual_teacher_safe_fusion_audit_v1"
BASE_OOF = REPO_ROOT / "runs/p90_videomaev2_distilled_teacher_v1/oof_logits.npz"
IV2_OOF = REPO_ROOT / "runs/p90_internvideo2_l_k400_teacher_v1/oof_logits.npz"


def load_visual_candidates() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(BASE_OOF, allow_pickle=False) as source:
        sample_ids = source["sample_ids"].astype(str)
        base_logits = np.asarray(source["early_late_logits"], dtype=np.float64)
    with np.load(IV2_OOF, allow_pickle=False) as source:
        if not np.array_equal(sample_ids, source["sample_ids"].astype(str)):
            raise ValueError("P90 visual OOF orders differ")
        iv2_early_late = np.asarray(source["early_late_logits"], dtype=np.float64)
        iv2_joint = np.asarray(
            source["early_late_plus_k400_logits"], dtype=np.float64
        )
    base_logp = log_softmax(base_logits, axis=1)
    iv2_logp = log_softmax(iv2_joint, axis=1)
    return sample_ids, {
        "videomaev2_distilled_base": softmax(base_logits, axis=1),
        "internvideo2_l_early_late": softmax(iv2_early_late, axis=1),
        "internvideo2_l_early_late_plus_k400": softmax(iv2_joint, axis=1),
        "videomaev2_base_plus_internvideo2_l_equal": softmax(
            0.5 * base_logp + 0.5 * iv2_logp, axis=1
        ),
    }


def main() -> None:
    grouping = load_grouping()
    old_imu_ids, old_imu_logits = load_imu()
    h1_raw = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2_raw = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_adjusted, h1_safe = safe_probability_and_prediction(
        h1_raw, old_imu_ids, old_imu_logits, grouping
    )
    h2_adjusted, h2_safe = safe_probability_and_prediction(
        h2_raw, old_imu_ids, old_imu_logits, grouping
    )
    h1 = list(h1_raw)
    h2 = list(h2_raw)
    h1[2] = h1_adjusted
    h2[2] = h2_adjusted
    h1 = tuple(h1)
    h2 = tuple(h2)
    h3_values = h3_protocol(old_imu_ids, old_imu_logits, grouping)
    h3_raw, h3_adjusted, h3_safe = h3_values[:3]
    h3 = list(h3_raw)
    h3[2] = h3_adjusted
    h3 = tuple(h3)
    splits = {
        "H1_selection": (h1, h1_safe),
        "H2_confirmation": (h2, h2_safe),
        "H3_independent_fold0": (h3, h3_safe),
    }

    reference_ids, probabilities = load_visual_candidates()
    report: dict[str, Any] = {
        "stage": "P90_visual_teacher_safe_residual_audit_v1",
        "protocol": (
            "Frozen P89 safe pipeline. Residual temperature/weight selected on H1 "
            "with no H1 user regression, then transferred unchanged to H2/H3."
        ),
        "safe_metrics": {
            split: classification_metrics(protocol[1], safe)
            for split, (protocol, safe) in splits.items()
        },
        "candidates": {},
    }
    saved: dict[str, np.ndarray] = {}
    for name, full_probability in probabilities.items():
        split_probability = {
            split: align(reference_ids, full_probability, protocol[0])
            for split, (protocol, _) in splits.items()
        }
        selected, h1_prediction, ranked = select_h1(
            h1, h1_safe, split_probability["H1_selection"], grouping
        )
        configuration = selected["configuration"]
        candidate: dict[str, Any] = {
            "H1_selected": selected,
            "oracle": {
                split: oracle(protocol[1], safe, split_probability[split])
                for split, (protocol, safe) in splits.items()
            },
            "top_h1_configurations": ranked[:5],
        }
        saved[f"{name}_h1_prediction"] = h1_prediction
        for split in ("H2_confirmation", "H3_independent_fold0"):
            protocol, safe = splits[split]
            confirmation, prediction = candidate_report(
                protocol,
                safe,
                split_probability[split],
                float(configuration["weight"]),
                float(configuration["temperature"]),
                grouping,
            )
            candidate[split] = confirmation
            saved[f"{name}_{split.lower()}_prediction"] = prediction
        report["candidates"][name] = candidate

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "selected_predictions.npz", **saved)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = {
        "safe_metrics": report["safe_metrics"],
        "selected": {
            name: {
                split: (
                    candidate["H1_selected"] if split == "H1_selection" else candidate[split]
                )
                for split in ("H1_selection", "H2_confirmation", "H3_independent_fold0")
            }
            for name, candidate in report["candidates"].items()
        },
        "oracle": {
            name: candidate["oracle"]
            for name, candidate in report["candidates"].items()
        },
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
