"""Strict P90 residual-teacher audit on the frozen P89 safe pipeline.

Candidate temperatures and residual weights are selected on P89 H1 only, with
the additional requirement that no held-out H1 user loses correct predictions.
The selected configuration is then transferred unchanged to H2 and the
independent original fold-0 H3.  All candidate probabilities are subject-
disjoint OOF predictions; label oracles are reported only as diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm
from p89_imu_probability_blend import evaluate
from p89_supported_template_gate import (
    h3_protocol,
    load_grouping,
    load_imu,
    safe_probability_and_prediction,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUTPUT = REPO_ROOT / "runs" / "p90_teacher_fusion_audit_v1"
WEIGHTS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def load_probability(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as source:
        sample_ids = source["sample_ids"].astype(str)
        if "probabilities" in source.files:
            probability = np.asarray(source["probabilities"], dtype=np.float64)
        else:
            probability = softmax(np.asarray(source["logits"], dtype=np.float64), axis=1)
    return sample_ids, probability


def align(
    source_ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray
) -> np.ndarray:
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids.astype(str))}
    if len(lookup) != len(source_ids):
        raise ValueError("duplicate source sample id")
    try:
        order = np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"candidate misses sample {error.args[0]}") from error
    return np.asarray(values)[order]


def temper(probability: np.ndarray, temperature: float) -> np.ndarray:
    values = np.log(np.clip(probability, 1e-10, 1.0)) / temperature
    return softmax(values, axis=1)


def users_audit(
    labels: np.ndarray,
    safe: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
) -> dict[str, dict[str, int]]:
    result = {}
    for user in sorted(set(users.astype(str).tolist())):
        selected = users.astype(str) == user
        safe_correct = int(np.sum(safe[selected] == labels[selected]))
        candidate_correct = int(np.sum(prediction[selected] == labels[selected]))
        result[user] = {
            "rows": int(np.sum(selected)),
            "safe_correct": safe_correct,
            "candidate_correct": candidate_correct,
            "gain": candidate_correct - safe_correct,
        }
    return result


def candidate_report(
    protocol_value: tuple[Any, ...],
    safe: np.ndarray,
    probability: np.ndarray,
    weight: float,
    temperature: float,
    grouping: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    # The residual starts from P89's already IMU-adjusted probability, while
    # retaining the original P87 initial prediction.  P89 safe itself is the
    # result of running joint_decode once from that exact state.  Replacing the
    # initial prediction with ``safe`` would decode a second time and would not
    # reproduce the baseline at weight zero.
    safe_protocol = list(protocol_value)
    result, prediction = evaluate(
        tuple(safe_protocol),
        temper(probability, temperature),
        np.ones(len(safe), dtype=bool),
        weight,
        "joint",
        grouping,
    )
    by_user = users_audit(
        protocol_value[1], safe, prediction, protocol_value[4].users
    )
    result.update(
        {
            "configuration": {
                "weight": weight,
                "temperature": temperature,
                "method": "joint",
            },
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_p89_safe": rescue_harm(
                protocol_value[1], safe, prediction
            ),
            "per_user_vs_p89_safe": by_user,
            "minimum_user_gain_vs_p89_safe": min(
                row["gain"] for row in by_user.values()
            ),
        }
    )
    return result, prediction


def select_h1(
    protocol_value: tuple[Any, ...],
    safe: np.ndarray,
    probability: np.ndarray,
    grouping: Any,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    candidates = []
    predictions = []
    for temperature in TEMPERATURES:
        for weight in WEIGHTS:
            report, prediction = candidate_report(
                protocol_value, safe, probability, weight, temperature, grouping
            )
            candidates.append(report)
            predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain_vs_p89_safe"] >= 0,
            candidates[index]["metrics"]["correct"],
            candidates[index]["metrics"]["balanced_accuracy"],
            candidates[index]["rescue_harm_vs_p89_safe"]["net"],
            -candidates[index]["rescue_harm_vs_p89_safe"]["harm"],
            -candidates[index]["configuration"]["weight"],
        ),
        reverse=True,
    )
    best = order[0]
    return candidates[best], predictions[best], [candidates[index] for index in order]


def oracle(labels: np.ndarray, safe: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    candidate = probability.argmax(axis=1)
    safe_correct = safe == labels
    candidate_correct = candidate == labels
    return {
        "candidate_top1_accuracy": float(candidate_correct.mean()),
        "safe_wrong_candidate_correct": int(np.sum(~safe_correct & candidate_correct)),
        "safe_correct_candidate_wrong": int(np.sum(safe_correct & ~candidate_correct)),
        "label_oracle_accuracy": float(np.mean(safe_correct | candidate_correct)),
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

    paths = {
        "imu_p90_crossfit_blend": REPO_ROOT
        / "runs/p90_imu_teacher_blend_v1/imu_p90_sensorwise_plus_deep_crossfit_oof.npz",
        "imu_deep_scratch": REPO_ROOT
        / "runs/p90_imu_ssl_teacher_v1/imu_hart128_scratch_oof.npz",
        "imu_deep_maskedssl": REPO_ROOT
        / "runs/p90_imu_ssl_teacher_v1/imu_hart128_maskedssl_oof.npz",
        "motionbert_full_front": REPO_ROOT
        / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front_linear_oof.npz",
        "motionbert_full_multiview": REPO_ROOT
        / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front-side-top_linear_oof.npz",
        "motionbert_action_multiview": REPO_ROOT
        / "runs/p90_motionbert_teacher_v1/motionbert_action_front-side-top_linear_oof.npz",
    }
    sources = {name: load_probability(path) for name, path in paths.items()}
    reference_ids = next(iter(sources.values()))[0]
    if any(not np.array_equal(ids, reference_ids) for ids, _ in sources.values()):
        raise ValueError("P90 candidate OOF orders differ")
    base_probabilities = {name: value[1] for name, value in sources.items()}
    base_probabilities["imu_deep_mean"] = 0.5 * (
        base_probabilities["imu_deep_scratch"]
        + base_probabilities["imu_deep_maskedssl"]
    )
    base_probabilities["imu_deep_plus_motionbert_front"] = 0.5 * (
        base_probabilities["imu_deep_mean"]
        + base_probabilities["motionbert_full_front"]
    )
    base_probabilities["motionbert_front_plus_multiview"] = 0.5 * (
        base_probabilities["motionbert_full_front"]
        + base_probabilities["motionbert_full_multiview"]
    )

    report: dict[str, Any] = {
        "stage": "P90_teacher_residual_fusion_audit_v1",
        "protocol": (
            "P89 safe probability is frozen. Temperature/weight are selected on H1 "
            "with no H1 user regression and transferred unchanged to H2 and H3."
        ),
        "safe_metrics": {
            split: classification_metrics(protocol[1], safe)
            for split, (protocol, safe) in splits.items()
        },
        "candidates": {},
    }
    saved: dict[str, np.ndarray] = {}
    h1_ranked: list[tuple[str, dict[str, Any]]] = []
    for name, full_probability in base_probabilities.items():
        aligned = {
            split: align(reference_ids, full_probability, protocol[0])
            for split, (protocol, _) in splits.items()
        }
        selected, h1_prediction, ranked = select_h1(
            h1, h1_safe, aligned["H1_selection"], grouping
        )
        configuration = selected["configuration"]
        candidate: dict[str, Any] = {
            "H1_selected": selected,
            "oracle": {
                split: oracle(protocol[1], safe, aligned[split])
                for split, (protocol, safe) in splits.items()
            },
            "selection_grid_size": len(ranked),
            "top_h1_configurations": ranked[:5],
        }
        saved[f"{name}_h1_prediction"] = h1_prediction
        for split in ("H2_confirmation", "H3_independent_fold0"):
            protocol, safe = splits[split]
            confirmation, prediction = candidate_report(
                protocol,
                safe,
                aligned[split],
                float(configuration["weight"]),
                float(configuration["temperature"]),
                grouping,
            )
            candidate[split] = confirmation
            saved[f"{name}_{split.lower()}_prediction"] = prediction
        report["candidates"][name] = candidate
        h1_ranked.append((name, selected))

    h1_ranked.sort(
        key=lambda item: (
            item[1]["minimum_user_gain_vs_p89_safe"] >= 0,
            item[1]["metrics"]["correct"],
            item[1]["metrics"]["balanced_accuracy"],
        ),
        reverse=True,
    )
    report["H1_candidate_order"] = [name for name, _ in h1_ranked]

    # LoRA exists only for fold0 and therefore cannot enter H1 selection.  Keep
    # it as an H3-only upper-bound/complementarity diagnostic.
    lora_path = REPO_ROOT / "runs/p90_videomae_lora_teacher_v1/videomae_large_ir_lora_r8_partial.npz"
    lora_ids, lora_probability = load_probability(lora_path)
    lora_h3 = align(lora_ids, lora_probability, h3[0])
    report["lora_h3_only_oracle"] = oracle(h3[1], h3_safe, lora_h3)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "selected_predictions.npz", **saved)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = {
        "safe_metrics": report["safe_metrics"],
        "H1_candidate_order": report["H1_candidate_order"],
        "selected": {
            name: {
                "H1": candidate["H1_selected"],
                "H2": candidate["H2_confirmation"],
                "H3": candidate["H3_independent_fold0"],
            }
            for name, candidate in report["candidates"].items()
        },
        "lora_h3_only_oracle": report["lora_h3_only_oracle"],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
