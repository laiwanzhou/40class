"""All-Train/Test counterpart of P347, with no submission side effect."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from p117_transductive_multicandidate_router import load_candidate_splits
from p191_source_truth_calibrated_p150_distillation import choose
from p346_distributionally_robust_rescue_harm_gate import (
    NAMES,
    P244,
    P255,
    P278,
    P279,
    P306,
    P307,
    P310,
    S,
    cat,
    part,
    scalar,
    score,
    train_pair,
)
from p347_loso_twohead_gate import subset


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "p357_full_nested_gate_test_audit_v1"


def align(values, source_ids, target_ids):
    lookup = {value: index for index, value in enumerate(np.asarray(source_ids).astype(str))}
    return np.asarray(values)[np.asarray([lookup[value] for value in np.asarray(target_ids).astype(str)])]


def softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def p87s_probability(target_ids):
    run = RUNS / "p87s_final_test_predictions_v1"
    with (run / "prediction_audit.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        ids = np.asarray([row["sample_id"] for row in csv.DictReader(handle)])
    probability = softmax(np.load(run / "base_student_logits_audit_only.npy"))
    return align(probability, ids, target_ids)


def test_part():
    p244 = np.load(RUNS / "p244_dual_physical_group_v1/predictions.npz")
    ids = p244["sample_ids"].astype(str)
    p255 = np.load(RUNS / "p255_repeat_augmented_physical_group_v1/predictions.npz")
    p308 = np.load(RUNS / "p308_union_repeat_physical_test_v1/test_predictions.npz")
    p309 = np.load(RUNS / "p309_union_repeat_group_test_v1/predictions.npz")
    p278 = np.load(RUNS / "p278_centroid_augmented_group_v1/predictions.npz")
    p279 = np.load(RUNS / "p279_p278_fixed_emission07_transition04_v1/predictions.npz")
    sig_ridge = np.load(RUNS / "p349_siglip2_ridge_test_v1/test_predictions.npz")
    sig_temporal = np.load(RUNS / "p356_siglip2_threeview_temporal_test_v1/test_predictions.npz")
    p310 = np.load(RUNS / "p310_union_repeat_precedence_teacher_v1/student_test_targets.npz")

    base = align(p310["emission_prediction"], p310["sample_ids"], ids).astype(int)
    base_probs = (
        p244["probability"].astype(float),
        align(p309["probability"], p309["sample_ids"], ids).astype(float),
        align(p255["probability"], p255["sample_ids"], ids).astype(float),
        align(p308["probability"], p308["sample_ids"], ids).astype(float),
    )
    candidate_probs = (
        p87s_probability(ids),
        align(sig_ridge["probability"], sig_ridge["sample_ids"], ids).astype(float),
        align(sig_temporal["probability"], sig_temporal["sample_ids"], ids).astype(float),
        align(p278["probability"], p278["sample_ids"], ids).astype(float),
        align(p309["probability"], p309["sample_ids"], ids).astype(float),
    )
    candidate = np.stack(
        (
            candidate_probs[0].argmax(1),
            candidate_probs[1].argmax(1),
            candidate_probs[2].argmax(1),
            align(p279["prediction"], p279["sample_ids"], ids).astype(int),
            align(p309["prediction"], p309["sample_ids"], ids).astype(int),
        ),
        axis=1,
    )
    rows = []
    for index, probability in enumerate(candidate_probs):
        proposed = candidate[:, index]
        vote = (candidate == proposed[:, None]).mean(1)
        features = np.concatenate(
            [
                *(np.log(np.clip(value, 1e-7, 1)) for value in (*base_probs, probability)),
                *(scalar(value, base, proposed) for value in (*base_probs, probability)),
                vote[:, None],
                np.eye(len(NAMES))[np.full(len(ids), index)],
                np.eye(40)[base],
                np.eye(40)[proposed],
            ],
            axis=1,
        )
        rows.append(features)
    values = np.stack(rows, axis=1).astype(np.float32)
    return {
        "ids": ids,
        "x": values,
        "base": base,
        "candidate": candidate,
        "disagree": candidate != base[:, None],
    }


def main() -> None:
    data = load_candidate_splits()
    z = {
        "p310": np.load(P310),
        "p244": np.load(P244),
        "p307": np.load(P307),
        "p255": np.load(P255),
        "p306": np.load(P306),
        "p328": np.load(RUNS / "p328_p87s_threefold_oof_audit_v1/oof_predictions.npz"),
        "p336": np.load(RUNS / "p336_siglip2_workspace_state_ridge_v1/oof_predictions.npz"),
        "p344": np.load(RUNS / "p344_siglip2_threeview_temporal_oof_v1/oof_predictions.npz"),
        "p279": np.load(P279),
        "p278": np.load(P278),
    }
    parts = {name: part(data, name, z) for name in S}
    source = cat([parts[name] for name in S])
    cohorts = np.concatenate([np.full(len(parts[name]["labels"]), name, object) for name in S])
    best = None
    for c_value in (0.003, 0.01, 0.03, 0.1):
        proposal = np.zeros(len(source["labels"]), dtype=int)
        robust_score = np.full(len(proposal), -np.inf)
        for user in sorted(set(source["users"].tolist())):
            held = source["users"] == user
            trained = ~held
            predicted, score_value = score([train_pair(subset(source, trained), c_value)], subset(source, held))
            proposal[held] = predicted
            robust_score[held] = score_value
        gain = (proposal == source["labels"]).astype(int) - (source["base"] == source["labels"]).astype(int)
        selected = choose(robust_score, gain, proposal != source["base"], source["users"], cohorts)
        key = (
            selected["minimum_user_gain"] >= 0,
            selected["minimum_cohort_gain"] >= 0,
            selected["net"],
            selected["rescue"],
            -selected["harm"],
            -selected["changed"],
            -c_value,
        )
        candidate = (key, c_value, selected)
        if best is None or candidate[0] > best[0]:
            best = candidate

    c_value, selected = best[1], best[2]
    test = test_part()
    proposal, robust_score = score([train_pair(source, c_value)], test)
    route = (proposal != test["base"]) & (robust_score >= selected["threshold"])
    prediction = test["base"].copy()
    prediction[route] = proposal[route]
    changed_rows = np.flatnonzero(route).tolist()
    frozen_rows = [row for row in changed_rows if row in (77, 283)]
    report = {
        "stage": "P357_full_nested_gate_Test_audit",
        "status": "safe" if not frozen_rows else "rejected_frozen_row_collision",
        "protocol": {
            "oof_selector": "leave-one-subject-out over all 2470 OOF rows",
            "candidate_pool": list(NAMES),
            "test_labels_read": False,
            "user_id_used_as_feature": False,
            "submission_generated": False,
        },
        "selection": {"C": c_value, **selected},
        "test": {
            "rows": len(prediction),
            "changes": len(changed_rows),
            "changed_rows_zero_based": changed_rows,
            "changed_pairs": [f"{test['base'][row]}->{prediction[row]}" for row in changed_rows],
            "frozen_row_collisions": frozen_rows,
            "test_labels_read": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "test_predictions.npz", sample_ids=test["ids"], base_prediction=test["base"], prediction=prediction, proposal=proposal, robust_score=robust_score, route=route)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
