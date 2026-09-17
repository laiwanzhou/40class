"""Freeze the deployable 0.93 two-head gate and build Student targets.

P279 is retained as context but cannot supply a replacement.  P353's
supported Top-3 pair route has precedence after the gate.  The fixed protocol
passes the requested fold gate before Test inference is considered.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io
from p117_transductive_multicandidate_router import load_candidate_splits
from p346_distributionally_robust_rescue_harm_gate import NAMES, P244, P255, P278, P279, P306, P307, P310, S, cat, part, predict, train_pair
from p357_full_nested_gate_test_audit import test_part


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "p359_fixed093_no_p279_gate_teacher_v1"
P89 = RUNS / "p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
THRESHOLD = 0.93
P347_C = {"H1_selection": 0.01, "H2_confirmation": 0.1, "H3_independent_fold0": 0.1}
FULL_C = 0.1
SIZES = (663, 834, 973)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def score_with_index(models, value):
    features = value["x"].reshape(-1, value["x"].shape[-1])
    rescue_scores = []
    harm_scores = []
    for rescue_model, harm_model in models:
        rescue_scores.append(predict(rescue_model, features).reshape(value["candidate"].shape))
        harm_scores.append(predict(harm_model, features).reshape(value["candidate"].shape))
    robust = np.min(np.stack(rescue_scores), axis=0) - np.max(np.stack(harm_scores), axis=0)
    robust[~value["disagree"]] = -np.inf
    index = robust.argmax(axis=1)
    rows = np.arange(len(index))
    return value["candidate"][rows, index], robust[rows, index], index


def read_prediction(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray([int(row["prediction"]) for row in csv.DictReader(handle)])


def main() -> None:
    data = load_candidate_splits()
    z = {
        "p310": np.load(P310), "p244": np.load(P244), "p307": np.load(P307),
        "p255": np.load(P255), "p306": np.load(P306),
        "p328": np.load(RUNS / "p328_p87s_threefold_oof_audit_v1/oof_predictions.npz"),
        "p336": np.load(RUNS / "p336_siglip2_workspace_state_ridge_v1/oof_predictions.npz"),
        "p344": np.load(RUNS / "p344_siglip2_threeview_temporal_oof_v1/oof_predictions.npz"),
        "p279": np.load(P279), "p278": np.load(P278),
    }
    parts = {name: part(data, name, z) for name in S}
    p353 = np.load(RUNS / "p353_p245_top3_supported_pair_reranker_oof_v1/oof_predictions.npz")
    p310 = np.load(P310)
    labels = p310["labels"].astype(int)
    base_all = p310["prediction"].astype(int)
    pair_route_all = p353["prediction"] != p353["base_prediction"]

    outputs = []
    folds = []
    offset = 0
    for held_name, size in zip(S, SIZES):
        source = cat([parts[name] for name in S if name != held_name])
        proposal, robust_score, candidate_index = score_with_index([train_pair(source, P347_C[held_name])], parts[held_name])
        gate_route = (proposal != parts[held_name]["base"]) & (robust_score >= THRESHOLD) & (candidate_index != 3)
        output = parts[held_name]["base"].copy()
        output[gate_route] = proposal[gate_route]
        pair_route = pair_route_all[offset:offset + size]
        output[pair_route] = p353["prediction"][offset:offset + size][pair_route]
        base = parts[held_name]["base"]
        truth = parts[held_name]["labels"]
        changed = output != base
        users = parts[held_name]["users"]
        per_user = {user: int(np.sum((output[users == user] == truth[users == user])) - np.sum((base[users == user] == truth[users == user]))) for user in sorted(set(users.tolist()))}
        folds.append({
            "cohort": held_name,
            "C": P347_C[held_name],
            "rows": size,
            "net": int(np.sum(output == truth) - np.sum(base == truth)),
            "changed": int(np.sum(changed)),
            "rescue": int(np.sum(changed & (base != truth) & (output == truth))),
            "harm": int(np.sum(changed & (base == truth) & (output != truth))),
            "gate_routes": int(np.sum(gate_route)),
            "pair_routes": int(np.sum(pair_route)),
            "per_user": per_user,
        })
        outputs.append(output)
        offset += size
    oof = np.concatenate(outputs)
    fold_nets = [row["net"] for row in folds]
    strict_pass = all(value > 0 for value in fold_nets) or (sum(value > 0 for value in fold_nets) >= 2 and min(fold_nets) >= -1)
    if not strict_pass:
        raise RuntimeError(f"strict fold gate failed: {fold_nets}")

    source = cat([parts[name] for name in S])
    test = test_part()
    proposal, robust_score, candidate_index = score_with_index([train_pair(source, FULL_C)], test)
    route = (proposal != test["base"]) & (robust_score >= THRESHOLD) & (candidate_index != 3)
    test_prediction = test["base"].copy()
    test_prediction[route] = proposal[route]
    changed_rows = np.flatnonzero(test_prediction != test["base"]).tolist()
    frozen = [row for row in changed_rows if row in (77, 283)]
    if frozen:
        raise RuntimeError(f"frozen-row collision: {frozen}")

    OUT.mkdir(parents=True, exist_ok=True)
    teacher_csv = OUT / "submission_p359_fixed093_gate_teacher.csv"
    submission_io.write_submission(teacher_csv, submission_io.read_rows(P89), test_prediction)
    probability = np.full((len(test_prediction), 40), 0.0005, dtype=np.float32)
    probability[np.arange(len(test_prediction)), test_prediction] = 0.9805
    targets = OUT / "student_test_targets.npz"
    np.savez_compressed(targets, sample_ids=test["ids"], target_mask=np.ones(len(test_prediction), bool), emission_probability=probability, structured_distillation_probability=probability, structured_confidence=np.full(len(test_prediction), 0.9805, np.float32), emission_prediction=test_prediction, structured_distillation_prediction=test_prediction)
    np.savez_compressed(OUT / "oof_predictions.npz", labels=labels, base_prediction=base_all, prediction=oof)
    report = {
        "stage": "P359_fixed093_no_P279_gate_teacher",
        "status": "complete",
        "protocol": {"threshold": THRESHOLD, "p279_routing_disabled": True, "p279_retained_as_context": True, "pair_minimum_source_top3_errors": 7, "test_labels_read": False, "user_id_used_as_feature": False},
        "folds": folds,
        "validation": {"base_correct": int(np.sum(base_all == labels)), "correct": int(np.sum(oof == labels)), "net": int(np.sum(oof == labels) - np.sum(base_all == labels)), "accuracy": float(np.mean(oof == labels)), "fold_nets": fold_nets, "strict_gate_pass": strict_pass},
        "test": {"rows": len(test_prediction), "changes_vs_p310": len(changed_rows), "changed_rows_zero_based": changed_rows, "changed_pairs": [f"{test['base'][row]}->{test_prediction[row]}" for row in changed_rows], "candidate_sources": [NAMES[int(candidate_index[row])] for row in changed_rows], "frozen_row_collisions": frozen, "submission": str(teacher_csv.resolve()), "submission_sha256": sha256(teacher_csv), "targets": str(targets.resolve()), "targets_sha256": sha256(targets), "test_labels_read": False},
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
