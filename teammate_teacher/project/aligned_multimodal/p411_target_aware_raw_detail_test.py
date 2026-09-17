"""Matched unlabeled Test deployment of the P410 target-aware raw-detail head."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import p399_candidate_conditioned_raw_detail_head_oof as p399
import p400_candidate_conditioned_raw_detail_test as p400
import p404_group_guarded_raw_detail_oof as p404
import p410_target_aware_weighted_raw_detail_oof as p410


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p411_target_aware_raw_detail_test_v1"


def main():
    print(
        "P411 uses unlabeled Test covariates to importance-weight all source OOF rows, then "
        "runs the P410 raw-detail head with thresholds selected by target-aware LOSO.",
        flush=True,
    )
    source = p400.train_source()
    best = None
    for k in p399.KS:
        for c_value in p399.CS:
            proposal = source["base"].copy()
            margin = np.full(len(proposal), -np.inf, dtype=float)
            for user in np.unique(source["users"]):
                validation = source["users"] == user
                train_part = p399.subset(source, ~validation)
                target_part = p399.subset(source, validation)
                weights, _ = p410.domain_weights(train_part, target_part)
                model = p399.fit_model(train_part, k, c_value, sample_weight=weights)
                proposal[validation], margin[validation] = p399.predict(model, target_part, k)
            threshold = p404.choose(source, proposal, margin)
            key = (
                threshold["minimum_cohort_gain"] >= 0,
                threshold["net"],
                threshold["rescue"],
                -threshold["harm"],
                -threshold["changed"],
                -k,
                -c_value,
            )
            if best is None or key > best[0]:
                best = (key, k, c_value, threshold)
    _, k, c_value, threshold = best
    targets = np.load(p400.P310)
    ids = targets["sample_ids"].astype(str)
    test = p400.test_part(ids)
    weights, domain_audit = p410.domain_weights(source, test)
    model = p399.fit_model(source, k, c_value, sample_weight=weights)
    proposal, margin = p399.predict(model, test, k)
    gap = p404.group_gap(test, proposal)
    route = (
        (proposal != test["base"])
        & (margin >= float(threshold["margin_threshold"]))
        & (gap >= float(threshold["group_gap_threshold"]))
    )
    prediction = test["base"].copy()
    prediction[route] = proposal[route]

    official = p400.read_csv(p400.OFFICIAL)
    official_ids = np.asarray([row["path"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] for row in official])
    position = {sample_id: index for index, sample_id in enumerate(ids)}
    official_prediction = np.asarray([prediction[position[sample_id]] for sample_id in official_ids])
    p315_rows = p400.read_csv(p400.P315)
    p315_prediction = np.asarray([int(row["prediction"]) for row in p315_rows])
    if not np.array_equal(p315_prediction, np.asarray([test["base"][position[sample_id]] for sample_id in official_ids])):
        raise RuntimeError("P315 CSV differs from P310 Test base")

    OUT.mkdir(parents=True, exist_ok=True)
    submission = OUT / "submission_p411_target_aware_raw_detail_teacher.csv"
    with submission.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        for row, value in zip(official, official_prediction, strict=True):
            writer.writerow({"path": row["path"], "prediction": int(value)})
    probability = np.full((len(prediction), 40), 0.0005, dtype=np.float32)
    probability[np.arange(len(prediction)), prediction] = 0.9805
    np.savez_compressed(
        OUT / "student_test_targets.npz",
        sample_ids=ids,
        target_mask=np.ones(len(ids), dtype=bool),
        emission_probability=probability,
        structured_distillation_probability=probability,
        structured_confidence=np.full(len(ids), 0.9805, dtype=np.float32),
        emission_prediction=prediction,
        structured_distillation_prediction=prediction,
    )
    changed = np.flatnonzero(route)
    report = {
        "stage": "P411_target_aware_raw_detail_Test",
        "status": "candidate" if len(changed) else "no_test_delta",
        "validation": {
            "p410_correct": 2217,
            "rows": 2470,
            "accuracy": 2217 / 2470,
            "net_vs_p310": 6,
            "fold_nets": [3, 2, 1],
            "strict_gate_pass": True,
        },
        "full_source_selection": {"k": k, "C": c_value, "threshold": threshold},
        "test_domain_audit": domain_audit,
        "test": {
            "rows": len(ids),
            "changes_vs_p315": int(len(changed)),
            "changed_rows_zero_based": changed.tolist(),
            "changed_sample_ids": ids[changed].tolist(),
            "changed_pairs": [f"{test['base'][row]}->{prediction[row]}" for row in changed],
            "candidate_margins": [float(margin[row]) for row in changed],
            "group_gaps": [float(gap[row]) for row in changed],
            "frozen_row_collisions": [int(row) for row in changed if int(row) in (77, 283, 328)],
            "submission": str(submission.resolve()),
            "test_labels_read": False,
        },
        "protocol": {
            "matched_oof": "P410",
            "test_covariates_used_for_source_weights": True,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: matched P410 source-to-Test importance weighting and unlabeled Test audit.\n"
        + json.dumps(report["test"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
