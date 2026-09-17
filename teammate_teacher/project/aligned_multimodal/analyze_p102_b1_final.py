"""Final frozen-gate audit for P102-B1 and the P102 stopping decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_p102_b0_mechanisms import duplicate_audit, outcome_breakdown, per_value_net
from audit_p102_session_closure import load_npz


HERE = Path(__file__).resolve().parent
DEFAULT_B1 = HERE / "runs/p102_b1_visual_listwise_session_oof_v1/b1_oof_predictions.npz"
DEFAULT_B1_SUMMARY = HERE / "runs/p102_b1_visual_listwise_session_oof_v1/summary.json"
DEFAULT_HARD = HERE / "runs/p102_hard_set_v1/hard_set.npz"
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_OUTPUT = HERE / "runs/p102_b1_visual_listwise_session_oof_v1/final_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1", type=Path, default=DEFAULT_B1)
    parser.add_argument("--b1-summary", type=Path, default=DEFAULT_B1_SUMMARY)
    parser.add_argument("--hard-set", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--session-oof", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def subject_bootstrap(
    labels: np.ndarray,
    users: np.ndarray,
    a_prediction: np.ndarray,
    b_prediction: np.ndarray,
    repetitions: int = 50_000,
    seed: int = 20260823,
) -> dict[str, Any]:
    unique = np.asarray(sorted(set(users.tolist())))
    rows = np.asarray([np.sum(users == user) for user in unique], dtype=np.int64)
    net = np.asarray(
        [
            np.sum((b_prediction == labels) & (users == user))
            - np.sum((a_prediction == labels) & (users == user))
            for user in unique
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    net_sample = net[sampled].sum(axis=1)
    row_sample = rows[sampled].sum(axis=1)
    delta = net_sample / row_sample
    return {
        "unit": "subject",
        "subjects": int(len(unique)),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "observed_net_rows": int(net.sum()),
        "observed_accuracy_delta": float(net.sum() / rows.sum()),
        "net_rows_ci95": [float(value) for value in np.quantile(net_sample, [0.025, 0.975])],
        "accuracy_delta_ci95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
        "probability_nonpositive": float(np.mean(net_sample <= 0)),
    }


def class_deltas(
    labels: np.ndarray, a_prediction: np.ndarray, b_prediction: np.ndarray
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for class_id in range(40):
        selected = labels == class_id
        support = int(selected.sum())
        a_correct = int(np.sum(selected & (a_prediction == labels)))
        b_correct = int(np.sum(selected & (b_prediction == labels)))
        result.append(
            {
                "class_id": class_id,
                "support": support,
                "a_correct": a_correct,
                "b_correct": b_correct,
                "net": b_correct - a_correct,
                "recall_delta": float((b_correct - a_correct) / max(support, 1)),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    b1 = load_npz(args.b1.resolve())
    hard = load_npz(args.hard_set.resolve())
    session = load_npz(args.session_oof.resolve())
    summary = json.loads(args.b1_summary.resolve().read_text(encoding="utf-8"))
    for archive, name in ((hard, "hard"), (session, "session")):
        if not np.array_equal(b1["sample_ids"].astype(str), archive["sample_ids"].astype(str)):
            raise RuntimeError(f"B1/{name} sample order differs")
    labels = np.asarray(b1["labels"], dtype=np.int64)
    users = b1["users"].astype(str)
    folds = np.asarray(b1["fold_ids"], dtype=np.int64)
    if len(labels) != 1941 or not np.asarray(b1["evaluated"], dtype=bool).all():
        raise RuntimeError("formal B1 OOF coverage is incomplete")
    if summary["data"]["h3_rows_selected"] != 0 or summary["data"]["h3_users_loaded"]:
        raise RuntimeError("H3 contract failed")

    names = ("full_local", "full_session", "shuffle_session", "zero_session")
    probabilities = {
        name: np.asarray(b1[f"{name}_probability"], dtype=np.float64) for name in names
    }
    a_probability = np.asarray(b1["a_probability"], dtype=np.float64)
    for name, probability in {"a": a_probability, **probabilities}.items():
        if probability.shape != (1941, 40) or not np.isfinite(probability).all():
            raise RuntimeError(f"invalid probability: {name}")
        if not np.allclose(probability.sum(axis=1), 1.0, atol=2e-6):
            raise RuntimeError(f"probability mass failed: {name}")

    a_prediction = a_probability.argmax(axis=1)
    predictions = {name: probability.argmax(axis=1) for name, probability in probabilities.items()}
    full_prediction = predictions["full_session"]
    category = hard["category"].astype(str)
    session_id = np.asarray(session["sequence_session_id"], dtype=np.int64)
    variant = {
        name: outcome_breakdown(
            labels, a_prediction, prediction, np.ones(len(labels), dtype=bool)
        )
        for name, prediction in predictions.items()
    }
    classes = class_deltas(labels, a_prediction, full_prediction)
    best_classes = sorted(classes, key=lambda value: (-value["net"], value["class_id"]))[:10]
    worst_classes = sorted(classes, key=lambda value: (value["net"], value["class_id"]))[:10]
    rare_harms = [value for value in classes if value["support"] <= 25 and value["net"] < 0]
    bootstrap = subject_bootstrap(labels, users, a_prediction, full_prediction)
    full_metrics = summary["variants"]["full_session"]["metrics"]
    zero_metrics = summary["variants"]["zero_session"]["metrics"]
    a_metrics = summary["a_baseline"]
    fold_net = [
        int(fold["variants"]["full_session"]["vs_a"]["net"]) for fold in summary["folds"]
    ]
    all_transition_masked = all(
        int(fold["session_decode"][name]["transition_fit_labels_masked_outside_source"])
        == int(fold["variants"][f"{name}_session"]["metrics"]["rows"])
        for fold in summary["folds"]
        for name in ("full", "shuffle", "zero")
    )
    all_candidate_mass_zero = all(
        float(fold["session_decode"][name]["final_mass_outside_candidate"]) == 0.0
        for fold in summary["folds"]
        for name in ("full", "shuffle", "zero")
    )

    result = {
        "status": "complete",
        "integrity": {
            "rows": 1941,
            "subjects": sorted(set(users.tolist())),
            "folds": sorted(set(map(int, folds.tolist()))),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
            "all_outer_held_labels_masked_before_transition_fit": all_transition_masked,
            "all_final_mass_outside_candidate_zero": all_candidate_mass_zero,
        },
        "variant_outcomes": variant,
        "candidate_category": {
            name: outcome_breakdown(labels, a_prediction, full_prediction, category == name)
            for name in ("correct", "A", "B", "C")
        },
        "fold": per_value_net(folds.astype(str), labels, a_prediction, full_prediction),
        "subject": per_value_net(users, labels, a_prediction, full_prediction),
        "subject_bootstrap": bootstrap,
        "class": {
            "best_net": best_classes,
            "worst_net": worst_classes,
            "rare_support_at_most_25_with_harm": rare_harms,
        },
        "session_consistency": {
            "a": duplicate_audit(session_id, a_prediction, labels, a_prediction),
            "b1_full_session": duplicate_audit(
                session_id, full_prediction, labels, a_prediction
            ),
            "shuffle_session": duplicate_audit(
                session_id, predictions["shuffle_session"], labels, a_prediction
            ),
            "zero_session": duplicate_audit(
                session_id, predictions["zero_session"], labels, a_prediction
            ),
        },
        "gate": {
            "net_rescue": int(variant["full_session"]["net"]),
            "net_at_least_20": bool(variant["full_session"]["net"] >= 20),
            "mcnemar_exact_p": float(summary["variants"]["full_session"]["vs_a"]["mcnemar_exact_p"]),
            "fold_net": fold_net,
            "all_folds_positive": bool(all(value > 0 for value in fold_net)),
            "subject_bootstrap_ci_excludes_nonpositive": bool(
                bootstrap["accuracy_delta_ci95"][0] > 0.0
            ),
            "macro_f1_delta": float(full_metrics["macro_f1"] - a_metrics["macro_f1"]),
            "full_macro_minus_zero": float(full_metrics["macro_f1"] - zero_metrics["macro_f1"]),
            "worst_subject_top1_delta": float(
                full_metrics["worst_subject"]["top1"] - a_metrics["worst_subject"]["top1"]
            ),
            "aligned_correct_minus_shuffle": int(
                full_metrics["top1_correct"]
                - summary["variants"]["shuffle_session"]["metrics"]["top1_correct"]
            ),
            "aligned_correct_minus_zero": int(
                full_metrics["top1_correct"] - zero_metrics["top1_correct"]
            ),
            "correspondence_strongly_supported": False,
            "major_system_value": False,
        },
        "final_decision": {
            "p102_b1_pass": False,
            "student_authorized": False,
            "stop_b_variants": True,
            "reason": (
                "B1 improves A by only +15, below the frozen +20 gate; its advantage "
                "over shuffle/zero is only +4/+1, fold nets are +12/0/+4/-1, worst-"
                "subject accuracy falls, and zero visual has higher macro-F1. B0 prototype "
                "scoring and B1 discriminative listwise scoring are substantially different "
                "failed mechanisms, so the pre-registered stopping rule is reached."
            ),
            "retained_system": "P100 coarse VS + source-only Session (1575/1941)",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
