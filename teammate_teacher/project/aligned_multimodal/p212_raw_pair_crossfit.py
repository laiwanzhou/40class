"""Strict outer-cross-fit raw-feature pair repairs over the frozen P205 teacher.

Every held cohort is predicted by recipes selected only from the other two cohorts.
The final Test recipe is selected from OOF-labelled Train rows and never reads Test labels.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

import p89_build_dual_consensus_submission as submission_io
from p89_pairwise_confusion_repair import FEATURE_SOURCES, aligned_rows, load_source
from p90_crossuser_visual_router import load_splits


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p212_raw_pair_crossfit_v1"
P205_OOF = HERE / "runs/p205_fixed_p203_p150_residual_v1/oof_predictions.npz"
P205_TEST = HERE / "runs/p205_fixed_p203_p150_residual_v1/submission_p205_fixed_residual.csv"
P89_TEST = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
COHORTS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
PAIR_POOL = (
    (24, 26), (8, 10), (23, 27), (0, 4), (20, 39), (25, 27),
    (11, 26), (7, 37), (32, 34), (6, 37), (21, 22), (24, 27),
)
ALPHAS = (10.0, 100.0, 1000.0)


def read_prediction(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray([int(row["prediction"]) for row in csv.DictReader(handle)])


def align_feature(
    source_ids: np.ndarray,
    values: np.ndarray,
    present: np.ndarray,
    target_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, found = aligned_rows(source_ids, target_ids)
    available = found.copy()
    available[found] &= present[rows[found]]
    output = np.zeros((len(target_ids), values.shape[1]), dtype=np.float32)
    output[available] = np.asarray(values[rows[available]], dtype=np.float32)
    return output, available


def fit_pair(x: np.ndarray, y: np.ndarray, pair: tuple[int, int], alpha: float):
    selected = np.isin(y, pair)
    if len(np.unique(y[selected])) != 2:
        return None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.asarray(x[selected], dtype=np.float32))
    model = RidgeClassifier(
        alpha=alpha, class_weight="balanced", solver="lsqr", tol=1e-4
    ).fit(scaled, y[selected])
    return scaler, model


def infer(fitted, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler, model = fitted
    scaled = scaler.transform(np.asarray(x, dtype=np.float32))
    prediction = model.predict(scaled).astype(np.int64)
    margin = np.abs(np.asarray(model.decision_function(scaled), dtype=np.float64))
    return prediction, margin


def choose_threshold(
    base: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    cohort: np.ndarray,
    available: np.ndarray,
    candidate: np.ndarray,
    margin: np.ndarray,
    pair: tuple[int, int],
) -> dict:
    eligible = available & np.isin(base, pair) & (candidate != base)
    gain = (candidate == labels).astype(np.int64) - (base == labels).astype(np.int64)
    if not eligible.any():
        return {"threshold": float("inf"), "selected": 0, "rescue": 0, "harm": 0,
                "net": 0, "minimum_cohort_gain": 0, "minimum_user_gain": 0}
    values = margin[eligible]
    thresholds = np.unique(np.concatenate((
        [np.inf], np.quantile(values, np.linspace(0.0, 1.0, 21)), [0.0]
    )))
    best = None
    for threshold in thresholds:
        selected = eligible & (margin >= threshold)
        per_cohort = {
            str(name): int(gain[selected & (cohort == name)].sum())
            for name in sorted(set(cohort.tolist()))
        }
        per_user = {
            str(user): int(gain[selected & (users == user)].sum())
            for user in sorted(set(users.tolist()))
        }
        rescue = int(np.sum(selected & (gain > 0)))
        harm = int(np.sum(selected & (gain < 0)))
        row = {
            "threshold": float(threshold), "selected": int(selected.sum()),
            "rescue": rescue, "harm": harm, "net": rescue - harm,
            "minimum_cohort_gain": min(per_cohort.values()),
            "minimum_user_gain": min(per_user.values()),
            "per_cohort_gain": per_cohort, "per_user_gain": per_user,
        }
        key = (
            row["minimum_cohort_gain"] >= 0,
            row["net"], row["rescue"], -row["harm"],
            row["minimum_user_gain"], -row["selected"], row["threshold"],
        )
        if best is None or key > best[0]:
            best = (key, row)
    return best[1]


def source_crossfit(
    data: dict, source_names: list[str], pair: tuple[int, int],
    feature_name: str, alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates, margins, available = [], [], []
    for target in source_names:
        train = [name for name in source_names if name != target]
        x_train = np.concatenate([data[name]["features"][feature_name] for name in train])
        y_train = np.concatenate([data[name]["labels"] for name in train])
        fitted = fit_pair(x_train, y_train, pair, alpha)
        x_target = data[target]["features"][feature_name]
        if fitted is None:
            pred = data[target]["base"].copy()
            score = np.zeros(len(pred), dtype=np.float64)
        else:
            pred, score = infer(fitted, x_target)
        candidates.append(pred)
        margins.append(score)
        available.append(data[target]["available"][feature_name])
    return np.concatenate(candidates), np.concatenate(margins), np.concatenate(available)


def select_rules(data: dict, source_names: list[str]) -> tuple[list[dict], list[dict]]:
    base = np.concatenate([data[name]["base"] for name in source_names])
    labels = np.concatenate([data[name]["labels"] for name in source_names])
    users = np.concatenate([data[name]["users"] for name in source_names])
    cohort = np.concatenate([
        np.full(len(data[name]["labels"]), name, dtype=object) for name in source_names
    ])
    audit = []
    for pair in PAIR_POOL:
        best = None
        for feature_name in FEATURE_SOURCES:
            for alpha in ALPHAS:
                candidate, margin, available = source_crossfit(
                    data, source_names, pair, feature_name, alpha
                )
                selection = choose_threshold(
                    base, labels, users, cohort, available, candidate, margin, pair
                )
                row = {"pair": list(pair), "feature": feature_name, "alpha": alpha,
                       **selection}
                key = (
                    row["minimum_cohort_gain"] >= 0,
                    row["net"], row["rescue"], -row["harm"],
                    row["minimum_user_gain"], -row["selected"],
                )
                if best is None or key > best[0]:
                    best = (key, row)
        chosen = best[1]
        chosen["eligible"] = bool(
            chosen["net"] >= 2 and chosen["rescue"] >= 3
            and chosen["minimum_cohort_gain"] >= 0
            and chosen["harm"] <= chosen["rescue"] // 2
        )
        audit.append(chosen)

    # Keep a disjoint set so no sample can receive incompatible pair proposals.
    eligible = sorted(
        [row for row in audit if row["eligible"]],
        key=lambda row: (row["net"], row["rescue"], -row["harm"]), reverse=True,
    )
    selected, used_classes = [], set()
    for row in eligible:
        if not used_classes.intersection(row["pair"]):
            selected.append(row)
            used_classes.update(row["pair"])
    return selected, audit


def apply_rules(data: dict, source_names: list[str], target: dict, rules: list[dict]):
    output = target["base"].copy()
    changed = np.zeros(len(output), dtype=bool)
    details = []
    x_train = {
        feature: np.concatenate([data[name]["features"][feature] for name in source_names])
        for feature in FEATURE_SOURCES
    }
    y_train = np.concatenate([data[name]["labels"] for name in source_names])
    for rule in rules:
        pair = tuple(rule["pair"])
        fitted = fit_pair(x_train[rule["feature"]], y_train, pair, rule["alpha"])
        if fitted is None:
            continue
        candidate, margin = infer(fitted, target["features"][rule["feature"]])
        accepted = (
            target["available"][rule["feature"]]
            & np.isin(target["base"], pair)
            & (candidate != target["base"])
            & (margin >= rule["threshold"])
        )
        output[accepted] = candidate[accepted]
        changed |= accepted
        details.append({**rule, "target_changes": int(accepted.sum())})
    return output, changed, details


def main() -> None:
    splits = load_splits()
    p205 = np.load(P205_OOF)
    p205_map = {
        sample_id: (int(label), int(prediction))
        for sample_id, label, prediction in zip(
            p205["sample_ids"].astype(str), p205["labels"], p205["prediction"]
        )
    }
    loaded = {name: load_source(name, "train") for name in FEATURE_SOURCES}
    data = {}
    for cohort_name in COHORTS:
        split = splits[cohort_name]
        ids = split.sample_ids.astype(str)
        data[cohort_name] = {
            "ids": ids,
            "users": split.users.astype(str),
            "labels": np.asarray([p205_map[value][0] for value in ids]),
            "base": np.asarray([p205_map[value][1] for value in ids]),
            "features": {}, "available": {},
        }
        for feature_name, (source_ids, values, present) in loaded.items():
            x, available = align_feature(source_ids, values, present, ids)
            data[cohort_name]["features"][feature_name] = x
            data[cohort_name]["available"][feature_name] = available

    report = {
        "stage": "P212_raw_pair_crossfit", "status": "complete",
        "protocol": {
            "base": "P205", "pair_pool": [list(pair) for pair in PAIR_POOL],
            "features": list(FEATURE_SOURCES), "alphas": list(ALPHAS),
            "selection": "other-two-cohort cross-fit; nonnegative source-cohort gain; disjoint pairs",
            "held_labels_used_for_selection": False, "test_labels_read": False,
        }, "cohorts": {},
    }
    held_outputs = {}
    for held in COHORTS:
        source_names = [name for name in COHORTS if name != held]
        rules, audit = select_rules(data, source_names)
        output, changed, details = apply_rules(data, source_names, data[held], rules)
        held_outputs[held] = output
        labels = data[held]["labels"]
        base = data[held]["base"]
        base_correct = int(np.sum(base == labels))
        correct = int(np.sum(output == labels))
        report["cohorts"][held] = {
            "source": source_names, "selected_rules": details, "source_audit": audit,
            "held": {"rows": len(labels), "base_correct": base_correct,
                     "correct": correct, "net": correct - base_correct,
                     "changed": int(changed.sum()),
                     "rescue": int(np.sum(changed & (base != labels) & (output == labels))),
                     "harm": int(np.sum(changed & (base == labels) & (output != labels)))},
        }
        print(json.dumps({"held": held, **report["cohorts"][held]["held"]}), flush=True)

    labels = np.concatenate([data[name]["labels"] for name in COHORTS])
    base = np.concatenate([data[name]["base"] for name in COHORTS])
    output = np.concatenate([held_outputs[name] for name in COHORTS])
    correct = int(np.sum(output == labels))
    base_correct = int(np.sum(base == labels))
    report["aggregate"] = {
        "rows": len(labels), "base_correct": base_correct, "correct": correct,
        "accuracy": float(correct / len(labels)), "net_vs_p205": correct - base_correct,
        "fold_nets": [report["cohorts"][name]["held"]["net"] for name in COHORTS],
        "gap_to_0.91": int(np.ceil(.91 * len(labels)) - correct),
    }

    test_ids = np.load(
        HERE / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )["sample_ids"].astype(str)
    test = {"ids": test_ids, "base": read_prediction(P205_TEST),
            "features": {}, "available": {}}
    for feature_name in FEATURE_SOURCES:
        source_ids, values, present = load_source(feature_name, "test")
        x, available = align_feature(source_ids, values, present, test_ids)
        test["features"][feature_name] = x
        test["available"][feature_name] = available
    rules, audit = select_rules(data, list(COHORTS))
    test_output, test_changed, details = apply_rules(data, list(COHORTS), test, rules)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p212_raw_pair_crossfit.csv"
    submission_io.write_submission(submission, submission_io.read_rows(P89_TEST), test_output)
    report["test"] = {
        "selected_rules": details, "oof_audit": audit,
        "changes_vs_p205": int(test_changed.sum()),
        "changed_rows": np.flatnonzero(test_changed).tolist(),
        "submission": str(submission.resolve()), "test_labels_read": False,
    }
    np.savez_compressed(
        OUTPUT / "predictions.npz", sample_ids=np.concatenate([data[n]["ids"] for n in COHORTS]),
        labels=labels, base_prediction=base, prediction=output,
        test_sample_ids=test_ids, test_base_prediction=test["base"], test_prediction=test_output,
        **{f"{name}_held_prediction": held_outputs[name] for name in COHORTS},
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"aggregate": report["aggregate"], "test": report["test"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
