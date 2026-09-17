"""Train source-only call/abstain models for the locked P105 specialists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze_p102_hard_set import source_crossfit_session_probability
from audit_p102_session_closure import load_npz
from audit_p87_sequence_decoder import align_metadata
from p100a_global_teacher_data import H3_USERS, load_p100a_data
from p104_modality_data import ModalityFeatures, load_modality_features
from train_p104_modality_specialists_oof import (
    PCA_COMPONENTS,
    Projection,
    cyclic_shuffle_source,
    family_mask,
    fit_specialist,
)
from validate_p105_specialist_bank_oof import (
    DEFAULT_METADATA,
    DEFAULT_NESTED,
    DEFAULT_SESSION,
    DEFAULT_SESSION_SUMMARY,
    SPECIALISTS,
    apply_routes,
    canonical_variants,
    concatenate,
    decoder_config,
    exact_edge_oracle_assignments,
    route_details,
    system_metrics,
    write_csv,
)


HERE = Path(__file__).resolve().parent
DEFAULT_P105 = HERE / "runs/p105_specialist_bank_oof_v1/summary.json"
DEFAULT_OUTPUT = HERE / "runs/p106_source_safe_routing_v1"
SEED = 20260823


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--session-summary", type=Path, default=DEFAULT_SESSION_SUMMARY)
    parser.add_argument("--nested-root", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--p105-summary", type=Path, default=DEFAULT_P105)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def probability_for_classes(
    classifier: LogisticRegression, values: np.ndarray, classes: list[int]
) -> np.ndarray:
    probability = classifier.predict_proba(values)
    lookup = {int(value): index for index, value in enumerate(classifier.classes_)}
    if set(lookup) != set(classes):
        raise RuntimeError("P106 specialist class order changed")
    return np.stack([probability[:, lookup[value]] for value in classes], axis=1)


def routing_features(
    a_probability: np.ndarray,
    specialist_probability: np.ndarray,
    classes: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(a_probability, dtype=np.float64)
    specialist = np.asarray(specialist_probability, dtype=np.float64)
    order = np.argsort(values, axis=1)[:, ::-1]
    ranks = np.empty_like(order)
    ranks[np.arange(len(values))[:, None], order] = np.arange(values.shape[1])[None, :] + 1
    top1 = order[:, 0]
    a_prediction = top1.astype(np.int64)
    specialist_local = specialist.argmax(axis=1)
    specialist_prediction = np.asarray(classes, dtype=np.int64)[specialist_local]
    left, right = classes
    entropy = -np.sum(values * np.log(np.maximum(values, 1e-12)), axis=1)
    entropy /= np.log(values.shape[1])
    predicted_a_probability = values[np.arange(len(values)), specialist_prediction]
    features = np.column_stack(
        (
            values[np.arange(len(values)), top1],
            values[np.arange(len(values)), order[:, 0]]
            - values[np.arange(len(values)), order[:, 1]],
            entropy,
            values[:, left],
            values[:, right],
            values[:, left] + values[:, right],
            np.minimum(values[:, left], values[:, right]),
            np.maximum(values[:, left], values[:, right]),
            ranks[:, left] / values.shape[1],
            ranks[:, right] / values.shape[1],
            np.isin(top1, classes).astype(np.float64),
            specialist.max(axis=1),
            np.abs(specialist[:, 0] - specialist[:, 1]),
            predicted_a_probability,
            (specialist_prediction == a_prediction).astype(np.float64),
        )
    ).astype(np.float64)
    candidate = (
        (ranks[:, left] <= 5)
        & (ranks[:, right] <= 5)
        & (specialist_prediction != a_prediction)
    )
    return features, candidate, a_prediction, specialist_prediction


def fit_call_model(
    features: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    a_prediction: np.ndarray,
    specialist_prediction: np.ndarray,
) -> tuple[StandardScaler | None, LogisticRegression | None, dict[str, Any]]:
    a_correct = a_prediction == labels
    specialist_correct = specialist_prediction == labels
    rescue = candidate & (~a_correct) & specialist_correct
    harm = candidate & a_correct & (~specialist_correct)
    decisive = rescue | harm
    targets = rescue[decisive].astype(np.int64)
    audit = {
        "candidate_rows": int(candidate.sum()),
        "decisive_rows": int(decisive.sum()),
        "rescue_examples": int(rescue.sum()),
        "harm_examples": int(harm.sum()),
        "target_classes": sorted(set(targets.tolist())),
    }
    if len(targets) == 0 or len(np.unique(targets)) != 2:
        audit.update(
            {
                "source_called_rows": 0,
                "source_rescue": 0,
                "source_harm": 0,
                "source_net": 0,
                "authorized": False,
                "reason": "source decisive rows do not contain both rescue and harm",
            }
        )
        return None, None, audit
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[decisive])
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight=None,
        random_state=SEED,
    )
    model.fit(scaled, targets)
    call_probability = model.predict_proba(scaler.transform(features))[:, 1]
    called = candidate & (call_probability > 0.5)
    source_rescue = int(np.sum(called & (~a_correct) & specialist_correct))
    source_harm = int(np.sum(called & a_correct & (~specialist_correct)))
    source_net = source_rescue - source_harm
    audit.update(
        {
            "source_called_rows": int(called.sum()),
            "source_rescue": source_rescue,
            "source_harm": source_harm,
            "source_net": source_net,
            "authorized": source_net > 0,
            "reason": "fixed probability > 0.5 and source net > 0",
        }
    )
    return scaler, model, audit


def call_probability(
    scaler: StandardScaler | None,
    model: LogisticRegression | None,
    features: np.ndarray,
) -> np.ndarray:
    if scaler is None or model is None:
        return np.zeros(len(features), dtype=np.float64)
    return model.predict_proba(scaler.transform(features))[:, 1]


def resolve_calls(
    call_masks: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    roster: tuple[dict[str, Any], ...] = SPECIALISTS,
) -> np.ndarray:
    rows = len(next(iter(call_masks.values())))
    routes = np.full(rows, "", dtype=object)
    best = np.full(rows, -np.inf, dtype=np.float64)
    for config in roster:
        key = config["family_key"]
        selected = call_masks[key] & (probabilities[key] > best)
        routes[selected] = key
        best[selected] = probabilities[key][selected]
    return routes


def main() -> None:
    args = parse_args()
    folds = [0] if args.smoke else [0, 1, 2, 3]
    configs = list(SPECIALISTS[:1] if args.smoke else SPECIALISTS)
    components = 8 if args.smoke else PCA_COMPONENTS
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = load_npz(args.session.resolve())
    session_summary = json.loads(args.session_summary.resolve().read_text(encoding="utf-8"))
    p105 = json.loads(args.p105_summary.resolve().read_text(encoding="utf-8"))
    data = load_p100a_data()
    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    fold_ids = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    if str(np.asarray(session["selected_system"]).item()) != "VS_session":
        raise RuntimeError("P106 requires the frozen A = VS + Session system")
    if not np.array_equal(sample_ids, data.sample_ids.astype(str)):
        raise RuntimeError("P100/P106 row order differs")
    if not np.array_equal(users, data.users.astype(str)) or not np.array_equal(labels, data.labels):
        raise RuntimeError("P100/P106 metadata differs")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached P106 routing")
    needed = sorted({name for config in configs for name in config["modalities"]})
    representations: dict[str, ModalityFeatures] = {
        name: load_modality_features(name, data=data) for name in needed
    }
    metadata = align_metadata(args.metadata.resolve(), sample_ids)
    source_a = {}
    for outer_fold in folds:
        probability, _ = source_crossfit_session_probability(
            outer_fold,
            args.nested_root.resolve(),
            sample_ids,
            users,
            fold_ids,
            labels,
            metadata,
            decoder_config(session_summary, outer_fold),
        )
        source_a[outer_fold] = probability

    evaluated = np.isin(fold_ids, folds)
    eval_rows = np.flatnonzero(evaluated).astype(np.int64)
    eval_position = {int(row): index for index, row in enumerate(eval_rows.tolist())}
    specialist_predictions = {
        variant: {
            config["family_key"]: np.full(len(eval_rows), -1, dtype=np.int64)
            for config in configs
        }
        for variant in ("aligned", "shuffle", "zero")
    }
    call_masks = {
        config["family_key"]: np.zeros(len(eval_rows), dtype=bool) for config in configs
    }
    call_probabilities = {
        config["family_key"]: np.zeros(len(eval_rows), dtype=np.float64) for config in configs
    }
    fold_audits: list[dict[str, Any]] = []
    for outer_fold in folds:
        source = fold_ids != outer_fold
        held = fold_ids == outer_fold
        source_rows = np.flatnonzero(source).astype(np.int64)
        held_rows = np.flatnonzero(held).astype(np.int64)
        inner_users = sorted(set(users[source].tolist()))
        if args.smoke:
            inner_users = inner_users[:2]
        inner_probability = {
            config["family_key"]: np.full((len(labels), 2), np.nan, dtype=np.float64)
            for config in configs
        }
        inner_covered = np.zeros(len(labels), dtype=bool)
        for inner_user in inner_users:
            train_all = np.flatnonzero(source & (users != inner_user)).astype(np.int64)
            validation_rows = np.flatnonzero(source & (users == inner_user)).astype(np.int64)
            projected: dict[str, np.ndarray] = {}
            for name in needed:
                projection = Projection.fit(representations[name].aligned, train_all, components)
                values = np.full((len(labels), projection.components), np.nan, dtype=np.float32)
                values[source_rows] = projection.transform(
                    representations[name].aligned[source_rows]
                )
                projected[name] = values
            for config in configs:
                family_train = np.flatnonzero(
                    source
                    & (users != inner_user)
                    & family_mask(labels, config["classes"])
                ).astype(np.int64)
                values = concatenate(projected, config["modalities"])
                classifier = fit_specialist(values, labels, family_train, config["classes"])
                inner_probability[config["family_key"]][validation_rows] = probability_for_classes(
                    classifier, values[validation_rows], config["classes"]
                )
            inner_covered[validation_rows] = True
        source_eval_rows = np.flatnonzero(source & inner_covered).astype(np.int64)
        for config in configs:
            family = config["family_key"]
            if not np.all(np.isfinite(inner_probability[family][source_eval_rows])):
                raise RuntimeError(f"P106 source-inner specialist OOF coverage failed: {family}")
        if not np.all(np.isfinite(source_a[outer_fold][source_eval_rows])):
            raise RuntimeError("P106 source-inner A OOF coverage failed")
        projections = {
            name: Projection.fit(representations[name].aligned, source_rows, components)
            for name in needed
        }
        source_projected = {
            name: projections[name].transform(representations[name].aligned[source_rows])
            for name in needed
        }
        source_position = {int(row): index for index, row in enumerate(source_rows.tolist())}
        shuffle_map = cyclic_shuffle_source(sample_ids, users, held)
        held_positions = np.asarray([eval_position[int(row)] for row in held_rows], dtype=np.int64)
        fold_audit = {"outer_fold": outer_fold, "families": []}
        for config in configs:
            family = config["family_key"]
            family_source_rows = np.flatnonzero(
                source & family_mask(labels, config["classes"])
            ).astype(np.int64)
            family_source_positions = np.asarray(
                [source_position[int(row)] for row in family_source_rows], dtype=np.int64
            )
            source_values = concatenate(source_projected, config["modalities"])
            classifier = fit_specialist(
                source_values,
                labels[source_rows],
                family_source_positions,
                config["classes"],
            )
            variants = canonical_variants(
                config,
                representations,
                projections,
                held_rows,
                shuffle_map[held_rows],
            )
            variant_probability = {
                variant: probability_for_classes(
                    classifier, values, config["classes"]
                )
                for variant, values in variants.items()
                if variant in {"aligned", "shuffle", "zero"}
            }
            for variant, probability in variant_probability.items():
                specialist_predictions[variant][family][held_positions] = np.asarray(
                    config["classes"], dtype=np.int64
                )[probability.argmax(axis=1)]
            source_features, source_candidate, source_prediction, source_specialist = routing_features(
                source_a[outer_fold][source_eval_rows],
                inner_probability[family][source_eval_rows],
                config["classes"],
            )
            scaler, router, audit = fit_call_model(
                source_features,
                source_candidate,
                labels[source_eval_rows],
                source_prediction,
                source_specialist,
            )
            held_features, held_candidate, _, _ = routing_features(
                a_probability[held_rows],
                variant_probability["aligned"],
                config["classes"],
            )
            probability = call_probability(scaler, router, held_features)
            call_probabilities[family][held_positions] = probability
            call_masks[family][held_positions] = (
                held_candidate & (probability > 0.5) & bool(audit["authorized"])
            )
            fold_audit["families"].append(
                {
                    "family_key": family,
                    "classes": config["classes"],
                    "modalities": config["modalities"],
                    "source_router": audit,
                    "held_candidate_rows": int(held_candidate.sum()),
                    "held_called_rows": int(call_masks[family][held_positions].sum()),
                }
            )
        fold_audits.append(fold_audit)
        print(f"P106 routing fold={outer_fold} families={len(configs)}", flush=True)

    if any(
        np.any(prediction < 0)
        for variants in specialist_predictions.values()
        for prediction in variants.values()
    ):
        raise RuntimeError("P106 specialist held coverage failed")
    routes = resolve_calls(call_masks, call_probabilities, tuple(configs))
    eval_labels = labels[eval_rows]
    eval_users = users[eval_rows]
    eval_folds = fold_ids[eval_rows]
    eval_a_probability = a_probability[eval_rows]
    a_prediction = eval_a_probability.argmax(axis=1)
    systems = {}
    for variant in ("aligned", "shuffle", "zero"):
        prediction = apply_routes(a_prediction, routes, specialist_predictions[variant])
        systems[variant] = system_metrics(
            eval_labels, a_prediction, prediction, eval_users, eval_folds
        )
    aligned_system = apply_routes(a_prediction, routes, specialist_predictions["aligned"])
    route_audit = route_details(
        routes, eval_labels, a_prediction, aligned_system, configs
    )
    oracle_routes = exact_edge_oracle_assignments(eval_labels, a_prediction, configs)
    oracle_prediction = apply_routes(
        a_prediction, oracle_routes, specialist_predictions["aligned"]
    )
    oracle = {
        "metrics": system_metrics(
            eval_labels, a_prediction, oracle_prediction, eval_users, eval_folds
        ),
        "routes": route_details(
            oracle_routes, eval_labels, a_prediction, oracle_prediction, configs
        ),
    }
    routing_rows = []
    for position, row in enumerate(eval_rows.tolist()):
        routing_rows.append(
            {
                "row_index": row,
                "sample_id": sample_ids[row],
                "subject": users[row],
                "outer_fold": int(fold_ids[row]),
                "label": int(labels[row]),
                "a_prediction": int(a_prediction[position]),
                "route": str(routes[position]),
                "system_prediction": int(aligned_system[position]),
                "oracle_route": str(oracle_routes[position]),
                "oracle_prediction": int(oracle_prediction[position]),
                **{
                    f"{config['family_key']}_call_probability": float(
                        call_probabilities[config["family_key"]][position]
                    )
                    for config in configs
                },
            }
        )
    write_csv(output / "routing_predictions.csv", routing_rows)
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "protocol": "P106 fixed source-only logistic call/abstain; probability > 0.5; no held threshold selection",
        "post_selection_limitation": p105["post_selection_limitation"],
        "data": {
            "rows": len(eval_rows),
            "folds": folds,
            "families": [value["family_key"] for value in configs],
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "features": [
            "a_top1_confidence",
            "a_top1_top2_margin",
            "a_normalized_entropy",
            "family_probabilities_sum_min_max_ranks",
            "a_top1_in_family",
            "specialist_confidence_margin",
            "a_probability_of_specialist_prediction",
            "specialist_agreement",
        ],
        "fold_router_audits": fold_audits,
        "source_safe_router": {
            "routes": route_audit,
            "systems": systems,
        },
        "exact_edge_oracle": oracle,
        "p105_reference": {
            "top3": p105["bank"]["systems"]["source_authorized_top3"],
            "exact_edge_oracle": p105["bank"]["systems"]["exact_edge_oracle"],
        },
        "h3_rows_selected": 0,
        "h3_users_loaded": [],
        "unified_b_teacher_trained": False,
        "student_started": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "source_safe_router": {
                    "aligned": systems["aligned"],
                    "shuffle_net": systems["shuffle"]["net"],
                    "zero_net": systems["zero"]["net"],
                    "routes": route_audit,
                },
                "exact_edge_oracle_net": oracle["metrics"]["net"],
                "p105_top3_net": p105["bank"]["systems"]["source_authorized_top3"]["aligned"]["net"],
                "h3_rows_selected": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
