"""Run P104 source-safe single-modality family specialist probes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from analyze_p102_hard_set import source_crossfit_session_probability
from audit_p102_session_closure import load_npz
from audit_p87_sequence_decoder import DecoderConfig, align_metadata
from p100a_global_teacher_data import FOLD_USERS, H3_USERS, load_p100a_data
from p104_modality_data import (
    ALL_MODALITIES,
    MODALITIES,
    ModalityFeatures,
    load_modality_features,
)


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_SESSION_SUMMARY = HERE / "runs/p102_session_closure_v1/summary.json"
DEFAULT_FAMILIES = HERE / "runs/p104_confusion_atlas_v1/fold_families.json"
DEFAULT_NESTED = HERE / "runs/p101_f1_coarse_anchor_oof_v1/nested_coarse_vs"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = HERE / "runs/p104_modality_specialists_oof_v1"
SEED = 20260823
PCA_COMPONENTS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--session-summary", type=Path, default=DEFAULT_SESSION_SUMMARY)
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--nested-root", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--modalities", default=",".join(MODALITIES))
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--family-limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_names(value: str, allowed: Iterable[str]) -> list[str]:
    output = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(output) - set(allowed))
    if not output or unknown:
        raise ValueError(f"invalid selection {value!r}; unknown={unknown}")
    return output


def parse_folds(value: str) -> list[int]:
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not folds or not set(folds) <= set(range(4)):
        raise ValueError(f"invalid P104 folds: {value}")
    return folds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty P104 CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def family_key(classes: Iterable[int]) -> str:
    return "__".join(map(str, sorted(map(int, classes))))


def family_mask(labels: np.ndarray, classes: Iterable[int]) -> np.ndarray:
    return np.isin(labels, np.asarray(list(classes), dtype=np.int64))


def cyclic_shuffle_source(
    sample_ids: np.ndarray, users: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    """Return a deterministic label-free within-subject cyclic source map."""

    mapping = np.arange(len(sample_ids), dtype=np.int64)
    for user in sorted(set(users[selected].tolist())):
        rows = np.flatnonzero(selected & (users == user)).astype(np.int64)
        order = rows[np.argsort(sample_ids[rows].astype(str))]
        if len(order) > 1:
            mapping[order] = np.roll(order, 1)
    if not np.array_equal(users[mapping[selected]], users[selected]):
        raise RuntimeError("P104 shuffle crossed subjects")
    return mapping


def deployable_trigger(probability: np.ndarray, classes: Iterable[int]) -> np.ndarray:
    members = set(map(int, classes))
    top3 = np.argsort(np.asarray(probability), axis=1)[:, ::-1][:, :3]
    return np.asarray(
        [
            int(row[0]) in members
            and any(int(value) in (members - {int(row[0])}) for value in row[1:])
            for row in top3
        ],
        dtype=bool,
    )


@dataclass
class Projection:
    scaler: StandardScaler
    pca: PCA
    components: int

    @classmethod
    def fit(
        cls, values: np.ndarray, train_rows: np.ndarray, components: int
    ) -> "Projection":
        scaler = StandardScaler(copy=True)
        standardized = scaler.fit_transform(np.asarray(values[train_rows], dtype=np.float32))
        actual = min(int(components), standardized.shape[0] - 1, standardized.shape[1])
        if actual < 2:
            raise RuntimeError("P104 projection has insufficient source rows/features")
        pca = PCA(
            n_components=actual,
            svd_solver="randomized",
            iterated_power=2,
            random_state=SEED,
        )
        pca.fit(standardized)
        return cls(scaler=scaler, pca=pca, components=actual)

    def transform(self, values: np.ndarray) -> np.ndarray:
        standardized = self.scaler.transform(np.asarray(values, dtype=np.float32))
        return self.pca.transform(standardized).astype(np.float32)

    def zero(self, rows: int) -> np.ndarray:
        raw = np.broadcast_to(self.scaler.mean_, (rows, len(self.scaler.mean_)))
        return self.transform(raw)


def fit_specialist(
    projected: np.ndarray, labels: np.ndarray, rows: np.ndarray, classes: list[int]
) -> LogisticRegression:
    selected_labels = labels[rows]
    if set(selected_labels.tolist()) != set(classes):
        raise RuntimeError(
            f"P104 family train support changed: expected={classes}, got={sorted(set(selected_labels.tolist()))}"
        )
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(projected[rows], selected_labels)
    return model


def metric_bundle(
    labels: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
    classes: list[int],
) -> dict[str, Any]:
    values = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(prediction, dtype=np.int64)
    if len(values) == 0:
        return {"rows": 0, "accuracy": None, "balanced_accuracy": None, "macro_f1": None}
    recalls: dict[str, float | None] = {}
    for class_id in classes:
        selected = values == class_id
        recalls[str(class_id)] = (
            float(np.mean(predicted[selected] == class_id)) if selected.any() else None
        )
    valid_recalls = [value for value in recalls.values() if value is not None]
    per_subject = {
        user: {
            "rows": int(np.sum(users == user)),
            "accuracy": float(np.mean(predicted[users == user] == values[users == user])),
        }
        for user in sorted(set(users.tolist()))
    }
    return {
        "rows": len(values),
        "correct": int(np.sum(predicted == values)),
        "accuracy": float(np.mean(predicted == values)),
        "balanced_accuracy": float(np.mean(valid_recalls)),
        "macro_f1": float(
            f1_score(values, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "per_class_recall": recalls,
        "per_subject": per_subject,
        "worst_subject": min(
            (
                {"subject": user, **record}
                for user, record in per_subject.items()
            ),
            key=lambda value: (value["accuracy"], value["subject"]),
        ),
    }


def intervention_metrics(
    labels: np.ndarray,
    a_prediction: np.ndarray,
    specialist: np.ndarray,
) -> dict[str, Any]:
    a_correct = a_prediction == labels
    specialist_correct = specialist == labels
    rescue = int(np.sum((~a_correct) & specialist_correct))
    harm = int(np.sum(a_correct & (~specialist_correct)))
    return {
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "oracle_upper_bound_correct": int(np.sum(a_correct | specialist_correct)),
        "a_error_specialist_correct": rescue,
        "a_correct_specialist_wrong": harm,
    }


def trigger_metrics(
    labels: np.ndarray,
    a_probability: np.ndarray,
    specialist_prediction: np.ndarray,
    classes: list[int],
) -> dict[str, Any]:
    trigger = deployable_trigger(a_probability, classes)
    a_prediction = a_probability.argmax(axis=1)
    system = a_prediction.copy()
    system[trigger] = specialist_prediction[trigger]
    a_correct = a_prediction == labels
    system_correct = system == labels
    return {
        "rows": int(trigger.sum()),
        "coverage": float(trigger.mean()),
        "true_family_rows": int(np.sum(trigger & family_mask(labels, classes))),
        "true_family_precision": float(
            np.mean(family_mask(labels[trigger], classes)) if trigger.any() else 0.0
        ),
        "rescue": int(np.sum((~a_correct) & system_correct)),
        "harm": int(np.sum(a_correct & (~system_correct))),
        "net": int(np.sum(system_correct) - np.sum(a_correct)),
        "system_correct": int(np.sum(system_correct)),
        "a_correct": int(np.sum(a_correct)),
    }


def decoder_config(summary: dict[str, Any], fold: int) -> DecoderConfig:
    selected = summary["folds"][fold]["selected"]
    return DecoderConfig(
        gap_seconds=float(selected["gap_seconds"]),
        transition_weight=float(selected["transition_weight"]),
        trigram_backoff=float(selected["trigram_backoff"]),
        beam_width=int(selected["beam_width"]),
    )


def selected_families(
    family_archive: dict[str, Any], fold: int, limit: int | None
) -> list[dict[str, Any]]:
    values = list(family_archive["folds"][fold]["selected_families"])
    return values if limit is None else values[:limit]


def evaluate_modality(
    modality: ModalityFeatures,
    folds_to_run: list[int],
    family_archive: dict[str, Any],
    family_limit: int | None,
    smoke: bool,
    sample_ids: np.ndarray,
    users: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    a_probability: np.ndarray,
    source_probability: dict[int, np.ndarray],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    components = 8 if smoke else PCA_COMPONENTS
    for outer_fold in folds_to_run:
        source = fold_ids != outer_fold
        held = ~source
        families = selected_families(family_archive, outer_fold, family_limit)
        source_users = sorted(set(users[source].tolist()))
        inner_users = source_users[:2] if smoke else source_users
        inner_prediction = {
            value["family_key"]: np.full(len(labels), -1, dtype=np.int64)
            for value in families
        }
        inner_covered = np.zeros(len(labels), dtype=bool)
        for inner_user in inner_users:
            inner_train = np.flatnonzero(source & (users != inner_user)).astype(np.int64)
            inner_held = source & (users == inner_user)
            projection = Projection.fit(modality.aligned, inner_train, components)
            projected = np.full((len(labels), projection.components), np.nan, dtype=np.float32)
            active = source & ((users != inner_user) | (users == inner_user))
            active_rows = np.flatnonzero(active).astype(np.int64)
            projected[active_rows] = projection.transform(modality.aligned[active_rows])
            for family in families:
                classes = list(map(int, family["classes"]))
                train_rows = np.flatnonzero(
                    source & (users != inner_user) & family_mask(labels, classes)
                ).astype(np.int64)
                validation_rows = np.flatnonzero(
                    inner_held & family_mask(labels, classes)
                ).astype(np.int64)
                if not len(validation_rows):
                    continue
                classifier = fit_specialist(projected, labels, train_rows, classes)
                inner_prediction[family["family_key"]][validation_rows] = classifier.predict(
                    projected[validation_rows]
                )
            inner_covered |= inner_held

        outer_source_rows = np.flatnonzero(source).astype(np.int64)
        held_rows = np.flatnonzero(held).astype(np.int64)
        final_projection = Projection.fit(modality.aligned, outer_source_rows, components)
        projected = np.full(
            (len(labels), final_projection.components), np.nan, dtype=np.float32
        )
        projected[outer_source_rows] = final_projection.transform(
            modality.aligned[outer_source_rows]
        )
        aligned_held = final_projection.transform(modality.aligned[held_rows])
        shuffle_map = cyclic_shuffle_source(sample_ids, users, held)
        variants = {
            "aligned": aligned_held,
            "shuffle": final_projection.transform(modality.aligned[shuffle_map[held_rows]]),
            "zero": final_projection.zero(len(held_rows)),
        }
        for name, values in modality.interventions.items():
            variants[name] = final_projection.transform(values[held_rows])
        held_position = {int(row): index for index, row in enumerate(held_rows.tolist())}
        for family in families:
            key = str(family["family_key"])
            classes = list(map(int, family["classes"]))
            family_source = source & family_mask(labels, classes)
            source_eval = inner_covered & family_mask(labels, classes)
            source_eval_rows = np.flatnonzero(source_eval).astype(np.int64)
            missing_inner = inner_prediction[key][source_eval_rows] < 0
            if missing_inner.any():
                raise RuntimeError(f"P104 source inner coverage failed: {modality.name}/{outer_fold}/{key}")
            source_metrics = metric_bundle(
                labels[source_eval_rows],
                inner_prediction[key][source_eval_rows],
                users[source_eval_rows],
                classes,
            )
            source_a_prediction = source_probability[outer_fold].argmax(axis=1)
            source_a_metrics = metric_bundle(
                labels[source_eval_rows],
                source_a_prediction[source_eval_rows],
                users[source_eval_rows],
                classes,
            )
            for row in source_eval_rows.tolist():
                prediction_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "family_key": key,
                        "classes": "|".join(map(str, classes)),
                        "modality": modality.name,
                        "split": "source_inner_oof",
                        "variant": "aligned",
                        "row_index": row,
                        "sample_id": sample_ids[row],
                        "subject": users[row],
                        "label": int(labels[row]),
                        "a_prediction": int(source_a_prediction[row]),
                        "specialist_prediction": int(inner_prediction[key][row]),
                    }
                )

            train_rows = np.flatnonzero(family_source).astype(np.int64)
            classifier = fit_specialist(projected, labels, train_rows, classes)
            all_predictions = {
                name: classifier.predict(values) for name, values in variants.items()
            }
            family_held_rows = np.flatnonzero(held & family_mask(labels, classes)).astype(
                np.int64
            )
            local_positions = np.asarray(
                [held_position[int(row)] for row in family_held_rows], dtype=np.int64
            )
            held_a_probability = a_probability[held_rows]
            held_a_prediction = held_a_probability.argmax(axis=1)
            a_family_prediction = held_a_prediction[local_positions]
            family_results: dict[str, Any] = {}
            for variant, predictions in all_predictions.items():
                selected_prediction = predictions[local_positions]
                family_results[variant] = {
                    "metrics": metric_bundle(
                        labels[family_held_rows],
                        selected_prediction,
                        users[family_held_rows],
                        classes,
                    ),
                    "intervention": intervention_metrics(
                        labels[family_held_rows], a_family_prediction, selected_prediction
                    ),
                }
                for row, prediction, a_prediction in zip(
                    family_held_rows.tolist(),
                    selected_prediction.tolist(),
                    a_family_prediction.tolist(),
                ):
                    prediction_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "family_key": key,
                            "classes": "|".join(map(str, classes)),
                            "modality": modality.name,
                            "split": "outer_held_family",
                            "variant": variant,
                            "row_index": row,
                            "sample_id": sample_ids[row],
                            "subject": users[row],
                            "label": int(labels[row]),
                            "a_prediction": int(a_prediction),
                            "specialist_prediction": int(prediction),
                        }
                    )
            trigger = deployable_trigger(held_a_probability, classes)
            trigger_family = trigger & family_mask(labels[held_rows], classes)
            a_hard_metrics = metric_bundle(
                labels[held_rows][trigger_family],
                held_a_prediction[trigger_family],
                users[held_rows][trigger_family],
                classes,
            )
            result = {
                "outer_fold": outer_fold,
                "held_users": list(FOLD_USERS[outer_fold]),
                "family_key": key,
                "classes": classes,
                "modality": modality.name,
                "source_sample_count": int(family_source.sum()),
                "source_inner_oof_rows": int(source_eval.sum()),
                "held_sample_count": len(family_held_rows),
                "held_available_fraction": float(
                    np.mean(modality.available[family_held_rows])
                ),
                "source_a": source_a_metrics,
                "source_inner_specialist": source_metrics,
                "a_family": metric_bundle(
                    labels[family_held_rows],
                    a_family_prediction,
                    users[family_held_rows],
                    classes,
                ),
                "a_deployable_trigger_family_subset": a_hard_metrics,
                "a_error_count": int(
                    np.sum(a_family_prediction != labels[family_held_rows])
                ),
                "variants": family_results,
                "source_vs_held_balanced_accuracy_gap": (
                    None
                    if source_metrics["balanced_accuracy"] is None
                    else float(
                        source_metrics["balanced_accuracy"]
                        - family_results["aligned"]["metrics"]["balanced_accuracy"]
                    )
                ),
                "deployable_trigger": trigger_metrics(
                    labels[held_rows],
                    held_a_probability,
                    all_predictions["aligned"],
                    classes,
                ),
                "projection": {
                    "components": final_projection.components,
                    "explained_variance": float(
                        np.sum(final_projection.pca.explained_variance_ratio_)
                    ),
                    "fit_rows_all_source_classes": len(outer_source_rows),
                },
            }
            results.append(result)
        print(
            f"P104 {modality.name} fold={outer_fold} families={len(families)} "
            f"source={int(source.sum())} held={int(held.sum())}",
            flush=True,
        )
    return results


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["family_key"], result["modality"])].append(result)
    output: list[dict[str, Any]] = []
    for (key, modality), values in sorted(grouped.items()):
        aligned_correct = sum(
            value["variants"]["aligned"]["metrics"]["correct"] for value in values
        )
        held_rows = sum(value["held_sample_count"] for value in values)
        a_correct = sum(value["a_family"]["correct"] for value in values)
        rescue = sum(
            value["variants"]["aligned"]["intervention"]["rescue"] for value in values
        )
        harm = sum(
            value["variants"]["aligned"]["intervention"]["harm"] for value in values
        )
        variant_correct = {
            variant: sum(
                value["variants"][variant]["metrics"]["correct"] for value in values
            )
            for variant in values[0]["variants"]
        }
        gaps = [
            value["source_vs_held_balanced_accuracy_gap"]
            for value in values
            if value["source_vs_held_balanced_accuracy_gap"] is not None
        ]
        output.append(
            {
                "family_key": key,
                "classes": values[0]["classes"],
                "modality": modality,
                "selected_folds": [value["outer_fold"] for value in values],
                "selected_fold_count": len(values),
                "held_rows": held_rows,
                "a_family_correct": a_correct,
                "a_family_accuracy": float(a_correct / max(held_rows, 1)),
                "aligned_correct": aligned_correct,
                "aligned_accuracy": float(aligned_correct / max(held_rows, 1)),
                "aligned_minus_a_correct": aligned_correct - a_correct,
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "variant_correct": variant_correct,
                "aligned_minus_shuffle_correct": aligned_correct
                - variant_correct["shuffle"],
                "aligned_minus_zero_correct": aligned_correct - variant_correct["zero"],
                "mean_source_vs_held_balanced_accuracy_gap": float(np.mean(gaps))
                if gaps
                else None,
                "deployable_trigger_net": int(
                    sum(value["deployable_trigger"]["net"] for value in values)
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    modalities = parse_names(args.modalities, ALL_MODALITIES)
    folds_to_run = parse_folds(args.folds)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = load_npz(args.session.resolve())
    session_summary = json.loads(args.session_summary.resolve().read_text(encoding="utf-8"))
    family_archive = json.loads(args.families.resolve().read_text(encoding="utf-8"))
    data = load_p100a_data()
    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    fold_ids = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    if not np.array_equal(sample_ids, data.sample_ids.astype(str)):
        raise RuntimeError("P100/P104 row order differs")
    if not np.array_equal(users, data.users.astype(str)) or not np.array_equal(labels, data.labels):
        raise RuntimeError("P100/P104 user or label alignment differs")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached P104 specialists")
    if str(np.asarray(session["selected_system"]).item()) != "VS_session":
        raise RuntimeError("P104 specialist baseline is not VS_session")

    metadata = align_metadata(args.metadata.resolve(), sample_ids)
    source_probabilities: dict[int, np.ndarray] = {}
    for outer_fold in folds_to_run:
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
        source_probabilities[outer_fold] = probability

    prediction_rows: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    modality_audit: dict[str, Any] = {}
    for name in modalities:
        representation = load_modality_features(name, data=data)
        modality_audit[name] = representation.audit
        fold_results.extend(
            evaluate_modality(
                representation,
                folds_to_run,
                family_archive,
                args.family_limit,
                args.smoke,
                sample_ids,
                users,
                labels,
                fold_ids,
                a_probability,
                source_probabilities,
                prediction_rows,
            )
        )
    write_csv(output / "single_predictions.csv", prediction_rows)
    aggregates = aggregate_results(fold_results)
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "protocol": "P104 all-family-samples, source-inner-OOF modality probes; no final B/router/Student",
        "data": {
            "rows": len(labels),
            "subjects": sorted(set(users.tolist())),
            "folds_run": folds_to_run,
            "modalities_run": modalities,
            "family_limit": args.family_limit,
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "recipe": {
            "projection": "outer/inner train-only StandardScaler + randomized PCA",
            "pca_components": 8 if args.smoke else PCA_COMPONENTS,
            "pca_random_state": SEED,
            "classifier": "balanced LogisticRegression C=1.0 lbfgs max_iter=2000",
            "training_population": "all outer-source rows whose true class belongs to the source-selected family",
            "source_metric": "leave-one-source-subject-out OOF",
            "counterfactuals": "aligned / within-subject cyclic shuffle / source-train mean zero; representation-specific reverse/swap when available",
        },
        "modality_audit": modality_audit,
        "fold_results": fold_results,
        "aggregates": aggregates,
        "student_started": False,
        "final_b_teacher_trained": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "fold_results": len(fold_results),
                "aggregates": aggregates,
                "h3_rows_selected": 0,
                "h3_users_loaded": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
