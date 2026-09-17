"""P102 shared, leakage-safe Session closure for P100 VS and P101-F3.

The per-outer-fold decoder recipe is selected only from the nine source subjects
using the already frozen P101 nested coarse-VS predictions.  Those predictions
come from models which exclude both the current outer fold and the prediction
row's subject group.  The selected recipe and source-only transition table are
then applied unchanged to both held VS and held F3 emissions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from audit_p87_sequence_decoder import (
    RecordingMetadata,
    align_metadata,
    build_sessions,
    choose_config,
)
from build_p87s_structured_targets import (
    backed_off_structured_probability,
    build_targets,
)
from p100a_global_teacher_data import FOLD_ROW_COUNTS, FOLD_USERS, H3_USERS, parse_sample_id


HERE = Path(__file__).resolve().parent
DEFAULT_VS = HERE / "runs/p100a_a0_global_teacher_oof_v1/VS_complete_oof.npz"
DEFAULT_F3 = HERE / "runs/p101_f3_causal_interaction_oof_v1/F3_VSI_complete_oof.npz"
DEFAULT_NESTED = HERE / "runs/p101_f1_coarse_anchor_oof_v1/nested_coarse_vs"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = HERE / "runs/p102_session_closure_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vs-oof", type=Path, default=DEFAULT_VS)
    parser.add_argument("--f3-oof", type=Path, default=DEFAULT_F3)
    parser.add_argument("--nested-root", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gap-seconds", type=float, nargs="+", default=(20.0, 30.0, 45.0))
    parser.add_argument(
        "--transition-weights", type=float, nargs="+", default=(0.25, 0.30, 0.35)
    )
    parser.add_argument("--trigram-backoffs", type=float, nargs="+", default=(1.0, 2.0, 5.0))
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def topk_correct(probability: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(np.asarray(probability), axis=1)[:, -int(k) :]
    return np.any(order == np.asarray(labels)[:, None], axis=1)


def true_rank(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability)
    labels = np.asarray(labels, dtype=np.int64)
    true_value = probability[np.arange(len(labels)), labels]
    return 1 + np.sum(probability > true_value[:, None], axis=1)


def classification_metrics(
    probability: np.ndarray, labels: np.ndarray, users: np.ndarray
) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    probability /= probability.sum(axis=1, keepdims=True)
    labels = np.asarray(labels, dtype=np.int64)
    users = np.asarray(users).astype(str)
    prediction = probability.argmax(axis=1)
    per_class: list[dict[str, float | int]] = []
    for class_id in range(probability.shape[1]):
        selected = labels == class_id
        true_positive = int(np.sum(selected & (prediction == class_id)))
        false_negative = int(np.sum(selected & (prediction != class_id)))
        false_positive = int(np.sum((~selected) & (prediction == class_id)))
        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        per_class.append(
            {
                "class_id": class_id,
                "support": int(selected.sum()),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    per_subject = {
        user: {
            "rows": int(np.sum(users == user)),
            "top1_correct": int(np.sum((users == user) & (prediction == labels))),
            "top1": float(np.mean(prediction[users == user] == labels[users == user])),
        }
        for user in sorted(set(users.tolist()))
    }
    worst_user = min(per_subject, key=lambda user: per_subject[user]["top1"])
    return {
        "rows": int(len(labels)),
        "top1_correct": int(np.sum(prediction == labels)),
        "top1": float(np.mean(prediction == labels)),
        "top3_correct": int(topk_correct(probability, labels, 3).sum()),
        "top3": float(topk_correct(probability, labels, 3).mean()),
        "top5_correct": int(topk_correct(probability, labels, 5).sum()),
        "top5": float(topk_correct(probability, labels, 5).mean()),
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "nll": float(-np.log(np.maximum(probability[np.arange(len(labels)), labels], 1e-12)).mean()),
        "worst_subject": {"user": worst_user, "top1": per_subject[worst_user]["top1"]},
        "per_subject": per_subject,
        "per_class": per_class,
    }


def exact_mcnemar_p(rescue: int, harm: int) -> float:
    changed = int(rescue + harm)
    if changed == 0:
        return 1.0
    tail = sum(math.comb(changed, value) for value in range(min(rescue, harm) + 1))
    return float(min(1.0, 2.0 * tail / (2.0**changed)))


def comparison(
    labels: np.ndarray,
    users: np.ndarray,
    raw_probability: np.ndarray,
    candidate_probability: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    users = np.asarray(users).astype(str)
    raw_prediction = np.asarray(raw_probability).argmax(axis=1)
    candidate_prediction = np.asarray(candidate_probability).argmax(axis=1)
    raw_correct = raw_prediction == labels
    candidate_correct = candidate_prediction == labels
    rescue_mask = (~raw_correct) & candidate_correct
    harm_mask = raw_correct & (~candidate_correct)
    raw_rank = true_rank(raw_probability, labels)
    candidate_rank = true_rank(candidate_probability, labels)
    by_subject: dict[str, Any] = {}
    for user in sorted(set(users.tolist())):
        selected = users == user
        rescue = int(np.sum(rescue_mask & selected))
        harm = int(np.sum(harm_mask & selected))
        by_subject[user] = {
            "rows": int(selected.sum()),
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
        }
    return {
        "rescue": int(rescue_mask.sum()),
        "harm": int(harm_mask.sum()),
        "net": int(rescue_mask.sum() - harm_mask.sum()),
        "changed": int(np.sum(raw_prediction != candidate_prediction)),
        "mcnemar_exact_p": exact_mcnemar_p(int(rescue_mask.sum()), int(harm_mask.sum())),
        "true_rank_improved": int(np.sum(candidate_rank < raw_rank)),
        "true_rank_harmed": int(np.sum(candidate_rank > raw_rank)),
        "mean_true_rank_delta": float(np.mean(candidate_rank - raw_rank)),
        "top5_rescue": int(np.sum((~topk_correct(raw_probability, labels, 5)) & topk_correct(candidate_probability, labels, 5))),
        "top5_harm": int(np.sum(topk_correct(raw_probability, labels, 5) & (~topk_correct(candidate_probability, labels, 5)))),
        "per_subject": by_subject,
    }


def validate_primary_inputs(vs: dict[str, np.ndarray], f3: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for archive, name in ((vs, "VS"), (f3, "F3")):
        required = {"sample_ids", "users", "fold_ids", "direct_probability", "direct_logits"}
        missing = sorted(required - set(archive))
        if missing:
            raise KeyError(f"{name} archive missing {missing}")
    sample_ids = np.asarray(vs["sample_ids"]).astype(str)
    users = np.asarray(vs["users"]).astype(str)
    folds = np.asarray(vs["fold_ids"], dtype=np.int64)
    if len(sample_ids) != 1941 or len(np.unique(sample_ids)) != 1941:
        raise RuntimeError("P102 development row contract changed")
    if not np.array_equal(sample_ids, np.asarray(f3["sample_ids"]).astype(str)):
        raise RuntimeError("VS/F3 sample order differs")
    if not np.array_equal(users, np.asarray(f3["users"]).astype(str)):
        raise RuntimeError("VS/F3 subject order differs")
    if not np.array_equal(folds, np.asarray(f3["fold_ids"], dtype=np.int64)):
        raise RuntimeError("VS/F3 fold order differs")
    parsed = [parse_sample_id(value) for value in sample_ids]
    labels = np.asarray([value[0] for value in parsed], dtype=np.int64)
    parsed_users = np.asarray([value[1] for value in parsed]).astype(str)
    if not np.array_equal(parsed_users, users):
        raise RuntimeError("subject ids do not match sample ids")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached P102")
    for fold, expected_users in enumerate(FOLD_USERS):
        selected = folds == fold
        if int(selected.sum()) != FOLD_ROW_COUNTS[fold]:
            raise RuntimeError(f"fold {fold} row count changed")
        if set(users[selected].tolist()) != set(expected_users):
            raise RuntimeError(f"fold {fold} subject contract changed")
    return sample_ids, users, folds, labels


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def validate_metadata(metadata: RecordingMetadata, sample_ids: np.ndarray, users: np.ndarray) -> dict[str, Any]:
    if not np.array_equal(metadata.sample_ids, sample_ids):
        raise RuntimeError("metadata sample order changed")
    if not np.array_equal(metadata.users, users):
        raise RuntimeError("metadata subjects differ from OOF subjects")
    if set(metadata.users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 metadata row was selected")
    timestamp = np.isfinite(metadata.starts) & (metadata.dates != "")
    return {
        "requested_rows": int(len(sample_ids)),
        "selected_rows": int(len(metadata.sample_ids)),
        "timestamp_rows": int(timestamp.sum()),
        "timestamp_coverage": float(timestamp.mean()),
        "h3_rows_selected": 0,
        "h3_users_loaded": [],
    }


def choose_shared_config(
    outer_fold: int,
    nested_root: Path,
    sample_ids: np.ndarray,
    users: np.ndarray,
    folds: np.ndarray,
    labels: np.ndarray,
    metadata: RecordingMetadata,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    path = nested_root.resolve() / f"outer{outer_fold}" / "nested_predictions.npz"
    nested = load_npz(path)
    if not np.array_equal(np.asarray(nested["sample_ids"]).astype(str), sample_ids):
        raise RuntimeError(f"nested sample order differs for outer {outer_fold}")
    if int(np.asarray(nested["outer_fold"]).item()) != outer_fold:
        raise RuntimeError("nested outer fold marker changed")
    covered = np.asarray(nested["covered"], dtype=bool)
    expected = folds != outer_fold
    if not np.array_equal(covered, expected):
        raise RuntimeError(f"nested source coverage differs for outer {outer_fold}")
    logits = np.asarray(nested["logits"], dtype=np.float64)
    if not np.isfinite(logits[covered]).all():
        raise RuntimeError("nested source logits contain non-finite values")
    masked_labels = labels.copy()
    masked_labels[~expected] = -10_000
    nested_log_probability = np.full_like(logits, -math.log(logits.shape[1]))
    nested_log_probability[covered] = log_softmax(logits[covered])
    config, grid = choose_config(
        masked_labels,
        folds,
        nested_log_probability,
        metadata,
        outer_fold=outer_fold,
        gap_seconds_candidates=tuple(map(float, args.gap_seconds)),
        transition_weights=tuple(map(float, args.transition_weights)),
        trigram_backoffs=tuple(map(float, args.trigram_backoffs)),
        beam_width=int(args.beam_width),
    )
    selected_key = (
        f"gap={config.gap_seconds:g},backoff={config.trigram_backoff:g},"
        f"weight={config.transition_weight:g}"
    )
    audit = {
        "nested_path": str(path),
        "nested_sha256": sha256(path),
        "source_rows": int(expected.sum()),
        "source_users": sorted(set(users[expected].tolist())),
        "outer_held_users": list(FOLD_USERS[outer_fold]),
        "source_held_overlap": sorted(set(users[expected].tolist()) & set(FOLD_USERS[outer_fold])),
        "source_safe_nested_coverage": int(covered.sum()),
        "selected_inner_accuracy": float(grid[selected_key]),
    }
    return config, grid, audit


def empty_result(rows: int, classes: int) -> dict[str, np.ndarray]:
    return {
        "structured_probability": np.zeros((rows, classes), dtype=np.float64),
        "session_probability": np.zeros((rows, classes), dtype=np.float64),
        "structured_map_prediction": np.full(rows, -1, dtype=np.int64),
        "structured_marginal_prediction": np.full(rows, -1, dtype=np.int64),
        "session_prediction": np.full(rows, -1, dtype=np.int64),
        "structured_weight": np.zeros(rows, dtype=np.float64),
        "sequence_session_id": np.full(rows, -1, dtype=np.int64),
    }


def select_final_baseline(summaries: dict[str, dict[str, Any]]) -> str:
    candidates = ("VS_session", "F3_session")
    def key(name: str) -> tuple[float, ...]:
        metrics = summaries[name]["metrics"]
        subject_nets = [value["net"] for value in summaries[name]["vs_raw"]["per_subject"].values()]
        return (
            float(metrics["top1_correct"]),
            float(metrics["macro_f1"]),
            float(metrics["worst_subject"]["top1"]),
            float(metrics["top5_correct"]),
            -float(np.var(subject_nets)),
            1.0 if name == "VS_session" else 0.0,
        )
    return max(candidates, key=key)


def main() -> None:
    args = parse_args()
    if args.posterior_temperature <= 0:
        raise ValueError("posterior-temperature must be positive")
    vs_path = args.vs_oof.resolve()
    f3_path = args.f3_oof.resolve()
    nested_root = args.nested_root.resolve()
    metadata_path = args.metadata.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    vs = load_npz(vs_path)
    f3 = load_npz(f3_path)
    sample_ids, users, folds, labels = validate_primary_inputs(vs, f3)
    metadata = align_metadata(metadata_path, sample_ids)
    metadata_audit = validate_metadata(metadata, sample_ids, users)

    raw_probability = {
        "VS": np.asarray(vs["direct_probability"], dtype=np.float64),
        "F3": np.asarray(f3["direct_probability"], dtype=np.float64),
    }
    raw_log_probability = {
        "VS": log_softmax(np.asarray(vs["direct_logits"], dtype=np.float64)),
        "F3": log_softmax(np.asarray(f3["direct_logits"], dtype=np.float64)),
    }
    results = {name: empty_result(len(labels), 40) for name in ("VS", "F3")}
    fold_audits: list[dict[str, Any]] = []

    for outer_fold in range(4):
        train = np.flatnonzero(folds != outer_fold).astype(np.int64)
        held = np.flatnonzero(folds == outer_fold).astype(np.int64)
        masked_labels = labels.copy()
        masked_labels[held] = -10_000
        config, grid, config_audit = choose_shared_config(
            outer_fold, nested_root, sample_ids, users, folds, labels, metadata, args
        )
        fold_record: dict[str, Any] = {
            "fold": outer_fold,
            "held_users": list(FOLD_USERS[outer_fold]),
            "held_rows": int(len(held)),
            "selected": {
                "gap_seconds": float(config.gap_seconds),
                "transition_weight": float(config.transition_weight),
                "trigram_backoff": float(config.trigram_backoff),
                "beam_width": int(config.beam_width),
                "posterior_temperature": float(args.posterior_temperature),
            },
            "selection_audit": config_audit,
            "inner_grid": grid,
            "teachers": {},
        }
        for name in ("VS", "F3"):
            target = build_targets(
                raw_log_probability[name],
                masked_labels,
                train,
                held,
                metadata,
                config,
                posterior_temperature=float(args.posterior_temperature),
            )
            structured = np.asarray(target["structured_probability"], dtype=np.float64)
            session_probability, structured_weight = backed_off_structured_probability(
                raw_probability[name], structured, beam_width=int(config.beam_width)
            )
            session_id = np.asarray(target["session_id"], dtype=np.int64)
            global_session_id = np.where(session_id >= 0, outer_fold * 10_000 + session_id, -1)
            results[name]["structured_probability"][held] = structured[held]
            results[name]["session_probability"][held] = session_probability[held]
            results[name]["structured_map_prediction"][held] = np.asarray(
                target["structured_map_prediction"], dtype=np.int64
            )[held]
            results[name]["structured_marginal_prediction"][held] = structured[held].argmax(axis=1)
            results[name]["session_prediction"][held] = session_probability[held].argmax(axis=1)
            results[name]["structured_weight"][held] = structured_weight[held]
            results[name]["sequence_session_id"][held] = global_session_id[held]
            held_metrics = classification_metrics(session_probability[held], labels[held], users[held])
            fold_record["teachers"][name] = {
                "raw_top1_correct": int(np.sum(raw_probability[name][held].argmax(axis=1) == labels[held])),
                "session": held_metrics,
                "raw_to_session": comparison(
                    labels[held], users[held], raw_probability[name][held], session_probability[held]
                ),
                "train_sessions": int(target["train_session_count"]),
                "held_sessions": int(target["holdout_session_count"]),
                "decoded_held_rows": int(np.sum(session_id[held] >= 0)),
                "mean_structured_weight": float(structured_weight[held].mean()),
            }
        fold_audits.append(fold_record)

    for name in ("VS", "F3"):
        if not np.isfinite(results[name]["session_probability"]).all():
            raise RuntimeError(f"{name} session probability is non-finite")
        if np.any(results[name]["session_prediction"] < 0):
            raise RuntimeError(f"{name} session prediction lacks OOF coverage")

    systems: dict[str, dict[str, Any]] = {}
    for name in ("VS", "F3"):
        systems[f"{name}_raw"] = {"metrics": classification_metrics(raw_probability[name], labels, users)}
        systems[f"{name}_session"] = {
            "metrics": classification_metrics(results[name]["session_probability"], labels, users),
            "vs_raw": comparison(
                labels, users, raw_probability[name], results[name]["session_probability"]
            ),
            "hard_map_vs_raw": comparison(
                labels,
                users,
                raw_probability[name],
                np.eye(40, dtype=np.float64)[results[name]["structured_map_prediction"]],
            ),
        }
    selected = select_final_baseline(systems)
    selected_teacher = selected.split("_")[0]
    selected_probability = results[selected_teacher]["session_probability"]
    selected_prediction = results[selected_teacher]["session_prediction"]

    np.savez_compressed(
        output / "session_oof_predictions.npz",
        sample_ids=sample_ids,
        users=users,
        labels=labels,
        fold_ids=folds,
        vs_raw_probability=raw_probability["VS"].astype(np.float32),
        f3_raw_probability=raw_probability["F3"].astype(np.float32),
        vs_session_probability=results["VS"]["session_probability"].astype(np.float32),
        f3_session_probability=results["F3"]["session_probability"].astype(np.float32),
        vs_structured_probability=results["VS"]["structured_probability"].astype(np.float32),
        f3_structured_probability=results["F3"]["structured_probability"].astype(np.float32),
        vs_structured_map_prediction=results["VS"]["structured_map_prediction"],
        f3_structured_map_prediction=results["F3"]["structured_map_prediction"],
        vs_structured_weight=results["VS"]["structured_weight"].astype(np.float32),
        f3_structured_weight=results["F3"]["structured_weight"].astype(np.float32),
        sequence_session_id=results["VS"]["sequence_session_id"],
        selected_system=np.asarray(selected),
        selected_probability=selected_probability.astype(np.float32),
        selected_prediction=selected_prediction,
    )
    summary = {
        "status": "complete",
        "protocol": (
            "P102 four-fold shared Session closure. Per outer fold, one decoder recipe "
            "is selected from nine source subjects using source-safe nested coarse-VS "
            "emissions, then applied unchanged to held P100 VS and P101-F3 emissions."
        ),
        "data": {
            "rows": int(len(labels)),
            "users": sorted(set(users.tolist())),
            "fold_counts": [int(np.sum(folds == fold)) for fold in range(4)],
            **metadata_audit,
        },
        "inputs": {
            "vs": {"path": str(vs_path), "sha256": sha256(vs_path)},
            "f3": {"path": str(f3_path), "sha256": sha256(f3_path)},
            "metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
            "nested_root": str(nested_root),
        },
        "recipe": {
            "selection_teacher": "nested coarse VS only",
            "shared_between_teachers": True,
            "gap_seconds": list(map(float, args.gap_seconds)),
            "transition_weights": list(map(float, args.transition_weights)),
            "trigram_backoffs": list(map(float, args.trigram_backoffs)),
            "beam_width": int(args.beam_width),
            "posterior_temperature": float(args.posterior_temperature),
            "probability": "entropy-adaptive structured posterior with finite-beam emission backoff",
        },
        "systems": systems,
        "folds": fold_audits,
        "selection_rule": (
            "Top-1 correct, macro-F1, worst-subject Top-1, Top-5 correct, lower "
            "subject-net variance, then VS deterministic tie-break"
        ),
        "selected_system": selected,
        "selected_teacher": selected_teacher,
        "h3_users_loaded": [],
        "h3_rows_selected": 0,
        "student_started": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

