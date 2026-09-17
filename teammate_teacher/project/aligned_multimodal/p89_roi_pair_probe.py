from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from audit_p87_sequence_decoder import (
    DEFAULT_TEACHER,
    DEFAULT_TRAIN_METADATA,
    DecoderConfig,
    align_metadata,
    build_sessions,
    classification_metrics,
    fit_transition_model,
)
from p88_aligned_repeat_holdout import AlignedRepeatConfig, decode_aligned_repeat
from p88_train_depth_residual import log_softmax_numpy


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_DIR / "runs" / "p89_roi_probe_cache_v1" / "roi_probe_features.npz"
DEFAULT_H1_RUN = PROJECT_DIR / "runs" / "p87s_fusion_holdout1_c7_structured12_v1"
DEFAULT_REPEAT = PROJECT_DIR / "runs" / "p88_aligned_repeat_h1_v1" / "summary.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p89_roi_pair_probe_h1_v1"

# Defined from action semantics and repeated confusion families, before looking
# at this probe's predictions.  They cover local object/hand ambiguities only.
PAIR_FAMILIES = (
    (6, 7), (6, 37), (6, 38),
    (7, 8), (7, 10), (7, 11),
    (8, 10), (8, 14), (8, 18),
    (9, 10), (10, 11), (10, 14),
    (17, 18), (18, 21), (18, 22),
    (19, 23), (19, 24), (20, 25),
    (20, 24), (21, 22), (23, 34),
    (24, 26), (37, 39), (37, 6),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P89 subject-disjoint local-ROI pairwise residual probe.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_H1_RUN)
    parser.add_argument("--repeat-summary", type=Path, default=DEFAULT_REPEAT)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-users", nargs="+", default=("user6", "user8", "user17", "user23"))
    parser.add_argument("--fixed-summary", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_id(p87_id: str) -> str:
    match = re.fullmatch(r"train__c(\d+)__(user\d+)__(.+)", p87_id)
    if match is None:
        raise ValueError(f"unexpected P87 sample id: {p87_id}")
    class_id, user, trial = int(match.group(1)), match.group(2), match.group(3)
    return f"{class_id}/{user}/{trial}"


def feature_key(cache_id: str) -> str:
    class_name, user, trial = cache_id.split("/")
    return f"{int(class_name.split('_', 1)[0])}/{user}/{trial}"


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {
        **classification_metrics(labels, predictions),
        "balanced_accuracy_sklearn": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1_sklearn": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def load_base(
    run_dir: Path,
    holdout_users: list[str],
    repeat_summary: Path,
    teacher_targets: Path,
    train_metadata: Path,
) -> dict[str, Any]:
    rows = read_csv(run_dir / "subject_holdout_predictions.csv")
    ids = np.asarray([canonical_id(row["sample_id"]) for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in rows])
    if set(users) != set(holdout_users):
        raise RuntimeError(f"holdout mismatch: {sorted(set(users))} != {sorted(holdout_users)}")
    logp = log_softmax_numpy(np.asarray(np.load(run_dir / "subject_holdout_logits.npy"), dtype=np.float64))

    audit = json.loads((run_dir / "decoder_audit.json").read_text(encoding="utf-8"))
    frozen = audit["selected_decoder_config"]
    decoder = DecoderConfig(
        float(frozen["gap_seconds"]), float(frozen["transition_weight"]),
        float(frozen["trigram_backoff"]), int(frozen["beam_width"]),
    )
    repeat_source = json.loads(repeat_summary.read_text(encoding="utf-8"))
    repeat = AlignedRepeatConfig(**repeat_source["best"]["config"])
    with np.load(teacher_targets, allow_pickle=False) as source:
        all_ids = source["oof_sample_ids"].astype(str)
        all_labels = source["oof_labels"].astype(np.int64)
    all_metadata = align_metadata(train_metadata, all_ids)
    fit = np.flatnonzero(~np.isin(all_metadata.users, holdout_users))
    transition = fit_transition_model(
        all_labels,
        build_sessions(fit, all_metadata, decoder.gap_seconds, "known_user"),
        40,
        decoder.trigram_backoff,
    )
    metadata = align_metadata(train_metadata, np.asarray([row["sample_id"] for row in rows]))
    indices = np.arange(len(labels))
    prediction, grouping = decode_aligned_repeat(logp, indices, metadata, transition, decoder, repeat)
    return {
        "ids": ids, "labels": labels, "users": users, "logp": logp,
        "prediction": prediction, "grouping": grouping,
    }


def train_pair_models(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    c_value: float,
) -> dict[tuple[int, int], Any]:
    models: dict[tuple[int, int], Any] = {}
    for pair in dict.fromkeys(tuple(sorted(pair)) for pair in PAIR_FAMILIES):
        selected = train[np.isin(y[train], pair)]
        if len(selected) < 12 or len(np.unique(y[selected])) != 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(C=c_value, class_weight="balanced", dual="auto", max_iter=5000),
        )
        model.fit(x[selected], y[selected])
        models[pair] = model
    return models


def proposals(
    models: dict[tuple[int, int], Any],
    x: np.ndarray,
    logp: np.ndarray,
    base: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alternative = np.full(len(x), -1, dtype=np.int64)
    strength = np.full(len(x), -np.inf, dtype=np.float64)
    base_gap = np.full(len(x), np.inf, dtype=np.float64)
    top = np.argsort(-logp, axis=1)[:, :top_k]
    for pair, model in models.items():
        first, second = pair
        eligible = np.flatnonzero(
            np.isin(base, pair)
            & np.asarray([first in row and second in row for row in top], dtype=bool)
        )
        if not len(eligible):
            continue
        decision = np.asarray(model.decision_function(x[eligible]), dtype=np.float64)
        scale = max(float(np.std(decision)), 1e-6)
        predicted = np.asarray(model.predict(x[eligible]), dtype=np.int64)
        confidence = np.abs(decision) / scale
        alternate = np.where(base[eligible] == first, second, first)
        wants_change = predicted == alternate
        gap = np.abs(logp[eligible, first] - logp[eligible, second])
        for local, sample in enumerate(eligible):
            if wants_change[local] and confidence[local] > strength[sample]:
                alternative[sample] = int(alternate[local])
                strength[sample] = float(confidence[local])
                base_gap[sample] = float(gap[local])
    return alternative, strength, base_gap


def changes(labels: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    rescue = int(np.sum((base != labels) & (candidate == labels)))
    harm = int(np.sum((base == labels) & (candidate != labels)))
    return {
        "rescues": rescue,
        "harms": harm,
        "net": rescue - harm,
        "changed": int(np.sum(base != candidate)),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.features.resolve(), allow_pickle=False) as source:
        cache_ids = source["sample_ids"].astype(str)
        y = source["labels"].astype(np.int64)
        users = source["users"].astype(str)
        x = source["features"].astype(np.float32)
    lookup = {feature_key(sample_id): index for index, sample_id in enumerate(cache_ids)}
    base = load_base(
        args.run_dir.resolve(), list(args.holdout_users), args.repeat_summary.resolve(),
        args.teacher_targets.resolve(), args.train_metadata.resolve(),
    )
    eval_indices = np.asarray([lookup[sample_id] for sample_id in base["ids"]], dtype=np.int64)
    if not np.array_equal(y[eval_indices], base["labels"]):
        raise RuntimeError("P87/cache label alignment failed")
    train = np.flatnonzero(~np.isin(users, args.holdout_users))

    if args.fixed_summary:
        selected = json.loads(args.fixed_summary.resolve().read_text(encoding="utf-8"))["selected_config"]
        c_values = [float(selected["c_value"])]
        top_ks = [int(selected["top_k"])]
        margins = [float(selected["minimum_specialist_margin"])]
        gaps = [float(selected["maximum_base_pair_gap"])]
        selected_here = False
    else:
        c_values = [0.0003, 0.001, 0.003, 0.01]
        top_ks = [2, 3, 5]
        margins = [0.0, 0.35, 0.7, 1.0, 1.5]
        gaps = [0.35, 0.7, 1.2, 2.0, 4.0]
        selected_here = True

    evaluations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for c_value in c_values:
        models = train_pair_models(x, y, train, c_value)
        for top_k in top_ks:
            alternative, strength, base_gap = proposals(
                models, x[eval_indices], base["logp"], base["prediction"], top_k
            )
            for margin in margins:
                for gap in gaps:
                    prediction = base["prediction"].copy()
                    switch = (alternative >= 0) & (strength >= margin) & (base_gap <= gap)
                    prediction[switch] = alternative[switch]
                    item = {
                        "config": {
                            "c_value": c_value,
                            "top_k": top_k,
                            "minimum_specialist_margin": margin,
                            "maximum_base_pair_gap": gap,
                        },
                        "metrics": metric(base["labels"], prediction),
                        "changes": changes(base["labels"], base["prediction"], prediction),
                    }
                    evaluations.append(item)
                    key = (
                        item["metrics"]["correct"], item["changes"]["net"],
                        -item["changes"]["harms"], -item["changes"]["changed"],
                    )
                    if best_key is None or key > best_key:
                        best_key, best = key, item
        print(f"finished C={c_value:g}", flush=True)

    assert best is not None
    summary = {
        "stage": "P89_local_ROI_pairwise_residual_probe_v1",
        "status": "complete",
        "selected_on_current_holdout": selected_here,
        "holdout_users": sorted(args.holdout_users),
        "train_samples": int(len(train)),
        "eval_samples": int(len(eval_indices)),
        "feature_dim": int(x.shape[1]),
        "pair_families": [list(pair) for pair in dict.fromkeys(tuple(sorted(pair)) for pair in PAIR_FAMILIES)],
        "base": metric(base["labels"], base["prediction"]),
        "base_grouping": base["grouping"],
        "selected_config": best["config"],
        "best": best,
        "grid_size": len(evaluations),
        "all_candidates": evaluations,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "all_candidates"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
