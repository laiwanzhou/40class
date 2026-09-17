"""Outer-safe selector between the P118 base and CLIP-augmented routers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import load_candidate_splits, one_hot


HERE = Path(__file__).resolve().parent
BASE = HERE / "runs/p118_candidate_conditioned_structured_maxnet_v2"
CLIP = HERE / "runs/p120_clip_augmented_structured_router_v1"
OUTPUT = HERE / "runs/p120_base_clip_router_meta_selector_v1"


def candidate_labels(
    sample_ids: np.ndarray,
    candidate_index: np.ndarray,
    bank,
    candidate_names: list[str],
    lookup,
) -> np.ndarray:
    output = np.full(len(sample_ids), -1, dtype=np.int64)
    for row, (sample_id, index) in enumerate(
        zip(sample_ids.astype(str), candidate_index.astype(np.int64))
    ):
        if index < 0:
            continue
        split_name, position = lookup[sample_id]
        output[row] = int(
            bank[split_name].candidates[candidate_names[index]][position].argmax()
        )
    return output


def router_output(
    safe: np.ndarray,
    candidate: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> np.ndarray:
    output = safe.copy()
    route = (candidate >= 0) & (candidate != safe) & (score >= threshold)
    output[route] = candidate[route]
    return output


def meta_features(
    safe: np.ndarray,
    base_output: np.ndarray,
    clip_output: np.ndarray,
    base_score: np.ndarray,
    clip_score: np.ndarray,
    base_threshold: float,
    clip_threshold: float,
    base_index: np.ndarray,
    clip_index: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    finite_base = np.where(np.isfinite(base_score), base_score, -1.0)
    finite_clip = np.where(np.isfinite(clip_score), clip_score, -1.0)
    base_identity = np.zeros((len(safe), candidate_count), dtype=np.float32)
    clip_identity = np.zeros_like(base_identity)
    valid = base_index >= 0
    base_identity[np.arange(len(safe))[valid], base_index[valid]] = 1.0
    valid = clip_index >= 0
    clip_identity[np.arange(len(safe))[valid], clip_index[valid]] = 1.0
    scalars = np.column_stack(
        (
            finite_base,
            finite_clip,
            finite_base - base_threshold,
            finite_clip - clip_threshold,
            finite_clip - finite_base,
            base_output != safe,
            clip_output != safe,
            base_output == clip_output,
            base_index == clip_index,
        )
    ).astype(np.float32)
    return np.concatenate(
        (
            scalars,
            base_identity,
            clip_identity,
            one_hot(safe),
            one_hot(base_output),
            one_hot(clip_output),
        ),
        axis=1,
    ).astype(np.float32)


def fit_score(train_x: np.ndarray, target: np.ndarray, predict_x: np.ndarray) -> np.ndarray:
    members = []
    models = (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.03, solver="liblinear", max_iter=1200),
        ),
        ExtraTreesClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=12031,
            n_jobs=-1,
        ),
    )
    for model in models:
        model.fit(train_x, target)
        members.append(model.predict_proba(predict_x)[:, 1])
    return np.mean(np.stack(members), axis=0)


def threshold_report(
    score: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    base_output: np.ndarray,
    clip_output: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    switch = (base_output != clip_output) & (score >= threshold)
    output = base_output.copy()
    output[switch] = clip_output[switch]
    per_user = {
        user: int(
            np.sum(output[users.astype(str) == user] == labels[users.astype(str) == user])
            - np.sum(
                base_output[users.astype(str) == user]
                == labels[users.astype(str) == user]
            )
        )
        for user in sorted(set(users.astype(str).tolist()))
    }
    return {
        "threshold": float(threshold),
        "correct": int(np.sum(output == labels)),
        "base_correct": int(np.sum(base_output == labels)),
        "net_vs_base": int(np.sum(output == labels) - np.sum(base_output == labels)),
        "switches": int(switch.sum()),
        "minimum_user_gain": int(min(per_user.values())),
        "positive_users": int(sum(value > 0 for value in per_user.values())),
        "per_user_gain": per_user,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alternative", type=Path, default=CLIP)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--alternative-name", type=str, default="clip")
    args = parser.parse_args()
    alternative_root = args.alternative.resolve()
    output_dir = args.output_dir.resolve()
    base_summary = json.loads((BASE / "summary.json").read_text(encoding="utf-8"))
    clip_summary = json.loads((alternative_root / "summary.json").read_text(encoding="utf-8"))
    base_saved = np.load(BASE / "predictions.npz")
    clip_saved = np.load(alternative_root / "predictions.npz")
    alternative_protocol = clip_summary.get("protocol", {})
    bank = load_candidate_splits(
        full_visual_bank=True,
        structured_bank=True,
        legacy_visual_bank=bool(alternative_protocol.get("legacy_visual_bank", False)),
        hand_object_bank=bool(alternative_protocol.get("hand_object_bank", False)),
        vjepa_dense_bank=bool(alternative_protocol.get("vjepa_dense_bank", False)),
        hierarchical_bank=bool(alternative_protocol.get("hierarchical_bank", False)),
        epic_bank=bool(alternative_protocol.get("epic_bank", False)),
        egovlp_bank=bool(alternative_protocol.get("egovlp_bank", False)),
    )
    base_candidate_names = list(base_summary["protocol"]["candidate_names"])
    alternative_candidate_names = list(clip_summary["protocol"]["candidate_names"])
    candidate_identity_count = max(
        len(base_candidate_names), len(alternative_candidate_names)
    )
    lookup = {
        sample_id: (split_name, row)
        for split_name, value in bank.items()
        for row, sample_id in enumerate(value.split.sample_ids.astype(str))
    }
    reports = {}
    payload = {}
    total_base = total_selected = 0
    for held_name, held in bank.items():
        source_ids = base_saved[f"{held_name}_source_sample_ids"]
        if not np.array_equal(source_ids, clip_saved[f"{held_name}_source_sample_ids"]):
            raise ValueError("source IDs differ between routers")
        labels = base_saved[f"{held_name}_source_labels"]
        users = base_saved[f"{held_name}_source_users"]
        safe = base_saved[f"{held_name}_source_safe_prediction"]
        base_threshold = float(
            base_summary["cohorts"][held_name]["selected_threshold"]["threshold"]
        )
        clip_threshold = float(
            clip_summary["cohorts"][held_name]["selected_threshold"]["threshold"]
        )
        base_candidate = candidate_labels(
            source_ids,
            base_saved[f"{held_name}_source_candidate_index"],
            bank,
            base_candidate_names,
            lookup,
        )
        clip_candidate = candidate_labels(
            source_ids,
            clip_saved[f"{held_name}_source_candidate_index"],
            bank,
            alternative_candidate_names,
            lookup,
        )
        base_output = router_output(
            safe,
            base_candidate,
            base_saved[f"{held_name}_source_route_score"],
            base_threshold,
        )
        clip_output = router_output(
            safe,
            clip_candidate,
            clip_saved[f"{held_name}_source_route_score"],
            clip_threshold,
        )
        source_x = meta_features(
            safe,
            base_output,
            clip_output,
            base_saved[f"{held_name}_source_route_score"],
            clip_saved[f"{held_name}_source_route_score"],
            base_threshold,
            clip_threshold,
            base_saved[f"{held_name}_source_candidate_index"],
            clip_saved[f"{held_name}_source_candidate_index"],
            candidate_identity_count,
        )
        disagreement = base_output != clip_output
        decisive = disagreement & ((base_output == labels) != (clip_output == labels))
        nested_score = np.zeros(len(labels), dtype=np.float64)
        for user in sorted(set(users.astype(str).tolist())):
            held_user = users.astype(str) == user
            train = decisive & ~held_user
            valid = held_user & disagreement
            target = (clip_output[train] == labels[train]).astype(np.int64)
            if valid.any() and len(np.unique(target)) == 2:
                nested_score[valid] = fit_score(source_x[train], target, source_x[valid])
        threshold_grid = [
            threshold_report(
                nested_score, labels, users, base_output, clip_output, threshold
            )
            for threshold in np.arange(0.20, 0.851, 0.025)
        ]
        threshold_grid.sort(
            key=lambda row: (
                row["net_vs_base"],
                row["minimum_user_gain"],
                row["positive_users"],
                -row["switches"],
            ),
            reverse=True,
        )
        selected = threshold_grid[0]
        source_switch = (base_output != clip_output) & (
            nested_score >= selected["threshold"]
        )
        source_meta_output = base_output.copy()
        source_meta_output[source_switch] = clip_output[source_switch]

        held_labels = held.split.labels
        held_safe = held.split.safe_prediction
        held_base = base_saved[f"{held_name}_router_prediction"]
        held_clip = clip_saved[f"{held_name}_router_prediction"]
        held_x = meta_features(
            held_safe,
            held_base,
            held_clip,
            base_saved[f"{held_name}_route_score"],
            clip_saved[f"{held_name}_route_score"],
            base_threshold,
            clip_threshold,
            base_saved[f"{held_name}_candidate_index"],
            clip_saved[f"{held_name}_candidate_index"],
            candidate_identity_count,
        )
        train_target = (clip_output[decisive] == labels[decisive]).astype(np.int64)
        held_score = fit_score(source_x[decisive], train_target, held_x)
        switch = (held_base != held_clip) & (held_score >= selected["threshold"])
        output = held_base.copy()
        output[switch] = held_clip[switch]
        held_report = threshold_report(
            held_score,
            held_labels,
            held.split.users,
            held_base,
            held_clip,
            selected["threshold"],
        )
        reports[held_name] = {
            "source_decisive_rows": int(decisive.sum()),
            "source_clip_wins": int(np.sum(decisive & (clip_output == labels))),
            "source_base_wins": int(np.sum(decisive & (base_output == labels))),
            "selected_threshold": selected,
            "held_result": held_report,
            "held_metrics": classification_metrics(held_labels, output),
            "top_source_thresholds": threshold_grid[:10],
        }
        total_base += int(np.sum(held_base == held_labels))
        total_selected += int(np.sum(output == held_labels))
        payload[f"{held_name}_sample_ids"] = held.split.sample_ids
        payload[f"{held_name}_labels"] = held_labels
        payload[f"{held_name}_base_prediction"] = held_base
        payload[f"{held_name}_clip_prediction"] = held_clip
        payload[f"{held_name}_prediction"] = output
        payload[f"{held_name}_meta_score"] = held_score
        payload[f"{held_name}_source_sample_ids"] = source_ids
        payload[f"{held_name}_source_labels"] = labels
        payload[f"{held_name}_source_users"] = users
        payload[f"{held_name}_source_safe_prediction"] = safe
        payload[f"{held_name}_source_base_prediction"] = base_output
        payload[f"{held_name}_source_alternative_prediction"] = clip_output
        payload[f"{held_name}_source_prediction"] = source_meta_output
        payload[f"{held_name}_source_meta_score"] = nested_score
    report = {
        "stage": f"P120_base_vs_{args.alternative_name}_router_outer_meta_selector_v1",
        "status": "complete",
        "protocol": {
            "source_meta_training": "nested LOUO router outputs from outer-training cohorts",
            "outer_held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": 2470,
            "base_correct": total_base,
            "correct": total_selected,
            "accuracy": total_selected / 2470,
            "net_vs_base": total_selected - total_base,
            "gap_to_0.91_correct": 2248 - total_selected,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "predictions.npz", **payload)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
