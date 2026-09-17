"""Strict source-OOF routing of new Depth/Thermal visual-token teachers.

The primary decision is the already frozen P90 cross-user visual router.  New
candidate decisions are trained from raw IR/Depth/Thermal teacher tokens.  For
H1+H2, every candidate logit and every routing score is leave-one-user-out.
The route threshold is selected there and applied once to untouched H3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax

from p90_crossuser_visual_router import build_features, load_splits
from p90_teacher_common import load_protocol
from p90_teacher_fusion_audit import align
from p90_videomaev2_distilled_teacher import class_sample_weights
from p91_sensor_residual_router import (
    Split,
    apply_route,
    concatenate,
    entropy,
    fit_candidate,
    fit_reduce,
    margin,
    one_hot,
    oof_scores,
    route_audit,
    select_threshold,
)
from train_p46_videomae_head import l2_normalize, make_model


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "runs/p91_visual_token_router_h3_v1"
VMAE = PROJECT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz"
IV2 = PROJECT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz"
DEPTH = PROJECT / "runs/p91_videomaev2_depth_fold0_v1/complete_features.npz"
THERMAL = PROJECT / "runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz"
ROUTER = PROJECT / "runs/p90_crossuser_visual_router_v1/full_predictions.npz"
EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def normalized_flat(data: dict[str, np.ndarray]) -> np.ndarray:
    values = l2_normalize(np.asarray(data["features"], dtype=np.float32))
    return values.reshape(len(values), -1)


def normalized_mean(data: dict[str, np.ndarray]) -> np.ndarray:
    values = l2_normalize(np.asarray(data["features"], dtype=np.float32))
    return l2_normalize(values.mean(axis=1)).reshape(len(values), -1)


def fit_logits(
    values: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = make_model(alpha)
    model.fit(
        values[train],
        labels[train],
        ridge__sample_weight=class_sample_weights(labels[train], 0.75),
    )
    return np.asarray(model.decision_function(values[target]), dtype=np.float32)


def candidate_logits(
    args: argparse.Namespace,
) -> tuple[
    list[str],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    protocol = load_protocol()
    raw = load_splits()
    source_ids = np.concatenate(
        (raw["H1_selection"].sample_ids, raw["H2_confirmation"].sample_ids)
    ).astype(str)
    source_users = np.concatenate(
        (raw["H1_selection"].users, raw["H2_confirmation"].users)
    ).astype(str)
    target_ids = raw["H3_independent_fold0"].sample_ids.astype(str)
    full_ids = protocol.sample_ids.astype(str)
    source_index = align(full_ids, np.arange(len(full_ids)), source_ids).astype(np.int64)
    target_index = align(full_ids, np.arange(len(full_ids)), target_ids).astype(np.int64)
    train_pool = np.zeros(len(full_ids), dtype=bool)
    train_pool[protocol.train_indices(0)] = True

    data = {name: load_npz(path) for name, path in (
        ("vmae", VMAE), ("iv2", IV2), ("depth", DEPTH), ("thermal", THERMAL)
    )}
    for name, item in data.items():
        if not np.array_equal(item["sample_ids"].astype(str), full_ids):
            raise ValueError(f"{name} cache order differs from P90 protocol")
    all_values = {name: normalized_flat(item) for name, item in data.items()}
    mean_values = {name: normalized_mean(item) for name, item in data.items()}
    matrices = {
        "ir_depth": np.concatenate(
            (all_values["vmae"], all_values["iv2"], all_values["depth"]), axis=1
        ),
        "ir_thermal": np.concatenate(
            (all_values["vmae"], all_values["iv2"], all_values["thermal"]), axis=1
        ),
        "ir_depth_thermal": np.concatenate(
            (
                all_values["vmae"],
                all_values["iv2"],
                all_values["depth"],
                all_values["thermal"],
            ),
            axis=1,
        ),
        "ir_depth_thermal_mean": np.concatenate(
            (
                mean_values["vmae"],
                mean_values["iv2"],
                mean_values["depth"],
                mean_values["thermal"],
            ),
            axis=1,
        ),
    }
    alphas = {
        "ir_depth": 3000.0,
        "ir_thermal": 3000.0,
        "ir_depth_thermal": 3000.0,
        "ir_depth_thermal_mean": 1000.0,
    }
    names = list(matrices)
    source_logits = {name: np.zeros((len(source_ids), 40), dtype=np.float32) for name in names}
    target_logits: dict[str, np.ndarray] = {}
    for user_number, user in enumerate(sorted(np.unique(source_users).tolist())):
        target_rows = np.flatnonzero(source_users == user)
        held_full = source_index[target_rows]
        train = train_pool.copy()
        train[held_full] = False
        print(
            f"  candidate OOF user={user} rows={len(target_rows)} "
            f"({user_number + 1}/{len(np.unique(source_users))})",
            flush=True,
        )
        for name in names:
            source_logits[name][target_rows] = fit_logits(
                matrices[name], protocol.labels, train, held_full, alphas[name]
            )
    final_train = train_pool
    for name in names:
        print(f"  candidate H3 fit={name} dim={matrices[name].shape[1]}", flush=True)
        target_logits[name] = fit_logits(
            matrices[name], protocol.labels, final_train, target_index, alphas[name]
        )
    raw_pca = {
        "depth": all_values["depth"],
        "thermal": all_values["thermal"],
    }
    return names, source_logits, target_logits, raw_pca


def split_gate_features(
    args: argparse.Namespace,
    candidate_names: list[str],
    source_logits: dict[str, np.ndarray],
    target_logits: dict[str, np.ndarray],
    raw_pca: dict[str, np.ndarray],
) -> dict[str, Split]:
    raw = load_splits()
    names = ["H1_selection", "H2_confirmation", "H3_independent_fold0"]
    router = load_npz(ROUTER)
    source_count = len(raw["H1_selection"].labels)
    source_ids = np.concatenate(
        (raw["H1_selection"].sample_ids, raw["H2_confirmation"].sample_ids)
    ).astype(str)
    target_ids = raw["H3_independent_fold0"].sample_ids.astype(str)
    full_ids = load_protocol().sample_ids.astype(str)

    logit_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split_name in names:
        if split_name == "H1_selection":
            chosen = slice(0, source_count)
            logit_by_split[split_name] = {key: value[chosen] for key, value in source_logits.items()}
        elif split_name == "H2_confirmation":
            chosen = slice(source_count, None)
            logit_by_split[split_name] = {key: value[chosen] for key, value in source_logits.items()}
        else:
            logit_by_split[split_name] = target_logits

    reduced_blocks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for block_number, (block_name, full_values) in enumerate(raw_pca.items()):
        source_values = align(full_ids, full_values, source_ids)
        target_values = align(full_ids, full_values, target_ids)
        reduced = fit_reduce(
            source_values,
            [
                source_values[:source_count],
                source_values[source_count:],
                target_values,
            ],
            args.pca_dim,
            args.seed + block_number,
        )
        for split_name, values in zip(names, reduced):
            reduced_blocks[split_name].append(values)

    output: dict[str, Split] = {}
    for split_name in names:
        split = raw[split_name]
        base = router[f"{split_name}_router_prediction"].astype(np.int64)
        route_score = router[f"{split_name}_route_score"].astype(np.float32)
        visual_features, _ = build_features(split)
        matrices = [visual_features.astype(np.float32), one_hot(base), route_score[:, None]]
        predictions = []
        for candidate_name in candidate_names:
            probability = softmax(logit_by_split[split_name][candidate_name], axis=1)
            prediction = probability.argmax(axis=1).astype(np.int64)
            predictions.append(prediction)
            matrices.extend(
                [
                    np.log(np.clip(probability, EPSILON, 1.0)).astype(np.float32),
                    one_hot(prediction),
                    np.column_stack(
                        (
                            probability.max(axis=1),
                            margin(probability),
                            entropy(probability),
                            prediction != base,
                            probability[np.arange(len(base)), prediction],
                            probability[np.arange(len(base)), base],
                        )
                    ).astype(np.float32),
                ]
            )
        matrices.extend(reduced_blocks[split_name])
        output[split_name] = Split(
            name=split_name,
            sample_ids=split.sample_ids.astype(str),
            labels=split.labels.astype(np.int64),
            users=split.users.astype(str),
            base_prediction=base,
            candidate_names=candidate_names,
            candidate_prediction=np.stack(predictions, axis=1),
            features=np.concatenate(matrices, axis=1).astype(np.float32),
        )
    return output


def oracle_audit(split: Split) -> dict[str, Any]:
    correct = split.base_prediction == split.labels
    by_candidate = {}
    for index, name in enumerate(split.candidate_names):
        candidate_correct = split.candidate_prediction[:, index] == split.labels
        by_candidate[name] = {
            "accuracy": float(candidate_correct.mean()),
            "rescue_over_base": int(np.sum(~correct & candidate_correct)),
            "harm_if_replaced": int(np.sum(correct & ~candidate_correct)),
            "union_oracle_accuracy": float(np.mean(correct | candidate_correct)),
        }
        correct |= candidate_correct
    return {
        "base_accuracy": float(np.mean(split.base_prediction == split.labels)),
        "candidate_diagnostics": by_candidate,
        "all_candidate_union_oracle_accuracy": float(correct.mean()),
        "all_candidate_union_oracle_correct": int(correct.sum()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_names, source_logits, target_logits, raw_pca = candidate_logits(args)
    splits = split_gate_features(
        args, candidate_names, source_logits, target_logits, raw_pca
    )
    source = concatenate(splits["H1_selection"], splits["H2_confirmation"])
    target = splits["H3_independent_fold0"]
    gate_scores = oof_scores(source, args)
    threshold, threshold_grid = select_threshold(source, gate_scores)
    source_prediction = apply_route(source, gate_scores, threshold)
    target_scores = np.zeros((len(target.labels), len(candidate_names)), dtype=np.float64)
    all_source = np.ones(len(source.labels), dtype=bool)
    for candidate_index in range(len(candidate_names)):
        target_scores[:, candidate_index] = fit_candidate(
            source.features,
            source.base_prediction,
            source.candidate_prediction[:, candidate_index],
            source.labels,
            all_source,
            target.features,
            args.seed + 10000 + candidate_index,
            args.trees,
        )
    target_prediction = apply_route(target, target_scores, threshold)
    source_audit = route_audit(source, source_prediction)
    target_audit = route_audit(target, target_prediction)
    report = {
        "protocol": (
            "Candidate classifiers and visual-token route scores are leave-one-user-out "
            "on H1+H2; threshold selected on H1+H2 only; H3 evaluated once."
        ),
        "candidate_names": candidate_names,
        "feature_dimensions": int(source.features.shape[1]),
        "threshold": threshold,
        "source_oof": source_audit,
        "target_h3": target_audit,
        "source_oracle": oracle_audit(source),
        "target_oracle": oracle_audit(target),
        "top_thresholds": sorted(
            threshold_grid,
            key=lambda row: (row["net"], -row["negative_users"], row["worst_user_net"]),
            reverse=True,
        )[:20],
    }
    np.savez_compressed(
        output / "predictions.npz",
        source_sample_ids=source.sample_ids,
        source_labels=source.labels,
        source_base_prediction=source.base_prediction,
        source_candidate_prediction=source.candidate_prediction,
        source_gate_scores=gate_scores.astype(np.float32),
        source_prediction=source_prediction,
        target_sample_ids=target.sample_ids,
        target_labels=target.labels,
        target_base_prediction=target.base_prediction,
        target_candidate_prediction=target.candidate_prediction,
        target_gate_scores=target_scores.astype(np.float32),
        target_prediction=target_prediction,
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
