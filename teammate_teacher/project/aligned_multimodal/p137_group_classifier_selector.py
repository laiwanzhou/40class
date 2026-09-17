"""Cross-cohort group classifier selectively augments P136 repeat propagation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import align_metadata, classification_metrics
from p117_transductive_multicandidate_router import load_candidate_splits, one_hot
from p118_candidate_conditioned_router import frozen_embedding_router_features
from p90_crossuser_visual_router import aligned_quality
from p134_frozen_repeat_consensus import CONFIG, METADATA, evidence, probability_lookup
from p136_peer_support_repeat_gate import peer_candidate, select_rule
from p88_aligned_repeat_holdout import align_probabilities
from p89_global_repeat_decoder import cluster_sessions, date_session_lists


HERE = Path(__file__).resolve().parent
P128 = HERE / "runs/p128_base_hierarchical_meta_selector_v1/predictions.npz"
P136 = HERE / "runs/p136_peer_support_repeat_gate_v1/predictions.npz"
OUTPUT = HERE / "runs/p137_group_classifier_selector_v1"


def group_features(
    sample_ids,
    base,
    lookup,
    embedding_lookup=None,
    include_quality_features: bool = False,
    posterior_feature_mode: str = "sqrt",
    group_config=CONFIG,
    group_feature_layout: str = "full",
    teacher_subset: str = "full",
    grouping_lookup=None,
    metadata_path=METADATA,
):
    raw = np.stack([lookup[value] for value in sample_ids.astype(str)])
    subset_indices = {
        "full": tuple(range(raw.shape[1])),
        "core": (0, 1, 2, 3, 4, 5, 6, 7, 8, 18),
        "visual": (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19),
        "nonvisual": (0, 1, 6, 7, 8, 15, 16, 17, 18),
    }
    expanded_names = (
        "thermal",
        "p12_imu",
        "motion_front",
        "skeleton",
        "pose",
        "object",
        "relation",
        "local_depth",
        "egovlp",
        "p142_token",
        "p144_hand_token",
        "p146_workspace_token",
        "p158_lavila_frame_token",
        "dense_blend",
    )
    if raw.shape[1] >= 30:
        for offset, name in enumerate(expanded_names, start=20):
            subset_indices[f"base_plus_{name}"] = (*range(20), offset)
    raw = raw[:, subset_indices[teacher_subset]]
    sqrt_probability = np.sqrt(np.clip(raw, 0.0, 1.0)).reshape(
        len(sample_ids), -1
    )
    hard = raw.argmax(axis=2)
    hard_one_hot = np.zeros_like(raw, dtype=np.float32)
    row_index = np.arange(len(sample_ids))[:, None]
    teacher_index = np.arange(raw.shape[1])[None, :]
    hard_one_hot[row_index, teacher_index, hard] = 1.0
    choices = {
        "sqrt": sqrt_probability,
        "sqrt_hard": np.concatenate(
            (sqrt_probability, hard_one_hot.reshape(len(sample_ids), -1)), axis=1
        ),
        "sqrt_raw": np.concatenate(
            (sqrt_probability, raw.reshape(len(sample_ids), -1)), axis=1
        ),
        "hard": hard_one_hot.reshape(len(sample_ids), -1),
    }
    own = choices[posterior_feature_mode].astype(np.float32)
    grouping_probability, _ = evidence(
        sample_ids,
        base,
        grouping_lookup if grouping_lookup is not None else lookup,
        "vote",
    )
    metadata = align_metadata(metadata_path, sample_ids)
    peers: dict[int, list[int]] = {}
    for sessions in date_session_lists(
        np.arange(len(sample_ids), dtype=np.int64), metadata, 30.0
    ):
        for group in cluster_sessions(
            sessions, grouping_probability, base, metadata, group_config
        ):
            reference = max(group, key=len)
            aligned = {int(row): [int(row)] for row in np.concatenate(group)}
            for session in group:
                if session is reference:
                    continue
                pairs, _ = align_probabilities(
                    grouping_probability[reference],
                    grouping_probability[session],
                    group_config.alignment_gap_penalty,
                )
                for reference_position, other_position in pairs:
                    left = int(reference[reference_position])
                    right = int(session[other_position])
                    aligned[left].append(right)
                    aligned[right].append(left)
            peers.update(aligned)

    aggregate = own.copy()
    base_one_hot = one_hot(base)
    aggregate_base = base_one_hot.copy()
    peer_count = np.zeros(len(base), dtype=np.float32)
    if embedding_lookup is not None:
        embedding = np.stack(
            [embedding_lookup[value] for value in sample_ids.astype(str)]
        ).astype(np.float32)
        aggregate_embedding = embedding.copy()
    if include_quality_features:
        quality, _ = aligned_quality(sample_ids)
        aggregate_quality = quality.copy()
    for row, aligned_rows in peers.items():
        aggregate[row] = own[aligned_rows].mean(axis=0)
        aggregate_base[row] = base_one_hot[aligned_rows].mean(axis=0)
        peer_count[row] = len(aligned_rows) - 1
        if embedding_lookup is not None:
            aggregate_embedding[row] = embedding[aligned_rows].mean(axis=0)
        if include_quality_features:
            aggregate_quality[row] = quality[aligned_rows].mean(axis=0)
    if group_feature_layout == "full":
        matrices = [own, aggregate, aggregate_base]
    elif group_feature_layout == "aggregate":
        matrices = [aggregate, aggregate_base]
    elif group_feature_layout == "own":
        matrices = [own, base_one_hot]
    elif group_feature_layout == "delta":
        matrices = [own, aggregate, aggregate - own, aggregate_base]
    else:
        raise ValueError(f"unknown group feature layout: {group_feature_layout}")
    if embedding_lookup is not None:
        matrices.extend((embedding, aggregate_embedding))
    if include_quality_features:
        matrices.extend(
            (quality, aggregate_quality, quality - aggregate_quality)
        )
    matrices.append(peer_count[:, None])
    return np.concatenate(matrices, axis=1).astype(np.float32)


def fit_probability(
    train_x,
    train_y,
    predict_x,
    regularization_c: float,
    peer_weight: float,
    class_weight_balanced: bool,
    class_frequency_power: float,
):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=regularization_c,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced" if class_weight_balanced else None,
        ),
    )
    sample_weight = np.where(train_x[:, -1] > 0, peer_weight, 1.0).astype(
        np.float64
    )
    if class_frequency_power > 0:
        counts = np.bincount(train_y, minlength=40).astype(np.float64)
        class_weight = (
            len(train_y) / np.maximum(40.0 * counts, 1.0)
        ) ** class_frequency_power
        sample_weight *= class_weight[train_y]
    model.fit(
        train_x,
        train_y,
        logisticregression__sample_weight=sample_weight,
    )
    output = np.zeros((len(predict_x), 40), dtype=np.float64)
    output[:, model.named_steps["logisticregression"].classes_.astype(np.int64)] = (
        model.predict_proba(predict_x)
    )
    return output


def apply_peer_rule(sample_ids, labels_for_selection, base, lookup):
    proposal, scores, peer_count, _ = peer_candidate(sample_ids, base, lookup)
    selected = select_rule(base, labels_for_selection, proposal, scores, peer_count)
    route = (
        (proposal != base)
        & (peer_count > 0)
        & (
            scores[:, int(selected["score_index"])]
            >= float(selected["threshold"])
        )
    )
    prediction = base.copy()
    prediction[route] = proposal[route]
    return prediction, selected


def choose_threshold(base, proposal, probability, labels):
    rows = np.arange(len(base))
    score = probability[rows, proposal] - probability[rows, base]
    disagreement = proposal != base
    values = np.unique(
        np.concatenate(
            (
                np.linspace(-0.10, 0.90, 201),
                np.quantile(score[disagreement], [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]),
            )
        )
    )
    candidates = []
    for threshold in values:
        route = disagreement & (score >= threshold)
        prediction = base.copy()
        prediction[route] = proposal[route]
        rescue = int(np.sum((base != labels) & (prediction == labels)))
        harm = int(np.sum((base == labels) & (prediction != labels)))
        candidates.append(
            {
                "threshold": float(threshold),
                "changed": int(route.sum()),
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
            }
        )
    positive = [row for row in candidates if row["net"] > 0]
    return (
        max(
            positive,
            key=lambda row: (
                row["net"],
                row["rescue"],
                -row["harm"],
                -row["changed"],
            ),
        )
        if positive
        else {"threshold": 2.0, "changed": 0, "rescue": 0, "harm": 0, "net": 0}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularization-c", type=float, default=0.01)
    parser.add_argument("--peer-weight", type=float, default=1.0)
    parser.add_argument("--frozen-embedding-features", action="store_true")
    parser.add_argument("--quality-features", action="store_true")
    parser.add_argument("--class-weight-balanced", action="store_true")
    parser.add_argument("--class-frequency-power", type=float, default=0.0)
    parser.add_argument(
        "--posterior-feature-mode",
        choices=("sqrt", "sqrt_hard", "sqrt_raw", "hard"),
        default="sqrt",
    )
    parser.add_argument(
        "--group-feature-layout",
        choices=("full", "aggregate", "own", "delta"),
        default="full",
    )
    parser.add_argument("--expanded-bank", action="store_true")
    parser.add_argument(
        "--source-group-oof-mode",
        choices=("cross_cohort", "loso_user"),
        default="cross_cohort",
    )
    parser.add_argument(
        "--teacher-subset",
        choices=(
            "full",
            "core",
            "visual",
            "nonvisual",
            "base_plus_thermal",
            "base_plus_p12_imu",
            "base_plus_motion_front",
            "base_plus_skeleton",
            "base_plus_pose",
            "base_plus_object",
            "base_plus_relation",
            "base_plus_local_depth",
            "base_plus_egovlp",
            "base_plus_p142_token",
            "base_plus_p144_hand_token",
            "base_plus_p146_workspace_token",
            "base_plus_p158_lavila_frame_token",
            "base_plus_dense_blend",
        ),
        default="full",
    )
    parser.add_argument("--group-rank", type=int, default=CONFIG.maximum_session_rank_distance)
    parser.add_argument("--group-overlap", type=float, default=CONFIG.minimum_path_overlap)
    parser.add_argument("--group-length", type=float, default=CONFIG.minimum_length_ratio)
    parser.add_argument(
        "--group-similarity", type=float, default=CONFIG.minimum_probability_similarity
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    group_config = replace(
        CONFIG,
        maximum_session_rank_distance=args.group_rank,
        minimum_path_overlap=args.group_overlap,
        minimum_length_ratio=args.group_length,
        minimum_probability_similarity=args.group_similarity,
    )
    core_data = load_candidate_splits(
        full_visual_bank=True,
        structured_bank=True,
        legacy_visual_bank=True,
        hand_object_bank=True,
        vjepa_dense_bank=True,
        nonvisual_bank=True,
        hierarchical_bank=True,
        epic_bank=True,
    )
    data = (
        load_candidate_splits(
            full_visual_bank=True,
            structured_bank=True,
            legacy_visual_bank=True,
            hand_object_bank=True,
            vjepa_dense_bank=True,
            nonvisual_bank=True,
            hierarchical_bank=True,
            epic_bank=True,
            expanded_bank=True,
        )
        if args.expanded_bank
        else core_data
    )
    lookup = probability_lookup(data)
    peer_lookup = probability_lookup(core_data)
    embedding_lookup = (
        frozen_embedding_router_features()
        if args.frozen_embedding_features
        else None
    )
    p128 = np.load(P128)
    p136 = np.load(P136)
    payload = {}
    cohorts = {}
    totals = {"correct": 0, "rescue": 0, "harm": 0, "changed": 0}
    split_names = list(data)

    for held_name in split_names:
        source_names = [name for name in split_names if name != held_name]
        source_ids = p128[f"{held_name}_source_sample_ids"].astype(str)
        source_labels = p128[f"{held_name}_source_labels"].astype(np.int64)
        source_users = p128[f"{held_name}_source_users"].astype(str)
        source_base = p128[f"{held_name}_source_prediction"].astype(np.int64)
        source_position = {
            value: row for row, value in enumerate(source_ids.astype(str))
        }
        source_peer = np.empty_like(source_base)
        source_group_probability = np.zeros((len(source_ids), 40), dtype=np.float64)
        cross_source = []
        train_feature_parts = []
        train_label_parts = []
        source_feature = None

        for target_name in source_names:
            calibration_name = next(
                name for name in source_names if name != target_name
            )
            target_ids = data[target_name].split.sample_ids.astype(str)
            calibration_ids = data[calibration_name].split.sample_ids.astype(str)
            target_rows = np.asarray(
                [source_position[value] for value in target_ids], dtype=np.int64
            )
            calibration_rows = np.asarray(
                [source_position[value] for value in calibration_ids], dtype=np.int64
            )
            target_x = group_features(
                target_ids,
                source_base[target_rows],
                lookup,
                embedding_lookup,
                args.quality_features,
                args.posterior_feature_mode,
                group_config,
                args.group_feature_layout,
                args.teacher_subset,
                peer_lookup,
            )
            calibration_x = group_features(
                calibration_ids,
                source_base[calibration_rows],
                lookup,
                embedding_lookup,
                args.quality_features,
                args.posterior_feature_mode,
                group_config,
                args.group_feature_layout,
                args.teacher_subset,
                peer_lookup,
            )
            source_group_probability[target_rows] = fit_probability(
                calibration_x,
                source_labels[calibration_rows],
                target_x,
                args.regularization_c,
                args.peer_weight,
                args.class_weight_balanced,
                args.class_frequency_power,
            )

            calibration_proposal, calibration_scores, calibration_peers, _ = (
                peer_candidate(
                    calibration_ids, source_base[calibration_rows], peer_lookup
                )
            )
            peer_rule = select_rule(
                source_base[calibration_rows],
                source_labels[calibration_rows],
                calibration_proposal,
                calibration_scores,
                calibration_peers,
            )
            target_proposal, target_scores, target_peers, _ = peer_candidate(
                target_ids, source_base[target_rows], peer_lookup
            )
            target_route = (
                (target_proposal != source_base[target_rows])
                & (target_peers > 0)
                & (
                    target_scores[:, int(peer_rule["score_index"])]
                    >= float(peer_rule["threshold"])
                )
            )
            target_peer = source_base[target_rows].copy()
            target_peer[target_route] = target_proposal[target_route]
            source_peer[target_rows] = target_peer
            cross_source.append(
                {
                    "calibration": calibration_name,
                    "target": target_name,
                    "peer_rule": peer_rule,
                }
            )
            train_feature_parts.append(target_x)
            train_label_parts.append(source_labels[target_rows])
            if source_feature is None:
                source_feature = np.empty(
                    (len(source_ids), target_x.shape[1]), dtype=np.float32
                )
            source_feature[target_rows] = target_x

        if args.source_group_oof_mode == "loso_user":
            assert source_feature is not None
            for user in sorted(set(source_users.tolist())):
                held_user = source_users == user
                source_group_probability[held_user] = fit_probability(
                    source_feature[~held_user],
                    source_labels[~held_user],
                    source_feature[held_user],
                    args.regularization_c,
                    args.peer_weight,
                    args.class_weight_balanced,
                    args.class_frequency_power,
                )

        source_group = source_group_probability.argmax(axis=1).astype(np.int64)
        selected = choose_threshold(
            source_peer,
            source_group,
            source_group_probability,
            source_labels,
        )
        source_rows = np.arange(len(source_ids))
        source_group_score = (
            source_group_probability[source_rows, source_group]
            - source_group_probability[source_rows, source_peer]
        )
        source_route = (source_group != source_peer) & (
            source_group_score >= float(selected["threshold"])
        )
        source_prediction = source_peer.copy()
        source_prediction[source_route] = source_group[source_route]

        sample_ids = p136[f"{held_name}_sample_ids"].astype(str)
        labels = p136[f"{held_name}_labels"].astype(np.int64)
        held_peer = p136[f"{held_name}_prediction"].astype(np.int64)
        held_p128 = p136[f"{held_name}_base_prediction"].astype(np.int64)
        held_x = group_features(
            sample_ids,
            held_p128,
            lookup,
            embedding_lookup,
            args.quality_features,
            args.posterior_feature_mode,
            group_config,
            args.group_feature_layout,
            args.teacher_subset,
            peer_lookup,
        )
        held_probability = fit_probability(
            np.concatenate(train_feature_parts),
            np.concatenate(train_label_parts),
            held_x,
            args.regularization_c,
            args.peer_weight,
            args.class_weight_balanced,
            args.class_frequency_power,
        )
        held_group = held_probability.argmax(axis=1).astype(np.int64)
        rows = np.arange(len(labels))
        held_score = held_probability[rows, held_group] - held_probability[
            rows, held_peer
        ]
        route = (held_group != held_peer) & (
            held_score >= float(selected["threshold"])
        )
        prediction = held_peer.copy()
        prediction[route] = held_group[route]
        rescue = int(np.sum((held_peer != labels) & (prediction == labels)))
        harm = int(np.sum((held_peer == labels) & (prediction != labels)))
        correct = int(np.sum(prediction == labels))
        changed = int(route.sum())
        totals["correct"] += correct
        totals["rescue"] += rescue
        totals["harm"] += harm
        totals["changed"] += changed
        cohorts[held_name] = {
            "cross_source": cross_source,
            "source_threshold": selected,
            "held": {
                "p136_correct": int(np.sum(held_peer == labels)),
                "selected_correct": correct,
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "changed": changed,
                "metrics": classification_metrics(labels, prediction),
            },
        }
        payload[f"{held_name}_sample_ids"] = sample_ids
        payload[f"{held_name}_labels"] = labels
        payload[f"{held_name}_p136_prediction"] = held_peer
        payload[f"{held_name}_group_prediction"] = held_group
        payload[f"{held_name}_group_probability"] = held_probability.astype(np.float32)
        payload[f"{held_name}_group_score"] = held_score
        payload[f"{held_name}_prediction"] = prediction
        payload[f"{held_name}_source_sample_ids"] = source_ids
        payload[f"{held_name}_source_labels"] = source_labels
        payload[f"{held_name}_source_p128_prediction"] = source_base
        payload[f"{held_name}_source_p136_prediction"] = source_peer
        payload[f"{held_name}_source_group_prediction"] = source_group
        payload[f"{held_name}_source_group_probability"] = (
            source_group_probability.astype(np.float32)
        )
        payload[f"{held_name}_source_group_score"] = source_group_score
        payload[f"{held_name}_source_prediction"] = source_prediction

    rows = sum(len(value.split.labels) for value in data.values())
    report = {
        "stage": "P137_group_classifier_selector_v1",
        "status": "complete_strict_outer_crossfit",
        "protocol": {
            "group_head": (
                "StandardScaler + LogisticRegression "
                f"C={args.regularization_c:g}"
            ),
            "aligned_peer_source_weight": args.peer_weight,
            "class_weight_balanced": args.class_weight_balanced,
            "class_frequency_power": args.class_frequency_power,
            "frozen_embedding_features": args.frozen_embedding_features,
            "frozen_embedding_projection_is_label_free": True,
            "quality_features": args.quality_features,
            "posterior_feature_mode": args.posterior_feature_mode,
            "group_geometry": {
                "maximum_session_rank_distance": group_config.maximum_session_rank_distance,
                "minimum_path_overlap": group_config.minimum_path_overlap,
                "minimum_length_ratio": group_config.minimum_length_ratio,
                "minimum_probability_similarity": group_config.minimum_probability_similarity,
            },
            "group_feature_layout": args.group_feature_layout,
            "teacher_subset": args.teacher_subset,
            "expanded_bank": args.expanded_bank,
            "candidate_count_including_safe": 1
            + len(next(iter(data.values())).candidates),
            "source_predictions": "two source cohorts predict each other",
            "source_group_oof_mode": args.source_group_oof_mode,
            "user_id_used_for_source_LOSO_only": (
                args.source_group_oof_mode == "loso_user"
            ),
            "held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": cohorts,
        "aggregate": {
            "rows": rows,
            **totals,
            "accuracy": totals["correct"] / rows,
            "net_vs_p136": totals["rescue"] - totals["harm"],
            "target_0.91_correct": int(np.ceil(0.91 * rows)),
            "gap_to_0.91_correct": int(np.ceil(0.91 * rows)) - totals["correct"],
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
