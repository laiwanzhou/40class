from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from audit_p87_sequence_decoder import (
    DEFAULT_TEACHER,
    DEFAULT_TRAIN_METADATA,
    DecoderConfig,
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
    fit_transition_model,
)
from p88_train_depth_residual import log_softmax_numpy, read_rows, rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_P12 = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_H1 = PROJECT_DIR / "runs/p87s_fusion_holdout1_c7_structured12_v1"
DEFAULT_H2 = PROJECT_DIR / "runs/p87s_fusion_confirm2_c2_structured12_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_thermal_router_h1_to_h2_v1"
H1_USERS = ("user6", "user8", "user17", "user23")
H2_USERS = ("user5", "user7", "user16", "user18", "user19")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Thermal rescue router by H1 user-LOSO, then confirm the frozen "
            "router once on H2. P87 and Thermal experts stay frozen."
        )
    )
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12)
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--h1-run", type=Path, default=DEFAULT_H1)
    parser.add_argument("--h2-run", type=Path, default=DEFAULT_H2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c-values", type=float, nargs="+", default=(0.01, 0.03, 0.1, 0.3, 1.0))
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95))
    parser.add_argument("--blend-weights", type=float, nargs="+", default=(0.50, 0.75, 1.0))
    return parser.parse_args()


def probability(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_numpy(logits))


def top_margin(values: np.ndarray) -> np.ndarray:
    top = np.partition(values, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def entropy(values: np.ndarray) -> np.ndarray:
    return -(values * np.log(np.maximum(values, 1e-12))).sum(axis=1)


def router_features(anchor_logits: np.ndarray, thermal_logits: np.ndarray) -> np.ndarray:
    anchor_probability = probability(anchor_logits)
    thermal_probability = probability(thermal_logits)
    anchor_prediction = anchor_probability.argmax(axis=1)
    thermal_prediction = thermal_probability.argmax(axis=1)
    rows = np.arange(len(anchor_prediction))
    continuous = np.column_stack(
        (
            anchor_probability.max(axis=1),
            top_margin(anchor_probability),
            entropy(anchor_probability),
            thermal_probability.max(axis=1),
            top_margin(thermal_probability),
            entropy(thermal_probability),
            anchor_probability[rows, thermal_prediction],
            thermal_probability[rows, anchor_prediction],
            thermal_probability.max(axis=1) - anchor_probability.max(axis=1),
            (anchor_prediction != thermal_prediction).astype(np.float64),
        )
    )
    anchor_onehot = np.eye(40, dtype=np.float64)[anchor_prediction]
    thermal_onehot = np.eye(40, dtype=np.float64)[thermal_prediction]
    return np.concatenate((continuous, anchor_onehot, thermal_onehot), axis=1)


def load_run(run_dir: Path) -> dict[str, np.ndarray]:
    rows = read_rows(run_dir / "subject_holdout_predictions.csv")
    return {
        "sample_ids": np.asarray([row["sample_id"] for row in rows]),
        "labels": np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        "users": np.asarray([row["user_id"] for row in rows]),
        "logits": np.asarray(
            np.load(run_dir / "subject_holdout_logits.npy", allow_pickle=False),
            dtype=np.float64,
        ),
    }


def align_thermal(
    sample_ids: np.ndarray, p12_ids: np.ndarray, thermal_logits: np.ndarray
) -> np.ndarray:
    lookup = {sample_id: index for index, sample_id in enumerate(p12_ids)}
    return thermal_logits[np.asarray([lookup[value] for value in sample_ids])]


def fit_router(features: np.ndarray, target: np.ndarray, c_value: float) -> LogisticRegression:
    if len(np.unique(target)) != 2:
        raise RuntimeError("Thermal rescue target lost a class")
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=20260815,
    )
    model.fit(features, target)
    return model


def decoder_for_run(
    run_dir: Path,
    holdout_users: tuple[str, ...],
    all_labels: np.ndarray,
    all_metadata,
) -> tuple[object, DecoderConfig]:
    audit = json.loads((run_dir / "decoder_audit.json").read_text(encoding="utf-8"))
    frozen = audit["selected_decoder_config"]
    config = DecoderConfig(
        gap_seconds=float(frozen["gap_seconds"]),
        transition_weight=float(frozen["transition_weight"]),
        trigram_backoff=float(frozen["trigram_backoff"]),
        beam_width=int(frozen["beam_width"]),
    )
    fit_indices = np.flatnonzero(~np.isin(all_metadata.users, list(holdout_users)))
    transition = fit_transition_model(
        all_labels,
        build_sessions(
            fit_indices, all_metadata, config.gap_seconds, grouping="known_user"
        ),
        num_classes=40,
        trigram_backoff=config.trigram_backoff,
    )
    return transition, config


def route_logits(
    anchor_logits: np.ndarray,
    thermal_logits: np.ndarray,
    router_score: np.ndarray,
    threshold: float,
    blend_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_probability = probability(anchor_logits)
    thermal_probability = probability(thermal_logits)
    disagree = anchor_probability.argmax(axis=1) != thermal_probability.argmax(axis=1)
    routed = disagree & (router_score >= float(threshold))
    result = anchor_probability.copy()
    result[routed] = (
        (1.0 - float(blend_weight)) * anchor_probability[routed]
        + float(blend_weight) * thermal_probability[routed]
    )
    result /= result.sum(axis=1, keepdims=True)
    return np.log(np.maximum(result, 1e-12)), routed


def evaluate_route(
    data: dict[str, np.ndarray],
    thermal_logits: np.ndarray,
    router_score: np.ndarray,
    threshold: float,
    blend_weight: float,
    train_metadata_path: Path,
    transition,
    decoder_config: DecoderConfig,
) -> dict[str, object]:
    sample_ids = data["sample_ids"]
    labels = data["labels"]
    metadata = align_metadata(train_metadata_path, sample_ids)
    indices = np.arange(len(labels), dtype=np.int64)
    sessions = build_sessions(
        indices, metadata, decoder_config.gap_seconds, grouping="anonymous_date"
    )
    base_logp = log_softmax_numpy(data["logits"])
    base_raw = base_logp.argmax(axis=1)
    base_decoded = decode_sessions(base_logp, sessions, transition, decoder_config)
    routed_logp, routed = route_logits(
        data["logits"], thermal_logits, router_score, threshold, blend_weight
    )
    raw = routed_logp.argmax(axis=1)
    decoded = decode_sessions(routed_logp, sessions, transition, decoder_config)
    return {
        "threshold": float(threshold),
        "blend_weight": float(blend_weight),
        "routed_rows": int(routed.sum()),
        "baseline_raw": classification_metrics(labels, base_raw),
        "baseline_decoded": classification_metrics(labels, base_decoded),
        "raw": classification_metrics(labels, raw),
        "decoded": classification_metrics(labels, decoded),
        "raw_rescue_harm": rescue_harm(labels, base_raw, raw),
        "decoded_rescue_harm": rescue_harm(labels, base_decoded, decoded),
    }


def main() -> None:
    args = parse_args()
    h1_run = args.h1_run.resolve()
    h2_run = args.h2_run.resolve()
    h1 = load_run(h1_run)
    h2 = load_run(h2_run)
    if set(h1["users"]) != set(H1_USERS) or set(h2["users"]) != set(H2_USERS):
        raise RuntimeError("P88 H1/H2 universes differ from the frozen protocol")
    with np.load(args.p12_oof.resolve(), allow_pickle=False) as p12:
        p12_ids = np.asarray(p12["sample_ids"]).astype(str)
        thermal_all = np.asarray(p12["thermal_candidate_logits"], dtype=np.float64)
    h1_thermal = align_thermal(h1["sample_ids"], p12_ids, thermal_all)
    h2_thermal = align_thermal(h2["sample_ids"], p12_ids, thermal_all)
    h1_features = router_features(h1["logits"], h1_thermal)
    h2_features = router_features(h2["logits"], h2_thermal)
    h1_anchor_prediction = h1["logits"].argmax(axis=1)
    h1_thermal_prediction = h1_thermal.argmax(axis=1)
    h1_disagree = h1_anchor_prediction != h1_thermal_prediction
    h1_target = (
        (h1_thermal_prediction == h1["labels"])
        & (h1_anchor_prediction != h1["labels"])
    ).astype(np.int64)
    with np.load(args.teacher_targets.resolve(), allow_pickle=False) as teacher:
        all_ids = np.asarray(teacher["oof_sample_ids"]).astype(str)
        all_labels = np.asarray(teacher["oof_labels"], dtype=np.int64)
    all_metadata = align_metadata(args.train_metadata, all_ids)
    h1_transition, h1_decoder = decoder_for_run(
        h1_run, H1_USERS, all_labels, all_metadata
    )
    h2_transition, h2_decoder = decoder_for_run(
        h2_run, H2_USERS, all_labels, all_metadata
    )

    best_key: tuple[int, int, int, float, float] | None = None
    best_selection: dict[str, object] | None = None
    selection_rows: list[dict[str, object]] = []
    selected_scores: np.ndarray | None = None
    for c_value in args.c_values:
        loso_scores = np.zeros(len(h1["labels"]), dtype=np.float64)
        for held_user in H1_USERS:
            fit = (h1["users"] != held_user) & h1_disagree
            held = h1["users"] == held_user
            router = fit_router(h1_features[fit], h1_target[fit], float(c_value))
            loso_scores[held] = router.predict_proba(h1_features[held])[:, 1]
        for threshold in args.thresholds:
            for blend_weight in args.blend_weights:
                result = evaluate_route(
                    h1,
                    h1_thermal,
                    loso_scores,
                    float(threshold),
                    float(blend_weight),
                    args.train_metadata.resolve(),
                    h1_transition,
                    h1_decoder,
                )
                row = {"c": float(c_value), **result}
                selection_rows.append(row)
                key = (
                    int(result["decoded"]["correct"]),  # type: ignore[index]
                    int(result["raw"]["correct"]),  # type: ignore[index]
                    -int(result["routed_rows"]),
                    float(threshold),
                    -float(blend_weight),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_selection = row
                    selected_scores = loso_scores.copy()
    assert best_selection is not None and selected_scores is not None
    chosen_c = float(best_selection["c"])
    final_router = fit_router(
        h1_features[h1_disagree], h1_target[h1_disagree], chosen_c
    )
    h2_scores = final_router.predict_proba(h2_features)[:, 1]
    h2_result = evaluate_route(
        h2,
        h2_thermal,
        h2_scores,
        float(best_selection["threshold"]),
        float(best_selection["blend_weight"]),
        args.train_metadata.resolve(),
        h2_transition,
        h2_decoder,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P88_H1_LOSO_thermal_rescue_router_H2_confirmation",
        "status": "complete",
        "protocol": (
            "P87 and subject-disjoint Thermal expert logits are frozen. C, threshold "
            "and blend are selected from H1 user-LOSO predictions only. The router is "
            "then refit on all H1 disagreements and evaluated once on H2."
        ),
        "h1_target_audit": {
            "rows": len(h1["labels"]),
            "disagreements": int(h1_disagree.sum()),
            "thermal_rescues": int(h1_target.sum()),
        },
        "h1_selected": best_selection,
        "h2_confirmation": h2_result,
        "selection_grid": selection_rows,
        "router_coefficients": {
            "intercept": final_router.intercept_.tolist(),
            "coefficient": final_router.coef_.tolist(),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
