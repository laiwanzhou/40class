from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import (
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
)
from p88_train_depth_residual import (
    DEFAULT_DECODER_AUDIT,
    DEFAULT_FEATURES,
    DEFAULT_METADATA,
    log_softmax_numpy,
    make_decoder,
    read_rows,
    rescue_harm,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p88_depth_linear_probe_h1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-label-only linear audit of frozen P88 Depth embeddings."
    )
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--decoder-audit", type=Path, default=DEFAULT_DECODER_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-users", nargs="+", required=True)
    parser.add_argument("--c-values", type=float, nargs="+", default=(0.001, 0.003, 0.01, 0.03, 0.1))
    parser.add_argument(
        "--blend-weights",
        type=float,
        nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30),
    )
    return parser.parse_args()


def probabilities(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_numpy(logits))


def main() -> None:
    args = parse_args()
    cache = args.feature_cache.resolve()
    rows = read_rows(cache / "rows.csv")
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in rows])
    held = np.flatnonzero(np.isin(users, list(args.holdout_users)))
    train = np.flatnonzero(~np.isin(users, list(args.holdout_users)))
    depth = np.asarray(
        np.load(cache / "depth_embedding_fp16.npy", mmap_mode="r"), dtype=np.float32
    )
    validity = np.asarray(
        np.load(cache / "depth_valid_statistics_fp16.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    features = np.concatenate((depth, validity), axis=1)
    scaler = StandardScaler().fit(features[train])
    train_features = scaler.transform(features[train])
    held_features = scaler.transform(features[held])
    anchor_logits = np.asarray(
        np.load(cache / "anchor_logits.npy", mmap_mode="r"), dtype=np.float64
    )
    anchor_probability = probabilities(anchor_logits)
    metadata = align_metadata(args.metadata, sample_ids)
    transition, decoder_config = make_decoder(
        labels, train, metadata, args.decoder_audit.resolve()
    )
    sessions = build_sessions(
        held, metadata, decoder_config.gap_seconds, grouping="anonymous_date"
    )
    anchor_raw = anchor_logits.argmax(axis=1)
    anchor_decoded = decode_sessions(
        log_softmax_numpy(anchor_logits), sessions, transition, decoder_config
    )
    candidates: list[dict[str, object]] = []
    best_key: tuple[int, int, float, float] | None = None
    best_payload: dict[str, object] | None = None
    for c_value in args.c_values:
        classifier = LogisticRegression(
            C=float(c_value),
            max_iter=600,
            solver="lbfgs",
            class_weight="balanced",
            random_state=20260815,
        )
        classifier.fit(train_features, labels[train])
        depth_probability = np.full((len(rows), 40), 1.0 / 40.0, dtype=np.float64)
        depth_probability[held] = classifier.predict_proba(held_features)
        depth_prediction = depth_probability.argmax(axis=1)
        direct = classification_metrics(labels[held], depth_prediction[held])
        for weight in args.blend_weights:
            mixed = anchor_probability.copy()
            mixed[held] = (
                (1.0 - float(weight)) * anchor_probability[held]
                + float(weight) * depth_probability[held]
            )
            mixed /= mixed.sum(axis=1, keepdims=True)
            raw = mixed.argmax(axis=1)
            decoded = decode_sessions(
                np.log(np.maximum(mixed, 1e-12)), sessions, transition, decoder_config
            )
            raw_metrics = classification_metrics(labels[held], raw[held])
            decoded_metrics = classification_metrics(labels[held], decoded[held])
            row = {
                "c": float(c_value),
                "blend_weight": float(weight),
                "depth_direct_accuracy": direct["accuracy"],
                "raw_accuracy": raw_metrics["accuracy"],
                "decoded_accuracy": decoded_metrics["accuracy"],
                "raw_rescue_harm": rescue_harm(
                    labels[held], anchor_raw[held], raw[held]
                ),
                "decoded_rescue_harm": rescue_harm(
                    labels[held], anchor_decoded[held], decoded[held]
                ),
            }
            candidates.append(row)
            key = (
                int(decoded_metrics["correct"]),
                int(raw_metrics["correct"]),
                -float(weight),
                -float(c_value),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_payload = row
    assert best_payload is not None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P88_registered_depth_linear_probe",
        "status": "complete",
        "protocol": (
            "Scaler and multinomial classifier fit only non-holdout subjects. C and "
            "blend weight are development diagnostics selected on H1 and cannot be "
            "claimed until an independent H2 refit."
        ),
        "holdout_users": sorted(map(str, args.holdout_users)),
        "baseline": {
            "raw": classification_metrics(labels[held], anchor_raw[held]),
            "decoded": classification_metrics(labels[held], anchor_decoded[held]),
        },
        "best": best_payload,
        "candidates": candidates,
        "config": {
            "feature_cache": str(cache),
            "metadata": str(args.metadata.resolve()),
            "decoder_audit": str(args.decoder_audit.resolve()),
            "output_dir": str(output),
            "holdout_users": list(args.holdout_users),
            "c_values": list(args.c_values),
            "blend_weights": list(args.blend_weights),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
