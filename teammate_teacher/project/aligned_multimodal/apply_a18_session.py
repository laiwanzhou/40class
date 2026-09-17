"""Fit the frozen Session mechanism on all 18 source subjects for deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from a18_full_teacher_data import A18_ROWS, A18_SOURCE_USER_SET
from audit_p102_session_closure import classification_metrics, comparison
from audit_p87_sequence_decoder import (
    DecoderConfig,
    align_metadata,
    build_sessions,
    decode_unique_beam_posterior,
    fit_transition_model,
)
from build_p87s_structured_targets import backed_off_structured_probability


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/a18_full_teacher.json"
DEFAULT_PREDICTIONS = HERE / "runs/a18_full_teacher_v1/full_fit_predictions.npz"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = HERE / "runs/a18_full_teacher_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    prediction_path = args.predictions.resolve()
    metadata_path = args.metadata.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    recipe = config["session"]
    decoder = DecoderConfig(
        gap_seconds=float(recipe["gap_seconds"]),
        transition_weight=float(recipe["transition_weight"]),
        trigram_backoff=float(recipe["trigram_backoff"]),
        beam_width=int(recipe["beam_width"]),
    )
    temperature = float(recipe["posterior_temperature"])
    with np.load(prediction_path, allow_pickle=False) as archive:
        sample_ids = np.asarray(archive["sample_ids"]).astype(str)
        users = np.asarray(archive["users"]).astype(str)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        logits = np.asarray(archive["direct_logits"], dtype=np.float64)
        raw_probability = np.asarray(archive["direct_probability"], dtype=np.float64)
    if len(sample_ids) != A18_ROWS or len(np.unique(sample_ids)) != A18_ROWS:
        raise RuntimeError("A18 full prediction row contract changed")
    if set(users.tolist()) != A18_SOURCE_USER_SET:
        raise RuntimeError("A18 full prediction subject contract changed")
    if logits.shape != (A18_ROWS, 40) or not np.isfinite(logits).all():
        raise RuntimeError("A18 full logits contract changed")
    metadata = align_metadata(metadata_path, sample_ids)
    if not np.array_equal(metadata.users, users):
        raise RuntimeError("A18 metadata subject alignment changed")

    all_indices = np.arange(A18_ROWS, dtype=np.int64)
    train_sessions = build_sessions(
        all_indices, metadata, decoder.gap_seconds, grouping="known_user"
    )
    transition = fit_transition_model(
        labels,
        train_sessions,
        num_classes=40,
        trigram_backoff=decoder.trigram_backoff,
    )
    np.savez_compressed(
        output / "session_transition_state.npz",
        start_log_probability=transition.start_log_probability,
        end_log_probability=transition.end_log_probability,
        bigram_log_probability=transition.bigram_log_probability,
        trigram_log_probability=transition.trigram_log_probability,
        gap_seconds=np.asarray(decoder.gap_seconds),
        transition_weight=np.asarray(decoder.transition_weight),
        trigram_backoff=np.asarray(decoder.trigram_backoff),
        beam_width=np.asarray(decoder.beam_width),
        posterior_temperature=np.asarray(temperature),
        source_sample_ids=sample_ids,
        source_users=users,
    )

    # Fit-set decoding is emitted only as a diagnostic.  The transition table was
    # fitted on these labels, so this cannot be interpreted as held performance.
    log_probability = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    structured = raw_probability.copy()
    structured_map = raw_probability.argmax(axis=1).astype(np.int64)
    session_id = np.full(A18_ROWS, -1, dtype=np.int64)
    diagnostic_sessions = build_sessions(
        all_indices, metadata, decoder.gap_seconds, grouping="anonymous_date"
    )
    for sequence_id, session in enumerate(diagnostic_sessions):
        posterior = decode_unique_beam_posterior(
            log_probability[session],
            transition,
            transition_weight=decoder.transition_weight,
            beam_width=decoder.beam_width,
            posterior_temperature=temperature,
        )
        structured[session] = posterior.marginals
        structured_map[session] = posterior.paths[0]
        session_id[session] = sequence_id
    session_probability, structured_weight = backed_off_structured_probability(
        raw_probability, structured, beam_width=decoder.beam_width
    )
    session_prediction = session_probability.argmax(axis=1)
    np.savez_compressed(
        output / "full_fit_session_predictions.npz",
        sample_ids=sample_ids,
        users=users,
        labels=labels,
        raw_probability=raw_probability.astype(np.float32),
        selected_probability=session_probability.astype(np.float32),
        selected_prediction=session_prediction,
        structured_probability=structured.astype(np.float32),
        structured_map_prediction=structured_map,
        structured_weight=structured_weight.astype(np.float32),
        sequence_session_id=session_id,
    )
    raw_metrics = classification_metrics(raw_probability, labels, users)
    session_metrics = classification_metrics(session_probability, labels, users)
    timestamp_mask = np.isfinite(metadata.starts) & (metadata.dates != "")
    summary = {
        "status": "complete",
        "protocol": (
            "One frozen Session recipe fitted on all 18 source subjects for final "
            "deployment. No split/grid/threshold/seed/blend selection."
        ),
        "inputs": {
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256(prediction_path),
            },
            "metadata": {
                "path": str(metadata_path),
                "sha256": sha256(metadata_path),
            },
            "config": {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
        },
        "recipe": {**recipe, "candidate_count": 1, "a18_grid_search": False},
        "fit": {
            "source_rows": A18_ROWS,
            "source_subjects": 18,
            "transition_train_sessions": len(train_sessions),
            "diagnostic_sessions": len(diagnostic_sessions),
            "timestamp_rows": int(timestamp_mask.sum()),
            "timestamp_coverage": float(timestamp_mask.mean()),
        },
        "systems": {
            "raw_VS_fit": {"metrics": raw_metrics},
            "A18_VS_session_fit": {
                "metrics": session_metrics,
                "vs_raw": comparison(
                    labels, users, raw_probability, session_probability
                ),
            },
        },
        "metric_scope": "training_fit_set_with_in_sample_transition_table",
        "generalization_claim_allowed": False,
        "deployment_artifact": str(output / "session_transition_state.npz"),
        "constraints": {
            "held_data_loaded": False,
            "held_label_selection_rows": 0,
            "h3_rows_selected": 0,
            "h3_confirmation_run": False,
            "b_teacher_started": False,
            "router_started": False,
            "confusion_family_loaded": False,
            "threshold_seed_blend_sweep": False,
        },
    }
    (output / "session_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["systems"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
