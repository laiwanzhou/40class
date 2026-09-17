from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DecoderConfig,
    align_metadata,
    build_sessions,
    consensus_test_config,
    decode_unique_beam_posterior,
    fit_transition_model,
)
from build_p87s_structured_targets import backed_off_structured_probability


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEACHER = PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
DEFAULT_TRAIN_METADATA = PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_TEST_METADATA = PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv"
DEFAULT_DECODER_SUMMARY = PROJECT_DIR / "runs/p87_sequence_decoder_v1/summary.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p87s_test_structured_targets_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final label-free P87-S posterior targets for the 401 Test rows "
            "covered by P85 Large emissions, using only the pre-frozen nested-OOF "
            "decoder consensus."
        )
    )
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_TRAIN_METADATA)
    parser.add_argument("--test-metadata", type=Path, default=DEFAULT_TEST_METADATA)
    parser.add_argument("--decoder-summary", type=Path, default=DEFAULT_DECODER_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_from_nested_summary(path: Path) -> DecoderConfig:
    summary = json.loads(path.read_text(encoding="utf-8"))
    folds = summary.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("decoder summary must contain the three strict nested folds")
    config = consensus_test_config(folds, beam_width=50)
    recorded = summary.get("test_config_from_outer_medians")
    expected = {
        "gap_seconds": config.gap_seconds,
        "transition_weight": config.transition_weight,
        "trigram_backoff": config.trigram_backoff,
        "beam_width": config.beam_width,
    }
    if recorded != expected:
        raise ValueError(
            f"decoder consensus changed: recorded={recorded}, recomputed={expected}"
        )
    return config


def main() -> None:
    args = parse_args()
    if args.posterior_temperature <= 0:
        raise ValueError("posterior-temperature must be positive")
    teacher_path = args.teacher_targets.resolve()
    train_metadata_path = args.train_metadata.resolve()
    test_metadata_path = args.test_metadata.resolve()
    decoder_summary_path = args.decoder_summary.resolve()
    teacher = np.load(teacher_path, allow_pickle=False)
    train_ids = teacher["oof_sample_ids"].astype(str)
    labels = teacher["oof_labels"].astype(np.int64)
    test_ids = teacher["test_sample_ids"].astype(str)
    emission_probability = teacher["test_teacher_probability"].astype(np.float64)
    emission_log_probability = teacher["test_teacher_log_probability"].astype(
        np.float64
    )
    if (
        len(train_ids) != 2914
        or len(test_ids) != 401
        or emission_probability.shape != (401, 40)
    ):
        raise RuntimeError("P85 Train/Test target universe changed")
    if not np.allclose(emission_probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("P85 Test emission rows are not normalized")

    config = config_from_nested_summary(decoder_summary_path)
    train_metadata = align_metadata(train_metadata_path, train_ids)
    test_metadata = align_metadata(test_metadata_path, test_ids)
    train_sessions = build_sessions(
        np.arange(len(train_ids)),
        train_metadata,
        config.gap_seconds,
        grouping="known_user",
    )
    transition = fit_transition_model(
        labels,
        train_sessions,
        num_classes=40,
        trigram_backoff=config.trigram_backoff,
    )
    test_sessions = build_sessions(
        np.arange(len(test_ids)),
        test_metadata,
        config.gap_seconds,
        grouping="anonymous_date",
    )
    structured_probability = emission_probability.copy()
    structured_map = emission_probability.argmax(axis=1).astype(np.int64)
    sequence_session_id = np.full(len(test_ids), -1, dtype=np.int64)
    for session_id, session in enumerate(test_sessions):
        posterior = decode_unique_beam_posterior(
            emission_log_probability[session],
            transition,
            transition_weight=config.transition_weight,
            beam_width=config.beam_width,
            posterior_temperature=float(args.posterior_temperature),
        )
        structured_probability[session] = posterior.marginals
        structured_map[session] = posterior.paths[0]
        sequence_session_id[session] = session_id
    distillation_probability, structured_weight = backed_off_structured_probability(
        emission_probability,
        structured_probability,
        beam_width=config.beam_width,
    )
    confidence = 1.0 - (
        -np.sum(
            structured_probability
            * np.log(np.maximum(structured_probability, 1e-300)),
            axis=1,
        )
        / np.log(structured_probability.shape[1])
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "structured_targets.npz"
    np.savez_compressed(
        target_path,
        sample_ids=test_ids,
        target_mask=np.ones(len(test_ids), dtype=bool),
        decoded_mask=sequence_session_id >= 0,
        emission_probability=emission_probability.astype(np.float32),
        structured_probability=structured_probability.astype(np.float32),
        structured_distillation_probability=distillation_probability.astype(np.float32),
        structured_distillation_weight=structured_weight.astype(np.float32),
        emission_prediction=emission_probability.argmax(axis=1).astype(np.int64),
        structured_map_prediction=structured_map,
        structured_marginal_prediction=structured_probability.argmax(axis=1).astype(
            np.int64
        ),
        structured_distillation_prediction=distillation_probability.argmax(axis=1).astype(
            np.int64
        ),
        structured_confidence=confidence.astype(np.float32),
        sequence_session_id=sequence_session_id,
        sequence_changed=(
            structured_map != emission_probability.argmax(axis=1)
        ),
    )
    summary = {
        "stage": "P87-S final Test structured posterior target generation",
        "protocol": (
            "No Test labels exist or are read. Decoder hyperparameters are the median "
            "consensus of the pre-existing three strict nested subject-disjoint OOF "
            "folds; no Test scan or Test adaptation is performed. The transition model "
            "uses all 2914 Train labels only. Targets cover exactly the 401 rows with "
            "P85 Large emissions; the four unreadable-IR Test rows receive no fabricated "
            "Large target."
        ),
        "train_rows": len(train_ids),
        "test_target_rows": len(test_ids),
        "missing_large_target_rows": 4,
        "train_sessions": len(train_sessions),
        "test_sessions": len(test_sessions),
        "decoded_rows": int(np.sum(sequence_session_id >= 0)),
        "sequence_changed_rows": int(
            np.sum(structured_map != emission_probability.argmax(axis=1))
        ),
        "mean_structured_weight": float(structured_weight.mean()),
        "mean_emission_to_structured_l1": float(
            np.abs(structured_probability - emission_probability).sum(axis=1).mean()
        ),
        "selected_config": {
            "gap_seconds": config.gap_seconds,
            "transition_weight": config.transition_weight,
            "trigram_backoff": config.trigram_backoff,
            "beam_width": config.beam_width,
            "posterior_temperature": float(args.posterior_temperature),
        },
        "teacher_targets": str(teacher_path),
        "teacher_targets_sha256": sha256(teacher_path),
        "decoder_summary": str(decoder_summary_path),
        "decoder_summary_sha256": sha256(decoder_summary_path),
        "output": str(target_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
