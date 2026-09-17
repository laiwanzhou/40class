from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import align_metadata, build_sessions, fit_transition_model
from build_p87s_test_structured_targets import config_from_nested_summary


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEACHER = PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
DEFAULT_METADATA = PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_DECODER_SUMMARY = PROJECT_DIR / "runs/p87_sequence_decoder_v1/summary.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p87s_tiny_decoder_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the nested-OOF-selected P87 transition decoder into a tiny "
            "standalone inference artifact."
        )
    )
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--decoder-summary", type=Path, default=DEFAULT_DECODER_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    teacher_path = args.teacher_targets.resolve()
    decoder_summary_path = args.decoder_summary.resolve()
    with np.load(teacher_path, allow_pickle=False) as teacher:
        sample_ids = teacher["oof_sample_ids"].astype(str)
        labels = teacher["oof_labels"].astype(np.int64)
    if len(sample_ids) != 2914 or labels.min() != 0 or labels.max() != 39:
        raise RuntimeError("P87 tiny decoder requires the exact 2914-row Train labels")
    config = config_from_nested_summary(decoder_summary_path)
    metadata = align_metadata(args.train_metadata.resolve(), sample_ids)
    sessions = build_sessions(
        np.arange(len(sample_ids)),
        metadata,
        config.gap_seconds,
        grouping="known_user",
    )
    transition = fit_transition_model(
        labels,
        sessions,
        num_classes=40,
        trigram_backoff=config.trigram_backoff,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "tiny_decoder.npz"
    np.savez_compressed(
        artifact,
        start_log_probability=transition.start_log_probability,
        end_log_probability=transition.end_log_probability,
        bigram_log_probability=transition.bigram_log_probability,
        trigram_log_probability=transition.trigram_log_probability,
        gap_seconds=np.asarray(config.gap_seconds, dtype=np.float64),
        transition_weight=np.asarray(config.transition_weight, dtype=np.float64),
        trigram_backoff=np.asarray(config.trigram_backoff, dtype=np.float64),
        beam_width=np.asarray(config.beam_width, dtype=np.int64),
    )
    summary = {
        "stage": "P87S_frozen_tiny_decoder",
        "protocol": (
            "Hyperparameters are the exact median consensus of three pre-existing "
            "strict nested subject-disjoint OOF folds. Transition probabilities are "
            "fit once on all 2914 Train labels; Test labels/emissions are never read."
        ),
        "train_rows": len(sample_ids),
        "train_sessions": len(sessions),
        "config": {
            "gap_seconds": config.gap_seconds,
            "transition_weight": config.transition_weight,
            "trigram_backoff": config.trigram_backoff,
            "beam_width": config.beam_width,
        },
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_mib": artifact.stat().st_size / 1024**2,
        "teacher_targets_used_for_train_labels_only": str(teacher_path),
        "teacher_targets_sha256": sha256(teacher_path),
        "decoder_summary": str(decoder_summary_path),
        "decoder_summary_sha256": sha256(decoder_summary_path),
        "large_model_required_at_inference": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
