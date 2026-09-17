from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DecoderConfig,
    TransitionModel,
    align_metadata,
    build_sessions,
    decode_sessions,
)
from p88_aligned_repeat_holdout import AlignedRepeatConfig, decode_aligned_repeat
from p88_train_depth_residual import log_softmax_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build isolated P88 repeat and calibrated-repeat Test submissions."
    )
    parser.add_argument("--p87-predictions", type=Path, default=Path("runs/p87s_final_test_predictions_v1"))
    parser.add_argument("--tiny-decoder", type=Path, default=Path("runs/p87s_tiny_decoder_v1/tiny_decoder.npz"))
    parser.add_argument("--test-metadata", type=Path, default=Path("data/p85_recording_metadata/test_recording_metadata.csv"))
    parser.add_argument("--repeat-summary", type=Path, default=Path("runs/p88_aligned_repeat_h1_v1/summary.json"))
    parser.add_argument("--class-bias", type=Path, default=Path("runs/p88_class_bias_h1_to_h2_v1/class_bias_combined.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/p88_final_test_predictions_v1"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_submission(path: Path, source_rows: list[dict[str, str]], prediction: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        for row, value in zip(source_rows, prediction, strict=True):
            writer.writerow({"path": row["path"], "prediction": int(value)})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def histogram(values: np.ndarray) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(map(int, values)).items())}


def main() -> None:
    args = parse_args()
    p87 = args.p87_predictions.resolve()
    audit_rows = read_rows(p87 / "prediction_audit.csv")
    sample_ids = np.asarray([row["sample_id"] for row in audit_rows])
    logits = np.asarray(np.load(p87 / "student_logits.npy"), dtype=np.float64)
    if logits.shape != (len(sample_ids), 40):
        raise RuntimeError("P87 Test logits changed shape")
    source_submission = read_rows(p87 / "submission_p87s_student_decoded.csv")
    if len(source_submission) != len(sample_ids):
        raise RuntimeError("P87 submission and audit row counts differ")

    with np.load(args.tiny_decoder.resolve(), allow_pickle=False) as saved:
        transition = TransitionModel(
            start_log_probability=np.asarray(saved["start_log_probability"]),
            end_log_probability=np.asarray(saved["end_log_probability"]),
            bigram_log_probability=np.asarray(saved["bigram_log_probability"]),
            trigram_log_probability=np.asarray(saved["trigram_log_probability"]),
        )
        decoder = DecoderConfig(
            gap_seconds=float(saved["gap_seconds"]),
            transition_weight=float(saved["transition_weight"]),
            trigram_backoff=float(saved["trigram_backoff"]),
            beam_width=int(saved["beam_width"]),
        )
    repeat_source = json.loads(args.repeat_summary.resolve().read_text(encoding="utf-8"))
    repeat = AlignedRepeatConfig(**repeat_source["best"]["config"])
    metadata = align_metadata(args.test_metadata.resolve(), sample_ids)
    indices = np.arange(len(sample_ids), dtype=np.int64)
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    base_logp = log_softmax_numpy(logits)
    reproduced = decode_sessions(base_logp, sessions, transition, decoder)
    saved_decoded = np.asarray([int(row["prediction"]) for row in source_submission])
    if not np.array_equal(reproduced, saved_decoded):
        raise RuntimeError(
            f"P88 failed to reproduce P87 decoded submission: "
            f"{int(np.sum(reproduced != saved_decoded))} differences"
        )

    repeat_prediction, repeat_grouping = decode_aligned_repeat(
        base_logp, indices, metadata, transition, decoder, repeat
    )
    class_bias = np.asarray(np.load(args.class_bias.resolve()), dtype=np.float64)
    if class_bias.shape != (40,):
        raise RuntimeError("P88 class bias must have shape [40]")
    calibrated_logp = log_softmax_numpy(logits + class_bias)
    calibrated_decoded = decode_sessions(
        calibrated_logp, sessions, transition, decoder
    )
    calibrated_repeat, calibrated_grouping = decode_aligned_repeat(
        calibrated_logp, indices, metadata, transition, decoder, repeat
    )

    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    repeat_path = output / "submission_p88_repeat.csv"
    calibrated_path = output / "submission_p88_calibrated_repeat.csv"
    write_submission(repeat_path, source_submission, repeat_prediction)
    write_submission(calibrated_path, source_submission, calibrated_repeat)
    with (output / "prediction_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = (
            "sample_id", "p87_decoded", "p88_repeat", "p88_calibrated_decoded",
            "p88_calibrated_repeat", "repeat_changed", "calibration_changed",
        )
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            writer.writerow({
                "sample_id": sample_id,
                "p87_decoded": int(saved_decoded[index]),
                "p88_repeat": int(repeat_prediction[index]),
                "p88_calibrated_decoded": int(calibrated_decoded[index]),
                "p88_calibrated_repeat": int(calibrated_repeat[index]),
                "repeat_changed": int(repeat_prediction[index] != saved_decoded[index]),
                "calibration_changed": int(calibrated_repeat[index] != repeat_prediction[index]),
            })
    summary = {
        "stage": "P88_final_Test_postprocessing", "status": "complete",
        "protocol": (
            "Read P87 logits without modifying any P87 artifact; apply the H1-selected/H2-confirmed "
            "variable-length repeat consensus and an optional cross-fitted 40-value class bias."
        ),
        "test_rows": len(sample_ids),
        "p87_reproduction_exact": True,
        "decoder_configuration": asdict(decoder),
        "repeat_configuration": asdict(repeat),
        "repeat_grouping": repeat_grouping,
        "calibrated_repeat_grouping": calibrated_grouping,
        "repeat_changed_vs_p87": int(np.sum(repeat_prediction != saved_decoded)),
        "calibrated_decoded_changed_vs_p87": int(np.sum(calibrated_decoded != saved_decoded)),
        "calibrated_repeat_changed_vs_repeat": int(np.sum(calibrated_repeat != repeat_prediction)),
        "calibrated_repeat_changed_vs_p87": int(np.sum(calibrated_repeat != saved_decoded)),
        "p87_histogram": histogram(saved_decoded),
        "p88_repeat_histogram": histogram(repeat_prediction),
        "p88_calibrated_repeat_histogram": histogram(calibrated_repeat),
        "class_bias_l2": float(np.linalg.norm(class_bias)),
        "class_bias_max_abs": float(np.max(np.abs(class_bias))),
        "submissions": {
            "conservative_repeat": {"path": str(repeat_path), "sha256": sha256(repeat_path)},
            "recommended_calibrated_repeat": {"path": str(calibrated_path), "sha256": sha256(calibrated_path)},
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
