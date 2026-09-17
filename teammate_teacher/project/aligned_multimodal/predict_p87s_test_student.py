from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapt_p87s_structured_student import model_build_args
from audit_p87_sequence_decoder import (
    DecoderConfig,
    TransitionModel,
    align_metadata,
    build_sessions,
    decode_sessions,
)
from p87s_test_data import P87STestCachedSequenceMotionDataset, collate_p87s_test
from p87s_deploy_model import load_p87s_deploy_checkpoint
from train_p86_mobind_fusion_proxy import build_model, model_forward


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SEQUENCE = PROJECT_DIR / "runs/p87s_test_mc3_sequence_v1"
DEFAULT_MOTION = PROJECT_DIR / "runs/p87s_test_motion_window_t16_v1"
DEFAULT_PIXELS = PROJECT_DIR / "runs/p87s_test_pixel_cache_t16_r160_v1"
DEFAULT_METADATA = PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv"
DEFAULT_DECODER = PROJECT_DIR / "runs/p87s_tiny_decoder_v1/tiny_decoder.npz"
DEFAULT_STRUCTURED_TARGETS = (
    PROJECT_DIR / "runs/p87s_test_structured_targets_v1/structured_targets.npz"
)
DEFAULT_TEST_CSV = PROJECT_DIR.parent / "Testing/test.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p87s_final_test_predictions_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the <=100 MiB P87-S Student and frozen tiny decoder on all Test rows."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--test-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--tiny-decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument(
        "--structured-targets",
        type=Path,
        default=DEFAULT_STRUCTURED_TARGETS,
        help=(
            "Optional Train-frozen P87-S Test targets used only for an unlabeled "
            "adaptation audit. They never alter final predictions."
        ),
    )
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_id(path_value: str) -> str:
    clean = path_value.replace("\\", "/").rstrip("/")
    return clean.rsplit("/", 1)[-1]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_student(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stage = checkpoint.get("stage")
    if stage in {
        "P87S_label_free_test_adaptation",
        "P162_P150_distilled_deployment",
    }:
        model, checkpoint = load_p87s_deploy_checkpoint(path)
        return model, {}, stage
    elif stage == "P87S_mobind_fusion_all2914_refit":
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        build_args = model_build_args(path, summary)
        base_summary = summary
    else:
        raise ValueError(f"unsupported final Student checkpoint stage: {stage}")
    model, _, _ = build_model(build_args)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, base_summary, stage


def load_decoder(path: Path) -> tuple[TransitionModel, DecoderConfig]:
    with np.load(path, allow_pickle=False) as data:
        model = TransitionModel(
            start_log_probability=np.asarray(data["start_log_probability"]),
            end_log_probability=np.asarray(data["end_log_probability"]),
            bigram_log_probability=np.asarray(data["bigram_log_probability"]),
            trigram_log_probability=np.asarray(data["trigram_log_probability"]),
        )
        config = DecoderConfig(
            gap_seconds=float(data["gap_seconds"]),
            transition_weight=float(data["transition_weight"]),
            trigram_backoff=float(data["trigram_backoff"]),
            beam_width=int(data["beam_width"]),
        )
    if model.trigram_log_probability.shape != (40, 40, 40):
        raise RuntimeError("tiny decoder class contract differs")
    return model, config


def infer_student(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    model.eval().to(device)
    logits_rows: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            sample_ids.extend(map(str, batch["sample_id"]))
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model_forward(model, batch)["logits"]
            logits_rows.append(logits.float().cpu().numpy())
    logits = np.concatenate(logits_rows)
    if logits.shape != (405, 40) or not np.isfinite(logits).all():
        raise RuntimeError("final Student logits are incomplete or nonfinite")
    return sample_ids, logits


def log_softmax_numpy(logits: np.ndarray) -> np.ndarray:
    result = logits.astype(np.float64)
    result -= np.logaddexp.reduce(result, axis=1, keepdims=True)
    return result


def entropy_from_log_probability(log_probability: np.ndarray) -> np.ndarray:
    probability = np.exp(log_probability)
    return -(probability * log_probability).sum(axis=1)


def class_histogram(prediction: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(prediction, dtype=np.int64), minlength=40)
    return {str(index): int(value) for index, value in enumerate(counts) if value}


def changed_pair_histogram(
    before: np.ndarray, after: np.ndarray
) -> dict[str, int]:
    changed = np.asarray(before) != np.asarray(after)
    pairs: dict[str, int] = {}
    for source, target in zip(
        np.asarray(before)[changed], np.asarray(after)[changed], strict=True
    ):
        key = f"{int(source)}->{int(target)}"
        pairs[key] = pairs.get(key, 0) + 1
    return dict(sorted(pairs.items(), key=lambda item: (-item[1], item[0])))


def optional_adaptation_audit(
    checkpoint: dict[str, Any],
    final_log_probability: np.ndarray,
    final_raw_prediction: np.ndarray,
    final_decoded_prediction: np.ndarray,
    sample_ids: list[str],
    loader: DataLoader,
    device: torch.device,
    transition: TransitionModel,
    config: DecoderConfig,
    sessions: list[np.ndarray],
    structured_targets_path: Path,
    output: Path,
) -> dict[str, Any] | None:
    """Audit adaptation without making base/teacher artifacts deployment requirements."""
    base_value = checkpoint.get("base_checkpoint")
    base_path = Path(str(base_value)).resolve() if base_value else None
    target_path = structured_targets_path.resolve()
    if base_path is None or not base_path.is_file() or not target_path.is_file():
        return None
    base_model, _, base_stage = load_student(base_path)
    baseline_ids, baseline_logits = infer_student(base_model, loader, device)
    base_model.to("cpu")
    if baseline_ids != sample_ids:
        raise RuntimeError("base and adapted Student inference row orders differ")
    baseline_log_probability = log_softmax_numpy(baseline_logits)
    baseline_raw = baseline_logits.argmax(axis=1).astype(np.int64)
    baseline_decoded = decode_sessions(
        baseline_log_probability, sessions, transition, config
    )
    np.save(output / "base_student_logits_audit_only.npy", baseline_logits)

    with np.load(target_path, allow_pickle=False) as targets:
        target_ids = np.asarray(targets["sample_ids"]).astype(str)
        structured_probability = np.asarray(
            targets["structured_distillation_probability"], dtype=np.float64
        )
        structured_prediction = np.asarray(
            targets["structured_distillation_prediction"], dtype=np.int64
        )
        emission_prediction = np.asarray(targets["emission_prediction"], dtype=np.int64)
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if len(target_ids) not in (401, 405) or any(value not in lookup for value in target_ids):
        raise RuntimeError("structured-target audit universe differs from Test inference")
    selected = np.asarray([lookup[value] for value in target_ids], dtype=np.int64)
    baseline_selected = baseline_raw[selected]
    final_selected = final_raw_prediction[selected]
    base_target_cross_entropy = float(
        -(structured_probability * baseline_log_probability[selected]).sum(axis=1).mean()
    )
    final_target_cross_entropy = float(
        -(structured_probability * final_log_probability[selected]).sum(axis=1).mean()
    )
    base_agree = baseline_selected == structured_prediction
    final_agree = final_selected == structured_prediction
    return {
        "status": "available_research_audit_not_required_for_deployment",
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": sha256(base_path),
        "base_checkpoint_stage": base_stage,
        "structured_targets": str(target_path),
        "structured_targets_sha256": sha256(target_path),
        "target_rows": len(target_ids),
        "base_to_adapted_raw_changed_rows_all405": int(
            np.sum(baseline_raw != final_raw_prediction)
        ),
        "base_to_adapted_decoded_changed_rows_all405": int(
            np.sum(baseline_decoded != final_decoded_prediction)
        ),
        "base_to_adapted_raw_change_pairs": changed_pair_histogram(
            baseline_raw, final_raw_prediction
        ),
        "base_to_adapted_decoded_change_pairs": changed_pair_histogram(
            baseline_decoded, final_decoded_prediction
        ),
        "base_mean_entropy_all405": float(
            entropy_from_log_probability(baseline_log_probability).mean()
        ),
        "adapted_mean_entropy_all405": float(
            entropy_from_log_probability(final_log_probability).mean()
        ),
        "base_structured_target_cross_entropy": base_target_cross_entropy,
        "adapted_structured_target_cross_entropy": final_target_cross_entropy,
        "structured_target_cross_entropy_delta": (
            final_target_cross_entropy - base_target_cross_entropy
        ),
        "base_structured_hard_agreement": float(base_agree.mean()),
        "adapted_structured_hard_agreement": float(final_agree.mean()),
        "changed_toward_structured_target": int(np.sum(~base_agree & final_agree)),
        "changed_away_from_structured_target": int(np.sum(base_agree & ~final_agree)),
        "base_emission_hard_agreement": float(
            np.mean(baseline_selected == emission_prediction)
        ),
        "adapted_emission_hard_agreement": float(
            np.mean(final_selected == emission_prediction)
        ),
        "accuracy_note": (
            "This is label-free diagnostics only; it cannot estimate Test accuracy and "
            "was not used for recipe selection."
        ),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    model, _, stage = load_student(checkpoint_path)
    checkpoint_metadata = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    dataset = P87STestCachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        temporal_augment=False,
    )
    if len(dataset) != 405:
        raise RuntimeError("final Student inference requires all 405 official Test rows")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate_p87s_test,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_ids, logits = infer_student(model, loader, device)
    model.to("cpu")
    log_probability = log_softmax_numpy(logits)
    raw_prediction = logits.argmax(axis=1).astype(np.int64)

    transition, config = load_decoder(args.tiny_decoder.resolve())
    metadata = align_metadata(args.test_metadata.resolve(), np.asarray(sample_ids))
    sessions = build_sessions(
        np.arange(len(sample_ids)),
        metadata,
        config.gap_seconds,
        grouping="anonymous_date",
    )
    decoded_prediction = decode_sessions(log_probability, sessions, transition, config)
    with args.test_csv.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        official_rows = list(csv.DictReader(handle))
    official_ids = [official_id(row["path"]) for row in official_rows]
    if len(official_rows) != 405 or set(official_ids) != set(sample_ids):
        raise RuntimeError("official Test CSV and final Student cache do not align")
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    raw_official = np.asarray([raw_prediction[lookup[value]] for value in official_ids])
    decoded_official = np.asarray([decoded_prediction[lookup[value]] for value in official_ids])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "student_logits.npy", logits)
    np.save(output / "student_log_probability.npy", log_probability.astype(np.float32))
    write_csv(
        output / "submission_p87s_student_raw.csv",
        [
            {"path": row["path"], "prediction": int(value)}
            for row, value in zip(official_rows, raw_official, strict=True)
        ],
    )
    write_csv(
        output / "submission_p87s_student_decoded.csv",
        [
            {"path": row["path"], "prediction": int(value)}
            for row, value in zip(official_rows, decoded_official, strict=True)
        ],
    )
    pixel_valid = np.load(Path(args.pixel_cache).resolve() / "view_valid.npy", mmap_mode="r")
    visual_missing = ~np.asarray(pixel_valid, dtype=bool).any(axis=(1, 2, 3))
    imu_global_mask = np.load(
        Path(args.motion_cache).resolve() / "imu_global_mask.npy", mmap_mode="r"
    )
    imu_missing = ~np.asarray(imu_global_mask, dtype=bool).any(axis=(1, 2))
    audit_rows = []
    for index, sample_id in enumerate(sample_ids):
        probability = np.exp(log_probability[index])
        audit_rows.append(
            {
                "sample_id": sample_id,
                "visual_missing": int(visual_missing[index]),
                "imu_missing": int(imu_missing[index]),
                "raw_prediction": int(raw_prediction[index]),
                "raw_confidence": float(probability.max()),
                "raw_entropy": float(
                    -(probability * log_probability[index]).sum()
                ),
                "decoded_prediction": int(decoded_prediction[index]),
                "decoder_changed": int(raw_prediction[index] != decoded_prediction[index]),
            }
        )
    write_csv(output / "prediction_audit.csv", audit_rows)
    adaptation_audit = optional_adaptation_audit(
        checkpoint_metadata,
        log_probability,
        raw_prediction,
        decoded_prediction,
        sample_ids,
        loader,
        device,
        transition,
        config,
        sessions,
        args.structured_targets,
        output,
    )
    final_probability = np.exp(log_probability)
    missing_modality_rows = [
        {
            "sample_id": sample_id,
            "visual_missing": bool(visual_missing[index]),
            "imu_missing": bool(imu_missing[index]),
            "raw_prediction": int(raw_prediction[index]),
            "raw_confidence": float(final_probability[index].max()),
            "decoded_prediction": int(decoded_prediction[index]),
        }
        for index, sample_id in enumerate(sample_ids)
        if visual_missing[index] or imu_missing[index]
    ]
    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "stage": "P87S_final_student_test_inference",
        "checkpoint_stage": stage,
        "test_rows": len(sample_ids),
        "visual_available_rows": int((~visual_missing).sum()),
        "visual_missing_rows": int(visual_missing.sum()),
        "imu_available_rows": int((~imu_missing).sum()),
        "imu_missing_rows": int(imu_missing.sum()),
        "decoded_sessions": len(sessions),
        "decoder_changed_rows": int(np.sum(raw_prediction != decoded_prediction)),
        "decoder_change_pairs": changed_pair_histogram(
            raw_prediction, decoded_prediction
        ),
        "raw_prediction_class_histogram": class_histogram(raw_prediction),
        "decoded_prediction_class_histogram": class_histogram(decoded_prediction),
        "raw_mean_confidence": float(final_probability.max(axis=1).mean()),
        "raw_mean_entropy": float(
            entropy_from_log_probability(log_probability).mean()
        ),
        "missing_modality_rows": missing_modality_rows,
        "student_parameters": parameters,
        "student_fp32_mib": parameters * 4 / 1024**2,
        "student_checkpoint_bytes": checkpoint_path.stat().st_size,
        "student_checkpoint_sha256": sha256(checkpoint_path),
        "tiny_decoder_bytes": args.tiny_decoder.resolve().stat().st_size,
        "tiny_decoder_sha256": sha256(args.tiny_decoder.resolve()),
        "combined_fp32_parameter_and_decoder_mib": (
            parameters * 4 + args.tiny_decoder.resolve().stat().st_size
        ) / 1024**2,
        "large_model_required_at_inference": False,
        "large_or_legacy_fallback_rows": 0,
        "optional_unlabeled_adaptation_audit": adaptation_audit,
        "accuracy_note": "Test labels are unavailable; score requires submission.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
