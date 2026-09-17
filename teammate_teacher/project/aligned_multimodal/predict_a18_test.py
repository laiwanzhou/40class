"""Run the full A18 Teacher and frozen A18 Session mechanism on Kaggle Test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_p87_sequence_decoder import (
    DecoderConfig,
    TransitionModel,
    align_metadata,
    build_sessions,
    decode_unique_beam_posterior,
)
from build_p87s_structured_targets import backed_off_structured_probability
from p100a_global_teacher_data import FoldNormalizer, P100AData, P100ADataset, _align
from p100a_global_teacher_model import P100AGlobalTeacher, P100AModelConfig
from train_p100a_global_teacher_oof import evaluate_model, make_loader, softmax_numpy


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_CHECKPOINT = HERE / "runs/a18_full_teacher_v1/a18_full_final.pt"
DEFAULT_SESSION = HERE / "runs/a18_full_teacher_v1/session_transition_state.npz"
DEFAULT_VMAE = REPO / "runs/p90_videomaev2_distilled_test_v1/complete_features.npz"
DEFAULT_IV2 = REPO / "runs/p90_internvideo2_l_k400_test_v1/complete_features.npz"
DEFAULT_FEATURES = HERE / "runs/a18_test_features_v1"
DEFAULT_MOTION = HERE / "runs/p87s_test_motion_window_t16_v1"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/test_recording_metadata.csv"
DEFAULT_TEST_CSV = REPO / "Testing/test.csv"
DEFAULT_P87S_LOG_PROBABILITY = (
    HERE / "runs/p87s_final_test_predictions_v1/student_log_probability.npy"
)
DEFAULT_P87S_AUDIT = HERE / "runs/p87s_final_test_predictions_v1/prediction_audit.csv"
DEFAULT_P87S_SUBMISSION = (
    HERE / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
)
DEFAULT_OUTPUT = HERE / "runs/a18_full_teacher_test_v1"
TEST_ROWS = 405
CLASSES = 40
MODALITIES = ("visual", "skeleton")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--session-state", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--vmae", type=Path, default=DEFAULT_VMAE)
    parser.add_argument("--iv2", type=Path, default=DEFAULT_IV2)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--test-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument(
        "--p87s-log-probability", type=Path, default=DEFAULT_P87S_LOG_PROBABILITY
    )
    parser.add_argument("--p87s-audit", type=Path, default=DEFAULT_P87S_AUDIT)
    parser.add_argument("--p87s-submission", type=Path, default=DEFAULT_P87S_SUBMISSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def official_id(path_value: str) -> str:
    return path_value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def mean_filled_alignment(
    source_ids: np.ndarray,
    values: np.ndarray,
    target_ids: np.ndarray,
    training_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_ids).astype(str)
    target = np.asarray(target_ids).astype(str)
    matrix = np.asarray(values)
    if len(source) != len(matrix) or len(np.unique(source)) != len(source):
        raise RuntimeError("visual Test feature sample-id contract changed")
    output = np.empty((len(target), *matrix.shape[1:]), dtype=np.float32)
    output[...] = np.asarray(training_mean, dtype=np.float32)
    lookup = {sample_id: index for index, sample_id in enumerate(source)}
    available = np.asarray([sample_id in lookup for sample_id in target], dtype=bool)
    selected = np.flatnonzero(available)
    source_rows = np.asarray([lookup[target[index]] for index in selected], dtype=np.int64)
    output[selected] = matrix[source_rows].astype(np.float32)
    return output, available


def class_histogram(prediction: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(prediction, dtype=np.int64), minlength=CLASSES)
    return {str(index): int(value) for index, value in enumerate(counts) if value}


def changed_pair_histogram(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    pairs: dict[str, int] = {}
    for source, target in zip(before, after, strict=True):
        if int(source) == int(target):
            continue
        key = f"{int(source)}->{int(target)}"
        pairs[key] = pairs.get(key, 0) + 1
    return dict(sorted(pairs.items(), key=lambda item: (-item[1], item[0])))


def entropy(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    return -np.sum(values * np.log(np.maximum(values, 1e-300)), axis=1)


def load_p87s_probability(
    probability_path: Path, audit_path: Path, target_ids: np.ndarray
) -> np.ndarray:
    log_probability = np.asarray(np.load(probability_path.resolve()), dtype=np.float64)
    rows = read_csv(audit_path)
    source_ids = np.asarray([row["sample_id"] for row in rows]).astype(str)
    if log_probability.shape != (len(source_ids), CLASSES):
        raise RuntimeError("P87S Test probability contract changed")
    probability = np.exp(log_probability)
    probability /= probability.sum(axis=1, keepdims=True)
    aligned = _align(source_ids, probability, target_ids)
    if aligned.shape != (TEST_ROWS, CLASSES) or not np.isfinite(aligned).all():
        raise RuntimeError("aligned P87S fallback probability is invalid")
    return aligned


def build_test_data(
    target_ids: np.ndarray,
    checkpoint: dict[str, Any],
    vmae_path: Path,
    iv2_path: Path,
    feature_dir: Path,
    motion_cache: Path,
) -> tuple[P100AData, np.ndarray]:
    means = checkpoint["normalizer_means"]
    with np.load(vmae_path.resolve(), allow_pickle=False) as archive:
        vmae_ids = np.asarray(archive["sample_ids"]).astype(str)
        vmae_feature = np.asarray(archive["features"]).reshape(-1, 6, 768)
        vmae_action = np.asarray(archive["action_logits"]).reshape(-1, 6, 710)
    with np.load(iv2_path.resolve(), allow_pickle=False) as archive:
        iv2_ids = np.asarray(archive["sample_ids"]).astype(str)
        iv2_feature = np.asarray(archive["features"]).reshape(-1, 6, 768)
        iv2_action = np.asarray(archive["action_logits"]).reshape(-1, 6, 400)
    visual_vmae, vmae_available = mean_filled_alignment(
        vmae_ids, vmae_feature, target_ids, means["visual_vmae"]
    )
    visual_vmae_action, vmae_action_available = mean_filled_alignment(
        vmae_ids, vmae_action, target_ids, means["visual_vmae_action"]
    )
    visual_iv2, iv2_available = mean_filled_alignment(
        iv2_ids, iv2_feature, target_ids, means["visual_iv2"]
    )
    visual_iv2_action, iv2_action_available = mean_filled_alignment(
        iv2_ids, iv2_action, target_ids, means["visual_iv2_action"]
    )
    visual_masks = (
        vmae_available,
        vmae_action_available,
        iv2_available,
        iv2_action_available,
    )
    if any(not np.array_equal(visual_masks[0], mask) for mask in visual_masks[1:]):
        raise RuntimeError("A18 Test visual feature availability differs by teacher")

    with np.load(
        feature_dir / "motionbert_pretrain_front_t81.npz", allow_pickle=False
    ) as archive:
        motionbert_ids = np.asarray(archive["sample_ids"]).astype(str)
        motionbert = np.asarray(archive["features"])
    with np.load(
        feature_dir / "hdgcn_six_stream_tokens.npz", allow_pickle=False
    ) as archive:
        hdgcn_ids = np.asarray(archive["sample_ids"]).astype(str)
        hdgcn = np.asarray(archive["tokens"])
    with np.load(feature_dir / "crossmodal_statistics.npz", allow_pickle=False) as archive:
        statistics_ids = np.asarray(archive["sample_ids"]).astype(str)
        statistics = np.asarray(archive["features"])
    motion_rows = read_csv(motion_cache / "rows.csv")
    motion_ids = np.asarray([row["sample_id"] for row in motion_rows]).astype(str)
    skeleton_sequence = np.load(
        motion_cache / "skeleton_features.npy", mmap_mode="r"
    ).reshape(len(motion_ids), 32, 17, 13)
    skeleton_mask = np.load(
        motion_cache / "skeleton_joint_mask.npy", mmap_mode="r"
    ).reshape(len(motion_ids), 32, 17)

    cross = _align(statistics_ids, statistics, target_ids).astype(np.float32)
    sequence = _align(motion_ids, skeleton_sequence, target_ids).astype(np.float32)
    mask = _align(motion_ids, skeleton_mask, target_ids).astype(np.float32)
    rows = len(target_ids)
    data = P100AData(
        sample_ids=target_ids,
        users=np.full(rows, "anonymous"),
        labels=np.zeros(rows, dtype=np.int64),
        fold_ids=np.full(rows, -1, dtype=np.int64),
        visual_vmae=visual_vmae,
        visual_iv2=visual_iv2,
        visual_vmae_action=visual_vmae_action,
        visual_iv2_action=visual_iv2_action,
        skeleton_motionbert=_align(
            motionbert_ids, motionbert, target_ids
        ).reshape(rows, 12, 768).astype(np.float32),
        skeleton_hdgcn=_align(hdgcn_ids, hdgcn, target_ids).astype(np.float32),
        skeleton_sequence=sequence,
        skeleton_mask=mask,
        skeleton_statistics=np.concatenate(
            (cross[:, :2629], cross[:, 5729:5740]), axis=1
        ).astype(np.float32),
        imu_sequence=np.empty((rows, 0), dtype=np.float32),
        imu_mask=np.empty((rows, 0), dtype=np.float32),
        imu_statistics=np.empty((rows, 0), dtype=np.float32),
        cross_statistics=np.empty((rows, 0), dtype=np.float32),
        skeleton_available=(mask.sum(axis=(1, 2)) > 0).astype(np.float32),
        imu_available=np.zeros(rows, dtype=np.float32),
    )
    if data.skeleton_motionbert.shape != (TEST_ROWS, 12, 768):
        raise RuntimeError("A18 Test MotionBERT shape changed")
    if data.skeleton_hdgcn.shape != (TEST_ROWS, 6, 16, 256):
        raise RuntimeError("A18 Test HD-GCN shape changed")
    if data.skeleton_statistics.shape != (TEST_ROWS, 2640):
        raise RuntimeError("A18 Test Skeleton statistics shape changed")
    if not data.skeleton_available.all():
        raise RuntimeError("A18 Test requires Skeleton coverage on all 405 rows")
    return data, visual_masks[0]


def load_session(path: Path) -> tuple[TransitionModel, DecoderConfig, float]:
    with np.load(path.resolve(), allow_pickle=False) as archive:
        transition = TransitionModel(
            start_log_probability=np.asarray(archive["start_log_probability"]),
            end_log_probability=np.asarray(archive["end_log_probability"]),
            bigram_log_probability=np.asarray(archive["bigram_log_probability"]),
            trigram_log_probability=np.asarray(archive["trigram_log_probability"]),
        )
        config = DecoderConfig(
            gap_seconds=float(archive["gap_seconds"]),
            transition_weight=float(archive["transition_weight"]),
            trigram_backoff=float(archive["trigram_backoff"]),
            beam_width=int(archive["beam_width"]),
        )
        temperature = float(archive["posterior_temperature"])
    if transition.trigram_log_probability.shape != (CLASSES, CLASSES, CLASSES):
        raise RuntimeError("A18 Session class contract changed")
    return transition, config, temperature


def apply_session(
    emission_probability: np.ndarray,
    sample_ids: np.ndarray,
    metadata_path: Path,
    session_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], DecoderConfig]:
    transition, config, temperature = load_session(session_path)
    metadata = align_metadata(metadata_path.resolve(), sample_ids)
    sessions = build_sessions(
        np.arange(len(sample_ids)), metadata, config.gap_seconds, "anonymous_date"
    )
    log_probability = np.log(np.maximum(emission_probability, 1e-300))
    structured = emission_probability.copy()
    structured_map = emission_probability.argmax(axis=1).astype(np.int64)
    session_id = np.full(len(sample_ids), -1, dtype=np.int64)
    for sequence_id, session in enumerate(sessions):
        posterior = decode_unique_beam_posterior(
            log_probability[session],
            transition,
            transition_weight=config.transition_weight,
            beam_width=config.beam_width,
            posterior_temperature=temperature,
        )
        structured[session] = posterior.marginals
        structured_map[session] = posterior.paths[0]
        session_id[session] = sequence_id
    selected, structured_weight = backed_off_structured_probability(
        emission_probability, structured, config.beam_width
    )
    return selected, structured_map, structured_weight, sessions, config


def audit_submission(path: Path, official_rows: list[dict[str, str]]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    predictions = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    checks = {
        "columns_exact": columns == ["path", "prediction"],
        "row_count_405": len(rows) == TEST_ROWS,
        "path_order_exact": [row["path"] for row in rows]
        == [row["path"] for row in official_rows],
        "prediction_range_0_39": bool(
            len(predictions) == TEST_ROWS
            and predictions.min(initial=0) >= 0
            and predictions.max(initial=0) < CLASSES
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"A18 submission audit failed: {checks}")
    return {"path": str(path), "sha256": sha256(path), "checks": checks}


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    official_rows = read_csv(args.test_csv)
    if len(official_rows) != TEST_ROWS:
        raise RuntimeError(f"official Test CSV row count changed: {len(official_rows)}")
    sample_ids = np.asarray([official_id(row["path"]) for row in official_rows])
    if len(np.unique(sample_ids)) != TEST_ROWS:
        raise RuntimeError("official Test CSV contains duplicate sample ids")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = P100AModelConfig(**checkpoint["model_config"])
    if tuple(model_config.modalities) != MODALITIES:
        raise RuntimeError("A18 checkpoint is not the frozen Visual+Skeleton model")
    data, visual_available = build_test_data(
        sample_ids,
        checkpoint,
        args.vmae.resolve(),
        args.iv2.resolve(),
        args.feature_dir.resolve(),
        args.motion_cache.resolve(),
    )
    missing_visual = ~visual_available
    expected_missing = {
        "SM_test_0012",
        "SM_test_0014",
        "SM_test_0154",
        "SM_test_0194",
    }
    if set(sample_ids[missing_visual].tolist()) != expected_missing:
        raise RuntimeError("A18 Test unreadable-IR contract changed")

    normalizer = FoldNormalizer(
        means={key: np.asarray(value) for key, value in checkpoint["normalizer_means"].items()},
        stds={key: np.asarray(value) for key, value in checkpoint["normalizer_stds"].items()},
    )
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    model = P100AGlobalTeacher(model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    result = evaluate_model(
        model,
        make_loader(
            P100ADataset(
                data,
                np.arange(TEST_ROWS, dtype=np.int64),
                normalizer,
                MODALITIES,
            ),
            args.batch_size,
            shuffle=False,
            seed=20260823,
        ),
        device,
    )
    model.to("cpu")
    if not np.array_equal(result["rows"], np.arange(TEST_ROWS)):
        raise RuntimeError("A18 Test inference row order changed")
    logits = np.asarray(result["logits"], dtype=np.float32)
    raw_a18_probability = softmax_numpy(logits).astype(np.float64)

    # The two P90 visual teachers cannot read four official rows.  Their tensors are
    # filled with the training mean only to obtain a finite diagnostic A18 output;
    # the deployable emission is replaced unconditionally by the already-frozen,
    # label-free P87S probability on exactly those four rows.
    p87s_probability = load_p87s_probability(
        args.p87s_log_probability, args.p87s_audit, sample_ids
    )
    emission_probability = raw_a18_probability.copy()
    emission_probability[missing_visual] = p87s_probability[missing_visual]
    selected_probability, structured_map, structured_weight, sessions, session_config = (
        apply_session(
            emission_probability,
            sample_ids,
            args.test_metadata,
            args.session_state,
        )
    )
    raw_a18_prediction = raw_a18_probability.argmax(axis=1).astype(np.int64)
    emission_prediction = emission_probability.argmax(axis=1).astype(np.int64)
    final_prediction = selected_probability.argmax(axis=1).astype(np.int64)
    top5 = np.argsort(-selected_probability, axis=1, kind="stable")[:, :5]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_submission = output / "submission_a18_full_teacher_session.csv"
    write_csv(
        final_submission,
        [
            {"path": row["path"], "prediction": int(prediction)}
            for row, prediction in zip(official_rows, final_prediction, strict=True)
        ],
    )
    raw_submission = output / "submission_a18_raw_diagnostic.csv"
    write_csv(
        raw_submission,
        [
            {"path": row["path"], "prediction": int(prediction)}
            for row, prediction in zip(official_rows, emission_prediction, strict=True)
        ],
    )
    audit_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids):
        audit_rows.append(
            {
                "sample_id": sample_id,
                "visual_available": int(visual_available[index]),
                "skeleton_available": int(data.skeleton_available[index]),
                "pure_a18_prediction": int(raw_a18_prediction[index]),
                "deploy_emission_prediction": int(emission_prediction[index]),
                "missing_visual_fallback": int(missing_visual[index]),
                "emission_confidence": float(emission_probability[index].max()),
                "session_prediction": int(final_prediction[index]),
                "session_changed": int(final_prediction[index] != emission_prediction[index]),
                "structured_map_prediction": int(structured_map[index]),
                "structured_weight": float(structured_weight[index]),
                "top1": int(top5[index, 0]),
                "top2": int(top5[index, 1]),
                "top3": int(top5[index, 2]),
                "top4": int(top5[index, 3]),
                "top5": int(top5[index, 4]),
            }
        )
    write_csv(output / "prediction_audit.csv", audit_rows)
    np.savez_compressed(
        output / "test_predictions.npz",
        sample_ids=sample_ids,
        visual_available=visual_available,
        raw_a18_logits=logits,
        raw_a18_probability=raw_a18_probability.astype(np.float32),
        deploy_emission_probability=emission_probability.astype(np.float32),
        selected_probability=selected_probability.astype(np.float32),
        selected_prediction=final_prediction,
        structured_map_prediction=structured_map,
        structured_weight=structured_weight.astype(np.float32),
    )

    previous_rows = read_csv(args.p87s_submission)
    previous_by_path = {row["path"]: int(row["prediction"]) for row in previous_rows}
    previous_prediction = np.asarray(
        [previous_by_path[row["path"]] for row in official_rows], dtype=np.int64
    )
    summary = {
        "status": "complete",
        "stage": "A18_full_teacher_Kaggle_Test_inference",
        "protocol": (
            "A18 full Visual+Skeleton checkpoint on all 405 official Test rows; fixed "
            "P87S probability fallback on four unreadable-IR rows; one frozen A18 "
            "Session posterior recipe; no labels, sweep, B, router, or specialist"
        ),
        "test_rows": TEST_ROWS,
        "inputs": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
            "session_state": {
                "path": str(args.session_state.resolve()),
                "sha256": sha256(args.session_state),
            },
            "vmae_test_features": {
                "path": str(args.vmae.resolve()),
                "sha256": sha256(args.vmae),
            },
            "iv2_test_features": {
                "path": str(args.iv2.resolve()),
                "sha256": sha256(args.iv2),
            },
            "feature_summary": {
                "path": str(args.feature_dir.resolve() / "summary.json"),
                "sha256": sha256(args.feature_dir.resolve() / "summary.json"),
            },
            "p87s_fallback_log_probability": {
                "path": str(args.p87s_log_probability.resolve()),
                "sha256": sha256(args.p87s_log_probability),
            },
            "previous_p87s_submission": {
                "path": str(args.p87s_submission.resolve()),
                "sha256": sha256(args.p87s_submission),
            },
            "official_test_csv": {
                "path": str(args.test_csv.resolve()),
                "sha256": sha256(args.test_csv),
            },
        },
        "coverage": {
            "visual_rows": int(visual_available.sum()),
            "visual_missing_rows": int(missing_visual.sum()),
            "visual_missing_sample_ids": sample_ids[missing_visual].tolist(),
            "skeleton_rows": int(data.skeleton_available.sum()),
            "p87s_probability_fallback_rows": int(missing_visual.sum()),
            "fallback_rule": "visual unreadable only; fixed four-row manifest contract",
        },
        "session": {
            "sessions": len(sessions),
            "gap_seconds": session_config.gap_seconds,
            "transition_weight": session_config.transition_weight,
            "trigram_backoff": session_config.trigram_backoff,
            "beam_width": session_config.beam_width,
            "changed_rows": int(np.sum(final_prediction != emission_prediction)),
            "change_pairs": changed_pair_histogram(
                emission_prediction, final_prediction
            ),
            "mean_structured_weight": float(structured_weight.mean()),
        },
        "diagnostics_without_test_labels": {
            "pure_a18_mean_confidence": float(raw_a18_probability.max(axis=1).mean()),
            "deploy_emission_mean_confidence": float(
                emission_probability.max(axis=1).mean()
            ),
            "selected_mean_confidence": float(
                selected_probability.max(axis=1).mean()
            ),
            "pure_a18_mean_entropy": float(entropy(raw_a18_probability).mean()),
            "selected_mean_entropy": float(entropy(selected_probability).mean()),
            "raw_class_histogram": class_histogram(emission_prediction),
            "final_class_histogram": class_histogram(final_prediction),
            "vs_previous_p87s_decoded_changed_rows": int(
                np.sum(final_prediction != previous_prediction)
            ),
            "vs_previous_p87s_decoded_agreement": float(
                np.mean(final_prediction == previous_prediction)
            ),
            "vs_previous_p87s_decoded_change_pairs": changed_pair_histogram(
                previous_prediction, final_prediction
            ),
            "accuracy_note": "Kaggle Test labels are unavailable; score requires submission.",
        },
        "submission": audit_submission(final_submission, official_rows),
        "raw_diagnostic_submission": audit_submission(raw_submission, official_rows),
        "constraints": {
            "test_labels_read": False,
            "test_labels_used_for_selection": False,
            "b_teacher_used": False,
            "router_used": False,
            "specialist_used": False,
            "threshold_seed_blend_sweep": False,
            "session_recipe_count": 1,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
