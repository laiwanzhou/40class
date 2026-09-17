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
    choose_config,
    decode_unique_beam_posterior,
    fit_transition_model,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEACHER = PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
DEFAULT_METADATA = PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p87s_holdout1_structured_targets_v1"
DEFAULT_HOLDOUT_USERS = ("user6", "user8", "user17", "user23")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe P87 structured posterior targets for a subject-disjoint "
            "pseudo-Test. Holdout labels are masked before any model/config fitting."
        )
    )
    parser.add_argument("--teacher-targets", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--holdout-users", nargs="+", default=DEFAULT_HOLDOUT_USERS
    )
    parser.add_argument("--expected-teacher-fold", type=int, default=1)
    parser.add_argument(
        "--gap-seconds", type=float, nargs="+", default=(20.0, 30.0, 45.0)
    )
    parser.add_argument(
        "--transition-weights", type=float, nargs="+", default=(0.25, 0.30, 0.35)
    )
    parser.add_argument(
        "--trigram-backoffs", type=float, nargs="+", default=(1.0, 2.0, 5.0)
    )
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, 1e-300)), axis=1
    )
    return entropy / np.log(probability.shape[1])


def backed_off_structured_probability(
    emission_probability: np.ndarray,
    structured_probability: np.ndarray,
    beam_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Entropy-adaptive structured target with a finite-beam support safeguard.

    A truncated beam cannot prove that zero retained mass means zero true posterior
    mass. Therefore at least 1/beam_width of the emission distribution is retained;
    the remaining structured weight is reduced further when path marginals are
    uncertain. This is label-free and does not introduce a scanned fixed blend.
    """

    if beam_width <= 1:
        raise ValueError("beam_width must exceed one for finite-beam backoff")
    confidence = 1.0 - normalized_entropy(structured_probability)
    structured_weight = confidence * (1.0 - 1.0 / float(beam_width))
    probability = (
        structured_weight[:, None] * structured_probability
        + (1.0 - structured_weight[:, None]) * emission_probability
    )
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, structured_weight


def build_targets(
    log_probability: np.ndarray,
    labels_with_holdout_masked: np.ndarray,
    train_indices: np.ndarray,
    holdout_indices: np.ndarray,
    metadata,
    config: DecoderConfig,
    posterior_temperature: float,
) -> dict[str, np.ndarray | int]:
    probability = np.exp(np.asarray(log_probability, dtype=np.float64))
    probability /= probability.sum(axis=1, keepdims=True)
    structured_probability = probability.copy()
    structured_map = probability.argmax(axis=1).astype(np.int64)
    session_id = np.full(len(probability), -1, dtype=np.int64)

    train_sessions = build_sessions(
        train_indices, metadata, config.gap_seconds, grouping="known_user"
    )
    model = fit_transition_model(
        labels_with_holdout_masked,
        train_sessions,
        num_classes=probability.shape[1],
        trigram_backoff=config.trigram_backoff,
    )
    holdout_sessions = build_sessions(
        holdout_indices,
        metadata,
        config.gap_seconds,
        grouping="anonymous_date",
    )
    for sequence_id, session in enumerate(holdout_sessions):
        posterior = decode_unique_beam_posterior(
            log_probability[session],
            model,
            transition_weight=config.transition_weight,
            beam_width=config.beam_width,
            posterior_temperature=posterior_temperature,
        )
        structured_probability[session] = posterior.marginals
        structured_map[session] = posterior.paths[0]
        session_id[session] = sequence_id

    return {
        "structured_probability": structured_probability,
        "structured_map_prediction": structured_map,
        "session_id": session_id,
        "train_session_count": len(train_sessions),
        "holdout_session_count": len(holdout_sessions),
    }


def main() -> None:
    args = parse_args()
    if args.posterior_temperature <= 0:
        raise ValueError("posterior-temperature must be positive")
    teacher_path = args.teacher_targets.resolve()
    metadata_path = args.train_metadata.resolve()
    teacher = np.load(teacher_path, allow_pickle=False)
    sample_ids = teacher["oof_sample_ids"].astype(str)
    teacher_folds = teacher["oof_folds"].astype(np.int64)
    log_probability = teacher["oof_teacher_log_probability"].astype(np.float64)
    emission_probability = teacher["oof_teacher_probability"].astype(np.float64)
    metadata = align_metadata(metadata_path, sample_ids)

    holdout_users = tuple(sorted(set(map(str, args.holdout_users))))
    holdout_mask = np.isin(metadata.users, holdout_users)
    if not np.any(holdout_mask):
        raise ValueError(f"No samples found for holdout users {holdout_users}")
    observed_holdout_users = tuple(sorted(set(metadata.users[holdout_mask].tolist())))
    if observed_holdout_users != holdout_users:
        raise ValueError(
            f"Requested holdout users {holdout_users}, observed {observed_holdout_users}"
        )
    holdout_folds = np.unique(teacher_folds[holdout_mask])
    if not np.array_equal(holdout_folds, [int(args.expected_teacher_fold)]):
        raise ValueError(
            "The pseudo-Test must have OOF teacher predictions from exactly the "
            f"expected unseen fold {args.expected_teacher_fold}; got {holdout_folds.tolist()}"
        )

    train_indices = np.flatnonzero(~holdout_mask)
    holdout_indices = np.flatnonzero(holdout_mask)
    labels = teacher["oof_labels"].astype(np.int64).copy()
    # This is the central leakage guard: no downstream fitting call can observe a
    # pseudo-Test label, even accidentally through the full aligned label vector.
    labels[holdout_mask] = -10_000
    protocol_folds = holdout_mask.astype(np.int64)
    config, inner_grid = choose_config(
        labels,
        protocol_folds,
        log_probability,
        metadata,
        outer_fold=1,
        gap_seconds_candidates=tuple(map(float, args.gap_seconds)),
        transition_weights=tuple(map(float, args.transition_weights)),
        trigram_backoffs=tuple(map(float, args.trigram_backoffs)),
        beam_width=int(args.beam_width),
    )
    target = build_targets(
        log_probability,
        labels,
        train_indices,
        holdout_indices,
        metadata,
        config,
        posterior_temperature=float(args.posterior_temperature),
    )
    structured_probability = np.asarray(target["structured_probability"])
    structured_map = np.asarray(target["structured_map_prediction"])
    session_id = np.asarray(target["session_id"])
    marginal_prediction = structured_probability.argmax(axis=1).astype(np.int64)
    confidence = 1.0 - normalized_entropy(structured_probability)
    distillation_probability, structured_weight = backed_off_structured_probability(
        emission_probability,
        structured_probability,
        beam_width=config.beam_width,
    )
    distillation_prediction = distillation_probability.argmax(axis=1).astype(np.int64)
    emission_prediction = emission_probability.argmax(axis=1)
    changed_map = structured_map != emission_prediction
    decoded_mask = session_id >= 0

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "structured_targets.npz"
    np.savez_compressed(
        target_path,
        sample_ids=sample_ids,
        users=metadata.users,
        teacher_folds=teacher_folds,
        target_mask=holdout_mask,
        decoded_mask=decoded_mask,
        emission_probability=emission_probability.astype(np.float32),
        structured_probability=structured_probability.astype(np.float32),
        structured_distillation_probability=distillation_probability.astype(np.float32),
        structured_distillation_weight=structured_weight.astype(np.float32),
        emission_prediction=emission_prediction.astype(np.int64),
        structured_map_prediction=structured_map.astype(np.int64),
        structured_marginal_prediction=marginal_prediction.astype(np.int64),
        structured_distillation_prediction=distillation_prediction.astype(np.int64),
        structured_confidence=confidence.astype(np.float32),
        sequence_session_id=session_id,
        sequence_changed=changed_map,
    )

    counts = {
        user: int(np.sum(metadata.users[holdout_mask] == user))
        for user in holdout_users
    }
    selected_key = (
        f"gap={config.gap_seconds:g},backoff={config.trigram_backoff:g},"
        f"weight={config.transition_weight:g}"
    )
    summary = {
        "stage": "P87-S leakage-safe structured posterior target generation",
        "protocol": (
            "All requested pseudo-Test subjects are excluded from config selection and "
            "transition fitting. Their labels are overwritten by -10000 before those "
            "calls. P85 emissions are OOF from the expected whole outer fold. Only "
            "anonymous recording date/time and teacher emissions are used for decoding."
        ),
        "holdout_users": list(holdout_users),
        "holdout_counts": counts,
        "holdout_rows": int(holdout_mask.sum()),
        "train_rows": int((~holdout_mask).sum()),
        "train_users": sorted(set(metadata.users[~holdout_mask].tolist())),
        "expected_teacher_fold": int(args.expected_teacher_fold),
        "selected_config": {
            "gap_seconds": config.gap_seconds,
            "transition_weight": config.transition_weight,
            "trigram_backoff": config.trigram_backoff,
            "beam_width": config.beam_width,
            "posterior_temperature": float(args.posterior_temperature),
            "inner_loso_accuracy": inner_grid[selected_key],
        },
        "target_mechanism": (
            "Softmax over retained sequence-path scores, followed by per-position "
            "marginalization. Distillation then uses entropy-adaptive emission backoff "
            "and retains at least 1/beam_width emission mass because a finite beam "
            "cannot justify exact zero support. No scanned fixed mixture is used."
        ),
        "train_sessions": int(target["train_session_count"]),
        "holdout_sessions": int(target["holdout_session_count"]),
        "decoded_holdout_rows": int(np.sum(decoded_mask & holdout_mask)),
        "changed_map_holdout_rows": int(np.sum(changed_map & holdout_mask)),
        "marginal_map_differs_from_sequence_map": int(
            np.sum((marginal_prediction != structured_map) & holdout_mask)
        ),
        "mean_structured_confidence_holdout": float(confidence[holdout_mask].mean()),
        "mean_structured_distillation_weight_holdout": float(
            structured_weight[holdout_mask].mean()
        ),
        "mean_emission_to_structured_l1_holdout": float(
            np.abs(structured_probability[holdout_mask] - emission_probability[holdout_mask])
            .sum(axis=1)
            .mean()
        ),
        "teacher_targets": str(teacher_path),
        "teacher_targets_sha256": sha256(teacher_path),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "output": str(target_path),
        "inner_grid": inner_grid,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
