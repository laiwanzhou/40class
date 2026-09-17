from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import p89_build_dual_consensus_submission as io
import p89_full40_scale_invariant_transfer as full40
import p89_supervised_router_transfer as router
from audit_p87_sequence_decoder import DecoderConfig, TransitionModel, align_metadata
from p89_global_repeat_decoder import GlobalRepeatConfig, decode_global_repeat


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_supervised_router_test_v1"


def features_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    stacked = np.asarray(probabilities, dtype=np.float64)
    ordered = np.sort(stacked, axis=2)
    entropy = -np.sum(
        stacked * np.log(np.maximum(stacked, 1e-12)), axis=2
    )
    diagnostics = np.stack(
        (
            entropy,
            ordered[:, :, -1],
            ordered[:, :, -1] - ordered[:, :, -2],
        ),
        axis=2,
    )
    aggregates = np.concatenate(
        (
            stacked.mean(axis=1),
            stacked.max(axis=1),
            stacked.std(axis=1),
            np.median(stacked, axis=1),
        ),
        axis=1,
    )
    return np.concatenate(
        (
            stacked.reshape(len(stacked), -1),
            diagnostics.reshape(len(stacked), -1),
            aggregates,
        ),
        axis=1,
    ).astype(np.float32)


def load_decoder() -> tuple[TransitionModel, DecoderConfig]:
    with np.load(
        PROJECT_DIR / "runs/p87s_tiny_decoder_v1/tiny_decoder.npz"
    ) as saved:
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
    return transition, decoder


def main() -> None:
    source = json.loads(
        (
            PROJECT_DIR
            / "runs/p89_supervised_router_h1_to_h2_v3/summary.json"
        ).read_text(encoding="utf-8")
    )
    configuration = source["selected_config"]
    global_source = json.loads(
        (PROJECT_DIR / "runs/p89_global_repeat_h1_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    global_config = GlobalRepeatConfig(**global_source["selected_config"])

    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    train_features = []
    train_labels = []
    expert_names = None
    for protocol_value in (h1, h2):
        expert_probability, names = full40.train_probabilities(protocol_value[0])
        if expert_names is not None and names != expert_names:
            raise RuntimeError("training expert order changed")
        expert_names = names
        stacked = np.concatenate(
            (protocol_value[2][:, None, :], expert_probability), axis=1
        )
        current = features_from_probabilities(stacked)
        reference, _ = router.router_features(protocol_value[0], protocol_value[2])
        if not np.allclose(current, reference, atol=1e-7):
            raise RuntimeError("router feature reconstruction failed")
        train_features.append(current)
        train_labels.append(protocol_value[1])
    x_train = np.concatenate(train_features, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    p87_dir = PROJECT_DIR / "runs/p87s_final_test_predictions_v1"
    source_rows = io.read_rows(p87_dir / "submission_p87s_student_decoded.csv")
    p87_prediction = np.asarray(
        [int(row["prediction"]) for row in source_rows], dtype=np.int64
    )
    test_base = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    sample_ids = test_base["sample_ids"].astype(str)
    routed_ids = test_base["detail_sample_ids"].astype(str)
    base_probability = np.asarray(test_base["base_probability"], dtype=np.float64)
    expert_probability, test_names = full40.test_probabilities(routed_ids)
    if test_names != expert_names:
        raise RuntimeError("Test expert order differs from training")
    lookup = {value: index for index, value in enumerate(sample_ids)}
    routed_positions = np.asarray(
        [lookup[value] for value in routed_ids], dtype=np.int64
    )
    x_test = features_from_probabilities(
        np.concatenate(
            (base_probability[routed_positions, None, :], expert_probability),
            axis=1,
        )
    )

    scaler = StandardScaler()
    model = router.make_model(
        str(configuration["model"]), float(configuration["regularization"])
    )
    model.fit(scaler.fit_transform(x_train), y_train)
    routed = router.probability(model, scaler.transform(x_test))
    routed = router.softmax(
        np.log(np.maximum(routed, 1e-12)) / float(configuration["temperature"])
    )
    weight = float(configuration["weight"])
    blended = base_probability.copy()
    blended[routed_positions] = (
        (1.0 - weight) * base_probability[routed_positions] + weight * routed
    )
    blended /= blended.sum(axis=1, keepdims=True)

    transition, decoder = load_decoder()
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        sample_ids,
    )
    indices = np.arange(len(sample_ids), dtype=np.int64)
    base_global, base_grouping = decode_global_repeat(
        np.log(np.maximum(base_probability, 1e-12)),
        indices,
        metadata,
        transition,
        decoder,
        global_config,
    )
    prediction, grouping = decode_global_repeat(
        np.log(np.maximum(blended, 1e-12)),
        indices,
        metadata,
        transition,
        decoder,
        global_config,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_supervised_router.csv"
    io.write_submission(submission, source_rows, prediction)
    np.savez_compressed(
        OUTPUT / "test_predictions.npz",
        sample_ids=sample_ids,
        routed_sample_ids=routed_ids,
        p87_prediction=p87_prediction,
        global_base_prediction=base_global,
        router_prediction=prediction,
        router_probability=routed.astype(np.float32),
        blended_probability=blended.astype(np.float32),
        expert_probability=expert_probability.astype(np.float32),
    )
    report = {
        "stage": "P89_supervised_router_Test_deployment_v1",
        "protocol": (
            "Freeze the H1-selected/H2-confirmed router configuration, refit on "
            "the union of H1 and H2 labels, and deploy to Test using the same 20 "
            "experts and frozen global-repeat decoder."
        ),
        "configuration": configuration,
        "fit_rows": int(len(y_train)),
        "expert_names": ["p87", *expert_names],
        "base_global_grouping": base_grouping,
        "router_grouping": grouping,
        "base_global_changes_vs_p87": int(np.sum(base_global != p87_prediction)),
        "router_changes_vs_p87": int(np.sum(prediction != p87_prediction)),
        "router_changes_vs_global_base": int(np.sum(prediction != base_global)),
        "submission": {
            "path": str(submission.resolve()),
            "sha256": io.digest(submission),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
