"""Source-OOF sequence gate over the deployable P177 group probability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io
from audit_p87_sequence_decoder import decode_sessions, fit_transition_model
from p117_transductive_multicandidate_router import load_candidate_splits
from p139_soft_sequence_gate import DECODER, emission, gate_features, select_gate, sessions_for


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p179_p177_soft_sequence_gate_v1"
P177 = HERE / "runs/p177_p128_vjepa_group_teacher_v1/predictions.npz"
P89 = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=P177)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args=parse_args()
    data = load_candidate_splits()
    source = np.load(args.source.resolve(), allow_pickle=False)
    split_values = {}
    for name in SPLITS:
        split_values[name] = {
            "ids": data[name].split.sample_ids.astype(str),
            "labels": data[name].split.labels.astype(np.int64),
            "base": source[f"{name}_held_prediction"].astype(np.int64),
            "probability": source[f"{name}_held_probability"].astype(np.float64),
        }
    reports = {}
    held_predictions = {}
    for held_name in SPLITS:
        source_names = [name for name in SPLITS if name != held_name]
        source_ids = np.concatenate([split_values[name]["ids"] for name in source_names])
        source_labels = np.concatenate([split_values[name]["labels"] for name in source_names])
        source_base = np.concatenate([split_values[name]["base"] for name in source_names])
        source_probability = np.concatenate(
            [split_values[name]["probability"] for name in source_names]
        )
        source_sessions = sessions_for(data, source_ids, source_names)
        transition = fit_transition_model(
            source_labels, source_sessions, 40, DECODER.trigram_backoff
        )
        source_sequence = decode_sessions(
            emission(source_probability, source_base),
            source_sessions,
            transition,
            DECODER,
        )
        source_features = gate_features(source_probability, source_base, source_sequence)
        selected = select_gate(source_base, source_sequence, source_features, source_labels)
        held = split_values[held_name]
        held_sessions = sessions_for(data, held["ids"], [held_name])
        held_sequence = decode_sessions(
            emission(held["probability"], held["base"]),
            held_sessions,
            transition,
            DECODER,
        )
        held_features = gate_features(held["probability"], held["base"], held_sequence)
        route = (held_sequence != held["base"]) & (
            held_features[:, int(selected["score_index"])] >= float(selected["threshold"])
        )
        prediction = held["base"].copy()
        prediction[route] = held_sequence[route]
        base_correct = held["base"] == held["labels"]
        final_correct = prediction == held["labels"]
        reports[held_name] = {
            "source_cohorts": source_names,
            "source_gate": selected,
            "held": {
                "base_correct": int(base_correct.sum()),
                "correct": int(final_correct.sum()),
                "net": int(final_correct.sum() - base_correct.sum()),
                "rescue": int(np.sum(~base_correct & final_correct)),
                "harm": int(np.sum(base_correct & ~final_correct)),
                "changed": int(route.sum()),
            },
        }
        held_predictions[held_name] = prediction

    all_ids = np.concatenate([split_values[name]["ids"] for name in SPLITS])
    all_labels = np.concatenate([split_values[name]["labels"] for name in SPLITS])
    all_base = np.concatenate([split_values[name]["base"] for name in SPLITS])
    all_probability = np.concatenate([split_values[name]["probability"] for name in SPLITS])
    all_sessions = sessions_for(data, all_ids, list(SPLITS))
    transition = fit_transition_model(all_labels, all_sessions, 40, DECODER.trigram_backoff)
    all_sequence = decode_sessions(
        emission(all_probability, all_base), all_sessions, transition, DECODER
    )
    all_features = gate_features(all_probability, all_base, all_sequence)
    final_gate = select_gate(all_base, all_sequence, all_features, all_labels)

    test_ids = source["sample_ids"].astype(str)
    test_base = source["prediction"].astype(np.int64)
    test_probability = source["probability"].astype(np.float64)
    # Reuse the metadata/session constructor by exposing a minimal object aligned
    # with the Test metadata through the canonical helper implementation.
    from audit_p87_sequence_decoder import align_metadata, build_sessions
    metadata = align_metadata(HERE / "data/p85_recording_metadata/test_recording_metadata.csv", test_ids)
    test_sessions = build_sessions(
        np.arange(len(test_ids), dtype=np.int64), metadata, DECODER.gap_seconds, "anonymous_date"
    )
    test_sequence = decode_sessions(
        emission(test_probability, test_base), test_sessions, transition, DECODER
    )
    test_features = gate_features(test_probability, test_base, test_sequence)
    test_route = (test_sequence != test_base) & (
        test_features[:, int(final_gate["score_index"])] >= float(final_gate["threshold"])
    )
    test_prediction = test_base.copy()
    test_prediction[test_route] = test_sequence[test_route]
    output_dir=args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    submission = output_dir / "submission_soft_sequence.csv"
    submission_io.write_submission(submission, submission_io.read_rows(P89), test_prediction)
    prediction = np.concatenate([held_predictions[name] for name in SPLITS])
    correct = int(np.sum(prediction == all_labels))
    report = {
        "stage": "P179_P177_soft_sequence_gate",
        "status": "complete",
        "protocol": {
            "outer_source_only_gate": True,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
            "decoder": {
                "gap_seconds": DECODER.gap_seconds,
                "transition_weight": DECODER.transition_weight,
                "trigram_backoff": DECODER.trigram_backoff,
                "beam_width": DECODER.beam_width,
            },
        },
        "cohorts": reports,
        "aggregate": {
            "rows": len(all_labels),
            "correct": correct,
            "accuracy": correct / len(all_labels),
            "p177_correct": int(np.sum(all_base == all_labels)),
            "net_vs_p177": correct - int(np.sum(all_base == all_labels)),
        },
        "final_refit_gate": final_gate,
        "test": {
            "base_changes_vs_p89": int(np.sum(test_base != source["base_prediction"])),
            "sequence_changed_vs_p177": int(test_route.sum()),
            "submission": str(submission.resolve()),
            "test_labels_read": False,
        },
    }
    np.savez_compressed(
        output_dir / "predictions.npz",
        sample_ids=test_ids,
        base_prediction=source["base_prediction"],
        p177_prediction=test_base,
        probability=test_probability.astype(np.float32),
        sequence_prediction=test_sequence,
        route=test_route,
        prediction=test_prediction,
        **{f"{name}_held_prediction": held_predictions[name] for name in SPLITS},
    )
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
