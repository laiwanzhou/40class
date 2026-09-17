from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import (
    DecoderConfig,
    align_metadata,
    build_sessions,
    classification_metrics,
    fit_transition_model,
)
from p88_train_depth_residual import rescue_harm
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_rescue_gate import aligned_imu
from p89_peer_supported_triple_repair import decode
from p89_deterministic_triple_repeat import triple_groups


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_peer_supported_triple_h3_audit_v1"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz") as source:
        all_ids = source["oof_sample_ids"].astype(str)
        all_labels = source["oof_labels"].astype(np.int64)
        folds = source["oof_folds"].astype(np.int64)
        all_probability = source["oof_teacher_probability"].astype(np.float64)
    all_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv", all_ids
    )
    summary = json.loads(
        (PROJECT_DIR / "runs/p87_sequence_decoder_v1/summary.json").read_text(encoding="utf-8")
    )
    fold_summary = next(item for item in summary["folds"] if int(item["fold"]) == 0)
    decoder = DecoderConfig(**fold_summary["selected"])
    fit = np.flatnonzero(folds != 0)
    transition = fit_transition_model(
        all_labels,
        build_sessions(fit, all_metadata, decoder.gap_seconds, "known_user"),
        40,
        decoder.trigram_backoff,
    )
    validation = np.flatnonzero(folds == 0)
    sample_ids = all_ids[validation]
    labels = all_labels[validation]
    probability = all_probability[validation].copy()
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv", sample_ids
    )
    with np.load(PROJECT_DIR / "runs/p87_sequence_decoder_v1/oof_predictions.npz") as source:
        p87 = source["sequence_predictions"].astype(np.int64)[validation]
    with np.load(PROJECT_DIR / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz") as source:
        imu_probability, present = aligned_imu(
            source["sample_ids"].astype(str),
            np.asarray(source["imu_logits"], dtype=np.float64),
            sample_ids,
            3.0,
        )
    probability[present] = 0.95 * probability[present] + 0.05 * imu_probability[present]
    probability /= probability.sum(axis=1, keepdims=True)
    indices = np.arange(len(sample_ids), dtype=np.int64)
    protocol = (
        sample_ids,
        labels,
        probability,
        p87,
        metadata,
        indices,
        build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date"),
        transition,
        decoder,
        None,
    )
    grouping_summary = json.loads(
        (PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json").read_text(encoding="utf-8")
    )
    grouping = GlobalRepeatConfig(**grouping_summary["H1_selected"]["configuration"])
    safe = joint_decode(
        probability,
        p87,
        protocol,
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
    )[0]
    prediction, grouping_audit = decode(protocol, probability, safe)
    group_audit = []
    for group_index, group in enumerate(triple_groups(protocol, "exact_three")):
        rows = np.concatenate(group)
        paths = [tuple(labels[session].tolist()) for session in group]
        starts = [float(np.min(metadata.starts[session])) for session in group]
        group_audit.append({
            "group": group_index,
            "users": [str(metadata.users[session[0]]) for session in group],
            "length": int(len(group[0])),
            "starts": starts,
            "gaps": [starts[1] - starts[0], starts[2] - starts[1]],
            "exact_label_path": len(set(paths)) == 1,
            "paths": [list(path) for path in paths],
            "safe_paths": [safe[session].astype(int).tolist() for session in group],
            "changes": int(np.sum(prediction[rows] != safe[rows])),
            "net": int(np.sum(prediction[rows] == labels[rows]) - np.sum(safe[rows] == labels[rows])),
        })
    users = metadata.users.astype(str)
    per_user = {
        user: int(
            np.sum(prediction[users == user] == labels[users == user])
            - np.sum(safe[users == user] == labels[users == user])
        )
        for user in sorted(set(users.tolist()))
    }
    report = {
        "stage": "P89_peer_supported_exact_triple_H3_fold0_audit_v1",
        "protocol": (
            "The exact-three equal-length peer-supported rule is frozen from H1/H2. "
            "H3 is the untouched original subject fold 0 with six users and its own "
            "strict nested decoder. The H3 base uses the P85 teacher, making this an "
            "independent structural confirmation rather than a direct P87-S score estimate."
        ),
        "users": sorted(set(users.tolist())),
        "safe_metrics": classification_metrics(labels, safe),
        "candidate_metrics": classification_metrics(labels, prediction),
        "rescue_harm_vs_safe": rescue_harm(labels, safe, prediction),
        "per_user_gain": per_user,
        "minimum_user_gain": int(min(per_user.values())),
        "grouping": grouping_audit,
        "exact_group_audit": {
            "groups": len(group_audit),
            "exact": int(sum(item["exact_label_path"] for item in group_audit)),
            "details": group_audit,
        },
    }
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        sample_ids=sample_ids,
        safe_prediction=safe,
        prediction=prediction,
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
