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
from p89_local_visual_neighbor_consistency import (
    TRAIN_FEATURES,
    align_feature_file,
    local_neighbors,
    smooth,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_local_visual_neighbor_h3_audit_v1"
CONFIGURATION = {
    "maximum_gap_seconds": 300.0,
    "neighbors": 2,
    "mutual": False,
    "minimum_similarity": 0.70,
    "weight": 0.10,
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz") as source:
        all_ids = source["oof_sample_ids"].astype(str)
        all_labels = source["oof_labels"].astype(np.int64)
        folds = source["oof_folds"].astype(np.int64)
        all_probability = source["oof_teacher_probability"].astype(np.float64)
    metadata_all = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv", all_ids
    )
    sequence_summary = json.loads(
        (PROJECT_DIR / "runs/p87_sequence_decoder_v1/summary.json").read_text(encoding="utf-8")
    )
    fold_summary = next(item for item in sequence_summary["folds"] if int(item["fold"]) == 0)
    decoder = DecoderConfig(**fold_summary["selected"])
    fit = np.flatnonzero(folds != 0)
    transition = fit_transition_model(
        all_labels,
        build_sessions(fit, metadata_all, decoder.gap_seconds, "known_user"),
        40,
        decoder.trigram_backoff,
    )
    validation_rows = np.flatnonzero(folds == 0)
    sample_ids = all_ids[validation_rows]
    labels = all_labels[validation_rows]
    probability = all_probability[validation_rows].copy()
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv", sample_ids
    )
    with np.load(PROJECT_DIR / "runs/p87_sequence_decoder_v1/oof_predictions.npz") as source:
        if not np.array_equal(source["sample_ids"].astype(str), all_ids):
            raise RuntimeError("P87 OOF alignment changed")
        p87 = source["sequence_predictions"].astype(np.int64)[validation_rows]
    with np.load(PROJECT_DIR / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz") as source:
        imu_probability, imu_present = aligned_imu(
            source["sample_ids"].astype(str),
            np.asarray(source["imu_logits"], dtype=np.float64),
            sample_ids,
            3.0,
        )
    probability[imu_present] = (
        0.95 * probability[imu_present] + 0.05 * imu_probability[imu_present]
    )
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
    features, present = align_feature_file(TRAIN_FEATURES, sample_ids)
    neighbors, _, graph = local_neighbors(
        features,
        present,
        metadata,
        CONFIGURATION["maximum_gap_seconds"],
        CONFIGURATION["neighbors"],
        CONFIGURATION["mutual"],
        CONFIGURATION["minimum_similarity"],
    )
    adjusted = smooth(probability, neighbors, CONFIGURATION["weight"])
    prediction = joint_decode(
        adjusted,
        p87,
        protocol,
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
        initial_prediction=safe,
        grouping_prediction=p87,
    )[0]
    users = metadata.users.astype(str)
    per_user = {
        user: int(
            np.sum(prediction[users == user] == labels[users == user])
            - np.sum(safe[users == user] == labels[users == user])
        )
        for user in sorted(set(users.tolist()))
    }
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        sample_ids=sample_ids,
        safe_prediction=safe,
        prediction=prediction,
    )
    report = {
        "stage": "P89_local_visual_neighbor_H3_fold0_audit_v1",
        "protocol": (
            "The local-neighbor configuration was frozen after H1/H2 stress analysis. "
            "H3 is the untouched original subject fold 0 (six users) and uses its own "
            "strict nested decoder/transition fit. The H3 base is the P85 teacher rather "
            "than P87-S, so this is an independent structural confirmation, not a direct score estimate."
        ),
        "configuration": CONFIGURATION,
        "users": sorted(set(users.tolist())),
        "safe_metrics": classification_metrics(labels, safe),
        "candidate_metrics": classification_metrics(labels, prediction),
        "rescue_harm_vs_safe": rescue_harm(labels, safe, prediction),
        "per_user_gain": per_user,
        "minimum_user_gain": int(min(per_user.values())),
        "graph": graph,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
