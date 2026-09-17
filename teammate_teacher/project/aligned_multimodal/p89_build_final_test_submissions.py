from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import (
    DecoderConfig,
    TransitionModel,
    align_metadata,
    build_sessions,
)
from p46_protocol import HARD_CLASS_IDS
from p88_session_template_decoder import apply_template_posterior, fit_templates
from p88_train_depth_residual import log_softmax_numpy
from p89_detail21_multiexpert_transfer import blend_detail
from p89_deployable_detail21_transfer import feature_matrix
from p89_global_repeat_decoder import GlobalRepeatConfig, decode_global_repeat


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_final_test_predictions_v1"
HARD_CLASSES = np.asarray(HARD_CLASS_IDS, dtype=np.int64)
P46_EXPERT_RUNS = (
    "p46_70_subject_calibrated_v2",
    "p46_dinov2_base_head_v1",
    "p46_validation70_final_v1",
    "p46_videomae_base_large_joint_v1",
    "p46_videomae_depth_head_v1",
    "p46_videomae_head_v1",
    "p46_videomae_ir_depth_head_v1",
    "p46_videomae_large_bagging_v1",
    "p46_videomae_large_head_v1",
    "p46_videomae_large_multiclip_head_v1",
    "p46_videomae_large_weighted_v1",
    "p46_videomae_relation_head_v1",
    "p46_videomae_ssv2_head_v1",
    "p46_videomae_subject_svm_v1",
    "p46_videomae_temporal_head_v2",
    "p46_videomae_thermal_head_v1",
)
P85_HEAD_KEYS = (
    "early_logits",
    "late_logits",
    "window_mean_logits",
    "early_late_logits",
    "temporal_delta_logits",
    "kinetics_logits",
)
P86_KEYS = (
    "baseline_logits",
    "drop_scene_logits",
    "drop_person_logits",
    "drop_workspace_logits",
    "drop_early_logits",
    "drop_late_logits",
    "swap_early_late_logits",
    "collapse_early_late_logits",
    "swap_person_workspace_logits",
    "collapse_view_identity_logits",
)
GLOBAL_REPEAT = GlobalRepeatConfig(
    maximum_session_rank_distance=3,
    maximum_start_gap_seconds=300.0,
    minimum_probability_similarity=0.84,
    minimum_path_overlap=0.2,
    minimum_length_ratio=0.8,
    consensus_weight=0.75,
    alignment_gap_penalty=0.2,
    maximum_group_size=3,
)
TEMPLATE_CONFIGURATION = {
    "prior_weight": 0.0,
    "posterior_temperature": 0.25,
    "strength": 0.05,
    "score_loss_gate": 0.5,
}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_numpy(np.asarray(values, dtype=np.float64)))


def align_values(
    source_ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray
) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    missing = [value for value in target_ids.astype(str) if value not in lookup]
    if missing:
        raise RuntimeError(f"expert misses {len(missing)} rows")
    order = np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    return np.asarray(values)[order]


def conditional_hard(probability40: np.ndarray) -> np.ndarray:
    value = np.asarray(probability40, dtype=np.float64)[:, HARD_CLASSES]
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)


def training_features(reference_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    probabilities: list[np.ndarray] = []
    names: list[str] = []
    for run in P46_EXPERT_RUNS:
        source = load(PROJECT_DIR / "runs" / run / "crossfit_logits.npz")
        logits = align_values(source["sample_ids"], source["logits"], reference_ids)
        probabilities.append(softmax(logits))
        names.append(run)

    teacher = load(PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz")
    values = align_values(
        teacher["oof_sample_ids"], teacher["oof_teacher_log_probability"], reference_ids
    )
    probabilities.append(conditional_hard(softmax(values)))
    names.append("p85_teacher")

    p85 = load(
        PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
    )
    for key in P85_HEAD_KEYS:
        values = align_values(p85["sample_ids"], p85[key], reference_ids)
        probabilities.append(conditional_hard(softmax(values)))
        names.append(f"p85_head_{key}")

    p86 = load(PROJECT_DIR / "runs/p86_teacher_mechanism_audit_v1/fixed_model_predictions.npz")
    for key in P86_KEYS:
        values = align_values(p86["sample_ids"], p86[key], reference_ids)
        probabilities.append(conditional_hard(softmax(values)))
        names.append(f"p86_mechanism_{key}")

    p12 = load(PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz")
    for key in ("skeleton_logits", "thermal_logits"):
        values = align_values(p12["sample_ids"], p12[key], reference_ids)
        probabilities.append(conditional_hard(softmax(values)))
        names.append(f"p12_{key.removesuffix('_logits')}")
    return feature_matrix(probabilities), names


def test_features(test_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    probabilities: list[np.ndarray] = []
    names: list[str] = []
    p46 = load(PROJECT_DIR / "runs/p89_multiexpert_test_logits_v1/p46_test_logits.npz")
    for run in P46_EXPERT_RUNS:
        values = align_values(p46["sample_ids"], p46[run], test_ids)
        probabilities.append(softmax(values))
        names.append(run)

    teacher = load(PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz")
    values = align_values(
        teacher["test_sample_ids"], teacher["test_teacher_log_probability"], test_ids
    )
    probabilities.append(conditional_hard(softmax(values)))
    names.append("p85_teacher")

    p85 = load(
        PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_test_logits.npz"
    )
    for key in P85_HEAD_KEYS:
        values = align_values(p85["sample_ids"], p85[key], test_ids)
        probabilities.append(conditional_hard(softmax(values)))
        names.append(f"p85_head_{key}")

    p86 = load(
        PROJECT_DIR / "runs/p89_multiexpert_test_logits_v1/p86_mechanism_test_logits.npz"
    )
    for key in P86_KEYS:
        stored_key = f"p86_mechanism_{key}"
        values = align_values(p86["sample_ids"], p86[stored_key], test_ids)
        probabilities.append(conditional_hard(softmax(values)))
        names.append(stored_key)

    p12_sd = load(PROJECT_DIR / "runs/p11_final_package/test_logits_sd_fp16.npz")
    values = align_values(p12_sd["sample_ids"], p12_sd["skeleton_logits"], test_ids)
    probabilities.append(conditional_hard(softmax(values)))
    names.append("p12_skeleton")
    p12_thermal = load(PROJECT_DIR / "runs/p11_final_package/test_candidate/test_logits.npz")
    values = align_values(
        p12_thermal["sample_ids"], p12_thermal["thermal_logits"], test_ids
    )
    probabilities.append(conditional_hard(softmax(values)))
    names.append("p12_thermal")
    return feature_matrix(probabilities), names


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_submission(
    path: Path, rows: list[dict[str, str]], prediction: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        for row, value in zip(rows, prediction, strict=True):
            writer.writerow({"path": row["path"], "prediction": int(value)})


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cascade(
    prediction: np.ndarray, all_ids: np.ndarray, detail_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    p46 = load(PROJECT_DIR / "runs/p89_multiexpert_test_logits_v1/p46_test_logits.npz")
    lookup = {value: index for index, value in enumerate(detail_ids.astype(str))}
    positions = np.asarray(
        [index for index, value in enumerate(all_ids.astype(str)) if value in lookup],
        dtype=np.int64,
    )
    detail_positions = np.asarray(
        [lookup[all_ids[index]] for index in positions], dtype=np.int64
    )
    output = prediction.copy()
    audits = {}
    for key, metric, threshold in (
        ("p46_mc_three_clip_kinetics", "margin", 0.98),
        ("p46_mc_late", "confidence", 0.98),
    ):
        probability = softmax(
            align_values(p46["sample_ids"], p46[key], detail_ids)
        )
        ordered = np.sort(probability, axis=1)
        score = (
            ordered[:, -1] - ordered[:, -2]
            if metric == "margin"
            else probability.max(axis=1)
        )
        gate = score[detail_positions] >= threshold
        selected_positions = positions[gate]
        replacement = HARD_CLASSES[probability[detail_positions[gate]].argmax(axis=1)]
        changed = int(np.sum(output[selected_positions] != replacement))
        output[selected_positions] = replacement
        audits[key] = int(gate.sum())
        audits[f"{key}_changed"] = changed
    return output, audits


def histogram(values: np.ndarray) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(map(int, values)).items())}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = load(
        PROJECT_DIR / "runs/p46_validation70_final_v1/crossfit_logits.npz"
    )
    detail_ids = reference["sample_ids"].astype(str)
    detail_labels = reference["labels"].astype(np.int64)
    train_x, train_names = training_features(detail_ids)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    model = LogisticRegression(
        C=0.001,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        tol=2e-4,
    )
    model.fit(scaled_train, detail_labels)

    p46_test = load(
        PROJECT_DIR / "runs/p89_multiexpert_test_logits_v1/p46_test_logits.npz"
    )
    test_detail_ids = p46_test["sample_ids"].astype(str)
    test_x, test_names = test_features(test_detail_ids)
    if train_names != test_names:
        raise RuntimeError("training/Test expert feature order differs")
    partial = model.predict_proba(scaler.transform(test_x))
    detail_probability = np.full(
        (len(test_detail_ids), len(HARD_CLASSES)), 1e-12, dtype=np.float64
    )
    class_to_column = {int(value): index for index, value in enumerate(HARD_CLASSES)}
    for source_column, class_id in enumerate(model.classes_.astype(np.int64)):
        detail_probability[:, class_to_column[int(class_id)]] = partial[:, source_column]
    detail_probability /= detail_probability.sum(axis=1, keepdims=True)
    joblib.dump(
        {"scaler": scaler, "model": model, "expert_names": train_names},
        OUTPUT / "detail21_stacker.joblib",
        compress=3,
    )

    p87 = PROJECT_DIR / "runs/p87s_final_test_predictions_v1"
    audit_rows = read_csv(p87 / "prediction_audit.csv")
    all_ids = np.asarray([row["sample_id"] for row in audit_rows])
    base_logits = np.asarray(np.load(p87 / "student_logits.npy"), dtype=np.float64)
    base_probability = softmax(base_logits)
    blended = blend_detail(
        base_probability,
        all_ids,
        test_detail_ids,
        detail_probability,
        temperature=0.5,
        weight=0.5,
    )

    with np.load(PROJECT_DIR / "runs/p87s_tiny_decoder_v1/tiny_decoder.npz") as saved:
        transition = TransitionModel(
            start_log_probability=saved["start_log_probability"],
            end_log_probability=saved["end_log_probability"],
            bigram_log_probability=saved["bigram_log_probability"],
            trigram_log_probability=saved["trigram_log_probability"],
        )
        decoder = DecoderConfig(
            gap_seconds=float(saved["gap_seconds"]),
            transition_weight=float(saved["transition_weight"]),
            trigram_backoff=float(saved["trigram_backoff"]),
            beam_width=int(saved["beam_width"]),
        )
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv", all_ids
    )
    indices = np.arange(len(all_ids), dtype=np.int64)
    robust_before_cascade, robust_grouping = decode_global_repeat(
        np.log(np.maximum(blended, 1e-12)),
        indices,
        metadata,
        transition,
        decoder,
        GLOBAL_REPEAT,
    )
    robust, robust_cascade = cascade(
        robust_before_cascade, all_ids, test_detail_ids
    )

    teacher = load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    train_ids = teacher["oof_sample_ids"].astype(str)
    train_labels = teacher["oof_labels"].astype(np.int64)
    train_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        train_ids,
    )
    train_indices = np.arange(len(train_ids), dtype=np.int64)
    template_sessions = build_sessions(
        train_indices, train_metadata, decoder.gap_seconds, "known_user"
    )
    templates = fit_templates(train_labels, template_sessions, maximum_length=10)
    test_sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    templated_logp, template_audit = apply_template_posterior(
        np.log(np.maximum(blended, 1e-12)),
        test_sessions,
        templates,
        transition,
        decoder,
        TEMPLATE_CONFIGURATION,
    )
    template_before_cascade, template_grouping = decode_global_repeat(
        templated_logp,
        indices,
        metadata,
        transition,
        decoder,
        GLOBAL_REPEAT,
    )
    template, template_cascade = cascade(
        template_before_cascade, all_ids, test_detail_ids
    )

    source_rows = read_csv(p87 / "submission_p87s_student_decoded.csv")
    if len(source_rows) != len(all_ids):
        raise RuntimeError("P87 source submission row count changed")
    paths = {
        "detail_global": OUTPUT / "submission_p89_detail_global.csv",
        "robust": OUTPUT / "submission_p89_robust.csv",
        "template": OUTPUT / "submission_p89_template.csv",
    }
    write_submission(paths["detail_global"], source_rows, robust_before_cascade)
    write_submission(paths["robust"], source_rows, robust)
    write_submission(paths["template"], source_rows, template)

    p87_prediction = np.asarray(
        [int(row["prediction"]) for row in source_rows], dtype=np.int64
    )
    with (OUTPUT / "prediction_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = (
            "sample_id",
            "p87",
            "p89_detail_global",
            "p89_robust",
            "p89_template",
            "robust_changed",
            "template_changed",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample_id in enumerate(all_ids):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "p87": int(p87_prediction[index]),
                    "p89_detail_global": int(robust_before_cascade[index]),
                    "p89_robust": int(robust[index]),
                    "p89_template": int(template[index]),
                    "robust_changed": int(robust[index] != p87_prediction[index]),
                    "template_changed": int(template[index] != robust[index]),
                }
            )
    np.savez_compressed(
        OUTPUT / "test_probabilities.npz",
        sample_ids=all_ids,
        base_probability=base_probability.astype(np.float32),
        blended_probability=blended.astype(np.float32),
        detail_sample_ids=test_detail_ids,
        detail_probability=detail_probability.astype(np.float32),
        robust_prediction=robust,
        template_prediction=template,
    )
    report = {
        "stage": "P89_final_Test_predictions_v1",
        "status": "complete",
        "protocol": (
            "P87 Test probabilities remain read-only. The Detail21 stacker uses the "
            "H1-selected C=0.001, temperature=0.5 and weight=0.5, refit on all "
            "labeled Detail21 rows. Sequence and confidence thresholds were fixed "
            "before Test inference."
        ),
        "test_rows": int(len(all_ids)),
        "detail_rows": int(len(test_detail_ids)),
        "expert_count": len(train_names),
        "feature_dim": int(train_x.shape[1]),
        "global_repeat": asdict(GLOBAL_REPEAT),
        "robust_grouping": robust_grouping,
        "robust_cascade": robust_cascade,
        "template_configuration": TEMPLATE_CONFIGURATION,
        "template_audit": template_audit,
        "template_grouping": template_grouping,
        "template_cascade": template_cascade,
        "changes": {
            "detail_global_vs_p87": int(np.sum(robust_before_cascade != p87_prediction)),
            "robust_vs_p87": int(np.sum(robust != p87_prediction)),
            "template_vs_p87": int(np.sum(template != p87_prediction)),
            "template_vs_robust": int(np.sum(template != robust)),
        },
        "histograms": {
            "p87": histogram(p87_prediction),
            "robust": histogram(robust),
            "template": histogram(template),
        },
        "submissions": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in paths.items()
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
