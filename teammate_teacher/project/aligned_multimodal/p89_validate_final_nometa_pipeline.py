from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import p89_build_final_test_submissions as deploy
from audit_p87_sequence_decoder import (
    align_metadata,
    build_sessions,
    classification_metrics,
)
from p46_protocol import HARD_CLASS_IDS
from p88_oof_candidate_ensemble import load_protocol
from p88_session_template_decoder import apply_template_posterior, fit_templates
from p88_train_depth_residual import log_softmax_numpy, rescue_harm
from p89_detail21_multiexpert_transfer import blend_detail, model_probability
from p89_global_repeat_decoder import decode_global_repeat


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_final_nometa_validation_v1"
HARD_CLASSES = np.asarray(HARD_CLASS_IDS, dtype=np.int64)
H1_USERS = ["user6", "user8", "user17", "user23"]
H2_USERS = ["user5", "user7", "user16", "user18", "user19"]
deploy.P46_EXPERT_RUNS = tuple(
    name
    for name in deploy.P46_EXPERT_RUNS
    if name
    not in {"p46_70_subject_calibrated_v2", "p46_validation70_final_v1"}
)


def protocol(run: str, users: list[str]):
    return load_protocol(
        SimpleNamespace(
            base_run=PROJECT_DIR / "runs" / run,
            holdout_users=users,
            teacher_targets=PROJECT_DIR
            / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz",
            train_metadata=PROJECT_DIR
            / "data/p85_recording_metadata/train_recording_metadata.csv",
            repeat_config_summary=PROJECT_DIR
            / "runs/p88_aligned_repeat_h1_v1/summary.json",
        )
    )


def decode(probability: np.ndarray, protocol_value):
    prediction, grouping = decode_global_repeat(
        np.log(np.maximum(probability, 1e-12)),
        protocol_value[5],
        protocol_value[4],
        protocol_value[7],
        protocol_value[8],
        deploy.GLOBAL_REPEAT,
    )
    return prediction, grouping


def cascade(
    prediction: np.ndarray, sample_ids: np.ndarray, detail_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    source = np.load(
        PROJECT_DIR
        / "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz"
    )
    source_ids = source["sample_ids"].astype(str)
    lookup = {value: index for index, value in enumerate(source_ids)}
    positions = np.asarray(
        [index for index, value in enumerate(sample_ids.astype(str)) if value in lookup],
        dtype=np.int64,
    )
    detail_positions = np.asarray(
        [lookup[sample_ids[index]] for index in positions], dtype=np.int64
    )
    output = prediction.copy()
    audit: dict[str, int] = {}
    for key, metric, threshold in (
        ("three_clip_kinetics_logits", "margin", 0.98),
        ("late_logits", "confidence", 0.98),
    ):
        probability = np.exp(log_softmax_numpy(source[key]))
        ordered = np.sort(probability, axis=1)
        score = (
            ordered[:, -1] - ordered[:, -2]
            if metric == "margin"
            else probability.max(axis=1)
        )
        gate = score[detail_positions] >= threshold
        target_positions = positions[gate]
        replacement = HARD_CLASSES[
            probability[detail_positions[gate]].argmax(axis=1)
        ]
        audit[f"{key}_gate"] = int(gate.sum())
        audit[f"{key}_changed"] = int(
            np.sum(output[target_positions] != replacement)
        )
        output[target_positions] = replacement
    return output, audit


def template_probability(
    probability: np.ndarray,
    protocol_value,
    excluded_users: list[str],
) -> tuple[np.ndarray, dict]:
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    all_labels = teacher["oof_labels"].astype(np.int64)
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    fit_indices = np.flatnonzero(~np.isin(metadata.users, excluded_users))
    fit_sessions = build_sessions(
        fit_indices, metadata, protocol_value[8].gap_seconds, "known_user"
    )
    templates = fit_templates(all_labels, fit_sessions, maximum_length=10)
    adjusted, audit = apply_template_posterior(
        np.log(np.maximum(probability, 1e-12)),
        protocol_value[6],
        templates,
        protocol_value[7],
        protocol_value[8],
        deploy.TEMPLATE_CONFIGURATION,
    )
    return np.exp(adjusted), audit


def result(
    labels: np.ndarray,
    base_prediction: np.ndarray,
    prediction: np.ndarray,
    grouping: dict,
    cascade_audit: dict,
) -> dict:
    return {
        "metrics": classification_metrics(labels, prediction),
        "rescue_harm_vs_base": rescue_harm(labels, base_prediction, prediction),
        "grouping": grouping,
        "cascade": cascade_audit,
    }


def main() -> None:
    h1 = protocol("p87s_fusion_holdout1_c7_structured12_v1", H1_USERS)
    h2 = protocol("p87s_fusion_confirm2_c2_structured12_v1", H2_USERS)
    reference = np.load(
        PROJECT_DIR / "runs/p46_validation70_final_v1/crossfit_logits.npz"
    )
    detail_ids = reference["sample_ids"].astype(str)
    detail_labels = reference["labels"].astype(np.int64)
    detail_users = reference["users"].astype(str)
    features, expert_names = deploy.training_features(detail_ids)
    fit_mask = ~np.isin(detail_users, H2_USERS)
    fit_indices = np.flatnonzero(fit_mask)
    crossfit = np.zeros((len(detail_ids), len(HARD_CLASSES)), dtype=np.float64)
    splitter = GroupKFold(n_splits=5)
    for train_relative, valid_relative in splitter.split(
        features[fit_mask], detail_labels[fit_mask], detail_users[fit_mask]
    ):
        train_rows = fit_indices[train_relative]
        valid_rows = fit_indices[valid_relative]
        scaler = StandardScaler()
        train_x = scaler.fit_transform(features[train_rows])
        model = LogisticRegression(
            C=0.001,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            tol=2e-4,
        )
        model.fit(train_x, detail_labels[train_rows])
        crossfit[valid_rows] = model_probability(
            model, scaler.transform(features[valid_rows])
        )
    probability1 = blend_detail(
        h1[2], h1[0], detail_ids, crossfit, temperature=0.5, weight=0.5
    )

    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[fit_mask])
    model = LogisticRegression(
        C=0.001,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        tol=2e-4,
    )
    model.fit(train_x, detail_labels[fit_mask])
    h2_mask = np.isin(detail_users, H2_USERS)
    probability2 = blend_detail(
        h2[2],
        h2[0],
        detail_ids[h2_mask],
        model_probability(model, scaler.transform(features[h2_mask])),
        temperature=0.5,
        weight=0.5,
    )

    report = {
        "stage": "P89_final_nometa_H1_select_H2_confirm_v1",
        "status": "complete",
        "protocol": (
            "33 directly deployable experts. Detail stacker and thresholds are "
            "selected on H1; H2 is evaluated once as confirmation. Templates for "
            "each holdout exclude every user in that holdout."
        ),
        "expert_count": len(expert_names),
        "feature_dim": int(features.shape[1]),
        "global_repeat": asdict(deploy.GLOBAL_REPEAT),
        "template_configuration": deploy.TEMPLATE_CONFIGURATION,
    }
    payload = {}
    for name, protocol_value, probability, excluded in (
        ("H1", h1, probability1, H1_USERS),
        ("H2", h2, probability2, H2_USERS),
    ):
        base_prediction, _ = decode(protocol_value[2], protocol_value)
        detail_prediction, detail_grouping = decode(probability, protocol_value)
        robust_prediction, robust_cascade = cascade(
            detail_prediction, protocol_value[0], detail_ids
        )
        templated_probability, template_audit = template_probability(
            probability, protocol_value, excluded
        )
        template_prediction0, template_grouping = decode(
            templated_probability, protocol_value
        )
        template_prediction, template_cascade = cascade(
            template_prediction0, protocol_value[0], detail_ids
        )
        report[name] = {
            "base": classification_metrics(protocol_value[1], base_prediction),
            "detail_global": result(
                protocol_value[1], base_prediction, detail_prediction, detail_grouping, {}
            ),
            "robust": result(
                protocol_value[1],
                base_prediction,
                robust_prediction,
                detail_grouping,
                robust_cascade,
            ),
            "template": {
                **result(
                    protocol_value[1],
                    base_prediction,
                    template_prediction,
                    template_grouping,
                    template_cascade,
                ),
                "template_audit": template_audit,
            },
        }
        payload[f"{name.lower()}_sample_ids"] = protocol_value[0]
        payload[f"{name.lower()}_labels"] = protocol_value[1]
        payload[f"{name.lower()}_probability"] = probability.astype(np.float32)
        payload[f"{name.lower()}_robust_prediction"] = robust_prediction
        payload[f"{name.lower()}_template_prediction"] = template_prediction

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(OUTPUT / "predictions.npz", **payload)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
