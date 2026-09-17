from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import (
    DecoderConfig,
    TransitionModel,
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_sessions,
)
from p88_oof_candidate_ensemble import CANDIDATE_SOURCES, load_candidate, load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_full40_scale_invariant_v1"
H1_USERS = ["user6", "user8", "user17", "user23"]
H2_USERS = ["user5", "user7", "user16", "user18", "user19"]
H1_RUN = "p87s_fusion_holdout1_c7_structured12_v1"
H2_RUN = "p87s_fusion_confirm2_c2_structured12_v1"
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


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_numpy(np.asarray(values, dtype=np.float64)))


def align_values(
    source_ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray
) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    missing = [value for value in target_ids.astype(str) if value not in lookup]
    if missing:
        raise RuntimeError(f"source misses {len(missing)} rows")
    return np.asarray(values)[
        np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    ]


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


def train_probabilities(sample_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    probabilities = []
    names = []
    for name in CANDIDATE_SOURCES:
        probabilities.append(softmax(load_candidate(name, sample_ids)))
        names.append(name)
    return np.stack(probabilities, axis=1).astype(np.float32), names


def test_probabilities(sample_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    probabilities = []
    names = []
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    probabilities.append(
        align_values(
            teacher["test_sample_ids"],
            teacher["test_teacher_probability"],
            sample_ids,
        )
    )
    names.append("p85_teacher")

    p85 = np.load(
        PROJECT_DIR
        / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_test_logits.npz"
    )
    for key in P85_HEAD_KEYS:
        probabilities.append(
            softmax(align_values(p85["sample_ids"], p85[key], sample_ids))
        )
        names.append(f"p85_head_{key}")

    p86 = np.load(
        PROJECT_DIR / "runs/p89_multiexpert_test_logits_v1/p86_mechanism_test_logits.npz"
    )
    for key in P86_KEYS:
        name = f"p86_mechanism_{key}"
        probabilities.append(
            softmax(align_values(p86["sample_ids"], p86[name], sample_ids))
        )
        names.append(name)

    p12 = np.load(PROJECT_DIR / "runs/p11_final_package/test_logits_sd_fp16.npz")
    probabilities.append(
        softmax(
            align_values(p12["sample_ids"], p12["skeleton_logits"], sample_ids)
        )
    )
    names.append("p12_skeleton")
    thermal = np.load(
        PROJECT_DIR / "runs/p11_final_package/test_candidate/test_logits.npz"
    )
    probabilities.append(
        softmax(
            align_values(
                thermal["sample_ids"], thermal["thermal_logits"], sample_ids
            )
        )
    )
    names.append("p12_thermal")
    return np.stack(probabilities, axis=1).astype(np.float32), names


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=2)
    result = np.empty_like(order, dtype=np.float32)
    rank_values = np.arange(values.shape[2], dtype=np.float32) / float(
        values.shape[2] - 1
    )
    rows = np.arange(values.shape[0])[:, None, None]
    experts = np.arange(values.shape[1])[None, :, None]
    result[rows, experts, order] = rank_values
    return result


def feature_matrix(probabilities: np.ndarray, kind: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    logp = np.log(np.maximum(values, 1e-12))
    normalized_logp = (logp - logp.mean(axis=2, keepdims=True)) / np.maximum(
        logp.std(axis=2, keepdims=True), 1e-6
    )
    rank = ranks(values)
    top = np.argmax(values, axis=2)
    votes = np.zeros((len(values), 40), dtype=np.float64)
    for class_id in range(40):
        votes[:, class_id] = np.mean(top == class_id, axis=1)
    aggregate = np.concatenate(
        (
            values.mean(axis=1),
            np.median(values, axis=1),
            values.max(axis=1),
            values.std(axis=1),
            votes,
        ),
        axis=1,
    )
    if kind == "probability":
        output = values.reshape(len(values), -1)
    elif kind == "normalized_logp":
        output = normalized_logp.reshape(len(values), -1)
    elif kind == "rank":
        output = rank.reshape(len(values), -1)
    elif kind == "robust_combined":
        output = np.concatenate(
            (
                normalized_logp.reshape(len(values), -1),
                rank.reshape(len(values), -1),
                aggregate,
            ),
            axis=1,
        )
    else:
        raise ValueError(f"unknown feature kind {kind}")
    return output.astype(np.float32)


def make_model(model_name: str, regularization: float):
    if model_name == "ridge":
        return RidgeClassifier(
            alpha=regularization, class_weight="balanced", solver="lsqr"
        )
    return LogisticRegression(
        C=regularization,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=600,
        tol=2e-4,
    )


def model_probability(model, values: np.ndarray, temperature: float) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probability = np.asarray(model.predict_proba(values), dtype=np.float64)
        probability = softmax(
            np.log(np.maximum(probability, 1e-12)) / temperature
        )
    else:
        probability = softmax(
            np.asarray(model.decision_function(values), dtype=np.float64)
            / temperature
        )
    if np.array_equal(np.asarray(model.classes_, dtype=np.int64), np.arange(40)):
        return probability
    output = np.full((len(values), 40), 1e-12, dtype=np.float64)
    output[:, np.asarray(model.classes_, dtype=np.int64)] = probability
    output /= output.sum(axis=1, keepdims=True)
    return output


def decode(probability: np.ndarray, protocol_value) -> np.ndarray:
    return decode_sessions(
        np.log(np.maximum(probability, 1e-12)),
        protocol_value[6],
        protocol_value[7],
        protocol_value[8],
    )


def evaluate(
    probability: np.ndarray,
    protocol_value,
    base_prediction: np.ndarray,
) -> dict[str, object]:
    prediction = decode(probability, protocol_value)
    return {
        "prediction": prediction,
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], base_prediction, prediction
        ),
    }


def read_submission(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)


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


def main() -> None:
    h1 = protocol(H1_RUN, H1_USERS)
    h2 = protocol(H2_RUN, H2_USERS)
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    all_labels = teacher["oof_labels"].astype(np.int64)
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    all_probability, expert_names = train_probabilities(all_ids)
    id_lookup = {value: index for index, value in enumerate(all_ids)}
    h1_rows = np.asarray([id_lookup[value] for value in h1[0]], dtype=np.int64)
    h2_rows = np.asarray([id_lookup[value] for value in h2[0]], dtype=np.int64)
    h1_fit = ~np.isin(metadata.users, H1_USERS)
    h2_fit = ~np.isin(metadata.users, H2_USERS)
    if np.any(h1_fit[h1_rows]) or np.any(h2_fit[h2_rows]):
        raise RuntimeError("holdout users leaked into full40 router fit")

    base1 = h1[3]
    base2 = h2[3]
    feature_kinds = ("probability", "normalized_logp", "rank", "robust_combined")
    model_configs = [
        *(("ridge", value) for value in (30.0, 100.0, 300.0, 1000.0, 3000.0)),
        *(("logistic", value) for value in (0.0001, 0.0003, 0.001, 0.003)),
    ]
    temperatures = (0.5, 0.75, 1.0, 1.5, 2.0)
    weights = (0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50)
    candidates = []
    best = None
    best_key = None
    cached_features = {
        kind: feature_matrix(all_probability, kind) for kind in feature_kinds
    }
    for kind in feature_kinds:
        features = cached_features[kind]
        for model_name, regularization in model_configs:
            scaler = StandardScaler()
            train_x = scaler.fit_transform(features[h1_fit])
            valid_x = scaler.transform(features[h1_rows])
            model = make_model(model_name, regularization)
            model.fit(train_x, all_labels[h1_fit])
            for temperature in temperatures:
                routed = model_probability(model, valid_x, temperature)
                for weight in weights:
                    blended = (1.0 - weight) * h1[2] + weight * routed
                    blended /= blended.sum(axis=1, keepdims=True)
                    result = evaluate(blended, h1, base1)
                    item = {
                        "feature_kind": kind,
                        "model": model_name,
                        "regularization": regularization,
                        "temperature": temperature,
                        "weight": weight,
                        "router_raw": classification_metrics(
                            h1[1], routed.argmax(axis=1)
                        ),
                        "metrics": result["metrics"],
                        "rescue_harm_vs_p87": result["rescue_harm_vs_p87"],
                    }
                    candidates.append(item)
                    key = (
                        item["metrics"]["correct"],
                        item["metrics"]["balanced_accuracy"],
                        item["rescue_harm_vs_p87"]["net"],
                        -item["rescue_harm_vs_p87"]["harm"],
                        -weight,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best = item
            print(f"finished {kind} {model_name} {regularization:g}", flush=True)
    assert best is not None

    chosen_kind = str(best["feature_kind"])
    chosen_model = str(best["model"])
    chosen_regularization = float(best["regularization"])
    chosen_temperature = float(best["temperature"])
    chosen_weight = float(best["weight"])
    features = cached_features[chosen_kind]
    h2_scaler = StandardScaler()
    h2_train_x = h2_scaler.fit_transform(features[h2_fit])
    h2_valid_x = h2_scaler.transform(features[h2_rows])
    h2_model = make_model(chosen_model, chosen_regularization)
    h2_model.fit(h2_train_x, all_labels[h2_fit])
    h2_routed = model_probability(h2_model, h2_valid_x, chosen_temperature)
    h2_blended = (1.0 - chosen_weight) * h2[2] + chosen_weight * h2_routed
    h2_blended /= h2_blended.sum(axis=1, keepdims=True)
    h2_result = evaluate(h2_blended, h2, base2)

    test_base = np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    )
    test_all_ids = test_base["sample_ids"].astype(str)
    test_detail_ids = test_base["detail_sample_ids"].astype(str)
    test_probability, test_names = test_probabilities(test_detail_ids)
    if test_names != expert_names:
        raise RuntimeError("full40 training/Test expert order differs")
    test_features = feature_matrix(test_probability, chosen_kind)
    final_scaler = StandardScaler()
    final_train_x = final_scaler.fit_transform(features)
    final_test_x = final_scaler.transform(test_features)
    final_model = make_model(chosen_model, chosen_regularization)
    final_model.fit(final_train_x, all_labels)
    test_routed = model_probability(final_model, final_test_x, chosen_temperature)

    full_probability = np.asarray(test_base["base_probability"], dtype=np.float64).copy()
    test_lookup = {value: index for index, value in enumerate(test_all_ids)}
    positions = np.asarray(
        [test_lookup[value] for value in test_detail_ids], dtype=np.int64
    )
    full_probability[positions] = (
        (1.0 - chosen_weight) * full_probability[positions]
        + chosen_weight * test_routed
    )
    full_probability /= full_probability.sum(axis=1, keepdims=True)

    tiny = np.load(PROJECT_DIR / "runs/p87s_tiny_decoder_v1/tiny_decoder.npz")
    transition = TransitionModel(
        start_log_probability=tiny["start_log_probability"],
        end_log_probability=tiny["end_log_probability"],
        bigram_log_probability=tiny["bigram_log_probability"],
        trigram_log_probability=tiny["trigram_log_probability"],
    )
    decoder = DecoderConfig(
        gap_seconds=float(tiny["gap_seconds"]),
        transition_weight=float(tiny["transition_weight"]),
        trigram_backoff=float(tiny["trigram_backoff"]),
        beam_width=int(tiny["beam_width"]),
    )
    test_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        test_all_ids,
    )
    indices = np.arange(len(test_all_ids), dtype=np.int64)
    sessions = build_sessions(
        indices, test_metadata, decoder.gap_seconds, "anonymous_date"
    )
    reproduced = decode_sessions(
        np.log(np.maximum(test_base["base_probability"], 1e-12)),
        sessions,
        transition,
        decoder,
    )
    prediction = decode_sessions(
        np.log(np.maximum(full_probability, 1e-12)),
        sessions,
        transition,
        decoder,
    )
    p87_rows, p87_prediction = read_submission(
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    if not np.array_equal(reproduced, p87_prediction):
        raise RuntimeError("failed to reproduce immutable P87 Test decoder")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_full40_scale_invariant.csv"
    write_submission(submission, p87_rows, prediction)
    joblib.dump(
        {
            "scaler": final_scaler,
            "model": final_model,
            "configuration": best,
            "expert_names": expert_names,
        },
        OUTPUT / "full40_router.joblib",
        compress=3,
    )
    np.savez_compressed(
        OUTPUT / "predictions.npz",
        sample_ids=test_all_ids,
        base_probability=test_base["base_probability"],
        routed_sample_ids=test_detail_ids,
        routed_probability=test_routed.astype(np.float32),
        blended_probability=full_probability.astype(np.float32),
        prediction=prediction,
    )
    report = {
        "stage": "P89_full40_scale_invariant_H1_select_H2_confirm_v1",
        "status": "confirmed" if h2_result["metrics"]["correct"] > int(np.sum(base2 == h2[1])) else "rejected",
        "protocol": (
            "Nineteen full-40-class subject-OOF experts. H1 model fitting excludes "
            "all H1 subjects; feature/model/temperature/weight are selected only "
            "on H1. H2 refits the frozen configuration excluding all H2 subjects. "
            "Test refits on all labeled rows and uses the same feature contract."
        ),
        "expert_names": expert_names,
        "selected": best,
        "H1_base": classification_metrics(h1[1], base1),
        "H2_base": classification_metrics(h2[1], base2),
        "H2": {
            "router_raw": classification_metrics(h2[1], h2_routed.argmax(axis=1)),
            "metrics": h2_result["metrics"],
            "rescue_harm_vs_p87": h2_result["rescue_harm_vs_p87"],
        },
        "feature_dimensions": {
            kind: int(value.shape[1]) for kind, value in cached_features.items()
        },
        "grid_size": len(candidates),
        "test": {
            "rows": int(len(test_all_ids)),
            "routed_rows": int(len(test_detail_ids)),
            "changes_vs_p87": int(np.sum(prediction != p87_prediction)),
            "submission": str(submission.resolve()),
            "sha256": digest(submission),
        },
        "all_H1_candidates": candidates,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "all_H1_candidates"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
