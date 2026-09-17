from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import p89_build_evidence_rollback as detail
import p89_build_final_test_submissions_nometa  # noqa: F401
import p89_build_final_test_submissions as detail_deploy
import p89_full40_consensus_gate as full_gate
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_dual_consensus_v1"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_prediction(path: Path) -> np.ndarray:
    return np.asarray(
        [int(row["prediction"]) for row in read_rows(path)], dtype=np.int64
    )


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


def merge(
    base: np.ndarray, detail_prediction: np.ndarray, full_prediction: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    detail_changed = detail_prediction != base
    full_changed = full_prediction != base
    conflict = detail_changed & full_changed & (detail_prediction != full_prediction)
    output = base.copy()
    output[detail_changed & ~conflict] = detail_prediction[detail_changed & ~conflict]
    output[full_changed & ~conflict] = full_prediction[full_changed & ~conflict]
    return output, {
        "detail_changes": int(detail_changed.sum()),
        "full40_changes": int(full_changed.sum()),
        "agreement_overlap": int(
            np.sum(detail_changed & full_changed & ~conflict)
        ),
        "conflicts_reverted_to_p87": int(conflict.sum()),
        "union_changes": int(np.sum(output != base)),
    }


def component_predictions(
    protocol_value,
    split: str,
    holdout_users: list[str],
    all_ids: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    detail_ids: np.ndarray,
    detail_probability: np.ndarray,
    full_probability: np.ndarray,
    full_features: np.ndarray,
    configuration: dict[str, object],
    stored_detail: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {value: index for index, value in enumerate(all_ids.astype(str))}
    rows = np.asarray(
        [lookup[value] for value in protocol_value[0].astype(str)], dtype=np.int64
    )

    detail_candidate = stored_detail[f"{split}_template_prediction"]
    detail_gate, _ = detail.evidence_gate(
        protocol_value[0],
        protocol_value[3],
        detail_candidate,
        stored_detail[f"{split}_probability"],
        detail_ids,
        detail_probability,
    )
    detail_output = protocol_value[3].copy()
    detail_output[detail_gate] = detail_candidate[detail_gate]

    fit = ~np.isin(users, holdout_users)
    scaler = StandardScaler()
    model = full40.make_model(
        str(configuration["model"]), float(configuration["regularization"])
    )
    model.fit(scaler.fit_transform(full_features[fit]), labels[fit])
    router_probability = full40.model_probability(
        model,
        scaler.transform(full_features[rows]),
        float(configuration["temperature"]),
    )
    weight = float(configuration["weight"])
    blended = (1.0 - weight) * protocol_value[2] + weight * router_probability
    blended /= blended.sum(axis=1, keepdims=True)
    full_candidate = full40.decode(blended, protocol_value)
    full_accepted, _ = full_gate.gate(
        protocol_value[0],
        protocol_value[3],
        full_candidate,
        router_probability,
        full_probability[rows],
    )
    full_output = protocol_value[3].copy()
    full_output[full_accepted] = full_candidate[full_accepted]
    return detail_output, full_output


def validate(
    protocol_value,
    split: str,
    holdout_users: list[str],
    all_ids: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    detail_ids: np.ndarray,
    detail_probability: np.ndarray,
    full_probability: np.ndarray,
    full_features: np.ndarray,
    configuration: dict[str, object],
    stored_detail: np.lib.npyio.NpzFile,
) -> dict[str, object]:
    detail_output, full_output = component_predictions(
        protocol_value,
        split,
        holdout_users,
        all_ids,
        labels,
        users,
        detail_ids,
        detail_probability,
        full_probability,
        full_features,
        configuration,
        stored_detail,
    )
    output, merge_audit = merge(
        protocol_value[3], detail_output, full_output
    )
    per_user = {}
    for user in holdout_users:
        rows = protocol_value[4].users == user
        per_user[user] = {
            "p87_correct": int(
                np.sum(protocol_value[3][rows] == protocol_value[1][rows])
            ),
            "candidate_correct": int(
                np.sum(output[rows] == protocol_value[1][rows])
            ),
            "changes": int(np.sum(output[rows] != protocol_value[3][rows])),
        }
    return {
        "metrics": classification_metrics(protocol_value[1], output),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], output
        ),
        "merge": merge_audit,
        "per_user": per_user,
    }


def main() -> None:
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    labels = teacher["oof_labels"].astype(np.int64)
    metadata = full40.align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    detail_reference = detail_deploy.load(
        PROJECT_DIR / "runs/p46_validation70_final_v1/crossfit_logits.npz"
    )
    detail_ids = detail_reference["sample_ids"].astype(str)
    detail_features, _ = detail_deploy.training_features(detail_ids)
    detail_probability = detail.expert_probability(detail_features)
    full_probability, expert_names = full40.train_probabilities(all_ids)
    source_summary = json.loads(
        (
            PROJECT_DIR
            / "runs/p89_full40_scale_invariant_v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    configuration = source_summary["selected"]
    full_features = full40.feature_matrix(
        full_probability, str(configuration["feature_kind"])
    )
    stored_detail = np.load(
        PROJECT_DIR / "runs/p89_final_nometa_validation_v1/predictions.npz"
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_result = validate(
        h1,
        "h1",
        full40.H1_USERS,
        all_ids,
        labels,
        metadata.users,
        detail_ids,
        detail_probability,
        full_probability,
        full_features,
        configuration,
        stored_detail,
    )
    h2_result = validate(
        h2,
        "h2",
        full40.H2_USERS,
        all_ids,
        labels,
        metadata.users,
        detail_ids,
        detail_probability,
        full_probability,
        full_features,
        configuration,
        stored_detail,
    )

    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = read_rows(p87_path)
    p87 = read_prediction(p87_path)
    detail_test = read_prediction(
        PROJECT_DIR
        / "runs/p89_evidence_rollback_v1/submission_p89_evidence_rollback.csv"
    )
    full_test = read_prediction(
        PROJECT_DIR
        / "runs/p89_full40_consensus_gate_v1/submission_p89_full40_consensus.csv"
    )
    prediction, test_merge = merge(p87, detail_test, full_test)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_dual_consensus.csv"
    write_submission(submission, source_rows, prediction)
    report = {
        "stage": "P89_dual_consensus_Detail21_plus_full40_v1",
        "protocol": (
            "Union of two independently validated, deployable correction gates. "
            "Detail21 requires family membership plus P46/P85 consensus. Full40 "
            "requires 14/19 expert votes plus router delta. Agreement is accepted, "
            "and any cross-component conflict reverts to immutable P87."
        ),
        "H1": h1_result,
        "H2": h2_result,
        "Test": {
            "merge": test_merge,
            "path": str(submission.resolve()),
            "sha256": digest(submission),
        },
        "full40_expert_names": expert_names,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
