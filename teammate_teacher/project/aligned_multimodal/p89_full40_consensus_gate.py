from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE = PROJECT_DIR / "runs/p89_full40_scale_invariant_v1"
OUTPUT = PROJECT_DIR / "runs/p89_full40_consensus_gate_v1"
MINIMUM_EXPERT_VOTES = 14
MINIMUM_ROUTER_DELTA = 0.10


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


def gate(
    sample_ids: np.ndarray,
    base_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    router_probability: np.ndarray,
    expert_probability: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    accepted = np.zeros(len(sample_ids), dtype=bool)
    audit = []
    expert_top = np.argmax(expert_probability, axis=2)
    for row in np.flatnonzero(candidate_prediction != base_prediction):
        votes = int(np.sum(expert_top[row] == candidate_prediction[row]))
        router_delta = float(
            router_probability[row, candidate_prediction[row]]
            - router_probability[row, base_prediction[row]]
        )
        keep = votes >= MINIMUM_EXPERT_VOTES and router_delta >= MINIMUM_ROUTER_DELTA
        accepted[row] = keep
        audit.append(
            {
                "row": int(row),
                "sample_id": str(sample_ids[row]),
                "p87": int(base_prediction[row]),
                "candidate": int(candidate_prediction[row]),
                "expert_votes": votes,
                "router_delta": router_delta,
                "accepted": bool(keep),
            }
        )
    return accepted, audit


def fit_holdout(
    protocol_value,
    holdout_users: list[str],
    all_ids: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    features: np.ndarray,
    expert_probability: np.ndarray,
    configuration: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    lookup = {value: index for index, value in enumerate(all_ids.astype(str))}
    rows = np.asarray(
        [lookup[value] for value in protocol_value[0].astype(str)], dtype=np.int64
    )
    fit = ~np.isin(users, holdout_users)
    scaler = StandardScaler()
    model = full40.make_model(
        str(configuration["model"]), float(configuration["regularization"])
    )
    model.fit(scaler.fit_transform(features[fit]), labels[fit])
    router_probability = full40.model_probability(
        model,
        scaler.transform(features[rows]),
        float(configuration["temperature"]),
    )
    weight = float(configuration["weight"])
    blended = (1.0 - weight) * protocol_value[2] + weight * router_probability
    blended /= blended.sum(axis=1, keepdims=True)
    candidate = full40.decode(blended, protocol_value)
    accepted, audit = gate(
        protocol_value[0],
        protocol_value[3],
        candidate,
        router_probability,
        expert_probability[rows],
    )
    prediction = protocol_value[3].copy()
    prediction[accepted] = candidate[accepted]
    per_user = {}
    for user in holdout_users:
        selected = protocol_value[4].users == user
        per_user[user] = {
            "p87_correct": int(
                np.sum(protocol_value[3][selected] == protocol_value[1][selected])
            ),
            "candidate_correct": int(
                np.sum(prediction[selected] == protocol_value[1][selected])
            ),
            "changes": int(np.sum(accepted[selected])),
        }
    return (
        {
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_p87": rescue_harm(
                protocol_value[1], protocol_value[3], prediction
            ),
            "changes": int(accepted.sum()),
            "per_user": per_user,
        },
        audit,
    )


def main() -> None:
    source_summary = json.loads(
        (SOURCE / "summary.json").read_text(encoding="utf-8")
    )
    configuration = source_summary["selected"]
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    labels = teacher["oof_labels"].astype(np.int64)
    metadata = full40.align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    train_probability, expert_names = full40.train_probabilities(all_ids)
    features = full40.feature_matrix(
        train_probability, str(configuration["feature_kind"])
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_result, h1_audit = fit_holdout(
        h1,
        full40.H1_USERS,
        all_ids,
        labels,
        metadata.users,
        features,
        train_probability,
        configuration,
    )
    h2_result, h2_audit = fit_holdout(
        h2,
        full40.H2_USERS,
        all_ids,
        labels,
        metadata.users,
        features,
        train_probability,
        configuration,
    )

    test = np.load(SOURCE / "predictions.npz")
    test_ids = test["sample_ids"].astype(str)
    routed_ids = test["routed_sample_ids"].astype(str)
    routed_probability = test["routed_probability"]
    candidate = test["prediction"].astype(np.int64)
    test_expert_probability, test_names = full40.test_probabilities(routed_ids)
    if test_names != expert_names:
        raise RuntimeError("full40 expert order differs")
    routed_lookup = {value: index for index, value in enumerate(routed_ids)}
    all_positions = np.asarray(
        [index for index, value in enumerate(test_ids) if value in routed_lookup],
        dtype=np.int64,
    )
    routed_positions = np.asarray(
        [routed_lookup[test_ids[index]] for index in all_positions], dtype=np.int64
    )
    p87_path = (
        PROJECT_DIR
        / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    source_rows = read_rows(p87_path)
    p87 = read_prediction(p87_path)
    partial_gate, partial_audit = gate(
        test_ids[all_positions],
        p87[all_positions],
        candidate[all_positions],
        routed_probability[routed_positions],
        test_expert_probability[routed_positions],
    )
    accepted = np.zeros(len(test_ids), dtype=bool)
    accepted[all_positions] = partial_gate
    prediction = p87.copy()
    prediction[accepted] = candidate[accepted]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p89_full40_consensus.csv"
    write_submission(submission, source_rows, prediction)
    report = {
        "stage": "P89_full40_consensus_gate_v1",
        "protocol": (
            "The full40 model/configuration is selected on H1 and confirmed on "
            "H2. A correction is deployed only when at least 14/19 full40 "
            "experts vote for it and the routed posterior exceeds the P87 class "
            "by at least 0.10. The gate uses no labels or class-family oracle."
        ),
        "source_configuration": configuration,
        "gate": {
            "minimum_expert_votes": MINIMUM_EXPERT_VOTES,
            "minimum_router_delta": MINIMUM_ROUTER_DELTA,
        },
        "H1": h1_result,
        "H2": h2_result,
        "Test": {
            "changes_vs_p87": int(accepted.sum()),
            "path": str(submission.resolve()),
            "sha256": digest(submission),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "prediction_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = (
            "row",
            "sample_id",
            "p87",
            "candidate",
            "expert_votes",
            "router_delta",
            "accepted",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in partial_audit:
            full_row = int(all_positions[int(row["row"])])
            writer.writerow({**row, "row": full_row})
    np.savez_compressed(
        OUTPUT / "validation_audits.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
