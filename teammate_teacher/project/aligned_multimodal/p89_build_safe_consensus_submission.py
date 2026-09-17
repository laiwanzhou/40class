from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import p89_build_final_test_submissions_nometa  # noqa: F401
import p89_build_final_test_submissions as deploy
import p89_validate_final_nometa_pipeline as validation
from audit_p87_sequence_decoder import classification_metrics
from p46_protocol import HARD_CLASS_IDS
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE = PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2"
OUTPUT = PROJECT_DIR / "runs/p89_safe_consensus_v1"
HARD_CLASSES = np.asarray(HARD_CLASS_IDS, dtype=np.int64)
EXPERT_COUNT = 33
CLASS_COUNT = 21


RULES = {
    # This restores the deployable Detail21 contract used by the successful P46
    # submission: the 21-way specialist may only reorder a row that the 40-way
    # base already assigned to the Detail21 family.
    "family_gated": {
        "require_p87_hard": True,
        "p46_vote": 0.0,
        "p85_vote": 0.0,
        "probability_delta": -1.0,
    },
    # Two separately trained visual collections must both support the correction.
    # The P46 threshold is a supermajority (10/14 after integer rounding), while
    # the P85 threshold requires all seven heads/teacher to agree in practice.
    "strict": {
        "require_p87_hard": True,
        "p46_vote": 0.70,
        "p85_vote": 0.85,
        "probability_delta": 0.0,
    },
    "relaxed": {
        "require_p87_hard": True,
        "p46_vote": 0.50,
        "p85_vote": 0.70,
        "probability_delta": 0.0,
    },
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_submission(
    path: Path, source_rows: list[dict[str, str]], prediction: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        for row, value in zip(source_rows, prediction, strict=True):
            writer.writerow({"path": row["path"], "prediction": int(value)})


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def expert_probability(features: np.ndarray) -> np.ndarray:
    return features[:, : EXPERT_COUNT * CLASS_COUNT].reshape(
        len(features), EXPERT_COUNT, CLASS_COUNT
    )


def gate_rows(
    sample_ids: np.ndarray,
    base_prediction: np.ndarray,
    detail_prediction: np.ndarray,
    blended_probability: np.ndarray,
    detail_ids: np.ndarray,
    probabilities: np.ndarray,
    rule: dict[str, float | bool],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    lookup = {value: index for index, value in enumerate(detail_ids.astype(str))}
    accepted = np.zeros(len(sample_ids), dtype=bool)
    audit: list[dict[str, object]] = []
    for row in np.flatnonzero(detail_prediction != base_prediction):
        sample_id = str(sample_ids[row])
        family_gate = int(base_prediction[row]) in set(HARD_CLASSES.tolist())
        detail_row = lookup.get(sample_id)
        if detail_row is None:
            continue
        top = HARD_CLASSES[np.argmax(probabilities[detail_row], axis=1)]
        p46_vote = float(np.mean(top[:14] == detail_prediction[row]))
        p85_vote = float(np.mean(top[14:21] == detail_prediction[row]))
        probability_delta = float(
            blended_probability[row, detail_prediction[row]]
            - blended_probability[row, base_prediction[row]]
        )
        keep = (
            (family_gate or not bool(rule["require_p87_hard"]))
            and
            p46_vote >= rule["p46_vote"]
            and p85_vote >= rule["p85_vote"]
            and probability_delta >= rule["probability_delta"]
        )
        accepted[row] = keep
        audit.append(
            {
                "row": int(row),
                "sample_id": sample_id,
                "p87": int(base_prediction[row]),
                "p89_detail": int(detail_prediction[row]),
                "p87_in_detail21": family_gate,
                "p46_vote": p46_vote,
                "p85_vote": p85_vote,
                "probability_delta": probability_delta,
                "accepted": bool(keep),
            }
        )
    return accepted, audit


def validation_result(
    split: str,
    run: str,
    users: list[str],
    stored: np.lib.npyio.NpzFile,
    detail_ids: np.ndarray,
    probabilities: np.ndarray,
    rule: dict[str, float | bool],
) -> dict[str, object]:
    protocol = validation.protocol(run, users)
    sample_ids, labels, _, p87_prediction = protocol[:4]
    blended = stored[f"{split}_probability"]
    detail_prediction, _ = validation.decode(blended, protocol)
    gate, _ = gate_rows(
        sample_ids,
        p87_prediction,
        detail_prediction,
        blended,
        detail_ids,
        probabilities,
        rule,
    )
    prediction = p87_prediction.copy()
    prediction[gate] = detail_prediction[gate]
    per_user = {}
    metadata = protocol[4]
    for user in users:
        rows = metadata.users == user
        per_user[user] = {
            "base_correct": int(np.sum(p87_prediction[rows] == labels[rows])),
            "candidate_correct": int(np.sum(prediction[rows] == labels[rows])),
            "changes": int(np.sum(gate[rows])),
        }
    return {
        "metrics": classification_metrics(labels, prediction),
        "rescue_harm_vs_p87": rescue_harm(labels, p87_prediction, prediction),
        "per_user": per_user,
    }


def main() -> None:
    reference = deploy.load(
        PROJECT_DIR / "runs/p46_validation70_final_v1/crossfit_logits.npz"
    )
    train_ids = reference["sample_ids"].astype(str)
    train_features, train_names = deploy.training_features(train_ids)
    train_probability = expert_probability(train_features)

    test_values = np.load(SOURCE / "test_probabilities.npz")
    sample_ids = test_values["sample_ids"].astype(str)
    detail_ids = test_values["detail_sample_ids"].astype(str)
    test_features, test_names = deploy.test_features(detail_ids)
    if train_names != test_names:
        raise RuntimeError("training/Test expert feature order differs")
    test_probability = expert_probability(test_features)

    p87_dir = PROJECT_DIR / "runs/p87s_final_test_predictions_v1"
    source_rows = read_rows(p87_dir / "submission_p87s_student_decoded.csv")
    p87_prediction = np.asarray(
        [int(row["prediction"]) for row in source_rows], dtype=np.int64
    )
    detail_rows = read_rows(SOURCE / "submission_p89_detail_global.csv")
    detail_prediction = np.asarray(
        [int(row["prediction"]) for row in detail_rows], dtype=np.int64
    )

    validation_values = np.load(
        PROJECT_DIR / "runs/p89_final_nometa_validation_v1/predictions.npz"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "stage": "P89_safe_cross_family_consensus_v1",
        "protocol": (
            "Start from the immutable P87 decoded prediction. Ignore the P89 "
            "cascade and template. Restore the missing deployment family gate: "
            "the Detail21 specialist can only reorder a row whose P87 top-1 is "
            "already in Detail21. Strict variants additionally require the P46 "
            "and P85 expert families to independently support the replacement."
        ),
        "expert_names": train_names,
        "rules": {},
    }
    all_audits: dict[str, list[dict[str, object]]] = {}
    for name, rule in RULES.items():
        gate, audit = gate_rows(
            sample_ids,
            p87_prediction,
            detail_prediction,
            test_values["blended_probability"],
            detail_ids,
            test_probability,
            rule,
        )
        prediction = p87_prediction.copy()
        prediction[gate] = detail_prediction[gate]
        path = OUTPUT / f"submission_p89_safe_consensus_{name}.csv"
        write_submission(path, source_rows, prediction)
        report["rules"][name] = {
            "configuration": rule,
            "H1": validation_result(
                "h1",
                "p87s_fusion_holdout1_c7_structured12_v1",
                validation.H1_USERS,
                validation_values,
                train_ids,
                train_probability,
                rule,
            ),
            "H2": validation_result(
                "h2",
                "p87s_fusion_confirm2_c2_structured12_v1",
                validation.H2_USERS,
                validation_values,
                train_ids,
                train_probability,
                rule,
            ),
            "test_changes_vs_p87": int(gate.sum()),
            "submission": {"path": str(path.resolve()), "sha256": digest(path)},
        }
        all_audits[name] = audit

    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "prediction_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = (
            "rule",
            "row",
            "sample_id",
            "p87",
            "p89_detail",
            "p87_in_detail21",
            "p46_vote",
            "p85_vote",
            "probability_delta",
            "accepted",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rule_name, audit in all_audits.items():
            for row in audit:
                writer.writerow({"rule": rule_name, **row})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
