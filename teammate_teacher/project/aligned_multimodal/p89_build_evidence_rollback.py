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
P87_DIR = PROJECT_DIR / "runs/p87s_final_test_predictions_v1"
P89_DIR = PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2"
OUTPUT = PROJECT_DIR / "runs/p89_evidence_rollback_v1"
HARD_CLASSES = np.asarray(HARD_CLASS_IDS, dtype=np.int64)
EXPERT_COUNT = 33
CLASS_COUNT = 21
P46_MINIMUM_VOTES = 5
P85_MINIMUM_VOTES = 5
PRE_SCORE_EQUIVALENT_SHA256 = (
    "2308e4cd85fa440d65e3e13691f39f4e6d7a58791320ede15fbedbe5af355109"
)


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


def expert_probability(features: np.ndarray) -> np.ndarray:
    return features[:, : EXPERT_COUNT * CLASS_COUNT].reshape(
        len(features), EXPERT_COUNT, CLASS_COUNT
    )


def evidence_gate(
    sample_ids: np.ndarray,
    p87: np.ndarray,
    candidate: np.ndarray,
    blended_probability: np.ndarray,
    detail_ids: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    lookup = {value: index for index, value in enumerate(detail_ids.astype(str))}
    accepted = np.zeros(len(sample_ids), dtype=bool)
    audit = []
    for row in np.flatnonzero((candidate != p87) & np.isin(p87, HARD_CLASSES)):
        detail_row = lookup.get(str(sample_ids[row]))
        if detail_row is None:
            continue
        top = HARD_CLASSES[np.argmax(probabilities[detail_row], axis=1)]
        p46_votes = int(np.sum(top[:14] == candidate[row]))
        p85_votes = int(np.sum(top[14:21] == candidate[row]))
        probability_delta = float(
            blended_probability[row, candidate[row]]
            - blended_probability[row, p87[row]]
        )
        keep = (
            p46_votes >= P46_MINIMUM_VOTES
            and p85_votes >= P85_MINIMUM_VOTES
            and probability_delta >= 0.0
        )
        accepted[row] = keep
        audit.append(
            {
                "row": int(row),
                "sample_id": str(sample_ids[row]),
                "p87": int(p87[row]),
                "candidate": int(candidate[row]),
                "p46_votes": p46_votes,
                "p85_votes": p85_votes,
                "probability_delta": probability_delta,
                "accepted": bool(keep),
            }
        )
    return accepted, audit


def validate(
    split: str,
    run: str,
    users: list[str],
    stored: np.lib.npyio.NpzFile,
    detail_ids: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    protocol = validation.protocol(run, users)
    sample_ids, labels, _, p87 = protocol[:4]
    candidate = stored[f"{split}_template_prediction"]
    gate, _ = evidence_gate(
        sample_ids,
        p87,
        candidate,
        stored[f"{split}_probability"],
        detail_ids,
        probabilities,
    )
    output = p87.copy()
    output[gate] = candidate[gate]
    return {
        "metrics": classification_metrics(labels, output),
        "rescue_harm_vs_p87": rescue_harm(labels, p87, output),
        "changes": int(gate.sum()),
    }


def main() -> None:
    reference = deploy.load(
        PROJECT_DIR / "runs/p46_validation70_final_v1/crossfit_logits.npz"
    )
    train_ids = reference["sample_ids"].astype(str)
    train_features, train_names = deploy.training_features(train_ids)
    train_probability = expert_probability(train_features)

    test_values = np.load(P89_DIR / "test_probabilities.npz")
    test_ids = test_values["sample_ids"].astype(str)
    detail_ids = test_values["detail_sample_ids"].astype(str)
    test_features, test_names = deploy.test_features(detail_ids)
    if train_names != test_names:
        raise RuntimeError("training/Test expert feature order differs")
    test_probability = expert_probability(test_features)

    source_rows = read_rows(P87_DIR / "submission_p87s_student_decoded.csv")
    p87 = np.asarray([int(row["prediction"]) for row in source_rows], dtype=np.int64)
    candidate = read_prediction(P89_DIR / "submission_p89_template.csv")
    gate, audit = evidence_gate(
        test_ids,
        p87,
        candidate,
        test_values["blended_probability"],
        detail_ids,
        test_probability,
    )
    output = p87.copy()
    output[gate] = candidate[gate]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "submission_p89_evidence_rollback.csv"
    write_submission(path, source_rows, output)
    output_hash = digest(path)
    if output_hash != PRE_SCORE_EQUIVALENT_SHA256:
        raise RuntimeError("evidence rollback is no longer prediction-equivalent")

    stored = np.load(
        PROJECT_DIR / "runs/p89_final_nometa_validation_v1/predictions.npz"
    )
    report = {
        "stage": "P89_family_and_cross_expert_evidence_rollback_v1",
        "protocol": (
            "Start from immutable P87. Require P87 top-1 inside Detail21, at "
            "least 5/14 P46 votes, at least 5/7 P85 votes, and positive P89 "
            "row-level probability delta. The resulting Test prediction is "
            "byte-identical to the four-change strict candidate generated "
            "before the family-gated leaderboard score was observed."
        ),
        "configuration": {
            "p46_minimum_votes": P46_MINIMUM_VOTES,
            "p85_minimum_votes": P85_MINIMUM_VOTES,
            "minimum_probability_delta": 0.0,
        },
        "H1": validate(
            "h1",
            "p87s_fusion_holdout1_c7_structured12_v1",
            validation.H1_USERS,
            stored,
            train_ids,
            train_probability,
        ),
        "H2": validate(
            "h2",
            "p87s_fusion_confirm2_c2_structured12_v1",
            validation.H2_USERS,
            stored,
            train_ids,
            train_probability,
        ),
        "Test": {
            "changes_vs_p87": int(gate.sum()),
            "path": str(path.resolve()),
            "sha256": output_hash,
            "prediction_equivalent_pre_score_sha256": PRE_SCORE_EQUIVALENT_SHA256,
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
            "p46_votes",
            "p85_votes",
            "probability_delta",
            "accepted",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
