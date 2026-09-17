from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io
import p89_deterministic_triple_repeat as triple
import p89_full40_scale_invariant_transfer as full40
from p89_imu_rescue_gate import softmax


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_deterministic_triple_stable_v2"
CONFIGURATION = triple.TripleConfig(
    run_policy="exact_three",
    method="shared_probability",
    minimum_similarity=0.0,
    minimum_overlap=0.0,
    consensus_weight=1.0,
    transition_scale=1.0,
)


def test_values():
    with np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    ) as source:
        sample_ids = source["sample_ids"].astype(str)
        probability = np.asarray(source["base_probability"], dtype=np.float64)
    with np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz") as source:
        lookup = {
            sample_id: index
            for index, sample_id in enumerate(source["sample_ids"].astype(str))
        }
        rows = np.asarray([lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)
        imu_probability = softmax(
            np.asarray(source["imu_logits"], dtype=np.float64)[rows], 3.0
        )
    probability = 0.95 * probability + 0.05 * imu_probability
    probability /= probability.sum(axis=1, keepdims=True)
    source_rows = submission_io.read_rows(triple.TEST_SAFE)
    safe = submission_io.read_prediction(triple.TEST_SAFE)
    protocol_value = triple.test_protocol(sample_ids, probability, safe)
    return sample_ids, probability, safe, source_rows, protocol_value


def missing_ir_fallback(sample_ids: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = prediction.copy()
    manifest = triple.read_csv(PROJECT_DIR / "data/p46_test_union_manifest.csv")
    missing = {
        row["official_sample_id"]
        for row in manifest
        if row["p46_ir_readable"] == "0"
    }
    with np.load(
        PROJECT_DIR / "runs/p11_final_package/test_candidate/test_logits.npz"
    ) as source:
        lookup = {
            sample_id: index
            for index, sample_id in enumerate(source["sample_ids"].astype(str))
        }
        fallback = np.asarray(source["routed_predictions"], dtype=np.int64)
    for index, sample_id in enumerate(sample_ids):
        if sample_id in missing:
            result[index] = fallback[lookup[sample_id]]
    return result


def main() -> None:
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    with np.load(triple.H1_SAFE) as source:
        h1_safe = np.asarray(source["h1_prediction"], dtype=np.int64)
        h2_safe = np.asarray(source["h2_prediction"], dtype=np.int64)
    h1_result, h1_prediction = triple.evaluate(
        h1, triple.prepared_probability(h1), h1_safe, CONFIGURATION
    )
    h2_result, h2_prediction = triple.evaluate(
        h2, triple.prepared_probability(h2), h2_safe, CONFIGURATION
    )
    sample_ids, probability, safe, source_rows, protocol_value = test_values()
    prediction, grouping = triple.decode(
        protocol_value, probability, safe, CONFIGURATION
    )
    combined = missing_ir_fallback(sample_ids, prediction)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    direct_path = OUTPUT / "submission_p89_deterministic_triple_stable.csv"
    combined_path = OUTPUT / "submission_p89_triple_stable_missing_ir.csv"
    submission_io.write_submission(direct_path, source_rows, prediction)
    submission_io.write_submission(combined_path, source_rows, combined)
    changes = np.flatnonzero(prediction != safe)
    combined_changes = np.flatnonzero(combined != safe)
    report = {
        "stage": "P89_pre_specified_deterministic_triple_stable_v2",
        "protocol": (
            "A single physics/protocol-motivated configuration: only maximal runs "
            "of exactly three consecutive equal-length sessions, arithmetic posterior "
            "mean, and the original frozen P87 transition weight. No thresholds or "
            "weights are tuned to H2 or Test."
        ),
        "configuration": CONFIGURATION.__dict__,
        "H1": h1_result,
        "H2": h2_result,
        "H1_true_repeat_audit": triple.true_repeat_audit(h1, CONFIGURATION),
        "H2_true_repeat_audit": triple.true_repeat_audit(h2, CONFIGURATION),
        "test": {
            "grouping": grouping,
            "direct_changes_vs_safe": int(len(changes)),
            "direct_changed_ids": sample_ids[changes].tolist(),
            "combined_changes_vs_safe": int(len(combined_changes)),
            "combined_changed_ids": sample_ids[combined_changes].tolist(),
        },
        "submissions": {
            "direct": {
                "path": str(direct_path.resolve()),
                "sha256": submission_io.digest(direct_path),
            },
            "with_missing_ir": {
                "path": str(combined_path.resolve()),
                "sha256": submission_io.digest(combined_path),
            },
        },
    }
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=h1_prediction,
        h2_prediction=h2_prediction,
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
