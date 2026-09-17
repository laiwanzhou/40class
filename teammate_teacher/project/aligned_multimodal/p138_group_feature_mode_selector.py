"""Source-only outer selector between two fixed P137 posterior feature modes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_p87_sequence_decoder import classification_metrics


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p138_group_feature_mode_selector_v1"
RUNS = {
    "sqrt": HERE / "runs/p137_group_classifier_selector_c003_w2_v3",
    "sqrt_raw": HERE / "runs/p137_group_classifier_selector_c003_w2_sqrtraw_v7",
}
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")


def main() -> None:
    summaries = {
        name: json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for name, path in RUNS.items()
    }
    predictions = {
        name: np.load(path / "predictions.npz") for name, path in RUNS.items()
    }
    payload = {}
    cohorts = {}
    total_correct = 0
    for held_name in SPLITS:
        candidates = []
        for name in RUNS:
            source = summaries[name]["cohorts"][held_name]["source_threshold"]
            candidates.append(
                {
                    "mode": name,
                    "source_net": int(source["net"]),
                    "source_rescue": int(source["rescue"]),
                    "source_harm": int(source["harm"]),
                    "source_changed": int(source["changed"]),
                    "source_threshold": float(source["threshold"]),
                }
            )
        selected = max(
            candidates,
            key=lambda row: (
                row["source_net"],
                row["source_rescue"],
                -row["source_harm"],
                -row["source_changed"],
            ),
        )
        source = predictions[str(selected["mode"])]
        sample_ids = source[f"{held_name}_sample_ids"].astype(str)
        labels = source[f"{held_name}_labels"].astype(np.int64)
        prediction = source[f"{held_name}_prediction"].astype(np.int64)
        correct = int(np.sum(prediction == labels))
        total_correct += correct
        cohorts[held_name] = {
            "candidates": candidates,
            "selected_mode": selected["mode"],
            "held_metrics": classification_metrics(labels, prediction),
        }
        payload[f"{held_name}_sample_ids"] = sample_ids
        payload[f"{held_name}_labels"] = labels
        payload[f"{held_name}_prediction"] = prediction

    rows = sum(len(payload[f"{name}_labels"]) for name in SPLITS)
    report = {
        "stage": "P138_group_feature_mode_selector_v1",
        "status": "complete_strict_outer_crossfit",
        "protocol": {
            "fixed_modes": list(RUNS),
            "selection_key": "source net, rescue, -harm, -changed",
            "held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": cohorts,
        "aggregate": {
            "rows": rows,
            "correct": total_correct,
            "accuracy": total_correct / rows,
            "target_0.91_correct": int(np.ceil(0.91 * rows)),
            "gap_to_0.91_correct": int(np.ceil(0.91 * rows)) - total_correct,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(OUTPUT / "predictions.npz", **payload)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
