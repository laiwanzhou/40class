from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_thermal_oof_fusion import align_thermal, fit_temperature, fused_logits
from evaluate_conditional_expert_routing import build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the final OOF-only Thermal calibrator/router, then generate local "
            "fixed-fusion and routed test candidates."
        )
    )
    parser.add_argument("--oof-base", type=Path, required=True)
    parser.add_argument("--oof-thermal-root", type=Path, required=True)
    parser.add_argument("--oof-thermal-name", default="val_logits_fp16.npz")
    parser.add_argument("--test-base", type=Path, required=True)
    parser.add_argument("--test-thermal", type=Path, required=True)
    parser.add_argument("--base-predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def load_thermal_oof(root: Path, filename: str) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {
        "sample_ids": [],
        "labels": [],
        "logits": [],
    }
    for fold in range(3):
        with np.load(root / f"fold_{fold}" / filename, allow_pickle=False) as data:
            values["sample_ids"].append(data["sample_ids"].astype(str))
            values["labels"].append(data["labels"].astype(np.int64))
            values["logits"].append(data["logits"].astype(np.float64))
    return {key: np.concatenate(parts) for key, parts in values.items()}


def align_test_thermal(
    reference_ids: np.ndarray, thermal_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(thermal_path, allow_pickle=False) as data:
        thermal_ids = data["sample_ids"].astype(str)
        thermal_logits = data["logits"].astype(np.float64)
    lookup = {
        sample_id: index for index, sample_id in enumerate(thermal_ids.tolist())
    }
    present = np.asarray(
        [sample_id in lookup for sample_id in reference_ids], dtype=bool
    )
    aligned = np.zeros((len(reference_ids), 40), dtype=np.float64)
    for index, sample_id in enumerate(reference_ids):
        if present[index]:
            aligned[index] = thermal_logits[lookup[sample_id]]
    return present, aligned


def write_submission(
    path: Path, official_paths: list[str], predictions: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "prediction"])
        writer.writerows(zip(official_paths, predictions.astype(int).tolist()))


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.oof_base.resolve(), allow_pickle=False) as data:
        oof_ids = data["sample_ids"].astype(str)
        labels = data["labels"].astype(np.int64)
        base_oof = data["fused_logits"].astype(np.float64)
    thermal_oof = load_thermal_oof(
        args.oof_thermal_root.resolve(), str(args.oof_thermal_name)
    )
    oof_present, aligned_thermal_oof = align_thermal(
        oof_ids, labels, thermal_oof
    )
    base_temperature = fit_temperature(
        base_oof[oof_present], labels[oof_present]
    )
    thermal_temperature = fit_temperature(
        aligned_thermal_oof[oof_present], labels[oof_present]
    )
    weight_rows = []
    selected: tuple[float, float, float] | None = None
    for weight in np.linspace(0.0, 1.0, 21):
        candidate = fused_logits(
            base_oof,
            aligned_thermal_oof,
            oof_present,
            float(weight),
            base_temperature,
            thermal_temperature,
        )
        predictions = candidate.argmax(1)
        accuracy = float(accuracy_score(labels, predictions))
        macro_f1 = float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        )
        weight_rows.append(
            {
                "thermal_weight": float(weight),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )
        score = (accuracy, macro_f1, -float(weight))
        if selected is None or score > selected:
            selected = score
    assert selected is not None
    weight = -selected[2]
    candidate_oof = fused_logits(
        base_oof,
        aligned_thermal_oof,
        oof_present,
        weight,
        base_temperature,
        thermal_temperature,
    )
    base_oof_predictions = base_oof.argmax(1)
    candidate_oof_predictions = candidate_oof.argmax(1)
    base_correct = base_oof_predictions == labels
    candidate_correct = candidate_oof_predictions == labels
    sensitive = oof_present & (base_correct != candidate_correct)
    gate_targets = candidate_correct[sensitive].astype(np.int64)
    if len(np.unique(gate_targets)) != 2:
        raise RuntimeError("Final gate training requires both beneficial and harmful cases.")
    gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.05,
            max_iter=2000,
            solver="lbfgs",
            random_state=int(args.seed),
        ),
    )
    gate.fit(
        build_features(base_oof, candidate_oof, include_class=True)[sensitive],
        gate_targets,
    )
    router_path = output_dir / "thermal_router.joblib"
    joblib.dump(gate, router_path, compress=3)

    with np.load(args.test_base.resolve(), allow_pickle=False) as data:
        test_ids = data["sample_ids"].astype(str)
        base_test = data["fused_logits"].astype(np.float64)
    test_present, aligned_thermal_test = align_test_thermal(
        test_ids, args.test_thermal.resolve()
    )
    candidate_test = fused_logits(
        base_test,
        aligned_thermal_test,
        test_present,
        weight,
        base_temperature,
        thermal_temperature,
    )
    test_features = build_features(
        base_test, candidate_test, include_class=True
    )
    route_probability = np.zeros(len(test_ids), dtype=np.float64)
    route_probability[test_present] = gate.predict_proba(
        test_features[test_present]
    )[:, 1]
    route_to_candidate = test_present & (route_probability >= 0.5)
    base_predictions = base_test.argmax(1)
    candidate_predictions = candidate_test.argmax(1)
    routed_predictions = base_predictions.copy()
    routed_predictions[route_to_candidate] = candidate_predictions[
        route_to_candidate
    ]

    with args.base_predictions_csv.resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        base_rows = list(csv.DictReader(handle))
    official_paths = [row["path"] for row in base_rows]
    if len(official_paths) != len(test_ids):
        raise ValueError("Base prediction CSV row count does not match test logits.")
    expected_ids = np.asarray(
        [Path(path.rstrip("/")).name for path in official_paths]
    )
    if not np.array_equal(expected_ids, test_ids):
        raise ValueError("Base prediction CSV order does not match test sample IDs.")

    fixed_csv = output_dir / "test_predictions_fixed_thermal.csv"
    routed_csv = output_dir / "test_predictions_routed_thermal.csv"
    write_submission(fixed_csv, official_paths, candidate_predictions)
    write_submission(routed_csv, official_paths, routed_predictions)
    probabilities = softmax(candidate_test)
    with (output_dir / "test_predictions_routed_detailed.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "base_prediction",
                "thermal_candidate_prediction",
                "routed_prediction",
                "thermal_present",
                "route_probability",
                "route_to_thermal",
                "candidate_confidence",
                "candidate_entropy",
            ]
        )
        writer.writerows(
            zip(
                test_ids,
                base_predictions,
                candidate_predictions,
                routed_predictions,
                test_present.astype(np.int64),
                route_probability,
                route_to_candidate.astype(np.int64),
                probabilities.max(axis=1),
                entropy(probabilities),
            )
        )
    np.savez_compressed(
        output_dir / "test_logits.npz",
        sample_ids=test_ids,
        base_logits=base_test.astype(np.float32),
        thermal_logits=aligned_thermal_test.astype(np.float32),
        fixed_candidate_logits=candidate_test.astype(np.float32),
        thermal_present=test_present.astype(np.int64),
        route_probability=route_probability.astype(np.float32),
        route_to_candidate=route_to_candidate.astype(np.int64),
        routed_predictions=routed_predictions.astype(np.int64),
    )
    summary = {
        "status": (
            "Local test candidates only. Test labels are unavailable, so no test "
            "accuracy claim is made."
        ),
        "oof_protocol": (
            "Temperatures, Thermal weight and the final low-capacity gate are fit "
            "only from labelled OOF predictions. Previously reported expected "
            "router accuracy remains the subject-disjoint cross-fitted result; "
            "the all-OOF fit metrics below are training diagnostics only."
        ),
        "calibration": {
            "base_temperature": base_temperature,
            "thermal_temperature": thermal_temperature,
            "thermal_weight": weight,
            "weight_scan": weight_rows,
        },
        "gate": {
            "features": 89,
            "regularization_C": 0.05,
            "sensitive_training_samples": int(sensitive.sum()),
            "candidate_beneficial": int(gate_targets.sum()),
            "candidate_harmful": int(len(gate_targets) - gate_targets.sum()),
            "model_path": str(router_path),
            "model_size_mib": router_path.stat().st_size / 2**20,
        },
        "test": {
            "samples": len(test_ids),
            "thermal_present": int(test_present.sum()),
            "thermal_missing_fallback": int((~test_present).sum()),
            "routed_to_thermal": int(route_to_candidate.sum()),
            "route_rate_among_present": float(
                route_to_candidate.sum() / max(1, test_present.sum())
            ),
            "fixed_changes_vs_base": int(
                np.sum(candidate_predictions != base_predictions)
            ),
            "routed_changes_vs_base": int(
                np.sum(routed_predictions != base_predictions)
            ),
            "fixed_predicted_classes": int(
                len(np.unique(candidate_predictions))
            ),
            "routed_predicted_classes": int(
                len(np.unique(routed_predictions))
            ),
            "fixed_class_counts": {
                str(key): int(value)
                for key, value in sorted(Counter(candidate_predictions).items())
            },
            "routed_class_counts": {
                str(key): int(value)
                for key, value in sorted(Counter(routed_predictions).items())
            },
        },
        "outputs": {
            "fixed_csv": str(fixed_csv),
            "routed_csv": str(routed_csv),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
