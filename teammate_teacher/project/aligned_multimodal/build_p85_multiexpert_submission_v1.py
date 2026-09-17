from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_P12_OOF = PROJECT_DIR / "runs/p12_complete_oof/complete_oof.npz"
DEFAULT_VISUAL_OOF = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)
DEFAULT_VISUAL_TEST = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_test_logits.npz"
)
DEFAULT_SD_TEST = PROJECT_DIR / "runs/p11_final_package/test_logits_sd_fp16.npz"
DEFAULT_IMU_TEST = PROJECT_DIR / "runs/p11_final_package/test_sd_imu_fp16.npz"
DEFAULT_THERMAL_TEST = PROJECT_DIR / "runs/p11_final_package/test_candidate/test_logits.npz"
DEFAULT_TEST_CSV = PROJECT_DIR.parent / "Testing/test.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p85_multiexpert_submission_v1"

P12_EXPERTS = (
    "skeleton",
    "depth",
    "sd",
    "imu",
    "sd_imu",
    "thermal",
    "thermal_candidate",
    "final",
)
VISUAL_EXPERTS = (
    "early",
    "late",
    "window_mean",
    "early_late",
    "temporal_delta",
    "kinetics",
)
EXPERTS = P12_EXPERTS + VISUAL_EXPERTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a low-capacity nonnegative global mixture on subject-disjoint OOF "
            "and build an auditable full-40 Test submission."
        )
    )
    parser.add_argument("--p12-oof", type=Path, default=DEFAULT_P12_OOF)
    parser.add_argument("--visual-oof", type=Path, default=DEFAULT_VISUAL_OOF)
    parser.add_argument("--visual-test", type=Path, default=DEFAULT_VISUAL_TEST)
    parser.add_argument("--sd-test", type=Path, default=DEFAULT_SD_TEST)
    parser.add_argument("--imu-test", type=Path, default=DEFAULT_IMU_TEST)
    parser.add_argument("--thermal-test", type=Path, default=DEFAULT_THERMAL_TEST)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def fit_simplex(log_probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if log_probabilities.ndim != 3 or log_probabilities.shape[2] != 40:
        raise RuntimeError(f"Expected [N,E,40] log probabilities, got {log_probabilities.shape}")

    def objective(theta: np.ndarray) -> float:
        weights = softmax(theta)
        scores = np.sum(log_probabilities * weights[None, :, None], axis=1)
        normalized = scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)
        return float(-np.mean(normalized[np.arange(len(labels)), labels]))

    result = minimize(
        objective,
        np.zeros(log_probabilities.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Simplex optimization failed: {result.message}")
    return softmax(result.x)


def official_id(path_value: str) -> str:
    return path_value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def align(reference: np.ndarray, values: dict[str, np.ndarray]) -> np.ndarray:
    ids = np.asarray(values["sample_ids"]).astype(str)
    lookup = {value: index for index, value in enumerate(ids)}
    if not set(reference).issubset(lookup):
        missing = sorted(set(reference) - set(lookup))[:5]
        raise RuntimeError(f"Sample alignment failed; missing {missing}")
    return np.asarray([lookup[value] for value in reference], dtype=np.int64)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    p12 = load(args.p12_oof)
    visual = load(args.visual_oof)
    sample_ids = np.asarray(visual["sample_ids"]).astype(str)
    labels = np.asarray(visual["labels"], dtype=np.int64)
    folds = np.asarray(visual["folds"], dtype=np.int64)
    p12_order = align(sample_ids, p12)
    if not np.array_equal(labels, np.asarray(p12["labels"], dtype=np.int64)[p12_order]):
        raise RuntimeError("P12 and visual OOF labels disagree")

    oof_logits: dict[str, np.ndarray] = {
        name: np.asarray(p12[f"{name}_logits"], dtype=np.float64)[p12_order]
        for name in P12_EXPERTS
    }
    oof_logits.update(
        {
            name: np.asarray(visual[f"{name}_logits"], dtype=np.float64)
            for name in VISUAL_EXPERTS
        }
    )
    oof_logp = np.stack([log_softmax(oof_logits[name], axis=1) for name in EXPERTS], axis=1)
    crossfit_scores = np.full((len(labels), 40), np.nan, dtype=np.float64)
    fold_weights: dict[str, dict[str, float]] = {}
    for held_fold in (0, 1, 2):
        fit_indices = np.flatnonzero(folds != held_fold)
        held_indices = np.flatnonzero(folds == held_fold)
        weights = fit_simplex(oof_logp[fit_indices], labels[fit_indices])
        crossfit_scores[held_indices] = np.sum(
            oof_logp[held_indices] * weights[None, :, None], axis=1
        )
        fold_weights[str(held_fold)] = {
            name: float(weight) for name, weight in zip(EXPERTS, weights, strict=True)
        }
    if not np.isfinite(crossfit_scores).all():
        raise RuntimeError("Cross-fit mixture did not cover all OOF rows")
    crossfit_prediction = crossfit_scores.argmax(axis=1)
    final_weights = fit_simplex(oof_logp, labels)
    refit_scores = np.sum(oof_logp * final_weights[None, :, None], axis=1)

    sd_test = load(args.sd_test)
    imu_test = load(args.imu_test)
    thermal_test = load(args.thermal_test)
    visual_test = load(args.visual_test)
    test_ids = np.asarray(thermal_test["sample_ids"]).astype(str)
    if len(test_ids) != 405 or len(set(test_ids)) != 405:
        raise RuntimeError("Expected 405 unique P12 Test predictions")
    sd_order = align(test_ids, sd_test)
    imu_order = align(test_ids, imu_test)
    route = np.asarray(thermal_test["route_to_candidate"]).astype(bool)
    routed_logits = np.where(
        route[:, None],
        np.asarray(thermal_test["fixed_candidate_logits"], dtype=np.float64),
        np.asarray(thermal_test["base_logits"], dtype=np.float64),
    )
    if not np.array_equal(routed_logits.argmax(axis=1), thermal_test["routed_predictions"]):
        raise RuntimeError("Reconstructed routed P12 Test logits disagree with saved predictions")
    test_p12_logits = {
        "skeleton": np.asarray(sd_test["skeleton_logits"], dtype=np.float64)[sd_order],
        "depth": np.asarray(sd_test["depth_logits"], dtype=np.float64)[sd_order],
        "sd": np.asarray(imu_test["sd_logits"], dtype=np.float64)[imu_order],
        "imu": np.asarray(imu_test["imu_logits"], dtype=np.float64)[imu_order],
        "sd_imu": np.asarray(imu_test["fused_logits"], dtype=np.float64)[imu_order],
        "thermal": np.asarray(thermal_test["thermal_logits"], dtype=np.float64),
        "thermal_candidate": np.asarray(thermal_test["fixed_candidate_logits"], dtype=np.float64),
        "final": routed_logits,
    }
    visual_ids = np.asarray(visual_test["sample_ids"]).astype(str)
    visual_lookup = {value: index for index, value in enumerate(visual_ids)}
    available = np.asarray([value in visual_lookup for value in test_ids])
    available_indices = np.flatnonzero(available)
    visual_order = np.asarray([visual_lookup[test_ids[index]] for index in available_indices])
    test_logp = np.zeros((len(available_indices), len(EXPERTS), 40), dtype=np.float64)
    for expert_index, name in enumerate(EXPERTS):
        if name in test_p12_logits:
            values = test_p12_logits[name][available_indices]
        else:
            values = np.asarray(visual_test[f"{name}_logits"], dtype=np.float64)[visual_order]
        test_logp[:, expert_index] = log_softmax(values, axis=1)
    fused_scores = np.sum(test_logp * final_weights[None, :, None], axis=1)
    base_prediction = routed_logits.argmax(axis=1)
    prediction = base_prediction.copy()
    prediction[available_indices] = fused_scores.argmax(axis=1)

    with args.test_csv.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        official_rows = list(csv.DictReader(handle))
    official_ids = np.asarray([official_id(row["path"]) for row in official_rows])
    test_lookup = {value: index for index, value in enumerate(test_ids)}
    if len(official_rows) != 405 or set(official_ids) != set(test_lookup):
        raise RuntimeError("Official Test CSV and expert Test predictions do not align")
    official_order = np.asarray([test_lookup[value] for value in official_ids])
    official_prediction = prediction[official_order]
    submission_rows = [
        {"path": row["path"], "prediction": int(value)}
        for row, value in zip(official_rows, official_prediction, strict=True)
    ]
    submission = output / "submission_p85_global_simplex_v1.csv"
    write_csv(submission, submission_rows)
    fusion_path = output / "fusion_logits.npz"
    temporary = fusion_path.with_suffix(fusion_path.suffix + ".building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            oof_sample_ids=sample_ids,
            oof_labels=labels,
            oof_folds=folds,
            oof_crossfit_scores=crossfit_scores.astype(np.float32),
            test_sample_ids=test_ids[available_indices],
            test_fused_scores=fused_scores.astype(np.float32),
            test_all_sample_ids=test_ids,
            test_base_predictions=base_prediction,
            test_visual_available=available.astype(np.uint8),
        )
    temporary.replace(fusion_path)

    summary = {
        "protocol": (
            "Global nonnegative log-probability mixture; weights learned on other "
            "subject folds for OOF evaluation and refit on all OOF rows for Test"
        ),
        "deployment_rule_note": (
            "Public Large VideoMAE is permitted as a distillation teacher, but this "
            "artifact directly uses teacher features at inference and is therefore "
            "not a compliant final deployment. Final inference weights, including "
            "ensembles, must be packaged below 100 MB."
        ),
        "experts": list(EXPERTS),
        "fold_weights": fold_weights,
        "final_weights": {
            name: float(weight) for name, weight in zip(EXPERTS, final_weights, strict=True)
        },
        "crossfit_oof": metrics(labels, crossfit_prediction),
        "same_oof_refit_diagnostic": metrics(labels, refit_scores.argmax(axis=1)),
        "test": {
            "rows": len(test_ids),
            "visual_available": int(available.sum()),
            "visual_missing_p12_fallback": int((~available).sum()),
            "changed_from_p12": int(np.sum(prediction != base_prediction)),
            "predicted_classes": int(len(set(prediction.tolist()))),
            "submission": str(submission),
        },
        "fusion_logits": str(fusion_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
