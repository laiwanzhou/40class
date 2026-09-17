"""Build label-free P150 OOF targets for the deployable P87-S Student.

The generated NPZ files intentionally contain no ground-truth labels. Each
cohort target is built only from P150's prediction for that held cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_P150 = HERE / "runs/p150_repeat_branch_confidence_selector_v1/predictions.npz"
DEFAULT_OUTPUT = HERE / "runs/p162_p150_student_targets_v1"
COHORTS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
CLASSES = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p150-predictions", type=Path, default=DEFAULT_P150)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    return parser.parse_args()


def smoothed_one_hot(prediction: np.ndarray, smoothing: float) -> np.ndarray:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("label smoothing must be in [0, 1)")
    if prediction.ndim != 1 or np.any((prediction < 0) | (prediction >= CLASSES)):
        raise ValueError("P150 prediction is not a valid 40-class vector")
    probability = np.full(
        (len(prediction), CLASSES), smoothing / CLASSES, dtype=np.float32
    )
    probability[np.arange(len(prediction)), prediction] += 1.0 - smoothing
    return probability


def main() -> None:
    args = parse_args()
    source_path = args.p150_predictions.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = np.load(source_path, allow_pickle=False)
    report = {
        "stage": "P162_P150_label_free_student_target_build",
        "status": "complete",
        "source": str(source_path),
        "label_smoothing": float(args.label_smoothing),
        "ground_truth_labels_written": 0,
        "cohorts": {},
    }
    seen: set[str] = set()
    all_ids: list[np.ndarray] = []
    all_probability: list[np.ndarray] = []
    total_rows = 0
    total_teacher_correct = 0
    for cohort in COHORTS:
        sample_ids = source[f"{cohort}_sample_ids"].astype(str)
        prediction = source[f"{cohort}_prediction"].astype(np.int64)
        labels = source[f"{cohort}_labels"].astype(np.int64)
        if len(sample_ids) != len(np.unique(sample_ids)):
            raise RuntimeError(f"duplicate sample IDs in {cohort}")
        overlap = seen.intersection(sample_ids.tolist())
        if overlap:
            raise RuntimeError(f"cohort overlap: {sorted(overlap)[:3]}")
        seen.update(sample_ids.tolist())
        probability = smoothed_one_hot(prediction, args.label_smoothing)
        all_ids.append(sample_ids)
        all_probability.append(probability)
        cohort_dir = output / cohort
        cohort_dir.mkdir(parents=True, exist_ok=True)
        target_path = cohort_dir / "structured_targets.npz"
        np.savez_compressed(
            target_path,
            sample_ids=sample_ids,
            target_mask=np.ones(len(sample_ids), dtype=bool),
            emission_probability=probability,
            structured_distillation_probability=probability,
            structured_confidence=probability.max(axis=1).astype(np.float32),
        )
        correct = int(np.sum(prediction == labels))
        total_rows += len(sample_ids)
        total_teacher_correct += correct
        report["cohorts"][cohort] = {
            "rows": int(len(sample_ids)),
            "teacher_correct_audit_only": correct,
            "teacher_accuracy_audit_only": correct / len(sample_ids),
            "target_path": str(target_path),
        }
    if total_rows != 2470 or total_teacher_correct != 2210:
        raise RuntimeError(
            f"P150 contract changed: rows={total_rows} correct={total_teacher_correct}"
        )
    combined_ids = np.concatenate(all_ids)
    combined_probability = np.concatenate(all_probability)
    combined_path = output / "all_structured_targets.npz"
    np.savez_compressed(
        combined_path,
        sample_ids=combined_ids,
        target_mask=np.ones(len(combined_ids), dtype=bool),
        emission_probability=combined_probability,
        structured_distillation_probability=combined_probability,
        structured_confidence=combined_probability.max(axis=1).astype(np.float32),
    )
    report["aggregate"] = {
        "rows": total_rows,
        "teacher_correct_audit_only": total_teacher_correct,
        "teacher_accuracy_audit_only": total_teacher_correct / total_rows,
        "student_target_correct_for_0.88": int(np.ceil(0.88 * total_rows)),
        "combined_target_path": str(combined_path),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
