from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from analyze_local_depth_oof_fusion import SMALL_ACTION_IDS, late_fuse_cross_fitted
from local_roi_data import sample_positions


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_BASELINE_SHARED = (
    PROJECT_DIR / "runs" / "p16_shared_full_local_oracle_oof" / "oof_logits.npz"
)
DEFAULT_CANDIDATE = (
    PROJECT_DIR / "runs" / "p18_peak_sampling_shared_oof" / "oof_logits.npz"
)
DEFAULT_PEAK_CACHE = PROJECT_DIR / "runs" / "p18_peak_local_depth_cache"
DEFAULT_HARD = PROJECT_DIR / "data" / "hard_local_v1" / "hard_action_protocol.json"
DEFAULT_TAXONOMY = (
    PROJECT_DIR / "data" / "six_modality_audit" / "small_action_taxonomy_v1.csv"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p18_peak_sampling_audit"
TARGET_PAIRS = ((9, 10), (8, 9), (10, 11), (21, 22), (17, 18), (20, 39))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered gate to the peak-sampling OOF experiment"
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--baseline-shared", type=Path, default=DEFAULT_BASELINE_SHARED)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--peak-cache", type=Path, default=DEFAULT_PEAK_CACHE)
    parser.add_argument("--hard", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def subset_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    class_ids: np.ndarray,
) -> dict[str, float | int]:
    mask = np.isin(labels, class_ids)
    return {"samples": int(mask.sum()), **metrics(labels[mask], logits[mask])}


def align_candidate(
    base_ids: np.ndarray,
    base_labels: np.ndarray,
    base_folds: np.ndarray,
    candidate: np.lib.npyio.NpzFile,
) -> dict[str, np.ndarray]:
    candidate_ids = candidate["sample_ids"].astype(str)
    location = {sample_id: index for index, sample_id in enumerate(candidate_ids)}
    if set(location) != set(base_ids.astype(str)):
        raise ValueError("Candidate OOF sample coverage differs from baseline")
    order = np.asarray([location[value] for value in base_ids.astype(str)], dtype=np.int64)
    if not np.array_equal(candidate["labels"][order].astype(int), base_labels):
        raise ValueError("Candidate labels differ")
    if not np.array_equal(candidate["held_fold"][order].astype(int), base_folds):
        raise ValueError("Candidate folds differ")
    return {
        key: candidate[key][order].astype(np.float64)
        for key in ("fused_logits", "full_aux_logits", "local_aux_logits")
    }


def ensure_candidate_oof(candidate_path: Path) -> None:
    if candidate_path.exists():
        return
    run_dir = candidate_path.parent
    arrays = [
        np.load(
            run_dir / f"fold_{fold}" / "best_val_logits.npz",
            allow_pickle=False,
        )
        for fold in range(3)
    ]
    combined = {
        key: np.concatenate([array[key] for array in arrays])
        for key in (
            "sample_ids",
            "labels",
            "held_fold",
            "fused_logits",
            "full_aux_logits",
            "local_aux_logits",
        )
    }
    if len(combined["sample_ids"]) != 2914:
        raise ValueError("Expected 2914 combined validation samples")
    if len(np.unique(combined["sample_ids"].astype(str))) != 2914:
        raise ValueError("Combined validation sample IDs are not unique")
    order = np.argsort(combined["sample_ids"].astype(str))
    np.savez_compressed(
        candidate_path,
        **{key: value[order] for key, value in combined.items()},
    )
    training_summary = {
        "status": "assembled_after_fold2_worker_recovery",
        "samples": 2914,
        "folds": [
            json.loads(
                (run_dir / f"fold_{fold}" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            for fold in range(3)
        ],
        "pooled_visual": metrics(
            combined["labels"].astype(int),
            combined["fused_logits"].astype(np.float64),
        ),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def pair_errors(
    labels: np.ndarray,
    predictions: np.ndarray,
    left: int,
    right: int,
) -> tuple[int, int, int]:
    left_to_right = int(np.sum((labels == left) & (predictions == right)))
    right_to_left = int(np.sum((labels == right) & (predictions == left)))
    return left_to_right, right_to_left, left_to_right + right_to_left


def write_matrix(
    output: Path,
    labels: np.ndarray,
    predictions: np.ndarray,
    rows: list[int],
    class_names: dict[int, str],
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(40))
    frame = pd.DataFrame(
        matrix[rows],
        columns=[f"pred_{index:02d}_{class_names[index]}" for index in range(40)],
    )
    frame.insert(0, "true_samples", matrix[rows].sum(axis=1))
    frame.insert(0, "true_action", [class_names[index] for index in rows])
    frame.insert(0, "true_class_id", rows)
    frame.to_csv(output, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = np.load(args.base.resolve(), allow_pickle=False)
    baseline_shared = np.load(args.baseline_shared.resolve(), allow_pickle=False)
    candidate_path = args.candidate.resolve()
    ensure_candidate_oof(candidate_path)
    candidate_file = np.load(candidate_path, allow_pickle=False)
    sample_ids = base["sample_ids"].astype(str)
    labels = base["labels"].astype(int)
    folds = base["folds"].astype(int)
    for source, name, fold_key in [
        (baseline_shared, "baseline shared", "held_fold"),
        (candidate_file, "candidate", "held_fold"),
    ]:
        if len(np.unique(source["sample_ids"].astype(str))) != len(sample_ids):
            raise ValueError(f"{name} has duplicate sample IDs")
        if set(source["sample_ids"].astype(str)) != set(sample_ids):
            raise ValueError(f"{name} sample coverage differs")
        if fold_key not in source:
            raise ValueError(f"{name} lacks held fold")

    baseline_aligned = align_candidate(sample_ids, labels, folds, baseline_shared)
    candidate_aligned = align_candidate(sample_ids, labels, folds, candidate_file)
    skeleton = base["skeleton_logits"].astype(np.float64)
    baseline_current, baseline_protocol = late_fuse_cross_fitted(
        labels,
        folds,
        skeleton,
        baseline_aligned["fused_logits"],
    )
    candidate_current, candidate_protocol = late_fuse_cross_fitted(
        labels,
        folds,
        skeleton,
        candidate_aligned["fused_logits"],
    )
    baseline_pred = baseline_current.argmax(1)
    candidate_pred = candidate_current.argmax(1)

    taxonomy = pd.read_csv(args.taxonomy.resolve())
    class_names = {
        int(row.class_id): str(row.action_name)
        for row in taxonomy[["class_id", "action_name"]].itertuples(index=False)
    }
    small_ids = [int(value) for value in SMALL_ACTION_IDS.tolist()]
    hard_ids = [
        int(value)
        for value in json.loads(args.hard.resolve().read_text(encoding="utf-8"))[
            "hard_class_ids"
        ]
    ]
    small = np.asarray(small_ids, dtype=int)
    hard = np.asarray(hard_ids, dtype=int)

    baseline_metrics = {
        "all": metrics(labels, baseline_current),
        "small": subset_metrics(labels, baseline_current, small),
        "hard": subset_metrics(labels, baseline_current, hard),
    }
    candidate_metrics = {
        "all": metrics(labels, candidate_current),
        "small": subset_metrics(labels, candidate_current, small),
        "hard": subset_metrics(labels, candidate_current, hard),
    }
    deltas = {
        subset: {
            key + "_pp": 100.0 * (
                float(candidate_metrics[subset][key])
                - float(baseline_metrics[subset][key])
            )
            for key in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        for subset in ("all", "small", "hard")
    }
    fold_rows: list[dict[str, object]] = []
    for fold in range(3):
        mask = folds == fold
        baseline_accuracy = float(np.mean(baseline_pred[mask] == labels[mask]))
        candidate_accuracy = float(np.mean(candidate_pred[mask] == labels[mask]))
        fold_rows.append(
            {
                "fold": fold,
                "samples": int(mask.sum()),
                "baseline_accuracy": baseline_accuracy,
                "candidate_accuracy": candidate_accuracy,
                "delta_pp": 100.0 * (candidate_accuracy - baseline_accuracy),
            }
        )
    positive_folds = sum(float(row["delta_pp"]) > 0.0 for row in fold_rows)

    pair_rows: list[dict[str, object]] = []
    baseline_target_errors = 0
    candidate_target_errors = 0
    for left, right in TARGET_PAIRS:
        baseline_directions = pair_errors(labels, baseline_pred, left, right)
        candidate_directions = pair_errors(labels, candidate_pred, left, right)
        baseline_target_errors += baseline_directions[2]
        candidate_target_errors += candidate_directions[2]
        pair_rows.append(
            {
                "class_a": left,
                "action_a": class_names[left],
                "class_b": right,
                "action_b": class_names[right],
                "baseline_a_to_b": baseline_directions[0],
                "baseline_b_to_a": baseline_directions[1],
                "baseline_bidirectional": baseline_directions[2],
                "candidate_a_to_b": candidate_directions[0],
                "candidate_b_to_a": candidate_directions[1],
                "candidate_bidirectional": candidate_directions[2],
                "error_reduction": baseline_directions[2] - candidate_directions[2],
                "error_reduction_percent": (
                    100.0
                    * (baseline_directions[2] - candidate_directions[2])
                    / baseline_directions[2]
                    if baseline_directions[2]
                    else 0.0
                ),
            }
        )
    target_pair_reduction_percent = (
        100.0
        * (baseline_target_errors - candidate_target_errors)
        / baseline_target_errors
    )
    pd.DataFrame(pair_rows).to_csv(
        output_dir / "target_pair_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline_matrix = confusion_matrix(labels, baseline_pred, labels=np.arange(40))
    candidate_matrix = confusion_matrix(labels, candidate_pred, labels=np.arange(40))
    union_ids = set(small_ids) | set(hard_ids)
    ranked_pairs: list[tuple[int, int, int]] = []
    for left in range(40):
        for right in range(left + 1, 40):
            if left not in union_ids and right not in union_ids:
                continue
            count = int(baseline_matrix[left, right] + baseline_matrix[right, left])
            ranked_pairs.append((count, left, right))
    ranked_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    top20_rows: list[dict[str, object]] = []
    for rank, (baseline_count, left, right) in enumerate(ranked_pairs[:20], start=1):
        pair_mask = np.isin(labels, [left, right])
        baseline_pair_accuracy = float(
            np.mean(baseline_pred[pair_mask] == labels[pair_mask])
        )
        candidate_pair_accuracy = float(
            np.mean(candidate_pred[pair_mask] == labels[pair_mask])
        )
        candidate_count = int(
            candidate_matrix[left, right] + candidate_matrix[right, left]
        )
        top20_rows.append(
            {
                "rank": rank,
                "class_a": left,
                "action_a": class_names[left],
                "class_b": right,
                "action_b": class_names[right],
                "pair_samples": int(pair_mask.sum()),
                "baseline_a_to_b": int(baseline_matrix[left, right]),
                "baseline_b_to_a": int(baseline_matrix[right, left]),
                "baseline_bidirectional": baseline_count,
                "candidate_a_to_b": int(candidate_matrix[left, right]),
                "candidate_b_to_a": int(candidate_matrix[right, left]),
                "candidate_bidirectional": candidate_count,
                "error_reduction": baseline_count - candidate_count,
                "baseline_pair_accuracy": baseline_pair_accuracy,
                "candidate_pair_accuracy": candidate_pair_accuracy,
                "pair_accuracy_delta_pp": 100.0
                * (candidate_pair_accuracy - baseline_pair_accuracy),
            }
        )
    pd.DataFrame(top20_rows).to_csv(
        output_dir / "baseline_top20_pair_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_sample_rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        baseline_correct = bool(baseline_pred[index] == labels[index])
        candidate_correct = bool(candidate_pred[index] == labels[index])
        subject = re.search(r"__user(\d+)__", sample_id)
        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "fold": int(folds[index]),
                "subject": f"user{subject.group(1)}" if subject else "",
                "class_id": int(labels[index]),
                "action_name": class_names[int(labels[index])],
                "is_small": int(labels[index] in small_ids),
                "is_hard": int(labels[index] in hard_ids),
                "baseline_prediction": int(baseline_pred[index]),
                "candidate_prediction": int(candidate_pred[index]),
                "baseline_correct": int(baseline_correct),
                "candidate_correct": int(candidate_correct),
                "outcome": (
                    "win"
                    if (not baseline_correct and candidate_correct)
                    else "loss"
                    if (baseline_correct and not candidate_correct)
                    else "both_correct"
                    if baseline_correct
                    else "both_wrong"
                ),
            }
        )
    pd.DataFrame(per_sample_rows).to_csv(
        output_dir / "per_sample_win_loss.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_class_rows: list[dict[str, object]] = []
    for class_id in range(40):
        mask = labels == class_id
        baseline_recall = float(np.mean(baseline_pred[mask] == labels[mask]))
        candidate_recall = float(np.mean(candidate_pred[mask] == labels[mask]))
        per_class_rows.append(
            {
                "class_id": class_id,
                "action_name": class_names[class_id],
                "samples": int(mask.sum()),
                "is_small": int(class_id in small_ids),
                "is_hard": int(class_id in hard_ids),
                "baseline_recall": baseline_recall,
                "candidate_recall": candidate_recall,
                "delta_pp": 100.0 * (candidate_recall - baseline_recall),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / "per_class_recall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    subjects = np.asarray(
        [
            int(re.search(r"__user(\d+)__", sample_id).group(1))
            for sample_id in sample_ids
        ]
    )
    subject_rows: list[dict[str, object]] = []
    for subject in sorted(np.unique(subjects).tolist()):
        subject_mask = subjects == subject
        row: dict[str, object] = {"subject": f"user{subject}"}
        for subset_name, subset_ids in [
            ("all", np.arange(40)),
            ("small", small),
            ("hard", hard),
        ]:
            mask = subject_mask & np.isin(labels, subset_ids)
            baseline_accuracy = float(np.mean(baseline_pred[mask] == labels[mask]))
            candidate_accuracy = float(np.mean(candidate_pred[mask] == labels[mask]))
            row[f"{subset_name}_n"] = int(mask.sum())
            row[f"{subset_name}_baseline_accuracy"] = baseline_accuracy
            row[f"{subset_name}_candidate_accuracy"] = candidate_accuracy
            row[f"{subset_name}_delta_pp"] = 100.0 * (
                candidate_accuracy - baseline_accuracy
            )
        subject_rows.append(row)
    pd.DataFrame(subject_rows).to_csv(
        output_dir / "per_subject_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_matrix(
        output_dir / "small_actions_confusion_21x40.csv",
        labels,
        candidate_pred,
        small_ids,
        class_names,
    )
    write_matrix(
        output_dir / "hard_classes_confusion_21x40.csv",
        labels,
        candidate_pred,
        hard_ids,
        class_names,
    )

    peak_cache = args.peak_cache.resolve()
    peak_ids = np.load(peak_cache / "sample_ids.npy", allow_pickle=False).astype(str)
    peak_location = {sample_id: index for index, sample_id in enumerate(peak_ids)}
    uniform = np.load(peak_cache / "uniform_positions.npy", allow_pickle=False)
    peaks = np.load(peak_cache / "peak_positions.npy", allow_pickle=False)
    lengths = np.load(peak_cache / "lengths.npy", allow_pickle=False)
    energy_offsets = np.load(
        peak_cache / "transition_offsets.npy",
        allow_pickle=False,
    )
    energy_flat = np.load(
        peak_cache / "transition_energy.npy",
        allow_pickle=False,
    )
    frame_rows: list[dict[str, object]] = []
    uniform_exact = 0
    peak_exact = 0
    uniform_within_one = 0
    peak_within_one = 0
    for sample_id in sample_ids:
        index = peak_location[sample_id]
        offset, count = energy_offsets[index].astype(int).tolist()
        energy = energy_flat[offset : offset + count]
        max_transition_position = int(np.argmax(energy) + 1) if len(energy) else 0
        uniform_12 = sample_positions(int(lengths[index]), 12, augment=False)
        selected = sorted(
            uniform[index].astype(int).tolist()
            + peaks[index].astype(int).tolist()
        )
        uniform_exact += int(max_transition_position in uniform_12)
        peak_exact += int(max_transition_position in selected)
        uniform_within_one += int(
            min(abs(max_transition_position - value) for value in uniform_12) <= 1
        )
        peak_within_one += int(
            min(abs(max_transition_position - value) for value in selected) <= 1
        )
        row: dict[str, object] = {
            "sample_id": sample_id,
            "length": int(lengths[index]),
            "uniform_positions": " ".join(
                str(value) for value in uniform[index].astype(int).tolist()
            ),
            "peak_positions": " ".join(
                str(value) for value in peaks[index].astype(int).tolist()
            ),
            "selected_validation_positions": " ".join(
                str(value) for value in selected
            ),
            "max_motion_transition_position": max_transition_position,
            "max_motion_energy": float(energy.max()) if len(energy) else 0.0,
        }
        for peak_index, position in enumerate(peaks[index].astype(int).tolist()):
            row[f"peak_{peak_index}_position"] = position
            row[f"peak_{peak_index}_energy"] = (
                float(energy[position - 1])
                if position >= 1 and position - 1 < len(energy)
                else 0.0
            )
        frame_rows.append(row)
    pd.DataFrame(frame_rows).to_csv(
        output_dir / "selected_frame_positions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    criteria = {
        "small_accuracy_delta_at_least_1_5pp": bool(
            deltas["small"]["accuracy_pp"] >= 1.5
        ),
        "hard_accuracy_delta_at_least_1_0pp": bool(
            deltas["hard"]["accuracy_pp"] >= 1.0
        ),
        "at_least_two_positive_folds": bool(positive_folds >= 2),
        "target_pair_errors_down_at_least_10_percent": bool(
            target_pair_reduction_percent >= 10.0
        ),
        "overall_accuracy_not_down_more_than_0_3pp": bool(
            deltas["all"]["accuracy_pp"] >= -0.3
        ),
    }
    summary = {
        "status": "complete",
        "experiment": "8_uniform_plus_4_local_depth_motion_peaks",
        "preregistered_gate_passed": bool(all(criteria.values())),
        "criteria": criteria,
        "baseline_current_system": baseline_metrics,
        "candidate_current_system": candidate_metrics,
        "delta": deltas,
        "folds": fold_rows,
        "positive_folds": positive_folds,
        "target_pairs": {
            "pairs": [list(pair) for pair in TARGET_PAIRS],
            "baseline_bidirectional_errors": baseline_target_errors,
            "candidate_bidirectional_errors": candidate_target_errors,
            "error_reduction_percent": target_pair_reduction_percent,
        },
        "sampling_peak_coverage": {
            "uniform12_exact_max_motion": uniform_exact / len(sample_ids),
            "peak12_exact_max_motion": peak_exact / len(sample_ids),
            "uniform12_within_one": uniform_within_one / len(sample_ids),
            "peak12_within_one": peak_within_one / len(sample_ids),
        },
        "cross_fitted_protocol": {
            "baseline": baseline_protocol,
            "candidate": candidate_protocol,
        },
        "win_loss": {
            outcome: sum(row["outcome"] == outcome for row in per_sample_rows)
            for outcome in ("win", "loss", "both_correct", "both_wrong")
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
