from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from analyze_local_depth_oof_fusion import SMALL_ACTION_IDS, late_fuse_cross_fitted
from analyze_peak_sampling_experiment import (
    align_candidate,
    metrics,
    pair_errors,
    subset_metrics,
    write_matrix,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_BASELINE_SHARED = (
    PROJECT_DIR / "runs" / "p16_shared_full_local_oracle_oof" / "oof_logits.npz"
)
DEFAULT_CANDIDATE = (
    PROJECT_DIR
    / "runs"
    / "p19_local_depth_difference_shared_oof"
    / "oof_logits.npz"
)
DEFAULT_HARD = PROJECT_DIR / "data" / "hard_local_v1" / "hard_action_protocol.json"
DEFAULT_TAXONOMY = (
    PROJECT_DIR / "data" / "six_modality_audit" / "small_action_taxonomy_v1.csv"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p19_local_depth_difference_audit"
DEFAULT_GATE = DEFAULT_OUTPUT / "preregistered_gate.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the fixed-12 Local Depth difference OOF experiment"
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--baseline-shared", type=Path, default=DEFAULT_BASELINE_SHARED)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--hard", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def assemble_candidate_oof(candidate_path: Path) -> None:
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
        raise ValueError("Combined validation IDs are not unique")
    order = np.argsort(combined["sample_ids"].astype(str))
    np.savez_compressed(
        candidate_path,
        **{key: value[order] for key, value in combined.items()},
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = json.loads(args.gate.resolve().read_text(encoding="utf-8"))
    if not bool(gate["frozen_before_training"]):
        raise ValueError("Gate is not marked frozen")

    base = np.load(args.base.resolve(), allow_pickle=False)
    baseline_shared_file = np.load(
        args.baseline_shared.resolve(),
        allow_pickle=False,
    )
    candidate_path = args.candidate.resolve()
    assemble_candidate_oof(candidate_path)
    candidate_file = np.load(candidate_path, allow_pickle=False)
    sample_ids = base["sample_ids"].astype(str)
    labels = base["labels"].astype(int)
    folds = base["folds"].astype(int)
    baseline_shared = align_candidate(
        sample_ids,
        labels,
        folds,
        baseline_shared_file,
    )
    candidate_shared = align_candidate(
        sample_ids,
        labels,
        folds,
        candidate_file,
    )
    skeleton = base["skeleton_logits"].astype(np.float64)
    baseline_current, baseline_protocol = late_fuse_cross_fitted(
        labels,
        folds,
        skeleton,
        baseline_shared["fused_logits"],
    )
    candidate_current, candidate_protocol = late_fuse_cross_fitted(
        labels,
        folds,
        skeleton,
        candidate_shared["fused_logits"],
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
    non_small = np.asarray(
        [class_id for class_id in range(40) if class_id not in set(small_ids)],
        dtype=int,
    )
    target_pairs = [
        (int(pair[0]), int(pair[1])) for pair in gate["target_pairs"]
    ]

    baseline_metrics = {
        "all": metrics(labels, baseline_current),
        "small": subset_metrics(labels, baseline_current, small),
        "hard": subset_metrics(labels, baseline_current, hard),
        "non_small": subset_metrics(labels, baseline_current, non_small),
        "shared_visual": metrics(labels, baseline_shared["fused_logits"]),
    }
    candidate_metrics = {
        "all": metrics(labels, candidate_current),
        "small": subset_metrics(labels, candidate_current, small),
        "hard": subset_metrics(labels, candidate_current, hard),
        "non_small": subset_metrics(labels, candidate_current, non_small),
        "shared_visual": metrics(labels, candidate_shared["fused_logits"]),
    }
    deltas: dict[str, dict[str, float]] = {}
    for subset in ("all", "small", "hard", "non_small", "shared_visual"):
        deltas[subset] = {
            key + "_pp": 100.0
            * (
                float(candidate_metrics[subset][key])
                - float(baseline_metrics[subset][key])
            )
            for key in ("accuracy", "balanced_accuracy", "macro_f1")
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
    improved_target_pairs = 0
    for left, right in target_pairs:
        baseline_directions = pair_errors(labels, baseline_pred, left, right)
        candidate_directions = pair_errors(labels, candidate_pred, left, right)
        improved = candidate_directions[2] < baseline_directions[2]
        improved_target_pairs += int(improved)
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
                "improved": int(improved),
            }
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
        candidate_count = int(
            candidate_matrix[left, right] + candidate_matrix[right, left]
        )
        baseline_accuracy = float(
            np.mean(baseline_pred[pair_mask] == labels[pair_mask])
        )
        candidate_accuracy = float(
            np.mean(candidate_pred[pair_mask] == labels[pair_mask])
        )
        top20_rows.append(
            {
                "rank": rank,
                "class_a": left,
                "action_a": class_names[left],
                "class_b": right,
                "action_b": class_names[right],
                "pair_samples": int(pair_mask.sum()),
                "baseline_bidirectional": baseline_count,
                "candidate_bidirectional": candidate_count,
                "error_reduction": baseline_count - candidate_count,
                "baseline_pair_accuracy": baseline_accuracy,
                "candidate_pair_accuracy": candidate_accuracy,
                "pair_accuracy_delta_pp": 100.0
                * (candidate_accuracy - baseline_accuracy),
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
        subject_match = re.search(r"__user(\d+)__", sample_id)
        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "fold": int(folds[index]),
                "subject": (
                    f"user{subject_match.group(1)}" if subject_match else ""
                ),
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

    thresholds = gate["criteria"]
    criteria = {
        "small_action_accuracy_delta": bool(
            deltas["small"]["accuracy_pp"]
            >= float(thresholds["small_action_accuracy_delta_pp_min"])
        ),
        "hard_class_accuracy_delta": bool(
            deltas["hard"]["accuracy_pp"]
            >= float(thresholds["hard_class_accuracy_delta_pp_min"])
        ),
        "positive_fold_count": bool(
            positive_folds >= int(thresholds["positive_fold_count_min"])
        ),
        "improved_target_pair_count": bool(
            improved_target_pairs
            >= int(thresholds["improved_target_pair_count_min"])
        ),
        "non_small_balanced_accuracy_delta": bool(
            deltas["non_small"]["balanced_accuracy_pp"]
            >= float(thresholds["non_small_balanced_accuracy_delta_pp_min"])
        ),
        "overall_accuracy_delta": bool(
            deltas["all"]["accuracy_pp"]
            >= float(thresholds["overall_accuracy_delta_pp_min"])
        ),
    }
    summary = {
        "status": "complete",
        "experiment": "fixed12_local_decoded_depth_difference",
        "preregistered_gate": gate,
        "preregistered_gate_passed": bool(all(criteria.values())),
        "criteria_passed": criteria,
        "baseline_current_system": baseline_metrics,
        "candidate_current_system": candidate_metrics,
        "delta": deltas,
        "folds": fold_rows,
        "positive_folds": positive_folds,
        "improved_target_pairs": improved_target_pairs,
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
