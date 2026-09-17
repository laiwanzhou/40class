from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from p46_protocol import HARD_CLASS_IDS, HARD_CLASS_TO_INDEX


PROJECT_DIR = Path(__file__).resolve().parent
GROUPS = {
    "hand_head_food_health": (7, 19, 37, 38, 39),
    "desktop_objects": (8, 9, 10, 11, 14, 15),
    "documents_devices": (18, 19, 20, 21, 22, 24, 25, 26),
    "clothes_cleaning": (13, 16),
    "standalone_35": (35,),
}


class HistoricalReplacementPairSampler:
    """Exact reproducer of the invalid 2026-08-06 sampler for failure audit only.

    This class intentionally samples with replacement. It is isolated from the
    training entry point and must never be used to train a model.
    """

    def __init__(
        self,
        lengths: list[int],
        labels: list[int],
        users: list[str],
        batch_size: int,
        seed: int,
        frame_budget: int,
        bucket_multiplier: int = 12,
    ) -> None:
        self.lengths = lengths
        self.labels = labels
        self.users = users
        self.batch_size = batch_size
        self.seed = seed
        self.frame_budget = frame_budget
        self.bucket_pairs = max(1, bucket_multiplier * batch_size // 2)
        self.epoch = 0
        self.by_class_user: dict[int, dict[str, list[int]]] = {}
        for index, (label, user) in enumerate(zip(labels, users)):
            self.by_class_user.setdefault(label, {}).setdefault(user, []).append(index)
        self.classes = sorted(self.by_class_user)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        pairs_needed = math.ceil(len(self.labels) / 2)
        class_order: list[int] = []
        while len(class_order) < pairs_needed:
            cycle = self.classes.copy()
            generator.shuffle(cycle)
            class_order.extend(cycle)
        pairs: list[tuple[int, int]] = []
        for label in class_order[:pairs_needed]:
            by_user = self.by_class_user[label]
            first_user, second_user = generator.sample(sorted(by_user), 2)
            first = generator.choice(by_user[first_user])
            candidates = by_user[second_user]
            nearest = min(
                abs(self.lengths[value] - self.lengths[first]) for value in candidates
            )
            second = generator.choice(
                [
                    value
                    for value in candidates
                    if abs(self.lengths[value] - self.lengths[first]) == nearest
                ]
            )
            pairs.append((first, second))
        generator.shuffle(pairs)
        batches: list[list[int]] = []
        for start in range(0, len(pairs), self.bucket_pairs):
            bucket = pairs[start : start + self.bucket_pairs]
            bucket.sort(
                key=lambda pair: max(self.lengths[pair[0]], self.lengths[pair[1]])
            )
            chosen: list[tuple[int, int]] = []
            for pair in bucket:
                proposed = chosen + [pair]
                flat = [index for value in proposed for index in value]
                cost = len(flat) * max(self.lengths[index] for index in flat)
                if chosen and (
                    len(flat) > self.batch_size or cost > self.frame_budget
                ):
                    batches.append([index for value in chosen for index in value])
                    chosen = [pair]
                else:
                    chosen = proposed
            if chosen:
                batches.append([index for value in chosen for index in value])
        generator.shuffle(batches)
        yield from batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the stopped P46 Step10 run.")
    parser.add_argument(
        "--run", type=Path, default=PROJECT_DIR / "runs" / "p46_step10_detail21"
    )
    parser.add_argument(
        "--p12", type=Path, default=PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
    )
    parser.add_argument(
        "--trial-summary",
        type=Path,
        default=PROJECT_DIR / "runs" / "p46_event_inputs_full" / "trial_summary.csv",
    )
    parser.add_argument("--epochs-audited", type=int, default=21)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.asarray(HARD_CLASS_IDS),
                average="macro",
                zero_division=0,
            )
        ),
    }


def class_name_map(p12_rows_path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in read_csv(p12_rows_path):
        result[int(row["class_id"])] = row["class_name"]
    return result


def per_class_rows(
    labels: np.ndarray,
    base: np.ndarray,
    p46_accuracy: np.ndarray,
    p46_macro: np.ndarray,
    names: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id in HARD_CLASS_IDS:
        mask = labels == class_id
        support = int(mask.sum())
        base_recall = float((base[mask] == class_id).mean())
        accuracy_recall = float((p46_accuracy[mask] == class_id).mean())
        macro_recall = float((p46_macro[mask] == class_id).mean())
        rows.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "support": support,
                "p12_recall": base_recall,
                "p46_best_accuracy_recall": accuracy_recall,
                "p46_best_macro_recall": macro_recall,
                "best_accuracy_delta_pp": 100.0 * (accuracy_recall - base_recall),
                "best_macro_delta_pp": 100.0 * (macro_recall - base_recall),
            }
        )
    return rows


def confusion_rows(
    labels: np.ndarray, predictions: np.ndarray, names: dict[int, str], limit: int = 30
) -> list[dict[str, Any]]:
    counts = Counter(
        (int(label), int(prediction))
        for label, prediction in zip(labels, predictions)
        if label != prediction
    )
    return [
        {
            "true_class_id": truth,
            "true_class_name": names.get(truth, str(truth)),
            "predicted_class_id": prediction,
            "predicted_class_name": names.get(prediction, str(prediction)),
            "count": count,
        }
        for (truth, prediction), count in counts.most_common(limit)
    ]


def sampler_audit(train_rows: list[dict[str, str]], epochs: int) -> dict[str, Any]:
    lengths = [int(row["frames"]) for row in train_rows]
    labels = [HARD_CLASS_TO_INDEX[int(row["class_id"])] for row in train_rows]
    users = [row["user_id"] for row in train_rows]
    sampler = HistoricalReplacementPairSampler(
        lengths,
        labels,
        users,
        batch_size=16,
        seed=20260806,
        frame_budget=1024,
    )
    epoch_rows: list[dict[str, Any]] = []
    class_unique: dict[int, list[float]] = defaultdict(list)
    cumulative_draws = np.zeros(len(train_rows), dtype=np.int64)
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(1000 + epoch)
        drawn = [index for batch in sampler for index in batch]
        np.add.at(cumulative_draws, np.asarray(drawn), 1)
        unique = set(drawn)
        epoch_rows.append(
            {
                "epoch": epoch,
                "draws": len(drawn),
                "unique_trials": len(unique),
                "duplicate_draws": len(drawn) - len(unique),
                "unique_coverage": len(unique) / len(train_rows),
            }
        )
        for detail_index, class_id in enumerate(HARD_CLASS_IDS):
            available = {index for index, label in enumerate(labels) if label == detail_index}
            class_unique[class_id].append(len(available & unique) / len(available))
    per_class_exposure: dict[str, dict[str, float | int]] = {}
    for detail_index, class_id in enumerate(HARD_CLASS_IDS):
        class_mask = np.asarray(labels) == detail_index
        exposure = cumulative_draws[class_mask]
        per_class_exposure[str(class_id)] = {
            "trials": int(class_mask.sum()),
            "mean_draws_per_trial": float(exposure.mean()),
            "minimum_draws": int(exposure.min()),
            "maximum_draws": int(exposure.max()),
            "never_drawn_trials": int((exposure == 0).sum()),
        }
    return {
        "epoch_rows": epoch_rows,
        "mean_unique_trials": float(np.mean([row["unique_trials"] for row in epoch_rows])),
        "mean_unique_coverage": float(np.mean([row["unique_coverage"] for row in epoch_rows])),
        "minimum_unique_coverage": float(min(row["unique_coverage"] for row in epoch_rows)),
        "maximum_unique_coverage": float(max(row["unique_coverage"] for row in epoch_rows)),
        "cumulative_exposure": {
            "mean_draws_per_trial": float(cumulative_draws.mean()),
            "standard_deviation": float(cumulative_draws.std()),
            "minimum_draws": int(cumulative_draws.min()),
            "maximum_draws": int(cumulative_draws.max()),
            "never_drawn_trials": int((cumulative_draws == 0).sum()),
        },
        "per_class_mean_unique_coverage": {
            str(class_id): float(np.mean(values)) for class_id, values in class_unique.items()
        },
        "per_class_cumulative_exposure": per_class_exposure,
    }


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    trial_rows = read_csv(args.trial_summary.resolve())
    by_source = {row["source_id"]: row for row in trial_rows}
    train_rows = [row for row in trial_rows if row["p46_split"] == "train"]

    accuracy_rows = read_csv(run / "best_accuracy_predictions.csv")
    macro_rows = read_csv(run / "best_macro_f1_predictions.csv")
    if [row["source_id"] for row in accuracy_rows] != [row["source_id"] for row in macro_rows]:
        raise RuntimeError("P46 prediction files do not have identical row order")
    source_ids = [row["source_id"] for row in accuracy_rows]
    sample_ids = [by_source[source_id]["sample_id"] for source_id in source_ids]
    users = np.asarray([row["user_id"] for row in accuracy_rows])
    labels = np.asarray([int(row["true_class_id"]) for row in accuracy_rows])
    p46_accuracy = np.asarray([int(row["predicted_class_id"]) for row in accuracy_rows])
    p46_macro = np.asarray([int(row["predicted_class_id"]) for row in macro_rows])

    with np.load(args.p12.resolve(), allow_pickle=False) as baseline:
        baseline_index = {str(value): index for index, value in enumerate(baseline["sample_ids"])}
        indices = np.asarray([baseline_index[sample_id] for sample_id in sample_ids])
        baseline_labels = baseline["labels"][indices].astype(np.int64)
        restricted_logits = baseline["sd_imu_logits"][indices][:, HARD_CLASS_IDS].astype(np.float32)
    if not np.array_equal(labels, baseline_labels):
        raise RuntimeError("P12/P46 labels do not align")
    p12 = np.asarray(HARD_CLASS_IDS)[restricted_logits.argmax(1)]

    names = class_name_map(args.p12.resolve().with_name("complete_oof_rows.csv"))
    p12_correct = p12 == labels
    p46_correct = p46_accuracy == labels
    outcome = {
        "both_correct": int((p12_correct & p46_correct).sum()),
        "p12_only_correct_new_errors": int((p12_correct & ~p46_correct).sum()),
        "p46_only_correct_rescues": int((~p12_correct & p46_correct).sum()),
        "both_wrong": int((~p12_correct & ~p46_correct).sum()),
    }
    outcome["oracle_either_correct"] = outcome["both_correct"] + outcome["p12_only_correct_new_errors"] + outcome["p46_only_correct_rescues"]
    outcome["oracle_either_accuracy"] = outcome["oracle_either_correct"] / len(labels)

    class_rows = per_class_rows(labels, p12, p46_accuracy, p46_macro, names)
    write_csv(run / "failure_per_class.csv", class_rows)
    write_csv(run / "failure_p46_confusions.csv", confusion_rows(labels, p46_accuracy, names))
    write_csv(run / "failure_p12_confusions.csv", confusion_rows(labels, p12, names))

    subject_rows: list[dict[str, Any]] = []
    for user in sorted(set(users)):
        mask = users == user
        base_accuracy = float((p12[mask] == labels[mask]).mean())
        event_accuracy = float((p46_accuracy[mask] == labels[mask]).mean())
        subject_rows.append(
            {
                "user_id": user,
                "support": int(mask.sum()),
                "p12_accuracy": base_accuracy,
                "p46_accuracy": event_accuracy,
                "delta_pp": 100.0 * (event_accuracy - base_accuracy),
            }
        )
    write_csv(run / "failure_per_subject.csv", subject_rows)

    group_rows: list[dict[str, Any]] = []
    for group, class_ids in GROUPS.items():
        mask = np.isin(labels, class_ids)
        group_rows.append(
            {
                "group": group,
                "class_ids": ",".join(map(str, class_ids)),
                "support": int(mask.sum()),
                "p12_accuracy": float((p12[mask] == labels[mask]).mean()),
                "p46_accuracy": float((p46_accuracy[mask] == labels[mask]).mean()),
                "delta_pp": 100.0
                * (
                    (p46_accuracy[mask] == labels[mask]).mean()
                    - (p12[mask] == labels[mask]).mean()
                ),
            }
        )
    write_csv(run / "failure_per_group.csv", group_rows)

    sampler = sampler_audit(train_rows, args.epochs_audited)
    write_csv(run / "failure_sampler_epochs.csv", sampler.pop("epoch_rows"))
    history = read_csv(run / "stage_b_history.csv")
    best_accuracy_epoch = max(history, key=lambda row: float(row["val_accuracy"]))
    best_macro_epoch = max(history, key=lambda row: float(row["val_macro_f1"]))
    minimum_loss_epoch = min(history, key=lambda row: float(row["val_loss"]))

    summary = {
        "trials": len(labels),
        "p12_restricted": metrics(labels, p12),
        "p46_best_accuracy_checkpoint": metrics(labels, p46_accuracy),
        "p46_best_macro_checkpoint": metrics(labels, p46_macro),
        "p46_best_accuracy_delta_vs_p12_pp": {
            key: 100.0 * (value - metrics(labels, p12)[key])
            for key, value in metrics(labels, p46_accuracy).items()
        },
        "p46_best_macro_delta_vs_p12_pp": {
            key: 100.0 * (value - metrics(labels, p12)[key])
            for key, value in metrics(labels, p46_macro).items()
        },
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_macro_epoch": best_macro_epoch,
        "minimum_validation_loss_epoch": minimum_loss_epoch,
        "train_validation_gap_pp": {
            "best_accuracy_epoch": 100.0
            * (
                float(best_accuracy_epoch["train_sampled_accuracy"])
                - float(best_accuracy_epoch["val_accuracy"])
            ),
            "best_macro_epoch": 100.0
            * (
                float(best_macro_epoch["train_sampled_macro_f1"])
                - float(best_macro_epoch["val_macro_f1"])
            ),
        },
        "correctness_overlap_best_accuracy": outcome,
        "sampler": sampler,
        "zero_recall_classes_best_accuracy": [
            row["class_id"] for row in class_rows if row["p46_best_accuracy_recall"] == 0.0
        ],
        "zero_recall_classes_best_macro": [
            row["class_id"] for row in class_rows if row["p46_best_macro_recall"] == 0.0
        ],
        "classes_p46_accuracy_better_than_p12": [
            row["class_id"] for row in class_rows if row["best_accuracy_delta_pp"] > 0.0
        ],
        "classes_p46_accuracy_worse_than_p12": [
            row["class_id"] for row in class_rows if row["best_accuracy_delta_pp"] < 0.0
        ],
    }
    (run / "failure_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
