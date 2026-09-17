from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OOF = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "hard_local_v1"
HARD_RECALL_THRESHOLD = 0.60
HARD_MASS_THRESHOLD = 0.50

# The seed graph was formed from symmetric OOF confusion counts >= 5.
# Closely related components were then expanded with their direct decision
# boundary classes. Overlap is intentional: one action may sit on two boundaries.
CONFUSION_GROUPS: dict[str, tuple[int, ...]] = {
    "G1_hand_to_face_food_health": (2, 6, 7, 19, 23, 37, 38, 39),
    "G2_tabletop_object_manipulation": (4, 8, 9, 10, 11, 14, 15),
    "G3_desk_document_device": (17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
    "G4_floor_cleaning": (12, 13),
    "G5_clothes_handling": (3, 5, 16, 31),
    "G6_lower_body_posture": (28, 29, 32, 34, 35, 36),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze hard classes, confusion groups, and Local trigger features"
    )
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def class_names() -> dict[int, str]:
    result: dict[int, str] = {}
    for path in sorted((PROJECT_DIR / "data" / "subject_folds").glob("fold_*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                result[int(row["class_id"])] = row["class_name"]
    if len(result) != 40:
        raise ValueError(f"Expected 40 class names, got {len(result)}")
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_names_for_class(class_id: int) -> list[str]:
    return [
        name for name, members in CONFUSION_GROUPS.items() if class_id in members
    ]


def shared_groups(first: int, second: int) -> list[str]:
    return [
        name
        for name, members in CONFUSION_GROUPS.items()
        if first in members and second in members
    ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = class_names()
    with np.load(args.oof.resolve(), allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str)
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int64)
        logits = data["final_logits"].astype(np.float64)
    probabilities = softmax(logits)
    order = np.argsort(-probabilities, axis=1)
    predictions = order[:, 0]
    top2_predictions = order[:, 1]
    margins = (
        probabilities[np.arange(len(labels)), predictions]
        - probabilities[np.arange(len(labels)), top2_predictions]
    )
    matrix = confusion_matrix(labels, predictions, labels=np.arange(40))
    recalls = np.diag(matrix) / matrix.sum(axis=1)
    hard_ids = np.flatnonzero(recalls < HARD_RECALL_THRESHOLD).astype(np.int64)
    hard_set = set(hard_ids.tolist())
    ungrouped = sorted(
        hard_set
        - {
            member
            for members in CONFUSION_GROUPS.values()
            for member in members
        }
    )
    if ungrouped:
        raise ValueError(f"Hard classes missing a confusion group: {ungrouped}")
    hard_mass = probabilities[:, hard_ids].sum(axis=1)
    top1_hard = np.isin(predictions, hard_ids)
    top2_pair_group = np.asarray(
        [
            bool(shared_groups(int(first), int(second)))
            and (int(first) in hard_set or int(second) in hard_set)
            for first, second in zip(predictions, top2_predictions, strict=True)
        ],
        dtype=bool,
    )
    candidate_trigger = (
        top1_hard
        | (hard_mass >= HARD_MASS_THRESHOLD)
        | top2_pair_group
    )
    true_hard = np.isin(labels, hard_ids)

    class_rows: list[dict[str, object]] = []
    for class_id in range(40):
        selected = labels == class_id
        wrong = selected & (predictions != labels)
        top_confusions = sorted(
            (
                (int(matrix[class_id, target]), target)
                for target in range(40)
                if target != class_id and matrix[class_id, target] > 0
            ),
            reverse=True,
        )[:5]
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "samples": int(selected.sum()),
                "correct": int((selected & (predictions == labels)).sum()),
                "errors": int(wrong.sum()),
                "recall": float(recalls[class_id]),
                "hard_class": int(class_id in hard_set),
                "mean_top1_top2_margin": float(margins[selected].mean()),
                "median_top1_top2_margin": float(np.median(margins[selected])),
                "mean_error_margin": (
                    float(margins[wrong].mean()) if wrong.any() else None
                ),
                "high_confidence_errors_margin_ge_0_30": int(
                    (wrong & (margins >= 0.30)).sum()
                ),
                "mean_hard_probability_mass": float(hard_mass[selected].mean()),
                "candidate_trigger_rate": float(candidate_trigger[selected].mean()),
                "confusion_groups": "|".join(group_names_for_class(class_id)),
                "top_confusions": json.dumps(
                    [
                        {
                            "class_id": target,
                            "class_name": names[target],
                            "count": count,
                        }
                        for count, target in top_confusions
                    ],
                    ensure_ascii=False,
                ),
            }
        )
    write_csv(output_dir / "class_difficulty.csv", class_rows)

    with (output_dir / "confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class_id", *range(40)])
        for class_id, values in enumerate(matrix):
            writer.writerow([class_id, *values.tolist()])

    edge_rows: list[dict[str, object]] = []
    for first in range(40):
        for second in range(first + 1, 40):
            forward = int(matrix[first, second])
            backward = int(matrix[second, first])
            if forward + backward == 0:
                continue
            edge_rows.append(
                {
                    "class_a": first,
                    "name_a": names[first],
                    "class_b": second,
                    "name_b": names[second],
                    "a_predicted_as_b": forward,
                    "b_predicted_as_a": backward,
                    "symmetric_confusions": forward + backward,
                    "shared_frozen_groups": "|".join(
                        shared_groups(first, second)
                    ),
                }
            )
    edge_rows.sort(
        key=lambda row: int(row["symmetric_confusions"]), reverse=True
    )
    write_csv(output_dir / "confusion_edges.csv", edge_rows)

    sample_rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        top1 = int(predictions[index])
        top2 = int(top2_predictions[index])
        sample_rows.append(
            {
                "sample_id": sample_id,
                "fold": int(folds[index]),
                "true_class_id": int(labels[index]),
                "true_class_name": names[int(labels[index])],
                "true_is_hard": int(true_hard[index]),
                "final_prediction": top1,
                "final_prediction_name": names[top1],
                "final_correct": int(top1 == labels[index]),
                "top1_probability": float(probabilities[index, top1]),
                "top2_prediction": top2,
                "top2_prediction_name": names[top2],
                "top2_probability": float(probabilities[index, top2]),
                "top1_top2_margin": float(margins[index]),
                "hard_probability_mass": float(hard_mass[index]),
                "top1_is_hard": int(top1_hard[index]),
                "top2_pair_in_frozen_group": int(top2_pair_group[index]),
                "top2_shared_groups": "|".join(shared_groups(top1, top2)),
                "candidate_local_trigger_v1": int(candidate_trigger[index]),
                "true_confusion_groups": "|".join(
                    group_names_for_class(int(labels[index]))
                ),
            }
        )
    write_csv(output_dir / "oof_route_features.csv", sample_rows)

    group_payload = {
        name: {
            "class_ids": list(members),
            "class_names": [names[class_id] for class_id in members],
            "hard_members": [
                class_id for class_id in members if class_id in hard_set
            ],
            "boundary_members": [
                class_id for class_id in members if class_id not in hard_set
            ],
        }
        for name, members in CONFUSION_GROUPS.items()
    }
    report = {
        "source_oof": str(args.oof.resolve()),
        "samples": int(len(labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "hard_class_rule": f"per-class Recall < {HARD_RECALL_THRESHOLD:.2f}",
        "hard_class_ids": hard_ids.tolist(),
        "hard_class_names": [names[class_id] for class_id in hard_ids],
        "hard_class_count": int(len(hard_ids)),
        "hard_true_samples": int(true_hard.sum()),
        "confusion_groups": group_payload,
        "candidate_trigger_v1": {
            "rule": (
                "top1 in hard classes OR hard-class probability mass >= 0.50 "
                "OR top1/top2 share a frozen confusion group with at least one hard member"
            ),
            "hard_probability_mass_threshold": HARD_MASS_THRESHOLD,
            "triggered_samples": int(candidate_trigger.sum()),
            "trigger_rate": float(candidate_trigger.mean()),
            "true_hard_coverage": float(candidate_trigger[true_hard].mean()),
            "non_hard_trigger_rate": float(candidate_trigger[~true_hard].mean()),
            "non_hard_triggered_samples": int(
                (candidate_trigger & ~true_hard).sum()
            ),
            "status": (
                "Frozen for new-500 sampling and boundary-negative construction; "
                "not yet accepted as the final deployment router."
            ),
        },
    }
    (output_dir / "hard_action_protocol.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
