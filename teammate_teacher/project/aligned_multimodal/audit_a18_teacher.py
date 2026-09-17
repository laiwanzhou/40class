"""A18 full-refit fit-set error structure and old-A9 scope-aware comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from a18_full_teacher_data import A18_ROWS, A18_SOURCE_USER_SET
from audit_p102_session_closure import classification_metrics


HERE = Path(__file__).resolve().parent
DEFAULT_A18 = HERE / "runs/a18_full_teacher_v1/full_fit_session_predictions.npz"
DEFAULT_OLD_A9 = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_MANIFEST = HERE / "data/manifest.csv"
DEFAULT_OUTPUT = HERE / "runs/a18_teacher_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a18-oof", type=Path, default=DEFAULT_A18)
    parser.add_argument("--old-a9-oof", type=Path, default=DEFAULT_OLD_A9)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Iterable[str] | None = None,
) -> None:
    records = list(rows)
    columns = list(fieldnames) if fieldnames is not None else []
    if records and not columns:
        columns = list(records[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records and not columns:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def class_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            class_id = int(row["class_id"])
            names.setdefault(class_id, row.get("class_name", str(class_id)))
    if set(names) != set(range(40)):
        raise RuntimeError("canonical manifest does not define all 40 classes")
    return names


def true_rank(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    true_value = probability[np.arange(len(labels)), labels]
    return 1 + np.sum(probability > true_value[:, None], axis=1)


def top_ids(probability: np.ndarray, k: int = 5) -> np.ndarray:
    return np.argsort(-probability, axis=1, kind="stable")[:, :k]


def per_class_rows(
    labels: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    names: dict[int, str],
    prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id in range(40):
        selected = labels == class_id
        correct = int(np.sum(selected & (prediction == class_id)))
        top5 = int(np.sum(np.any(top_ids(probability[selected]) == class_id, axis=1)))
        support = int(selected.sum())
        rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                f"{prefix}_support": support,
                f"{prefix}_top1_correct": correct,
                f"{prefix}_top1_recall": correct / max(support, 1),
                f"{prefix}_top5_correct": top5,
                f"{prefix}_top5_recall": top5 / max(support, 1),
            }
        )
    return rows


def confusion_rows(
    labels: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
    names: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    directed: Counter[tuple[int, int]] = Counter()
    undirected: Counter[tuple[int, int]] = Counter()
    subjects: dict[tuple[int, int], set[str]] = defaultdict(set)
    unordered_subjects: dict[tuple[int, int], set[str]] = defaultdict(set)
    for truth, pred, user in zip(labels, prediction, users):
        if truth == pred:
            continue
        edge = (int(truth), int(pred))
        pair = tuple(sorted(edge))
        directed[edge] += 1
        undirected[pair] += 1
        subjects[edge].add(str(user))
        unordered_subjects[pair].add(str(user))
    directed_rows = [
        {
            "true_class": truth,
            "true_name": names[truth],
            "predicted_class": pred,
            "predicted_name": names[pred],
            "errors": count,
            "subject_count": len(subjects[(truth, pred)]),
            "subjects": "|".join(sorted(subjects[(truth, pred)])),
        }
        for (truth, pred), count in directed.most_common()
    ]
    pair_rows = [
        {
            "class_a": left,
            "class_a_name": names[left],
            "class_b": right,
            "class_b_name": names[right],
            "bidirectional_errors": count,
            "a_to_b": directed[(left, right)],
            "b_to_a": directed[(right, left)],
            "subject_count": len(unordered_subjects[(left, right)]),
            "subjects": "|".join(sorted(unordered_subjects[(left, right)])),
        }
        for (left, right), count in undirected.most_common()
    ]
    return directed_rows, pair_rows


def validate_a18(values: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    required = {"sample_ids", "users", "labels", "selected_probability"}
    missing = sorted(required - set(values))
    if missing:
        raise KeyError(f"A18 audit input missing {missing}")
    sample_ids = np.asarray(values["sample_ids"]).astype(str)
    users = np.asarray(values["users"]).astype(str)
    labels = np.asarray(values["labels"], dtype=np.int64)
    probability = np.asarray(values["selected_probability"], dtype=np.float64)
    if len(sample_ids) != A18_ROWS or len(np.unique(sample_ids)) != A18_ROWS:
        raise RuntimeError("A18 audit row contract changed")
    if set(users.tolist()) != A18_SOURCE_USER_SET:
        raise RuntimeError("A18 audit subject contract changed")
    if probability.shape != (A18_ROWS, 40) or not np.isfinite(probability).all():
        raise RuntimeError("A18 audit probability contract changed")
    probability /= probability.sum(axis=1, keepdims=True)
    return sample_ids, users, labels, probability


def plot_confusion(matrix: np.ndarray, names: dict[int, str], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(18, 16), dpi=180)
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("A18 Full Teacher confusion matrix (fit set, n=2,914)")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(40), [str(value) for value in range(40)], rotation=90)
    axis.set_yticks(range(40), [str(value) for value in range(40)])
    for row in range(40):
        for column in range(40):
            value = int(matrix[row, column])
            if value:
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color="white" if value > matrix.max() * 0.5 else "black",
                )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    a18_path = args.a18_oof.resolve()
    old_path = args.old_a9_oof.resolve()
    manifest_path = args.manifest.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = class_names(manifest_path)
    sample_ids, users, labels, probability = validate_a18(load_npz(a18_path))
    prediction = probability.argmax(axis=1)
    ranks = true_rank(probability, labels)
    top5 = top_ids(probability, 5)
    metrics = classification_metrics(probability, labels, users)
    matrix = confusion_matrix(labels, prediction, labels=np.arange(40))
    np.savetxt(output / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    plot_confusion(matrix, names, output / "confusion_matrix.png")
    directed, pairs = confusion_rows(labels, prediction, users, names)
    write_csv(output / "confusion_directed.csv", directed)
    write_csv(output / "confusion_pairs.csv", pairs)

    error_rows: list[dict[str, Any]] = []
    for row in np.flatnonzero(prediction != labels):
        error_rows.append(
            {
                "sample_id": sample_ids[row],
                "subject": users[row],
                "true_class": int(labels[row]),
                "true_name": names[int(labels[row])],
                "predicted_class": int(prediction[row]),
                "predicted_name": names[int(prediction[row])],
                "true_rank": int(ranks[row]),
                "true_in_top5": bool(ranks[row] <= 5),
                "predicted_probability": float(probability[row, prediction[row]]),
                "true_probability": float(probability[row, labels[row]]),
                "top5_classes": "|".join(map(str, top5[row].tolist())),
                "top5_names": "|".join(names[int(value)] for value in top5[row]),
                "top5_probabilities": "|".join(
                    f"{probability[row, value]:.8f}" for value in top5[row]
                ),
            }
        )
    write_csv(output / "errors.csv", error_rows)
    top5_miss_rows = [row for row in error_rows if not row["true_in_top5"]]
    ranking_rows = [row for row in error_rows if row["true_in_top5"]]
    write_csv(
        output / "top5_miss_samples.csv",
        top5_miss_rows,
        fieldnames=list(error_rows[0]) if error_rows else None,
    )
    write_csv(output / "top5_ranking_errors.csv", ranking_rows)

    per_subject = []
    for user in sorted(set(users.tolist())):
        selected = users == user
        subject_prediction = prediction[selected]
        subject_labels = labels[selected]
        per_subject.append(
            {
                "subject": user,
                "rows": int(selected.sum()),
                "top1_correct": int(np.sum(subject_prediction == subject_labels)),
                "top1_errors": int(np.sum(subject_prediction != subject_labels)),
                "top1": float(np.mean(subject_prediction == subject_labels)),
                "top5_correct": int(np.sum(ranks[selected] <= 5)),
                "top5_miss": int(np.sum(ranks[selected] > 5)),
                "top5": float(np.mean(ranks[selected] <= 5)),
            }
        )
    write_csv(output / "per_subject.csv", per_subject)
    a18_class = per_class_rows(labels, prediction, probability, names, "a18")

    old = load_npz(old_path)
    old_ids = np.asarray(old["sample_ids"]).astype(str)
    old_probability = np.asarray(old["selected_probability"], dtype=np.float64)
    old_lookup = {sample_id: row for row, sample_id in enumerate(old_ids)}
    common_rows = np.asarray(
        [row for row, sample_id in enumerate(sample_ids) if sample_id in old_lookup],
        dtype=np.int64,
    )
    old_rows = np.asarray([old_lookup[sample_ids[row]] for row in common_rows])
    if len(common_rows) != len(old_ids):
        raise RuntimeError("old A9 sample set is not an exact subset of A18")
    common_labels = labels[common_rows]
    common_users = users[common_rows]
    common_new_probability = probability[common_rows]
    common_old_probability = old_probability[old_rows]
    common_new_prediction = common_new_probability.argmax(axis=1)
    common_old_prediction = common_old_probability.argmax(axis=1)
    old_metrics = classification_metrics(
        common_old_probability, common_labels, common_users
    )
    new_common_metrics = classification_metrics(
        common_new_probability, common_labels, common_users
    )
    old_class = per_class_rows(
        common_labels,
        common_old_prediction,
        common_old_probability,
        names,
        "old_a9",
    )
    new_class = per_class_rows(
        common_labels,
        common_new_prediction,
        common_new_probability,
        names,
        "a18_common12",
    )
    class_comparison: list[dict[str, Any]] = []
    for old_row, new_row in zip(old_class, new_class):
        merged = {**old_row, **{key: value for key, value in new_row.items() if key not in {"class_id", "class_name"}}}
        merged["top1_correct_delta"] = (
            new_row["a18_common12_top1_correct"] - old_row["old_a9_top1_correct"]
        )
        merged["top1_recall_delta_pp"] = 100.0 * (
            new_row["a18_common12_top1_recall"] - old_row["old_a9_top1_recall"]
        )
        merged["top5_correct_delta"] = (
            new_row["a18_common12_top5_correct"] - old_row["old_a9_top5_correct"]
        )
        class_comparison.append(merged)
    write_csv(output / "per_class_a18_full.csv", a18_class)
    write_csv(output / "per_class_old_a9_comparison.csv", class_comparison)

    old_correct = common_old_prediction == common_labels
    new_correct = common_new_prediction == common_labels
    resolved_indices = common_rows[(~old_correct) & new_correct]
    regressed_indices = common_rows[old_correct & (~new_correct)]
    still_wrong_indices = common_rows[(~old_correct) & (~new_correct)]
    resolved_rows = [
        {
            "sample_id": sample_ids[row],
            "subject": users[row],
            "true_class": int(labels[row]),
            "true_name": names[int(labels[row])],
            "old_prediction": int(common_old_prediction[np.where(common_rows == row)[0][0]]),
            "new_prediction": int(prediction[row]),
        }
        for row in resolved_indices
    ]
    write_csv(output / "old_a9_errors_resolved_by_a18.csv", resolved_rows)

    # Hard candidates are derived only from the new A18 error pool.  Repeated
    # boundaries need support from at least three subjects; Top-5 misses remain
    # representation candidates regardless of pair frequency.
    pair_lookup = {
        tuple(sorted((int(row["class_a"]), int(row["class_b"])))): row
        for row in pairs
    }
    hard_candidates: list[dict[str, Any]] = []
    for row in error_rows:
        pair = tuple(sorted((int(row["true_class"]), int(row["predicted_class"]))))
        pair_record = pair_lookup[pair]
        top5_miss = not bool(row["true_in_top5"])
        repeated_boundary = (
            int(pair_record["bidirectional_errors"]) >= 5
            and int(pair_record["subject_count"]) >= 3
        )
        specialist_eligible = top5_miss or repeated_boundary
        hard_candidates.append(
            {
                **row,
                "candidate_type": (
                    "representation_top5_miss"
                    if top5_miss
                    else (
                        "repeated_top5_ranking_boundary"
                        if repeated_boundary
                        else "isolated_fit_residual_observation_only"
                    )
                ),
                "pair_errors": int(pair_record["bidirectional_errors"]),
                "pair_subject_count": int(pair_record["subject_count"]),
                "specialist_priority": (
                    "high"
                    if top5_miss and repeated_boundary
                    else ("medium" if specialist_eligible else "not_authorized")
                ),
                "specialist_eligible": specialist_eligible,
                "provenance": "A18_new_error_pool_only",
            }
        )
    write_csv(output / "hard_case_candidates.csv", hard_candidates)

    improved_classes = sorted(
        class_comparison,
        key=lambda row: (-row["top1_correct_delta"], row["class_id"]),
    )
    difficult_classes = sorted(
        a18_class,
        key=lambda row: (row["a18_top1_recall"], -row["a18_support"], row["class_id"]),
    )
    summary = {
        "status": "complete",
        "protocol": (
            "Fresh A18 full-refit fit-set error audit. No P104/P105/P106/P107 "
            "family, trigger, router, threshold, blend, or specialist result is loaded."
        ),
        "inputs": {
            "a18": {"path": str(a18_path), "sha256": sha256(a18_path)},
            "old_a9": {"path": str(old_path), "sha256": sha256(old_path)},
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        },
        "a18": {
            "metrics": metrics,
            "metric_scope": "training_fit_set_with_in_sample_session_transition",
            "generalization_claim_allowed": False,
            "top1_errors": int(np.sum(prediction != labels)),
            "top5_contains_correct": int(np.sum(ranks <= 5)),
            "top5_misses": int(np.sum(ranks > 5)),
            "top5_ranking_errors": int(np.sum((prediction != labels) & (ranks <= 5))),
            "subjects": len(set(users.tolist())),
            "most_difficult_classes": difficult_classes[:10],
            "top_directed_confusions": directed[:20],
            "top_class_pairs": pairs[:20],
        },
        "old_a9_comparison_on_common_12_subjects": {
            "comparison_scope": (
                "non_comparable_apparent_delta: A18 is fit-set and old A9 is "
                "subject-disjoint OOF; do not interpret as generalization gain"
            ),
            "rows": int(len(common_rows)),
            "old_a9": old_metrics,
            "a18": new_common_metrics,
            "top1_correct_delta": int(
                new_common_metrics["top1_correct"] - old_metrics["top1_correct"]
            ),
            "top1_delta_pp": 100.0 * (
                new_common_metrics["top1"] - old_metrics["top1"]
            ),
            "top5_correct_delta": int(
                new_common_metrics["top5_correct"] - old_metrics["top5_correct"]
            ),
            "top5_delta_pp": 100.0 * (
                new_common_metrics["top5"] - old_metrics["top5"]
            ),
            "old_wrong_new_correct": int(len(resolved_indices)),
            "old_correct_new_wrong": int(len(regressed_indices)),
            "both_wrong": int(len(still_wrong_indices)),
            "most_improved_classes": improved_classes[:10],
            "most_regressed_classes": sorted(
                class_comparison,
                key=lambda row: (row["top1_correct_delta"], row["class_id"]),
            )[:10],
        },
        "hard_case_candidates": {
            "rows": len(hard_candidates),
            "specialist_eligible_rows": sum(
                bool(row["specialist_eligible"]) for row in hard_candidates
            ),
            "observation_only_rows": sum(
                not bool(row["specialist_eligible"]) for row in hard_candidates
            ),
            "top5_miss_rows": sum(
                row["candidate_type"] == "representation_top5_miss"
                for row in hard_candidates
            ),
            "ranking_boundary_rows": sum(
                row["candidate_type"] == "repeated_top5_ranking_boundary"
                for row in hard_candidates
            ),
            "derivation": (
                "all new A18 residual errors are retained as observations; Specialist "
                "eligibility requires Top-5 miss or bidirectional pair >=5 errors "
                "across >=3 subjects"
            ),
        },
        "constraints": {
            "source_safe": True,
            "held_label_selection_rows": 0,
            "h3_rows_selected": 0,
            "h3_confirmation_run": False,
            "b_teacher_trained": False,
            "router_trained": False,
            "old_confusion_family_loaded": False,
            "threshold_seed_blend_sweep": False,
        },
        "answerability": {
            "true_cross_subject_remaining_boundaries_answered": False,
            "reason": (
                "The user-directed no-split full refit has no unseen-subject "
                "evaluation. Listed boundaries are residual fit errors only."
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"a18": summary["a18"], "comparison": summary["old_a9_comparison_on_common_12_subjects"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
