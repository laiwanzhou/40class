"""Audit the 158 P89 errors missed by every frozen non-redundant modality teacher.

The subset definition intentionally reproduces the historical 353/195/158
inventory.  Later P91/P92 models are evaluated only after the subset is frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax

from p91_unrestricted_fusion_teacher import build_cohorts


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "hard158_topk_audit" / "analysis.json",
    )
    return parser.parse_args()


def class_names() -> dict[int, str]:
    result: dict[int, str] = {}
    with (HERE / "data/manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["class_id"])] = row["class_name"]
    return result


def ranks(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-probability, axis=1, kind="stable")
    return np.argmax(order == labels[:, None], axis=1).astype(np.int64) + 1


def top_label(probability: np.ndarray, names: dict[int, str]) -> list[str]:
    prediction = probability.argmax(axis=1)
    return [f"{int(value)} {names[int(value)]}" for value in prediction]


def aligned_rows(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {str(value): row for row, value in enumerate(source_ids.astype(str))}
    return np.asarray([lookup[str(value)] for value in target_ids], dtype=np.int64)


def blended_scores(logits: np.ndarray, base: np.ndarray, weight: float) -> np.ndarray:
    neural = softmax(logits.astype(np.float64), axis=1)
    anchor = np.full((len(base), 40), 0.06 / 39.0, dtype=np.float64)
    anchor[np.arange(len(base)), base.astype(np.int64)] = 0.94
    return weight * np.log(np.clip(neural, 1e-8, 1.0)) + (1.0 - weight) * np.log(
        np.clip(anchor, 1e-8, 1.0)
    )


def load_p91_scores(split: str, target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if split == "H2_confirmation":
        path = PROJECT / "runs/p91_hierarchical_multimodal_h3_v3/inner_predictions.npz"
        with np.load(path, allow_pickle=False) as source:
            rows = aligned_rows(source["sample_ids"], target_ids)
            logits = source["direct_logits"][rows]
            base = source["teacher_prediction"][rows]
            weight = float(source["selected_constant_weight"])
        return logits, blended_scores(logits, base, weight)
    if split == "H3_independent_fold0":
        path = PROJECT / "runs/p91_hierarchical_multimodal_h3_v3/predictions.npz"
        with np.load(path, allow_pickle=False) as source:
            rows = aligned_rows(source["sample_ids"], target_ids)
            logits = source["direct_logits"][rows]
            base = source["base_prediction"][rows]
            weight = float(source["selected_blend_weight"])
        return logits, blended_scores(logits, base, weight)
    return None


def load_p92_h3(target_ids: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    vjepa_path = PROJECT / "runs/p92_vjepa2_vitl_ssv2_12view_fold0_v1/fold0_logits.npz"
    with np.load(vjepa_path, allow_pickle=False) as source:
        rows = aligned_rows(source["sample_ids"], target_ids)
        for key in source.files:
            if key.endswith("_logits"):
                output[key] = source[key][rows]
    token_path = PROJECT / "runs/p92_vjepa2_token_fusion_h3_v1/predictions.npz"
    if token_path.exists():
        with np.load(token_path, allow_pickle=False) as source:
            rows = aligned_rows(source["sample_ids"], target_ids)
            output["p92_token_fusion_logits"] = source["logits"][rows]
    return output


def topk_count(rank_values: np.ndarray, ks: tuple[int, ...] = (1, 2, 3, 5, 10)) -> dict[str, int]:
    return {f"top{k}": int(np.sum(rank_values <= k)) for k in ks}


def main() -> None:
    args = parse_args()
    names = class_names()
    cohorts = build_cohorts()
    split_names = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
    expert_names = cohorts[split_names[0]].expert_names
    # Historical non-redundant pool.  Index 2 is a highly redundant IV2 signal
    # that was not part of the frozen 195 count.
    pool_indices = [3, 4, 5, 6, 7, 8]
    pool_names = [expert_names[index] for index in pool_indices]

    records: list[dict[str, Any]] = []
    class_totals: Counter[int] = Counter()
    class_errors: Counter[int] = Counter()
    split_summary: list[dict[str, Any]] = []
    expert_rank_blocks: list[np.ndarray] = []
    safe_raw_rank_blocks: list[np.ndarray] = []
    hard_labels: list[np.ndarray] = []

    for split_name in split_names:
        cohort = cohorts[split_name]
        probability = cohort.expert_probability
        labels = cohort.labels
        safe = cohort.safe_prediction
        pool_probability = probability[:, pool_indices]
        pool_top1 = pool_probability.argmax(axis=2)
        errors = safe != labels
        recoverable = errors & np.any(pool_top1 == labels[:, None], axis=1)
        hard = errors & ~recoverable
        class_totals.update(labels.tolist())
        class_errors.update(labels[errors].tolist())
        split_summary.append(
            {
                "split": split_name,
                "rows": int(len(labels)),
                "safe_errors": int(errors.sum()),
                "recoverable_top1_union": int(recoverable.sum()),
                "all_teachers_top1_wrong": int(hard.sum()),
            }
        )
        selected = np.flatnonzero(hard)
        selected_ids = cohort.sample_ids[selected]
        selected_labels = labels[selected]
        selected_probability = pool_probability[selected]
        selected_ranks = np.stack(
            [ranks(selected_probability[:, expert], selected_labels) for expert in range(len(pool_indices))],
            axis=1,
        )
        safe_raw_ranks = ranks(probability[selected, 1], selected_labels)
        expert_rank_blocks.append(selected_ranks)
        safe_raw_rank_blocks.append(safe_raw_ranks)
        hard_labels.append(selected_labels)

        p91 = load_p91_scores(split_name, selected_ids)
        if p91 is not None:
            p91_direct_logits, p91_blended = p91
            p91_direct_rank = ranks(p91_direct_logits, selected_labels)
            p91_blended_rank = ranks(p91_blended, selected_labels)
            p91_direct_top = p91_direct_logits.argmax(axis=1)
            p91_blended_top = p91_blended.argmax(axis=1)
        else:
            p91_direct_rank = p91_blended_rank = np.full(len(selected), -1, dtype=np.int64)
            p91_direct_top = p91_blended_top = np.full(len(selected), -1, dtype=np.int64)

        p92_scores = load_p92_h3(selected_ids) if split_name == "H3_independent_fold0" else {}
        if p92_scores:
            vjepa_keys = [key for key in p92_scores if key != "p92_token_fusion_logits"]
            vjepa_ranks = np.stack([ranks(p92_scores[key], selected_labels) for key in vjepa_keys], axis=1)
            vjepa_min_rank = vjepa_ranks.min(axis=1)
            token_logits = p92_scores.get("p92_token_fusion_logits")
            token_rank = ranks(token_logits, selected_labels) if token_logits is not None else np.full(len(selected), -1)
            token_top = token_logits.argmax(axis=1) if token_logits is not None else np.full(len(selected), -1)
        else:
            vjepa_min_rank = token_rank = np.full(len(selected), -1, dtype=np.int64)
            token_top = np.full(len(selected), -1, dtype=np.int64)

        for local, row in enumerate(selected):
            item: dict[str, Any] = {
                "split": split_name,
                "sample_id": str(cohort.sample_ids[row]),
                "user": str(cohort.users[row]),
                "true_class_id": int(labels[row]),
                "true_class": names[int(labels[row])],
                "safe_prediction_id": int(safe[row]),
                "safe_prediction": names[int(safe[row])],
                "safe_raw_true_rank": int(safe_raw_ranks[local]),
                "existing_pool_min_true_rank": int(selected_ranks[local].min()),
                "existing_pool_top2": bool(selected_ranks[local].min() <= 2),
                "existing_pool_top3": bool(selected_ranks[local].min() <= 3),
                "existing_pool_top5": bool(selected_ranks[local].min() <= 5),
                "existing_pool_top10": bool(selected_ranks[local].min() <= 10),
                "p91_direct_true_rank": int(p91_direct_rank[local]),
                "p91_direct_top1": "" if p91_direct_top[local] < 0 else f"{int(p91_direct_top[local])} {names[int(p91_direct_top[local])]}",
                "p91_blended_true_rank": int(p91_blended_rank[local]),
                "p91_blended_top1": "" if p91_blended_top[local] < 0 else f"{int(p91_blended_top[local])} {names[int(p91_blended_top[local])]}",
                "vjepa_candidate_min_true_rank": int(vjepa_min_rank[local]),
                "p92_token_true_rank": int(token_rank[local]),
                "p92_token_top1": "" if token_top[local] < 0 else f"{int(token_top[local])} {names[int(token_top[local])]}",
            }
            for expert_offset, expert_name in enumerate(pool_names):
                top = int(selected_probability[local, expert_offset].argmax())
                item[f"{expert_name}_top1"] = f"{top} {names[top]}"
                item[f"{expert_name}_true_rank"] = int(selected_ranks[local, expert_offset])
            records.append(item)

    all_ranks = np.concatenate(expert_rank_blocks)
    all_safe_raw_ranks = np.concatenate(safe_raw_rank_blocks)
    all_labels = np.concatenate(hard_labels)
    if len(records) != 158:
        raise RuntimeError(f"historical hard subset mismatch: expected 158, got {len(records)}")

    class_rows = []
    for class_id, count in Counter(all_labels.tolist()).most_common():
        selected = all_labels == class_id
        min_rank = all_ranks[selected].min(axis=1)
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "hard158_count": int(count),
                "share_of_hard158": float(count / len(all_labels)),
                "source_rows": int(class_totals[class_id]),
                "safe_errors": int(class_errors[class_id]),
                "hard_share_of_class_rows": float(count / class_totals[class_id]),
                "hard_share_of_safe_errors": float(count / class_errors[class_id]),
                **topk_count(min_rank),
            }
        )

    confusion_rows = []
    for (prediction, truth), count in Counter(
        (row["safe_prediction_id"], row["true_class_id"]) for row in records
    ).most_common():
        confusion_rows.append(
            {
                "safe_prediction_id": prediction,
                "safe_prediction": names[prediction],
                "true_class_id": truth,
                "true_class": names[truth],
                "count": int(count),
            }
        )

    topk_rows = [
        {
            "model_or_union": "existing_nonredundant_teacher_union",
            "eligible_rows": len(all_labels),
            **topk_count(all_ranks.min(axis=1)),
        },
        {
            "model_or_union": "p89_safe_raw_probability",
            "eligible_rows": len(all_labels),
            **topk_count(all_safe_raw_ranks),
        },
    ]
    for column, label in (
        ("p91_direct_true_rank", "P91_direct_head_H2_H3"),
        ("p91_blended_true_rank", "P91_source_selected_blend_H2_H3"),
        ("vjepa_candidate_min_true_rank", "VJEPA_candidate_union_H3"),
        ("p92_token_true_rank", "P92_token_fusion_direct_H3"),
    ):
        values = np.asarray([row[column] for row in records], dtype=np.int64)
        values = values[values > 0]
        topk_rows.append({"model_or_union": label, "eligible_rows": len(values), **topk_count(values)})
    for expert_offset, expert_name in enumerate(pool_names):
        topk_rows.append(
            {
                "model_or_union": expert_name,
                "eligible_rows": len(all_labels),
                **topk_count(all_ranks[:, expert_offset]),
            }
        )

    output = {
        "definition": {
            "source_rows": 2470,
            "p89_safe_errors": 353,
            "historical_top1_recoverable": 195,
            "all_frozen_teacher_top1_wrong": 158,
            "pool_names": pool_names,
            "note": "Subset is frozen before P91/P92 evaluation; H1 has no honest P91/P92 OOF score.",
        },
        "split_summary": split_summary,
        "topk_summary": topk_rows,
        "class_summary": class_rows,
        "confusion_summary": confusion_rows,
        "records": sorted(
            records,
            key=lambda row: (
                -row["existing_pool_min_true_rank"],
                -row["true_class_id"],
                row["sample_id"],
            ),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("definition", "split_summary", "topk_summary")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
