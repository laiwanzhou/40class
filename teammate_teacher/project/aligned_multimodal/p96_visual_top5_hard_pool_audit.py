"""Build sample-level visual Top-5 hard pools from strict OOF predictions.

The only selected samples are those for which the frozen primary visual teacher
is wrong at Top-1 while the true label is ranked from 2 through 5.  Every
selected sample keeps its own five candidates.  Repeated Top-1-versus-truth
pairs are summarised only to identify reusable specialist training tasks.

Pool construction uses source OOF only.  H2 is a frozen confirmation report;
it cannot add or modify source pools.  There is intentionally no H3 path and no
deployment router in this script.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import log_softmax, softmax

from p90_teacher_common import REPO_ROOT, load_protocol
from p91_unrestricted_fusion_teacher import build_cohorts


VMAE_OOF = REPO_ROOT / "runs/p90_videomaev2_distilled_teacher_v1/oof_logits.npz"
IV2_OOF = REPO_ROOT / "runs/p90_internvideo2_l_k400_teacher_v1/oof_logits.npz"
CLASS_MANIFEST = REPO_ROOT / "aligned_multimodal/data/p27_strong_inner/fold_0.csv"
OUTPUT = REPO_ROOT / "runs/p96_primary_visual_top5_hard_pools_v1"


def align_indices(all_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {str(value): row for row, value in enumerate(all_ids)}
    return np.asarray([lookup[str(value)] for value in target_ids], dtype=np.int64)


def load_class_names() -> dict[int, str]:
    with CLASS_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["class_id"]): row["class_name"].split("_", 1)[-1]
        for row in rows
    }


def load_primary_visual_scores() -> tuple[np.ndarray, np.ndarray]:
    with np.load(VMAE_OOF, allow_pickle=False) as source:
        sample_ids = source["sample_ids"].astype(str)
        vmae_logits = np.asarray(source["early_late_logits"], dtype=np.float64)
    with np.load(IV2_OOF, allow_pickle=False) as source:
        if not np.array_equal(sample_ids, source["sample_ids"].astype(str)):
            raise ValueError("VideoMAE and InternVideo2 OOF sample orders differ")
        iv2_logits = np.asarray(
            source["early_late_plus_k400_logits"], dtype=np.float64
        )
    scores = 0.5 * log_softmax(vmae_logits, axis=1)
    scores += 0.5 * log_softmax(iv2_logits, axis=1)
    return sample_ids, scores


def pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def pair_id(key: tuple[int, int]) -> str:
    return f"pair_{key[0]:02d}_{key[1]:02d}"


def build_hard_rows(
    split: str,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    scores: np.ndarray,
    names: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probabilities = softmax(scores, axis=1)
    order = np.argsort(-scores, axis=1)
    ranks = np.argmax(order == labels[:, None], axis=1) + 1
    selected = (ranks >= 2) & (ranks <= 5)
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(selected):
        top5 = order[index, :5].astype(int)
        truth = int(labels[index])
        top1 = int(top5[0])
        key = pair_key(top1, truth)
        row: dict[str, Any] = {
            "split": split,
            "sample_id": str(sample_ids[index]),
            "user": str(users[index]),
            "true_class_id": truth,
            "true_class_name": names[truth],
            "true_rank": int(ranks[index]),
            "visual_top1_id": top1,
            "visual_top1_name": names[top1],
            "visual_top1_probability": float(probabilities[index, top1]),
            "visual_top1_top2_margin": float(
                probabilities[index, top5[0]] - probabilities[index, top5[1]]
            ),
            "true_class_probability": float(probabilities[index, truth]),
            "dynamic_pool_size": 5,
            "pair_pool_id": pair_id(key),
        }
        for rank_index, class_id in enumerate(top5, start=1):
            row[f"top{rank_index}_class_id"] = int(class_id)
            row[f"top{rank_index}_class_name"] = names[int(class_id)]
            row[f"top{rank_index}_probability"] = float(
                probabilities[index, class_id]
            )
        rows.append(row)

    rank_counts = Counter(int(row["true_rank"]) for row in rows)
    ordered_signatures = {
        tuple(int(row[f"top{rank}_class_id"]) for rank in range(1, 6))
        for row in rows
    }
    unordered_signatures = {
        tuple(sorted(int(row[f"top{rank}_class_id"]) for rank in range(1, 6)))
        for row in rows
    }
    summary = {
        "total_samples": len(labels),
        "visual_top1_correct": int(np.sum(ranks == 1)),
        "visual_top1_wrong": int(np.sum(ranks != 1)),
        "selected_top1_wrong_truth_rank_2_to_5": len(rows),
        "selected_rate_of_all_samples": float(len(rows) / len(labels)),
        "selected_rate_of_top1_errors": float(len(rows) / np.sum(ranks != 1)),
        "true_rank_distribution": {
            str(rank): rank_counts.get(rank, 0) for rank in range(2, 6)
        },
        "unique_ordered_top5_signatures": len(ordered_signatures),
        "unique_unordered_top5_signatures": len(unordered_signatures),
        "selected_true_class_count": len(
            {int(row["true_class_id"]) for row in rows}
        ),
    }
    return rows, summary


def collect_pair_stats(
    rows: list[dict[str, Any]],
) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], set[str]], Counter[tuple[int, int]]]:
    counts: Counter[tuple[int, int]] = Counter()
    users: dict[tuple[int, int], set[str]] = defaultdict(set)
    directions: Counter[tuple[int, int]] = Counter()
    for row in rows:
        top1 = int(row["visual_top1_id"])
        truth = int(row["true_class_id"])
        key = pair_key(top1, truth)
        counts[key] += 1
        users[key].add(str(row["user"]))
        directions[(top1, truth)] += 1
    return counts, users, directions


def pool_tier(source_count: int, source_users: int) -> str:
    if source_count >= 2 and source_users >= 2:
        return "shared_cross_user"
    if source_count >= 2:
        return "repeated_single_user"
    return "singleton_tail"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def true_class_rows(
    source_rows: list[dict[str, Any]],
    h2_rows: list[dict[str, Any]],
    names: dict[int, str],
) -> list[dict[str, Any]]:
    source_counts = Counter(int(row["true_class_id"]) for row in source_rows)
    h2_counts = Counter(int(row["true_class_id"]) for row in h2_rows)
    source_users: dict[int, set[str]] = defaultdict(set)
    h2_users: dict[int, set[str]] = defaultdict(set)
    source_ranks: dict[int, Counter[int]] = defaultdict(Counter)
    for row in source_rows:
        class_id = int(row["true_class_id"])
        source_users[class_id].add(str(row["user"]))
        source_ranks[class_id][int(row["true_rank"])] += 1
    for row in h2_rows:
        h2_users[int(row["true_class_id"])].add(str(row["user"]))
    output = []
    for class_id in sorted(source_counts, key=lambda value: (-source_counts[value], value)):
        count = source_counts[class_id]
        user_count = len(source_users[class_id])
        if count >= 3 and user_count >= 2:
            tier = "recurrent_hard_true_class"
        elif count >= 2:
            tier = "repeated_tail_true_class"
        else:
            tier = "singleton_tail_true_class"
        output.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "source_hard_samples": count,
                "source_users": user_count,
                "source_true_rank2": source_ranks[class_id][2],
                "source_true_rank3": source_ranks[class_id][3],
                "source_true_rank4": source_ranks[class_id][4],
                "source_true_rank5": source_ranks[class_id][5],
                "class_tier": tier,
                "H2_hard_samples": h2_counts[class_id],
                "H2_users": len(h2_users[class_id]),
            }
        )
    return output


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_report(result: dict[str, Any]) -> str:
    source = result["splits"]["source_oof"]
    h2 = result["splits"]["H2_confirmation"]
    pair_summary = result["pair_pool_summary"]
    lines = [
        "# 主视觉 Top-5 困难样本池审计",
        "",
        "## 唯一筛选规则",
        "",
        "只保留冻结主视觉教师 Top-1 错误且真实类别排名为 Top-2 至 Top-5 的 OOF 样本。Top-1 已正确的样本不进入困难池；真实类别不在 Top-5 的错误样本也不进入本池，而应留给表示增强路线。",
        "",
        "每条困难样本保留它自己的 5 类动态候选池。类别不会因为属于某个语义 family 而整类进入；只有它在某条入选困难样本的真实 Top-5 中实际出现，才会出现在该样本池。",
        "",
        "## 数量",
        "",
        "| 划分 | 总样本 | 视觉Top-1错误 | 入选困难样本 | 错误覆盖率 | 唯一无序Top-5池 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Source OOF | {source['total_samples']} | "
            f"{source['visual_top1_wrong']} | "
            f"{source['selected_top1_wrong_truth_rank_2_to_5']} | "
            f"{percentage(source['selected_rate_of_top1_errors'])} | "
            f"{source['unique_unordered_top5_signatures']} |"
        ),
        (
            f"| H2冻结确认 | {h2['total_samples']} | "
            f"{h2['visual_top1_wrong']} | "
            f"{h2['selected_top1_wrong_truth_rank_2_to_5']} | "
            f"{percentage(h2['selected_rate_of_top1_errors'])} | "
            f"{h2['unique_unordered_top5_signatures']} |"
        ),
        "",
        "Source 的 185 条困难样本中有 177 种无序 Top-5 组合，说明几乎每条候选集合都不同，不适合压成少数人工静态大池。",
        "",
        "## 实际真实困难类",
        "",
        "Source 入选样本的真实标签涉及 35 类，但只有满足 `至少3条且跨至少2个用户` 的类别被标为 recurrent；其余仍是样本级长尾，不为它们单独建立类别池。没有入选样本的类别不会作为真实困难类加入。完整清单见 `source_hard_true_classes.csv`。",
        "",
        "未作为真实困难类入选：`0 Wash face`、`16 Fold clothes`、`17 Tap keyboard`、`31 Stretching`、`33 Lie down`。它们仍可能作为某条实际 Top-5 的竞争候选出现，但不会整类加入困难样本池。",
        "",
        "## 二分类共享模板",
        "",
        "每条困难样本同时产生一个 `错误Top-1 ↔ 真实类别` 二分类对子。它只用于统计哪些局部判别任务可以共享专项头，不替代该样本完整的动态 Top-5 池。",
        "",
        f"- Source 共 {pair_summary['source_unique_pair_pools']} 种实际二分类对子。",
        f"- {pair_summary['shared_cross_user_pair_pools']} 种在至少两个 source 用户重复出现，覆盖 {pair_summary['source_samples_in_shared_cross_user_pairs']}/{source['selected_top1_wrong_truth_rank_2_to_5']} 条 source 困难样本。",
        f"- {pair_summary['repeated_single_user_pair_pools']} 种只在单用户重复，暂不升级为跨用户专项头。",
        f"- {pair_summary['singleton_tail_pair_pools']} 种只出现一次，保留在通用动态 Top-5 长尾池。",
        f"- 冻结的 cross-user 对子在 H2 覆盖 {pair_summary['h2_samples_in_source_shared_cross_user_pairs']}/{h2['selected_top1_wrong_truth_rank_2_to_5']} 条困难样本；H2 不反向新增 source 池。",
        "",
        "| 二分类对子 | Source样本 | Source用户 | H2样本 | H2用户 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["shared_cross_user_pairs"]:
        lines.append(
            f"| {row['class_a_id']} {row['class_a_name']} ↔ "
            f"{row['class_b_id']} {row['class_b_name']} | "
            f"{row['source_samples']} | {row['source_users']} | "
            f"{row['H2_samples']} | {row['H2_users']} |"
        )
    lines.extend(
        [
            "",
            "## 当前边界",
            "",
            "本文件只完成困难样本和候选池定义。由于筛选时使用了 OOF 真值，它是训练/验证审计产物，不能直接作为测试时路由规则。下一阶段应先在这些动态 Top-5 池内训练 candidate-conditioned 重排器并做严格跨用户评估；池内判别足够强之后，才研究不看标签时如何路由。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    protocol = load_protocol()
    cohorts = build_cohorts()
    h1 = cohorts["H1_selection"]
    embargo = cohorts["E0_p87_sequence_source"]
    h2 = cohorts["H2_confirmation"]
    source_ids = np.concatenate((h1.sample_ids, embargo.sample_ids)).astype(str)
    source_labels = np.concatenate((h1.labels, embargo.labels)).astype(np.int64)
    source_users = np.concatenate((h1.users, embargo.users)).astype(str)

    visual_ids, all_scores = load_primary_visual_scores()
    if not np.array_equal(visual_ids, protocol.sample_ids.astype(str)):
        raise ValueError("primary visual OOF and protocol sample orders differ")
    source_scores = all_scores[align_indices(visual_ids, source_ids)]
    h2_scores = all_scores[align_indices(visual_ids, h2.sample_ids)]
    names = load_class_names()

    source_rows, source_summary = build_hard_rows(
        "source_oof",
        source_ids,
        source_labels,
        source_users,
        source_scores,
        names,
    )
    h2_rows, h2_summary = build_hard_rows(
        "H2_confirmation",
        h2.sample_ids.astype(str),
        h2.labels.astype(np.int64),
        h2.users.astype(str),
        h2_scores,
        names,
    )

    source_counts, source_pair_users, source_directions = collect_pair_stats(source_rows)
    h2_counts, h2_pair_users, h2_directions = collect_pair_stats(h2_rows)
    pair_rows: list[dict[str, Any]] = []
    for key, count in sorted(
        source_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        source_user_count = len(source_pair_users[key])
        tier = pool_tier(count, source_user_count)
        forward = source_directions[(key[0], key[1])]
        reverse = source_directions[(key[1], key[0])]
        pair_rows.append(
            {
                "pair_pool_id": pair_id(key),
                "pool_size": 2,
                "class_a_id": key[0],
                "class_a_name": names[key[0]],
                "class_b_id": key[1],
                "class_b_name": names[key[1]],
                "source_samples": count,
                "source_users": source_user_count,
                "source_a_top1_b_truth": forward,
                "source_b_top1_a_truth": reverse,
                "pool_tier": tier,
                "H2_samples": h2_counts[key],
                "H2_users": len(h2_pair_users[key]),
                "H2_a_top1_b_truth": h2_directions[(key[0], key[1])],
                "H2_b_top1_a_truth": h2_directions[(key[1], key[0])],
            }
        )

    pair_lookup = {row["pair_pool_id"]: row for row in pair_rows}
    for row in source_rows:
        pair = pair_lookup[str(row["pair_pool_id"])]
        row["source_pair_samples"] = pair["source_samples"]
        row["source_pair_users"] = pair["source_users"]
        row["pair_pool_tier"] = pair["pool_tier"]
    for row in h2_rows:
        pair = pair_lookup.get(str(row["pair_pool_id"]))
        row["source_pair_samples"] = pair["source_samples"] if pair else 0
        row["source_pair_users"] = pair["source_users"] if pair else 0
        row["pair_pool_tier"] = pair["pool_tier"] if pair else "unseen_in_source"

    class_rows = true_class_rows(source_rows, h2_rows, names)

    source_signatures = {
        tuple(sorted(int(row[f"top{rank}_class_id"]) for rank in range(1, 6)))
        for row in source_rows
    }
    h2_signature_seen = sum(
        tuple(sorted(int(row[f"top{rank}_class_id"]) for rank in range(1, 6)))
        in source_signatures
        for row in h2_rows
    )
    tier_counts = Counter(str(row["pool_tier"]) for row in pair_rows)
    shared_pair_ids = {
        str(row["pair_pool_id"])
        for row in pair_rows
        if row["pool_tier"] == "shared_cross_user"
    }
    result = {
        "experiment_id": "p96_primary_visual_top5_hard_pools_v1",
        "status": "HARD_SAMPLE_POOLS_FROZEN_ROUTING_NOT_STARTED",
        "selection_rule": (
            "Frozen primary visual Top-1 is wrong and the true class rank is 2..5."
        ),
        "candidate_generator": "videomaev2_base_plus_internvideo2_l_equal",
        "dynamic_pool_size": 5,
        "protocol": (
            "Source OOF constructs sample pools and pair templates; H2 confirms "
            "without adding pools; no H3 path and no deployment router."
        ),
        "splits": {
            "source_oof": source_summary,
            "H2_confirmation": {
                **h2_summary,
                "exact_unordered_top5_signature_seen_in_source": h2_signature_seen,
                "exact_unordered_top5_signature_seen_rate": float(
                    h2_signature_seen / len(h2_rows)
                ),
            },
        },
        "pair_pool_summary": {
            "source_unique_pair_pools": len(pair_rows),
            "shared_cross_user_pair_pools": tier_counts["shared_cross_user"],
            "repeated_single_user_pair_pools": tier_counts["repeated_single_user"],
            "singleton_tail_pair_pools": tier_counts["singleton_tail"],
            "source_samples_in_shared_cross_user_pairs": sum(
                str(row["pair_pool_id"]) in shared_pair_ids for row in source_rows
            ),
            "h2_samples_in_source_shared_cross_user_pairs": sum(
                str(row["pair_pool_id"]) in shared_pair_ids for row in h2_rows
            ),
            "h2_samples_with_any_pair_seen_in_source": sum(
                str(row["pair_pool_id"]) in pair_lookup for row in h2_rows
            ),
        },
        "hard_true_class_summary": {
            "source_true_classes": len(class_rows),
            "source_recurrent_true_classes": sum(
                row["class_tier"] == "recurrent_hard_true_class"
                for row in class_rows
            ),
            "source_repeated_tail_true_classes": sum(
                row["class_tier"] == "repeated_tail_true_class"
                for row in class_rows
            ),
            "source_singleton_tail_true_classes": sum(
                row["class_tier"] == "singleton_tail_true_class"
                for row in class_rows
            ),
            "classes_with_no_selected_source_sample": [
                {"class_id": class_id, "class_name": names[class_id]}
                for class_id in range(40)
                if class_id not in {int(row["class_id"]) for row in class_rows}
            ],
        },
        "shared_cross_user_pairs": [
            row for row in pair_rows if row["pool_tier"] == "shared_cross_user"
        ],
        "routing": {
            "status": "DEFERRED",
            "reason": (
                "The current task defines label-audited hard pools only. "
                "Pool-internal discrimination must be validated before routing."
            ),
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "source_hard_samples.csv", source_rows)
    write_csv(OUTPUT / "h2_hard_samples_confirmation.csv", h2_rows)
    write_csv(OUTPUT / "source_pair_pools.csv", pair_rows)
    write_csv(OUTPUT / "source_hard_true_classes.csv", class_rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
