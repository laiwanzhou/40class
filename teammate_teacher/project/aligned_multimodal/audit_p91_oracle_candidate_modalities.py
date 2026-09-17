"""Candidate-conditioned Skeleton / IMU capability audit for frozen P91 H2.

The audit never trains a model or modifies P91.  It reconstructs P91's saved
H2 probability, takes each low-confidence sample's own Top-5 classes, and
restricts frozen subject-disjoint Skeleton and IMU OOF scores to that dynamic
candidate set.  Skeleton+IMU is a fixed, untuned 0.5/0.5 average after
candidate-wise normalization.  H3 is not read.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax

from audit_p91_confidence095_hard_pool import (
    IMU,
    PROJECT,
    SKELETON,
    align,
    load_names,
    load_npz,
    load_p96_pairs,
    pair_id,
    pct,
    reconstruct_p91_h2,
    write_csv,
)


OUTPUT = PROJECT / "runs/p91_oracle_candidate_modalities_v1"
THRESHOLD = 0.95
TOP_K = 5
MODALITY_ORDER = ("skeleton", "imu", "skeleton_imu")
MODALITY_LABEL = {
    "skeleton": "Skeleton",
    "imu": "IMU",
    "skeleton_imu": "Skeleton+IMU",
}


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def candidate_prediction(candidates: np.ndarray, scores: np.ndarray) -> np.ndarray:
    positions = np.argmax(scores, axis=1)
    return np.take_along_axis(candidates, positions[:, None], axis=1).reshape(-1)


def candidate_rank(
    candidates: np.ndarray, scores: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    score_order = np.argsort(-scores, axis=1)
    ordered_candidates = np.take_along_axis(candidates, score_order, axis=1)
    contained = np.any(ordered_candidates == labels[:, None], axis=1)
    ranks = np.full(len(labels), -1, dtype=np.int64)
    ranks[contained] = (
        np.argmax(ordered_candidates[contained] == labels[contained, None], axis=1) + 1
    )
    return ranks


def direction_text(
    indices: np.ndarray,
    labels: np.ndarray,
    base_prediction: np.ndarray,
    names: dict[int, str],
) -> str:
    counts = Counter((int(labels[row]), int(base_prediction[row])) for row in indices)
    return ";".join(
        f"{truth}:{names[truth]}->{pred}:{names[pred]}={count}"
        for (truth, pred), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    )


def modality_summary(
    key: str,
    prediction: np.ndarray,
    labels: np.ndarray,
    primary_target: np.ndarray,
    candidate_eligible: np.ndarray,
    protected_correct: np.ndarray,
    base_correct: int,
) -> dict[str, Any]:
    correct = prediction == labels
    rescue = int(np.sum(primary_target & correct))
    harm = int(np.sum(protected_correct & ~correct))
    eligible_correct = int(np.sum(candidate_eligible & correct))
    oracle_gate_correct = base_correct + rescue
    low_gate_correct = base_correct + rescue - harm
    total = len(labels)
    return {
        "modality": key,
        "primary_target_samples": int(primary_target.sum()),
        "primary_target_correct": rescue,
        "oracle_candidate_accuracy": safe_rate(rescue, int(primary_target.sum())),
        "rescue": rescue,
        "candidate_eligible_samples": int(candidate_eligible.sum()),
        "candidate_eligible_correct": eligible_correct,
        "candidate_eligible_accuracy": safe_rate(
            eligible_correct, int(candidate_eligible.sum())
        ),
        "protected_correct_samples": int(protected_correct.sum()),
        "harm": harm,
        "harm_rate": safe_rate(harm, int(protected_correct.sum())),
        "net_change_if_all_low_are_reranked": rescue - harm,
        "oracle_error_gate_h2_correct": oracle_gate_correct,
        "oracle_error_gate_h2_accuracy": float(oracle_gate_correct / total),
        "oracle_error_gate_uplift_pp": float(100.0 * rescue / total),
        "all_low_gate_h2_correct": low_gate_correct,
        "all_low_gate_h2_accuracy": float(low_gate_correct / total),
        "all_low_gate_change_pp": float(100.0 * (rescue - harm) / total),
    }


def main() -> None:
    names = load_names()
    p96_pairs = load_p96_pairs()
    p91 = reconstruct_p91_h2()
    sample_ids = np.asarray(p91["sample_ids"]).astype(str)
    labels = np.asarray(p91["labels"]).astype(np.int64)
    users = np.asarray(p91["users"]).astype(str)
    p91_probability = np.asarray(p91["probability"]).astype(np.float64)
    p91_prediction = np.asarray(p91["prediction"]).astype(np.int64)
    confidence = np.asarray(p91["confidence"]).astype(np.float64)

    candidates = np.argsort(-p91_probability, axis=1)[:, :TOP_K]
    p91_candidate_probability = np.take_along_axis(
        p91_probability, candidates, axis=1
    )
    candidate_size = np.asarray(
        [len(np.unique(row)) for row in candidates], dtype=np.int64
    )
    true_in_candidate = np.any(candidates == labels[:, None], axis=1)
    p91_true_rank = np.argmax(
        np.argsort(-p91_probability, axis=1) == labels[:, None], axis=1
    ) + 1
    low = confidence < THRESHOLD
    p91_correct_mask = p91_prediction == labels
    low_error = low & ~p91_correct_mask
    primary_target = low_error & true_in_candidate
    candidate_eligible = low & true_in_candidate
    protected_correct = low & p91_correct_mask
    unreachable_error = low_error & ~true_in_candidate

    skeleton = load_npz(SKELETON)
    skeleton_logits = align(
        skeleton["sample_ids"].astype(str),
        skeleton["skeleton_logits"].astype(np.float64),
        sample_ids,
    )
    skeleton_labels = align(
        skeleton["sample_ids"].astype(str),
        skeleton["labels"].astype(np.int64),
        sample_ids,
    )
    imu = load_npz(IMU)
    imu_probability = align(
        imu["sample_ids"].astype(str),
        imu["probabilities"].astype(np.float64),
        sample_ids,
    )
    imu_labels = align(
        imu["sample_ids"].astype(str),
        imu["labels"].astype(np.int64),
        sample_ids,
    )
    if not np.array_equal(skeleton_labels, labels):
        raise RuntimeError("aligned Skeleton labels differ from P91 H2 labels")
    if not np.array_equal(imu_labels, labels):
        raise RuntimeError("aligned IMU labels differ from P91 H2 labels")

    skeleton_candidate = softmax(
        np.take_along_axis(skeleton_logits, candidates, axis=1), axis=1
    )
    imu_candidate = np.take_along_axis(imu_probability, candidates, axis=1)
    imu_candidate /= np.clip(imu_candidate.sum(axis=1, keepdims=True), 1e-12, None)
    fusion_candidate = 0.5 * skeleton_candidate + 0.5 * imu_candidate
    score_by_modality = {
        "skeleton": skeleton_candidate,
        "imu": imu_candidate,
        "skeleton_imu": fusion_candidate,
    }
    prediction_by_modality = {
        key: candidate_prediction(candidates, scores)
        for key, scores in score_by_modality.items()
    }
    rank_by_modality = {
        key: candidate_rank(candidates, scores, labels)
        for key, scores in score_by_modality.items()
    }

    base_correct = int(p91_correct_mask.sum())
    modality_metrics = {
        key: modality_summary(
            key,
            prediction_by_modality[key],
            labels,
            primary_target,
            candidate_eligible,
            protected_correct,
            base_correct,
        )
        for key in MODALITY_ORDER
    }

    sample_rows: list[dict[str, Any]] = []
    for row in np.flatnonzero(low):
        item: dict[str, Any] = {
            "sample_id": sample_ids[row],
            "user": users[row],
            "true_class_id": int(labels[row]),
            "true_class_name": names[int(labels[row])],
            "p91_prediction_id": int(p91_prediction[row]),
            "p91_prediction_name": names[int(p91_prediction[row])],
            "p91_confidence": float(confidence[row]),
            "p91_correct": int(p91_correct_mask[row]),
            "p91_true_rank": int(p91_true_rank[row]),
            "candidate_size": int(candidate_size[row]),
            "truth_in_candidate": int(true_in_candidate[row]),
            "primary_error_target": int(primary_target[row]),
            "protected_correct": int(protected_correct[row]),
            "unreachable_error": int(unreachable_error[row]),
        }
        for position in range(TOP_K):
            class_id = int(candidates[row, position])
            item[f"candidate_{position + 1}_id"] = class_id
            item[f"candidate_{position + 1}_name"] = names[class_id]
            item[f"candidate_{position + 1}_p91_probability"] = float(
                p91_candidate_probability[row, position]
            )
            item[f"candidate_{position + 1}_skeleton_probability"] = float(
                skeleton_candidate[row, position]
            )
            item[f"candidate_{position + 1}_imu_probability"] = float(
                imu_candidate[row, position]
            )
            item[f"candidate_{position + 1}_fusion_probability"] = float(
                fusion_candidate[row, position]
            )
        for key in MODALITY_ORDER:
            prediction = int(prediction_by_modality[key][row])
            modality_correct = prediction == int(labels[row])
            item[f"{key}_prediction_id"] = prediction
            item[f"{key}_prediction_name"] = names[prediction]
            item[f"{key}_true_rank_in_candidate"] = (
                int(rank_by_modality[key][row]) if true_in_candidate[row] else None
            )
            item[f"{key}_correct"] = int(modality_correct)
            item[f"{key}_rescue"] = int(primary_target[row] and modality_correct)
            item[f"{key}_harm"] = int(protected_correct[row] and not modality_correct)
        sample_rows.append(item)

    user_rows: list[dict[str, Any]] = []
    for user in sorted(np.unique(users)):
        selected = users == user
        item = {
            "user": user,
            "low_samples": int(np.sum(selected & low)),
            "primary_target_samples": int(np.sum(selected & primary_target)),
            "candidate_eligible_samples": int(np.sum(selected & candidate_eligible)),
            "protected_correct_samples": int(np.sum(selected & protected_correct)),
            "unreachable_errors": int(np.sum(selected & unreachable_error)),
        }
        for key in MODALITY_ORDER:
            correct = prediction_by_modality[key] == labels
            rescue = int(np.sum(selected & primary_target & correct))
            harm = int(np.sum(selected & protected_correct & ~correct))
            eligible_correct = int(np.sum(selected & candidate_eligible & correct))
            item[f"{key}_rescue"] = rescue
            item[f"{key}_target_accuracy"] = safe_rate(
                rescue, int(item["primary_target_samples"])
            )
            item[f"{key}_harm"] = harm
            item[f"{key}_harm_rate"] = safe_rate(
                harm, int(item["protected_correct_samples"])
            )
            item[f"{key}_candidate_accuracy"] = safe_rate(
                eligible_correct, int(item["candidate_eligible_samples"])
            )
        user_rows.append(item)

    grouped_pairs: dict[str, list[int]] = defaultdict(list)
    for row in np.flatnonzero(primary_target):
        grouped_pairs[pair_id(int(labels[row]), int(p91_prediction[row]))].append(int(row))
    pair_rows: list[dict[str, Any]] = []
    for key, indices in grouped_pairs.items():
        rows = np.asarray(indices, dtype=np.int64)
        class_a, class_b = (int(value) for value in key.split("_")[1:])
        item: dict[str, Any] = {
            "pair_id": key,
            "class_a_id": class_a,
            "class_a_name": names[class_a],
            "class_b_id": class_b,
            "class_b_name": names[class_b],
            "target_errors": len(rows),
            "users": len(set(users[rows].tolist())),
            "user_ids": ";".join(sorted(set(users[rows].tolist()))),
            "directions": direction_text(rows, labels, p91_prediction, names),
            "truth_rank_2": int(np.sum(p91_true_rank[rows] == 2)),
            "truth_rank_3": int(np.sum(p91_true_rank[rows] == 3)),
            "truth_rank_4": int(np.sum(p91_true_rank[rows] == 4)),
            "truth_rank_5": int(np.sum(p91_true_rank[rows] == 5)),
        }
        for modality in MODALITY_ORDER:
            scores = score_by_modality[modality]
            prediction = prediction_by_modality[modality]
            rescue = int(np.sum(prediction[rows] == labels[rows]))
            true_position = np.argmax(
                candidates[rows] == labels[rows, None], axis=1
            )
            base_position = np.argmax(
                candidates[rows] == p91_prediction[rows, None], axis=1
            )
            pairwise_truth_preferred = int(
                np.sum(
                    scores[rows, true_position]
                    > scores[rows, base_position]
                )
            )
            item[f"{modality}_rescue"] = rescue
            item[f"{modality}_candidate_accuracy"] = safe_rate(rescue, len(rows))
            item[f"{modality}_pairwise_truth_preferred"] = pairwise_truth_preferred
            item[f"{modality}_pairwise_preference_rate"] = safe_rate(
                pairwise_truth_preferred, len(rows)
            )
        p96 = p96_pairs.get(key)
        item["p96_source_samples"] = int(p96["source_samples"]) if p96 else 0
        item["p96_source_users"] = int(p96["source_users"]) if p96 else 0
        item["p96_pool_tier"] = p96["pool_tier"] if p96 else "unseen_in_p96_source"
        pair_rows.append(item)
    pair_rows.sort(key=lambda row: (-row["target_errors"], -row["users"], row["pair_id"]))

    rescue_masks = {
        key: primary_target & (prediction_by_modality[key] == labels)
        for key in MODALITY_ORDER
    }
    harm_masks = {
        key: protected_correct & (prediction_by_modality[key] != labels)
        for key in MODALITY_ORDER
    }
    single_union = rescue_masks["skeleton"] | rescue_masks["imu"]
    complementarity = {
        "skeleton_and_imu_both_rescue": int(
            np.sum(rescue_masks["skeleton"] & rescue_masks["imu"])
        ),
        "skeleton_or_imu_rescue_union": int(np.sum(single_union)),
        "fusion_unique_rescues_beyond_both_singles": int(
            np.sum(
                rescue_masks["skeleton_imu"]
                & ~rescue_masks["skeleton"]
                & ~rescue_masks["imu"]
            )
        ),
        "single_modality_union_rescues_lost_by_fusion": int(
            np.sum(single_union & ~rescue_masks["skeleton_imu"])
        ),
        "fusion_harms": int(np.sum(harm_masks["skeleton_imu"])),
    }

    perfect_candidate_rescue = int(primary_target.sum())
    perfect_candidate_correct = base_correct + perfect_candidate_rescue
    summary = {
        "experiment_id": "p91_oracle_candidate_modalities_v1",
        "status": "complete_frozen_candidate_conditioned_h2_audit",
        "protocol": {
            "evaluated_split": "P91 frozen H2 subject-disjoint confirmation",
            "threshold": THRESHOLD,
            "candidate_source": "per-sample P91 Top-5 probability ranking",
            "candidate_set_is_dynamic_per_sample": True,
            "observed_candidate_size_distribution": {
                str(size): int(np.sum(candidate_size[low] == size))
                for size in sorted(np.unique(candidate_size[low]))
            },
            "skeleton_source": "P89 fixed subject-disjoint OOF logits",
            "imu_source": "P90 subject-disjoint 3-fold OOF probabilities",
            "fusion": "fixed 0.5/0.5 average of candidate-normalized Skeleton and IMU probabilities",
            "fusion_tuned_on_h2": False,
            "H1_limitation": (
                "Comparable P91 champion H1 probabilities were not saved; no H1 candidate set was regenerated because that would require retraining."
            ),
            "H3_read": False,
            "training_performed": False,
            "p91_pipeline_modified": False,
        },
        "sets": {
            "h2_samples": len(labels),
            "p91_h2_correct": base_correct,
            "p91_h2_accuracy": float(base_correct / len(labels)),
            "low_confidence_samples": int(low.sum()),
            "low_confidence_correct_protected": int(protected_correct.sum()),
            "low_confidence_errors": int(low_error.sum()),
            "primary_target_errors_truth_in_top5": int(primary_target.sum()),
            "unreachable_errors_truth_not_in_top5": int(unreachable_error.sum()),
            "all_low_truth_in_top5": int(candidate_eligible.sum()),
        },
        "baselines_and_ceiling": {
            "uniform_random_candidate_accuracy": float(1.0 / TOP_K),
            "p91_accuracy_on_all_candidate_eligible_low_samples": safe_rate(
                int(np.sum(candidate_eligible & p91_correct_mask)),
                int(candidate_eligible.sum()),
            ),
            "perfect_candidate_selector_rescue": perfect_candidate_rescue,
            "perfect_candidate_selector_h2_correct": perfect_candidate_correct,
            "perfect_candidate_selector_h2_accuracy": float(
                perfect_candidate_correct / len(labels)
            ),
            "perfect_candidate_selector_uplift_pp": float(
                100.0 * perfect_candidate_rescue / len(labels)
            ),
        },
        "modalities": modality_metrics,
        "complementarity": complementarity,
        "pair_summary": {
            "unique_target_pairs": len(pair_rows),
            "pairs_with_at_least_2_target_errors": sum(
                int(row["target_errors"]) >= 2 for row in pair_rows
            ),
            "cross_user_pairs": sum(int(row["users"]) >= 2 for row in pair_rows),
        },
    }

    if int(low.sum()) != 351 or int(low_error.sum()) != 79:
        raise RuntimeError("P91 low-confidence set no longer matches the fixed audit")
    if int(primary_target.sum()) != 63 or int(unreachable_error.sum()) != 16:
        raise RuntimeError("unexpected P91 Top-5 target partition")
    if int(candidate_eligible.sum()) != 335 or int(protected_correct.sum()) != 272:
        raise RuntimeError("unexpected candidate-eligible/protected partition")
    if not np.all(candidates[:, 0] == p91_prediction):
        raise RuntimeError("P91 Top-1 differs from first candidate")
    if not np.all(candidate_size == TOP_K):
        raise RuntimeError("duplicate class found in P91 Top-5")
    for scores in score_by_modality.values():
        if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-10):
            raise RuntimeError("candidate probabilities do not sum to one")
    if sum(int(row["target_errors"]) for row in pair_rows) != int(primary_target.sum()):
        raise RuntimeError("pair table does not cover every primary target")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "candidate_conditioned_samples_h2.csv", sample_rows)
    write_csv(OUTPUT / "candidate_conditioned_users_h2.csv", user_rows)
    write_csv(OUTPUT / "candidate_conditioned_pairs_h2.csv", pair_rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    frequent_pairs = [row for row in pair_rows if int(row["target_errors"]) >= 2]
    lines = [
        "# P91 Top-5 oracle candidate-conditioned 模态能力审计",
        "",
        "## 结论",
        "",
        (
            "在63个‘P91低置信、当前预测错误且真实类位于Top-5’的主要目标上，"
            f"Skeleton仅救回{modality_metrics['skeleton']['rescue']}个（{pct(modality_metrics['skeleton']['oracle_candidate_accuracy'])}），"
            f"IMU救回{modality_metrics['imu']['rescue']}个（{pct(modality_metrics['imu']['oracle_candidate_accuracy'])}），"
            f"固定等权Skeleton+IMU救回{modality_metrics['skeleton_imu']['rescue']}个（{pct(modality_metrics['skeleton_imu']['oracle_candidate_accuracy'])}）。"
        ),
        (
            f"三者均未超过5候选的均匀随机参考{pct(1.0 / TOP_K)}。"
            "因此，使用当前冻结Skeleton/IMU教师分数时，缩小到P91 Top-5并没有把困难错误变成容易的局部排序问题。"
        ),
        "",
        "## 验证边界",
        "",
        "- 只读取P91冻结H2预测，并按每个样本自身的Top-5构造动态候选集合；本批样本的实际候选数均为5。",
        "- Skeleton来自P89固定subject-disjoint OOF；IMU来自P90 subject-disjoint三折OOF。",
        "- Skeleton+IMU固定使用候选内归一化概率的0.5/0.5平均，没有在H2选择权重。",
        "- 没有训练probe、specialist、router或最终模型；没有修改P91；没有读取H3。",
        "- 仓库没有保存可比的P91 champion H1概率；本轮不重训生成H1候选。因此这是无参数的H2冻结能力确认，不从H2反向调方案。",
        "",
        "## 测试集合",
        "",
        "| 集合 | 样本数 |",
        "|---|---:|",
        f"| P91 H2 | {len(labels)} |",
        f"| confidence < 0.95 | {int(low.sum())} |",
        f"| 低置信当前错误 | {int(low_error.sum())} |",
        f"| 主要目标：当前错误且真实类在Top-5 | {int(primary_target.sum())} |",
        f"| 当前错误但真实类不在Top-5 | {int(unreachable_error.sum())} |",
        f"| 保护集：低置信但P91当前正确 | {int(protected_correct.sum())} |",
        "",
        "## Candidate-conditioned结果",
        "",
        "主要accuracy只在63个当前错误、真实类在Top-5的样本上计算；rescue等于其中选对的数量。",
        "harm是在272个P91低置信但当前正确的样本上，被候选排序改错的数量。",
        "",
        "| 模态 | 主要目标accuracy | 全部335条候选内accuracy | rescue | oracle错误门控H2上限 | harm | harm rate | 全部低置信直接重排的净变化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in MODALITY_ORDER:
        metrics = modality_metrics[key]
        lines.append(
            f"| {MODALITY_LABEL[key]} | {pct(metrics['oracle_candidate_accuracy'])} | "
            f"{pct(metrics['candidate_eligible_accuracy'])} | "
            f"{metrics['rescue']}/{metrics['primary_target_samples']} | "
            f"{pct(metrics['oracle_error_gate_h2_accuracy'])} "
            f"(+{metrics['oracle_error_gate_uplift_pp']:.2f}pp) | "
            f"{metrics['harm']}/{metrics['protected_correct_samples']} | "
            f"{pct(metrics['harm_rate'])} | {metrics['net_change_if_all_low_are_reranked']} |"
        )
    lines.extend(
        [
            "",
            (
                f"完美Top-5选择器的理论上限是救回{perfect_candidate_rescue}个，"
                f"把H2从{pct(base_correct / len(labels))}提高到{pct(perfect_candidate_correct / len(labels))}（+{100.0 * perfect_candidate_rescue / len(labels):.2f}pp）。"
                "当前冻结模态只实现了其中9–11个，而且这仍假设oracle知道哪些样本当前是错的。"
            ),
            "",
            "若对全部低置信样本直接使用候选排序，Skeleton、IMU、融合分别净损失125、132、112个正确预测；因此当前分数不能安全承担共享reranker。",
            "",
            "## 用户分布",
            "",
            "| 用户 | 主要目标 | Skeleton rescue | IMU rescue | 融合 rescue | Skeleton harm | IMU harm | 融合 harm |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in user_rows:
        lines.append(
            f"| {row['user']} | {row['primary_target_samples']} | "
            f"{row['skeleton_rescue']} ({pct(row['skeleton_target_accuracy'])}) | "
            f"{row['imu_rescue']} ({pct(row['imu_target_accuracy'])}) | "
            f"{row['skeleton_imu_rescue']} ({pct(row['skeleton_imu_target_accuracy'])}) | "
            f"{row['skeleton_harm']} | {row['imu_harm']} | {row['skeleton_imu_harm']} |"
        )
    lines.extend(
        [
            "",
            "没有一个模态在五个用户上表现出稳定优势；救回主要集中于user7和user16，而user18、user19几乎没有被救回。",
            "",
            "## 高频confusion pair分层",
            "",
            "candidate accuracy表示在完整Top-5中选中真实类；括号内是仅比较真实类与P91错误Top-1时，模态偏向真实类的次数。",
            "",
            "| pair | 样本/用户 | Skeleton | IMU | Skeleton+IMU | P96 Source |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frequent_pairs:
        lines.append(
            f"| {row['class_a_id']} {row['class_a_name']} ↔ {row['class_b_id']} {row['class_b_name']} | "
            f"{row['target_errors']}/{row['users']} | "
            f"{row['skeleton_rescue']}/{row['target_errors']} "
            f"(pairwise {row['skeleton_pairwise_truth_preferred']}/{row['target_errors']}) | "
            f"{row['imu_rescue']}/{row['target_errors']} "
            f"(pairwise {row['imu_pairwise_truth_preferred']}/{row['target_errors']}) | "
            f"{row['skeleton_imu_rescue']}/{row['target_errors']} "
            f"(pairwise {row['skeleton_imu_pairwise_truth_preferred']}/{row['target_errors']}) | "
            f"{row['p96_source_samples']}/{row['p96_source_users']} |"
        )
    lines.extend(
        [
            "",
            "## 局部模态信息判断",
            "",
            "- 三个跨用户稳定关系均没有可靠rescue：Drink_water/Take_medicine为0/6，Phone_call/Headphones融合仅1/5，Sweep/Mop为0/2。",
            "- Take_off/Put_on_clothes中Skeleton与融合为2/3；Wipe_hands/Tableware中IMU与融合为2/2，但都只来自单用户，不能升级为稳定结论。",
            "- Put_on_clothes/Body_temperature和Check_time/Body_temperature出现较强pairwise偏好，却在完整Top-5中为0 rescue，说明二分类可分不等于动态候选集合可排对，也不支持建立大量pair specialist。",
            "- 示例中的Read_documents/Turn_pages在本次63条主要目标中没有对应错误，无法估计；相邻的Write/Turn_pages有2条，三路完整Top-5 rescue均为0/2。",
            (
                f"- Skeleton与IMU救回集合的并集为{complementarity['skeleton_or_imu_rescue_union']}个，"
                f"固定融合只救回{modality_metrics['skeleton_imu']['rescue']}个，没有产生任何两路都未救回的独有rescue，"
                f"并丢失{complementarity['single_modality_union_rescues_lost_by_fusion']}个单模态可救样本。"
            ),
            "",
            "## 下一步裁决",
            "",
            "当前结果不支持直接进入共享candidate reranker：主要目标准确率没有超过随机参考，且对原正确样本的harm为45%–53%。",
            "应优先增强或重新验证teacher representation，尤其是Skeleton的手-头/手-躯干时序关系、IMU的方向与接触微动作；先要求这些表示在跨用户稳定pair上产生可重复的候选内优势。",
            "本结论只针对当前冻结Skeleton/IMU教师分数，不等价于原始Skeleton/IMU信号本身没有信息。若后续要区分‘表示弱’还是‘全局分类头弱’，应另行预注册一个H1-only共享线性probe，再冻结到H2；本轮没有实施。",
            "",
            "## 产物",
            "",
            "- `candidate_conditioned_samples_h2.csv`：351条低置信样本的动态Top-5、三路候选概率、预测、rescue与harm。",
            "- `candidate_conditioned_users_h2.csv`：用户分层结果。",
            "- `candidate_conditioned_pairs_h2.csv`：主要目标的pair分层及pairwise/candidate差异。",
            "- `summary.json`：机器可读汇总。",
            "",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
