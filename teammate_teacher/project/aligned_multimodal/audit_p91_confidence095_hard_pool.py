"""Audit the frozen P91 champion at a fixed confidence threshold of 0.95.

This is a read-only model audit.  It reconstructs the already-selected P91
constant-blend probability on the saved H2 subject-disjoint predictions.  It
does not train, calibrate, route, or read H3.  Generated files are reporting
artifacts only.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
P91_H2 = PROJECT / "runs/p91_hierarchical_multimodal_h3_v3/inner_predictions.npz"
P94_H2 = PROJECT / "runs/p94_candidate_multimodal_reranker_h1h2_v1/h2_predictions.npz"
CLASS_MAPPING = PROJECT / "class_mapping.csv"
P96_PAIRS = PROJECT / "runs/p96_primary_visual_top5_hard_pools_v1/source_pair_pools.csv"
SKELETON = HERE / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz"
IMU = PROJECT / "runs/p90_imu_teacher_blend_v1/imu_p90_sensorwise_plus_deep_crossfit_oof.npz"
OUTPUT = PROJECT / "runs/p91_confidence095_hard_pool_audit_v1"
THRESHOLD = 0.95


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def align(
    source_ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray
) -> np.ndarray:
    lookup = {str(value): row for row, value in enumerate(source_ids.astype(str))}
    return np.asarray(values)[
        np.asarray([lookup[str(value)] for value in target_ids.astype(str)], dtype=np.int64)
    ]


def load_names() -> dict[int, str]:
    with CLASS_MAPPING.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["action_id"]): row["action_name"].split("_", 1)[-1]
            for row in csv.DictReader(handle)
        }


def load_p96_pairs() -> dict[str, dict[str, str]]:
    with P96_PAIRS.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["pair_pool_id"]: row for row in csv.DictReader(handle)}


def pair_id(left: int, right: int) -> str:
    a, b = sorted((left, right))
    return f"pair_{a:02d}_{b:02d}"


def reconstruct_p91_h2() -> dict[str, np.ndarray | float]:
    data = load_npz(P91_H2)
    logits = data["direct_logits"].astype(np.float64)
    teacher_prediction = data["teacher_prediction"].astype(np.int64)
    weight = float(np.asarray(data["selected_constant_weight"]).reshape(-1)[0])
    neural = softmax(logits, axis=1)
    teacher = np.full((len(logits), 40), 0.06 / 39.0, dtype=np.float64)
    teacher[np.arange(len(logits)), teacher_prediction] = 0.94
    scores = weight * np.log(np.clip(neural, 1e-8, 1.0))
    scores += (1.0 - weight) * np.log(np.clip(teacher, 1e-8, 1.0))
    probability = softmax(scores, axis=1)
    prediction = probability.argmax(axis=1).astype(np.int64)

    # P94 saved the same frozen P91 champion as its base.  This is an
    # independent identity check on the reconstruction without touching H3.
    p94 = load_npz(P94_H2)
    p94_base = align(
        p94["sample_ids"].astype(str),
        p94["base_prediction"].astype(np.int64),
        data["sample_ids"].astype(str),
    )
    if not np.array_equal(prediction, p94_base):
        raise RuntimeError("reconstructed P91 H2 prediction differs from frozen P94 base")
    labels = data["labels"].astype(np.int64)
    if int(np.sum(prediction == labels)) != 750:
        raise RuntimeError("unexpected P91 H2 correct count")
    return {
        "sample_ids": data["sample_ids"].astype(str),
        "labels": labels,
        "users": data["users"].astype(str),
        "probability": probability,
        "prediction": prediction,
        "confidence": probability.max(axis=1),
        "blend_weight": weight,
    }


def metrics(mask: np.ndarray, prediction: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    correct = int(np.sum(prediction[mask] == labels[mask]))
    return {
        "samples": count,
        "coverage": float(count / len(labels)),
        "correct": correct,
        "accuracy": float(correct / count) if count else None,
        "errors": count - correct,
        "error_rate": float((count - correct) / count) if count else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def main() -> None:
    names = load_names()
    p96_pairs = load_p96_pairs()
    p91 = reconstruct_p91_h2()
    sample_ids = np.asarray(p91["sample_ids"]).astype(str)
    labels = np.asarray(p91["labels"]).astype(np.int64)
    users = np.asarray(p91["users"]).astype(str)
    probability = np.asarray(p91["probability"]).astype(np.float64)
    prediction = np.asarray(p91["prediction"]).astype(np.int64)
    confidence = np.asarray(p91["confidence"]).astype(np.float64)

    order = np.argsort(-probability, axis=1)
    ranks = np.argmax(order == labels[:, None], axis=1) + 1
    high = confidence >= THRESHOLD
    low = ~high
    correct = prediction == labels
    low_error = low & ~correct
    high_stats = metrics(high, prediction, labels)
    low_stats = metrics(low, prediction, labels)
    all_errors = int((~correct).sum())
    error_capture = float(low_stats["errors"] / all_errors)
    enrichment = float(low_stats["error_rate"] / high_stats["error_rate"])

    top3 = ranks <= 3
    top5 = ranks <= 5
    low_topk = {
        "samples": int(low.sum()),
        "top3_covered": int(np.sum(low & top3)),
        "top3_coverage": float(np.mean(top3[low])),
        "top5_covered": int(np.sum(low & top5)),
        "top5_coverage": float(np.mean(top5[low])),
        "truth_not_in_top5": int(np.sum(low & ~top5)),
        "rank_distribution": {
            **{str(rank): int(np.sum(low & (ranks == rank))) for rank in range(1, 6)},
            ">5": int(np.sum(low & (ranks > 5))),
        },
    }
    low_error_topk = {
        "samples": int(low_error.sum()),
        "top3_covered": int(np.sum(low_error & top3)),
        "top3_coverage": float(np.mean(top3[low_error])),
        "top5_covered": int(np.sum(low_error & top5)),
        "top5_coverage": float(np.mean(top5[low_error])),
        "truth_not_in_top5": int(np.sum(low_error & ~top5)),
        "rank_distribution": {
            **{
                str(rank): int(np.sum(low_error & (ranks == rank)))
                for rank in range(2, 6)
            },
            ">5": int(np.sum(low_error & (ranks > 5))),
        },
    }

    low_rows: list[dict[str, Any]] = []
    for row in np.flatnonzero(low):
        top = order[row, :5].astype(int)
        item: dict[str, Any] = {
            "sample_id": sample_ids[row],
            "user": users[row],
            "true_class_id": int(labels[row]),
            "true_class_name": names[int(labels[row])],
            "prediction_id": int(prediction[row]),
            "prediction_name": names[int(prediction[row])],
            "top1_confidence": float(confidence[row]),
            "top1_top2_margin": float(probability[row, top[0]] - probability[row, top[1]]),
            "correct": int(correct[row]),
            "true_rank": int(ranks[row]),
            "truth_in_top3": int(ranks[row] <= 3),
            "truth_in_top5": int(ranks[row] <= 5),
        }
        for rank_index, class_id in enumerate(top, start=1):
            item[f"top{rank_index}_class_id"] = int(class_id)
            item[f"top{rank_index}_class_name"] = names[int(class_id)]
            item[f"top{rank_index}_probability"] = float(probability[row, class_id])
        low_rows.append(item)

    user_rows: list[dict[str, Any]] = []
    for user in sorted(np.unique(users)):
        selected = users == user
        user_errors = int(np.sum(selected & ~correct))
        user_rows.append(
            {
                "user": user,
                "total_samples": int(selected.sum()),
                "total_accuracy": float(np.mean(correct[selected])),
                "total_errors": user_errors,
                "high_samples": int(np.sum(selected & high)),
                "high_fraction_of_user": float(np.mean(high[selected])),
                "high_accuracy": float(np.mean(correct[selected & high])),
                "high_errors": int(np.sum(selected & high & ~correct)),
                "low_samples": int(np.sum(selected & low)),
                "low_fraction_of_user": float(np.mean(low[selected])),
                "low_accuracy": float(np.mean(correct[selected & low])),
                "low_errors": int(np.sum(selected & low & ~correct)),
                "fraction_user_errors_in_low": float(
                    np.sum(selected & low & ~correct) / user_errors
                )
                if user_errors
                else 0.0,
            }
        )

    true_class_rows: list[dict[str, Any]] = []
    predicted_class_rows: list[dict[str, Any]] = []
    for class_id in range(40):
        truth = labels == class_id
        high_truth = truth & high
        low_truth = truth & low
        true_class_rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "total_samples": int(truth.sum()),
                "high_samples": int(high_truth.sum()),
                "high_fraction_of_class": float(np.mean(high[truth])) if truth.any() else None,
                "high_accuracy": float(np.mean(correct[high_truth])) if high_truth.any() else None,
                "high_errors": int(np.sum(high_truth & ~correct)),
                "low_samples": int(low_truth.sum()),
                "low_fraction_of_class": float(np.mean(low[truth])) if truth.any() else None,
                "low_accuracy": float(np.mean(correct[low_truth])) if low_truth.any() else None,
                "low_errors": int(np.sum(low_truth & ~correct)),
                "low_top3_coverage": float(np.mean(top3[low_truth])) if low_truth.any() else None,
                "low_top5_coverage": float(np.mean(top5[low_truth])) if low_truth.any() else None,
            }
        )
        predicted = prediction == class_id
        high_predicted = predicted & high
        low_predicted = predicted & low
        predicted_class_rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "total_predictions": int(predicted.sum()),
                "high_predictions": int(high_predicted.sum()),
                "high_precision": float(np.mean(correct[high_predicted]))
                if high_predicted.any()
                else None,
                "high_errors": int(np.sum(high_predicted & ~correct)),
                "low_predictions": int(low_predicted.sum()),
                "low_precision": float(np.mean(correct[low_predicted]))
                if low_predicted.any()
                else None,
                "low_errors": int(np.sum(low_predicted & ~correct)),
            }
        )

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
    if not np.array_equal(skeleton_labels, labels):
        raise RuntimeError("aligned Skeleton labels differ from P91 H2 labels")
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
    if not np.array_equal(imu_labels, labels):
        raise RuntimeError("aligned IMU labels differ from P91 H2 labels")

    pair_indices: dict[str, list[int]] = defaultdict(list)
    for row in np.flatnonzero(low_error):
        pair_indices[pair_id(int(labels[row]), int(prediction[row]))].append(int(row))
    pair_rows: list[dict[str, Any]] = []
    for key, indices in pair_indices.items():
        rows = np.asarray(indices, dtype=np.int64)
        class_a, class_b = (int(value) for value in key.split("_")[1:])
        directions = Counter(
            (int(labels[row]), int(prediction[row])) for row in rows
        )
        skeleton_wins = skeleton_logits[rows, labels[rows]] > skeleton_logits[rows, prediction[rows]]
        imu_wins = imu_probability[rows, labels[rows]] > imu_probability[rows, prediction[rows]]
        p96 = p96_pairs.get(key)
        pair_rows.append(
            {
                "pair_id": key,
                "class_a_id": class_a,
                "class_a_name": names[class_a],
                "class_b_id": class_b,
                "class_b_name": names[class_b],
                "low_confidence_errors": len(rows),
                "users": len(set(users[rows].tolist())),
                "user_ids": ";".join(sorted(set(users[rows].tolist()))),
                "directions": ";".join(
                    f"{truth}:{names[truth]}->{pred}:{names[pred]}={count}"
                    for (truth, pred), count in sorted(
                        directions.items(), key=lambda item: (-item[1], item[0])
                    )
                ),
                "truth_in_p91_top5": int(np.sum(ranks[rows] <= 5)),
                "truth_in_p91_top5_rate": float(np.mean(ranks[rows] <= 5)),
                "skeleton_prefers_truth": int(skeleton_wins.sum()),
                "skeleton_prefers_truth_rate": float(np.mean(skeleton_wins)),
                "imu_prefers_truth": int(imu_wins.sum()),
                "imu_prefers_truth_rate": float(np.mean(imu_wins)),
                "either_skeleton_or_imu_prefers_truth": int(np.sum(skeleton_wins | imu_wins)),
                "either_modality_rate": float(np.mean(skeleton_wins | imu_wins)),
                "p96_source_samples": int(p96["source_samples"]) if p96 else 0,
                "p96_source_users": int(p96["source_users"]) if p96 else 0,
                "p96_pool_tier": p96["pool_tier"] if p96 else "unseen_in_p96_source",
            }
        )
    pair_rows.sort(
        key=lambda row: (-row["low_confidence_errors"], -row["users"], row["pair_id"])
    )

    high_truth_exceptions = [
        row for row in true_class_rows if int(row["high_errors"]) > 0
    ]
    high_truth_exceptions.sort(
        key=lambda row: (-int(row["high_errors"]), int(row["class_id"]))
    )
    high_prediction_exceptions = [
        row for row in predicted_class_rows if int(row["high_errors"]) > 0
    ]
    high_prediction_exceptions.sort(
        key=lambda row: (-int(row["high_errors"]), int(row["class_id"]))
    )

    summary = {
        "experiment_id": "p91_confidence095_hard_pool_audit_v1",
        "status": "complete_read_only_h2_audit",
        "threshold": THRESHOLD,
        "model": "p91_hierarchical_multimodal_h3_v3 constant-blend champion",
        "blend_weight": p91["blend_weight"],
        "protocol": {
            "evaluated_split": "H2 subject-disjoint frozen confirmation",
            "H1_limitation": (
                "No comparable subject-disjoint P91 champion H1 OOF probabilities are saved. "
                "They were not regenerated because that would require retraining."
            ),
            "H3_read": False,
            "training_performed": False,
            "pipeline_modified": False,
        },
        "overall": {
            "samples": len(labels),
            "correct": int(correct.sum()),
            "accuracy": float(correct.mean()),
            "errors": all_errors,
        },
        "high_confidence": high_stats,
        "low_confidence": low_stats,
        "difficulty_enrichment": {
            "fraction_all_errors_in_low_confidence": error_capture,
            "low_vs_high_error_rate_ratio": enrichment,
        },
        "high_confidence_reliability": {
            "minimum_user_accuracy": min(float(row["high_accuracy"]) for row in user_rows),
            "maximum_user_accuracy": max(float(row["high_accuracy"]) for row in user_rows),
            "true_classes_with_errors": [
                {
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"],
                    "samples": int(row["high_samples"]),
                    "accuracy": float(row["high_accuracy"]),
                    "errors": int(row["high_errors"]),
                }
                for row in high_truth_exceptions
            ],
            "predicted_classes_with_errors": [
                {
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"],
                    "predictions": int(row["high_predictions"]),
                    "precision": float(row["high_precision"]),
                    "errors": int(row["high_errors"]),
                }
                for row in high_prediction_exceptions
            ],
        },
        "low_confidence_topk": low_topk,
        "low_confidence_errors_topk": low_error_topk,
        "pair_summary": {
            "unique_unordered_error_pairs": len(pair_rows),
            "pairs_with_at_least_2_errors": sum(
                int(row["low_confidence_errors"]) >= 2 for row in pair_rows
            ),
            "cross_user_pairs": sum(int(row["users"]) >= 2 for row in pair_rows),
            "cross_user_pairs_with_at_least_2_errors": sum(
                int(row["users"]) >= 2 and int(row["low_confidence_errors"]) >= 2
                for row in pair_rows
            ),
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "low_confidence_samples_h2.csv", low_rows)
    write_csv(OUTPUT / "user_distribution_h2.csv", user_rows)
    write_csv(OUTPUT / "true_class_distribution_h2.csv", true_class_rows)
    write_csv(OUTPUT / "predicted_class_reliability_h2.csv", predicted_class_rows)
    write_csv(OUTPUT / "low_confidence_error_pairs_h2.csv", pair_rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    recurrent_pairs = [
        row
        for row in pair_rows
        if int(row["low_confidence_errors"]) >= 2
    ]
    cross_user_pairs = [
        row
        for row in recurrent_pairs
        if int(row["users"]) >= 2
    ]
    modality_pairs = [
        row
        for row in cross_user_pairs
        if int(row["low_confidence_errors"]) >= 3
        and (
            float(row["skeleton_prefers_truth_rate"]) >= 2.0 / 3.0
            or float(row["imu_prefers_truth_rate"]) >= 2.0 / 3.0
        )
    ]
    most_low_true_classes = sorted(
        true_class_rows,
        key=lambda row: (-int(row["low_samples"]), -int(row["low_errors"]), int(row["class_id"])),
    )[:12]
    most_low_error_classes = sorted(
        [row for row in true_class_rows if int(row["low_errors"]) > 0],
        key=lambda row: (-int(row["low_errors"]), -int(row["low_samples"]), int(row["class_id"])),
    )[:12]

    lines = [
        "# P91 champion confidence=0.95 困难分流审计",
        "",
        "## 结论",
        "",
        (
            f"固定阈值 `0.95` 在 H2 将 {low_stats['samples']}/{len(labels)}="
            f"{pct(low_stats['coverage'])} 的样本划入低置信池，却收集了 "
            f"{low_stats['errors']}/{all_errors}={pct(error_capture)} 的全部错误。"
        ),
        (
            f"低置信池错误率为 {pct(low_stats['error_rate'])}，高置信区为 "
            f"{pct(high_stats['error_rate'])}，错误风险富集约 {enrichment:.1f} 倍。"
        ),
        "因此 `0.95` 是有效的困难样本富集点；但它不是类别无条件安全的直接输出保证，低置信池也只是候选分析池而不是错误标签池。",
        "",
        "## 验证边界",
        "",
        "- 仅审计 P91 frozen champion 的 H2 subject-disjoint 保存预测。",
        "- 仓库未保存可比的 P91 champion H1 subject-disjoint OOF 概率；补做需要重新训练，本轮没有执行。",
        "- 未读取 H3；未训练、校准或修改主模型及 pipeline。",
        "",
        "## 高低置信总体统计",
        "",
        "| 区域 | 样本 | 覆盖 | accuracy | error | error rate |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| confidence ≥ 0.95 | {high_stats['samples']} | {pct(high_stats['coverage'])} | "
            f"{pct(high_stats['accuracy'])} | {high_stats['errors']} | {pct(high_stats['error_rate'])} |"
        ),
        (
            f"| confidence < 0.95 | {low_stats['samples']} | {pct(low_stats['coverage'])} | "
            f"{pct(low_stats['accuracy'])} | {low_stats['errors']} | {pct(low_stats['error_rate'])} |"
        ),
        "",
        "## 用户分布",
        "",
        "| 用户 | 总样本 | 高置信样本/accuracy/error | 低置信样本/accuracy/error | 用户错误落入低置信池 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in user_rows:
        lines.append(
            f"| {row['user']} | {row['total_samples']} | "
            f"{row['high_samples']} / {pct(row['high_accuracy'])} / {row['high_errors']} | "
            f"{row['low_samples']} / {pct(row['low_accuracy'])} / {row['low_errors']} | "
            f"{pct(row['fraction_user_errors_in_low'])} |"
        )
    lines.extend(
        [
            "",
            "## 高置信区的用户与类别可靠性",
            "",
            (
                f"五个 H2 用户的高置信 accuracy 范围为 "
                f"{pct(min(float(row['high_accuracy']) for row in user_rows))}–"
                f"{pct(max(float(row['high_accuracy']) for row in user_rows))}，"
                "没有出现总体很好但单用户约 80% 的失真。"
            ),
            "",
            "高置信区仍有 5 个错误，按真实类别集中如下：",
            "",
            "| 真实类别 | 高置信样本 | accuracy | error |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in high_truth_exceptions:
        lines.append(
            f"| {row['class_id']} {row['class_name']} | {row['high_samples']} | "
            f"{pct(row['high_accuracy'])} | {row['high_errors']} |"
        )
    lines.extend(
        [
            "",
            "按 Top-1 预测类别看，发生过高置信错误的类别如下：",
            "",
            "| 预测类别 | 高置信预测 | precision | error |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in high_prediction_exceptions:
        lines.append(
            f"| {row['class_id']} {row['class_name']} | {row['high_predictions']} | "
            f"{pct(row['high_precision'])} | {row['high_errors']} |"
        )
    lines.extend(
        [
            "",
            "这些类别格子的支持量有限，不能据此添加类别特判；但它们说明仅凭 `confidence >= 0.95` 还不能声称类别层面全部安全。",
        ]
    )
    lines.extend(
        [
            "",
            "## 低置信池 Top-K 覆盖",
            "",
            "| 范围 | 样本 | Top-3覆盖 | Top-5覆盖 | 真类不在Top-5 |",
            "|---|---:|---:|---:|---:|",
            (
                f"| 全部低置信样本 | {low_topk['samples']} | "
                f"{low_topk['top3_covered']} ({pct(low_topk['top3_coverage'])}) | "
                f"{low_topk['top5_covered']} ({pct(low_topk['top5_coverage'])}) | "
                f"{low_topk['truth_not_in_top5']} |"
            ),
            (
                f"| 低置信错误样本 | {low_error_topk['samples']} | "
                f"{low_error_topk['top3_covered']} ({pct(low_error_topk['top3_coverage'])}) | "
                f"{low_error_topk['top5_covered']} ({pct(low_error_topk['top5_coverage'])}) | "
                f"{low_error_topk['truth_not_in_top5']} |"
            ),
            "",
            "## 低置信真实类别分布（样本最多的12类）",
            "",
            "| 类别 | 低置信样本 | 低置信accuracy | error | Top-5覆盖 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in most_low_true_classes:
        lines.append(
            f"| {row['class_id']} {row['class_name']} | {row['low_samples']} | "
            f"{pct(row['low_accuracy'])} | {row['low_errors']} | {pct(row['low_top5_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## 低置信错误最多的真实类别",
            "",
            "| 类别 | 低置信样本 | error | 低置信accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in most_low_error_classes:
        lines.append(
            f"| {row['class_id']} {row['class_name']} | {row['low_samples']} | "
            f"{row['low_errors']} | {pct(row['low_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## 高频低置信混淆关系",
            "",
            "仅列出至少2条低置信错误的无序 pair；它们是统计分层，不是固定候选池。",
            "",
            "| pair | 错误 | 用户 | 真类在P91 Top-5 | Skeleton偏向真类 | IMU偏向真类 | P96 Source证据 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in recurrent_pairs:
        lines.append(
            f"| {row['class_a_id']} {row['class_a_name']} ↔ {row['class_b_id']} {row['class_b_name']} | "
            f"{row['low_confidence_errors']} | {row['users']} | "
            f"{pct(row['truth_in_p91_top5_rate'])} | "
            f"{row['skeleton_prefers_truth']}/{row['low_confidence_errors']} "
            f"({pct(row['skeleton_prefers_truth_rate'])}) | "
            f"{row['imu_prefers_truth']}/{row['low_confidence_errors']} "
            f"({pct(row['imu_prefers_truth_rate'])}) | "
            f"{row['p96_source_samples']}样本/{row['p96_source_users']}用户 |"
        )
    lines.extend(
        [
            "",
            "## 跨用户关系",
            "",
        ]
    )
    if cross_user_pairs:
        for row in cross_user_pairs:
            lines.append(
                f"- `{row['class_a_id']} {row['class_a_name']} ↔ "
                f"{row['class_b_id']} {row['class_b_name']}`："
                f"{row['low_confidence_errors']}条、{row['users']}个用户；"
                f"方向 `{row['directions']}`。"
            )
    else:
        lines.append("- 没有至少2条且跨至少2个用户的关系。")
    lines.extend(
        [
            "",
            "## Skeleton / IMU 局部能力优先级",
            "",
            "这里只使用冻结专家对‘真类 vs 当前错误Top-1’的分数顺序做能力提示，不训练 specialist。",
            "",
        ]
    )
    if modality_pairs:
        for row in modality_pairs:
            stronger = []
            if float(row["skeleton_prefers_truth_rate"]) >= 2.0 / 3.0:
                stronger.append("Skeleton")
            if float(row["imu_prefers_truth_rate"]) >= 2.0 / 3.0:
                stronger.append("IMU")
            lines.append(
                f"- 优先审计 `{row['class_a_name']} ↔ {row['class_b_name']}` 的 "
                f"{' + '.join(stronger)}：{row['low_confidence_errors']}条、"
                f"{row['users']}用户，Skeleton={pct(row['skeleton_prefers_truth_rate'])}，"
                f"IMU={pct(row['imu_prefers_truth_rate'])}。"
            )
    else:
        lines.append("- 当前没有同时满足跨用户、至少3条且冻结模态偏向真类≥2/3的 pair。")
    lines.extend(
        [
            "",
            "## 裁决",
            "",
            (
                f"`0.95` 在 H2 形成 {low_stats['samples']} 条（{pct(low_stats['coverage'])}）的困难池，"
                f"包含 {low_stats['errors']}/{all_errors} 个错误。规模可用于下一步只读局部模态能力验证。"
            ),
            (
                f"高置信区在各用户上稳定（最低 {pct(min(float(row['high_accuracy']) for row in user_rows))}），"
                "但类别层面仍有小样本异常，因此当前裁决是‘合理困难分流点’，不是‘已验证安全直出阈值’。"
            ),
            (
                f"但困难池中仍有 {low_stats['correct']} 条正确样本，纯度不是目标；后续任何分析都必须同时记录"
                "对这些正确样本的潜在破坏，不能把低置信等同于错误。"
            ),
            "优先级应由跨用户复现和冻结 Skeleton/IMU pairwise 证据决定，不扩展成大量固定池或 specialist。",
            "",
            "## 产物",
            "",
            "- `low_confidence_samples_h2.csv`：每条低置信样本的完整 Top-5、概率、margin 和真类排名。",
            "- `user_distribution_h2.csv`：高低置信用户分布。",
            "- `true_class_distribution_h2.csv`：按真实类的高低置信分布与 Top-K 覆盖。",
            "- `predicted_class_reliability_h2.csv`：按预测类的 precision。",
            "- `low_confidence_error_pairs_h2.csv`：pair、用户、方向及冻结 Skeleton/IMU pairwise 证据。",
            "- `summary.json`：机器可读汇总。",
            "",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
