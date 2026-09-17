from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from p27_data import read_csv
from p27r_weak_labels import (
    GROUP_TARGET_NAMES,
    P27RSignalExtractor,
    build_weak_targets,
    feature_groups,
    fit_fold_parameters,
    save_weak_label_npz,
    sha256,
    write_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
RESEARCH_DOCS = REPO_DIR / "docs" / "research"
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "p27_r_fold0_pilot.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_r0_weak_label_audit"
VALIDATION_DIR = REPO_DIR / "analysis_bundle" / "hard_cases_v1" / "six_modal_validation_v2"
FOCUS_CLASS_NAMES = {
    9: "Pour drinks",
    10: "Stir drinks",
    19: "Phone call",
    21: "Read",
    22: "Turn pages",
    24: "Use phone",
    25: "Watch TV",
    26: "Play games",
    37: "Take medicine",
}
PAIR_DEFINITIONS = [
    (19, 24, "Phone call vs Use phone"),
    (24, 26, "Use phone vs Play games"),
    (25, 26, "Watch TV vs Play games"),
    (25, 24, "Watch TV vs Use phone"),
    (37, 7, "Take medicine vs Eat"),
    (9, 10, "Pour vs Stir"),
    (21, 22, "Read vs Turn pages"),
]


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P27-R fold-0 weak event labels")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manual-checks", type=Path)
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def evidence_audit() -> dict[str, Any]:
    decision = read_csv(VALIDATION_DIR / "decision_rules.csv")
    validated = read_csv(VALIDATION_DIR / "validated_examples.csv")
    blind = read_csv(VALIDATION_DIR / "all_blind_results.csv")
    blind_by_code = {row["code"]: row for row in blind}
    image_files = list(VALIDATION_DIR.rglob("*.jpg")) + list(
        VALIDATION_DIR.rglob("*.png")
    )
    missing_images = []
    available_blind_images = 0
    for row in blind:
        explicit = row["image_file"].strip()
        if explicit:
            found = (VALIDATION_DIR / explicit).is_file()
        else:
            found = any(row["code"] in path.stem for path in image_files)
        available_blind_images += int(found)
        if explicit and not found:
            missing_images.append(row["code"])
    validated_mismatches: list[str] = []
    for row in validated:
        source = blind_by_code.get(row["code"])
        if source is None:
            validated_mismatches.append(f"{row['code']}: missing from all_blind_results")
            continue
        for key in ("sample_id", "true_class_id", "human_top1_id", "human_correct"):
            if row[key] != source[key]:
                validated_mismatches.append(f"{row['code']}: {key} differs")
    summary_path = (
        RESEARCH_DOCS
        / "20_events_and_sequence"
        / "30_困难样本与小动作六模态直观分析总结.md"
    )
    summary_text = summary_path.read_text(encoding="utf-8")
    referenced_ids = sorted(set(re.findall(r"train__c\d{2}__user\d+__[\d-]+", summary_text)))
    known_ids = {row["sample_id"] for row in blind}
    hard_metadata = REPO_DIR / "analysis_bundle" / "hard_cases_v1" / "metadata.csv"
    if hard_metadata.is_file():
        known_ids.update(row["sample_id"] for row in read_csv(hard_metadata))
    missing_references = [sample_id for sample_id in referenced_ids if sample_id not in known_ids]
    referenced_paths = sorted(
        set(re.findall(r"`(analysis_bundle/[^`]+)`", summary_text))
    )
    missing_path_references = [
        value for value in referenced_paths if not (REPO_DIR / value).exists()
    ]
    return {
        "all_blind_rows": len(blind),
        "all_blind_correct": sum(as_bool(row["human_correct"]) for row in blind),
        "unique_codes": len(blind_by_code),
        "decision_rule_classes": len({int(row["class_id"]) for row in decision}),
        "validated_classes": len({int(row["true_class_id"]) for row in validated}),
        "validated_rows": len(validated),
        "available_blind_images": available_blind_images,
        "missing_images": missing_images,
        "validated_mismatches": validated_mismatches,
        "summary_referenced_sample_ids": len(referenced_ids),
        "summary_missing_references": missing_references,
        "summary_referenced_paths": referenced_paths,
        "summary_missing_path_references": missing_path_references,
        "passed": (
            len(blind) == 116
            and sum(as_bool(row["human_correct"]) for row in blind) == 43
            and len(blind_by_code) == len(blind)
            and len(decision) == 21
            and len(validated) == 21
            and not missing_images
            and not validated_mismatches
            and not missing_references
            and len(referenced_paths) >= 2
            and not missing_path_references
        ),
    }


def subject_eta_squared(
    values: np.ndarray, subjects: np.ndarray
) -> float:
    global_mean = float(values.mean())
    total = float(np.square(values - global_mean).sum())
    if total <= 1e-12:
        return 1.0
    between = 0.0
    for subject in np.unique(subjects):
        group = values[subjects == subject]
        between += len(group) * float((group.mean() - global_mean) ** 2)
    return float(between / total)


def ridge_audit(
    features: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
) -> dict[str, float | int]:
    train = valid & (folds != outer_fold)
    held = valid & (folds == outer_fold)
    train_mean = float(values[train].mean())
    baseline = np.full(int(held.sum()), train_mean, dtype=np.float64)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(features[train], values[train])
    predicted = model.predict(features[held])
    baseline_mae = float(mean_absolute_error(values[held], baseline))
    ridge_mae = float(mean_absolute_error(values[held], predicted))
    improvement = (
        float((baseline_mae - ridge_mae) / baseline_mae)
        if baseline_mae > 1e-12
        else 0.0
    )
    return {
        "train_samples": int(train.sum()),
        "held_samples": int(held.sum()),
        "mean_baseline_mae": baseline_mae,
        "ridge_mae": ridge_mae,
        "ridge_improvement_fraction": improvement,
    }


def pair_audit(
    targets: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_fold = folds != outer_fold
    held_fold = folds == outer_fold
    for class_a, class_b, pair_name in PAIR_DEFINITIONS:
        for group, matrix in targets.items():
            if group == "signed_tilt_diagnostic":
                continue
            valid = masks[group] > 0
            train_a = valid & train_fold & (labels == class_a)
            train_b = valid & train_fold & (labels == class_b)
            if train_a.sum() < 3 or train_b.sum() < 3:
                continue
            effects = []
            for column in range(matrix.shape[1]):
                left = matrix[train_a, column]
                right = matrix[train_b, column]
                pooled = float(np.sqrt(0.5 * (left.var() + right.var())))
                effects.append(
                    float((left.mean() - right.mean()) / max(pooled, 1e-6))
                )
            best_column = int(np.argmax(np.abs(effects)))
            direction = np.sign(effects[best_column])
            agreements = []
            held_differences = []
            for subject in np.unique(subjects[held_fold]):
                subject_a = (
                    valid
                    & held_fold
                    & (subjects == subject)
                    & (labels == class_a)
                )
                subject_b = (
                    valid
                    & held_fold
                    & (subjects == subject)
                    & (labels == class_b)
                )
                if subject_a.any() and subject_b.any():
                    difference = float(
                        matrix[subject_a, best_column].mean()
                        - matrix[subject_b, best_column].mean()
                    )
                    held_differences.append(difference)
                    agreements.append(float(np.sign(difference) == direction))
            rows.append(
                {
                    "pair": pair_name,
                    "class_a": class_a,
                    "class_b": class_b,
                    "group": group,
                    "target": GROUP_TARGET_NAMES[group][best_column],
                    "outer_train_effect_size": effects[best_column],
                    "held_subjects_compared": len(agreements),
                    "held_direction_agreement": (
                        float(np.mean(agreements)) if agreements else None
                    ),
                    "held_mean_difference": (
                        float(np.mean(held_differences)) if held_differences else None
                    ),
                }
            )
    return rows


def load_manual_checks(path: Path | None) -> tuple[list[dict[str, str]], dict[str, float]]:
    if path is None or not path.is_file():
        return [], {}
    rows = read_csv(path)
    by_group: dict[str, list[int]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(int(row["aligned"]))
    return rows, {
        group: float(np.mean(values)) for group, values in by_group.items()
    }


def label_audit(
    raw,
    targets: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    config: dict[str, Any],
    manual_fraction: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, str]]:
    outer_fold = int(config["outer_fold"])
    gates = config["r0_gates"]
    rows: list[dict[str, Any]] = []
    passed: dict[str, list[str]] = {}
    group_status: dict[str, str] = {}
    for group, matrix in targets.items():
        valid = masks[group] > 0
        train = valid & (raw.folds != outer_fold)
        held = valid & (raw.folds == outer_fold)
        group_passed: list[str] = []
        for column, name in enumerate(GROUP_TARGET_NAMES[group]):
            values = matrix[:, column]
            iqr = float(np.quantile(values[train], 0.75) - np.quantile(values[train], 0.25))
            eta = subject_eta_squared(values[valid], raw.subjects[valid])
            held_total = int(np.sum(raw.folds == outer_fold))
            held_coverage = float(held.sum() / max(held_total, 1))
            ridge = ridge_audit(
                features[group], values, valid, raw.folds, outer_fold
            )
            structural_reason = ""
            if group == "signed_tilt_diagnostic":
                structural_reason = (
                    "relative quaternion axes have no audited semantic mapping to a "
                    "common pour/tilt direction across device mounting and subjects"
                )
            numerical_pass = (
                int(train.sum()) >= int(gates["minimum_valid_train_samples"])
                and int(held.sum()) >= int(gates["minimum_valid_held_samples"])
                and held_coverage >= float(gates["minimum_held_coverage"])
                and iqr >= float(gates["minimum_nonconstant_iqr"])
                and eta <= float(gates["maximum_subject_eta_squared"])
                and float(ridge["ridge_improvement_fraction"])
                >= float(gates["minimum_ridge_mae_improvement_fraction"])
            )
            manual = manual_fraction.get(group)
            manual_pass = (
                manual is not None
                and manual >= float(gates["minimum_manual_trace_alignment_fraction"])
            )
            final_pass = numerical_pass and manual_pass and not structural_reason
            if final_pass:
                group_passed.append(name)
            rows.append(
                {
                    "group": group,
                    "target": name,
                    "source": config["candidate_groups"].get(
                        group.replace("_diagnostic", ""), {}
                    ).get("source", "imu_quaternion"),
                    "train_valid": int(train.sum()),
                    "held_valid": int(held.sum()),
                    "held_coverage": held_coverage,
                    "train_iqr": iqr,
                    "subject_eta_squared": eta,
                    **ridge,
                    "manual_alignment_fraction": manual,
                    "numerical_pass": numerical_pass,
                    "manual_pass": manual_pass,
                    "passed": final_pass,
                    "reason": structural_reason,
                }
            )
        passed[group] = group_passed
        if group == "signed_tilt_diagnostic":
            group_status[group] = "rejected_semantic_axis_unavailable"
        elif group_passed:
            group_status[group] = "passed"
        elif group not in manual_fraction:
            group_status[group] = "provisional_waiting_manual_trace_audit"
        else:
            group_status[group] = "rejected"
    group_status["object_shape_context"] = (
        "rejected_no_object_segmentation_or_calibrated_depth_ir_object_correspondence"
    )
    return rows, passed, group_status


def plot_trace_panels(
    output: Path,
    raw,
    targets: dict[str, np.ndarray],
    sequences: dict[str, np.ndarray],
) -> None:
    validated = read_csv(VALIDATION_DIR / "validated_examples.csv")
    lookup = {sample_id: index for index, sample_id in enumerate(raw.sample_ids.tolist())}
    selected = [
        row
        for row in validated
        if int(row["true_class_id"]) in {19, 24, 25, 26, 37}
        and row["sample_id"] in lookup
    ]
    fig, axes = plt.subplots(len(selected), 1, figsize=(11, 2.2 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for axis, row in zip(axes, selected, strict=True):
        index = lookup[row["sample_id"]]
        trace = sequences["coarse_head_event"][index]
        for column, name in enumerate(GROUP_TARGET_NAMES["coarse_head_event"]):
            axis.plot(np.linspace(0, 1, 12), trace[:, column], marker="o", label=name)
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel(row["true_class_name"].replace("_", " "))
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("normalized progress")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    fig.suptitle("P27-R0 coarse hand-to-head event traces on blind-validated examples")
    fig.tight_layout()
    fig.savefig(output / "head_event_traces.png", dpi=160)
    plt.close(fig)

    selected = [
        row
        for row in validated
        if int(row["true_class_id"]) in {9, 10, 21, 22, 24, 25, 26, 37}
        and row["sample_id"] in lookup
    ]
    fig, axes = plt.subplots(len(selected), 1, figsize=(11, 2.0 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for axis, row in zip(axes, selected, strict=True):
        index = lookup[row["sample_id"]]
        activity = sequences["imu_wrist_activity"][index]
        axis.plot(np.linspace(0, 1, 32), activity[:, 0], label="left wrist activity")
        axis.plot(np.linspace(0, 1, 32), activity[:, 1], label="right wrist activity")
        motion = targets["motion_shape"][index]
        axis.set_title(
            row["true_class_name"].replace("_", " ")
            + " | periodicity={:.2f}, irregularity={:.2f}, events={:.2f}, interval={:.2f}".format(
                *motion
            ),
            fontsize=9,
        )
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("normalized progress")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("P27-R0 wrist motion shape traces on blind-validated examples")
    fig.tight_layout()
    fig.savefig(output / "motion_shape_traces.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(selected), 1, figsize=(11, 2.0 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for axis, row in zip(axes, selected, strict=True):
        index = lookup[row["sample_id"]]
        visual = sequences["visual_motion"][index]
        imu = sequences["imu_wrist_activity"][index].max(axis=1)
        imu12 = np.interp(np.linspace(0, 31, 12), np.arange(32), imu)
        axis.plot(np.linspace(0, 1, 12), visual, marker="o", label="broad ROI visual motion")
        axis.plot(np.linspace(0, 1, 12), imu12, marker="s", label="wrist IMU activity")
        axis.set_title(
            row["true_class_name"].replace("_", " ")
            + " | soft alignment={:.2f}".format(
                targets["visual_imu_soft_alignment"][index, 2]
            ),
            fontsize=9,
        )
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("normalized progress")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("P27-R0 visual/IMU soft alignment diagnostic")
    fig.tight_layout()
    fig.savefig(output / "visual_imu_alignment_traces.png", dpi=160)
    plt.close(fig)


def plot_audit_summaries(
    output: Path,
    rows: list[dict[str, Any]],
    raw,
    targets: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
) -> None:
    names = [f"{row['group']}:{row['target']}" for row in rows]
    baseline = [float(row["mean_baseline_mae"]) for row in rows]
    ridge = [float(row["ridge_mae"]) for row in rows]
    y = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(11, max(6, 0.35 * len(rows))))
    axis.barh(y - 0.18, baseline, height=0.35, label="outer-train mean")
    axis.barh(y + 0.18, ridge, height=0.35, label="held-subject Ridge")
    axis.set_yticks(y, names, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel("held fold 0 MAE (lower is better)")
    axis.legend()
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "event_prediction_baselines.png", dpi=160)
    plt.close(fig)

    chosen = [
        ("coarse_head_event", 1, "head_zone_dwell"),
        ("hand_roles", 1, "role_stability"),
        ("motion_shape", 0, "periodicity"),
        ("motion_shape", 2, "sparse_event_count"),
        ("visual_imu_soft_alignment", 2, "visual_wrist_peak_alignment"),
    ]
    subjects = sorted(np.unique(raw.subjects).tolist())
    heat = np.full((len(subjects), len(chosen)), np.nan, dtype=np.float32)
    for col, (group, target_index, _) in enumerate(chosen):
        valid = masks[group] > 0
        values = targets[group][:, target_index]
        for row_index, subject in enumerate(subjects):
            selected = valid & (raw.subjects == subject)
            if selected.any():
                heat[row_index, col] = float(values[selected].mean())
        column = heat[:, col]
        finite = np.isfinite(column)
        if finite.any():
            heat[finite, col] = (column[finite] - column[finite].mean()) / max(
                float(column[finite].std()), 1e-6
            )
    fig, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(heat, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
    axis.set_yticks(np.arange(len(subjects)), subjects)
    axis.set_xticks(
        np.arange(len(chosen)), [item[2] for item in chosen], rotation=35, ha="right"
    )
    axis.set_title("Subject mean z-scores (diagnostic; lower uniformity is preferred)")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output / "subject_stability.png", dpi=160)
    plt.close(fig)


def markdown_report(
    config: dict[str, Any],
    evidence: dict[str, Any],
    parameters: dict[str, float],
    audit_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    group_status: dict[str, str],
    passed: dict[str, list[str]],
    manual_rows: list[dict[str, str]],
    finalized: bool,
    artifacts: list[dict[str, Any]],
) -> str:
    lines = [
        "# P27-R0 困难动作事件弱标签审计",
        "",
        f"- 协议：`{config['protocol']}`，outer fold = {config['outer_fold']}。",
        f"- 状态：{'已完成最终人工时序抽检' if finalized else '数值审计完成，等待人工时序抽检后定稿'}。",
        "- 本报告只审计弱标签；未恢复 P27-A、未加入 Thermal/Radar、未训练分类模型。",
        "",
        "## 1. 证据交叉核对",
        "",
        f"- `all_blind_results.csv`：{evidence['all_blind_rows']} 条，人工正确 {evidence['all_blind_correct']} 条。",
        f"- 决策规则/验证类别：{evidence['decision_rule_classes']}/{evidence['validated_classes']} 类。",
        f"- 总结文档引用样本：{evidence['summary_referenced_sample_ids']} 个；图片、行字段和引用一致性：{'通过' if evidence['passed'] else '失败'}。",
        "",
        "盲看证据直接对应本轮原语：单侧手—头接近与停留；抬起—停留—回落；单手持续倾斜与双手周期操作；稀疏点击/翻页；左右手主辅关系；视觉与腕部 IMU 峰值软一致性。",
        "",
        "## 2. fold 0 训练侧拟合参数",
        "",
        "| 参数 | 数值 |",
        "|---|---:|",
    ]
    for key, value in parameters.items():
        lines.append(f"| {key} | {value:.6g} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines += [
        "",
        "所有分位数、尺度和事件 prominence 只由 outer-train subjects 拟合，held subjects 未参与。",
        "",
        "## 3. 候选标签裁决",
        "",
        "| 组 | 状态 | 通过标签 |",
        "|---|---|---|",
    ]
    for group, status in group_status.items():
        lines.append(
            f"| {group} | {status} | {', '.join(passed.get(group, [])) or '—'} |"
        )
    lines += [
        "",
        "| 标签 | held覆盖 | train IQR | subject η² | 均值MAE | Ridge MAE | 改善 | 人工对齐 | 裁决 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in audit_rows:
        manual = (
            "—"
            if row["manual_alignment_fraction"] is None
            else f"{100*float(row['manual_alignment_fraction']):.1f}%"
        )
        lines.append(
            "| {group}/{target} | {coverage:.1f}% | {iqr:.3f} | {eta:.3f} | "
            "{base:.4f} | {ridge:.4f} | {improve:+.1f}% | {manual} | {result} |".format(
                group=row["group"],
                target=row["target"],
                coverage=100 * float(row["held_coverage"]),
                iqr=float(row["train_iqr"]),
                eta=float(row["subject_eta_squared"]),
                base=float(row["mean_baseline_mae"]),
                ridge=float(row["ridge_mae"]),
                improve=100 * float(row["ridge_improvement_fraction"]),
                manual=manual,
                result="通过" if row["passed"] else "淘汰/待定",
            )
        )
    lines += [
        "",
        "### 明确不可用项",
        "",
        "- **单次有符号倾斜**：相对四元数保留变化，但当前没有证明不同设备佩戴方向、左右腕和 subjects 共享同一语义旋转轴；数值可预测不等于“倒入方向”可解释，因此不进入监督。",
        "- **宽平面/窄物体上下文**：现有 ROI 是 person + nearby context，且没有物体分割或 Depth/IR 物体级空间对应；不从背景边缘伪造物体宽窄标签。",
        "- **精确手—嘴/耳接触**：Skeleton 只有粗头部关节，不能称为嘴/耳接触；保留名称严格限定为 coarse head-zone。",
        "",
        "## 4. 重点混淆对支持证据",
        "",
        "| 混淆对 | 最相关标签 | outer-train effect | held subjects | 方向一致率 |",
        "|---|---|---:|---:|---:|",
    ]
    best_pairs: dict[str, dict[str, Any]] = {}
    for row in pair_rows:
        current = best_pairs.get(row["pair"])
        if current is None or abs(float(row["outer_train_effect_size"])) > abs(
            float(current["outer_train_effect_size"])
        ):
            best_pairs[row["pair"]] = row
    for pair, row in best_pairs.items():
        agreement = row["held_direction_agreement"]
        lines.append(
            f"| {pair} | {row['group']}/{row['target']} | "
            f"{float(row['outer_train_effect_size']):+.2f} | {row['held_subjects_compared']} | "
            f"{'—' if agreement is None else f'{100*float(agreement):.1f}%'} |"
        )
    lines += [
        "",
        "这里的真实类别只用于审计解释性和方向稳定性，从未参与弱标签公式、阈值拟合或目标生成。",
        "",
        "## 5. 原始样本人工抽检",
        "",
        f"- 人工检查记录：{len(manual_rows)} 条。",
        "- 时序图：`head_event_traces.png`、`motion_shape_traces.png`、`visual_imu_alignment_traces.png`。",
        "- 基线与 subject 稳定性：`event_prediction_baselines.png`、`subject_stability.png`。",
        "",
        "## 6. R0 训练门",
        "",
    ]
    any_passed = any(passed.get(group) for group in passed if group != "signed_tilt_diagnostic")
    if finalized and any_passed:
        lines.append(
            "**弱标签门通过。** 仅允许通过的标签进入 P27-R fold 0 residual pilot；淘汰项必须保持关闭。"
        )
    elif finalized:
        lines.append("**弱标签门未通过。停止，不启动 P27-R 分类训练。**")
    else:
        lines.append("**尚未作最终训练裁决。必须先完成时序人工抽检。**")
    lines += [
        "",
        "## 7. 产物",
        "",
        "| 文件 | bytes | SHA256 |",
        "|---|---:|---|",
    ]
    for artifact in artifacts:
        lines.append(
            f"| `{artifact['path']}` | {artifact['bytes']} | `{artifact['sha256']}` |"
        )
    lines += [
        "",
        "协议偏离：无。R0 阶段没有使用 held fold 选择阈值、没有启动旧 P27-A、没有训练模型。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    evidence = evidence_audit()
    if not evidence["passed"]:
        raise RuntimeError(f"blind evidence audit failed: {evidence}")
    extractor = P27RSignalExtractor(outer_fold=int(config["outer_fold"]))
    raw = extractor.extract()
    parameters = fit_fold_parameters(raw, config, int(config["outer_fold"]))
    targets, sequences, masks = build_weak_targets(raw, parameters)
    features = feature_groups(raw)

    manual_path = (
        args.manual_checks.resolve()
        if args.manual_checks is not None
        else output / "manual_trace_checks.csv"
    )
    manual_rows, manual_fraction = load_manual_checks(manual_path)
    audit_rows, passed, group_status = label_audit(
        raw, targets, masks, features, config, manual_fraction
    )
    pair_rows = pair_audit(
        targets,
        masks,
        raw.labels,
        raw.subjects,
        raw.folds,
        int(config["outer_fold"]),
    )

    weak_path = output / "weak_labels_fold0.npz"
    save_weak_label_npz(
        weak_path, raw, targets, sequences, masks, parameters
    )
    (output / "evidence_crosscheck.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "fold0_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output / "candidate_label_audit.csv", audit_rows)
    write_csv(output / "confusion_pair_audit.csv", pair_rows)
    plot_trace_panels(output, raw, targets, sequences)
    plot_audit_summaries(output, audit_rows, raw, targets, masks)

    generated = [
        output / "weak_labels_fold0.npz",
        output / "evidence_crosscheck.json",
        output / "fold0_parameters.json",
        output / "candidate_label_audit.csv",
        output / "confusion_pair_audit.csv",
        output / "head_event_traces.png",
        output / "motion_shape_traces.png",
        output / "visual_imu_alignment_traces.png",
        output / "event_prediction_baselines.png",
        output / "subject_stability.png",
    ]
    if manual_path.is_file():
        generated.append(manual_path)
    artifacts = [
        {
            "path": str(path.relative_to(REPO_DIR)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in generated
    ]
    finalized = bool(args.finalize and manual_rows)
    report = markdown_report(
        config,
        evidence,
        parameters,
        audit_rows,
        pair_rows,
        group_status,
        passed,
        manual_rows,
        finalized,
        artifacts,
    )
    report_path = (
        RESEARCH_DOCS
        / "20_events_and_sequence"
        / "31_P27-R0困难动作事件弱标签审计.md"
    )
    report_path.write_text(report, encoding="utf-8")
    summary = {
        "protocol": config["protocol"],
        "outer_fold": int(config["outer_fold"]),
        "evidence": evidence,
        "parameters": parameters,
        "group_status": group_status,
        "passed_targets": passed,
        "manual_trace_checks": len(manual_rows),
        "finalized": finalized,
        "training_gate_passed": bool(
            finalized
            and any(
                passed.get(group)
                for group in passed
                if group != "signed_tilt_diagnostic"
            )
        ),
        "artifacts": artifacts,
        "report": str(report_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
