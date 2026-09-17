from __future__ import annotations

import csv
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
RESEARCH_DOCS = REPO_DIR / "docs" / "research"
P12_PATH = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
R0_DIR = PROJECT_DIR / "runs" / "p27_r0_weak_label_audit"
PILOT_DIR = PROJECT_DIR / "runs" / "p27_r_fold0_pilot"
REPORT_PATH = (
    RESEARCH_DOCS
    / "20_events_and_sequence"
    / "32_P27-R困难动作事件弱标签重构与单折验证_基础logits阻塞版.md"
)
SMALL_IDS = np.asarray(
    [1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39],
    dtype=np.int64,
)
HARD_IDS = np.asarray(
    [7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 35, 37, 38, 39],
    dtype=np.int64,
)
FOCUS = {
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(P12_PATH, allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str)
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int64)
        predictions = data["final_predictions"].astype(np.int64)
        imu_present = data["imu_present"].astype(bool)
        thermal_present = data["thermal_present"].astype(bool)
    held = folds == 0
    held_labels = labels[held]
    held_predictions = predictions[held]
    held_ids = sample_ids[held]
    held_subjects = np.asarray([sample_id.split("__")[2] for sample_id in held_ids])
    overall = metrics(held_labels, held_predictions)
    small = np.isin(held_labels, SMALL_IDS)
    hard = np.isin(held_labels, HARD_IDS)
    result = {
        "overall": overall,
        "small": {"samples": int(small.sum()), **metrics(held_labels[small], held_predictions[small])},
        "hard": {"samples": int(hard.sum()), **metrics(held_labels[hard], held_predictions[hard])},
    }
    subject_rows = []
    for subject in sorted(np.unique(held_subjects), key=lambda value: int(value[4:])):
        selected = held_subjects == subject
        values = metrics(held_labels[selected], held_predictions[selected])
        subject_rows.append({"subject": subject, "samples": int(selected.sum()), **values})
    focus_rows = []
    for class_id, class_name in FOCUS.items():
        selected = held_labels == class_id
        focus_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "samples": int(selected.sum()),
                "recall": float(np.mean(held_predictions[selected] == class_id)) if selected.any() else None,
            }
        )
    pattern_rows = []
    held_imu = imu_present[held]
    held_thermal = thermal_present[held]
    for imu_value, thermal_value in ((1, 1), (1, 0), (0, 1), (0, 0)):
        selected = (held_imu == bool(imu_value)) & (held_thermal == bool(thermal_value))
        if selected.any():
            pattern_rows.append(
                {
                    "imu_present": imu_value,
                    "thermal_present": thermal_value,
                    "samples": int(selected.sum()),
                    **metrics(held_labels[selected], held_predictions[selected]),
                }
            )
    write_csv(PILOT_DIR / "p12_per_subject.csv", subject_rows)
    write_csv(PILOT_DIR / "p12_focus_recall.csv", focus_rows)
    write_csv(PILOT_DIR / "p12_missing_pattern.csv", pattern_rows)

    r0_summary = json.loads((R0_DIR / "summary.json").read_text(encoding="utf-8"))
    candidate_rows = list(
        csv.DictReader(
            (R0_DIR / "candidate_label_audit.csv").open(encoding="utf-8-sig", newline="")
        )
    )
    passed_labels = [
        row for row in candidate_rows if row["passed"].strip().lower() == "true"
    ]
    preflight = json.loads(
        (PILOT_DIR / "p12_base_preflight.json").read_text(encoding="utf-8")
    )
    summary = {
        "protocol": "p27-r-fold0-v1-fixed-before-r0-audit",
        "status": "blocked_before_model_training",
        "r0_training_gate_passed": r0_summary["training_gate_passed"],
        "passed_event_targets": r0_summary["passed_targets"],
        "p12_fold0_reference": result,
        "p12_base_preflight": {
            "status": preflight["status"],
            "contaminated_outer_train_rows": preflight["nested_purity_audit"][
                "outer_train_rows_with_held-user-contaminated_source_experts"
            ],
            "outer_train_rows": preflight["outer_split"]["train_rows"],
            "training_started": preflight["training_started"],
        },
        "comparisons": {
            "frozen_p12": result,
            "ce_only_residual": None,
            "event_supervised_residual": None,
            "reason": preflight["blocking_condition"],
        },
        "model_size": {
            "new_trainable_model": None,
            "new_checkpoint": None,
            "note": "Residual model was not instantiated; no new deployment cost was created.",
        },
        "rescue_new_error": None,
        "protocol_deviation": False,
        "verdict": "do_not_enter_full_three_fold",
    }
    summary_path = PILOT_DIR / "blocked_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# P27-R 困难动作事件弱标签重构与单折验证：基础 logits 阻塞版",
        "",
        "## 最终状态",
        "",
        "**P27-R0 完成；fold 0 residual pilot 在模型训练前被协议门阻塞。没有启动 CE-only 或事件监督训练。**",
        "",
        "阻塞不是算力问题，而是严格 outer-fold 纯度问题：当前没有一份 P12 基础 logits 同时满足“冻结的 P12 决策函数、覆盖 fold 0 outer-train 与 held、嵌套 outer-fold-pure”。",
        "",
        "## 1. P27-R0 结论",
        "",
        "- 盲测证据逐行核对通过：116 条结果、43 条人工正确、21 个困难类规则与 21 个 validated exemplars 一致。",
        "- 共审计 2,933 条 P27 union 样本；所有分位数和尺度只在 fold 0 outer-train subjects 上拟合。",
        "- 最终只保留 `left_activity_share` 与 `role_stability`。",
        "- 粗头区阶段、周期/稀疏事件、视觉—IMU 峰值、有符号倾斜、物体宽窄全部淘汰或暂不可用。",
        "",
        "| 通过标签 | held覆盖 | train IQR | subject η² | 均值 MAE | Ridge MAE | 相对改善 | 人工对齐 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in passed_labels:
        lines.append(
            f"| {row['target']} | {100*float(row['held_coverage']):.1f}% | "
            f"{float(row['train_iqr']):.3f} | {float(row['subject_eta_squared']):.3f} | "
            f"{float(row['mean_baseline_mae']):.4f} | {float(row['ridge_mae']):.4f} | "
            f"{100*float(row['ridge_improvement_fraction']):+.1f}% | "
            f"{100*float(row['manual_alignment_fraction']):.1f}% |"
        )
    lines += [
        "",
        "关键反例：Play games/Watch TV 的粗头区伪接近；Pour 的短平滑单峰被判成比 Stir 更周期；宽 ROI 视觉峰与腕部峰在 Pour/Turn pages 上为零一致性。这些反例直接来自 blind-validated 原图抽检，因此没有为了凑结构保留失败标签。",
        "",
        "## 2. P12 基础 logits 阻塞",
        "",
        f"- fold 0 outer-train 共 {preflight['outer_split']['train_rows']} 条，held 共 {preflight['outer_split']['held_rows']} 条。",
        f"- 直接取 `p12_complete_oof` 的 outer-train 行时，{preflight['nested_purity_audit']['outer_train_rows_with_held-user-contaminated_source_experts']}/{preflight['outer_split']['train_rows']} 条来自训练时见过 fold 0 held users 的 fold 1/2 专家。",
        "- fold 0 专家回放 outer-train 虽不含 held users，但得到的是 in-sample logits；其误差与置信度分布不等价于 held OOF，残差学习会失真。",
        "- full18 P12 见过所有 subjects，明确禁止。",
        "- fold-specific RF 与 router 的可调用模型状态未持久化；重拟合仍不能解决 train 侧需要 nested OOF 的问题。",
        "",
        "唯一严格方案是：在 12 个 outer-train subjects 内再做 subject-disjoint inner OOF，重训一套 inner P12 replicas（含 RF、Thermal 校准和 router），用 inner OOF logits 训练 residual，再仅一次评估 fold 0 held。该动作会新增多套 P12 训练，超出“冻结原 P12 专家”的既定含义，因此本轮没有自行扩大协议。",
        "",
        "## 3. fold 0 只读 P12 参考",
        "",
        "| 模型 | Overall | Balanced | Macro-F1 | Small | Hard |",
        "|---|---:|---:|---:|---:|---:|",
        f"| frozen P12 | {100*overall['accuracy']:.2f}% | {100*overall['balanced_accuracy']:.2f}% | {100*overall['macro_f1']:.2f}% | {100*result['small']['accuracy']:.2f}% | {100*result['hard']['accuracy']:.2f}% |",
        "| CE-only residual | N/A | N/A | N/A | N/A | N/A |",
        "| event residual | N/A | N/A | N/A | N/A | N/A |",
        "",
        "### 重点类别",
        "",
        "| 类别 | n | P12 recall |",
        "|---|---:|---:|",
    ]
    for row in focus_rows:
        recall = row["recall"]
        lines.append(
            f"| {row['class_name']} | {row['samples']} | "
            f"{'N/A' if recall is None else f'{100*float(recall):.2f}%'} |"
        )
    lines += [
        "",
        "### 逐 subject",
        "",
        "| subject | n | accuracy | balanced | macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in subject_rows:
        lines.append(
            f"| {row['subject']} | {row['samples']} | {100*row['accuracy']:.2f}% | "
            f"{100*row['balanced_accuracy']:.2f}% | {100*row['macro_f1']:.2f}% |"
        )
    lines += [
        "",
        "缺失模态模式见 `p12_missing_pattern.csv`；由于 residual 未训练，rescue/new-error、zero/shuffle、模型显存和新增推理成本均为 N/A。",
        "",
        "## 4. 协议与产物",
        "",
        "- 协议偏离：**无**。遇到基础 logits 硬阻塞后按预设停止。",
        "- 没有恢复 P27-A、没有加入 Thermal/Radar 新分支、没有使用 full-data 模型、没有覆盖 P11/P12。",
        "- 训练进程：0；新增 checkpoint：0；新增部署参数：0。",
        "",
        "| 文件 | SHA256 |",
        "|---|---|",
    ]
    artifacts = [
        R0_DIR / "summary.json",
        R0_DIR / "candidate_label_audit.csv",
        R0_DIR / "manual_trace_checks.csv",
        PILOT_DIR / "p12_base_preflight.json",
        PILOT_DIR / "p12_per_subject.csv",
        PILOT_DIR / "p12_focus_recall.csv",
        PILOT_DIR / "p12_missing_pattern.csv",
        summary_path,
    ]
    for path in artifacts:
        lines.append(f"| `{path.relative_to(REPO_DIR).as_posix()}` | `{sha256(path)}` |")
    lines += [
        "",
        "## 裁决",
        "",
        "**目前不值得进入完整三折。** R0 证明了两个腕部角色标签可用，但 fold 0 residual 的基础 logits 协议尚未闭环；在 CE-only/event 对照没有合法结果前，不能声称事件监督改善了困难类或跨 subject 泛化。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    manifest_paths = [
        RESEARCH_DOCS
        / "20_events_and_sequence"
        / "30_困难样本与小动作六模态直观分析总结.md",
        RESEARCH_DOCS
        / "20_events_and_sequence"
        / "31_P27-R0困难动作事件弱标签审计.md",
        REPORT_PATH,
        PROJECT_DIR / "configs" / "p27_r_fold0_pilot.json",
        PROJECT_DIR / "p27r_weak_labels.py",
        PROJECT_DIR / "audit_p27_r0.py",
        PROJECT_DIR / "audit_p27r_p12_base_preflight.py",
        PROJECT_DIR / "summarize_p27r_blocked.py",
        R0_DIR / "weak_labels_fold0.npz",
        R0_DIR / "summary.json",
        R0_DIR / "candidate_label_audit.csv",
        R0_DIR / "confusion_pair_audit.csv",
        R0_DIR / "manual_trace_checks.csv",
        R0_DIR / "head_event_traces.png",
        R0_DIR / "motion_shape_traces.png",
        R0_DIR / "visual_imu_alignment_traces.png",
        R0_DIR / "event_prediction_baselines.png",
        R0_DIR / "subject_stability.png",
        PILOT_DIR / "p12_base_preflight.json",
        PILOT_DIR / "blocked_summary.json",
        PILOT_DIR / "p12_per_subject.csv",
        PILOT_DIR / "p12_focus_recall.csv",
        PILOT_DIR / "p12_missing_pattern.csv",
    ]
    artifact_manifest = {
        "protocol": summary["protocol"],
        "training_started": False,
        "checkpoint_files": [],
        "large_files_committed": False,
        "files": [
            {
                "path": path.relative_to(REPO_DIR).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "commit_policy": (
                    "local_only_derived_label_cache"
                    if path.name == "weak_labels_fold0.npz"
                    else "commit"
                ),
            }
            for path in manifest_paths
        ],
    }
    (PILOT_DIR / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
