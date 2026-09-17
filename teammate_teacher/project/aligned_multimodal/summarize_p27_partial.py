from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
RESEARCH_DOCS = REPO_DIR / "docs" / "research"
RUN_DIR = PROJECT_DIR / "runs" / "p27_a"
OUTPUT_DIR = PROJECT_DIR / "runs" / "p27_a_partial_report"
P12_PATH = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
MANIFEST_PATH = PROJECT_DIR / "runs" / "p27_0_audit" / "p27_train_manifest.csv"
HARD_PATH = PROJECT_DIR / "data" / "hard_local_v1" / "hard_action_protocol.json"
REPORT_PATH = (
    RESEARCH_DOCS
    / "20_events_and_sequence"
    / "29_P27-A事件级多模态联合表示实验结果_中止版.md"
)
VARIANTS = ("a0", "a1", "a2")
SMALL_IDS = np.asarray(
    (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39),
    dtype=np.int64,
)
FOCUS_IDS = (9, 10, 19, 21, 22, 24, 25, 26, 37)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def groups(
    labels: np.ndarray,
    predictions: np.ndarray,
    hard_ids: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    return {
        "all": metric(labels, predictions),
        "small": metric(
            labels[np.isin(labels, SMALL_IDS)],
            predictions[np.isin(labels, SMALL_IDS)],
        ),
        "hard": metric(
            labels[np.isin(labels, hard_ids)],
            predictions[np.isin(labels, hard_ids)],
        ),
    }


def load_held(fold: int, variant: str) -> dict[str, np.ndarray]:
    with np.load(
        RUN_DIR / f"fold_{fold}" / variant / "held_outputs.npz",
        allow_pickle=False,
    ) as archive:
        return {
            key: np.asarray(archive[key])
            for key in ("sample_ids", "subjects", "labels", "predictions", "logits")
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hard = json.loads(HARD_PATH.read_text(encoding="utf-8"))
    hard_ids = np.asarray(hard["hard_class_ids"], dtype=np.int64)
    manifest_rows = read_csv(MANIFEST_PATH)
    manifest = {row["sample_id"]: row for row in manifest_rows}
    names = {int(row["class_id"]): row["class_name"] for row in manifest_rows}
    fold0 = {variant: load_held(0, variant) for variant in VARIANTS}
    fold1_a0 = load_held(1, "a0")
    for variant in VARIANTS[1:]:
        for key in ("sample_ids", "subjects", "labels"):
            if not np.array_equal(fold0["a0"][key], fold0[variant][key]):
                raise ValueError(f"fold 0 {variant} differs on {key}")
    with np.load(P12_PATH, allow_pickle=False) as archive:
        p12 = {
            "sample_ids": archive["sample_ids"].astype(str),
            "labels": archive["labels"].astype(np.int64),
            "folds": archive["folds"].astype(np.int64),
            "predictions": archive["final_predictions"].astype(np.int64),
        }
    p12_by_id = {
        sample_id: (int(label), int(prediction), int(fold))
        for sample_id, label, prediction, fold in zip(
            p12["sample_ids"],
            p12["labels"],
            p12["predictions"],
            p12["folds"],
            strict=True,
        )
    }
    summary: dict[str, Any] = {
        "status": "stopped_by_user",
        "protocol": "p27-a-v1-fixed-before-training",
        "formal_completion": {
            "fold_0": ["a0", "a1", "a2"],
            "fold_1": ["a0"],
            "fold_2": [],
        },
        "three_fold_oof_complete": False,
        "gate_decision_valid": False,
        "folds": {},
    }
    rows_class: list[dict[str, Any]] = []
    rows_subject: list[dict[str, Any]] = []
    rows_rescue: list[dict[str, Any]] = []
    reference = fold0["a0"]
    for variant in VARIANTS:
        prediction = fold0[variant]["predictions"]
        summary["folds"].setdefault("0", {})[variant] = groups(
            reference["labels"], prediction, hard_ids
        )
        for class_id in range(40):
            mask = reference["labels"] == class_id
            rows_class.append(
                {
                    "fold": 0,
                    "model": f"P27-{variant.upper()}",
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "samples": int(mask.sum()),
                    "recall": float((prediction[mask] == class_id).mean()),
                    "is_focus": int(class_id in FOCUS_IDS),
                    "is_small": int(class_id in set(SMALL_IDS.tolist())),
                    "is_hard": int(class_id in set(hard_ids.tolist())),
                }
            )
        for subject in sorted(np.unique(reference["subjects"].astype(str))):
            mask = reference["subjects"].astype(str) == subject
            for scope, item in groups(
                reference["labels"][mask], prediction[mask], hard_ids
            ).items():
                rows_subject.append(
                    {
                        "fold": 0,
                        "model": f"P27-{variant.upper()}",
                        "subject": subject,
                        "scope": scope,
                        **item,
                    }
                )
    for variant in ("a1", "a2"):
        candidate = fold0[variant]["predictions"]
        control = reference["predictions"]
        for scope, mask in (
            ("all", np.ones(len(reference["labels"]), dtype=bool)),
            ("small", np.isin(reference["labels"], SMALL_IDS)),
            ("hard", np.isin(reference["labels"], hard_ids)),
        ):
            rescue = mask & (candidate == reference["labels"]) & (
                control != reference["labels"]
            )
            new_error = mask & (candidate != reference["labels"]) & (
                control == reference["labels"]
            )
            rows_rescue.append(
                {
                    "fold": 0,
                    "candidate": f"P27-{variant.upper()}",
                    "reference": "P27-A0",
                    "scope": scope,
                    "rescues": int(rescue.sum()),
                    "new_errors": int(new_error.sum()),
                    "net": int(rescue.sum() - new_error.sum()),
                }
            )
    for fold, held in ((0, reference), (1, fold1_a0)):
        ids = held["sample_ids"].astype(str)
        common = np.asarray([sample_id in p12_by_id for sample_id in ids])
        p12_labels = np.asarray([p12_by_id[sample_id][0] for sample_id in ids[common]])
        p12_predictions = np.asarray(
            [p12_by_id[sample_id][1] for sample_id in ids[common]]
        )
        summary["folds"].setdefault(str(fold), {})["p12_common"] = groups(
            p12_labels, p12_predictions, hard_ids
        )
        if fold == 1:
            summary["folds"]["1"]["a0"] = groups(
                held["labels"], held["predictions"], hard_ids
            )
    event_metrics: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    for fold, variants in ((0, VARIANTS), (1, ("a0",))):
        for variant in variants:
            metrics_path = RUN_DIR / f"fold_{fold}" / variant / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            event_metrics[f"fold_{fold}_{variant}"] = metrics["event_prediction"]
            runtime[f"fold_{fold}_{variant}"] = {
                key: metrics[key]
                for key in (
                    "train_seconds",
                    "parameters",
                    "fp16_size_mib",
                    "peak_cuda_memory_mib",
                    "milliseconds_per_sample",
                )
            }
    summary["event_prediction"] = event_metrics
    summary["runtime"] = runtime
    summary["fold0_rescue_new_error"] = rows_rescue
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_csv(OUTPUT_DIR / "fold0_per_class_recall.csv", rows_class)
    write_csv(OUTPUT_DIR / "fold0_per_subject.csv", rows_subject)
    write_csv(OUTPUT_DIR / "fold0_rescue_new_error.csv", rows_rescue)
    np.savez_compressed(
        OUTPUT_DIR / "partial_oof.npz",
        fold0_sample_ids=reference["sample_ids"],
        fold0_subjects=reference["subjects"],
        fold0_labels=reference["labels"],
        fold0_a0_logits=fold0["a0"]["logits"].astype(np.float16),
        fold0_a1_logits=fold0["a1"]["logits"].astype(np.float16),
        fold0_a2_logits=fold0["a2"]["logits"].astype(np.float16),
        fold1_a0_sample_ids=fold1_a0["sample_ids"],
        fold1_a0_subjects=fold1_a0["subjects"],
        fold1_a0_labels=fold1_a0["labels"],
        fold1_a0_logits=fold1_a0["logits"].astype(np.float16),
    )
    focus_lookup = {
        (row["model"], row["class_id"]): row["recall"] for row in rows_class
    }
    figure, axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(FOCUS_IDS))
    width = 0.26
    for offset, model in enumerate(("P27-A0", "P27-A1", "P27-A2")):
        axis.bar(
            x + (offset - 1) * width,
            [100.0 * focus_lookup[(model, class_id)] for class_id in FOCUS_IDS],
            width=width,
            label=model,
        )
    axis.set_xticks(
        x,
        [names[class_id].split("_", 1)[-1].replace("_", " ") for class_id in FOCUS_IDS],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Fold 0 recall (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fold0_focus_recall.png", dpi=160)
    plt.close(figure)

    hash_paths: list[Path] = []
    for fold, variants in ((0, VARIANTS), (1, ("a0",))):
        for variant in variants:
            directory = RUN_DIR / f"fold_{fold}" / variant
            hash_paths.extend(
                directory / name
                for name in (
                    "history.csv",
                    "held_outputs.npz",
                    "metrics.json",
                    "final.pt",
                    "deployment_fp16.pt",
                )
            )
    hash_paths.extend(
        OUTPUT_DIR / name
        for name in (
            "summary.json",
            "fold0_per_class_recall.csv",
            "fold0_per_subject.csv",
            "fold0_rescue_new_error.csv",
            "partial_oof.npz",
            "fold0_focus_recall.png",
        )
    )
    write_json(
        OUTPUT_DIR / "artifact_hashes.json",
        {
            "status": "partial_after_user_stop",
            "files": [
                {
                    "path": portable_path(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "git_policy": (
                        "manifest_only"
                        if path.suffix == ".pt" or path.stat().st_size > 10 * 1024 * 1024
                        else "eligible"
                    ),
                }
                for path in hash_paths
            ],
        },
    )

    fold0_metrics = summary["folds"]["0"]
    fold1_metrics = summary["folds"]["1"]
    lines = [
        "# P27-A 事件级多模态联合表示实验结果（用户中止版）",
        "",
        "## 状态与有效边界",
        "",
        "- 用户要求停止后，所有 `train_p27_a.py` 进程已终止。",
        "- 完整结果：fold 0 的 A0/A1/A2，以及 fold 1 的 A0。",
        "- 未完成：fold 1 的 A1/A2、fold 2 的 A0/A1/A2、三折合并 OOF 与模型消融。",
        "- 因此预注册的三折门槛不能裁决；下面的 A0/A1/A2 比较只对 fold 0 有效。",
        "",
        "## P27-0",
        "",
        "- 没有发现会使实验无效的数据硬阻塞。真实四模态 parser 并集为 2933 条；Depth 2931、IR 2933、Skeleton 2931、IMU 2863。",
        "- 17 条 filename schema 例外通过“完整时间键优先、trial 内唯一 frame counter 才回退”修复。",
        "- Depth/IR 只支持共享宽 ROI，不支持紧手框或精确像素配准；模型未使用 Skeleton 到图像的伪投影。",
        "- 详细审计见 `docs/research/20_events_and_sequence/28_P27-0事件级多模态联合表示审计.md`。",
        "",
        "## fold 0 严格对比（986 条）",
        "",
        "| 模型 | Overall | Balanced | Macro-F1 | Small | Hard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, key in (("P12（共同样本）", "p12_common"), ("P27-A0", "a0"), ("P27-A1", "a1"), ("P27-A2", "a2")):
        item = fold0_metrics[key]
        lines.append(
            f"| {model} | {pct(item['all']['accuracy'])} | "
            f"{pct(item['all']['balanced_accuracy'])} | "
            f"{pct(item['all']['macro_f1'])} | "
            f"{pct(item['small']['accuracy'])} | {pct(item['hard']['accuracy'])} |"
        )
    a0 = fold0_metrics["a0"]
    a2 = fold0_metrics["a2"]
    lines.extend(
        [
            "",
            f"fold 0 上 A2 相对 A0：Overall {(a2['all']['accuracy']-a0['all']['accuracy'])*100:+.2f} pp，"
            f"Small {(a2['small']['accuracy']-a0['small']['accuracy'])*100:+.2f} pp，"
            f"Hard {(a2['hard']['accuracy']-a0['hard']['accuracy'])*100:+.2f} pp。三项均未达到建议门槛。",
            "",
            "A1 的冻结事件表示分类明显低于 A0；A2 虽比 A1 恢复，但仍未恢复到 A0。fold 0 不能支持“事件学习改善分类泛化”的结论。",
            "",
            "## fold 0 重点类别 recall",
            "",
            "| 类别 | A0 | A1 | A2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for class_id in FOCUS_IDS:
        lines.append(
            f"| {class_id} {names[class_id]} | "
            f"{pct(focus_lookup[('P27-A0', class_id)])} | "
            f"{pct(focus_lookup[('P27-A1', class_id)])} | "
            f"{pct(focus_lookup[('P27-A2', class_id)])} |"
        )
    lines.extend(
        [
            "",
            "Watch TV 与 Play games 在三个 variant 上均为 0；A2 的 Phone call 也降为 0。"
            "当前事件目标没有解决最初指定的静态持物/微小手势类别。",
            "",
            "## fold 1 已完成的 A0（980 条，仅补充）",
            "",
            "| 模型 | Overall | Balanced | Macro-F1 | Small | Hard |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, key in (("P12（共同样本）", "p12_common"), ("P27-A0", "a0")):
        item = fold1_metrics[key]
        lines.append(
            f"| {model} | {pct(item['all']['accuracy'])} | "
            f"{pct(item['all']['balanced_accuracy'])} | "
            f"{pct(item['all']['macro_f1'])} | "
            f"{pct(item['small']['accuracy'])} | {pct(item['hard']['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## 事件目标与机制观察",
            "",
            f"- fold 0 A1：Skeleton 相对均值 MAE 改善 {event_metrics['fold_0_a1']['skeleton']['relative_improvement']*100:+.1f}%，"
            f"shared-motion {event_metrics['fold_0_a1']['shared_motion']['relative_improvement']*100:+.1f}%，"
            f"IMU {event_metrics['fold_0_a1']['imu']['relative_improvement']*100:+.1f}%；"
            f"visual {event_metrics['fold_0_a1']['visual']['relative_improvement']*100:+.1f}%，"
            f"clip {event_metrics['fold_0_a1']['clip']['relative_improvement']*100:+.1f}%。",
            f"- fold 0 A2：Skeleton {event_metrics['fold_0_a2']['skeleton']['relative_improvement']*100:+.1f}%，"
            f"shared-motion {event_metrics['fold_0_a2']['shared_motion']['relative_improvement']*100:+.1f}%，"
            f"IMU {event_metrics['fold_0_a2']['imu']['relative_improvement']*100:+.1f}%；"
            f"visual {event_metrics['fold_0_a2']['visual']['relative_improvement']*100:+.1f}%，"
            f"clip {event_metrics['fold_0_a2']['clip']['relative_improvement']*100:+.1f}%。",
            "- A2 每轮事件权重都触及预注册上限 2.0，但事件梯度仍远小于分类梯度；这是失败机制证据，不在本次中止实验中修改。",
            "",
            "## 实际结构与部署",
            "",
            "- Depth/IR 独立 stem + 单个共享 ResNet18 trunk；Full/Local 共享权重，Local 用 feature-space 宽 ROIAlign。",
            "- Skeleton 保留 12 段身体关系；原始 IMU 5×32×10 经小 TCN，按 `t-1/t/t+1` 受限窗口与视觉/Skeleton 交互。",
            "- 192-D 事件序列为分类主路径，仅保留 32-D 全局上下文旁路。",
            f"- 参数 {runtime['fold_0_a2']['parameters']:,}；理论 FP16 {runtime['fold_0_a2']['fp16_size_mib']:.2f} MiB；"
            f"训练峰值显存 {runtime['fold_0_a2']['peak_cuda_memory_mib']:.1f} MiB，部署大小满足 40 MiB 目标。",
            "",
            "## 协议偏离与产物",
            "",
            "- 已完成 variant 内没有 held early stopping、蒸馏、Thermal/Radar、full-subject refit 或根据 held 调参。",
            "- 唯一未完成项是用户主动中止导致的折数不足；所以不报告完整三折胜负。",
            "- 部分 OOF、逐类、逐 subject、rescue/new-error、图和哈希位于 `aligned_multimodal/runs/p27_a_partial_report/`。",
            "- checkpoint 不提交 Git；路径、字节数和 SHA256 记录在 `artifact_hashes.json`。",
            "",
            "## 当前结论",
            "",
            "P27-0 工程路线可运行且部署规模可行，但已完成的 fold 0 表明当前事件预训练定义没有转化为更好的跨 subject 分类；A1 明显失败，A2 只部分恢复。由于实验被中止，不能把这一单折结果外推为三折最终裁决。",
            "",
            "下一步唯一建议：若以后恢复 P27，只先审计并重定义 visual/clip 弱标签及其尺度，让它们在 outer-train 内明显优于均值基线后，再按原三折协议重启；不要直接加入 Thermal 或 Radar。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hash_manifest_path = OUTPUT_DIR / "artifact_hashes.json"
    hash_manifest = json.loads(hash_manifest_path.read_text(encoding="utf-8"))
    hash_manifest["files"].append(
        {
            "path": portable_path(REPORT_PATH),
            "bytes": REPORT_PATH.stat().st_size,
            "sha256": sha256(REPORT_PATH),
            "git_policy": "eligible",
        }
    )
    write_json(hash_manifest_path, hash_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
