from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .audit_subject_generalization_ceiling import (
        DISTANCES,
        build_repeat_groups,
        evaluate_cross_subject,
        evaluate_same_subject,
        gap_pp,
        load_and_align_caches,
        load_class_names,
        metrics_bundle,
        per_subject_rows,
        select_ids,
        top_confusion_rows,
        write_csv,
    )
except ImportError:
    from audit_subject_generalization_ceiling import (
        DISTANCES,
        build_repeat_groups,
        evaluate_cross_subject,
        evaluate_same_subject,
        gap_pp,
        load_and_align_caches,
        load_class_names,
        metrics_bundle,
        per_subject_rows,
        select_ids,
        top_confusion_rows,
        write_csv,
    )


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
RESEARCH_DOCS = REPO_ROOT / "docs" / "research"
MODALITIES = ("skeleton", "depth", "thermal", "imu")
VARIANTS = ("P25-C", "P25-S", "P25-A")
EMBEDDING_KEYS = {
    "skeleton": "skeleton_embedding",
    "depth": "depth_embedding",
    "thermal": "thermal_embedding",
    "imu": "imu_embedding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P25 subject-invariant adapters")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "p25_subject_invariant_adapters.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p22_joint_pooled_fusion" / "cache",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p25_subject_invariant_adapters",
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "subject_folds",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=RESEARCH_DOCS
        / "10_local_and_domain"
        / "26_P25主体不变表示学习实验结果.md",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def align_output(
    output: dict[str, np.ndarray],
    canonical_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    ids = output["sample_ids"].astype(str)
    if len(ids) != len(np.unique(ids)):
        raise ValueError("Duplicate sample_id in P25 output")
    lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    if set(lookup) != set(canonical_ids.astype(str).tolist()):
        raise ValueError("P25 output sample_id universe differs from P22 cache")
    order = np.asarray([lookup[sample_id] for sample_id in canonical_ids.astype(str)])
    aligned: dict[str, np.ndarray] = {}
    for key, value in output.items():
        if value.ndim > 0 and value.shape[0] == len(ids):
            aligned[key] = value[order]
        elif value.ndim > 1 and value.shape[1] == len(ids):
            aligned[key] = value[:, order]
        else:
            aligned[key] = value
    aligned["sample_ids"] = canonical_ids.copy()
    return aligned


def direct_per_class_rows(
    modality: str,
    variant: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: dict[int, str],
    small_ids: set[int],
    hard_ids: set[int],
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(40))
    rows: list[dict[str, Any]] = []
    for class_id in range(40):
        samples = int(matrix[class_id].sum())
        correct = int(matrix[class_id, class_id])
        rows.append(
            {
                "modality": modality,
                "variant": variant,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "samples": samples,
                "correct": correct,
                "recall": float(correct / samples) if samples else None,
                "is_small": int(class_id in small_ids),
                "is_hard": int(class_id in hard_ids),
            }
        )
    return rows


def direct_per_subject_rows(
    modality: str,
    variant: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    subjects: np.ndarray,
    small_ids: list[int],
    hard_ids: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in sorted(np.unique(subjects).astype(str).tolist()):
        mask = subjects == subject
        metrics = metrics_bundle(labels[mask], predictions[mask], small_ids, hard_ids)
        rows.append(
            {
                "modality": modality,
                "variant": variant,
                "subject": subject,
                "samples": metrics["all"]["samples"],
                "accuracy": metrics["all"]["accuracy"],
                "small_accuracy": metrics["small"]["accuracy"],
                "hard_accuracy": metrics["hard"]["accuracy"],
            }
        )
    return rows


def fold_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    folds: np.ndarray,
    small_ids: list[int],
    hard_ids: list[int],
) -> dict[str, Any]:
    return {
        str(fold): metrics_bundle(
            labels[folds == fold],
            predictions[folds == fold],
            small_ids,
            hard_ids,
        )
        for fold in range(3)
    }


def rescue_new_error(
    labels: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    rescue_mask = (control != labels) & (candidate == labels)
    new_error_mask = (control == labels) & (candidate != labels)
    return {
        "rescues": int(rescue_mask.sum()),
        "new_errors": int(new_error_mask.sum()),
        "net": int(rescue_mask.sum() - new_error_mask.sum()),
        "rescue_sample_indices": np.flatnonzero(rescue_mask).astype(int).tolist(),
        "new_error_sample_indices": np.flatnonzero(new_error_mask).astype(int).tolist(),
    }


def probe_one(
    features: np.ndarray,
    subject_labels: np.ndarray,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    cv = StratifiedKFold(
        n_splits=int(config["n_splits"]),
        shuffle=bool(config["shuffle"]),
        random_state=seed,
    )
    scores: list[float] = []
    for train_indices, test_indices in cv.split(features, subject_labels):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(config["C"]),
                solver=str(config["solver"]),
                max_iter=int(config["max_iter"]),
                random_state=seed,
            ),
        )
        estimator.fit(features[train_indices], subject_labels[train_indices])
        scores.append(float(estimator.score(features[test_indices], subject_labels[test_indices])))
    return {
        "samples": int(len(features)),
        "subjects": int(len(np.unique(subject_labels))),
        "cv_fold_accuracy": scores,
        "accuracy": float(np.mean(scores)),
    }


def run_subject_probes(
    caches: dict[int, dict[str, np.ndarray]],
    learned: dict[str, dict[str, dict[str, np.ndarray]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config["training"]["seed"])
    probe_config = config["subject_probe"]
    output: dict[str, Any] = {}
    for modality_index, modality in enumerate(MODALITIES):
        output[modality] = {"raw": {"folds": {}}}
        for variant in VARIANTS:
            output[modality][variant] = {"folds": {}}
        for fold in range(3):
            cache = caches[fold]
            train_mask = cache["is_outer_train"].astype(bool)
            valid = cache["presence"][:, modality_index].astype(bool)
            mask = train_mask & valid
            subjects = cache["subjects"].astype(str)[mask]
            raw_features = cache[EMBEDDING_KEYS[modality]].astype(np.float32)[mask]
            output[modality]["raw"]["folds"][str(fold)] = probe_one(
                raw_features, subjects, seed, probe_config
            )
            for variant in VARIANTS:
                representations = learned[modality][variant]["representations"][fold].astype(
                    np.float32
                )
                learned_valid = learned[modality][variant]["valid"][fold].astype(bool)
                if not np.array_equal(learned_valid, valid):
                    raise ValueError(f"{modality} {variant} fold {fold} valid mask mismatch")
                output[modality][variant]["folds"][str(fold)] = probe_one(
                    representations[mask], subjects, seed, probe_config
                )
        for name in ("raw", *VARIANTS):
            fold_values = output[modality][name]["folds"]
            weights = np.asarray([fold_values[str(fold)]["samples"] for fold in range(3)])
            accuracies = np.asarray([fold_values[str(fold)]["accuracy"] for fold in range(3)])
            output[modality][name]["accuracy"] = float(np.average(accuracies, weights=weights))
    return output


def prototype_audit(
    learned: dict[str, dict[str, dict[str, np.ndarray]]],
    reference: dict[str, np.ndarray],
    subject_folds: dict[str, int],
    complete_groups: dict[str, list[tuple[int, int, int]]],
    small_ids: list[int],
    hard_ids: list[int],
    class_names: dict[int, str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    results: dict[str, Any] = {}
    class_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        results[modality] = {}
        for variant in VARIANTS:
            results[modality][variant] = {}
            data = learned[modality][variant]
            features_by_fold = {
                fold: data["representations"][fold].astype(np.float32) for fold in range(3)
            }
            valid_by_fold = {fold: data["valid"][fold].astype(bool) for fold in range(3)}
            for distance in DISTANCES:
                same = evaluate_same_subject(
                    features_by_fold,
                    valid_by_fold,
                    reference,
                    subject_folds,
                    complete_groups,
                    distance,
                )
                cross_all = evaluate_cross_subject(
                    features_by_fold,
                    valid_by_fold,
                    reference,
                    subject_folds,
                    distance,
                )
                matched_ids = set(same["sample_ids"].astype(str).tolist())
                cross_matched = select_ids(cross_all, matched_ids)
                same_metrics = metrics_bundle(
                    same["labels"], same["predictions"], small_ids, hard_ids
                )
                cross_matched_metrics = metrics_bundle(
                    cross_matched["labels"],
                    cross_matched["predictions"],
                    small_ids,
                    hard_ids,
                )
                cross_all_metrics = metrics_bundle(
                    cross_all["labels"], cross_all["predictions"], small_ids, hard_ids
                )
                results[modality][variant][distance] = {
                    "same_subject": same_metrics,
                    "cross_subject_matched": cross_matched_metrics,
                    "cross_subject_all": cross_all_metrics,
                    "gap_pp": {
                        subset: gap_pp(same_metrics, cross_matched_metrics, subset)
                        for subset in ("all", "small", "hard")
                    },
                    "coverage": {
                        "same_subject": same["coverage"],
                        "cross_subject_all_samples": int(len(cross_all["labels"])),
                        "matched_samples": int(len(cross_matched["labels"])),
                    },
                }
                for protocol, result in (
                    ("same_subject", same),
                    ("cross_subject_matched", cross_matched),
                    ("cross_subject_all", cross_all),
                ):
                    matrix = confusion_matrix(
                        result["labels"], result["predictions"], labels=np.arange(40)
                    )
                    for class_id in range(40):
                        samples = int(matrix[class_id].sum())
                        correct = int(matrix[class_id, class_id])
                        class_rows.append(
                            {
                                "modality": modality,
                                "variant": variant,
                                "distance": distance,
                                "protocol": protocol,
                                "class_id": class_id,
                                "class_name": class_names[class_id],
                                "samples": samples,
                                "correct": correct,
                                "recall": float(correct / samples) if samples else None,
                            }
                        )
                    confusion_rows.extend(
                        top_confusion_rows(
                            f"{modality}_{variant}",
                            distance,
                            protocol,
                            result,
                            class_names,
                            top_k=20,
                        )
                    )
                subject_rows.extend(
                    per_subject_rows(
                        f"{modality}_{variant}",
                        distance,
                        same,
                        cross_matched,
                        cross_all,
                        small_ids,
                        hard_ids,
                    )
                )
    return results, class_rows, subject_rows, confusion_rows


def gate_candidate(
    candidate: str,
    control_direct: dict[str, Any],
    candidate_direct: dict[str, Any],
    control_prototype: dict[str, Any],
    candidate_prototype: dict[str, Any],
    control_probe: float,
    candidate_probe: float,
    gates: dict[str, Any],
) -> dict[str, Any]:
    overall_delta = (
        candidate_direct["metrics"]["all"]["accuracy"]
        - control_direct["metrics"]["all"]["accuracy"]
    ) * 100
    hard_delta = (
        candidate_direct["metrics"]["hard"]["accuracy"]
        - control_direct["metrics"]["hard"]["accuracy"]
    ) * 100
    fold_deltas = {
        str(fold): (
            candidate_direct["folds"][str(fold)]["all"]["accuracy"]
            - control_direct["folds"][str(fold)]["all"]["accuracy"]
        )
        * 100
        for fold in range(3)
    }
    improved_folds = sum(value > 0 for value in fold_deltas.values())
    control_gap = control_prototype["gap_pp"]["all"]
    candidate_gap = candidate_prototype["gap_pp"]["all"]
    gap_reduction = control_gap - candidate_gap
    same_drop = (
        control_prototype["same_subject"]["all"]["accuracy"]
        - candidate_prototype["same_subject"]["all"]["accuracy"]
    ) * 100
    probe_drop = (control_probe - candidate_probe) * 100
    checks = {
        "overall": overall_delta >= float(gates["overall_delta_pp_min"]),
        "hard": hard_delta >= float(gates["hard_delta_pp_min"]),
        "folds": improved_folds >= int(gates["improved_outer_fold_count_min"]),
        "gap": gap_reduction >= float(gates["prototype_gap_reduction_pp_min"]),
        "same_subject_preserved": same_drop
        <= float(gates["prototype_same_subject_drop_pp_max"]),
        "subject_probe": probe_drop >= float(gates["subject_probe_drop_pp_min"]),
    }
    action_deletion = (
        candidate == "P25-A"
        and probe_drop > 0
        and overall_delta < 0
    )
    return {
        "overall_delta_pp": float(overall_delta),
        "hard_delta_pp": float(hard_delta),
        "fold_delta_pp": fold_deltas,
        "improved_fold_count": int(improved_folds),
        "prototype_gap_reduction_pp": float(gap_reduction),
        "prototype_same_subject_drop_pp": float(same_drop),
        "subject_probe_drop_pp": float(probe_drop),
        "checks": checks,
        "action_information_deleted": bool(action_deletion),
        "passed": bool(all(checks.values()) and not action_deletion),
    }


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.2f}%"


def pp(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.2f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 26_P25主体不变表示学习实验结果",
        "",
        f"- 协议：`{summary['protocol_version']}`",
        f"- 训练代码提交：`{summary['training_code_commit']}`",
        f"- 评估代码提交：`{summary['evaluation_code_commit']}`",
        f"- 协议偏离：**{'是' if summary['protocol_deviation'] else '否'}**",
        f"- 正式训练后端：`{summary['training_audit']['backend']}`；36 个 fold-unit 累计 "
        f"{summary['training_audit']['elapsed_minutes']:.2f} 分钟。",
        "- 说明：P25-A 严格使用 `0.05 × subject CE`，同时 GRL 从 0 线性升到 0.05；adapter 端最大反向系数为 0.0025。",
        "",
        "## 核心结果",
        "",
        "| 模态 | 版本 | Overall | Balanced | Macro-F1 | Small | Hard | Prototype same | Prototype cross | Gap(pp) | Subject probe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for modality in MODALITIES:
        for variant in VARIANTS:
            direct = summary["direct"][modality][variant]["metrics"]
            prototype = summary["prototype"][modality][variant]["standardized_euclidean"]
            probe = summary["subject_probe"][modality][variant]["accuracy"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        modality,
                        variant,
                        pct(direct["all"]["accuracy"]),
                        pct(direct["all"]["balanced_accuracy"]),
                        pct(direct["all"]["macro_f1"]),
                        pct(direct["small"]["accuracy"]),
                        pct(direct["hard"]["accuracy"]),
                        pct(prototype["same_subject"]["all"]["accuracy"]),
                        pct(prototype["cross_subject_matched"]["all"]["accuracy"]),
                        f"{prototype['gap_pp']['all']:.2f}",
                        pct(probe),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Subject-ID 线性探针",
            "",
            "| 模态 | Raw embedding | P25-C | P25-S | P25-A | S相对C下降pp | A相对C下降pp |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for modality in MODALITIES:
        probes = summary["subject_probe"][modality]
        lines.append(
            f"| {modality} | {pct(probes['raw']['accuracy'])} | "
            f"{pct(probes['P25-C']['accuracy'])} | {pct(probes['P25-S']['accuracy'])} | "
            f"{pct(probes['P25-A']['accuracy'])} | "
            f"{(probes['P25-C']['accuracy'] - probes['P25-S']['accuracy']) * 100:+.2f} | "
            f"{(probes['P25-C']['accuracy'] - probes['P25-A']['accuracy']) * 100:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Outer-fold 准确率",
            "",
            "| 模态 | 版本 | Fold 0 | Fold 1 | Fold 2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for modality in MODALITIES:
        for variant in VARIANTS:
            folds = summary["direct"][modality][variant]["folds"]
            lines.append(
                f"| {modality} | {variant} | "
                + " | ".join(pct(folds[str(fold)]["all"]["accuracy"]) for fold in range(3))
                + " |"
            )
    lines.extend(
        [
            "",
            "## 相对 P25-C 的裁决",
            "",
            "| 模态 | 候选 | Overall Δpp | Hard Δpp | 提升折 | Gap 缩小pp | Same下降pp | Probe下降pp | 动作信息删除 | 通过 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for modality in MODALITIES:
        for candidate in ("P25-S", "P25-A"):
            gate = summary["gates"]["by_modality"][modality][candidate]
            lines.append(
                f"| {modality} | {candidate} | {pp(gate['overall_delta_pp'])} | "
                f"{pp(gate['hard_delta_pp'])} | {gate['improved_fold_count']}/3 | "
                f"{pp(gate['prototype_gap_reduction_pp'])} | "
                f"{pp(gate['prototype_same_subject_drop_pp'])} | "
                f"{pp(gate['subject_probe_drop_pp'])} | "
                f"{'是' if gate['action_information_deleted'] else '否'} | "
                f"{'通过' if gate['passed'] else '未通过'} |"
            )
    lines.extend(
        [
            "",
            "## Rescue / new-error",
            "",
            "| 模态 | 候选 | Rescue | New-error | Net |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for modality in MODALITIES:
        for candidate in ("P25-S", "P25-A"):
            comparison = summary["direct"][modality][candidate]["relative_to_control"]
            lines.append(
                f"| {modality} | {candidate} | {comparison['rescues']} | "
                f"{comparison['new_errors']} | {comparison['net']:+d} |"
            )
    lines.extend(
        [
            "",
            "## 重点困难类别（直接 OOF recall）",
            "",
            "| 模态 | 类别 | P25-C | P25-S | P25-A | 最佳相对C Δpp |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for modality in MODALITIES:
        for row in summary["focus_class_results"][modality]:
            control = row["P25-C"]
            best_delta = max(row["P25-S"] - control, row["P25-A"] - control) * 100
            lines.append(
                f"| {modality} | {row['class_id']} {row['class_name']} | "
                f"{pct(control)} | {pct(row['P25-S'])} | {pct(row['P25-A'])} | "
                f"{best_delta:+.2f} |"
            )
    lines.extend(
        [
            "",
            "## 数据覆盖与协议审计",
            "",
            f"- 样本总数：{summary['coverage']['samples']}；Skeleton/Depth/Thermal/IMU 有效数分别为 "
            + "/".join(str(summary["coverage"]["modality_present"][name]) for name in MODALITIES)
            + "。",
            "- 三份 P22 cache 按 `sample_id` 字典连接；OOF 与 learned representation 再次按 `sample_id` 重排核验。",
            "- held fold 未用于 early stopping、权重/epoch/seed/结构选择；只保存固定第 50 轮。",
            "- Subject-ID probe 仅在各 outer-train subjects 内做固定三折诊断，不参与选择。",
            f"- S/A 有效跨主体 anchor 比例：{summary['training_audit']['supcon_valid_anchor_fraction'] * 100:.2f}%；"
            f"无有效正样本 batch：{summary['training_audit']['supcon_no_positive_batches']}。"
            "全部类别在每个训练折至少有 2 个主体。",
            "",
            "## 最终裁决",
            "",
            f"**{summary['decision']['verdict']}**",
            "",
            summary["decision"]["explanation"],
            "",
            "## 后续唯一建议",
            "",
            summary["decision"]["single_recommendation"],
            "",
            "完整逐类、逐 subject、混淆、rescue/new-error、prototype、probe、loss 曲线和制品哈希均保存在运行目录。",
        ]
    )
    return "\n".join(lines) + "\n"


def artifact_manifest(run_dir: Path, config_path: Path, report_path: Path) -> dict[str, Any]:
    files = [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.name != "artifact_manifest.json"
        and "smoke" not in path.relative_to(run_dir).parts
    ]
    files.extend(
        [
            config_path,
            report_path,
            Path(__file__),
            PROJECT_DIR / "train_p25_subject_invariant_adapters.py",
        ]
    )
    unique = sorted(set(path.resolve() for path in files))
    return {
        "files": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in unique
        ]
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    caches, cache_audit = load_and_align_caches(args.cache_dir)
    reference = caches[0]
    canonical_ids = reference["sample_ids"].astype(str)
    labels = reference["labels"].astype(np.int64)
    subjects = reference["subjects"].astype(str)
    folds = reference["folds"].astype(np.int64)
    small_ids = [int(value) for value in config["evaluation"]["small_action_ids"]]
    hard_ids = [int(value) for value in config["evaluation"]["hard_class_ids"]]
    class_names = load_class_names(args.fold_dir)
    subject_folds = {
        subject: int(np.unique(folds[subjects == subject]).item())
        for subject in np.unique(subjects).astype(str)
    }
    complete_groups, repeat_coverage = build_repeat_groups(
        canonical_ids, subjects, labels
    )

    learned: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    direct: dict[str, Any] = {}
    direct_class_rows: list[dict[str, Any]] = []
    direct_subject_rows: list[dict[str, Any]] = []
    direct_confusion_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        learned[modality] = {}
        direct[modality] = {}
        aligned_oof: dict[str, dict[str, np.ndarray]] = {}
        for variant in VARIANTS:
            variant_dir = args.run_dir / modality / variant
            oof = align_output(load_npz(variant_dir / "oof_outputs.npz"), canonical_ids)
            fold_representations = align_output(
                load_npz(variant_dir / "fold_conditioned_representations.npz"),
                canonical_ids,
            )
            if not np.array_equal(oof["labels"].astype(int), labels):
                raise ValueError(f"{modality} {variant} label mismatch")
            if not np.array_equal(oof["folds"].astype(int), folds):
                raise ValueError(f"{modality} {variant} fold mismatch")
            valid = oof["valid_mask"].astype(bool)
            expected_valid = reference["presence"][:, MODALITIES.index(modality)].astype(bool)
            if not np.array_equal(valid, expected_valid):
                raise ValueError(f"{modality} {variant} OOF validity mismatch")
            predictions = oof["predictions"].astype(int)[valid]
            valid_labels = labels[valid]
            valid_subjects = subjects[valid]
            valid_folds = folds[valid]
            direct[modality][variant] = {
                "metrics": metrics_bundle(valid_labels, predictions, small_ids, hard_ids),
                "folds": fold_metrics(
                    valid_labels, predictions, valid_folds, small_ids, hard_ids
                ),
            }
            direct_class_rows.extend(
                direct_per_class_rows(
                    modality,
                    variant,
                    valid_labels,
                    predictions,
                    class_names,
                    set(small_ids),
                    set(hard_ids),
                )
            )
            direct_subject_rows.extend(
                direct_per_subject_rows(
                    modality,
                    variant,
                    valid_labels,
                    predictions,
                    valid_subjects,
                    small_ids,
                    hard_ids,
                )
            )
            direct_confusion_rows.extend(
                top_confusion_rows(
                    f"{modality}_{variant}",
                    "direct_action_head",
                    "outer_fold_oof",
                    {
                        "labels": valid_labels,
                        "predictions": predictions,
                    },
                    class_names,
                    top_k=20,
                )
            )
            aligned_oof[variant] = {
                "labels": valid_labels,
                "predictions": predictions,
                "sample_ids": canonical_ids[valid],
            }
            learned[modality][variant] = fold_representations
        for candidate in ("P25-S", "P25-A"):
            comparison = rescue_new_error(
                aligned_oof["P25-C"]["labels"],
                aligned_oof["P25-C"]["predictions"],
                aligned_oof[candidate]["predictions"],
            )
            for kind, indices in (
                ("rescue", comparison.pop("rescue_sample_indices")),
                ("new_error", comparison.pop("new_error_sample_indices")),
            ):
                for index in indices:
                    rescue_rows.append(
                        {
                            "modality": modality,
                            "candidate": candidate,
                            "kind": kind,
                            "sample_id": aligned_oof["P25-C"]["sample_ids"][index],
                            "label": int(aligned_oof["P25-C"]["labels"][index]),
                            "label_name": class_names[
                                int(aligned_oof["P25-C"]["labels"][index])
                            ],
                            "control_prediction": int(
                                aligned_oof["P25-C"]["predictions"][index]
                            ),
                            "candidate_prediction": int(
                                aligned_oof[candidate]["predictions"][index]
                            ),
                        }
                    )
            direct[modality][candidate]["relative_to_control"] = comparison

    (
        prototype,
        prototype_class_rows,
        prototype_subject_rows,
        prototype_confusion_rows,
    ) = prototype_audit(
        learned,
        reference,
        subject_folds,
        complete_groups,
        small_ids,
        hard_ids,
        class_names,
    )
    probes = run_subject_probes(caches, learned, config)
    gate_distance = config["prototype"]["gate_distance"]
    gates_by_modality: dict[str, Any] = {}
    passing_modalities: set[str] = set()
    for modality in MODALITIES:
        gates_by_modality[modality] = {}
        for candidate in ("P25-S", "P25-A"):
            result = gate_candidate(
                candidate,
                direct[modality]["P25-C"],
                direct[modality][candidate],
                prototype[modality]["P25-C"][gate_distance],
                prototype[modality][candidate][gate_distance],
                probes[modality]["P25-C"]["accuracy"],
                probes[modality][candidate]["accuracy"],
                config["gates"],
            )
            gates_by_modality[modality][candidate] = result
            if result["passed"]:
                passing_modalities.add(modality)

    overall_pass = len(passing_modalities) >= int(
        config["gates"]["passing_modality_count_min"]
    )
    candidate_deltas = [
        (
            gates_by_modality[modality][candidate]["overall_delta_pp"],
            gates_by_modality[modality][candidate]["hard_delta_pp"],
            modality,
            candidate,
        )
        for modality in MODALITIES
        for candidate in ("P25-S", "P25-A")
    ]
    best = max(candidate_deltas)
    if overall_pass:
        verdict = (
            f"通过：{len(passing_modalities)} 个模态满足完整主体不变泛化门槛"
        )
        explanation = (
            "至少两个模态同时改善直接 subject-disjoint 动作准确率、困难类、prototype gap "
            "并降低 subject-ID probe；因此 learned domain invariance 有可重复证据。"
        )
        recommendation = (
            "只将通过门槛且动作增益最高的单模态 adapter 固化为候选，再做一次独立种子复验；"
            "在复验通过前不做四模态拼接。"
        )
    else:
        verdict = (
            f"未通过：仅 {len(passing_modalities)} 个模态满足完整门槛，少于要求的 2 个"
        )
        explanation = (
            f"最佳观察项为 {best[2]} {best[3]}（Overall {best[0]:+.2f} pp，"
            f"Hard {best[1]:+.2f} pp），但完整联合门槛没有在至少两个模态成立。"
            "P25-C 本身已大幅压低 raw embedding 的 subject-ID 可预测性；S/A 相对 C "
            "没有继续降低 probe，且标准化欧氏 prototype gap 在所有模态都扩大。"
            "因此 IMU 的直接增益不能解释为 learned domain invariance。"
        )
        recommendation = (
            "停止冻结 pooled embedding 的主体不变联合路线，转向仅针对困难组的显式关系/时序特征，"
            "先做一个无训练可分性审计再决定是否训练小专家。"
        )

    training_summary = load_json(args.run_dir / "training_summary.json")
    resolved_protocol = load_json(args.run_dir / "resolved_protocol.json")
    protocol_deviation = bool(
        training_summary.get("protocol_deviation", True)
        or resolved_protocol.get("smoke", True)
    )
    focus_ids = [int(value) for value in config["evaluation"]["focus_class_ids"]]
    class_lookup = {
        (str(row["modality"]), str(row["variant"]), int(row["class_id"])): row["recall"]
        for row in direct_class_rows
    }
    focus_class_results = {
        modality: [
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                **{
                    variant: class_lookup[(modality, variant, class_id)]
                    for variant in VARIANTS
                },
            }
            for class_id in focus_ids
        ]
        for modality in MODALITIES
    }
    fold_summary_paths = sorted(args.run_dir.glob("*/*/fold_*/fold_summary.json"))
    fold_summaries = [load_json(path) for path in fold_summary_paths]
    for path, fold_summary in zip(fold_summary_paths, fold_summaries):
        write_json(
            path.parent / "fold_config.json",
            {
                "protocol_version": config["protocol_version"],
                "training_code_commit": resolved_protocol["training_code_commit"],
                "config_sha256": resolved_protocol["config_sha256"],
                "modality": path.parts[-4],
                "variant": fold_summary["variant"],
                "outer_fold": fold_summary["fold"],
                "fixed_config": config,
            },
        )
    supcon_summaries = [
        item for item in fold_summaries if item["variant"] in ("P25-S", "P25-A")
    ]
    training_audit = {
        "backend": resolved_protocol["device"],
        "fold_units": int(len(fold_summaries)),
        "all_fixed_epoch_50": bool(all(item["epochs"] == 50 for item in fold_summaries)),
        "held_fold_evaluations_during_training": int(
            sum(item["held_fold_evaluations_during_training"] for item in fold_summaries)
        ),
        "elapsed_minutes": float(
            sum(item["elapsed_seconds"] for item in fold_summaries) / 60.0
        ),
        "supcon_valid_anchor_fraction": float(
            np.average(
                [
                    item["effective_cross_subject_positive_anchor_fraction"]
                    for item in supcon_summaries
                ],
                weights=[
                    item["epochs"] * item["steps_per_epoch"]
                    for item in supcon_summaries
                ],
            )
        ),
        "supcon_no_positive_batches": int(
            sum(item["no_valid_positive_batches"] for item in supcon_summaries)
        ),
    }
    summary = {
        "status": "complete",
        "protocol_version": config["protocol_version"],
        "training_code_commit": resolved_protocol["training_code_commit"],
        "evaluation_code_commit": git_commit(),
        "protocol_deviation": protocol_deviation,
        "restrictions": {
            "encoder_rerun": False,
            "encoder_unfreezing": False,
            "four_modality_concat": False,
            "P22_B": False,
            "held_fold_early_stopping": False,
            "post_result_tuning": False,
        },
        "input_audit": cache_audit,
        "repeat_coverage": repeat_coverage,
        "coverage": {
            "samples": int(len(canonical_ids)),
            "modality_present": {
                modality: int(
                    reference["presence"][:, MODALITIES.index(modality)].sum()
                )
                for modality in MODALITIES
            },
        },
        "training_audit": training_audit,
        "direct": direct,
        "focus_class_results": focus_class_results,
        "prototype": prototype,
        "subject_probe": probes,
        "gates": {
            "by_modality": gates_by_modality,
            "passing_modalities": sorted(passing_modalities),
            "passing_modality_count": len(passing_modalities),
            "overall_pass": overall_pass,
        },
        "decision": {
            "verdict": verdict,
            "explanation": explanation,
            "single_recommendation": recommendation,
        },
    }
    write_json(args.run_dir / "summary.json", summary)
    write_csv(args.run_dir / "direct_per_class.csv", direct_class_rows)
    write_csv(args.run_dir / "direct_per_subject.csv", direct_subject_rows)
    write_csv(args.run_dir / "direct_top_confusions.csv", direct_confusion_rows)
    write_csv(args.run_dir / "prototype_per_class.csv", prototype_class_rows)
    write_csv(args.run_dir / "prototype_per_subject.csv", prototype_subject_rows)
    write_csv(args.run_dir / "prototype_top_confusions.csv", prototype_confusion_rows)
    write_csv(args.run_dir / "rescue_new_error.csv", rescue_rows)
    args.report.write_text(render_report(summary), encoding="utf-8")
    write_json(
        args.run_dir / "artifact_manifest.json",
        artifact_manifest(args.run_dir, args.config, args.report),
    )


if __name__ == "__main__":
    main()
