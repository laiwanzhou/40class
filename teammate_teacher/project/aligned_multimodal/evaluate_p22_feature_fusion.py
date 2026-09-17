from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
RESEARCH_DOCS = REPO_ROOT / "docs" / "research"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p22_feature_fusion_model import MODALITY_ORDER, build_p22_model


MODEL_NAMES = ("P22-L", "P22-F", "P22-U")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed P22-A three-fold experiment and all preregistered ablations"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "p22_joint_pooled_fusion.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p22_joint_pooled_fusion" / "cache",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p22_joint_pooled_fusion" / "experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p22_joint_pooled_fusion" / "analysis",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=RESEARCH_DOCS
        / "10_local_and_domain"
        / "23_P22联合多模态Pooled特征融合实验结果.md",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def basic_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float | int]:
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def subset_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_ids: list[int],
) -> dict[str, float | int]:
    mask = np.isin(labels, np.asarray(class_ids))
    return basic_metrics(labels[mask], predictions[mask])


def model_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    folds: np.ndarray,
    subjects: np.ndarray,
    presence: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    small_ids = config["evaluation"]["small_action_ids"]
    hard_ids = config["evaluation"]["hard_class_ids"]
    by_fold = {
        str(fold): basic_metrics(labels[folds == fold], predictions[folds == fold])
        for fold in range(3)
    }
    by_subject = {
        subject: basic_metrics(
            labels[subjects == subject], predictions[subjects == subject]
        )
        for subject in sorted(np.unique(subjects).tolist())
    }
    pattern = np.asarray(
        ["".join(str(int(value)) for value in row) for row in presence]
    )
    by_presence = {
        value: basic_metrics(labels[pattern == value], predictions[pattern == value])
        for value in sorted(np.unique(pattern).tolist())
    }
    return {
        "all": basic_metrics(labels, predictions),
        "small": subset_metrics(labels, predictions, small_ids),
        "hard": subset_metrics(labels, predictions, hard_ids),
        "per_fold": by_fold,
        "per_subject": by_subject,
        "per_presence_pattern": by_presence,
    }


def per_class_recall(
    labels: np.ndarray, predictions: np.ndarray
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(40))
    denominators = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        denominators,
        out=np.zeros(40, dtype=np.float64),
        where=denominators > 0,
    )
    return [
        {
            "class_id": class_id,
            "samples": int(denominators[class_id]),
            "recall": float(recalls[class_id]),
            "correct": int(matrix[class_id, class_id]),
        }
        for class_id in range(40)
    ]


def top_bidirectional_pairs(
    labels: np.ndarray, predictions: np.ndarray, count: int = 20
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(40))
    pairs: list[dict[str, Any]] = []
    for first in range(40):
        for second in range(first + 1, 40):
            forward = int(matrix[first, second])
            reverse = int(matrix[second, first])
            if forward + reverse:
                pairs.append(
                    {
                        "class_a": first,
                        "class_b": second,
                        "a_to_b": forward,
                        "b_to_a": reverse,
                        "total": forward + reverse,
                    }
                )
    return sorted(
        pairs,
        key=lambda item: (-int(item["total"]), int(item["class_a"]), int(item["class_b"])),
    )[:count]


def subject_bootstrap(
    labels: np.ndarray,
    candidate_predictions: np.ndarray,
    reference_predictions: np.ndarray,
    subjects: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    unique_subjects = np.unique(subjects)
    counts = np.asarray([(subjects == subject).sum() for subject in unique_subjects])
    candidate_correct = np.asarray(
        [
            ((candidate_predictions == labels) & (subjects == subject)).sum()
            for subject in unique_subjects
        ]
    )
    reference_correct = np.asarray(
        [
            ((reference_predictions == labels) & (subjects == subject)).sum()
            for subject in unique_subjects
        ]
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        selected = rng.integers(0, len(unique_subjects), size=len(unique_subjects))
        denominator = counts[selected].sum()
        deltas[repeat] = (
            candidate_correct[selected].sum() - reference_correct[selected].sum()
        ) / denominator
    observed = float(
        (candidate_predictions == labels).mean()
        - (reference_predictions == labels).mean()
    )
    return {
        "delta_pp": observed * 100.0,
        "subject_cluster_bootstrap_95_ci_pp": [
            float(np.quantile(deltas, 0.025) * 100.0),
            float(np.quantile(deltas, 0.975) * 100.0),
        ],
        "probability_delta_positive": float((deltas > 0).mean()),
        "subjects": int(len(unique_subjects)),
        "repeats": repeats,
        "seed": seed,
    }


def win_loss(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, int]:
    candidate_correct = candidate == labels
    reference_correct = reference == labels
    return {
        "rescue": int((candidate_correct & ~reference_correct).sum()),
        "new_error": int((~candidate_correct & reference_correct).sum()),
        "both_correct": int((candidate_correct & reference_correct).sum()),
        "both_wrong": int((~candidate_correct & ~reference_correct).sum()),
    }


def cache_tensors(
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    embeddings = {
        modality: torch.from_numpy(
            cache[f"{modality}_embedding"][indices].astype(np.float32)
        ).to(device)
        for modality in MODALITY_ORDER
    }
    logits = torch.from_numpy(
        cache["per_modality_logits"][indices].astype(np.float32)
    ).to(device)
    presence = torch.from_numpy(
        cache["presence"][indices].astype(np.float32)
    ).to(device)
    return embeddings, logits, presence


def infer_arrays(
    model: torch.nn.Module,
    embeddings_np: dict[str, np.ndarray],
    logits_np: np.ndarray,
    presence_np: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(presence_np), batch_size):
            stop = min(start + batch_size, len(presence_np))
            embeddings = {
                modality: torch.from_numpy(value[start:stop].astype(np.float32)).to(device)
                for modality, value in embeddings_np.items()
            }
            logits = torch.from_numpy(
                logits_np[start:stop].astype(np.float32)
            ).to(device)
            presence = torch.from_numpy(
                presence_np[start:stop].astype(np.float32)
            ).to(device)
            outputs.append(
                model(embeddings, logits, presence).detach().cpu().numpy()
            )
    return np.concatenate(outputs).astype(np.float32)


def ablation_inputs(
    cache: dict[str, np.ndarray],
    held_indices: np.ndarray,
    modality: str,
    action: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    embeddings = {
        name: cache[f"{name}_embedding"][held_indices].astype(np.float32).copy()
        for name in MODALITY_ORDER
    }
    logits = cache["per_modality_logits"][held_indices].astype(np.float32).copy()
    presence = cache["presence"][held_indices].astype(np.float32).copy()
    modality_index = MODALITY_ORDER.index(modality)
    if action == "zero":
        presence[:, modality_index] = 0.0
        embeddings[modality].fill(0.0)
        logits[:, modality_index].fill(0.0)
    elif action == "shuffle":
        present_rows = np.flatnonzero(presence[:, modality_index] > 0)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(present_rows)
        original_embedding = embeddings[modality].copy()
        original_logits = logits[:, modality_index].copy()
        embeddings[modality][present_rows] = original_embedding[shuffled]
        logits[present_rows, modality_index] = original_logits[shuffled]
    else:
        raise ValueError(action)
    return embeddings, logits, presence


def run_ablations(
    config: dict[str, Any],
    cache_dir: Path,
    experiment_dir: Path,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    batch_size = int(config["training"]["batch_size"])
    shuffle_seed = int(config["evaluation"]["shuffle_seed"])
    ablation_logits = {
        f"{model}_{action}_{modality}": np.zeros((2914, 40), dtype=np.float32)
        for model in MODEL_NAMES
        for action in ("zero", "shuffle")
        for modality in MODALITY_ORDER
    }
    started = time.time()
    for fold in range(3):
        cache = load_npz(cache_dir / f"fold_{fold}_cache.npz")
        held_indices = np.flatnonzero(cache["folds"] == fold)
        for model_name in MODEL_NAMES:
            checkpoint = torch.load(
                experiment_dir / model_name / f"fold_{fold}" / "final.pt",
                map_location="cpu",
                weights_only=False,
            )
            model = build_p22_model(
                model_name,
                projection_dim=int(config["models"]["P22-F"]["projection_dim"]),
                hidden_dim=int(config["models"]["P22-F"]["hidden_dim"]),
                dropout=float(config["models"]["P22-F"]["dropout"]),
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            model.requires_grad_(False)
            model = model.to(device)
            for action in ("zero", "shuffle"):
                for modality_index, modality in enumerate(MODALITY_ORDER):
                    embeddings, logits, presence = ablation_inputs(
                        cache,
                        held_indices,
                        modality,
                        action,
                        shuffle_seed + fold * 100 + modality_index,
                    )
                    key = f"{model_name}_{action}_{modality}"
                    ablation_logits[key][held_indices] = infer_arrays(
                        model,
                        embeddings,
                        logits,
                        presence,
                        batch_size,
                        device,
                    )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"ablation fold={fold} complete", flush=True)
    return ablation_logits, {"elapsed_seconds": time.time() - started}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    summary: dict[str, Any],
) -> None:
    metrics = summary["metrics"]
    comparisons = summary["comparisons"]
    gates = summary["gates"]
    lines = [
        "# 23_P22联合多模态Pooled特征融合实验结果",
        "",
        f"**协议：** `{summary['protocol']}`  ",
        f"**阶段一代码提交：** `{summary['stage_one_code_commit']}`  ",
        f"**协议偏离：** `{'是' if summary['protocol_deviation'] else '否'}`  ",
        "**训练方式：** 三折 outer-fold、固定末轮、无 held-fold early stopping。",
        "",
        "## 核心结果",
        "",
        "| 模型 | Overall | Balanced | Macro-F1 | 小动作 | 困难类 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        item = metrics[model_name]
        lines.append(
            f"| {model_name} | {item['all']['accuracy']*100:.2f}% | "
            f"{item['all']['balanced_accuracy']*100:.2f}% | "
            f"{item['all']['macro_f1']*100:.2f}% | "
            f"{item['small']['accuracy']*100:.2f}% | "
            f"{item['hard']['accuracy']*100:.2f}% |"
        )
    p12 = metrics["P12"]
    lines.append(
        f"| P12 | {p12['all']['accuracy']*100:.2f}% | "
        f"{p12['all']['balanced_accuracy']*100:.2f}% | "
        f"{p12['all']['macro_f1']*100:.2f}% | "
        f"{p12['small']['accuracy']*100:.2f}% | "
        f"{p12['hard']['accuracy']*100:.2f}% |"
    )
    lines += [
        "",
        "## 机制与替代裁决",
        "",
        f"- P22-F 相对 P22-L overall："
        f"`{comparisons['P22-F_vs_P22-L']['all_accuracy_delta_pp']:+.2f} pp`。",
        f"- P22-U 相对 P22-F overall："
        f"`{comparisons['P22-U_vs_P22-F']['all_accuracy_delta_pp']:+.2f} pp`。",
        f"- P22-F 相对 P12 overall："
        f"`{comparisons['P22-F_vs_P12']['all_accuracy_delta_pp']:+.2f} pp`。",
        f"- P22-U 相对 P12 overall："
        f"`{comparisons['P22-U_vs_P12']['all_accuracy_delta_pp']:+.2f} pp`。",
        f"- 机制门槛：`{'通过' if gates['mechanism_passed'] else '未通过'}`。",
        f"- 强机制证据：`{'通过' if gates['strong_mechanism_passed'] else '未通过'}`。",
        f"- P22-F 升级准确率主线："
        f"`{'通过' if gates['accuracy_reference']['P22-F']['passed'] else '未通过'}`。",
        f"- P22-U 升级准确率主线："
        f"`{'通过' if gates['accuracy_reference']['P22-U']['passed'] else '未通过'}`。",
        "",
        "## 三折",
        "",
        "| fold | P22-L | P22-F | P22-U | P12 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fold in range(3):
        lines.append(
            f"| {fold} | "
            f"{metrics['P22-L']['per_fold'][str(fold)]['accuracy']*100:.2f}% | "
            f"{metrics['P22-F']['per_fold'][str(fold)]['accuracy']*100:.2f}% | "
            f"{metrics['P22-U']['per_fold'][str(fold)]['accuracy']*100:.2f}% | "
            f"{metrics['P12']['per_fold'][str(fold)]['accuracy']*100:.2f}% |"
        )
    lines += [
        "",
        "## 资源",
        "",
        f"- P22-F 参数：`{summary['resources']['P22-F']['parameters']:,}`。",
        f"- P22-F 最大 checkpoint："
        f"`{summary['resources']['P22-F']['max_checkpoint_mib']:.3f} MiB`。",
        f"- 估计完整部署模型："
        f"`{summary['resources']['P22-F']['estimated_total_model_mib']:.3f} MiB`。",
        f"- 三折训练与最终验证耗时："
        f"`{summary['resources']['training_elapsed_seconds']:.1f} s`。",
        f"- 消融推理耗时："
        f"`{summary['resources']['ablation_elapsed_seconds']:.1f} s`。",
        "",
        "## 最终结论",
        "",
        summary["conclusion"],
        "",
        "## 下一步唯一建议",
        "",
        summary["next_recommendation"],
        "",
        "完整逐 subject、逐类 recall、混淆矩阵、前 20 双向混淆对、"
        "缺失模式、bootstrap、zero-out、shuffle 和 rescue/new-error "
        "保存在 `aligned_multimodal/runs/p22_joint_pooled_fusion/analysis/`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    cache_dir = args.cache_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite P22 analysis: {output_dir}")
    if args.report.resolve().exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.report.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=False)
    training_summary = json.loads(
        (experiment_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    if training_summary["status"] != "fixed_three_fold_training_complete":
        raise RuntimeError("P22 training is incomplete")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P22 ablation evaluation requires CUDA")

    model_oof = {
        model_name: load_npz(experiment_dir / f"{model_name}_oof_logits.npz")
        for model_name in MODEL_NAMES
    }
    reference = load_npz(
        PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
    )
    canonical = model_oof["P22-F"]
    for model_name, archive in model_oof.items():
        for key in ("sample_ids", "labels", "subjects", "folds"):
            if not np.array_equal(archive[key], canonical[key]):
                raise ValueError(f"OOF identity mismatch: {model_name}/{key}")
    if not np.array_equal(reference["sample_ids"], canonical["sample_ids"]):
        raise ValueError("P12 sample order differs from P22")
    if not np.array_equal(reference["labels"], canonical["labels"]):
        raise ValueError("P12 labels differ from P22")
    if not np.array_equal(reference["folds"], canonical["folds"]):
        raise ValueError("P12 folds differ from P22")

    labels = canonical["labels"].astype(np.int64)
    subjects = canonical["subjects"].astype(str)
    folds = canonical["folds"].astype(np.int64)
    first_cache = load_npz(cache_dir / "fold_0_cache.npz")
    presence = first_cache["presence"].astype(np.uint8)
    predictions = {
        model_name: model_oof[model_name]["logits"].argmax(1)
        for model_name in MODEL_NAMES
    }
    predictions["P12"] = reference["final_predictions"].astype(np.int64)
    metrics = {
        model_name: model_metrics(
            labels,
            model_predictions,
            folds,
            subjects,
            presence,
            config,
        )
        for model_name, model_predictions in predictions.items()
    }

    repeats = int(config["evaluation"]["bootstrap_repeats"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    comparison_pairs = (
        ("P22-F", "P22-L"),
        ("P22-U", "P22-F"),
        ("P22-F", "P12"),
        ("P22-U", "P12"),
    )
    comparisons: dict[str, Any] = {}
    for candidate, base in comparison_pairs:
        key = f"{candidate}_vs_{base}"
        candidate_metrics = metrics[candidate]
        base_metrics = metrics[base]
        comparisons[key] = {
            "all_accuracy_delta_pp": (
                float(candidate_metrics["all"]["accuracy"])
                - float(base_metrics["all"]["accuracy"])
            )
            * 100.0,
            "small_accuracy_delta_pp": (
                float(candidate_metrics["small"]["accuracy"])
                - float(base_metrics["small"]["accuracy"])
            )
            * 100.0,
            "hard_accuracy_delta_pp": (
                float(candidate_metrics["hard"]["accuracy"])
                - float(base_metrics["hard"]["accuracy"])
            )
            * 100.0,
            "per_fold_accuracy_delta_pp": {
                str(fold): (
                    float(candidate_metrics["per_fold"][str(fold)]["accuracy"])
                    - float(base_metrics["per_fold"][str(fold)]["accuracy"])
                )
                * 100.0
                for fold in range(3)
            },
            "bootstrap": subject_bootstrap(
                labels,
                predictions[candidate],
                predictions[base],
                subjects,
                repeats,
                bootstrap_seed,
            ),
            "win_loss": win_loss(
                labels, predictions[candidate], predictions[base]
            ),
        }

    ablation_logits, ablation_runtime = run_ablations(
        config, cache_dir, experiment_dir, device
    )
    np.savez_compressed(
        output_dir / "ablation_logits.npz",
        sample_ids=canonical["sample_ids"],
        labels=labels,
        subjects=subjects,
        folds=folds,
        presence=presence,
        **{
            key: value.astype(np.float16)
            for key, value in ablation_logits.items()
        },
    )
    ablations: dict[str, Any] = {}
    for key, logits in ablation_logits.items():
        model_name, action, modality = key.split("_", 2)
        changed_predictions = logits.argmax(1)
        present_mask = presence[:, MODALITY_ORDER.index(modality)] > 0
        ablations[key] = {
            "all": basic_metrics(labels, changed_predictions),
            "present_subset": basic_metrics(
                labels[present_mask], changed_predictions[present_mask]
            ),
            "delta_vs_original_pp": (
                (changed_predictions == labels).mean()
                - (predictions[model_name] == labels).mean()
            )
            * 100.0,
            "present_subset_delta_vs_original_pp": (
                (changed_predictions[present_mask] == labels[present_mask]).mean()
                - (
                    predictions[model_name][present_mask]
                    == labels[present_mask]
                ).mean()
            )
            * 100.0,
            "win_loss": win_loss(
                labels, changed_predictions, predictions[model_name]
            ),
        }

    output_rows_subject: list[dict[str, Any]] = []
    output_rows_class: list[dict[str, Any]] = []
    top_pairs: dict[str, Any] = {}
    for model_name in (*MODEL_NAMES, "P12"):
        for subject, values in metrics[model_name]["per_subject"].items():
            output_rows_subject.append(
                {"model": model_name, "subject": subject, **values}
            )
        for values in per_class_recall(labels, predictions[model_name]):
            output_rows_class.append({"model": model_name, **values})
        matrix = confusion_matrix(
            labels, predictions[model_name], labels=np.arange(40)
        )
        np.savetxt(
            output_dir / f"{model_name}_confusion_matrix.csv",
            matrix,
            delimiter=",",
            fmt="%d",
        )
        top_pairs[model_name] = top_bidirectional_pairs(
            labels, predictions[model_name]
        )
    write_csv(
        output_dir / "per_subject_metrics.csv",
        [
            "model",
            "subject",
            "samples",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
        output_rows_subject,
    )
    write_csv(
        output_dir / "per_class_recall.csv",
        ["model", "class_id", "samples", "recall", "correct"],
        output_rows_class,
    )
    (output_dir / "top20_bidirectional_pairs.json").write_text(
        json.dumps(top_pairs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    mechanism = comparisons["P22-F_vs_P22-L"]
    mechanism_gate = config["gates"]["mechanism"]
    fold_deltas = list(mechanism["per_fold_accuracy_delta_pp"].values())
    mechanism_checks = {
        "overall": mechanism["all_accuracy_delta_pp"]
        >= float(mechanism_gate["overall_delta_pp_min"]),
        "small": mechanism["small_accuracy_delta_pp"]
        >= float(mechanism_gate["small_delta_pp_min"]),
        "hard": mechanism["hard_accuracy_delta_pp"]
        >= float(mechanism_gate["hard_delta_pp_min"]),
        "fold_count": sum(delta > 0 for delta in fold_deltas)
        >= int(mechanism_gate["improved_fold_count_min"]),
        "worst_fold": min(fold_deltas)
        >= float(mechanism_gate["worst_fold_delta_pp_min"]),
        "bootstrap": float(
            mechanism["bootstrap"]["probability_delta_positive"]
        )
        >= float(mechanism_gate["bootstrap_probability_positive_min"]),
    }
    strong_mechanism = mechanism["all_accuracy_delta_pp"] >= float(
        config["gates"]["strong_mechanism"]["overall_delta_pp_min"]
    )

    encoder_size_mib = (
        (PROJECT_DIR / "runs" / "p11_final_package" / "skeleton_final_fp16.pt").stat().st_size
        + (PROJECT_DIR / "runs" / "p11_final_package" / "depth_final_fp16.pt").stat().st_size
        + (
            REPO_ROOT
            / "thermal_baseline"
            / "runs"
            / "p11_thermal_imagenet_full18"
            / "final_epoch_fp16.pt"
        ).stat().st_size
        + max(
            (
                PROJECT_DIR
                / "runs"
                / "p20_tiny_imu_student_oof"
                / f"fold_{fold}"
                / "final.pt"
            ).stat().st_size
            for fold in range(3)
        )
    ) / 1024**2
    resources: dict[str, Any] = {
        "training_elapsed_seconds": float(training_summary["elapsed_seconds"]),
        "ablation_elapsed_seconds": float(ablation_runtime["elapsed_seconds"]),
        "frozen_encoder_estimated_mib": encoder_size_mib,
    }
    for model_name in MODEL_NAMES:
        fold_items = [
            item for item in training_summary["folds"]
            if item["model"] == model_name
        ]
        max_checkpoint_mib = max(
            float(item["checkpoint_bytes"]) / 1024**2 for item in fold_items
        )
        resources[model_name] = {
            "parameters": int(fold_items[0]["parameters"]),
            "max_checkpoint_mib": max_checkpoint_mib,
            "estimated_total_model_mib": encoder_size_mib + max_checkpoint_mib,
            "max_peak_vram_mib": max(
                float(item["peak_vram_mib"]) for item in fold_items
            ),
            "total_inference_seconds": sum(
                float(item["inference_seconds"]) for item in fold_items
            ),
        }

    accuracy_reference: dict[str, Any] = {}
    accuracy_gate = config["gates"]["accuracy_reference"]
    for model_name in ("P22-F", "P22-U"):
        comparison = comparisons[f"{model_name}_vs_P12"]
        model_item = metrics[model_name]
        fold_delta_values = list(
            comparison["per_fold_accuracy_delta_pp"].values()
        )
        checks = {
            "overall": float(model_item["all"]["accuracy"])
            >= float(accuracy_gate["overall_accuracy_min"]),
            "small": float(model_item["small"]["accuracy"])
            >= float(accuracy_gate["small_action_accuracy_min"]),
            "hard": float(model_item["hard"]["accuracy"])
            >= float(accuracy_gate["hard_class_accuracy_min"]),
            "fold_count": sum(delta > 0 for delta in fold_delta_values)
            >= int(accuracy_gate["improved_fold_count_min"]),
            "bootstrap_ci_lower": float(
                comparison["bootstrap"]["subject_cluster_bootstrap_95_ci_pp"][0]
            )
            > float(accuracy_gate["bootstrap_ci95_lower_pp_min"]),
            "model_size": float(
                resources[model_name]["estimated_total_model_mib"]
            )
            < float(accuracy_gate["model_size_mib_max"]),
        }
        accuracy_reference[model_name] = {
            "passed": all(checks.values()),
            "checks": checks,
        }

    mechanism_passed = all(mechanism_checks.values())
    any_accuracy_passed = any(
        item["passed"] for item in accuracy_reference.values()
    )
    if any_accuracy_passed:
        conclusion = (
            "P22-A 达到预注册准确率升级门槛，可作为新的准确率主线候选；"
            "P22-U 仍只按诊断身份解释。"
        )
        next_recommendation = (
            "只进行一次人工复核，确认 full18 Tiny IMU refit 与最终打包协议；"
            "不要自动启动 P22-B。"
        )
    elif mechanism_passed:
        conclusion = (
            "P22-F 证明 pooled feature 相对 matched logits 有稳定机制增益，"
            "但未达到替换 P12 的强门槛。"
        )
        next_recommendation = (
            "保留 P22-F 为研究候选并停止自动训练，由人工决定是否值得单独审计 P22-B。"
        )
    else:
        conclusion = (
            "P22-F 未通过预注册机制门槛，当前 pooled feature 融合假设不成立或不稳定，"
            "不应替换 P12。"
        )
        next_recommendation = (
            "停止 P22 路线，不启动 P22-B；继续以 P12 作为可部署准确率主线。"
        )

    summary = {
        "status": "complete",
        "protocol": config["protocol_version"],
        "stage_one_code_commit": training_summary["code_commit"],
        "protocol_deviation": False,
        "metrics": metrics,
        "comparisons": comparisons,
        "ablations": ablations,
        "top20_bidirectional_pairs": top_pairs,
        "gates": {
            "mechanism_passed": mechanism_passed,
            "mechanism_checks": mechanism_checks,
            "strong_mechanism_passed": strong_mechanism,
            "accuracy_reference": accuracy_reference,
        },
        "resources": resources,
        "conclusion": conclusion,
        "next_recommendation": next_recommendation,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.report.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
