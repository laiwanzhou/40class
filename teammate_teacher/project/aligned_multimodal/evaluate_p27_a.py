from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from p27_data import P27Dataset, compute_fold_imu_stats, read_csv
from p27_model import P27EventModel
from train_p27_a import evaluate, move_batch, seed_everything


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_RUN = PROJECT_DIR / "runs" / "p27_a"
DEFAULT_P12 = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_HARD = PROJECT_DIR / "data" / "hard_local_v1" / "hard_action_protocol.json"
DEFAULT_REPORT = REPO_DIR / "29_P27-A事件级多模态联合表示实验结果.md"
VARIANTS = ("a0", "a1", "a2")
SMALL_IDS = np.asarray(
    (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39),
    dtype=np.int64,
)
TARGET_IDS = (9, 10, 19, 21, 22, 24, 25, 26, 37)
ABLATIONS = (
    "event_zero",
    "event_shuffle",
    "global_shuffle",
    "ir_zero",
    "ir_shuffle",
    "imu_zero",
    "imu_shuffle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict P27-A OOF and ablation audit")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--p12", type=Path, default=DEFAULT_P12)
    parser.add_argument("--hard-protocol", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-ablations", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_block(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    if not len(labels):
        return {
            "samples": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
        }
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def metric_groups(
    labels: np.ndarray,
    predictions: np.ndarray,
    hard_ids: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    small = np.isin(labels, SMALL_IDS)
    hard = np.isin(labels, hard_ids)
    return {
        "all": metric_block(labels, predictions),
        "small": metric_block(labels[small], predictions[small]),
        "hard": metric_block(labels[hard], predictions[hard]),
    }


def load_variant(run_dir: Path, variant: str) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    for fold in range(3):
        path = run_dir / f"fold_{fold}" / variant / "held_outputs.npz"
        if not path.is_file():
            raise FileNotFoundError(f"formal OOF is incomplete: {path}")
        with np.load(path, allow_pickle=False) as archive:
            part = {key: np.asarray(archive[key]) for key in archive.files}
        part["folds"] = np.full(len(part["labels"]), fold, dtype=np.int64)
        parts.append(part)
    common = ("sample_ids", "subjects", "labels", "predictions", "logits", "folds")
    result = {key: np.concatenate([part[key] for part in parts]) for key in common}
    if len(np.unique(result["sample_ids"].astype(str))) != len(result["sample_ids"]):
        raise ValueError(f"duplicate sample IDs in {variant} OOF")
    order = np.argsort(result["sample_ids"].astype(str))
    return {key: value[order] for key, value in result.items()}


def validate_variants(arrays: dict[str, dict[str, np.ndarray]]) -> None:
    reference = arrays["a0"]
    for variant in VARIANTS[1:]:
        candidate = arrays[variant]
        for key in ("sample_ids", "subjects", "labels", "folds"):
            if not np.array_equal(reference[key], candidate[key]):
                raise ValueError(f"{variant} differs from A0 on {key}")
    if len(reference["labels"]) != 2933:
        raise ValueError(f"expected 2933 P27 union OOF rows, found {len(reference['labels'])}")


def load_manifest(path: Path) -> tuple[dict[str, dict[str, str]], dict[int, str]]:
    rows = read_csv(path)
    by_id = {row["sample_id"]: row for row in rows}
    names = {int(row["class_id"]): row["class_name"] for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate sample IDs in P27 manifest")
    return by_id, names


def load_p12_common(
    path: Path,
    p27: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        p12 = {
            "sample_ids": archive["sample_ids"].astype(str),
            "labels": archive["labels"].astype(np.int64),
            "folds": archive["folds"].astype(np.int64),
            "predictions": archive["final_predictions"].astype(np.int64),
            "logits": archive["final_logits"].astype(np.float32),
        }
    p27_index = {
        sample_id: index
        for index, sample_id in enumerate(p27["sample_ids"].astype(str).tolist())
    }
    if not set(p12["sample_ids"]).issubset(p27_index):
        missing = sorted(set(p12["sample_ids"]) - set(p27_index))
        raise ValueError(f"P12 sample IDs absent from P27 union: {missing[:3]}")
    indices = np.asarray([p27_index[item] for item in p12["sample_ids"]], dtype=np.int64)
    if not np.array_equal(p12["labels"], p27["labels"][indices]):
        raise ValueError("P12/P27 labels differ on the 2914 common samples")
    if not np.array_equal(p12["folds"], p27["folds"][indices]):
        raise ValueError("P12/P27 held folds differ on the common samples")
    return p12, indices


def recalls(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    values = np.full(40, np.nan, dtype=np.float64)
    for class_id in range(40):
        mask = labels == class_id
        if mask.any():
            values[class_id] = float((predictions[mask] == class_id).mean())
    return values


def delta_pp(candidate: float, reference: float) -> float:
    return 100.0 * (candidate - reference)


def completed_metrics(run_dir: Path, fold: int, variant: str) -> dict[str, Any]:
    path = run_dir / f"fold_{fold}" / variant / "metrics.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("variant") != variant
        or int(result.get("fold", -1)) != fold
        or not bool(result.get("formal", False))
    ):
        raise ValueError(f"not a compatible formal result: {path}")
    return result


def aggregate_event_metrics(run_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for variant in VARIANTS:
        components: defaultdict[str, list[tuple[float, float, int]]] = defaultdict(list)
        for fold in range(3):
            metrics = completed_metrics(run_dir, fold, variant)
            with np.load(
                run_dir / f"fold_{fold}" / variant / "held_outputs.npz",
                allow_pickle=False,
            ) as archive:
                for name, item in metrics["event_prediction"].items():
                    target = archive[f"{name}_targets"]
                    mask = archive[f"{name}_masks"]
                    while mask.ndim < target.ndim:
                        mask = np.expand_dims(mask, axis=1)
                    count = int(np.broadcast_to(mask, target.shape).sum())
                    components[name].append(
                        (
                            float(item["mae"]),
                            float(item["train_mean_baseline_mae"]),
                            count,
                        )
                    )
        result[variant] = {}
        for name, values in components.items():
            weight = np.asarray([item[2] for item in values], dtype=np.float64)
            mae = float(np.average([item[0] for item in values], weights=weight))
            baseline = float(np.average([item[1] for item in values], weights=weight))
            result[variant][name] = {
                "mae": mae,
                "train_mean_baseline_mae": baseline,
                "relative_improvement": (baseline - mae) / max(baseline, 1e-12),
            }
    return result


def make_rows(
    arrays: dict[str, dict[str, np.ndarray]],
    p12: dict[str, np.ndarray],
    p12_indices: np.ndarray,
    manifest: dict[str, dict[str, str]],
    names: dict[int, str],
    hard_ids: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    reference = arrays["a0"]
    overall_rows: list[dict[str, Any]] = []
    per_fold: list[dict[str, Any]] = []
    per_subject: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        groups = metric_groups(reference["labels"], arrays[variant]["predictions"], hard_ids)
        for group, metrics in groups.items():
            overall_rows.append({"model": f"P27-{variant.upper()}", "scope": group, **metrics})
        common_groups = metric_groups(
            reference["labels"][p12_indices],
            arrays[variant]["predictions"][p12_indices],
            hard_ids,
        )
        for group, metrics in common_groups.items():
            overall_rows.append(
                {
                    "model": f"P27-{variant.upper()}",
                    "scope": f"common_{group}",
                    **metrics,
                }
            )
        for fold in range(3):
            mask = reference["folds"] == fold
            groups = metric_groups(
                reference["labels"][mask],
                arrays[variant]["predictions"][mask],
                hard_ids,
            )
            for group, metrics in groups.items():
                per_fold.append(
                    {
                        "model": f"P27-{variant.upper()}",
                        "fold": fold,
                        "scope": group,
                        **metrics,
                    }
                )
        for subject in sorted(np.unique(reference["subjects"].astype(str))):
            mask = reference["subjects"].astype(str) == subject
            groups = metric_groups(
                reference["labels"][mask],
                arrays[variant]["predictions"][mask],
                hard_ids,
            )
            for group, metrics in groups.items():
                per_subject.append(
                    {
                        "model": f"P27-{variant.upper()}",
                        "subject": subject,
                        "fold": int(reference["folds"][mask][0]),
                        "scope": group,
                        **metrics,
                    }
                )
        class_recalls = recalls(reference["labels"], arrays[variant]["predictions"])
        for class_id in range(40):
            mask = reference["labels"] == class_id
            per_class.append(
                {
                    "model": f"P27-{variant.upper()}",
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "samples": int(mask.sum()),
                    "recall": float(class_recalls[class_id]),
                    "is_small": int(class_id in set(SMALL_IDS.tolist())),
                    "is_hard": int(class_id in set(hard_ids.tolist())),
                    "is_focus": int(class_id in TARGET_IDS),
                }
            )
        pattern_indices: defaultdict[str, list[int]] = defaultdict(list)
        for index, sample_id in enumerate(reference["sample_ids"].astype(str)):
            row = manifest[sample_id]
            pattern = "".join(
                [
                    "D" if int(row["depth_usable"]) else "-",
                    "I" if int(row["ir_usable"]) else "-",
                    "S" if int(row["skeleton_usable"]) else "-",
                    "M" if int(row["imu_usable"]) else "-",
                ]
            )
            pattern_indices[pattern].append(index)
        for pattern, indices_list in sorted(pattern_indices.items()):
            indices = np.asarray(indices_list, dtype=np.int64)
            missing_rows.append(
                {
                    "model": f"P27-{variant.upper()}",
                    "presence_pattern": pattern,
                    **metric_block(
                        reference["labels"][indices],
                        arrays[variant]["predictions"][indices],
                    ),
                }
            )
    p12_groups = metric_groups(p12["labels"], p12["predictions"], hard_ids)
    for group, metrics in p12_groups.items():
        overall_rows.append({"model": "P12", "scope": group, **metrics})
        overall_rows.append({"model": "P12", "scope": f"common_{group}", **metrics})
    for fold in range(3):
        mask = p12["folds"] == fold
        groups = metric_groups(
            p12["labels"][mask], p12["predictions"][mask], hard_ids
        )
        for group, metrics in groups.items():
            per_fold.append({"model": "P12", "fold": fold, "scope": group, **metrics})
    p12_subjects = reference["subjects"][p12_indices].astype(str)
    for subject in sorted(np.unique(p12_subjects)):
        mask = p12_subjects == subject
        groups = metric_groups(
            p12["labels"][mask], p12["predictions"][mask], hard_ids
        )
        for group, metrics in groups.items():
            per_subject.append(
                {
                    "model": "P12",
                    "subject": subject,
                    "fold": int(p12["folds"][mask][0]),
                    "scope": group,
                    **metrics,
                }
            )
    p12_recalls = recalls(p12["labels"], p12["predictions"])
    for class_id in range(40):
        mask = p12["labels"] == class_id
        per_class.append(
            {
                "model": "P12",
                "class_id": class_id,
                "class_name": names[class_id],
                "samples": int(mask.sum()),
                "recall": float(p12_recalls[class_id]),
                "is_small": int(class_id in set(SMALL_IDS.tolist())),
                "is_hard": int(class_id in set(hard_ids.tolist())),
                "is_focus": int(class_id in TARGET_IDS),
            }
        )
    return overall_rows, per_fold, per_subject, per_class, missing_rows


def rescue_rows(
    arrays: dict[str, dict[str, np.ndarray]],
    p12: dict[str, np.ndarray],
    p12_indices: np.ndarray,
    hard_ids: np.ndarray,
) -> list[dict[str, Any]]:
    reference = arrays["a0"]
    rows: list[dict[str, Any]] = []
    comparisons = [
        ("P27-A1", arrays["a1"]["predictions"], "P27-A0", reference["predictions"], np.arange(len(reference["labels"]))),
        ("P27-A2", arrays["a2"]["predictions"], "P27-A0", reference["predictions"], np.arange(len(reference["labels"]))),
    ]
    for variant in VARIANTS:
        comparisons.append(
            (
                f"P27-{variant.upper()}",
                arrays[variant]["predictions"][p12_indices],
                "P12",
                p12["predictions"],
                p12_indices,
            )
        )
    for candidate_name, candidate, reference_name, control, indices in comparisons:
        labels = reference["labels"][indices]
        subjects = reference["subjects"][indices].astype(str)
        for scope, scope_mask in (
            ("all", np.ones(len(labels), dtype=bool)),
            ("small", np.isin(labels, SMALL_IDS)),
            ("hard", np.isin(labels, hard_ids)),
        ):
            candidate_correct = candidate == labels
            control_correct = control == labels
            rescue = scope_mask & candidate_correct & ~control_correct
            new_error = scope_mask & ~candidate_correct & control_correct
            rows.append(
                {
                    "candidate": candidate_name,
                    "reference": reference_name,
                    "scope": scope,
                    "subject": "ALL",
                    "rescues": int(rescue.sum()),
                    "new_errors": int(new_error.sum()),
                    "net": int(rescue.sum() - new_error.sum()),
                }
            )
            if candidate_name == "P27-A2" and reference_name == "P27-A0":
                for subject in sorted(np.unique(subjects)):
                    subject_mask = subjects == subject
                    rows.append(
                        {
                            "candidate": candidate_name,
                            "reference": reference_name,
                            "scope": scope,
                            "subject": subject,
                            "rescues": int((rescue & subject_mask).sum()),
                            "new_errors": int((new_error & subject_mask).sum()),
                            "net": int(rescue[subject_mask].sum() - new_error[subject_mask].sum()),
                        }
                    )
    return rows


def ablation_loader(dataset: P27Dataset, config: dict, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.inference_mode()
def model_latency(
    model: P27EventModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float | int]:
    raw = next(iter(loader))
    batch = move_batch(raw, device)
    for _ in range(5):
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    repeats = 20
    started = time.perf_counter()
    for _ in range(repeats):
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "batch_size": int(len(batch["label"])),
        "repeats": repeats,
        "milliseconds_per_batch": 1000.0 * elapsed / repeats,
        "milliseconds_per_sample": 1000.0 * elapsed / (repeats * len(batch["label"])),
    }


def run_ablations(
    run_dir: Path,
    config: dict,
    manifest_path: Path,
    hard_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["use_amp"] and device.type == "cuda")
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=float(config["label_smoothing"])
    )
    per_fold: dict[str, Any] = {}
    runtime_folds: dict[str, Any] = {}
    for fold in range(3):
        imu_mean, imu_std = compute_fold_imu_stats(manifest_path, fold)
        dataset = P27Dataset(
            manifest_path,
            fold,
            train=False,
            num_frames=int(config["num_frames"]),
            image_height=int(config["image_height"]),
            image_width=int(config["image_width"]),
            roi_padding=float(config["roi_padding"]),
        )
        checkpoint_path = run_dir / f"fold_{fold}" / "a2" / "final.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = P27EventModel(
            torch.from_numpy(imu_mean),
            torch.from_numpy(imu_std),
            event_dim=int(config["event_dim"]),
            dropout=float(config["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        seed = int(config["seed"]) + 1000 + fold
        latency_loader = ablation_loader(dataset, config, seed)
        runtime_folds[str(fold)] = model_latency(model, latency_loader, device, use_amp)
        fold_results: dict[str, Any] = {}
        for ablation in ABLATIONS:
            seed_everything(seed)
            loader = ablation_loader(dataset, config, seed)
            metrics, arrays = evaluate(
                model,
                loader,
                criterion,
                device,
                config,
                use_amp,
                ablation=ablation,
            )
            fold_results[ablation] = {
                **metric_groups(arrays["labels"], arrays["predictions"], hard_ids),
                "elapsed_seconds": metrics["elapsed_seconds"],
            }
        per_fold[str(fold)] = fold_results
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate: dict[str, Any] = {}
    for ablation in ABLATIONS:
        aggregate[ablation] = {}
        for scope in ("all", "small", "hard"):
            weights = np.asarray(
                [per_fold[str(fold)][ablation][scope]["samples"] for fold in range(3)],
                dtype=np.float64,
            )
            aggregate[ablation][scope] = {
                metric: float(
                    np.average(
                        [
                            per_fold[str(fold)][ablation][scope][metric]
                            for fold in range(3)
                        ],
                        weights=weights,
                    )
                )
                for metric in ("accuracy",)
            }
            aggregate[ablation][scope]["samples"] = int(weights.sum())
    return {"per_fold": per_fold, "aggregate": aggregate}, runtime_folds


def checkpoint_runtime(run_dir: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for variant in VARIANTS:
        fold_metrics = [completed_metrics(run_dir, fold, variant) for fold in range(3)]
        rows[variant] = {
            "parameters": int(fold_metrics[0]["parameters"]),
            "theoretical_fp16_mib": float(fold_metrics[0]["fp16_size_mib"]),
            "deployment_checkpoint_mib": [
                (run_dir / f"fold_{fold}" / variant / "deployment_fp16.pt").stat().st_size
                / (1024**2)
                for fold in range(3)
            ],
            "peak_cuda_memory_mib": [
                float(item["peak_cuda_memory_mib"]) for item in fold_metrics
            ],
            "train_seconds": [float(item["train_seconds"]) for item in fold_metrics],
            "end_to_end_ms_per_sample": [
                float(item["milliseconds_per_sample"]) for item in fold_metrics
            ],
        }
    return rows


def save_oof(
    run_dir: Path,
    arrays: dict[str, dict[str, np.ndarray]],
    p12: dict[str, np.ndarray],
    p12_indices: np.ndarray,
    manifest: dict[str, dict[str, str]],
) -> None:
    reference = arrays["a0"]
    p12_by_index = np.full(len(reference["labels"]), -1, dtype=np.int64)
    p12_by_index[p12_indices] = p12["predictions"]
    np.savez_compressed(
        run_dir / "complete_oof.npz",
        sample_ids=reference["sample_ids"],
        subjects=reference["subjects"],
        labels=reference["labels"],
        folds=reference["folds"],
        a0_logits=arrays["a0"]["logits"].astype(np.float16),
        a1_logits=arrays["a1"]["logits"].astype(np.float16),
        a2_logits=arrays["a2"]["logits"].astype(np.float16),
        a0_predictions=arrays["a0"]["predictions"],
        a1_predictions=arrays["a1"]["predictions"],
        a2_predictions=arrays["a2"]["predictions"],
        p12_predictions=p12_by_index,
        p12_common=(p12_by_index >= 0).astype(np.uint8),
    )
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(reference["sample_ids"].astype(str)):
        row = manifest[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "subject": reference["subjects"][index],
                "fold": int(reference["folds"][index]),
                "label": int(reference["labels"][index]),
                "depth_present": int(row["depth_usable"]),
                "ir_present": int(row["ir_usable"]),
                "skeleton_present": int(row["skeleton_usable"]),
                "imu_present": int(row["imu_usable"]),
                "p12_prediction": int(p12_by_index[index]),
                "a0_prediction": int(arrays["a0"]["predictions"][index]),
                "a1_prediction": int(arrays["a1"]["predictions"][index]),
                "a2_prediction": int(arrays["a2"]["predictions"][index]),
            }
        )
    write_csv(run_dir / "complete_oof_rows.csv", rows)


def plot_results(
    run_dir: Path,
    summary: dict[str, Any],
    per_class: list[dict[str, Any]],
    names: dict[int, str],
) -> None:
    models = ("P12", "P27-A0", "P27-A1", "P27-A2")
    scopes = ("all", "small", "hard")
    lookup = {
        (row["model"], row["scope"]): row
        for row in summary["overall_rows"]
    }
    figure, axis = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(scopes))
    width = 0.19
    for offset, model in enumerate(models):
        values = [100.0 * lookup[(model, scope)]["accuracy"] for scope in scopes]
        axis.bar(x + (offset - 1.5) * width, values, width=width, label=model)
    axis.set_xticks(x, ("Overall", "Small", "Hard"))
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(run_dir / "metrics_comparison.png", dpi=160)
    plt.close(figure)

    recall_lookup = {
        (row["model"], int(row["class_id"])): 100.0 * float(row["recall"])
        for row in per_class
    }
    figure, axis = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(TARGET_IDS))
    for offset, model in enumerate(models):
        values = [recall_lookup[(model, class_id)] for class_id in TARGET_IDS]
        axis.bar(x + (offset - 1.5) * width, values, width=width, label=model)
    axis.set_xticks(
        x,
        [names[class_id].split("_", 1)[-1].replace("_", " ") for class_id in TARGET_IDS],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Recall (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(run_dir / "focus_class_recall.png", dpi=160)
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_hash_manifest(run_dir: Path) -> dict[str, Any]:
    paths: list[Path] = []
    for fold in range(3):
        for variant in VARIANTS:
            directory = run_dir / f"fold_{fold}" / variant
            paths.extend(
                directory / name
                for name in (
                    "history.csv",
                    "held_outputs.npz",
                    "metrics.json",
                    "final.pt",
                    "deployment_fp16.pt",
                )
            )
    paths.extend(
        run_dir / name
        for name in (
            "config_used.json",
            "complete_oof.npz",
            "complete_oof_rows.csv",
            "summary.json",
            "per_fold.csv",
            "per_subject.csv",
            "per_class_recall.csv",
            "missing_patterns.csv",
            "rescue_new_error.csv",
            "event_prediction.json",
            "ablations.json",
            "runtime.json",
            "metrics_comparison.png",
            "focus_class_recall.png",
        )
    )
    return {
        "files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "git_policy": (
                    "manifest_only"
                    if path.suffix == ".pt" or path.stat().st_size > 10 * 1024 * 1024
                    else "eligible"
                ),
            }
            for path in paths
            if path.is_file()
        ]
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def build_report(
    path: Path,
    summary: dict[str, Any],
    per_fold: list[dict[str, Any]],
    per_subject: list[dict[str, Any]],
    per_class: list[dict[str, Any]],
    event_metrics: dict[str, Any],
    ablations: dict[str, Any],
    runtime: dict[str, Any],
    names: dict[int, str],
) -> None:
    overall = {
        (row["model"], row["scope"]): row for row in summary["overall_rows"]
    }
    lines = [
        "# P27-A 事件级多模态联合表示实验结果",
        "",
        "## 协议",
        "",
        "- 三折 subject-disjoint；每折只在固定最终 epoch 评估 held subjects。",
        "- P27 使用真实 parser 的四模态并集 2933 条；与 P12 比较使用共同 2914 条。",
        "- A0/A1/A2 使用相同输入和结构；主实验无 P12 蒸馏、Thermal、Radar。",
        "",
        "## 核心 OOF",
        "",
        "| 模型 | Overall | Balanced | Macro-F1 | Small | Hard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("P12", "P27-A0", "P27-A1", "P27-A2"):
        all_metrics = overall[(model, "all")]
        lines.append(
            f"| {model} | {pct(all_metrics['accuracy'])} | "
            f"{pct(all_metrics['balanced_accuracy'])} | {pct(all_metrics['macro_f1'])} | "
            f"{pct(overall[(model, 'small')]['accuracy'])} | "
            f"{pct(overall[(model, 'hard')]['accuracy'])} |"
        )
    a0 = overall[("P27-A0", "all")]
    a2 = overall[("P27-A2", "all")]
    small_delta = delta_pp(
        overall[("P27-A2", "small")]["accuracy"],
        overall[("P27-A0", "small")]["accuracy"],
    )
    hard_delta = delta_pp(
        overall[("P27-A2", "hard")]["accuracy"],
        overall[("P27-A0", "hard")]["accuracy"],
    )
    fold_lookup = {
        (row["model"], int(row["fold"]), row["scope"]): row for row in per_fold
    }
    fold_wins = sum(
        fold_lookup[("P27-A2", fold, "all")]["accuracy"]
        > fold_lookup[("P27-A0", fold, "all")]["accuracy"]
        for fold in range(3)
    )
    lines.extend(
        [
            "",
            "## 预注册裁决",
            "",
            f"- A2 − A0 Overall：{delta_pp(a2['accuracy'], a0['accuracy']):+.2f} pp（门槛 +1.0 pp）。",
            f"- A2 − A0 Small：{small_delta:+.2f} pp（门槛 +1.5 pp）。",
            f"- A2 − A0 Hard：{hard_delta:+.2f} pp（门槛 +2.0 pp）。",
            f"- fold 胜出：{fold_wins}/3（门槛至少 2/3）。",
            f"- 共同 2914 条上 P27-A2 为 {pct(overall[('P27-A2', 'common_all')]['accuracy'])}，相对 P12 "
            f"{delta_pp(overall[('P27-A2', 'common_all')]['accuracy'], overall[('P12', 'common_all')]['accuracy']):+.2f} pp。",
            "",
            "## 三折",
            "",
            "| 模型 | fold 0 | fold 1 | fold 2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in ("P12", "P27-A0", "P27-A1", "P27-A2"):
        values = [
            pct(fold_lookup[(model, fold, "all")]["accuracy"]) for fold in range(3)
        ]
        lines.append(f"| {model} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            "## 重点类别 recall",
            "",
            "| 类别 | P12 | A0 | A1 | A2 | A2−A0 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    class_lookup = {
        (row["model"], int(row["class_id"])): row for row in per_class
    }
    for class_id in TARGET_IDS:
        values = [
            class_lookup[(model, class_id)]["recall"]
            for model in ("P12", "P27-A0", "P27-A1", "P27-A2")
        ]
        lines.append(
            f"| {class_id} {names[class_id]} | "
            f"{pct(values[0])} | {pct(values[1])} | {pct(values[2])} | "
            f"{pct(values[3])} | {delta_pp(values[3], values[1]):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 事件预测与消融",
            "",
            "| 事件分量 | A0 相对均值 | A1 相对均值 | A2 相对均值 |",
            "|---|---:|---:|---:|",
        ]
    )
    for component in ("skeleton", "visual", "imu", "shared_motion", "clip"):
        lines.append(
            f"| {component} | "
            f"{event_metrics['a0'][component]['relative_improvement']*100:+.1f}% | "
            f"{event_metrics['a1'][component]['relative_improvement']*100:+.1f}% | "
            f"{event_metrics['a2'][component]['relative_improvement']*100:+.1f}% |"
        )
    if ablations:
        lines.extend(
            [
                "",
                "| A2 消融 | Overall | Small | Hard |",
                "|---|---:|---:|---:|",
            ]
        )
        for name in ABLATIONS:
            item = ablations["aggregate"][name]
            lines.append(
                f"| {name} | {pct(item['all']['accuracy'])} | "
                f"{pct(item['small']['accuracy'])} | {pct(item['hard']['accuracy'])} |"
            )
    positive_subjects = sum(
        row["net"] > 0
        for row in summary["rescue_rows"]
        if row["candidate"] == "P27-A2"
        and row["reference"] == "P27-A0"
        and row["scope"] == "hard"
        and row["subject"] != "ALL"
    )
    negative_subjects = sum(
        row["net"] < 0
        for row in summary["rescue_rows"]
        if row["candidate"] == "P27-A2"
        and row["reference"] == "P27-A0"
        and row["scope"] == "hard"
        and row["subject"] != "ALL"
    )
    runtime_a2 = runtime["variants"]["a2"]
    lines.extend(
        [
            "",
            "## 跨 subject 与部署",
            "",
            f"- Hard rescue/new-error 净值为正的 subject：{positive_subjects}；净值为负：{negative_subjects}。完整表见 `runs/p27_a/rescue_new_error.csv`。",
            f"- 参数量：{runtime_a2['parameters']:,}；理论 FP16：{runtime_a2['theoretical_fp16_mib']:.2f} MiB；实际 checkpoint 最大 {max(runtime_a2['deployment_checkpoint_mib']):.2f} MiB。",
            f"- 训练峰值显存最大 {max(runtime_a2['peak_cuda_memory_mib']):.1f} MiB。",
            f"- held 端到端（含数据读取）加权均值约 {np.mean(runtime_a2['end_to_end_ms_per_sample']):.2f} ms/sample；模型纯前向见 `runs/p27_a/runtime.json`。",
            "",
            "## 产物",
            "",
            "- 完整 OOF、逐 fold、逐 subject、逐类 recall、缺失模式、rescue/new-error、事件指标和消融均位于 `aligned_multimodal/runs/p27_a/`。",
            "- checkpoint 与大型产物不提交 Git；绝对路径、字节数和 SHA256 见 `artifact_hashes.json`。",
            "- 弱标签数值抽检及 ROI montage 见 `aligned_multimodal/runs/p27_0_audit/`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config_used.json").read_text(encoding="utf-8"))
    if config["protocol"] != "p27-a-v1-fixed-before-training":
        raise ValueError("unexpected P27 protocol")
    hard_protocol = json.loads(args.hard_protocol.resolve().read_text(encoding="utf-8"))
    hard_ids = np.asarray(hard_protocol["hard_class_ids"], dtype=np.int64)
    manifest_path = PROJECT_DIR / config["manifest"]
    manifest, names = load_manifest(manifest_path)
    arrays = {variant: load_variant(run_dir, variant) for variant in VARIANTS}
    validate_variants(arrays)
    p12, p12_indices = load_p12_common(args.p12.resolve(), arrays["a0"])
    overall_rows, per_fold, per_subject, per_class, missing_rows = make_rows(
        arrays, p12, p12_indices, manifest, names, hard_ids
    )
    rescue = rescue_rows(arrays, p12, p12_indices, hard_ids)
    event_metrics = aggregate_event_metrics(run_dir)
    save_oof(run_dir, arrays, p12, p12_indices, manifest)
    write_csv(run_dir / "per_fold.csv", per_fold)
    write_csv(run_dir / "per_subject.csv", per_subject)
    write_csv(run_dir / "per_class_recall.csv", per_class)
    write_csv(run_dir / "missing_patterns.csv", missing_rows)
    write_csv(run_dir / "rescue_new_error.csv", rescue)
    json_dump(run_dir / "event_prediction.json", event_metrics)
    ablations: dict[str, Any] = {}
    model_runtime: dict[str, Any] = {}
    if not args.skip_ablations:
        ablations, model_runtime = run_ablations(
            run_dir, config, manifest_path, hard_ids
        )
    json_dump(run_dir / "ablations.json", ablations)
    runtime = {
        "variants": checkpoint_runtime(run_dir),
        "a2_model_only_per_fold": model_runtime,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    json_dump(run_dir / "runtime.json", runtime)
    summary = {
        "protocol": config["protocol"],
        "p27_union_samples": int(len(arrays["a0"]["labels"])),
        "p12_common_samples": int(len(p12["labels"])),
        "small_class_ids": SMALL_IDS.tolist(),
        "hard_class_ids": hard_ids.tolist(),
        "focus_class_ids": list(TARGET_IDS),
        "overall_rows": overall_rows,
        "rescue_rows": rescue,
    }
    json_dump(run_dir / "summary.json", summary)
    plot_results(run_dir, summary, per_class, names)
    build_report(
        args.report.resolve(),
        summary,
        per_fold,
        per_subject,
        per_class,
        event_metrics,
        ablations,
        runtime,
        names,
    )
    json_dump(run_dir / "artifact_hashes.json", build_hash_manifest(run_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
