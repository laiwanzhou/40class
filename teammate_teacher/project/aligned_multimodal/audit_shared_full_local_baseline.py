from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score
from torch import nn

from analyze_local_depth_oof_fusion import SMALL_ACTION_IDS
from quantize_checkpoint_storage import convert_state_dict
from train_shared_full_local_oof import (
    DEFAULT_FOLD_DIR,
    DEFAULT_FULL_CACHE,
    DEFAULT_LOCAL_CACHE,
    FullLocalCacheDataset,
    SharedFullLocalModel,
    create_loader,
    run_epoch,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT_DIR / "runs" / "p16_shared_full_local_oracle_oof"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p17_shared_baseline_freeze"
DEFAULT_ROWS = (
    PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof_rows.csv"
)
DEFAULT_HARD = (
    PROJECT_DIR / "data" / "hard_local_v1" / "hard_action_protocol.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the current shared-BN Full+Local baseline, create actual "
            "FP16-storage checkpoints, and verify reloaded validation logits."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--full-cache", type=Path, default=DEFAULT_FULL_CACHE)
    parser.add_argument("--local-cache", type=Path, default=DEFAULT_LOCAL_CACHE)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--hard-protocol", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def align_logits(
    source_ids: np.ndarray,
    source_logits: np.ndarray,
    target_ids: list[str],
) -> np.ndarray:
    location = {
        sample_id: index
        for index, sample_id in enumerate(source_ids.astype(str))
    }
    if set(location) != set(target_ids):
        raise ValueError("Stored and reloaded validation IDs differ")
    return np.stack([source_logits[location[sample_id]] for sample_id in target_ids])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 reload verification requires CUDA")
    torch.backends.cudnn.benchmark = True
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    fold_reports: list[dict[str, object]] = []

    for fold in range(3):
        source = run_dir / f"fold_{fold}" / "best.pt"
        target_dir = output_dir / f"fold_{fold}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "best_fp16.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        converted_state, tensor_report = convert_state_dict(
            checkpoint["model_state_dict"]
        )
        converted = dict(checkpoint)
        converted["model_state_dict"] = converted_state
        converted["storage_dtype"] = "float16"
        converted["storage_source"] = str(source)
        torch.save(converted, target)

        reloaded = torch.load(target, map_location="cpu", weights_only=False)
        model = SharedFullLocalModel().to(device)
        model.load_state_dict(reloaded["model_state_dict"], strict=True)
        dataset = FullLocalCacheDataset(
            args.full_cache.resolve(),
            args.local_cache.resolve(),
            args.fold_dir.resolve() / f"fold_{fold}.csv",
            fold,
            "val",
            augment=False,
        )
        loader = create_loader(
            dataset,
            int(args.batch_size),
            max(0, int(args.workers)),
            False,
            20260728 + fold,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=False)
        _, logits, labels, sample_ids, _ = run_epoch(
            model,
            loader,
            criterion,
            device,
            None,
            scaler,
            0.25,
        )
        with np.load(
            run_dir / f"fold_{fold}" / "best_val_logits.npz",
            allow_pickle=False,
        ) as baseline:
            if not np.array_equal(
                labels,
                align_logits(
                    baseline["sample_ids"],
                    baseline["labels"][:, None],
                    sample_ids,
                )[:, 0].astype(np.int64),
            ):
                raise ValueError(f"Fold {fold} labels changed")
            comparisons: dict[str, object] = {}
            for runtime_key, stored_key in (
                ("fused", "fused_logits"),
                ("full", "full_aux_logits"),
                ("local", "local_aux_logits"),
            ):
                stored = align_logits(
                    baseline["sample_ids"],
                    baseline[stored_key].astype(np.float32),
                    sample_ids,
                )
                current = logits[runtime_key].astype(np.float32)
                comparisons[runtime_key] = {
                    "max_abs_logit_delta": float(
                        np.max(np.abs(current - stored))
                    ),
                    "mean_abs_logit_delta": float(
                        np.mean(np.abs(current - stored))
                    ),
                    "prediction_changes": int(
                        np.sum(current.argmax(1) != stored.argmax(1))
                    ),
                    "stored_accuracy": float(
                        accuracy_score(labels, stored.argmax(1))
                    ),
                    "reloaded_fp16_accuracy": float(
                        accuracy_score(labels, current.argmax(1))
                    ),
                }
        fold_reports.append(
            {
                "fold": fold,
                "source_bytes": source.stat().st_size,
                "source_mib": source.stat().st_size / 2**20,
                "source_sha256": sha256(source),
                "fp16_bytes": target.stat().st_size,
                "fp16_mib": target.stat().st_size / 2**20,
                "fp16_sha256": sha256(target),
                "compression_ratio": target.stat().st_size / source.stat().st_size,
                **tensor_report,
                "reload_comparison": comparisons,
            }
        )
        del loader, dataset, model
        torch.cuda.empty_cache()

    with np.load(run_dir / "oof_logits.npz", allow_pickle=False) as oof:
        sample_ids = oof["sample_ids"].astype(str)
        labels = oof["labels"].astype(np.int64)
        folds = oof["held_fold"].astype(np.int64)
        fused_logits = oof["fused_logits"].astype(np.float32)
        full_logits = oof["full_aux_logits"].astype(np.float32)
        local_logits = oof["local_aux_logits"].astype(np.float32)
    row_metadata = {
        row["sample_id"]: row
        for row in read_csv(args.rows.resolve())
    }
    hard_protocol = json.loads(
        args.hard_protocol.resolve().read_text(encoding="utf-8")
    )
    hard_ids = set(int(value) for value in hard_protocol["hard_class_ids"])
    probabilities = softmax(fused_logits)
    predictions = fused_logits.argmax(1)
    full_predictions = full_logits.argmax(1)
    local_predictions = local_logits.argmax(1)
    per_sample: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        metadata = row_metadata[sample_id]
        sorted_probability = np.sort(probabilities[index])
        per_sample.append(
            {
                "sample_id": sample_id,
                "fold": int(folds[index]),
                "class_id": int(labels[index]),
                "class_name": metadata["class_name"],
                "user_id": metadata["user_id"],
                "trial_id": metadata["trial_id"],
                "is_small_action": int(
                    int(labels[index]) in set(SMALL_ACTION_IDS.tolist())
                ),
                "is_hard_class": int(int(labels[index]) in hard_ids),
                "fused_prediction": int(predictions[index]),
                "full_prediction": int(full_predictions[index]),
                "local_prediction": int(local_predictions[index]),
                "fused_correct": int(predictions[index] == labels[index]),
                "fused_confidence": float(sorted_probability[-1]),
                "fused_margin": float(
                    sorted_probability[-1] - sorted_probability[-2]
                ),
            }
        )
    write_csv(output_dir / "per_sample_predictions.csv", per_sample)

    class_names = {
        int(row["class_id"]): row["class_name"]
        for row in row_metadata.values()
    }
    per_class: list[dict[str, object]] = []
    for class_id in range(40):
        mask = labels == class_id
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "samples": int(mask.sum()),
                "is_small_action": int(
                    class_id in set(SMALL_ACTION_IDS.tolist())
                ),
                "is_hard_class": int(class_id in hard_ids),
                "fused_recall": float(
                    recall_score(
                        labels[mask],
                        predictions[mask],
                        labels=[class_id],
                        average=None,
                        zero_division=0,
                    )[0]
                ),
                "full_recall": float(
                    np.mean(full_predictions[mask] == labels[mask])
                ),
                "local_recall": float(
                    np.mean(local_predictions[mask] == labels[mask])
                ),
            }
        )
    write_csv(output_dir / "per_class_recall.csv", per_class)

    subject_correct: dict[str, list[int]] = defaultdict(list)
    for row in per_sample:
        subject_correct[str(row["user_id"])].append(int(row["fused_correct"]))
    per_subject = [
        {
            "user_id": subject,
            "samples": len(values),
            "accuracy": float(np.mean(values)),
        }
        for subject, values in sorted(subject_correct.items())
    ]
    write_csv(output_dir / "per_subject_accuracy.csv", per_subject)

    architecture = {
        "status": "exploratory_oracle_assisted",
        "deployable_oof": False,
        "wide_local": False,
        "visual_encoder": "one shared ResNet-18+TSM VisualEncoder",
        "shared_layers": "depth stem, bn1, layer1, layer2, layer3, layer4",
        "batch_norm": (
            "one shared BN set; Full and Local are concatenated along batch "
            "before one encoder call"
        ),
        "temporal_sampling": (
            "fixed midpoint from each of 12 segments in both training and validation"
        ),
        "augmentation": "paired horizontal flip only",
        "full_project": "independent Linear(1024,256)+LayerNorm+GELU",
        "local_project": "independent Linear(1024,256)+LayerNorm+GELU",
        "fusion": (
            "feature gate over Full/Local projections, then classifier over "
            "[Full, Local, gated feature]"
        ),
        "heads": "fused, Full auxiliary, Local auxiliary",
        "loss": "L_fused + 0.25*L_full + 0.25*L_local",
        "initialization": "fold-matched ImageNet Full Depth checkpoint",
        "freeze_schedule": "none; all layers train from epoch 1",
        "learning_rates": {"backbone": 1e-4, "heads": 3e-4},
        "knowledge_distillation": False,
        "skeleton_in_network": False,
        "skeleton_fusion": "separate cross-fitted logit late fusion",
        "parameters": int(sum(value.numel() for value in SharedFullLocalModel().parameters())),
    }
    report = {
        "architecture": architecture,
        "fold_fp16_verification": fold_reports,
        "canonical_oof": {
            "samples": len(labels),
            "visual_fused_accuracy": float(
                accuracy_score(labels, predictions)
            ),
        },
    }
    (output_dir / "architecture_and_fp16_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
