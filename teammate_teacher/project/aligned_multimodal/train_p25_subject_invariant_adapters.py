from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .audit_subject_generalization_ceiling import load_and_align_caches
except ImportError:
    from audit_subject_generalization_ceiling import load_and_align_caches


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
MODALITIES = ("skeleton", "depth", "thermal", "imu")
VARIANTS = ("P25-C", "P25-S", "P25-A")
EMBEDDING_KEYS = {
    "skeleton": "skeleton_embedding",
    "depth": "depth_embedding",
    "thermal": "thermal_embedding",
    "imu": "imu_embedding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P25 frozen-feature subject-invariant adapters")
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
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p25_subject_invariant_adapters",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--folds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One epoch and one step; writes to a separate path.")
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


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return values.view_as(values)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * grad_output, None


def gradient_reverse(values: torch.Tensor, strength: float) -> torch.Tensor:
    return GradientReversalFunction.apply(values, strength)


class SubjectInvariantAdapter(nn.Module):
    def __init__(
        self,
        raw_dim: int,
        subject_count: int,
        hidden_dim: int = 128,
        representation_dim: int = 64,
        dropout: float = 0.20,
        action_classes: int = 40,
    ) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, representation_dim),
        )
        self.action_head = nn.Linear(representation_dim, action_classes)
        self.subject_head = nn.Linear(representation_dim, subject_count)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.adapter(values), dim=1)

    def forward(
        self,
        values: torch.Tensor,
        grl_strength: float = 0.0,
        use_subject_head: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        representation = self.encode(values)
        action_logits = self.action_head(representation)
        subject_logits = None
        if use_subject_head:
            subject_logits = self.subject_head(gradient_reverse(representation, grl_strength))
        return action_logits, subject_logits, representation


def cross_subject_supervised_contrastive_loss(
    representation: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    batch_size = representation.shape[0]
    nonself = ~torch.eye(batch_size, dtype=torch.bool, device=representation.device)
    same_class = labels[:, None].eq(labels[None, :])
    different_subject = subjects[:, None].ne(subjects[None, :])
    positive = same_class & different_subject & nonself
    valid_anchor = positive.any(dim=1)

    logits = representation @ representation.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * nonself
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive.sum(dim=1)
    per_anchor = -(log_prob * positive).sum(dim=1) / positive_count.clamp_min(1)
    loss = per_anchor[valid_anchor].mean() if valid_anchor.any() else representation.sum() * 0.0
    possible_pairs = int(nonself.sum().item())
    positive_pairs = int(positive.sum().item())
    return loss, {
        "anchors": int(batch_size),
        "valid_anchors": int(valid_anchor.sum().item()),
        "valid_anchor_fraction": float(valid_anchor.float().mean().item()),
        "positive_directed_pairs": positive_pairs,
        "positive_pair_fraction": float(positive_pairs / possible_pairs) if possible_pairs else 0.0,
    }


def class_subject_index(
    indices: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    output: dict[int, dict[str, np.ndarray]] = {}
    for class_id in sorted(np.unique(labels[indices]).astype(int).tolist()):
        subject_map: dict[str, np.ndarray] = {}
        class_indices = indices[labels[indices] == class_id]
        for subject in sorted(np.unique(subjects[class_indices]).astype(str).tolist()):
            subject_map[subject] = class_indices[subjects[class_indices] == subject]
        output[class_id] = subject_map
    return output


def balanced_batches(
    index: dict[int, dict[str, np.ndarray]],
    steps: int,
    classes_per_batch: int,
    subject_slots_per_class: int,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    classes = np.asarray(sorted(index), dtype=np.int64)
    if len(classes) < classes_per_batch:
        raise ValueError(f"Only {len(classes)} classes, need {classes_per_batch}")
    batches: list[np.ndarray] = []
    fallback_slots = 0
    sampled_slots = 0
    sampled_classes: set[int] = set()
    for _ in range(steps):
        chosen_classes = rng.choice(classes, size=classes_per_batch, replace=False)
        batch: list[int] = []
        for class_id in chosen_classes.tolist():
            sampled_classes.add(int(class_id))
            subject_map = index[int(class_id)]
            available = np.asarray(sorted(subject_map), dtype=str)
            replace = len(available) < subject_slots_per_class
            if replace:
                extra_count = subject_slots_per_class - len(available)
                chosen_subjects = np.concatenate(
                    [rng.permutation(available), rng.choice(available, size=extra_count, replace=True)]
                )
                rng.shuffle(chosen_subjects)
                fallback_slots += extra_count
            else:
                chosen_subjects = rng.choice(
                    available,
                    size=subject_slots_per_class,
                    replace=False,
                )
            sampled_slots += subject_slots_per_class
            for subject in chosen_subjects.tolist():
                candidates = subject_map[str(subject)]
                batch.append(int(rng.choice(candidates)))
        batch_array = np.asarray(batch, dtype=np.int64)
        rng.shuffle(batch_array)
        batches.append(batch_array)
    no_cross_subject_classes = [
        int(class_id) for class_id, subject_map in index.items() if len(subject_map) < 2
    ]
    return batches, {
        "steps": int(steps),
        "sampled_slots": int(sampled_slots),
        "fallback_repeated_subject_slots": int(fallback_slots),
        "classes_sampled": sorted(sampled_classes),
        "no_cross_subject_positive_classes": no_cross_subject_classes,
    }


@torch.no_grad()
def infer_all(
    model: SubjectInvariantAdapter,
    features: np.ndarray | torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    representations: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        feature_slice = features[start : start + batch_size]
        values = (
            feature_slice
            if isinstance(feature_slice, torch.Tensor)
            else torch.from_numpy(feature_slice).to(device=device)
        )
        action_logits, _, representation = model(values)
        logits.append(action_logits.cpu().numpy().astype(np.float32))
        representations.append(representation.cpu().numpy().astype(np.float32))
    return np.concatenate(logits), np.concatenate(representations)


def train_one(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    train_mask: np.ndarray,
    held_mask: np.ndarray,
    raw_dim: int,
    variant: str,
    fold: int,
    config: dict[str, Any],
    device: torch.device,
    unit_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    training = config["training"]
    adapter_config = config["adapter"]
    variant_config = config["variants"][variant]
    sampler_config = config["sampler"]
    supcon_config = config["supervised_contrastive"]
    grl_config = config["gradient_reversal"]
    seed = int(training["seed"])
    seed_everything(seed, bool(training["deterministic_algorithms"]))

    train_indices = np.flatnonzero(train_mask)
    held_indices = np.flatnonzero(held_mask)
    subject_names = sorted(np.unique(subjects[train_indices]).astype(str).tolist())
    subject_to_index = {subject: index for index, subject in enumerate(subject_names)}
    subject_targets = np.asarray([subject_to_index.get(subject, -1) for subject in subjects], dtype=np.int64)
    indexed = class_subject_index(train_indices, labels, subjects)
    # The frozen cache is small enough to remain on device. Indexing the resident
    # tensors is numerically equivalent to copying each selected NumPy batch, while
    # avoiding 800 synchronous host-to-device transfers per fold-unit.
    feature_tensor = torch.from_numpy(features).to(device=device)
    label_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.long)
    subject_target_tensor = torch.from_numpy(subject_targets).to(
        device=device, dtype=torch.long
    )

    model = SubjectInvariantAdapter(
        raw_dim=raw_dim,
        subject_count=len(subject_names),
        hidden_dim=int(adapter_config["hidden_dim"]),
        representation_dim=int(adapter_config["representation_dim"]),
        dropout=float(adapter_config["dropout"]),
        action_classes=int(adapter_config["action_classes"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    epochs = 1 if smoke else int(training["epochs"])
    normal_steps = int(math.ceil(len(train_indices) / int(training["batch_size"])))
    steps = 1 if smoke else normal_steps
    curves: list[dict[str, Any]] = []
    start_time = time.time()
    no_positive_batches = 0
    aggregate_valid_anchors = 0
    aggregate_anchors = 0
    aggregate_positive_pairs = 0
    aggregate_possible_pairs = 0

    for epoch_index in range(epochs):
        model.train()
        batches, sampler_stats = balanced_batches(
            indexed,
            steps=steps,
            classes_per_batch=int(sampler_config["classes_per_batch"]),
            subject_slots_per_class=int(sampler_config["subject_slots_per_class"]),
            seed=seed + fold * 1000 + epoch_index,
        )
        grl_strength = (
            float(grl_config["end"]) * epoch_index / max(1, int(training["epochs"]) - 1)
            if variant == "P25-A"
            else 0.0
        )
        sums = defaultdict(float)
        epoch_correct = 0
        epoch_samples = 0
        epoch_valid_anchors = 0
        epoch_anchors = 0
        epoch_positive_pairs = 0
        epoch_possible_pairs = 0
        epoch_no_positive_batches = 0

        for batch_indices in batches:
            batch_index_tensor = torch.from_numpy(batch_indices).to(
                device=device, dtype=torch.long
            )
            values = feature_tensor.index_select(0, batch_index_tensor)
            action_targets = label_tensor.index_select(0, batch_index_tensor)
            batch_subjects = subject_target_tensor.index_select(0, batch_index_tensor)
            use_subject = variant == "P25-A"
            action_logits, subject_logits, representation = model(
                values,
                grl_strength=grl_strength,
                use_subject_head=use_subject,
            )
            action_loss = F.cross_entropy(action_logits, action_targets)
            if variant in ("P25-S", "P25-A"):
                contrastive_loss, positive_stats = cross_subject_supervised_contrastive_loss(
                    representation,
                    action_targets,
                    batch_subjects,
                    temperature=float(supcon_config["temperature"]),
                )
            else:
                contrastive_loss = representation.sum() * 0.0
                positive_stats = {
                    "anchors": len(batch_indices),
                    "valid_anchors": 0,
                    "valid_anchor_fraction": 0.0,
                    "positive_directed_pairs": 0,
                    "positive_pair_fraction": 0.0,
                }
            subject_loss = (
                F.cross_entropy(subject_logits, batch_subjects)
                if subject_logits is not None
                else representation.sum() * 0.0
            )
            total_loss = (
                action_loss
                + float(variant_config["cross_subject_supcon_weight"]) * contrastive_loss
                + float(variant_config["subject_adversarial_ce_weight"]) * subject_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            sums["total_loss"] += float(total_loss.detach().item())
            sums["action_loss"] += float(action_loss.detach().item())
            sums["contrastive_loss"] += float(contrastive_loss.detach().item())
            sums["subject_loss"] += float(subject_loss.detach().item())
            epoch_correct += int((action_logits.argmax(1) == action_targets).sum().item())
            epoch_samples += int(len(batch_indices))
            valid_anchors = int(positive_stats["valid_anchors"])
            anchors = int(positive_stats["anchors"])
            positive_pairs = int(positive_stats["positive_directed_pairs"])
            epoch_valid_anchors += valid_anchors
            epoch_anchors += anchors
            epoch_positive_pairs += positive_pairs
            epoch_possible_pairs += anchors * max(0, anchors - 1)
            if variant in ("P25-S", "P25-A") and valid_anchors == 0:
                epoch_no_positive_batches += 1

        no_positive_batches += epoch_no_positive_batches
        aggregate_valid_anchors += epoch_valid_anchors
        aggregate_anchors += epoch_anchors
        aggregate_positive_pairs += epoch_positive_pairs
        aggregate_possible_pairs += epoch_possible_pairs
        curves.append(
            {
                "epoch": epoch_index + 1,
                "total_loss": sums["total_loss"] / steps,
                "action_loss": sums["action_loss"] / steps,
                "contrastive_loss": sums["contrastive_loss"] / steps,
                "subject_loss": sums["subject_loss"] / steps,
                "train_sampled_accuracy": float(epoch_correct / epoch_samples),
                "grl_strength": float(grl_strength),
                "valid_cross_subject_positive_anchor_fraction": (
                    float(epoch_valid_anchors / epoch_anchors) if epoch_anchors else 0.0
                ),
                "cross_subject_positive_pair_fraction": (
                    float(epoch_positive_pairs / epoch_possible_pairs)
                    if epoch_possible_pairs
                    else 0.0
                ),
                "no_valid_positive_batches": int(epoch_no_positive_batches),
                "sampler": sampler_stats,
            }
        )

    all_logits, all_representations = infer_all(model, feature_tensor, device)
    held_predictions = all_logits[held_indices].argmax(1)
    held_accuracy = float((held_predictions == labels[held_indices]).mean())
    elapsed = float(time.time() - start_time)
    unit_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = unit_dir / "final_epoch.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "raw_dim": raw_dim,
            "subject_names": subject_names,
            "variant": variant,
            "fold": fold,
            "config": config,
            "epoch": epochs,
            "fixed_final_epoch": not smoke,
        },
        checkpoint_path,
    )
    np.savez_compressed(
        unit_dir / "fold_conditioned_outputs.npz",
        logits=all_logits.astype(np.float32),
        representations=all_representations.astype(np.float16),
    )
    result = {
        "variant": variant,
        "fold": fold,
        "raw_dim": raw_dim,
        "train_samples": int(len(train_indices)),
        "held_samples": int(len(held_indices)),
        "train_subjects": subject_names,
        "held_subjects": sorted(np.unique(subjects[held_indices]).astype(str).tolist()),
        "epochs": epochs,
        "steps_per_epoch": steps,
        "held_fold_accuracy_final_epoch": held_accuracy,
        "held_fold_evaluations_during_training": 0,
        "effective_cross_subject_positive_anchor_fraction": (
            float(aggregate_valid_anchors / aggregate_anchors) if aggregate_anchors else 0.0
        ),
        "cross_subject_positive_pair_fraction": (
            float(aggregate_positive_pairs / aggregate_possible_pairs)
            if aggregate_possible_pairs
            else 0.0
        ),
        "no_valid_positive_batches": int(no_positive_batches),
        "classes_without_cross_subject_positive": [
            int(class_id) for class_id, values in indexed.items() if len(values) < 2
        ],
        "classes_with_fewer_than_four_subjects": [
            int(class_id) for class_id, values in indexed.items() if len(values) < 4
        ],
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "curves": curves,
    }
    write_json(unit_dir / "fold_summary.json", result)
    return result


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    config = load_json(args.config)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    caches, cache_audit = load_and_align_caches(args.cache_dir)
    reference = caches[0]
    sample_ids = reference["sample_ids"].astype(str)
    labels = reference["labels"].astype(np.int64)
    subjects = reference["subjects"].astype(str)
    folds = reference["folds"].astype(np.int64)
    modality_order = tuple(reference["modality_order"].astype(str).tolist())
    if modality_order != MODALITIES:
        raise ValueError(f"Unexpected modality order: {modality_order}")

    output_dir = args.output_dir / "smoke" if args.smoke else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_protocol = {
        "config": config,
        "config_path": str(args.config.relative_to(REPO_ROOT)),
        "config_sha256": sha256(args.config),
        "cache_audit": cache_audit,
        "cache_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in sorted(args.cache_dir.glob("fold_*_cache.npz"))
        },
        "training_code_commit": git_commit(),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "smoke": bool(args.smoke),
    }
    write_json(output_dir / "resolved_protocol.json", resolved_protocol)

    summaries: dict[str, Any] = {}
    for modality in args.modalities:
        modality_index = MODALITIES.index(modality)
        raw_dim = int(config["modalities"][modality])
        summaries[modality] = {}
        for variant in args.variants:
            variant_dir = output_dir / modality / variant
            if variant_dir.exists() and not args.overwrite:
                raise FileExistsError(f"{variant_dir} exists; refusing to overwrite")
            summaries[modality][variant] = {}
            oof_logits = np.full((len(sample_ids), 40), np.nan, dtype=np.float32)
            oof_representations = np.full((len(sample_ids), 64), np.nan, dtype=np.float32)
            valid_oof = np.zeros(len(sample_ids), dtype=bool)
            fold_conditioned_representations = np.zeros((3, len(sample_ids), 64), dtype=np.float16)
            fold_conditioned_valid = np.zeros((3, len(sample_ids)), dtype=bool)

            for fold in args.folds:
                cache = caches[fold]
                features = np.ascontiguousarray(
                    cache[EMBEDDING_KEYS[modality]].astype(np.float32)
                )
                if features.shape != (len(sample_ids), raw_dim):
                    raise ValueError(f"{modality} fold {fold} feature shape {features.shape}")
                presence = cache["presence"][:, modality_index].astype(bool)
                train_mask = (folds != fold) & presence
                held_mask = (folds == fold) & presence
                result = train_one(
                    features=features,
                    labels=labels,
                    subjects=subjects,
                    train_mask=train_mask,
                    held_mask=held_mask,
                    raw_dim=raw_dim,
                    variant=variant,
                    fold=fold,
                    config=config,
                    device=device,
                    unit_dir=variant_dir / f"fold_{fold}",
                    smoke=args.smoke,
                )
                with np.load(
                    variant_dir / f"fold_{fold}" / "fold_conditioned_outputs.npz",
                    allow_pickle=False,
                ) as archive:
                    all_logits = archive["logits"].astype(np.float32)
                    all_representations = archive["representations"].astype(np.float32)
                oof_logits[held_mask] = all_logits[held_mask]
                oof_representations[held_mask] = all_representations[held_mask]
                valid_oof[held_mask] = True
                fold_conditioned_representations[fold] = all_representations.astype(np.float16)
                fold_conditioned_valid[fold] = presence
                summaries[modality][variant][str(fold)] = result

            np.savez_compressed(
                variant_dir / "oof_outputs.npz",
                sample_ids=sample_ids,
                labels=labels,
                subjects=subjects,
                folds=folds,
                valid_mask=valid_oof,
                logits=oof_logits,
                predictions=np.asarray(
                    [
                        int(np.argmax(oof_logits[index])) if valid_oof[index] else -1
                        for index in range(len(sample_ids))
                    ],
                    dtype=np.int64,
                ),
                representations=oof_representations.astype(np.float16),
            )
            np.savez_compressed(
                variant_dir / "fold_conditioned_representations.npz",
                sample_ids=sample_ids,
                folds=folds,
                valid=fold_conditioned_valid,
                representations=fold_conditioned_representations,
            )
            write_json(variant_dir / "training_summary.json", summaries[modality][variant])

    write_json(
        output_dir / "training_summary.json",
        {
            "protocol_version": config["protocol_version"],
            "modalities": summaries,
            "protocol_deviation": False,
            "held_fold_model_selection": False,
            "completed_at_unix": time.time(),
        },
    )


if __name__ == "__main__":
    main()
