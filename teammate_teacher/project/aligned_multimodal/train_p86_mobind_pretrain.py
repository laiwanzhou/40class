from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from p86_cached_motion_data import MOTION_FIELDS
from p86_mobind_lite_data import P86MoBindMotionDataset, collate_p86_mobind
from p86_mobind_lite_model import P86MoBindLite
from train_p86_visual_student_oof import class_weights, metric_dict


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MOTION = PROJECT_DIR / "runs/p86_motion_window_cache_t16_v1"
DEFAULT_TEACHER_LOGITS = (
    PROJECT_DIR
    / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)
DEFAULT_TEACHER_FEATURES = (
    PROJECT_DIR
    / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
)
DEFAULT_IMU_TEACHER = (
    PROJECT_DIR
    / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
)
TRAIN_USERS = {
    "user16",
    "user17",
    "user18",
    "user19",
    "user23",
    "user5",
    "user6",
    "user7",
    "user8",
}
PROXY_USERS = {"user20", "user22", "user24", "user3", "user4", "user9"}
PERMANENT_USERS = {"user1", "user2", "user21"}
TRAIN_USER_TO_INDEX = {
    user: index for index, user in enumerate(sorted(TRAIN_USERS))
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train P86 MoBind-lite on the fixed training-only subject split."
    )
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument(
        "--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES
    )
    parser.add_argument("--imu-teacher-logits", type=Path)
    parser.add_argument("--imu-event-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--alignment-width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--semantic-weight", type=float, default=2.0)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--local-weight", type=float, default=0.05)
    parser.add_argument("--global-weight", type=float, default=0.15)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.75)
    parser.add_argument("--teacher-feature-weight", type=float, default=0.25)
    parser.add_argument("--imu-teacher-weight", type=float, default=0.0)
    parser.add_argument("--imu-teacher-temperature", type=float, default=1.0)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--skeleton-initial-checkpoint",
        type=Path,
        help=(
            "Load only skeleton_* parameters from this checkpoint. This permits "
            "a strong Skeleton semantic anchor to be paired with a separately "
            "trained IMU branch before asymmetric alignment training."
        ),
    )
    parser.add_argument(
        "--imu-initial-checkpoint",
        type=Path,
        help="Load only imu_* parameters from this checkpoint.",
    )
    parser.add_argument("--freeze-skeleton-teacher", action="store_true")
    parser.add_argument("--imu-to-skeleton-token-weight", type=float, default=0.0)
    parser.add_argument("--imu-to-skeleton-semantic-weight", type=float, default=0.0)
    parser.add_argument("--imu-to-skeleton-logit-weight", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=float, default=0.35)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--imu-instance-normalization", action="store_true")
    parser.add_argument("--domain-adversarial-weight", type=float, default=0.0)
    parser.add_argument("--domain-reversal-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--final-refit",
        action="store_true",
        help=(
            "Train all 2470 non-permanent samples and evaluate the frozen "
            "user1/user2/user21 split once after fixed-epoch training."
        ),
    )
    parser.add_argument(
        "--defer-final-validation",
        action="store_true",
        help=(
            "With --final-refit, save the 2470-trained motion checkpoint without "
            "reading the permanent validation labels.  The unified final model "
            "can then perform the single terminal evaluation."
        ),
    )
    parser.add_argument(
        "--all-label-refit",
        action="store_true",
        help=(
            "Terminal refit on all 2914 true-labeled Train rows. Uses the fixed epoch "
            "count, starts from scratch and creates no validation loader."
        ),
    )
    parser.add_argument(
        "--subject-holdout-users",
        nargs="+",
        help=(
            "Leakage-safe P87-S protocol: train the motion model from scratch on all "
            "other subjects and evaluate these held subjects once after fixed epochs."
        ),
    )
    parser.add_argument(
        "--train-users",
        nargs="+",
        help=(
            "Optional explicit training-user allowlist for subject-holdout mode. "
            "All other users remain excluded and are never evaluated."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_branch(
    model: P86MoBindLite,
    checkpoint_path: Path,
    prefix: str,
) -> dict[str, Any]:
    """Load one complete modality branch and reject silent architecture drift."""
    checkpoint = torch.load(
        checkpoint_path.resolve(), map_location="cpu", weights_only=False
    )
    source = checkpoint["model_state"]
    target = model.state_dict()
    selected = {key: value for key, value in source.items() if key.startswith(prefix)}
    expected = {key for key in target if key.startswith(prefix)}
    missing = sorted(expected - selected.keys())
    unexpected = sorted(selected.keys() - expected)
    shape_mismatch = sorted(
        key
        for key, value in selected.items()
        if key in target and value.shape != target[key].shape
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"Incompatible {prefix} branch in {checkpoint_path}: "
            f"missing={missing}, unexpected={unexpected}, "
            f"shape_mismatch={shape_mismatch}"
        )
    model.load_state_dict(selected, strict=False)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "prefix": prefix,
        "tensors": len(selected),
        "parameters": sum(value.numel() for value in selected.values()),
    }


def loader(
    dataset: P86MoBindMotionDataset,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate_p86_mobind,
        # Proxy runs preserve their historical batching.  Both terminal refit
        # protocols must consume every labeled sample in every epoch.
        drop_last=(
            shuffle
            and not args.final_refit
            and not args.all_label_refit
            and not args.subject_holdout_users
            and len(dataset) >= args.batch_size
        ),
    )


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def motion_fields(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    motion = {field: batch[field] for field in MOTION_FIELDS}
    for field in ("imu_event_features", "imu_event_valid"):
        if field in batch:
            motion[field] = batch[field]
    return motion


def diagonal_contrastive(
    first: torch.Tensor,
    second: torch.Tensor,
    first_mask: torch.Tensor,
    second_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bidirectionally retrieve the matching index within each independent group."""

    logits = torch.matmul(first, second.transpose(-1, -2)) / temperature
    size = logits.shape[-1]
    target = torch.arange(size, device=logits.device)

    def direction(
        scores: torch.Tensor, query_mask: torch.Tensor, candidate_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_positive = query_mask & candidate_mask
        scores = scores.masked_fill(~candidate_mask.unsqueeze(-2), -1e4)
        flat_scores = scores.reshape(-1, size)
        flat_valid = valid_positive.reshape(-1)
        flat_target = target.view(*([1] * (scores.ndim - 2)), size).expand(
            *scores.shape[:-1]
        ).reshape(-1)
        if not flat_valid.any():
            zero = scores.sum() * 0.0
            return zero, zero.detach(), zero.detach()
        selected_scores = flat_scores[flat_valid]
        selected_target = flat_target[flat_valid]
        loss = F.cross_entropy(selected_scores, selected_target)
        correct = (selected_scores.argmax(dim=-1) == selected_target).sum()
        count = flat_valid.sum()
        return loss, correct, count

    forward_loss, forward_correct, forward_count = direction(
        logits, first_mask, second_mask
    )
    reverse_loss, reverse_correct, reverse_count = direction(
        logits.transpose(-1, -2), second_mask, first_mask
    )
    return (
        0.5 * (forward_loss + reverse_loss),
        forward_correct + reverse_correct,
        forward_count + reverse_count,
    )


def supervised_cross_modal_contrastive(
    first: torch.Tensor,
    second: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = torch.matmul(first, second.transpose(0, 1)) / temperature
    candidate_valid = valid.unsqueeze(0).expand_as(logits)
    positive = labels[:, None].eq(labels[None, :]) & candidate_valid
    query_valid = valid & positive.any(dim=1)
    logits = logits.masked_fill(~candidate_valid, -1e4)
    positive_logits = logits.masked_fill(~positive, -1e4)
    loss = -(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(logits, dim=1)
    )
    if not query_valid.any():
        return logits.sum() * 0.0
    forward = loss[query_valid].mean()
    reverse = supervised_cross_modal_contrastive_one_direction(
        logits.transpose(0, 1), positive.transpose(0, 1), valid
    )
    return 0.5 * (forward + reverse)


def supervised_cross_modal_contrastive_one_direction(
    logits: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    candidate_valid = valid.unsqueeze(0).expand_as(logits)
    positive = positive & candidate_valid
    query_valid = valid & positive.any(dim=1)
    logits = logits.masked_fill(~candidate_valid, -1e4)
    positive_logits = logits.masked_fill(~positive, -1e4)
    loss = -(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(logits, dim=1)
    )
    return loss[query_valid].mean() if query_valid.any() else logits.sum() * 0.0


def pooled_alignment(
    tokens: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = mask.to(tokens.dtype).unsqueeze(-1)
    pooled = (tokens * weight).sum(dim=dimensions) / weight.sum(
        dim=dimensions
    ).clamp_min(1.0)
    valid = mask.any(dim=dimensions)
    return F.normalize(pooled, dim=-1), valid


def reconstruction_loss(output: dict[str, torch.Tensor], modality: str) -> torch.Tensor:
    predicted = output[f"{modality}_reconstruction"]
    target = output[f"{modality}_reconstruction_target"]
    mask = output[f"{modality}_reconstruction_mask"]
    if not mask.any():
        return predicted.sum() * 0.0
    return F.smooth_l1_loss(predicted[mask], target[mask])


def distillation_loss(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float
) -> torch.Tensor:
    temperature = float(temperature)
    return F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2


def teacher_feature_loss(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def asymmetric_token_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    student_mask: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> torch.Tensor:
    """Match exact part/time pairs without moving the privileged teacher."""
    valid = student_mask & teacher_mask
    if not valid.any():
        return student.sum() * 0.0
    similarity = F.cosine_similarity(student, teacher.detach(), dim=-1)
    return (1.0 - similarity[valid]).mean()


def losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    weights: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    labels = batch["label"]
    skeleton_ce = F.cross_entropy(
        output["skeleton_logits"],
        labels,
        weight=weights,
        label_smoothing=args.label_smoothing,
    )
    imu_ce = F.cross_entropy(
        output["imu_logits"],
        labels,
        weight=weights,
        label_smoothing=args.label_smoothing,
    )
    skeleton_alignment = output["skeleton_alignment"]
    imu_alignment = output["imu_alignment"]
    skeleton_mask = output["skeleton_mask"]
    imu_mask = output["imu_mask"]

    # Time retrieval is performed independently for each sample/window/body part.
    skeleton_time = skeleton_alignment.permute(0, 1, 3, 2, 4)
    imu_time = imu_alignment.permute(0, 1, 3, 2, 4)
    skeleton_time_mask = skeleton_mask.permute(0, 1, 3, 2)
    imu_time_mask = imu_mask.permute(0, 1, 3, 2)
    token, token_correct, token_count = diagonal_contrastive(
        skeleton_time,
        imu_time,
        skeleton_time_mask,
        imu_time_mask,
        args.temperature,
    )

    # Body-part retrieval is performed after time pooling within each window.
    skeleton_local, skeleton_local_mask = pooled_alignment(
        skeleton_alignment, skeleton_mask, (2,)
    )
    imu_local, imu_local_mask = pooled_alignment(
        imu_alignment, imu_mask, (2,)
    )
    local, local_correct, local_count = diagonal_contrastive(
        skeleton_local,
        imu_local,
        skeleton_local_mask,
        imu_local_mask,
        args.temperature,
    )

    skeleton_global, skeleton_global_mask = pooled_alignment(
        skeleton_alignment, skeleton_mask, (1, 2, 3)
    )
    imu_global, imu_global_mask = pooled_alignment(
        imu_alignment, imu_mask, (1, 2, 3)
    )
    global_valid = skeleton_global_mask & imu_global_mask
    global_alignment = supervised_cross_modal_contrastive(
        skeleton_global,
        imu_global,
        labels,
        global_valid,
        args.temperature,
    )
    reconstruction = 0.5 * (
        reconstruction_loss(output, "skeleton")
        + reconstruction_loss(output, "imu")
    )
    teacher_logits = batch["teacher_logits"]
    distillation = 0.5 * (
        distillation_loss(
            output["skeleton_logits"],
            teacher_logits,
            args.distillation_temperature,
        )
        + distillation_loss(
            output["imu_logits"],
            teacher_logits,
            args.distillation_temperature,
        )
    )
    teacher_features = batch["teacher_features"]
    feature_distillation = 0.5 * (
        teacher_feature_loss(
            output["skeleton_teacher_features"], teacher_features
        )
        + teacher_feature_loss(output["imu_teacher_features"], teacher_features)
    )
    imu_teacher_distillation = output["imu_logits"].sum() * 0.0
    if args.imu_teacher_weight > 0 and "imu_teacher_logits" in batch:
        teacher_valid = batch["imu_teacher_valid"]
        if teacher_valid.any():
            imu_teacher_distillation = distillation_loss(
                output["imu_logits"][teacher_valid],
                batch["imu_teacher_logits"][teacher_valid],
                args.imu_teacher_temperature,
            )
    skeleton_token_distillation = asymmetric_token_distillation(
        output["imu_alignment"],
        output["skeleton_alignment"],
        imu_mask,
        skeleton_mask,
    )
    skeleton_semantic_distillation = teacher_feature_loss(
        output["imu_compact"], output["skeleton_compact"].detach()
    )
    skeleton_logit_distillation = distillation_loss(
        output["imu_logits"],
        output["skeleton_logits"].detach(),
        args.distillation_temperature,
    )
    semantic = 0.5 * (skeleton_ce + imu_ce)
    if "skeleton_domain_logits" in output:
        domain_target = torch.tensor(
            [TRAIN_USER_TO_INDEX[user] for user in batch["user_id"]],
            dtype=torch.long,
            device=labels.device,
        )
        domain = 0.5 * (
            F.cross_entropy(output["skeleton_domain_logits"], domain_target)
            + F.cross_entropy(output["imu_domain_logits"], domain_target)
        )
    else:
        domain = semantic * 0.0
    total = (
        args.semantic_weight * semantic
        + args.token_weight * token
        + args.local_weight * local
        + args.global_weight * global_alignment
        + args.reconstruction_weight * reconstruction
        + args.distillation_weight * distillation
        + args.teacher_feature_weight * feature_distillation
        + args.imu_teacher_weight * imu_teacher_distillation
        + args.imu_to_skeleton_token_weight * skeleton_token_distillation
        + args.imu_to_skeleton_semantic_weight * skeleton_semantic_distillation
        + args.imu_to_skeleton_logit_weight * skeleton_logit_distillation
        + args.domain_adversarial_weight * domain
    )
    return {
        "loss": total,
        "semantic": semantic,
        "skeleton_ce": skeleton_ce,
        "imu_ce": imu_ce,
        "token": token,
        "local": local,
        "global": global_alignment,
        "reconstruction": reconstruction,
        "distillation": distillation,
        "feature_distillation": feature_distillation,
        "imu_teacher_distillation": imu_teacher_distillation,
        "skeleton_token_distillation": skeleton_token_distillation,
        "skeleton_semantic_distillation": skeleton_semantic_distillation,
        "skeleton_logit_distillation": skeleton_logit_distillation,
        "domain": domain,
        "token_correct": token_correct.to(total.dtype),
        "token_count": token_count.to(total.dtype),
        "local_correct": local_correct.to(total.dtype),
        "local_count": local_count.to(total.dtype),
    }


def train(
    model: P86MoBindLite,
    data: DataLoader,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = class_weights(labels, args.class_weight_power, device)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_skeleton_teacher:
            # requires_grad_(False) does not disable dropout after the parent
            # model enters training mode, so keep the privileged teacher fixed.
            for module in (
                model.skeleton_encoder,
                model.skeleton_head,
                model.skeleton_projection,
                model.skeleton_teacher_projection,
                model.skeleton_context,
                model.skeleton_reconstruction,
            ):
                module.eval()
        ratio = args.minimum_learning_rate / args.learning_rate + 0.5 * (
            1.0 - args.minimum_learning_rate / args.learning_rate
        ) * (1.0 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1)))
        learning_rate = args.learning_rate * ratio
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        sums = {
            key: 0.0
            for key in (
                "loss",
                "semantic",
                "skeleton_ce",
                "imu_ce",
                "token",
                "local",
                "global",
                "reconstruction",
                "distillation",
                "feature_distillation",
                "imu_teacher_distillation",
                "skeleton_token_distillation",
                "skeleton_semantic_distillation",
                "skeleton_logit_distillation",
                "domain",
            )
        }
        token_correct = token_count = local_correct = local_count = 0.0
        samples = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(data):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(motion_fields(batch), mask_ratio=args.mask_ratio)
                values = losses(output, batch, weights, args)
            scaler.scale(values["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["label"])
            samples += count
            for key in sums:
                sums[key] += float(values[key].detach()) * count
            token_correct += float(values["token_correct"].detach())
            token_count += float(values["token_count"].detach())
            local_correct += float(values["local_correct"].detach())
            local_count += float(values["local_count"].detach())
        if args.all_label_refit and not args.max_train_batches and samples != len(data.dataset):
            raise RuntimeError(
                "all-label MoBind refit did not consume all 2914 rows in this epoch: "
                f"observed={samples}, expected={len(data.dataset)}"
            )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            **{f"train_{key}": value / max(samples, 1) for key, value in sums.items()},
            "train_token_retrieval": token_correct / max(token_count, 1.0),
            "train_part_retrieval": local_correct / max(local_count, 1.0),
            "train_samples": samples,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return history


def evaluate(
    model: P86MoBindLite,
    data: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows = []
    token_correct = token_count = local_correct = local_count = 0.0
    with torch.inference_mode():
        for batch_index, batch in enumerate(data):
            if args.max_eval_batches and batch_index >= args.max_eval_batches:
                break
            batch = to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(motion_fields(batch), mask_ratio=0.0)
            _, tc, tn = diagonal_contrastive(
                output["skeleton_alignment"].permute(0, 1, 3, 2, 4),
                output["imu_alignment"].permute(0, 1, 3, 2, 4),
                output["skeleton_mask"].permute(0, 1, 3, 2),
                output["imu_mask"].permute(0, 1, 3, 2),
                args.temperature,
            )
            skeleton_local, skeleton_local_mask = pooled_alignment(
                output["skeleton_alignment"], output["skeleton_mask"], (2,)
            )
            imu_local, imu_local_mask = pooled_alignment(
                output["imu_alignment"], output["imu_mask"], (2,)
            )
            _, lc, ln = diagonal_contrastive(
                skeleton_local,
                imu_local,
                skeleton_local_mask,
                imu_local_mask,
                args.temperature,
            )
            token_correct += float(tc)
            token_count += float(tn)
            local_correct += float(lc)
            local_count += float(ln)
            skeleton_probability = torch.softmax(
                output["skeleton_logits"].float(), dim=1
            ).cpu()
            imu_probability = torch.softmax(output["imu_logits"].float(), dim=1).cpu()
            labels = batch["label"].cpu()
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "user_id": batch["user_id"][index],
                        "label": int(labels[index]),
                        "skeleton_prediction": int(skeleton_probability[index].argmax()),
                        "skeleton_confidence": float(skeleton_probability[index].max()),
                        "imu_prediction": int(imu_probability[index].argmax()),
                        "imu_confidence": float(imu_probability[index].max()),
                    }
                )
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    users = [row["user_id"] for row in rows]
    metrics = {
        "skeleton": metric_dict(
            labels,
            np.asarray([row["skeleton_prediction"] for row in rows]),
            users,
        ),
        "imu": metric_dict(
            labels,
            np.asarray([row["imu_prediction"] for row in rows]),
            users,
        ),
        "token_retrieval_accuracy": token_correct / max(token_count, 1.0),
        "token_retrieval_count": int(token_count),
        "part_retrieval_accuracy": local_correct / max(local_count, 1.0),
        "part_retrieval_count": int(local_count),
    }
    return metrics, rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    subject_holdout_mode = bool(args.subject_holdout_users)
    all_label_mode = bool(args.all_label_refit)
    if args.train_users and not subject_holdout_mode:
        raise ValueError("--train-users requires --subject-holdout-users")
    if args.final_refit and all_label_mode:
        raise ValueError("--final-refit and --all-label-refit are mutually exclusive")
    if args.initial_checkpoint and (
        args.skeleton_initial_checkpoint or args.imu_initial_checkpoint
    ):
        raise ValueError(
            "--initial-checkpoint cannot be combined with modality-specific "
            "initial checkpoints"
        )
    if (args.final_refit or all_label_mode) and (
        args.initial_checkpoint
        or args.skeleton_initial_checkpoint
        or args.imu_initial_checkpoint
        or args.freeze_skeleton_teacher
    ):
        raise ValueError(
            "terminal refit must relearn both motion encoders from scratch; proxy "
            "checkpoints and a frozen Skeleton branch are not allowed"
        )
    if args.defer_final_validation and not args.final_refit:
        raise ValueError("--defer-final-validation requires --final-refit")
    if subject_holdout_mode and (
        args.final_refit or args.defer_final_validation or all_label_mode
    ):
        raise ValueError("subject holdout mode cannot be combined with final-refit modes")
    if all_label_mode and args.domain_adversarial_weight > 0:
        raise ValueError("all-label refit has no frozen 18-subject domain index")
    if subject_holdout_mode and (
        args.initial_checkpoint
        or args.skeleton_initial_checkpoint
        or args.imu_initial_checkpoint
        or args.freeze_skeleton_teacher
    ):
        raise ValueError(
            "P87-S subject holdout motion training must start from scratch; checkpoint "
            "initialization could contain held-subject label information"
        )
    if subject_holdout_mode and args.domain_adversarial_weight > 0:
        raise ValueError(
            "domain-adversarial user indexing is not defined for arbitrary holdout mode"
        )
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 2
    seed_all(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    full = P86MoBindMotionDataset(
        args.motion_cache,
        teacher_logits=args.teacher_logits,
        teacher_features=args.teacher_features,
        imu_teacher_logits=args.imu_teacher_logits,
        imu_event_features=args.imu_event_features,
    )
    users = full.users
    proxy_train_indices = np.flatnonzero(np.isin(users, sorted(TRAIN_USERS)))
    proxy_indices = np.flatnonzero(np.isin(users, sorted(PROXY_USERS)))
    permanent_indices = np.flatnonzero(np.isin(users, sorted(PERMANENT_USERS)))
    if (len(proxy_train_indices), len(proxy_indices), len(permanent_indices)) != (
        1497,
        973,
        444,
    ):
        raise RuntimeError("P86 fixed subject split changed")
    if subject_holdout_mode:
        requested_users = set(map(str, args.subject_holdout_users))
        observed_users = set(users[np.isin(users, sorted(requested_users))].tolist())
        if observed_users != requested_users:
            raise ValueError(
                f"Requested holdout users {sorted(requested_users)}, observed "
                f"{sorted(observed_users)}"
            )
        if args.train_users:
            requested_train_users = set(map(str, args.train_users))
            if requested_train_users & requested_users:
                raise ValueError("motion train and holdout user allowlists overlap")
            observed_train_users = set(
                users[np.isin(users, sorted(requested_train_users))].tolist()
            )
            if observed_train_users != requested_train_users:
                raise ValueError(
                    f"Requested train users {sorted(requested_train_users)}, observed "
                    f"{sorted(observed_train_users)}"
                )
            training_indices = np.flatnonzero(
                np.isin(users, sorted(requested_train_users))
            )
        else:
            training_indices = np.flatnonzero(
                ~np.isin(users, sorted(requested_users))
            )
        evaluation_indices = np.flatnonzero(np.isin(users, sorted(requested_users)))
        if set(users[training_indices].tolist()) & requested_users:
            raise RuntimeError("subject holdout isolation failed")
    elif all_label_mode:
        training_indices = np.arange(len(full), dtype=np.int64)
        evaluation_indices = np.empty(0, dtype=np.int64)
        if len(training_indices) != 2914:
            raise RuntimeError("terminal motion refit requires exactly 2914 labels")
    elif args.final_refit:
        training_indices = np.concatenate((proxy_train_indices, proxy_indices))
        evaluation_indices = permanent_indices
    else:
        training_indices = proxy_train_indices
        evaluation_indices = proxy_indices
    training = P86MoBindMotionDataset(
        args.motion_cache,
        training_indices,
        temporal_augment=True,
        teacher_logits=args.teacher_logits,
        teacher_features=args.teacher_features,
        imu_teacher_logits=args.imu_teacher_logits,
        imu_event_features=args.imu_event_features,
    )
    evaluation = None
    if not args.defer_final_validation and not all_label_mode:
        evaluation = P86MoBindMotionDataset(
            args.motion_cache,
            evaluation_indices,
            temporal_augment=False,
            teacher_logits=args.teacher_logits,
            teacher_features=args.teacher_features,
            imu_teacher_logits=args.imu_teacher_logits,
            imu_event_features=args.imu_event_features,
        )
    if args.imu_event_features:
        with np.load(args.imu_event_features.resolve(), allow_pickle=False) as data:
            imu_event_feature_width = int(data["features"].shape[1])
    else:
        imu_event_feature_width = 0
    model = P86MoBindLite(
        width=args.width,
        alignment_width=args.alignment_width,
        dropout=args.dropout,
        imu_instance_normalization=args.imu_instance_normalization,
        imu_event_feature_width=imu_event_feature_width,
        domain_classes=(
            len(TRAIN_USER_TO_INDEX) if args.domain_adversarial_weight > 0 else 0
        ),
        domain_reversal_scale=args.domain_reversal_scale,
    )
    initialization: list[dict[str, Any]] = []
    if args.initial_checkpoint:
        initial = torch.load(
            args.initial_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        model.load_state_dict(initial["model_state"], strict=True)
        initialization.append(
            {
                "checkpoint": str(args.initial_checkpoint.resolve()),
                "prefix": "all",
                "tensors": len(initial["model_state"]),
                "parameters": sum(
                    value.numel() for value in initial["model_state"].values()
                ),
            }
        )
    else:
        if args.skeleton_initial_checkpoint:
            initialization.append(
                load_checkpoint_branch(
                    model, args.skeleton_initial_checkpoint, "skeleton_"
                )
            )
        if args.imu_initial_checkpoint:
            initialization.append(
                load_checkpoint_branch(model, args.imu_initial_checkpoint, "imu_")
            )
    if args.freeze_skeleton_teacher:
        for name, parameter in model.named_parameters():
            if name.startswith("skeleton_"):
                parameter.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = train(
        model,
        loader(training, args, True),
        training.labels,
        args,
        device,
    )
    metrics = None
    predictions: list[dict[str, Any]] = []
    if not args.defer_final_validation and not all_label_mode:
        assert evaluation is not None
        metrics, predictions = evaluate(
            model, loader(evaluation, args, False), args, device
        )
    write_rows(output / "training_history.csv", history)
    if predictions:
        write_rows(
            output
            / (
                "subject_holdout_predictions.csv"
                if subject_holdout_mode
                else (
                    "final_validation_predictions.csv"
                    if args.final_refit
                    else "proxy_predictions.csv"
                )
            ),
            predictions,
        )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    stage = (
        "P87S_mobind_subject_holdout"
        if subject_holdout_mode
        else (
            "P87S_mobind_all2914_refit"
            if all_label_mode
            else (
                "P86_mobind_lite_pretrain_final2470"
                if args.final_refit
                else "P86_mobind_lite_pretrain"
            )
        )
    )
    checkpoint = {
        "stage": stage,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": {
            "width": args.width,
            "alignment_width": args.alignment_width,
            "classes": 40,
            "dropout": args.dropout,
            "imu_instance_normalization": args.imu_instance_normalization,
            "imu_event_feature_width": imu_event_feature_width,
            "domain_classes": (
                len(TRAIN_USER_TO_INDEX)
                if args.domain_adversarial_weight > 0
                else 0
            ),
            "domain_reversal_scale": args.domain_reversal_scale,
        },
        "initialization": initialization,
        "proxy_metrics": (
            metrics
            if not args.final_refit and not subject_holdout_mode and not all_label_mode
            else None
        ),
        "final_validation_metrics": metrics if args.final_refit else None,
        "subject_holdout_metrics": metrics if subject_holdout_mode else None,
        "training_subjects": sorted(set(users[training_indices].tolist())),
        "holdout_subjects": sorted(set(users[evaluation_indices].tolist())),
        "excluded_subjects": sorted(
            set(users.tolist())
            - set(users[training_indices].tolist())
            - set(users[evaluation_indices].tolist())
        ),
    }
    torch.save(checkpoint, output / "mobind_lite.pt")
    summary = {
        "stage": stage,
        "status": (
            "smoke"
            if args.smoke
            else (
                "formal_subject_holdout"
                if subject_holdout_mode
                else (
                    "formal_terminal_refit"
                    if all_label_mode
                    else (
                        "formal_final_pretrain"
                        if args.final_refit and args.defer_final_validation
                        else ("formal_final" if args.final_refit else "formal_proxy")
                    )
                )
            )
        ),
        "protocol": (
            "Train a fresh MoBind motion model on every subject except the explicit "
            "P87-S pseudo-Test subjects, then read held labels once after fixed-epoch "
            "training. No held-subject checkpoint initialization or epoch selection."
            if subject_holdout_mode
            else (
            "Terminal refit from scratch on all 2914 true-labeled Train rows with a "
            "pre-frozen epoch count. No validation loader, early stop, Test input or "
            "checkpoint selection is used."
            if all_label_mode
            else
            "Train all 15 non-permanent subjects from scratch. Permanent validation "
            "is deferred to the terminal unified-model evaluation."
            if args.final_refit and args.defer_final_validation
            else "Train all 15 non-permanent subjects and evaluate fixed user1/user2/"
            "user21 exactly once after fixed-epoch training. No validation-driven "
            "selection is performed."
            if args.final_refit
            else "Train nine candidate-train subjects and evaluate six disjoint proxy "
            "subjects exactly once after fixed-epoch training. Permanent user1/user2/"
            "user21 validation is never indexed by training or evaluation."
            )
        ),
        "counts": {
            "train": len(training),
            (
                "subject_holdout"
                if subject_holdout_mode
                else (
                    "validation"
                    if all_label_mode
                    else (
                        "final_validation_held_out"
                        if args.final_refit and args.defer_final_validation
                        else ("final_validation" if args.final_refit else "proxy")
                    )
                )
            ): len(evaluation_indices),
            "permanent_untouched": (
                len(permanent_indices)
                if not subject_holdout_mode and not all_label_mode
                and (args.defer_final_validation or not args.final_refit)
                else 0
            ),
        },
        "training_subjects": sorted(set(users[training_indices].tolist())),
        "holdout_subjects": sorted(set(users[evaluation_indices].tolist())),
        "excluded_subjects": sorted(
            set(users.tolist())
            - set(users[training_indices].tolist())
            - set(users[evaluation_indices].tolist())
        ),
        "proxy_metrics": (
            metrics
            if not args.final_refit and not subject_holdout_mode and not all_label_mode
            else None
        ),
        "final_validation_metrics": metrics if args.final_refit else None,
        "subject_holdout_metrics": metrics if subject_holdout_mode else None,
        "parameters": parameters,
        "fp32_mib": parameters * 4 / 1024**2,
        "initialization": initialization,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
