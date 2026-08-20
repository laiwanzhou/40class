from __future__ import annotations

from collections.abc import Iterable
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from src.models.thermal_multistream import ThermalMultiStreamStudent
from src.models.thermal_x3d_xs import ThermalX3DXSBaseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTES = {"b_x3d_xs", "a_multistream"}
OBJECTIVES = {"direct", "kd"}
CHECKPOINT_RANK = [
    "macro_f1_fixed_0_39",
    "accuracy",
    "worst_user_accuracy",
    "lower_epoch",
]
TENSOR_SHAPES = {
    "full_rgb": [3, 16, 3, 160, 160],
    "crop_rgb": [3, 16, 3, 160, 160],
    "motion": [3, 16, 1, 160, 160],
    "pose": [3, 16, 56],
    "availability": [4],
    "quality": [8],
}


class TrainingAuthorizationError(PermissionError):
    """Raised before any output is created when formal training is not authorized."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_generation2_config(config: dict[str, Any]) -> None:
    _require(config.get("schema_version") == 1, "schema_version must be 1")
    _require(config.get("route") in ROUTES, "unsupported generation-2 route")
    _require(config.get("objective") in OBJECTIVES, "unsupported objective")
    student = config.get("student", {})
    data = config.get("data", {})
    optimization = config.get("optimization", {})
    _require(student.get("initialization") == "random", "student must initialize randomly")
    _require(student.get("pretrained") is False, "pretrained student weights are forbidden")
    _require(student.get("num_classes") == 40, "student must emit 40 logits")
    _require(data.get("modality") == "thermal_only", "only Thermal inputs are allowed")
    _require(
        data.get("development_split")
        == "metadata/splits/train12_val2_user6_user7_development.json",
        "development split must remain fixed",
    )
    _require(data.get("class_ids") == list(range(40)), "class map must be fixed to 0..39")
    _require(data.get("tensor_shapes") == TENSOR_SHAPES, "v2 tensor shapes changed")
    for field in (
        "read_heldout4_labels",
        "read_competition_test",
        "read_quarantined_evidence",
        "import_ir_depth_inputs",
    ):
        _require(data.get(field) is False, f"forbidden data flag must be false: {field}")
    _require(optimization.get("seed") == 20260715, "seed changed")
    _require(optimization.get("maximum_epochs") == 50, "maximum_epochs must be 50")
    _require(optimization.get("automatic_resume") is False, "automatic resume is forbidden")
    _require(
        optimization.get("automatic_extension") is False,
        "automatic extension is forbidden",
    )
    _require(
        optimization.get("checkpoint_rank") == CHECKPOINT_RANK,
        "checkpoint rank changed",
    )
    if config["objective"] == "kd":
        _require(config["route"] == "a_multistream", "KD is only defined for route A")


def load_generation2_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("generation-2 config must be a mapping")
    validate_generation2_config(config)
    config["config_path"] = str(path.resolve())
    return config


def require_training_authorization(config: dict[str, Any], *, token: str | None) -> None:
    if config.get("training_authorized") is not True:
        raise TrainingAuthorizationError("configuration does not authorize formal training")
    if not token or token != config.get("authorization_token"):
        raise TrainingAuthorizationError("exact training authorization token is required")


def build_generation2_student(config: dict[str, Any]) -> nn.Module:
    validate_generation2_config(config)
    if config["route"] == "b_x3d_xs":
        model = ThermalX3DXSBaseline(num_classes=40)
        model.initialization_provenance = {
            "student": "random",
            "pretrained_student_weights": False,
            "raster": model.backbone.initialization_provenance,
        }
        return model
    return ThermalMultiStreamStudent(num_classes=40)


def compute_generation2_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    eligible: torch.Tensor,
    objective: str,
    label_smoothing: float,
    teacher_logits: torch.Tensor | None = None,
    temperature: float = 4.0,
    hard_label_weight: float = 0.5,
    teacher_kl_weight: float = 0.5,
) -> torch.Tensor:
    if objective not in OBJECTIVES:
        raise ValueError("unsupported generation-2 objective")
    if logits.ndim != 2 or labels.shape != logits.shape[:1] or eligible.shape != labels.shape:
        raise ValueError("loss inputs have incompatible shapes")
    mask = eligible.to(device=logits.device, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("batch contains no loss-eligible sample")
    hard = functional.cross_entropy(
        logits,
        labels,
        reduction="none",
        label_smoothing=label_smoothing,
    )[mask].mean()
    if objective == "direct":
        return hard
    if teacher_logits is None or teacher_logits.shape != logits.shape:
        raise ValueError("KD requires matching fixed teacher logits")
    if temperature <= 0:
        raise ValueError("KD temperature must be positive")
    per_sample_kl = functional.kl_div(
        functional.log_softmax(logits / temperature, dim=1),
        functional.softmax(teacher_logits.to(logits.device) / temperature, dim=1),
        reduction="none",
    ).sum(dim=1) * temperature**2
    return hard_label_weight * hard + teacher_kl_weight * per_sample_kl[mask].mean()


def fixed_label_metrics(
    *, labels: np.ndarray, logits: np.ndarray, users: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    users = np.asarray(users, dtype=str)
    if logits.shape != (len(labels), 40) or users.shape != labels.shape:
        raise ValueError("prediction archive must contain N x 40 logits and N users")
    predictions = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    matrix = confusion_matrix(labels, predictions, labels=np.arange(40))
    totals = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        totals,
        out=np.zeros(40, dtype=np.float64),
        where=totals != 0,
    )
    user_accuracy = {
        user: float(accuracy_score(labels[users == user], predictions[users == user]))
        for user in sorted(set(users.tolist()))
    }
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "nll": float(-log_probabilities[np.arange(len(labels)), labels].mean()),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(40),
                average="macro",
                zero_division=0,
            )
        ),
        "worst_user_accuracy": min(user_accuracy.values()),
        "user_accuracy": user_accuracy,
        "per_class_recall": recalls.tolist(),
        "zero_recall_classes": int((recalls == 0).sum()),
        "confusion_matrix": matrix.tolist(),
    }


def checkpoint_rank(metrics: dict[str, Any], *, epoch: int) -> tuple[float, float, float, int]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
        float(metrics["worst_user_accuracy"]),
        -int(epoch),
    )


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer_and_scheduler(
    model: nn.Module, *, config: dict[str, Any], steps_per_epoch: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    validate_generation2_config(config)
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    warmup_steps = int(optimization["warmup_epochs"]) * steps_per_epoch
    total_steps = int(optimization["maximum_epochs"]) * steps_per_epoch

    def scale(step: int) -> float:
        completed = step + 1
        if warmup_steps and completed <= warmup_steps:
            return completed / warmup_steps
        progress = (completed - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


MODEL_INPUT_KEYS = (
    "full_rgb",
    "crop_rgb",
    "motion",
    "pose",
    "window_mask",
    "pose_mask",
    "availability",
    "quality",
)


def _forward_generation2(
    model: nn.Module, batch: dict[str, Any], route: str, device: torch.device
) -> dict[str, torch.Tensor]:
    values = {
        key: batch[key].to(device, non_blocking=True)
        for key in MODEL_INPUT_KEYS
        if key in batch
    }
    if route == "b_x3d_xs":
        return model(
            values["full_rgb"],
            window_mask=values["window_mask"],
            availability=values["availability"],
            quality=values["quality"],
        )
    return model(**values)


def run_generation2_epoch(
    *,
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    validate_generation2_config(config)
    training = optimizer is not None
    model.train(training)
    optimization = config["optimization"]
    physical = int(optimization["physical_batch_trials"])
    effective = int(optimization["effective_batch_trials"])
    if effective % physical:
        raise ValueError("effective batch must be divisible by physical batch")
    accumulation = effective // physical
    if training:
        optimizer.zero_grad(set_to_none=True)
    labels_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    users_all: list[np.ndarray] = []
    loss_sum = 0.0
    eligible_samples = 0
    batches = 0
    optimizer_steps = 0
    pending = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)
            eligible = batch["loss_eligible"].to(device, non_blocking=True).bool()
            if not bool(eligible.any()):
                batches += 1
                continue
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=training and device.type == "cuda",
            ):
                output = _forward_generation2(model, batch, config["route"], device)
                loss = compute_generation2_loss(
                    logits=output["logits"],
                    labels=labels,
                    eligible=eligible,
                    objective=config["objective"],
                    label_smoothing=float(optimization["label_smoothing"]),
                    teacher_logits=batch.get("teacher_logits"),
                    temperature=float(config.get("distillation", {}).get("temperature", 4.0)),
                    hard_label_weight=float(
                        config.get("distillation", {}).get("hard_label_weight", 0.5)
                    ),
                    teacher_kl_weight=float(
                        config.get("distillation", {}).get("teacher_kl_weight", 0.5)
                    ),
                )
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(output["logits"]).all()):
                raise FloatingPointError("non-finite generation-2 loss or logits")
            count = int(eligible.sum().item())
            loss_sum += float(loss.detach()) * count
            eligible_samples += count
            selected = eligible.detach().cpu().numpy().astype(bool)
            labels_all.append(labels.detach().cpu().numpy()[selected])
            logits_all.append(output["logits"].detach().float().cpu().numpy()[selected])
            users_all.append(np.asarray(batch["user_id"], dtype=str)[selected])
            if training:
                (loss / accumulation).backward()
                pending += 1
                if pending == accumulation:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(optimization["gradient_clip_norm"])
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    optimizer_steps += 1
                    pending = 0
            batches += 1
    if batches == 0 or eligible_samples == 0:
        raise ValueError("loader produced no loss-eligible sample")
    if training and pending:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(optimization["gradient_clip_norm"])
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        optimizer_steps += 1
    metrics = fixed_label_metrics(
        labels=np.concatenate(labels_all),
        logits=np.concatenate(logits_all),
        users=np.concatenate(users_all),
    )
    metrics.update(
        {
            "loss": loss_sum / eligible_samples,
            "eligible_samples": eligible_samples,
            "batches": batches,
            "optimizer_steps": optimizer_steps,
        }
    )
    return metrics


def collect_generation2_predictions(
    *,
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    validate_generation2_config(config)
    model.eval()
    sample_ids: list[np.ndarray] = []
    users: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    availability: list[np.ndarray] = []
    quality: list[np.ndarray] = []
    stream_norms: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            eligible = batch["loss_eligible"].bool().cpu().numpy()
            output = _forward_generation2(model, batch, config["route"], device)
            if not bool(torch.isfinite(output["logits"]).all()):
                raise FloatingPointError("non-finite selected-checkpoint logits")
            sample_ids.append(np.asarray(batch["sample_id"], dtype=str)[eligible])
            users.append(np.asarray(batch["user_id"], dtype=str)[eligible])
            labels.append(batch["label"].cpu().numpy()[eligible])
            logits.append(output["logits"].float().cpu().numpy()[eligible])
            availability.append(output["availability"].cpu().numpy()[eligible])
            quality.append(output["quality"].float().cpu().numpy()[eligible])
            if "stream_norms" in output:
                stream_norms.append(output["stream_norms"].float().cpu().numpy()[eligible])
    if not labels:
        raise ValueError("loader produced no prediction batches")
    archive = {
        "sample_ids": np.concatenate(sample_ids),
        "users": np.concatenate(users),
        "labels": np.concatenate(labels),
        "logits": np.concatenate(logits),
        "availability": np.concatenate(availability),
        "quality": np.concatenate(quality),
    }
    if stream_norms:
        archive["stream_norms"] = np.concatenate(stream_norms)
    archive["predictions"] = archive["logits"].argmax(axis=1)
    archive["metrics"] = fixed_label_metrics(
        labels=archive["labels"], logits=archive["logits"], users=archive["users"]
    )
    return archive
