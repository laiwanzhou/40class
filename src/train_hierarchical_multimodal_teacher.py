from __future__ import annotations

import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
import yaml

from scripts.cache_ir_depth_videomaev2_p2a import (
    _atomic_npz,
    _atomic_write_text,
    _metrics,
)
from src.data.hierarchical_multimodal_dataset import make_midfusion_dataset
from src.data.body_normalization_state import (
    BodyNormalizationState,
    apply_body_normalization_state,
    body_normalization_provenance,
    fit_body_normalization_state,
)
from src.experiments.hierarchical_midfusion_config import (
    assert_grouped_cv_authorized,
    load_midfusion_config,
    project_path,
)
from src.models.body_motion_segment_encoder import BodyMotionSegmentEncoder
from src.models.hierarchical_action_query_fusion import HierarchicalActionQueryFusion
from src.models.hierarchical_multimodal_teacher import (
    GroupDropout,
    HierarchicalMultimodalTeacher,
)
from src.models.ir_depth_videomaev2_teacher import (
    build_official_videomaev2_vit_b,
    sha256_file,
)
from src.models.structured_ir_depth_visual_encoder import (
    StructuredIRDepthVisualEncoder,
    VideoMAESegmentBackboneAdapter,
)
from src.training.hierarchical_multimodal_losses import hierarchical_teacher_loss


ModelFactory = Callable[[dict[str, Any]], HierarchicalMultimodalTeacher]
DatasetFactory = Callable[[dict[str, Any]], Dataset[dict[str, object]]]
FixedDatasetFactory = Callable[
    [dict[str, Any], str], Dataset[dict[str, object]]
]
CANDIDATE_MODALITIES = {
    "visual_only": ("ir", "depth_color"),
    "visual_skeleton": ("ir", "depth_color", "skeleton"),
    "visual_imu": ("ir", "depth_color", "imu"),
    "visual_skeleton_imu": ("ir", "depth_color", "skeleton", "imu"),
}


class EpochClassUserBalancedSampler(Sampler[int]):
    def __init__(
        self,
        *,
        labels: np.ndarray,
        users: np.ndarray,
        samples: int,
        seed: int,
    ) -> None:
        self.samples = int(samples)
        self.seed = int(seed)
        self.epoch = 0
        groups: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (label, user) in enumerate(zip(labels, users, strict=True)):
            groups[int(label)][str(user)].append(index)
        self.groups = {
            label: {user: tuple(indices) for user, indices in values.items()}
            for label, values in groups.items()
        }
        self.classes = tuple(sorted(self.groups))
        if not self.classes:
            raise ValueError("balanced sampler received no classes")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.samples):
            label = rng.choice(self.classes)
            user = rng.choice(tuple(sorted(self.groups[label])))
            yield rng.choice(self.groups[label][user])

    def __len__(self) -> int:
        return self.samples


def partition_fold_indices(
    users: np.ndarray, *, validation_users: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    user_values = np.asarray(users).astype(str)
    validation = np.isin(user_values, np.asarray(sorted(validation_users)))
    fit_indices = np.flatnonzero(~validation)
    validation_indices = np.flatnonzero(validation)
    if not len(fit_indices) or not len(validation_indices):
        raise ValueError("grouped fold partition is empty")
    if set(fit_indices.tolist()) & set(validation_indices.tolist()):
        raise ValueError("grouped fold indices overlap")
    return fit_indices, validation_indices


def final_evaluation_candidates(selected: str) -> tuple[str]:
    if selected not in CANDIDATE_MODALITIES:
        raise ValueError(f"unknown midfusion candidate: {selected}")
    return (selected,)


def fit_class_prior(labels: np.ndarray, *, classes: int = 40) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=classes).astype(
        np.float64
    )
    probabilities = (counts + 1.0) / (counts.sum() + classes)
    return torch.from_numpy(np.log(probabilities).astype(np.float32))


def pool_fold_predictions(
    *,
    sample_count: int,
    classes: int,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    pooled = np.full((sample_count, classes), np.nan, dtype=np.float32)
    ownership = np.zeros(sample_count, dtype=np.int64)
    for indices, logits in folds:
        indices = np.asarray(indices, dtype=np.int64)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (len(indices), classes):
            raise ValueError("fold prediction shape changed")
        if bool((indices < 0).any()) or bool((indices >= sample_count).any()):
            raise ValueError("fold prediction index is outside population")
        if bool((ownership[indices] != 0).any()):
            raise ValueError("fold prediction ownership repeated")
        pooled[indices] = logits
        ownership[indices] += 1
    if not bool((ownership == 1).all()) or not np.isfinite(pooled).all():
        raise ValueError("fold predictions do not cover every sample exactly once")
    return pooled


def select_grouped_candidate(
    metrics: dict[str, dict[str, Any]],
) -> str:
    order = tuple(CANDIDATE_MODALITIES)
    if set(metrics) != set(order):
        raise ValueError("grouped candidate metric set changed")
    return max(
        order,
        key=lambda name: (
            float(metrics[name]["accuracy"]),
            float(metrics[name]["macro_f1"]),
            float(metrics[name]["worst_user_accuracy"]),
            -float(metrics[name]["nll"]),
            -order.index(name),
        ),
    )


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _string_sequence_sha256(values: np.ndarray) -> str:
    payload = json.dumps(
        np.asarray(values).astype(str).tolist(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_sha256(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fit_body_normalization(
    dataset: Dataset[dict[str, object]], fit_indices: np.ndarray
) -> dict[str, Any]:
    state = fit_body_normalization_state(dataset, fit_indices)
    apply_body_normalization_state(dataset, state)
    return body_normalization_provenance(state)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_teacher(config: dict[str, Any]) -> HierarchicalMultimodalTeacher:
    model_config = config["model"]
    checkpoint = Path(str(model_config["visual_checkpoint"]))
    if sha256_file(checkpoint) != model_config["visual_checkpoint_sha256"]:
        raise RuntimeError("VideoMAE checkpoint provenance changed")
    backbone, _ = build_official_videomaev2_vit_b(
        checkpoint_path=checkpoint,
        num_classes=40,
        with_cp=True,
    )
    dim = int(model_config["teacher_dim"])
    visual = StructuredIRDepthVisualEncoder(
        backbone=VideoMAESegmentBackboneAdapter(
            backbone=backbone, frozen_prefix_blocks=8, segment_count=8
        ),
        output_dim=dim,
    )
    body = BodyMotionSegmentEncoder(
        output_dim=dim, heads=int(model_config["attention_heads"])
    )
    fusion = HierarchicalActionQueryFusion(
        dim=dim,
        classes=40,
        heads=int(model_config["attention_heads"]),
        layers=int(model_config["fusion_layers"]),
    )
    return HierarchicalMultimodalTeacher(
        visual_encoder=visual,
        body_encoder=body,
        fusion=fusion,
        dim=dim,
        classes=40,
    )


def _default_smoke_dataset(config: dict[str, Any]) -> Dataset[dict[str, object]]:
    clean_view = project_path(
        str(config["data"]["skeleton_clean_views"])
    ) / "selected_final/clean_view.csv"
    if not clean_view.is_file():
        raise FileNotFoundError(
            f"missing selected-final Skeleton clean view: {clean_view}"
        )
    dataset = make_midfusion_dataset(
        config,
        partition="train",
        metadata_only=False,
        skeleton_clean_view=clean_view,
        training=False,
    )
    complete = next(
        index
        for index, trial in enumerate(dataset.trials)
        if all(trial.availability[name] for name in ("ir", "depth_color", "skeleton", "imu"))
    )
    missing_imu = next(
        index
        for index, trial in enumerate(dataset.trials)
        if trial.availability["ir"]
        and trial.availability["depth_color"]
        and trial.availability["skeleton"]
        and not trial.availability["imu"]
    )
    return Subset(dataset, [complete, missing_imu])


def _parameter_groups(
    model: HierarchicalMultimodalTeacher,
) -> dict[str, list[torch.nn.Parameter]]:
    visual = [
        parameter
        for parameter in model.visual_encoder.parameters()
        if parameter.requires_grad
    ]
    skeleton = [
        *model.body_encoder.skeleton_projection.parameters(),
        *model.body_encoder.skeleton_temporal.parameters(),
    ]
    imu = [
        *model.body_encoder.imu_projection.parameters(),
        *model.body_encoder.imu_role_score.parameters(),
        *model.body_encoder.imu_temporal.parameters(),
    ]
    body_ids = {id(parameter) for parameter in skeleton + imu}
    fusion = [
        parameter
        for module in (
            model.body_encoder.cross_attention,
            model.body_encoder.cross_norm,
            model.fusion,
            model.context_head,
            model.wrist_head,
            model.body_head,
        )
        for parameter in module.parameters()
        if parameter.requires_grad and id(parameter) not in body_ids
    ]
    return {"visual": visual, "skeleton": skeleton, "imu": imu, "fusion": fusion}


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_smoke(
    config_path: Path,
    *,
    output_root: Path,
    model_factory: ModelFactory | None = None,
    dataset_factory: DatasetFactory | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    config = load_midfusion_config(config_path)
    _set_seed(int(config["training"]["seed"]))
    device = device or torch.device("cuda")
    model = (model_factory or build_teacher)(config).to(device)
    dataset = (dataset_factory or _default_smoke_dataset)(config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    groups = _parameter_groups(model)
    before = {
        name: [parameter.detach().cpu().clone() for parameter in parameters]
        for name, parameters in groups.items()
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    finite_gradients = True
    users: set[str] = set()
    last_batch: dict[str, Any] | None = None
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        users.update(str(value) for value in batch["user_id"])
        if users & {"user6", "user7"}:
            raise RuntimeError("validation users entered smoke gradients")
        batch = _move_batch(batch, device)
        last_batch = batch
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(batch, dropout_policy=GroupDropout.disabled())
            losses = hierarchical_teacher_loss(
                output,
                batch["label"].long(),
                epoch=1,
                natural_pattern=True,
            )
        losses["loss"].backward()
        finite_gradients &= all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not finite_gradients:
            raise FloatingPointError("non-finite hierarchical teacher smoke gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    if last_batch is None:
        raise RuntimeError("empty hierarchical teacher smoke dataset")

    changed = []
    for name, parameters in groups.items():
        if any(
            not torch.equal(start, parameter.detach().cpu())
            for start, parameter in zip(before[name], parameters, strict=True)
        ):
            changed.append(name)

    model.eval()
    body_only = _clone_batch(last_batch)
    body_only["visual_view_availability"].zero_()
    no_core = _clone_batch(last_batch)
    no_core["visual_view_availability"].zero_()
    no_core["skeleton_mask"].zero_()
    no_core["imu_role_mask"].zero_()
    with torch.inference_mode():
        body_output = model(body_only, dropout_policy=GroupDropout.disabled())
        no_core_output = model(no_core, dropout_policy=GroupDropout.disabled())
    report = {
        "status": "smoke_passed",
        "sample_users_entered_gradient": sorted(users),
        "finite_gradients": finite_gradients,
        "changed_parameter_groups": changed,
        "body_only_finite": bool(torch.isfinite(body_output["logits"]).all()),
        "no_core_finite": bool(torch.isfinite(no_core_output["logits"]).all()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_cuda_mib": float(
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }
    output_root.mkdir(parents=True)
    _atomic_write_text(
        output_root / "resolved_config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )
    _atomic_write_text(
        output_root / "smoke_report.json", json.dumps(report, indent=2) + "\n"
    )
    return report


def _dataset_identity(
    dataset: Dataset[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all(hasattr(dataset, name) for name in ("labels", "user_ids", "sample_ids")):
        return (
            np.asarray(getattr(dataset, "labels"), dtype=np.int64),
            np.asarray(getattr(dataset, "user_ids")).astype(str),
            np.asarray(getattr(dataset, "sample_ids")).astype(str),
        )
    if hasattr(dataset, "trials"):
        trials = list(getattr(dataset, "trials"))
        return (
            np.asarray([trial.class_id for trial in trials], dtype=np.int64),
            np.asarray([trial.user_id for trial in trials]),
            np.asarray([trial.sample_id for trial in trials]),
        )
    labels, users, samples = [], [], []
    for index in range(len(dataset)):
        item = dataset[index]
        labels.append(int(item["label"]))
        users.append(str(item["user_id"]))
        samples.append(str(item["sample_id"]))
    return np.asarray(labels), np.asarray(users), np.asarray(samples)


def _candidate_eligible(
    dataset: Dataset[dict[str, object]], index: int, candidate: str
) -> bool:
    if not hasattr(dataset, "trials"):
        return True
    trial = getattr(dataset, "trials")[index]
    return any(bool(trial.availability[name]) for name in CANDIDATE_MODALITIES[candidate])


class _JoinedSplitDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        train_dataset: Dataset[dict[str, object]],
        validation_dataset: Dataset[dict[str, object]],
    ) -> None:
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        train_identity = _dataset_identity(train_dataset)
        validation_identity = _dataset_identity(validation_dataset)
        self.labels = np.concatenate((train_identity[0], validation_identity[0]))
        self.user_ids = np.concatenate((train_identity[1], validation_identity[1]))
        self.sample_ids = np.concatenate((train_identity[2], validation_identity[2]))
        if hasattr(train_dataset, "trials") and hasattr(validation_dataset, "trials"):
            self.trials = [
                *list(getattr(train_dataset, "trials")),
                *list(getattr(validation_dataset, "trials")),
            ]

    def __len__(self) -> int:
        return len(self.train_dataset) + len(self.validation_dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < len(self.train_dataset):
            return self.train_dataset[index]
        return self.validation_dataset[index - len(self.train_dataset)]


@torch.no_grad()
def _predict_indices(
    *,
    model: HierarchicalMultimodalTeacher,
    dataset: Dataset[dict[str, object]],
    indices: np.ndarray,
    candidate: str,
    prior_logits: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=1, shuffle=False)
    logits, labels, users, samples, core = [], [], [], [], []
    diagnostics: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "group_attention",
            "segment_attention",
            "context_logits",
            "wrist_logits",
            "body_logits",
            "effective_group_mask",
            "visual_mask",
            "body_mask",
            "availability",
            "skeleton_quality",
            "imu_quality",
            "skeleton_mask",
            "imu_role_mask",
        )
    }
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch,
                dropout_policy=GroupDropout.disabled(),
                enabled_modalities=CANDIDATE_MODALITIES[candidate],
            )
        values = output["logits"].float()
        available = output["core_available"].bool()
        values = torch.where(
            available[:, None], values, prior_logits.to(device)[None]
        )
        logits.append(values.cpu())
        labels.append(batch["label"].long().cpu())
        users.extend(str(value) for value in batch["user_id"])
        samples.extend(str(value) for value in batch["sample_id"])
        core.append(available.cpu())
        for name in (
            "group_attention",
            "segment_attention",
            "context_logits",
            "wrist_logits",
            "body_logits",
            "effective_group_mask",
            "visual_mask",
            "body_mask",
        ):
            diagnostics[name].append(output[name].detach().cpu())
        diagnostics["availability"].append(batch["availability"].detach().cpu())
        batch_size = values.shape[0]
        diagnostics["skeleton_quality"].append(
            batch.get(
                "skeleton_quality",
                torch.zeros(batch_size, 8, 4, device=device),
            ).detach().cpu()
        )
        diagnostics["imu_quality"].append(
            batch.get(
                "imu_quality",
                torch.zeros(batch_size, 8, 5, 3, device=device),
            ).detach().cpu()
        )
        diagnostics["skeleton_mask"].append(batch["skeleton_mask"].detach().cpu())
        diagnostics["imu_role_mask"].append(batch["imu_role_mask"].detach().cpu())
    logits_np = torch.cat(logits).numpy()
    labels_np = torch.cat(labels).numpy()
    users_np = np.asarray(users)
    result = {
        "logits": logits_np,
        "labels": labels_np,
        "user_ids": users_np,
        "sample_ids": np.asarray(samples),
        "core_available": torch.cat(core).numpy(),
        "metrics": _metrics(labels_np, logits_np, users_np),
    }
    result.update(
        {name: torch.cat(values).numpy() for name, values in diagnostics.items()}
    )
    return result


def _save_prediction_archive(path: Path, prediction: dict[str, Any]) -> None:
    _atomic_npz(
        path,
        **{
            name: np.asarray(value)
            for name, value in prediction.items()
            if name != "metrics"
        },
    )


def train_candidate_fold(
    *,
    config: dict[str, Any],
    candidate: str,
    dataset: Dataset[dict[str, object]],
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    run_dir: Path,
    model_factory: ModelFactory | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if candidate not in CANDIDATE_MODALITIES:
        raise ValueError(f"unknown midfusion candidate: {candidate}")
    if (run_dir / "summary.json").exists():
        raise FileExistsError(f"completed fold already exists: {run_dir}")
    if run_dir.exists() and not (run_dir / "latest_checkpoint.pt").is_file():
        raise FileExistsError(f"non-resumable fold directory exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    labels, users, sample_ids = _dataset_identity(dataset)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if set(fit_indices.tolist()) & set(validation_indices.tolist()):
        raise ValueError("fit and validation indices overlap")
    fit_users = set(users[fit_indices].tolist())
    validation_users = set(users[validation_indices].tolist())
    if fit_users & validation_users:
        raise ValueError("fit and validation users overlap")
    fit_sample_ids_sha256 = _string_sequence_sha256(sample_ids[fit_indices])
    validation_sample_ids_sha256 = _string_sequence_sha256(
        sample_ids[validation_indices]
    )
    config_sha256 = _config_sha256(config)
    eligible_fit = np.asarray(
        [
            index
            for index in fit_indices.tolist()
            if _candidate_eligible(dataset, index, candidate)
        ],
        dtype=np.int64,
    )
    if not len(eligible_fit):
        raise ValueError("candidate has no usable fit rows")
    training = config["training"]
    seed = int(training["seed"])
    _set_seed(seed)
    device = device or torch.device("cuda")
    model = (model_factory or build_teacher)(config).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    prior_logits = fit_class_prior(labels[fit_indices], classes=40)
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    latest = run_dir / "latest_checkpoint.pt"
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload["candidate"] != candidate:
            raise RuntimeError("resume candidate changed")
        if payload.get("evaluation_protocol") != config["evaluation_protocol"]:
            raise RuntimeError("resume evaluation protocol changed")
        if payload.get("config_sha256") != config_sha256:
            raise RuntimeError("resume config changed")
        if payload.get("fit_sample_ids_sha256") != fit_sample_ids_sha256:
            raise RuntimeError("resume fit samples changed")
        if (
            payload.get("validation_sample_ids_sha256")
            != validation_sample_ids_sha256
        ):
            raise RuntimeError("resume validation samples changed")
        if payload.get("fit_user_ids") != sorted(fit_users):
            raise RuntimeError("resume fit users changed")
        if payload.get("validation_user_ids") != sorted(validation_users):
            raise RuntimeError("resume validation users changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        history = list(payload["history"])
        start_epoch = int(payload["epoch"]) + 1
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if device.type == "cuda" and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(payload["cuda_rng_state"], device)

    fit_dataset = Subset(dataset, eligible_fit.tolist())
    subset_labels = labels[eligible_fit]
    subset_users = users[eligible_fit]
    sampler = EpochClassUserBalancedSampler(
        labels=subset_labels,
        users=subset_users,
        samples=len(eligible_fit),
        seed=seed,
    )
    loader = DataLoader(fit_dataset, batch_size=1, sampler=sampler, num_workers=0)
    accumulation_target = int(training["gradient_accumulation"])
    fixed_epochs = int(training["fixed_epochs"])
    for epoch in range(start_epoch, fixed_epochs + 1):
        epoch_started = time.perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        loss_total = 0.0
        supervised_rows = 0
        for step, batch in enumerate(loader, start=1):
            batch = _move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    batch,
                    dropout_policy=GroupDropout(
                        context=float(training["context_dropout"]),
                        wrist=float(training["wrist_dropout"]),
                        body=float(training["body_dropout"]),
                        visual=float(training["visual_dropout"]),
                    ),
                    enabled_modalities=CANDIDATE_MODALITIES[candidate],
                )
                losses = hierarchical_teacher_loss(
                    output,
                    batch["label"].long(),
                    epoch=epoch,
                    natural_pattern=True,
                )
            losses["loss"].backward()
            accumulated += 1
            loss_total += float(losses["loss"].detach())
            supervised_rows += int(losses["supervised_rows"].detach())
            is_last = step == len(loader)
            if accumulated == accumulation_target or is_last:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated)
                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("non-finite candidate gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
        history.append(
            {
                "epoch": epoch,
                "mean_train_loss": loss_total / max(len(loader), 1),
                "supervised_rows": supervised_rows,
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        _atomic_torch_save(
            latest,
            {
                "candidate": candidate,
                "evaluation_protocol": config["evaluation_protocol"],
                "config_sha256": config_sha256,
                "fit_sample_ids_sha256": fit_sample_ids_sha256,
                "validation_sample_ids_sha256": validation_sample_ids_sha256,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "prior_logits": prior_logits,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
                "fit_user_ids": sorted(fit_users),
                "validation_user_ids": sorted(validation_users),
            },
        )
        print(
            json.dumps(
                {
                    "stage": "candidate_training",
                    "candidate": candidate,
                    "run_dir": str(run_dir),
                    **history[-1],
                }
            ),
            flush=True,
        )

    train_prediction = _predict_indices(
        model=model,
        dataset=dataset,
        indices=fit_indices,
        candidate=candidate,
        prior_logits=prior_logits,
        device=device,
    )
    _save_prediction_archive(run_dir / "train_predictions.npz", train_prediction)
    prediction = _predict_indices(
        model=model,
        dataset=dataset,
        indices=validation_indices,
        candidate=candidate,
        prior_logits=prior_logits,
        device=device,
    )
    _save_prediction_archive(run_dir / "validation_predictions.npz", prediction)
    summary = {
        "candidate": candidate,
        "evaluation_protocol": config["evaluation_protocol"],
        "config_sha256": config_sha256,
        "fit_sample_ids_sha256": fit_sample_ids_sha256,
        "validation_sample_ids_sha256": validation_sample_ids_sha256,
        "epochs_completed": fixed_epochs,
        "history": history,
        "fit_user_ids": sorted(fit_users),
        "fit_sample_ids": train_prediction["sample_ids"].astype(str).tolist(),
        "validation_user_ids": sorted(validation_users),
        "train_sample_ids": train_prediction["sample_ids"].astype(str).tolist(),
        "validation_sample_ids": prediction["sample_ids"].astype(str).tolist(),
        "train_evaluation_count": 1,
        "validation_evaluation_count": 1,
        "train_metrics": train_prediction["metrics"],
        "validation_metrics": prediction["metrics"],
        "checkpoint": str(latest),
        "checkpoint_sha256": sha256_file(latest),
        "predictions": str(run_dir / "validation_predictions.npz"),
        "predictions_sha256": sha256_file(run_dir / "validation_predictions.npz"),
        "train_predictions": str(run_dir / "train_predictions.npz"),
        "train_predictions_sha256": sha256_file(run_dir / "train_predictions.npz"),
    }
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def train_candidate_split(
    *,
    config: dict[str, Any],
    candidate: str,
    train_dataset: Dataset[dict[str, object]],
    validation_dataset: Dataset[dict[str, object]],
    run_dir: Path,
    model_factory: ModelFactory | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    train_labels, train_users, train_samples = _dataset_identity(train_dataset)
    validation_labels, validation_users, validation_samples = _dataset_identity(
        validation_dataset
    )
    del train_labels, validation_labels
    if set(train_users.tolist()) & set(validation_users.tolist()):
        raise ValueError("train and validation users overlap")
    if set(train_samples.tolist()) & set(validation_samples.tolist()):
        raise ValueError("train and validation samples overlap")
    joined = _JoinedSplitDataset(train_dataset, validation_dataset)
    fit_indices = np.arange(len(train_dataset), dtype=np.int64)
    validation_indices = np.arange(
        len(train_dataset), len(joined), dtype=np.int64
    )
    return train_candidate_fold(
        config=config,
        candidate=candidate,
        dataset=joined,
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        run_dir=run_dir,
        model_factory=model_factory,
        device=device,
    )


def _default_fixed_dataset_factory(
    config: dict[str, Any], partition: str
) -> Dataset[dict[str, object]]:
    clean_view = project_path(str(config["data"]["skeleton_clean_views"])) / (
        "selected_final/clean_view.csv"
    )
    if not clean_view.is_file():
        raise FileNotFoundError(f"missing selected-final Skeleton clean view: {clean_view}")
    return make_midfusion_dataset(
        config,
        partition=partition,
        metadata_only=False,
        skeleton_clean_view=clean_view,
        training=partition == "train",
    )


def _persist_normalization_state(
    path: Path, state: BodyNormalizationState
) -> None:
    arrays = {
        "skeleton_mean": state.skeleton_mean,
        "skeleton_std": state.skeleton_std,
        "imu_mean": state.imu_mean,
        "imu_std": state.imu_std,
        "fit_sample_ids": np.asarray(state.fit_sample_ids),
        "fit_user_ids": np.asarray(state.fit_user_ids),
        "skeleton_samples": np.asarray(state.skeleton_samples, dtype=np.int64),
        "imu_samples": np.asarray(state.imu_samples, dtype=np.int64),
    }
    if path.is_file():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(existing[name], value)
                for name, value in arrays.items()
            ):
                raise RuntimeError("persisted normalization state changed")
        return
    _atomic_npz(path, **arrays)


def _fixed_report_paths(
    config: dict[str, Any], output_root: Path | None
) -> tuple[Path, Path, Path]:
    if output_root is not None:
        root = output_root.resolve()
        return (
            root,
            root / "fixed_validation_report.json",
            root / "fixed_validation_report.md",
        )
    root = project_path(str(config["outputs"]["root"])) / "fixed_user6_user7"
    return (
        root,
        project_path(str(config["outputs"]["fixed_validation_report_json"])),
        project_path(
            str(config["outputs"]["fixed_validation_report_markdown"])
        ),
    )


def run_fixed_validation(
    config_path: Path,
    *,
    output_root: Path | None = None,
    dataset_factory: FixedDatasetFactory | None = None,
    model_factory: ModelFactory | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    config = load_midfusion_config(config_path)
    if config["evaluation_protocol"] != "fixed_user6_user7":
        raise RuntimeError("fixed validation protocol changed")
    root, report_path, markdown_path = _fixed_report_paths(config, output_root)
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError("fixed-validation report already exists")
    root.mkdir(parents=True, exist_ok=True)
    factory = dataset_factory or _default_fixed_dataset_factory
    train_dataset = factory(config, "train")
    validation_dataset = factory(config, "validation")
    train_labels, train_users, train_samples = _dataset_identity(train_dataset)
    validation_labels, validation_users, validation_samples = _dataset_identity(
        validation_dataset
    )
    if set(train_users.tolist()) & set(validation_users.tolist()):
        raise RuntimeError("fixed train and validation users overlap")
    if set(train_samples.tolist()) & set(validation_samples.tolist()):
        raise RuntimeError("fixed train and validation samples overlap")
    if dataset_factory is None:
        if len(train_dataset) != 2039 or len(validation_dataset) != 388:
            raise RuntimeError("fixed canonical population changed")
        if set(train_users.tolist()) != set(config["population"]["train_user_ids"]):
            raise RuntimeError("fixed train users changed")
        if set(validation_users.tolist()) != {"user6", "user7"}:
            raise RuntimeError("fixed validation users changed")
        if set(train_labels.tolist()) != set(range(40)) or set(
            validation_labels.tolist()
        ) != set(range(40)):
            raise RuntimeError("fixed population class coverage changed")

    normalization_state = fit_body_normalization_state(
        train_dataset, np.arange(len(train_dataset), dtype=np.int64)
    )
    validation_user_set = set(validation_users.astype(str).tolist())
    if set(normalization_state.fit_user_ids) & validation_user_set:
        raise RuntimeError("validation users entered normalization")
    apply_body_normalization_state(train_dataset, normalization_state)
    apply_body_normalization_state(validation_dataset, normalization_state)
    normalization_path = root / "normalization_state.npz"
    _persist_normalization_state(normalization_path, normalization_state)
    normalization = {
        **body_normalization_provenance(normalization_state),
        "state_path": str(normalization_path),
        "state_sha256": sha256_file(normalization_path),
    }
    provenance = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_path("."),
            text=True,
        ).strip(),
        "config_path": str(config_path.resolve()),
        "config_file_sha256": sha256_file(config_path.resolve()),
        "trainer_source_sha256": sha256_file(Path(__file__).resolve()),
        "runner_source_sha256": sha256_file(
            project_path("scripts/run_hierarchical_multimodal_teacher.py")
        ),
        "selected_final_clean_view_sha256": None,
    }
    if dataset_factory is None:
        clean_view = project_path(
            str(config["data"]["skeleton_clean_views"])
        ) / "selected_final/clean_view.csv"
        provenance["selected_final_clean_view_sha256"] = sha256_file(clean_view)
    run_config = copy.deepcopy(config)
    run_config["runtime_provenance"] = {
        "normalization_state_sha256": normalization["state_sha256"],
        **provenance,
    }
    run_config_sha256 = _config_sha256(run_config)

    candidate_results: dict[str, dict[str, Any]] = {}
    validation_logits: dict[str, np.ndarray] = {}
    for candidate in CANDIDATE_MODALITIES:
        candidate_dir = root / candidate
        summary_path = candidate_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = train_candidate_split(
                config=run_config,
                candidate=candidate,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                run_dir=candidate_dir,
                model_factory=model_factory,
                device=device,
            )
        if summary["fit_user_ids"] != sorted(set(train_users.tolist())):
            raise RuntimeError("candidate fit ownership changed")
        if summary["validation_user_ids"] != sorted(
            set(validation_users.tolist())
        ):
            raise RuntimeError("candidate validation ownership changed")
        if summary.get("config_sha256") != run_config_sha256:
            raise RuntimeError("candidate config provenance changed")
        if sha256_file(Path(summary["checkpoint"])) != summary["checkpoint_sha256"]:
            raise RuntimeError("candidate checkpoint hash changed")
        if (
            sha256_file(Path(summary["train_predictions"]))
            != summary["train_predictions_sha256"]
        ):
            raise RuntimeError("candidate train prediction hash changed")
        validation_archive = Path(summary["predictions"])
        if sha256_file(validation_archive) != summary["predictions_sha256"]:
            raise RuntimeError("candidate validation prediction hash changed")
        with np.load(validation_archive, allow_pickle=False) as archive:
            if not np.array_equal(
                archive["sample_ids"].astype(str), validation_samples.astype(str)
            ):
                raise RuntimeError("candidate validation sample order changed")
            validation_logits[candidate] = archive["logits"].astype(np.float32)
        train_accuracy = float(summary["train_metrics"]["accuracy"])
        validation_accuracy = float(summary["validation_metrics"]["accuracy"])
        candidate_results[candidate] = {
            "config_sha256": summary["config_sha256"],
            "train_metrics": summary["train_metrics"],
            "validation_metrics": summary["validation_metrics"],
            "accuracy_generalization_gap": train_accuracy - validation_accuracy,
            "checkpoint": summary["checkpoint"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "train_predictions": summary["train_predictions"],
            "train_predictions_sha256": summary["train_predictions_sha256"],
            "validation_predictions": summary["predictions"],
            "validation_predictions_sha256": summary["predictions_sha256"],
        }

    candidate_metrics = {
        name: result["validation_metrics"]
        for name, result in candidate_results.items()
    }
    selected = select_grouped_candidate(candidate_metrics)
    anchor_logits = validation_logits["visual_only"]
    comparisons = {
        candidate: _comparison(
            validation_labels, anchor_logits, validation_logits[candidate]
        )
        for candidate in CANDIDATE_MODALITIES
    }
    research_category = _research_category(
        config,
        selected_metrics=candidate_metrics[selected],
        visual_metrics=candidate_metrics["visual_only"],
        comparison=comparisons[selected],
    )
    report = {
        "stage": "P5-HMF0-fixed-user6-user7",
        "status": "completed",
        "evaluation_protocol": "fixed_user6_user7",
        "development_validation": True,
        "independent_final_test": False,
        "train_population_samples": len(train_dataset),
        "validation_population_samples": len(validation_dataset),
        "train_user_ids": sorted(set(train_users.tolist())),
        "validation_user_ids": sorted(set(validation_users.tolist())),
        "validation_users_entered_training": False,
        "normalization": normalization,
        "provenance": provenance,
        "candidate_order": list(CANDIDATE_MODALITIES),
        "selection_order": [
            "accuracy",
            "macro_f1",
            "worst_user_accuracy",
            "negative_nll",
            "fixed_candidate_order",
        ],
        "selected_candidate": selected,
        "research_category": research_category,
        "student_planning_authorized": research_category
        in {"full_teacher_worthy", "teacher_target_reached"},
        "candidate_results": candidate_results,
        "comparison_to_visual_only": comparisons,
        "config_sha256": run_config_sha256,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")
    lines = [
        "# Hierarchical multimodal teacher fixed user6/user7 validation",
        "",
        "- Development validation: `True`",
        "- Independent final test: `False`",
        f"- Selected candidate: `{selected}`",
        "",
        "| Candidate | Train Accuracy | Validation Accuracy | Macro-F1 | Worst-user | Gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate, result in candidate_results.items():
        train_metrics = result["train_metrics"]
        validation_metrics = result["validation_metrics"]
        lines.append(
            f"| {candidate} | {train_metrics['accuracy']:.6f} | "
            f"{validation_metrics['accuracy']:.6f} | "
            f"{validation_metrics['macro_f1']:.6f} | "
            f"{validation_metrics['worst_user_accuracy']:.6f} | "
            f"{result['accuracy_generalization_gap']:.6f} |"
        )
    _atomic_write_text(markdown_path, "\n".join(lines) + "\n")
    return report


def _comparison(
    labels: np.ndarray, anchor_logits: np.ndarray, candidate_logits: np.ndarray
) -> dict[str, int]:
    anchor = anchor_logits.argmax(axis=1)
    candidate = candidate_logits.argmax(axis=1)
    return {
        "rescued": int(((candidate == labels) & (anchor != labels)).sum()),
        "harmed": int(((candidate != labels) & (anchor == labels)).sum()),
        "net": int((candidate == labels).sum() - (anchor == labels).sum()),
        "disagreement": int((candidate != anchor).sum()),
    }


def _research_category(
    config: dict[str, Any],
    *,
    selected_metrics: dict[str, Any],
    visual_metrics: dict[str, Any],
    comparison: dict[str, int],
) -> str:
    gates = config["decision_gates"]
    accuracy = float(selected_metrics["accuracy"])
    macro_f1 = float(selected_metrics["macro_f1"])
    worst_user = float(selected_metrics["worst_user_accuracy"])
    if (
        accuracy >= float(gates["teacher_target_accuracy"])
        and macro_f1 >= float(gates["teacher_target_macro_f1"])
        and worst_user >= float(gates["teacher_target_worst_user"])
    ):
        return "teacher_target_reached"
    if (
        accuracy >= float(gates["full_teacher_worthy_accuracy"])
        and comparison["net"] > 0
        and worst_user >= float(gates["full_teacher_worthy_worst_user"])
    ):
        return "full_teacher_worthy"
    if (
        accuracy >= float(gates["reject_below_accuracy"])
        and accuracy < float(gates["full_teacher_worthy_accuracy"])
        and comparison["net"] > 0
        and worst_user
        >= float(visual_metrics["worst_user_accuracy"]) - 0.01
    ):
        return "promising"
    return "reject"


def run_grouped_cv(config_path: Path) -> dict[str, Any]:
    config = load_midfusion_config(config_path)
    assert_grouped_cv_authorized(config)
    report_path = project_path(str(config["outputs"]["grouped_report_json"]))
    markdown_path = project_path(str(config["outputs"]["grouped_report_markdown"]))
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError("grouped-CV report already exists")
    output_root = project_path(str(config["outputs"]["root"])) / "grouped_cv"
    output_root.mkdir(parents=True, exist_ok=True)
    fold_results: dict[str, list[dict[str, Any]]] = {
        candidate: [] for candidate in CANDIDATE_MODALITIES
    }
    fold_predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        candidate: [] for candidate in CANDIDATE_MODALITIES
    }
    canonical_labels: np.ndarray | None = None
    canonical_users: np.ndarray | None = None
    canonical_samples: np.ndarray | None = None

    for fold in config["grouped_folds"]:
        fold_index = int(fold["fold"])
        clean_view = project_path(
            str(config["data"]["skeleton_clean_views"])
        ) / f"fold_{fold_index}/clean_view.csv"
        dataset = make_midfusion_dataset(
            config,
            partition="train",
            metadata_only=False,
            skeleton_clean_view=clean_view,
            training=True,
        )
        labels, users, samples = _dataset_identity(dataset)
        if canonical_labels is None:
            canonical_labels, canonical_users, canonical_samples = labels, users, samples
        elif not (
            np.array_equal(labels, canonical_labels)
            and np.array_equal(users, canonical_users)
            and np.array_equal(samples, canonical_samples)
        ):
            raise RuntimeError("fold datasets changed canonical ordering")
        fit_indices, validation_indices = partition_fold_indices(
            users, validation_users=set(fold["validation_user_ids"])
        )
        if set(users[fit_indices].tolist()) != set(fold["fit_user_ids"]):
            raise RuntimeError("fold fit membership differs from contract")
        normalization = fit_body_normalization(dataset, fit_indices)
        normalization_path = output_root / f"fold_{fold_index}/normalization.json"
        normalization_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            normalization_path,
            json.dumps(
                {
                    **normalization,
                    "fit_user_ids": sorted(set(users[fit_indices].tolist())),
                    "validation_user_ids": sorted(
                        set(users[validation_indices].tolist())
                    ),
                },
                indent=2,
            )
            + "\n",
        )
        for candidate in CANDIDATE_MODALITIES:
            run_dir = output_root / candidate / f"fold_{fold_index}"
            if (run_dir / "summary.json").is_file():
                summary = json.loads(
                    (run_dir / "summary.json").read_text(encoding="utf-8")
                )
            else:
                summary = train_candidate_fold(
                    config=config,
                    candidate=candidate,
                    dataset=dataset,
                    fit_indices=fit_indices,
                    validation_indices=validation_indices,
                    run_dir=run_dir,
                    device=torch.device("cuda"),
                )
            archive_path = run_dir / "validation_predictions.npz"
            with np.load(archive_path, allow_pickle=False) as archive:
                archive_samples = archive["sample_ids"].astype(str)
                expected_samples = samples[validation_indices].astype(str)
                if not np.array_equal(archive_samples, expected_samples):
                    raise RuntimeError("fold prediction sample order changed")
                fold_predictions[candidate].append(
                    (validation_indices, archive["logits"].astype(np.float32))
                )
            fold_results[candidate].append(
                {
                    "fold": fold_index,
                    "fit_user_ids": summary["fit_user_ids"],
                    "validation_user_ids": summary["validation_user_ids"],
                    "metrics": summary["validation_metrics"],
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "predictions_sha256": summary["predictions_sha256"],
                    "normalization_sha256": sha256_file(normalization_path),
                }
            )

    assert canonical_labels is not None
    assert canonical_users is not None
    assert canonical_samples is not None
    pooled_logits: dict[str, np.ndarray] = {}
    candidate_metrics: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATE_MODALITIES:
        logits = pool_fold_predictions(
            sample_count=len(canonical_labels),
            classes=40,
            folds=fold_predictions[candidate],
        )
        pooled_logits[candidate] = logits
        candidate_metrics[candidate] = _metrics(
            canonical_labels, logits, canonical_users
        )
        _atomic_npz(
            output_root / f"{candidate}_pooled_predictions.npz",
            sample_ids=canonical_samples,
            user_ids=canonical_users,
            labels=canonical_labels,
            logits=logits,
            predictions=logits.argmax(axis=1),
        )
    selected = select_grouped_candidate(candidate_metrics)
    anchor = pooled_logits["visual_only"]
    report = {
        "stage": "P5-HMF0-grouped-cv",
        "status": "completed",
        "population_samples": len(canonical_labels),
        "selected_candidate": selected,
        "selection_order": [
            "accuracy",
            "macro_f1",
            "worst_user_accuracy",
            "negative_nll",
            "fixed_candidate_order",
        ],
        "candidate_metrics": candidate_metrics,
        "comparison_to_visual_only": {
            candidate: _comparison(canonical_labels, anchor, pooled_logits[candidate])
            for candidate in CANDIDATE_MODALITIES
        },
        "fold_results": fold_results,
        "validation_users_entered_training": False,
        "user6_user7_evaluated": False,
    }
    _atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")
    lines = [
        "# Hierarchical multimodal teacher grouped-CV",
        "",
        f"- Selected candidate: `{selected}`",
        "- user6/user7 evaluated: `False`",
        "",
        "| Candidate | Accuracy | Macro-F1 | Worst-user | Rescue/Harm |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATE_MODALITIES:
        metrics = candidate_metrics[candidate]
        comparison = report["comparison_to_visual_only"][candidate]
        lines.append(
            f"| {candidate} | {metrics['accuracy']:.6f} | "
            f"{metrics['macro_f1']:.6f} | {metrics['worst_user_accuracy']:.6f} | "
            f"{comparison['rescued']}/{comparison['harmed']} |"
        )
    _atomic_write_text(markdown_path, "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "status": report["status"],
                "selected_candidate": selected,
            }
        ),
        flush=True,
    )
    return report
