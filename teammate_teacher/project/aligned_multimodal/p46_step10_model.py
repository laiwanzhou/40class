from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from p46_event_model import P46EventTokenEncoder


DETAIL_CLASSES = 21
MODALITY_TARGET_WIDTH = 4
MODALITY_NAMES = ("visual", "skeleton", "imu")


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, source: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return source.view_as(source)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def gradient_reverse(source: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(source, scale)


class P46Step10Model(nn.Module):
    """P46 Step 10 heads around the shared Steps [7]-[9] event encoder.

    The Detail21 head is a training/evaluation probe for Step 10.  It is not the
    routed Step 11 expert and it never receives Base top-k candidates.
    """

    def __init__(
        self,
        width: int = 192,
        dropout: float = 0.12,
        subjects: int = 12,
    ) -> None:
        super().__init__()
        self.encoder = P46EventTokenEncoder(width=width, dropout=dropout)
        self.motion_alignment = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 128, bias=False)
        )
        self.visual_alignment = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 128, bias=False)
        )
        self.modality_reconstruction = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Linear(192, len(MODALITY_NAMES) * MODALITY_TARGET_WIDTH),
        )
        self.order_head = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )
        self.detail_head = nn.Sequential(
            nn.LayerNorm(384), nn.Dropout(0.15), nn.Linear(384, DETAIL_CLASSES)
        )
        self.contrast_projection = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 128, bias=False)
        )
        self.subject_head = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 128), nn.GELU(), nn.Linear(128, subjects)
        )

    @staticmethod
    def _sequence_halves(
        sequence: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = sequence.shape[1]
        position = torch.linspace(
            0.0, 1.0, steps, device=sequence.device, dtype=sequence.dtype
        )[None]
        first_mask = frame_mask & (position <= 0.45)
        last_mask = frame_mask & (position >= 0.55)
        # One-frame and extremely short trials still have a defined order pair.
        first_mask = torch.where(first_mask.any(1, keepdim=True), first_mask, frame_mask)
        last_mask = torch.where(last_mask.any(1, keepdim=True), last_mask, frame_mask)

        def pool(mask: torch.Tensor) -> torch.Tensor:
            weight = mask.to(sequence.dtype).unsqueeze(-1)
            return (sequence * weight).sum(1) / weight.sum(1).clamp_min(1.0)

        return pool(first_mask), pool(last_mask)

    def forward(
        self, batch: dict[str, torch.Tensor], subject_adversarial_scale: float = 0.0
    ) -> dict[str, torch.Tensor]:
        output = self.encoder(batch)
        first, last = self._sequence_halves(
            output["event_temporal_sequence"], batch["frame_mask"]
        )
        embedding = output["trial_embedding"]
        return {
            **output,
            "alignment_motion": F.normalize(
                self.motion_alignment(output["motion_query"]).float(), dim=-1, eps=1e-6
            ),
            "alignment_visual": F.normalize(
                self.visual_alignment(output["attended_visual"]).float(), dim=-1, eps=1e-6
            ),
            "modality_reconstruction": self.modality_reconstruction(embedding),
            "order_forward_logits": self.order_head(torch.cat((first, last), dim=-1)),
            "order_reverse_logits": self.order_head(torch.cat((last, first), dim=-1)),
            "detail_logits": self.detail_head(embedding),
            "contrast_embedding": F.normalize(
                self.contrast_projection(embedding).float(), dim=-1, eps=1e-6
            ),
            "subject_logits": self.subject_head(
                gradient_reverse(embedding, subject_adversarial_scale)
            ),
        }


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]
) -> torch.Tensor:
    weight = mask.to(values.dtype)
    return (values * weight).sum(dimensions) / weight.sum(dimensions).clamp_min(1.0)


def modality_summary_targets(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Twelve bounded low-dimensional targets: four per masked modality."""

    frame = batch["frame_mask"]
    frame_float = frame.to(batch["arm_spatial_features"].dtype)

    def frame_reduce(value: torch.Tensor) -> torch.Tensor:
        reduced = value.flatten(start_dim=2).abs().mean(dim=2)
        return _masked_mean(reduced, frame_float, (1,))

    arm = torch.log1p(frame_reduce(batch["arm_spatial_features"]))
    detail = torch.log1p(frame_reduce(batch["detail_spatial_features"]))
    context = torch.log1p(frame_reduce(batch["context_features"]))
    visual_frame = batch["detail_spatial_features"].flatten(start_dim=2).float().mean(2)
    visual_change = visual_frame.new_zeros(visual_frame.shape)
    if visual_frame.shape[1] > 1:
        visual_change[:, 1:] = (visual_frame[:, 1:] - visual_frame[:, :-1]).abs()
    visual_change = torch.log1p(_masked_mean(visual_change, frame_float, (1,)))

    joint = batch["skeleton_joint_mask"] & frame.unsqueeze(-1)
    joint_weight = joint.to(batch["skeleton_features"].dtype)
    position = torch.linalg.vector_norm(batch["skeleton_features"][..., 0:3], dim=-1)
    velocity = torch.linalg.vector_norm(batch["skeleton_features"][..., 6:9], dim=-1)
    acceleration = torch.linalg.vector_norm(batch["skeleton_features"][..., 9:12], dim=-1)
    skeleton_position = torch.log1p(_masked_mean(position, joint_weight, (1, 2)))
    skeleton_velocity = torch.log1p(_masked_mean(velocity, joint_weight, (1, 2)))
    skeleton_acceleration = torch.log1p(
        _masked_mean(acceleration, joint_weight, (1, 2))
    )
    skeleton_quality = _masked_mean(
        batch["skeleton_frame_quality"], frame_float, (1,)
    ).clamp(0.0, 1.0)

    imu_mask = batch["imu_point_mask"]
    imu_weight = imu_mask.to(batch["imu_values"].dtype)
    acceleration_imu = torch.linalg.vector_norm(batch["imu_values"][..., 0:3], dim=-1)
    gyro_imu = torch.linalg.vector_norm(batch["imu_values"][..., 3:6], dim=-1)
    imu_acceleration = torch.log1p(_masked_mean(acceleration_imu, imu_weight, (1, 2)))
    imu_gyro = torch.log1p(_masked_mean(gyro_imu, imu_weight, (1, 2)))
    imu_point_fraction = imu_weight.mean(dim=(1, 2))
    imu_device_fraction = batch["imu_device_mask"].float().mean(dim=1)
    return torch.stack(
        (
            arm,
            detail,
            context,
            visual_change,
            skeleton_position,
            skeleton_velocity,
            skeleton_acceleration,
            skeleton_quality,
            imu_acceleration,
            imu_gyro,
            imu_point_fraction,
            imu_device_fraction,
        ),
        dim=1,
    )


def mask_modalities(
    batch: dict[str, torch.Tensor], assignment: torch.Tensor
) -> dict[str, torch.Tensor | list[str]]:
    """Mask one complete modality per sample without changing temporal length."""

    output: dict[str, torch.Tensor | list[str]] = dict(batch)

    def replace(keys: tuple[str, ...], selected: torch.Tensor, boolean: bool = False) -> None:
        for key in keys:
            value = batch[key].clone()
            value[selected] = False if boolean else 0
            output[key] = value

    visual = assignment == 0
    replace(
        (
            "arm_spatial_features",
            "detail_spatial_features",
            "local_geometry_features",
            "oriented_roi_geometry",
            "local_roi_quality",
            "local_roi_clipped_ratio",
            "pose_quality_factor",
            "context_features",
            "context_quality",
        ),
        visual,
    )
    replace(
        ("oriented_angle_valid", "local_roi_valid", "context_valid"),
        visual,
        boolean=True,
    )

    skeleton = assignment == 1
    replace(
        (
            "skeleton_features",
            "skeleton_relations",
            "skeleton_frame_quality",
            "body_axes_camera",
        ),
        skeleton,
    )
    replace(
        (
            "skeleton_feature_mask",
            "skeleton_joint_mask",
            "skeleton_relation_mask",
            "body_axes_raw_valid",
        ),
        skeleton,
        boolean=True,
    )

    imu = assignment == 2
    replace(
        (
            "imu_values",
            "imu_raw_vectors",
            "imu_time_seconds",
            "imu_interval_counts",
        ),
        imu,
    )
    frame_index = batch["imu_frame_index"].clone()
    frame_index[imu] = -1
    output["imu_frame_index"] = frame_index
    replace(("imu_point_mask", "imu_device_mask"), imu, boolean=True)
    return output  # type: ignore[return-value]


def left_right_swap_batch(
    batch: dict[str, torch.Tensor | list[str]],
) -> dict[str, torch.Tensor | list[str]]:
    """Synchronous semantic left/right flip across visual, Skeleton and IMU."""

    output = dict(batch)
    two = torch.tensor((1, 0), device=batch["frame_mask"].device)
    detail = torch.tensor((1, 0, 2), device=two.device)
    local = torch.tensor((1, 0, 3, 2, 4), device=two.device)
    joints = torch.tensor(
        (0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13),
        device=two.device,
    )
    relations = torch.tensor(
        (1, 0, 2, 4, 3, 6, 5, 8, 7, 9, 10, 11, 12, 13, 14, 16, 15, 17),
        device=two.device,
    )
    devices = torch.tensor((0, 2, 1, 4, 3), device=two.device)

    output["arm_spatial_features"] = batch["arm_spatial_features"].index_select(3, two).flip(5)
    output["detail_spatial_features"] = batch["detail_spatial_features"].index_select(3, detail).flip(5)
    output["local_geometry_features"] = batch["local_geometry_features"].index_select(2, local).flip(4)
    for key in (
        "oriented_angle_valid",
        "local_roi_valid",
        "local_roi_quality",
        "local_roi_source",
        "local_roi_clipped_ratio",
    ):
        output[key] = batch[key].index_select(2, local)
    geometry = batch["oriented_roi_geometry"].index_select(2, local).clone()
    geometry[..., 0] = 1.0 - geometry[..., 0]
    geometry[..., 5] *= -1.0
    output["oriented_roi_geometry"] = geometry

    skeleton_features = batch["skeleton_features"].index_select(2, joints).clone()
    skeleton_features[..., (0, 3, 6, 9)] *= -1.0
    output["skeleton_features"] = skeleton_features
    output["skeleton_feature_mask"] = batch["skeleton_feature_mask"].index_select(2, joints)
    output["skeleton_joint_mask"] = batch["skeleton_joint_mask"].index_select(2, joints)
    skeleton_relations = batch["skeleton_relations"].index_select(2, relations).clone()
    skeleton_relations[..., 12] *= -1.0
    output["skeleton_relations"] = skeleton_relations
    output["skeleton_relation_mask"] = batch["skeleton_relation_mask"].index_select(2, relations)
    axes = batch["body_axes_camera"].clone()
    axes[..., 0] *= -1.0
    axes[..., 2] *= -1.0
    output["body_axes_camera"] = axes

    for key in (
        "imu_values",
        "imu_raw_vectors",
        "imu_time_seconds",
        "imu_frame_index",
        "imu_point_mask",
        "imu_device_mask",
    ):
        output[key] = batch[key].index_select(1, devices)
    output["imu_interval_counts"] = batch["imu_interval_counts"].index_select(2, devices)
    return output


def selected_modality_reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor, assignment: torch.Tensor
) -> torch.Tensor:
    prediction = prediction.reshape(len(prediction), len(MODALITY_NAMES), MODALITY_TARGET_WIDTH)
    target = target.reshape(len(target), len(MODALITY_NAMES), MODALITY_TARGET_WIDTH)
    row = torch.arange(len(prediction), device=prediction.device)
    return F.smooth_l1_loss(prediction[row, assignment], target[row, assignment])


def same_part_alignment_loss(
    output: dict[str, torch.Tensor], maximum_per_trial: int = 12
) -> torch.Tensor:
    motion_values: list[torch.Tensor] = []
    visual_values: list[torch.Tensor] = []
    for index in range(output["event_mask"].shape[0]):
        valid = output["event_mask"][index].flatten()
        if not valid.any():
            continue
        weight = (
            output["soft_event_gate"][index]
            + output["contact_proxy"][index]
            + 0.05
        ).flatten()
        candidate = torch.nonzero(valid, as_tuple=False).squeeze(1)
        count = min(maximum_per_trial, len(candidate))
        chosen = candidate[torch.topk(weight[candidate], count).indices]
        motion_values.append(output["alignment_motion"][index].reshape(-1, 128)[chosen])
        visual_values.append(output["alignment_visual"][index].reshape(-1, 128)[chosen])
    if not motion_values:
        return output["trial_embedding"].sum() * 0.0
    motion = torch.cat(motion_values)
    visual = torch.cat(visual_values)
    if len(motion) < 2:
        return motion.sum() * 0.0
    logits = motion.float() @ visual.float().transpose(0, 1) / 0.10
    target = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def weak_contact_targets(
    batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    geometry = batch["local_geometry_features"].float()
    non_body_depth = geometry[..., 2] * (1.0 - geometry[..., 5]).clamp(0.0, 1.0)
    objectness = non_body_depth[..., 1:4, 1:4].mean(dim=(-1, -2))
    valid = batch["local_roi_valid"] & batch["frame_mask"].unsqueeze(-1)
    left = torch.sqrt((objectness[:, :, 2] * objectness[:, :, 4]).clamp_min(0.0))
    right = torch.sqrt((objectness[:, :, 3] * objectness[:, :, 4]).clamp_min(0.0))
    left = left * (valid[:, :, 2] & valid[:, :, 4])
    right = right * (valid[:, :, 3] & valid[:, :, 4])
    target = geometry.new_zeros(*geometry.shape[:2], 10)
    mask = torch.zeros_like(target, dtype=torch.bool)
    for part in (3, 5):
        target[:, :, part] = left
        mask[:, :, part] = valid[:, :, 2] & valid[:, :, 4]
    for part in (4, 6):
        target[:, :, part] = right
        mask[:, :, part] = valid[:, :, 3] & valid[:, :, 4]
    target[:, :, 9] = torch.maximum(left, right)
    mask[:, :, 9] = valid[:, :, 4]
    return target.clamp(0.0, 1.0), mask


def contact_and_phase_losses(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contact_target, contact_mask = weak_contact_targets(batch)
    contact_weight = (0.25 + output["soft_event_gate"].detach()) * contact_mask
    # contact_proxy is already a probability.  Compute BCE explicitly in FP32;
    # torch intentionally rejects probability-space BCE inside autocast.
    probability = output["contact_proxy"].float().clamp(1e-5, 1.0 - 1e-5)
    target_float = contact_target.float()
    contact_raw = -(
        target_float * torch.log(probability)
        + (1.0 - target_float) * torch.log1p(-probability)
    )
    contact_loss = (contact_raw * contact_weight).sum() / contact_weight.sum().clamp_min(1.0)

    motion = (
        torch.log1p(output["skeleton_motion"].detach().clamp_min(0.0))
        + torch.log1p(output["imu_motion"].detach().clamp_min(0.0))
        + torch.log1p(output["visual_motion"].detach().clamp_min(0.0))
    ).mean(dim=2)
    scale = torch.quantile(motion, 0.80, dim=1, keepdim=True).clamp_min(1e-4)
    motion = (motion / scale).clamp(0.0, 1.0)
    contact = contact_target[:, :, (5, 6, 9)].amax(dim=2)
    previous = torch.zeros_like(contact)
    previous[:, 1:] = contact[:, :-1]
    change = contact - previous
    phase = torch.zeros_like(contact, dtype=torch.long)  # approach
    phase[(contact > 0.30) & (motion > 0.55)] = 2  # manipulate
    phase[(contact > 0.45) & (motion <= 0.12)] = 4  # static hold
    phase[change > 0.08] = 1  # contact onset
    phase[change < -0.08] = 3  # release
    phase_mask = batch["frame_mask"] & (
        (motion > 0.95)
        | (change.abs() > 0.08)
        | ((contact > 0.45) & (motion < 0.12))
    )
    empty = ~phase_mask.any(dim=1)
    if empty.any():
        strongest = motion.argmax(dim=1)
        phase_mask[empty, strongest[empty]] = True
    phase_raw = F.cross_entropy(
        output["phase_logits"].transpose(1, 2), phase, reduction="none"
    )
    phase_weight = (0.25 + motion + contact) * phase_mask
    phase_loss = (phase_raw * phase_weight).sum() / phase_weight.sum().clamp_min(1.0)
    phase_coverage = phase_mask.float().sum() / batch["frame_mask"].float().sum().clamp_min(1.0)
    return contact_loss, phase_loss, phase_coverage


def temporal_order_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    positive = torch.ones(
        len(output["order_forward_logits"]), dtype=torch.long, device=output["order_forward_logits"].device
    )
    negative = torch.zeros_like(positive)
    return 0.5 * (
        F.cross_entropy(output["order_forward_logits"], positive)
        + F.cross_entropy(output["order_reverse_logits"], negative)
    )


def hardest_rival_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    true = logits.gather(1, labels[:, None]).squeeze(1)
    rival = logits.masked_fill(
        F.one_hot(labels, logits.shape[1]).bool(), torch.finfo(logits.dtype).min
    ).amax(dim=1)
    return F.relu(margin - true + rival).mean()


def cross_subject_supervised_contrastive(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    subject_index: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.sum() * 0.0
    similarity = embedding.float() @ embedding.float().transpose(0, 1) / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = (
        labels[:, None].eq(labels[None, :])
        & subject_index[:, None].ne(subject_index[None, :])
        & ~identity
    )
    valid = positive.any(dim=1)
    if not valid.any():
        return embedding.sum() * 0.0
    logits = similarity.masked_fill(identity, -1e4)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    mean_positive = (log_probability * positive).sum(1) / positive.sum(1).clamp_min(1)
    return -mean_positive[valid].mean()


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
