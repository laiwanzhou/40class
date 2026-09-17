from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p86_mobind_lite_model import (  # noqa: E402
    MotionSemanticHead,
    P86JointMotionEncoder,
    P86MoBindLite,
    P86MoBindMotionResidual,
    P86SeparateMotionEncoder,
)
from train_p86_mobind_pretrain import (  # noqa: E402
    asymmetric_token_distillation,
    load_checkpoint_branch,
    loader as mobind_loader,
)


def motion_batch(batch: int, steps: int, present: bool) -> dict[str, torch.Tensor]:
    skeleton_joint_mask = torch.full((batch, 2, steps, 17), present)
    skeleton_feature_mask = skeleton_joint_mask.unsqueeze(-1).expand(
        -1, -1, -1, -1, 13
    )
    skeleton_relation_mask = torch.full((batch, 2, steps, 18), present)
    imu_bin_mask = torch.full((batch, 2, steps, 5), present)
    imu_sequence_mask = imu_bin_mask.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
    return {
        "skeleton_features": torch.randn(batch, 2, steps, 17, 13),
        "skeleton_feature_mask": skeleton_feature_mask,
        "skeleton_joint_mask": skeleton_joint_mask,
        "skeleton_relations": torch.randn(batch, 2, steps, 18),
        "skeleton_relation_mask": skeleton_relation_mask,
        "skeleton_frame_quality": torch.ones(batch, 2, steps),
        "imu_sequences": torch.randn(batch, 2, steps, 5, 4, 16),
        "imu_sequence_mask": imu_sequence_mask,
        "imu_bin_statistics": torch.randn(batch, 2, steps, 5, 52),
        "imu_bin_mask": imu_bin_mask,
        "imu_global_statistics": torch.randn(batch, 5, 48),
        "imu_global_mask": torch.tensor([1.0, 1.0])
        .view(1, 1, 2)
        .expand(batch, 5, 2),
    }


def test_all_label_mobind_loader_keeps_terminal_partial_batch() -> None:
    args = SimpleNamespace(
        batch_size=64,
        workers=0,
        final_refit=False,
        all_label_refit=True,
        subject_holdout_users=[],
    )
    data = mobind_loader(list(range(2914)), args, shuffle=True)
    assert not data.drop_last
    assert len(data) == 46


def test_mobind_preserves_part_and_time_tokens() -> None:
    model = P86MoBindLite(width=32, alignment_width=16, dropout=0.0).train()
    output = model(motion_batch(3, 6, present=True), mask_ratio=0.25)
    assert output["skeleton_tokens"].shape == (3, 2, 6, 5, 32)
    assert output["imu_tokens"].shape == (3, 2, 6, 5, 32)
    assert output["skeleton_alignment"].shape == (3, 2, 6, 5, 16)
    assert output["imu_alignment"].shape == (3, 2, 6, 5, 16)
    assert output["skeleton_logits"].shape == (3, 40)
    assert output["imu_logits"].shape == (3, 40)
    assert output["skeleton_compact"].shape == (3, 64)
    assert output["imu_compact"].shape == (3, 64)
    assert output["skeleton_reconstruction"].shape == (3, 2 * 6 * 5, 32)
    assert output["imu_reconstruction_mask"].dtype == torch.bool
    assert sum(parameter.numel() for parameter in model.parameters()) < 3_000_000


def test_single_modality_residual_has_exact_missing_fallback() -> None:
    pretrained = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    residual = P86MoBindMotionResidual(
        "skeleton",
        pretrained.skeleton_encoder,
        pretrained.skeleton_head,
        pretrained.skeleton_teacher_projection,
        motion_width=32,
        dropout=0.0,
    ).eval()
    visual = torch.randn(2, 2, 3, 512)
    fused, audit = residual(
        visual, torch.rand(2, 2, 4), motion_batch(2, 4, present=False)
    )
    assert torch.equal(fused, visual)
    assert not audit["motion_available"].any()
    assert audit["motion_part_attention"].shape == (2, 2, 3, 4, 5)
    global_visual = torch.randn(2, 512)
    global_fused, _ = residual.fuse_global(
        global_visual,
        audit["motion_semantic_embedding"],
        audit["motion_available"],
    )
    assert torch.equal(global_fused, global_visual)


def test_imu_residual_backpropagates_to_continuous_encoder() -> None:
    pretrained = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    residual = P86MoBindMotionResidual(
        "imu",
        pretrained.imu_encoder,
        pretrained.imu_head,
        pretrained.imu_teacher_projection,
        motion_width=32,
        dropout=0.0,
    )
    visual = torch.randn(2, 2, 3, 512)
    fused, audit = residual(
        visual, torch.rand(2, 2, 4), motion_batch(2, 4, present=True)
    )
    assert 0.0 < float(audit["motion_residual_strength"].detach()) <= 1.0
    (fused.square().mean() + audit["motion_logits"].square().mean()).backward()
    assert pretrained.imu_encoder.raw_stem[1].weight.grad is not None
    assert pretrained.imu_encoder.point_blocks[0].depthwise.weight.grad is not None
    assert (
        pretrained.imu_encoder.device_encoder.layers[0].self_attn.in_proj_weight.grad
        is not None
    )
    assert pretrained.imu_encoder.statistics_projection[1].weight.grad is not None


def test_imu_event_features_enter_semantic_forward() -> None:
    model = P86MoBindLite(
        width=32,
        alignment_width=16,
        dropout=0.0,
        imu_event_feature_width=10,
    )
    batch = motion_batch(2, 4, present=True)
    batch["imu_event_features"] = torch.randn(2, 10)
    batch["imu_event_valid"] = torch.tensor([True, False])
    output = model(batch)
    output["imu_logits"].square().mean().backward()
    assert model.imu_encoder.event_projection is not None
    assert model.imu_encoder.event_projection[1].weight.grad is not None


def test_imu_event_features_can_condition_reliability_only() -> None:
    pretrained = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    residual = P86MoBindMotionResidual(
        "imu",
        pretrained.imu_encoder,
        pretrained.imu_head,
        pretrained.imu_teacher_projection,
        motion_width=32,
        dropout=0.0,
        reliability_event_feature_width=10,
    )
    batch = motion_batch(2, 4, present=True)
    batch["imu_event_features"] = torch.randn(2, 10)
    batch["imu_event_valid"] = torch.tensor([True, False])
    fused, audit = residual(torch.randn(2, 2, 3, 512), torch.rand(2, 2, 4), batch)
    (fused.square().mean() + audit["motion_reliability"].square().mean()).backward()
    assert residual.event_reliability_projection is not None
    assert residual.event_reliability_projection[1].weight.grad is not None


def test_conditional_global_fusion_uses_both_modalities() -> None:
    pretrained = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    residual = P86MoBindMotionResidual(
        "imu",
        pretrained.imu_encoder,
        pretrained.imu_head,
        pretrained.imu_teacher_projection,
        motion_width=32,
        dropout=0.0,
        global_fusion_mode="conditional",
    )
    visual = torch.randn(2, 512, requires_grad=True)
    motion = torch.randn(2, residual.semantic_input_width, requires_grad=True)
    available = torch.ones(2, 2, dtype=torch.bool)
    fused, _ = residual.fuse_global(visual, motion, available)
    fused.square().mean().backward()
    assert visual.grad is not None
    assert motion.grad is not None
    assert residual.global_motion_context is not None
    assert residual.global_motion_context[1].weight.grad is not None


def test_asymmetric_token_distillation_only_updates_student() -> None:
    student = torch.randn(2, 2, 4, 5, 16, requires_grad=True)
    teacher = torch.randn(2, 2, 4, 5, 16, requires_grad=True)
    valid = torch.ones(2, 2, 4, 5, dtype=torch.bool)
    loss = asymmetric_token_distillation(student, teacher, valid, valid)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_modality_specific_initialization_builds_exact_hybrid(
    tmp_path: Path,
) -> None:
    skeleton_source = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    imu_source = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    target = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    with torch.no_grad():
        for name, parameter in skeleton_source.named_parameters():
            if name.startswith("skeleton_"):
                parameter.fill_(0.125)
        for name, parameter in imu_source.named_parameters():
            if name.startswith("imu_"):
                parameter.fill_(0.25)
    skeleton_path = tmp_path / "skeleton.pt"
    imu_path = tmp_path / "imu.pt"
    torch.save({"model_state": skeleton_source.state_dict()}, skeleton_path)
    torch.save({"model_state": imu_source.state_dict()}, imu_path)

    skeleton_audit = load_checkpoint_branch(target, skeleton_path, "skeleton_")
    imu_audit = load_checkpoint_branch(target, imu_path, "imu_")
    assert skeleton_audit["tensors"] > 0
    assert imu_audit["tensors"] > 0
    for name, value in target.state_dict().items():
        if name.startswith("skeleton_"):
            torch.testing.assert_close(value, skeleton_source.state_dict()[name])
        elif name.startswith("imu_"):
            torch.testing.assert_close(value, imu_source.state_dict()[name])


def test_grouped_reliability_modulates_feature_channels() -> None:
    pretrained = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    residual = P86MoBindMotionResidual(
        "imu",
        pretrained.imu_encoder,
        pretrained.imu_head,
        pretrained.imu_teacher_projection,
        motion_width=32,
        dropout=0.0,
        reliability_groups=16,
    )
    fused, audit = residual(
        torch.randn(2, 2, 3, 512),
        torch.rand(2, 2, 4),
        motion_batch(2, 4, present=True),
    )
    assert fused.shape == (2, 2, 3, 512)
    assert audit["motion_reliability"].shape == (2, 2, 3, 16)
    global_fused, global_gate = residual.fuse_global(
        torch.randn(2, 512),
        audit["motion_semantic_embedding"],
        audit["motion_available"],
    )
    assert global_fused.shape == (2, 512)
    assert global_gate.shape == (2, 16)


def test_skeleton_encoder_uses_exact_global_time_position() -> None:
    model = P86MoBindLite(
        width=32,
        alignment_width=16,
        dropout=0.0,
        skeleton_time_position=True,
    )
    residual = P86MoBindMotionResidual(
        "skeleton",
        model.skeleton_encoder,
        model.skeleton_head,
        model.skeleton_teacher_projection,
        motion_width=32,
        dropout=0.0,
    )
    time = torch.tensor(
        [[[0.05, 0.12, 0.21, 0.33], [0.62, 0.73, 0.86, 1.0]]],
        dtype=torch.float32,
    ).expand(2, -1, -1)
    fused, _ = residual(
        torch.randn(2, 2, 3, 512),
        time,
        motion_batch(2, 4, present=True),
    )
    fused.square().mean().backward()
    assert model.skeleton_encoder.time_projection is not None
    assert model.skeleton_encoder.time_projection[0].weight.grad is not None


def test_skeleton_multistream_encoder_preserves_all_signal_groups() -> None:
    model = P86MoBindLite(
        width=32,
        alignment_width=16,
        dropout=0.0,
        skeleton_multistream=True,
    )
    residual = P86MoBindMotionResidual(
        "skeleton",
        model.skeleton_encoder,
        model.skeleton_head,
        model.skeleton_teacher_projection,
        motion_width=32,
        dropout=0.0,
    )
    fused, _ = residual(
        torch.randn(2, 2, 3, 512),
        torch.rand(2, 2, 4),
        motion_batch(2, 4, present=True),
    )
    fused.square().mean().backward()
    encoder = model.skeleton_encoder
    assert encoder.stream_stems is not None
    assert encoder.stream_fusion is not None
    assert encoder.multistream_residual_logit.grad is not None
    for stem in encoder.stream_stems:
        assert stem[1].weight.grad is not None
    assert encoder.stream_fusion[1].weight.grad is not None


def test_skeleton_adaptive_graph_is_zero_initialized_residual() -> None:
    torch.manual_seed(5)
    fixed = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    torch.manual_seed(5)
    adaptive = P86MoBindLite(
        width=32,
        alignment_width=16,
        dropout=0.0,
        skeleton_adaptive_graph=True,
    )
    incompatible = adaptive.load_state_dict(fixed.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        "skeleton_encoder.blocks.0.adaptive_adjacency",
        "skeleton_encoder.blocks.1.adaptive_adjacency",
    }
    batch = motion_batch(2, 4, present=True)
    inputs = (
        batch["skeleton_features"],
        batch["skeleton_feature_mask"],
        batch["skeleton_joint_mask"],
        batch["skeleton_relations"],
        batch["skeleton_relation_mask"],
        batch["skeleton_frame_quality"],
    )
    fixed.eval()
    adaptive.eval()
    fixed_tokens, _ = fixed.skeleton_encoder(*inputs)
    adaptive_tokens, _ = adaptive.skeleton_encoder(*inputs)
    torch.testing.assert_close(adaptive_tokens, fixed_tokens)
    adaptive_tokens.square().mean().backward()
    for block in adaptive.skeleton_encoder.blocks:
        assert block.adaptive_adjacency is not None
        assert block.adaptive_adjacency.grad is not None


def test_joint_motion_tokens_align_parts_and_preserve_skeleton_fallback() -> None:
    pretrained = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    joint = P86JointMotionEncoder(
        pretrained.skeleton_encoder,
        pretrained.imu_encoder,
        pretrained.skeleton_projection,
        pretrained.imu_projection,
        pretrained.imu_head,
        width=32,
        alignment_width=16,
        dropout=0.0,
        maximum_imu_residual=0.25,
    )
    batch = motion_batch(2, 4, present=True)
    tokens, valid = joint(batch, torch.rand(2, 2, 4))
    assert tokens.shape == (2, 2, 4, 5, 32)
    assert valid.all()
    assert joint.last_token_reliability is not None
    assert float(joint.last_token_reliability) <= 0.25
    tokens.square().mean().backward()
    assert pretrained.skeleton_encoder.joint_stem[1].weight.grad is not None
    assert pretrained.imu_encoder.raw_stem[1].weight.grad is not None
    for parameter in joint.fresh_parameters():
        assert parameter.grad is not None

    missing_imu = motion_batch(2, 4, present=True)
    for key in ("imu_sequence_mask", "imu_bin_mask", "imu_global_mask"):
        missing_imu[key] = torch.zeros_like(missing_imu[key])
    expected, expected_mask = pretrained.skeleton_encoder(
        missing_imu["skeleton_features"],
        missing_imu["skeleton_feature_mask"],
        missing_imu["skeleton_joint_mask"],
        missing_imu["skeleton_relations"],
        missing_imu["skeleton_relation_mask"],
        missing_imu["skeleton_frame_quality"],
    )
    actual, actual_mask = joint(missing_imu)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_mask, expected_mask)


def test_separate_motion_tokens_keep_both_modalities_and_private_statistics() -> None:
    pretrained = P86MoBindLite(width=32, alignment_width=16, dropout=0.0)
    separate = P86SeparateMotionEncoder(
        pretrained.skeleton_encoder,
        pretrained.imu_encoder,
        width=32,
    )
    batch = motion_batch(2, 4, present=True)
    tokens, valid = separate(batch, torch.rand(2, 2, 4))
    assert tokens.shape == (2, 2, 4, 10, 32)
    assert valid.shape == (2, 2, 4, 10)
    assert valid.all()
    tokens.square().mean().backward()
    assert separate.modality_embedding.grad is not None
    assert separate.statistics_lift[1].weight.grad is not None

    residual = P86MoBindMotionResidual(
        "separate",
        separate,
        pretrained.skeleton_head,
        pretrained.skeleton_teacher_projection,
        motion_width=32,
        dropout=0.0,
    )
    fused, audit = residual(
        torch.randn(2, 2, 3, 512),
        torch.rand(2, 2, 4),
        batch,
    )
    assert fused.shape == (2, 2, 3, 512)
    assert audit["motion_logits"].shape == (2, 40)

    dropout_encoder = P86SeparateMotionEncoder(
        pretrained.skeleton_encoder,
        pretrained.imu_encoder,
        width=32,
        modality_dropout=0.49,
    ).train()
    torch.manual_seed(0)
    _, dropout_mask = dropout_encoder(batch, torch.rand(2, 2, 4))
    assert dropout_mask.flatten(1).any(dim=1).all()
    assert (~dropout_mask).any()
