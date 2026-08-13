from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.fusion.expert_evidence import ExpertEvidence
from scripts.build_x3d_s_ir_evidence import (
    FINALIZE_EPOCHS,
    QUALITY_MAPPING,
    QUALITY_MAPPING_SHA256,
    SOURCE_EPOCHS,
    finalization_policy,
    single_validation_view,
)


def fixture_evidence(**overrides: object) -> ExpertEvidence:
    rows = 3
    base = ExpertEvidence(
        role="oof_train14",
        expert_id="ir_x3d_s_k400_pure",
        sample_ids=np.asarray(["a", "b", "c"]),
        user_ids=np.asarray(["u1", "u2", "u3"]),
        logits=np.zeros((rows, 40), dtype=np.float32),
        availability=np.ones((rows, 1), dtype=bool),
        quality=np.ones((rows, 2), dtype=np.float32),
        quality_mask=np.ones((rows, 2), dtype=bool),
        fusion_quality_score=np.ones((rows, 1), dtype=np.float32),
        class_map_hash="a" * 64,
        model_sha256="b" * 64,
        config_sha256="c" * 64,
        deployed_weight_bytes=123,
        preprocessing_dependencies=("yolo11n-pose", "pose-guided-ir-context"),
        quality_mapping="constant_1",
        quality_mapping_sha256="d" * 64,
        labels=np.asarray([0, 1, 2]),
        embeddings=np.ones((rows, 4), dtype=np.float32),
        engineered_summary=None,
        diagnostics={"num_frames": np.asarray([10, 20, 30])},
    )
    return replace(base, **overrides)


def test_evidence_allows_optional_embedding_but_requires_provenance() -> None:
    evidence = fixture_evidence(
        embeddings=None, engineered_summary=np.ones((3, 5), dtype=np.float32)
    )
    evidence.validate()
    assert evidence.logits.shape == (3, 40)
    assert evidence.model_sha256
    assert evidence.config_sha256


def test_evidence_rejects_duplicate_or_unknown_samples() -> None:
    with pytest.raises(ValueError, match="duplicate sample"):
        fixture_evidence(sample_ids=np.asarray(["a", "a", "c"])).validate()
    with pytest.raises(ValueError, match="unknown sample"):
        fixture_evidence(sample_ids=np.asarray(["a", "", "c"])).validate()


def test_heldout_evidence_forbids_labels() -> None:
    with pytest.raises(ValueError, match="labels forbidden"):
        fixture_evidence(role="heldout", labels=np.asarray([0, 1, 2])).validate()


def test_oof_requires_labels_and_quality_score_range() -> None:
    with pytest.raises(ValueError, match="labels required"):
        fixture_evidence(labels=None).validate()
    with pytest.raises(ValueError, match="fusion quality"):
        fixture_evidence(fusion_quality_score=np.asarray([[1.1], [1.0], [1.0]])).validate()


def test_heldout_roundtrip_physically_omits_labels(tmp_path: Path) -> None:
    path = tmp_path / "heldout_evidence.npz"
    evidence = fixture_evidence(role="heldout", labels=None)
    evidence.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert "labels" not in archive.files
    loaded = ExpertEvidence.load(path)
    loaded.validate()
    assert loaded.role == "heldout"
    assert loaded.labels is None


def test_ir_quality_mapping_and_finalization_are_frozen_before_training() -> None:
    import hashlib

    assert QUALITY_MAPPING == "constant_1_for_first_generation_fusion"
    assert QUALITY_MAPPING_SHA256 == hashlib.sha256(QUALITY_MAPPING.encode()).hexdigest()
    assert sorted(SOURCE_EPOCHS)[4] == FINALIZE_EPOCHS == 11
    policy = finalization_policy()
    assert policy["final_seed"] == 20260715
    assert policy["scheduler_horizon_epochs"] == 30
    assert policy["selection_uses_heldout"] is False
    assert policy["heldout_labels_serialized"] is False


def test_label_free_inference_removes_the_single_validation_view_dimension() -> None:
    import torch

    clips = torch.zeros((2, 3, 1, 3, 13, 182, 182))
    normalized = single_validation_view(clips)
    assert normalized.shape == (2, 3, 3, 13, 182, 182)
    with pytest.raises(ValueError, match="V=1"):
        single_validation_view(torch.zeros((2, 3, 2, 3, 13, 182, 182)))
