from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.build_midfusion_skeleton_clean_views import projection_scope_specs
from src.data.clean_skeleton_segments import (
    apply_skeleton_normalization,
    fit_skeleton_normalization,
    load_skeleton_segments,
    resample_skeleton_segments,
)
from src.experiments.hierarchical_midfusion_config import load_midfusion_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml"


def synthetic_h36m_poses(frames: np.ndarray) -> np.ndarray:
    base = np.stack(
        (
            np.linspace(0.0, 1.6, 17),
            np.sin(np.linspace(0.0, np.pi, 17)),
            np.cos(np.linspace(0.0, np.pi, 17)),
        ),
        axis=1,
    )
    return np.stack([base + np.asarray([0.01 * frame, 0.0, 0.0]) for frame in frames])


def test_skeleton_segment_loader_never_interpolates_across_gap() -> None:
    frames = np.asarray([0, 1, 10, 11])
    segment_ids = np.asarray([0, 0, 1, 1])
    poses = synthetic_h36m_poses(frames)

    result = resample_skeleton_segments(
        frames, segment_ids, poses, segment_count=8
    )

    assert not result.mask[2:6].any()
    assert torch.count_nonzero(result.features[~result.mask]) == 0


def test_skeleton_features_are_xyz_plus_segment_local_velocity() -> None:
    frames = np.arange(8)
    segment_ids = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    poses = synthetic_h36m_poses(frames)

    result = resample_skeleton_segments(
        frames, segment_ids, poses, segment_count=8
    )

    assert result.features.shape == (8, 17, 6)
    assert torch.allclose(result.features[0, :, 3:], torch.zeros(17, 3))
    assert torch.allclose(result.features[4, :, 3:], torch.zeros(17, 3))


def test_projection_scopes_fit_only_fold_fit_users() -> None:
    config = load_midfusion_config(CONFIG)

    scopes = projection_scope_specs(config)

    assert len(scopes) == 4
    for scope in scopes:
        assert set(scope["projection_fit_user_ids"]).isdisjoint(
            scope["scope_validation_user_ids"]
        )
        assert set(scope["projection_fit_user_ids"]) == set(
            scope["fold_fit_user_ids"]
        )
    final = scopes[-1]
    assert final["scope"] == "selected_final"
    assert set(final["projection_fit_user_ids"]) == set(
        config["population"]["train_user_ids"]
    )
    assert set(final["scope_validation_user_ids"]) == {"user6", "user7"}


def test_load_skeleton_segments_reads_selected_candidate(tmp_path: Path) -> None:
    rows = []
    for frame_id in range(8):
        path = tmp_path / f"{frame_id}.json"
        pose = synthetic_h36m_poses(np.asarray([frame_id]))[0]
        path.write_text(
            __import__("json").dumps([{"keypoints": pose.tolist()}]),
            encoding="utf-8",
        )
        rows.append(
            {
                "sample_id": "sample",
                "frame_id": frame_id,
                "retained_segment_index": 0 if frame_id < 4 else 1,
                "skeleton_json_path": path.name,
                "candidate_index": 0,
                "use_for_frame_training": True,
            }
        )

    result = load_skeleton_segments(
        pd.DataFrame(rows), data_root=tmp_path, segment_count=8
    )

    assert result.features.shape == (8, 17, 6)
    assert result.mask.all()
    assert torch.isfinite(result.features).all()


def test_skeleton_normalization_uses_only_valid_segments() -> None:
    frames = np.arange(8)
    result = resample_skeleton_segments(
        frames,
        np.zeros(8, dtype=np.int64),
        synthetic_h36m_poses(frames),
        segment_count=8,
    )
    mean, std = fit_skeleton_normalization([result])
    normalized = apply_skeleton_normalization(result, mean, std)

    assert mean.shape == (17, 6)
    assert std.shape == (17, 6)
    assert torch.isfinite(normalized.features).all()
    assert torch.count_nonzero(normalized.features[~normalized.mask]) == 0
