from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.motionbert_skeleton_dataset import (
    MotionBERTSkeletonDataset,
    build_motionbert_sequence,
)
from src.experiments.motionbert_p6b_config import load_motionbert_p6b_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml"


def _pose(frame: int) -> np.ndarray:
    pose = np.zeros((17, 3), dtype=np.float64)
    pose[:, 0] = np.linspace(-1.0, 1.0, 17) + frame * 0.01
    pose[:, 1] = np.linspace(0.0, 2.0, 17)
    pose[:, 2] = np.sin(np.linspace(0.0, np.pi, 17))
    pose[1] = [-0.3, 0.0, 0.0]
    pose[4] = [0.3, 0.0, 0.0]
    pose[11] = [-0.6, 1.2, 0.1]
    pose[14] = [0.6, 1.2, 0.1]
    return pose


def _fixture_rows(tmp_path: Path) -> pd.DataFrame:
    rows = []
    frame = 0
    for segment, length in ((0, 4), (1, 7), (2, 7)):
        for _ in range(length):
            path = tmp_path / f"pose_{frame}.json"
            path.write_text(
                json.dumps([{"keypoints": _pose(frame).tolist()}]),
                encoding="utf-8",
            )
            rows.append(
                {
                    "sample_id": "sample",
                    "frame_id": frame,
                    "retained_segment_index": segment,
                    "skeleton_json_path": path.name,
                    "candidate_index": 0,
                    "use_for_frame_training": True,
                }
            )
            frame += 1
        frame += 10
    return pd.DataFrame(rows)


def test_longest_segment_is_selected_without_cross_gap_interpolation(
    tmp_path: Path,
) -> None:
    rows = _fixture_rows(tmp_path)
    projection = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    result = build_motionbert_sequence(
        rows, data_root=tmp_path, projection=projection, frames=96
    )

    segment_one_frames = rows.loc[
        rows["retained_segment_index"].eq(1), "frame_id"
    ].to_numpy()
    assert result.selected_segment_index == 1
    assert result.source_frame_ids.min() >= segment_one_frames.min()
    assert result.source_frame_ids.max() <= segment_one_frames.max()
    assert result.sequence.shape == (96, 17, 3)
    assert torch.isfinite(result.sequence).all()
    assert torch.all(result.sequence[:, :, 2] == 1)
    assert result.selected_frame_count == 7
    assert result.total_retained_frame_count == 18


def test_real_population_and_projection_ownership_are_frozen() -> None:
    config = load_motionbert_p6b_config(CONFIG)
    train = MotionBERTSkeletonDataset(config, partition="train")
    validation = MotionBERTSkeletonDataset(config, partition="validation")

    assert len(train) == 2039
    assert len(validation) == 388
    assert train.supported_count == 1956
    assert validation.supported_count == 385
    assert train.projection_sha256 == validation.projection_sha256
    assert set(train.projection_fit_user_ids).isdisjoint({"user6", "user7"})
    assert set(train.labels.tolist()) == set(range(40))
    assert set(validation.labels.tolist()) == set(range(40))
