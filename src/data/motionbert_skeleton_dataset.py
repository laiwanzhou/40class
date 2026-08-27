from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.canonical_multimodal_index import CanonicalTrial, build_canonical_trials
from src.experiments.motionbert_p6b_config import project_path


@dataclass(frozen=True)
class MotionBERTSkeletonSample:
    sequence: torch.Tensor
    source_frame_ids: np.ndarray
    selected_segment_index: int
    selected_frame_count: int
    total_retained_frame_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.casefold().isin({"true", "1", "yes"})


def _read_candidate(path: Path, candidate_index: int) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not 0 <= candidate_index < len(payload):
        raise ValueError(f"invalid Skeleton candidate {candidate_index}: {path}")
    pose = np.asarray(payload[candidate_index].get("keypoints"), dtype=np.float64)
    if pose.shape != (17, 3) or not np.isfinite(pose).all():
        raise ValueError(f"invalid H36M-17 Skeleton pose: {path}")
    return pose


def _normalize_h36m_pose(pose: np.ndarray) -> np.ndarray:
    root = (pose[1] + pose[4]) * 0.5
    shoulders = (pose[11] + pose[14]) * 0.5
    scale = np.linalg.norm(shoulders - root) + 0.5 * np.linalg.norm(
        pose[11] - pose[14]
    )
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("invalid H36M root/shoulder scale")
    normalized = (pose - root[None]) / float(scale)
    if not np.isfinite(normalized).all():
        raise ValueError("non-finite normalized H36M pose")
    return normalized


def build_motionbert_sequence(
    clean_rows: pd.DataFrame,
    *,
    data_root: Path,
    projection: np.ndarray,
    frames: int = 96,
) -> MotionBERTSkeletonSample:
    required = {
        "sample_id",
        "frame_id",
        "retained_segment_index",
        "skeleton_json_path",
        "candidate_index",
        "use_for_frame_training",
    }
    if not required.issubset(clean_rows.columns):
        raise ValueError(
            f"MotionBERT clean rows miss {sorted(required - set(clean_rows.columns))}"
        )
    if np.asarray(projection).shape != (3, 2) or frames < 2:
        raise ValueError("MotionBERT projection or frame budget changed")
    rows = clean_rows[_truthy(clean_rows["use_for_frame_training"])].copy()
    rows = rows.dropna(subset=["candidate_index", "retained_segment_index"])
    if rows.empty or rows["sample_id"].astype(str).nunique() != 1:
        raise ValueError("MotionBERT sequence expects one retained trial")
    rows["retained_segment_index"] = rows["retained_segment_index"].astype(int)
    counts = rows.groupby("retained_segment_index", sort=True).size()
    maximum = int(counts.max())
    selected_segment = int(counts[counts.eq(maximum)].index.min())
    selected = rows[rows["retained_segment_index"].eq(selected_segment)].copy()
    selected = selected.sort_values("frame_id")
    source_frames = selected["frame_id"].to_numpy(dtype=np.float64)
    if len(np.unique(source_frames)) != len(source_frames):
        raise ValueError("MotionBERT source frame IDs are not unique")
    poses = np.stack(
        [
            _normalize_h36m_pose(
                _read_candidate(
                    data_root / str(row.skeleton_json_path),
                    int(row.candidate_index),
                )
            )
            for row in selected.itertuples()
        ]
    )
    projected = poses @ np.asarray(projection, dtype=np.float64)
    target_frames = np.linspace(
        source_frames[0], source_frames[-1], frames, dtype=np.float64
    )
    output = np.empty((frames, 17, 3), dtype=np.float32)
    for joint in range(17):
        output[:, joint, 0] = np.interp(
            target_frames, source_frames, projected[:, joint, 0]
        )
        output[:, joint, 1] = np.interp(
            target_frames, source_frames, projected[:, joint, 1]
        )
    output[:, :, 2] = 1.0
    if not np.isfinite(output).all():
        raise ValueError("non-finite MotionBERT input sequence")
    return MotionBERTSkeletonSample(
        sequence=torch.from_numpy(output),
        source_frame_ids=target_frames.astype(np.float32),
        selected_segment_index=selected_segment,
        selected_frame_count=len(selected),
        total_retained_frame_count=len(rows),
    )


class MotionBERTSkeletonDataset(Dataset[dict[str, object]]):
    def __init__(self, config: dict[str, Any], *, partition: str) -> None:
        if partition not in {"train", "validation"}:
            raise ValueError("partition must be train or validation")
        population = config["population"]
        data = config["data"]
        self.trials = build_canonical_trials(
            project_path(str(population["manifest"])),
            project_path(str(population["split"])),
            Path(str(data["root"])),
            partition=partition,
        )
        clean_path = project_path(str(data["clean_view"]))
        clean = pd.read_csv(
            clean_path,
            encoding="utf-8-sig",
            dtype={"sample_id": str, "user_id": str},
        )
        self.lookup = {
            str(sample_id): rows.reset_index(drop=True)
            for sample_id, rows in clean.groupby("sample_id", sort=False)
        }
        provenance_path = project_path(str(data["clean_view_provenance"]))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        self.projection = np.asarray(provenance["projection_matrix"], dtype=np.float64)
        if self.projection.shape != (3, 2):
            raise ValueError("MotionBERT projection shape changed")
        self.projection_sha256 = _sha256(provenance_path)
        self.projection_fit_user_ids = tuple(
            str(value) for value in provenance["projection_fit_user_ids"]
        )
        if set(self.projection_fit_user_ids) & {"user6", "user7"}:
            raise ValueError("validation users entered MotionBERT projection fit")
        self.data_root = Path(str(data["root"]))
        self.frames = int(config["input"]["frames"])
        self.sample_ids = np.asarray([trial.sample_id for trial in self.trials])
        self.user_ids = np.asarray([trial.user_id for trial in self.trials])
        self.labels = np.asarray([trial.class_id for trial in self.trials], dtype=np.int64)
        self.supported_count = sum(
            trial.sample_id in self.lookup for trial in self.trials
        )
        expected_supported = 1956 if partition == "train" else 385
        if self.supported_count != expected_supported:
            raise ValueError(
                f"MotionBERT supported {partition} population changed: {self.supported_count}"
            )

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, index: int) -> dict[str, object]:
        trial: CanonicalTrial = self.trials[index]
        rows = self.lookup.get(trial.sample_id)
        if rows is None:
            sequence = torch.zeros(self.frames, 17, 3)
            available = False
            selected_segment = -1
            selected_count = 0
            total_count = 0
            source_frames = np.zeros(self.frames, dtype=np.float32)
            failure_reason = "skeleton_unavailable"
        else:
            sample = build_motionbert_sequence(
                rows,
                data_root=self.data_root,
                projection=self.projection,
                frames=self.frames,
            )
            sequence = sample.sequence
            available = True
            selected_segment = sample.selected_segment_index
            selected_count = sample.selected_frame_count
            total_count = sample.total_retained_frame_count
            source_frames = sample.source_frame_ids
            failure_reason = ""
        discarded_fraction = (
            1.0 - selected_count / total_count if total_count else 0.0
        )
        return {
            "sequence": sequence,
            "available": torch.tensor(available),
            "quality": torch.tensor(
                [selected_count, total_count, discarded_fraction],
                dtype=torch.float32,
            ),
            "source_frame_ids": torch.from_numpy(source_frames.copy()),
            "selected_segment_index": selected_segment,
            "sample_id": trial.sample_id,
            "user_id": trial.user_id,
            "label": trial.class_id,
            "failure_reason": failure_reason,
        }
