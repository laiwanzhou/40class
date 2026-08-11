from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.x3d_clip_dataset import X3DClipDataset, partition_trial_windows
from scripts.audit_x3d_s_run import select_overfit_sample_ids


REAL_MANIFEST = Path(
    r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track"
    r"\roi640_depth_ordinal_256\combined_frame_manifest.csv"
)


def _assert_exact_window_partition(num_frames: int) -> None:
    windows = partition_trial_windows(num_frames)
    assert windows[0][0] == 0
    assert windows[-1][1] == num_frames
    assert all(0 <= start < end <= num_frames for start, end in windows)
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))
    coverage = np.concatenate([np.arange(start, end) for start, end in windows])
    np.testing.assert_array_equal(coverage, np.arange(num_frames))


def test_real_manifest_is_split_safe_ordered_and_adaptive() -> None:
    frame = pd.read_csv(REAL_MANIFEST, encoding="utf-8-sig")

    classes = frame[["class_id", "action_name"]].drop_duplicates()
    assert len(classes) == 40
    assert sorted(classes["class_id"].astype(int)) == list(range(40))
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "val"]
    assert set(train["user_id"]).isdisjoint(validation["user_id"])
    assert set(train["sample_id"]).isdisjoint(validation["sample_id"])
    assert not frame["ir_context_path"].astype(str).str.lower().str.contains(
        r"competition[-_ ]?test", regex=True
    ).any()

    lengths: dict[str, int] = {}
    for sample_id, trial in frame.groupby("sample_id", sort=False):
        indices = trial.sort_values("source_frame_index")["source_frame_index"].to_numpy()
        np.testing.assert_array_equal(indices, np.arange(len(trial)))
        lengths[str(sample_id)] = len(trial)
        _assert_exact_window_partition(len(trial))

    shortest_id = min(lengths, key=lengths.get)
    longest_id = max(lengths, key=lengths.get)
    assert lengths[shortest_id] == 1
    assert lengths[longest_id] == 236

    train_dataset = X3DClipDataset(frame, split="train", training=True)
    boundary_indices = [train_dataset.sample_ids.index(shortest_id), train_dataset.sample_ids.index(longest_id)]
    by_id = {
        train_dataset.sample_ids[index]: train_dataset[index]
        for index in boundary_indices
    }
    assert tuple(by_id[shortest_id]["clips"].shape) == (1, 1, 3, 13, 182, 182)
    assert tuple(by_id[longest_id]["clips"].shape) == (8, 1, 3, 13, 182, 182)

    deterministic = X3DClipDataset(frame, split="val", training=False)
    first = deterministic[0]
    second = deterministic[0]
    torch.testing.assert_close(first["clips"], second["clips"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(first["source_indices"], second["source_indices"])
    torch.testing.assert_close(first["window_bounds"], second["window_bounds"])


def test_overfit_selection_has_sixteen_trials_eight_classes_and_mixed_clip_counts() -> None:
    rows = []
    for class_id in range(8):
        for trial_index, frames in enumerate((8, 16, 33 if class_id == 0 else 24)):
            rows.append(
                {
                    "sample_id": f"c{class_id}-{trial_index}",
                    "class_id": class_id,
                    "frames": frames,
                }
            )
    trials = pd.DataFrame(rows)

    selected = select_overfit_sample_ids(trials)
    chosen = trials[trials["sample_id"].isin(selected)]

    assert len(selected) == 16
    assert chosen["class_id"].nunique() == 8
    assert (chosen["frames"] <= 32).any()
    assert (chosen["frames"] > 32).any()
