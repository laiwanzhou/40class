from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p100a_global_teacher_data import (  # noqa: E402
    CANONICAL_VARIANTS as P100_VARIANTS,
    FOLD_USERS,
    load_p100a_data,
)
from p100a_global_teacher_model import (  # noqa: E402
    P100AGlobalTeacher,
    P100AModelConfig,
)
from p101_f1_coarse_anchor_model import (  # noqa: E402
    P101F1CoarseAnchoredTeacher,
    P101F1Config,
)
from train_p101_f2_source_safe_adapter_oof import (  # noqa: E402
    allocate_group_batch_counts,
    load_non_anchor_state,
    make_epoch_loader,
    non_anchor_state,
    source_safe_inner_folds,
)


def small_anchor() -> P100AGlobalTeacher:
    return P100AGlobalTeacher(
        P100AModelConfig(
            modalities=P100_VARIANTS["VS"],
            model_dim=64,
            heads=4,
            modality_layers=1,
            fusion_layers=1,
            fusion_latents=4,
            dropout=0.0,
            evidence_dropout=0.0,
        )
    )


def small_model(anchor: P100AGlobalTeacher) -> P101F1CoarseAnchoredTeacher:
    return P101F1CoarseAnchoredTeacher(
        anchor,
        P101F1Config(
            model_dim=64,
            motion_dim=32,
            heads=4,
            dropout=0.0,
            evidence_layers=1,
        ),
    )


def test_source_safe_groups_partition_every_outer_train_row() -> None:
    data = load_p100a_data()
    for outer_fold in range(4):
        train, held = data.indices_for_fold(outer_fold)
        groups = []
        for inner_fold in source_safe_inner_folds(outer_fold):
            rows = train[np.isin(data.users[train], np.asarray(FOLD_USERS[inner_fold]))]
            assert set(data.users[rows].tolist()) == set(FOLD_USERS[inner_fold])
            groups.append(rows)
        merged = np.concatenate(groups)
        np.testing.assert_array_equal(np.sort(merged), np.sort(train))
        assert len(np.unique(merged)) == len(train)
        assert not np.intersect1d(merged, held).size


def test_group_batch_allocation_matches_original_optimizer_budget() -> None:
    sizes = [483, 513, 485]
    counts = allocate_group_batch_counts(sizes, 16)
    assert sum(counts) == math.ceil(sum(sizes) / 16)
    for size, count in zip(sizes, counts):
        assert abs(size / count - 16) < 1


def test_epoch_loader_has_fixed_steps_and_complete_coverage() -> None:
    dataset = TensorDataset(torch.arange(101))
    loader = make_epoch_loader(dataset, 7, 123)
    batches = [batch[0].numpy() for batch in loader]
    assert len(batches) == 7
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(101))


def test_adapter_checkpoint_excludes_and_preserves_outer_anchor() -> None:
    source = small_model(small_anchor())
    with torch.no_grad():
        source.residual_projection.network[4].weight.normal_(std=0.1)
    state = non_anchor_state(source)
    assert state and not any(name.startswith("anchor.") for name in state)
    target = small_model(small_anchor())
    before = {
        name: tensor.clone() for name, tensor in target.anchor.state_dict().items()
    }
    audit = load_non_anchor_state(target, state)
    for name, tensor in target.anchor.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert audit["adapter_tensors"] == len(state)
    forty_class = [
        module
        for module in target.modules()
        if isinstance(module, torch.nn.Linear) and module.out_features == 40
    ]
    assert len(forty_class) == 1
