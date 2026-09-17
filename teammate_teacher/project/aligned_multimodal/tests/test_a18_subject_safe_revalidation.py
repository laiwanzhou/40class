from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ALIGNED = HERE.parent
if str(ALIGNED) not in sys.path:
    sys.path.insert(0, str(ALIGNED))

from a18_full_teacher_data import load_a18_data  # noqa: E402
from revalidate_a18_subject_safe import (  # noqa: E402
    checkpoint_selection_key,
    protocol_fold_ids,
)


def test_revalidation_reuses_complete_p90_subject_folds() -> None:
    data = load_a18_data()
    folds = protocol_fold_ids(data.sample_ids, data.users)
    assert sorted(np.unique(folds).tolist()) == [0, 1, 2]
    assert len(folds) == 2914
    for fold in range(3):
        held = folds == fold
        train = ~held
        assert len(set(data.users[held])) == 6
        assert len(set(data.users[train])) == 12
        assert not (set(data.users[held]) & set(data.users[train]))


def test_checkpoint_key_has_no_train_fit_input() -> None:
    metrics = {"top1": 0.8, "macro_f1": 0.7, "nll": 0.9}
    key = checkpoint_selection_key(metrics, mean_subject_top1=0.75, epoch=4)
    assert key == (0.75, 0.8, 0.7, -0.9, -4)


def test_final_epoch_is_control_and_session_recipe_is_frozen() -> None:
    config = json.loads(
        (ALIGNED / "configs/a18_subject_safe_revalidation.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["training"]["epochs"] == 24
    assert config["checkpoint_selection"]["eligible_epoch_max"] == 23
    assert config["checkpoint_selection"]["final_epoch_is_control_only"] == 24
    assert config["checkpoint_selection"]["train_fit_used"] is False
    assert config["session"]["gap_seconds"] == 30.0
    assert config["session"]["transition_weight"] == 0.3
    assert config["session"]["trigram_backoff"] == 1.0
    assert config["session"]["beam_width"] == 50
