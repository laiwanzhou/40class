from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_x3d_s_experiment import (
    create_frozen_oof_assignment,
    generate_train14_oof_assignment,
)


def _trial_frame() -> pd.DataFrame:
    rows = []
    for user_index in range(14):
        for class_id in range(40):
            rows.append(
                {
                    "sample_id": f"sample-u{user_index}-c{class_id}",
                    "user_id": f"user{user_index}",
                    "class_id": class_id,
                    "action_name": f"class-{class_id}",
                }
            )
    return pd.DataFrame(rows)


def test_phase4_assignment_is_user_disjoint_complete_and_deterministic() -> None:
    frame = _trial_frame()
    allowed_users = set(frame["user_id"].astype(str))

    first = generate_train14_oof_assignment(
        frame,
        allowed_users=allowed_users,
        random_state=20260715,
    )
    second = generate_train14_oof_assignment(
        frame,
        allowed_users=allowed_users,
        random_state=20260715,
    )

    assert first == second
    validation_users = []
    for fold in first["folds"]:
        train = set(fold["train_user_ids"])
        validation = set(fold["validation_user_ids"])
        assert train.isdisjoint(validation)
        assert train | validation == allowed_users
        assert fold["train_class_count"] == 40
        assert set(fold["epoch_selection"]["fit_user_ids"]).isdisjoint(
            fold["epoch_selection"]["validation_user_ids"]
        )
        assert set(fold["epoch_selection"]["fit_user_ids"]) | set(
            fold["epoch_selection"]["validation_user_ids"]
        ) == train
        assert fold["epoch_selection"]["fit_class_count"] == 40
        validation_users.extend(fold["validation_user_ids"])
    assert sorted(validation_users) == sorted(allowed_users)
    assert first["combined_validation_class_count"] == 40


def test_frozen_assignment_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "train14_oof_3fold.json"
    assignment = generate_train14_oof_assignment(
        _trial_frame(),
        allowed_users={f"user{index}" for index in range(14)},
        random_state=20260715,
    )

    digest = create_frozen_oof_assignment(path, assignment)

    assert len(digest) == 64
    assert json.loads(path.read_text(encoding="utf-8")) == assignment
    with pytest.raises(FileExistsError, match="already exists"):
        create_frozen_oof_assignment(path, assignment)
