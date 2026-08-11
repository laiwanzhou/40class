from __future__ import annotations

import pytest

from scripts.run_x3d_s_fold2_only import CANONICAL_SEED, FOLD_INDEX, build_arg_parser, select_fold2


def frozen_assignment() -> dict[str, object]:
    return {
        "folds": [
            {
                "fold": 0,
                "train_user_ids": ["u1", "u2"],
                "validation_user_ids": ["u3"],
            },
            {
                "fold": 1,
                "train_user_ids": ["u1", "u3"],
                "validation_user_ids": ["u2"],
            },
            {
                "fold": 2,
                "train_user_ids": ["u2", "u3"],
                "validation_user_ids": ["u1"],
            },
        ]
    }


def test_fold2_only_selector_returns_frozen_fold_two() -> None:
    selected = select_fold2(frozen_assignment(), allowed_users={"u1", "u2", "u3"})
    assert selected.fold == FOLD_INDEX == 2
    assert selected.train_user_ids == ("u2", "u3")
    assert selected.validation_user_ids == ("u1",)


def test_fold2_only_selector_still_validates_complete_assignment() -> None:
    assignment = frozen_assignment()
    assignment["folds"] = assignment["folds"][:2]
    with pytest.raises(ValueError, match="exactly three folds"):
        select_fold2(assignment, allowed_users={"u1", "u2", "u3"})


def test_fold2_only_cli_defaults_to_canonical_seed() -> None:
    args = build_arg_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--oof-fold-assignment",
            "assignment.json",
            "--run-id",
            "continuation",
            "--source-run-id",
            "source",
        ]
    )
    assert args.seed == CANONICAL_SEED == 20260715
