from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.run_x3d_s_fold0_dev import (
    CANONICAL_SEED,
    DEV_OUTPUT_ROOT,
    collect_canonical_artifact_hashes,
    select_fold0,
    validate_dev_contract,
)


def _assignment() -> dict[str, object]:
    users = {"u0", "u1", "u2"}
    return {
        "folds": [
            {
                "fold": 0,
                "train_user_ids": ["u1", "u2"],
                "validation_user_ids": ["u0"],
            },
            {
                "fold": 1,
                "train_user_ids": ["u0", "u2"],
                "validation_user_ids": ["u1"],
            },
            {
                "fold": 2,
                "train_user_ids": ["u0", "u1"],
                "validation_user_ids": ["u2"],
            },
        ],
        "users": sorted(users),
    }


def test_select_fold0_returns_only_frozen_development_fold() -> None:
    fold = select_fold0(_assignment(), allowed_users={"u0", "u1", "u2"})

    assert fold.fold == 0
    assert fold.train_user_ids == ("u1", "u2")
    assert fold.validation_user_ids == ("u0",)


@pytest.mark.parametrize("run_id", ["strict_v3_seed20260715", "phase4_retry", "phase5"])
def test_fold0_dev_rejects_canonical_run_id(run_id: str) -> None:
    with pytest.raises(ValueError, match="canonical"):
        validate_dev_contract(
            output_root=DEV_OUTPUT_ROOT,
            run_id=run_id,
            seed=CANONICAL_SEED,
        )


def test_fold0_dev_rejects_non_development_output_root() -> None:
    with pytest.raises(ValueError, match="development output root"):
        validate_dev_contract(
            output_root=Path("outputs/x3d_s_ir_context_oof"),
            run_id="fold0_dev_a1",
            seed=CANONICAL_SEED,
        )


def test_fold0_dev_is_frozen_to_canonical_seed() -> None:
    with pytest.raises(ValueError, match=str(CANONICAL_SEED)):
        validate_dev_contract(
            output_root=DEV_OUTPUT_ROOT,
            run_id="fold0_dev_a1",
            seed=20260716,
        )


def test_collect_canonical_hashes_is_read_only(tmp_path: Path) -> None:
    artifact = tmp_path / "formal_outer_refit.pt"
    artifact.write_bytes(b"frozen")
    before_bytes = artifact.read_bytes()
    before_mtime = artifact.stat().st_mtime_ns

    hashes = collect_canonical_artifact_hashes([artifact])

    assert hashes[str(artifact.resolve())] == hashlib.sha256(b"frozen").hexdigest()
    assert artifact.read_bytes() == before_bytes
    assert artifact.stat().st_mtime_ns == before_mtime


def test_collect_canonical_hashes_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_canonical_artifact_hashes([tmp_path / "missing.pt"])
