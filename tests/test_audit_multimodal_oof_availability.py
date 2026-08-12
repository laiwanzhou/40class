from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_multimodal_oof_availability.py"
SPEC = importlib.util.spec_from_file_location("audit_multimodal_oof_availability", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_validate_oof_accepts_nested_user_partition() -> None:
    oof = {"folds": [
        {
            "fold": 0, "train_user_ids": ["u2"], "validation_user_ids": ["u1"],
            "epoch_selection": {"fit_user_ids": ["u2"], "validation_user_ids": []},
        },
        {
            "fold": 1, "train_user_ids": ["u1"], "validation_user_ids": ["u2"],
            "epoch_selection": {"fit_user_ids": ["u1"], "validation_user_ids": []},
        },
    ]}

    AUDIT.validate_oof(oof, {"u1", "u2"})


def test_validate_oof_rejects_reused_outer_validation_user() -> None:
    oof = {"folds": [
        {
            "fold": 0, "train_user_ids": ["u2"], "validation_user_ids": ["u1"],
            "epoch_selection": {"fit_user_ids": ["u2"], "validation_user_ids": []},
        },
        {
            "fold": 1, "train_user_ids": ["u2"], "validation_user_ids": ["u1"],
            "epoch_selection": {"fit_user_ids": ["u2"], "validation_user_ids": []},
        },
    ]}

    with pytest.raises(ValueError, match="partition"):
        AUDIT.validate_oof(oof, {"u1", "u2"})


def test_scope_row_distinguishes_canonical_and_modality_class_loss() -> None:
    manifest = pd.DataFrame({
        "user_id": ["u1", "u1"], "class_id": [0, 1], "sample_id": ["a", "b"],
    })
    row = AUDIT.scope_row(
        manifest, ["u1"], "IR", pd.Series([True, False]), 0, "outer_validation", "manifest_present"
    )

    assert row["canonical_class_count"] == 2
    assert row["class_count"] == 1
    assert row["modality_specific_missing_class_ids"] == "1"
