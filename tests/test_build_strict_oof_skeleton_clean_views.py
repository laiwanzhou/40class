from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_strict_oof_skeleton_clean_views.py"
SPEC = importlib.util.spec_from_file_location("build_strict_oof_skeleton_clean_views", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_build_view_rejects_projection_fit_validation_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        BUILDER.build_view(
            pd.DataFrame(), {"u1"}, {"u1"}, {"u1"}, {}, Path("."),
            projection=None, confidence_threshold=0.25, margin_threshold=0.20,
        )


def test_generated_provenance_has_disjoint_users() -> None:
    root = Path(__file__).resolve().parents[1] / "reports/skeleton_strict_oof_clean_views"
    for fold in range(3):
        for scope in ("inner_selection", "formal_outer"):
            provenance = __import__("json").loads(
                (root / f"fold_{fold}" / scope / "provenance.json").read_text(encoding="utf-8")
            )
            assert provenance["fit_validation_overlap"] == []
            assert provenance["margin_threshold"] == 0.20
