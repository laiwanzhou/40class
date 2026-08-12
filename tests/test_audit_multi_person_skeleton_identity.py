from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_multi_person_skeleton_identity.py"
SPEC = importlib.util.spec_from_file_location("audit_multi_person_skeleton_identity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_frame_key_matches_ir_and_skeleton_names() -> None:
    expected = "2025-05-31_17-45-41.716_00000048"

    assert AUDIT.frame_key(Path(f"IR_{expected}.png")) == expected
    assert AUDIT.frame_key(Path(f"Color_{expected}.json")) == expected


def test_candidate_rmse_prefers_matching_projection() -> None:
    observed = np.zeros((17, 2), dtype=np.float64)
    observed_mask = np.zeros(17, dtype=bool)
    observed_mask[AUDIT.COMMON_YOLO_INDICES] = True
    candidate = np.zeros((17, 3), dtype=np.float64)
    projection = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    matching_rmse, joints, _ = AUDIT.candidate_rmse(observed, observed_mask, candidate, projection)
    candidate[AUDIT.COMMON_YOLO_INDICES, 0] = 1.0
    mismatching_rmse, _, _ = AUDIT.candidate_rmse(observed, observed_mask, candidate, projection)

    assert joints == len(AUDIT.COMMON_YOLO_INDICES)
    assert matching_rmse == 0.0
    assert mismatching_rmse == 1.0
