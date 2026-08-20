from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/thermal_a1_student_implementation.json"


def test_a1_report_records_zero_training_and_locked_next_gate() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))

    assert payload["stage"] == "A1"
    assert payload["status"] == "completed"
    assert payload["zero_training"] is True
    assert payload["models"]["a_multistream"]["parameters"] < 10_000_000
    assert payload["models"]["a_multistream"]["state_dict_bytes"] < 45_000_000
    assert payload["models"]["a_multistream"]["fusion_dim"] == 780
    assert payload["trainer"]["formal_cli_execution"] == "locked_until_a2_passes"
    assert payload["trainer"]["training_authorized"] is False
    assert payload["pose_cache"]["status"] == "absent_required_before_formal_training"
    assert payload["next_action"] == "a2_runtime_probe"
    assert payload["evidence_boundary"]["heldout4_labels_read"] is False
    assert payload["evidence_boundary"]["ir_x3d_modified"] is False


def test_a1_report_source_hashes_match_committed_inputs() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))

    for relative_path, expected in payload["source_sha256"].items():
        digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == expected
