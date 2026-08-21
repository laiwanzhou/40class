from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/thermal_teacher_environment_probe.json"


def test_generated_c1_probe_passes_frozen_gates() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))

    assert payload["stage"] == "C1-qualification"
    assert payload["status"] == "passed"
    assert payload["zero_formal_training"] is True
    assert payload["throwaway_optimizer_steps"] == 1
    assert payload["input_contract"]["modality"] == "thermal_only"
    assert payload["input_contract"]["views"] == ["full", "thermal_yolo_context"]
    assert payload["input_contract"]["windows"] == 3
    assert payload["input_contract"]["frames_per_window"] == 16
    assert payload["pretrained_load"]["missing_keys"] == []
    assert payload["pretrained_load"]["unexpected_keys"] == []
    assert payload["pretrained_load"]["checkpoint_sha256"] == (
        "91a641e6c2ab531d1aca5f4321b4d802ec5c3babc15df855cdb6e39c6a1107c8"
    )
    smoke = payload["real_data_train_smoke"]
    assert smoke["physical_batch_trials"] == 1
    assert smoke["effective_batch_trials"] == 8
    assert smoke["finite_loss"] is True
    assert smoke["finite_logits"] is True
    assert smoke["finite_gradients"] is True
    assert smoke["optimizer_state_changed"] is True
    assert smoke["peak_allocated_mib"] < 7300
    assert payload["teacher_is_training_only"] is True
    assert payload["student_deployment_imports_teacher"] is False
