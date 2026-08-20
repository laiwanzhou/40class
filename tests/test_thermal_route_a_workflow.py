from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "validate_thermal_route_a_workflow.py"
WORKFLOW_PATH = ROOT / "configs" / "experiments" / "thermal_route_a_workflow.yaml"
RUNBOOK_PATH = (
    ROOT
    / "docs"
    / "superpowers"
    / "workflows"
    / "2026-08-20-thermal-route-a.md"
)
DEVELOPMENT_SPLIT_PATH = (
    ROOT / "metadata" / "splits" / "train12_val2_user6_user7_development.json"
)

EXPECTED_STAGE_ORDER = (
    "a0_input_audit",
    "a1_student_implementation",
    "a2_runtime_probe",
    "a3_direct_training",
    "a4_teacher_logits_handoff",
    "a5_kd_training",
    "a6_paired_decision",
)


def load_validator():
    assert SCRIPT_PATH.is_file(), f"missing workflow validator: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("thermal_route_a_workflow", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_workflow() -> dict:
    assert WORKFLOW_PATH.is_file(), f"missing workflow config: {WORKFLOW_PATH}"
    payload = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_committed_route_a_workflow_is_valid_and_waiting_for_a0_review() -> None:
    validator = load_validator()
    workflow = load_workflow()

    assert validator.validate_workflow(workflow) == []
    assert tuple(stage["id"] for stage in workflow["stages"]) == EXPECTED_STAGE_ORDER
    assert validator.next_action(workflow) == "a0_input_audit"
    assert workflow["status"] == "a0_pending_human_montage_approval"
    assert workflow["stages"][0]["status"] == "in_progress"
    assert all(stage["status"] == "pending" for stage in workflow["stages"][1:])
    assert not workflow["authorization"]["a_direct_training"]
    assert not workflow["authorization"]["a_kd_training"]
    assert RUNBOOK_PATH.is_file()


def test_workflow_carries_its_authoritative_development_membership() -> None:
    workflow = load_workflow()

    assert DEVELOPMENT_SPLIT_PATH.is_file()
    split = yaml.safe_load(DEVELOPMENT_SPLIT_PATH.read_text(encoding="utf-8"))
    assert workflow["evidence_boundary"]["development_split"] == str(
        DEVELOPMENT_SPLIT_PATH.relative_to(ROOT)
    ).replace("\\", "/")
    assert split["train_user_ids"] == [
        "user1",
        "user2",
        "user3",
        "user5",
        "user8",
        "user9",
        "user16",
        "user18",
        "user19",
        "user20",
        "user21",
        "user22",
    ]
    assert split["validation_user_ids"] == ["user6", "user7"]
    assert split["heldout_user_ids"] == ["user4", "user17", "user23", "user24"]


def test_route_a_variants_share_one_student_contract() -> None:
    validator = load_validator()
    workflow = load_workflow()

    contract_id = workflow["student_contract"]["id"]
    assert workflow["variants"]["a_direct"]["student_contract_id"] == contract_id
    assert workflow["variants"]["a_kd"]["student_contract_id"] == contract_id

    mismatched = copy.deepcopy(workflow)
    mismatched["variants"]["a_kd"]["student_contract_id"] = "different_student"
    assert "A-direct and A-KD must share one student contract" in validator.validate_workflow(
        mismatched
    )


def test_training_stage_cannot_start_without_explicit_authorization() -> None:
    validator = load_validator()
    workflow = load_workflow()
    unauthorized = copy.deepcopy(workflow)

    for stage in unauthorized["stages"][:3]:
        stage["status"] = "completed"
    unauthorized["stages"][3]["status"] = "in_progress"
    unauthorized["current_stage"] = "a3_direct_training"

    errors = validator.validate_workflow(unauthorized)
    assert "a3_direct_training requires explicit authorization" in errors


@pytest.mark.parametrize(
    "field",
    ("read_heldout4_labels", "read_competition_test", "read_quarantined_evidence"),
)
def test_forbidden_evidence_access_is_rejected(field: str) -> None:
    validator = load_validator()
    workflow = load_workflow()
    unsafe = copy.deepcopy(workflow)
    unsafe["evidence_boundary"][field] = True

    assert f"forbidden evidence flag must be false: {field}" in validator.validate_workflow(
        unsafe
    )


def test_kd_requires_one_passing_teacher_and_forbids_ensemble() -> None:
    validator = load_validator()
    workflow = load_workflow()
    invalid = copy.deepcopy(workflow)
    invalid["teacher_handoff"]["allow_teacher_ensemble"] = True

    errors = validator.validate_workflow(invalid)
    assert "teacher ensemble is forbidden for generation 2" in errors
    assert workflow["teacher_handoff"]["required_metrics"] == {
        "accuracy_min": 0.60,
        "macro_f1_min": 0.45,
        "worst_user_accuracy_min": 0.50,
    }
