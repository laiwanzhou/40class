from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW = (
    PROJECT_ROOT / "configs" / "experiments" / "thermal_route_a_workflow.yaml"
)

REQUIRED_STAGE_ORDER = (
    "a0_input_audit",
    "a1_student_implementation",
    "a2_runtime_probe",
    "a3_direct_training",
    "a4_teacher_logits_handoff",
    "a5_kd_training",
    "a6_paired_decision",
)
REQUIRED_DEPENDENCIES = {
    "a0_input_audit": (),
    "a1_student_implementation": ("a0_input_audit",),
    "a2_runtime_probe": ("a0_input_audit", "a1_student_implementation"),
    "a3_direct_training": ("a2_runtime_probe",),
    "a4_teacher_logits_handoff": ("a3_direct_training",),
    "a5_kd_training": ("a4_teacher_logits_handoff",),
    "a6_paired_decision": ("a3_direct_training", "a5_kd_training"),
}
TRAINING_AUTHORIZATION = {
    "a3_direct_training": "a_direct_training",
    "a5_kd_training": "a_kd_training",
}
ALLOWED_STAGE_STATUSES = {"pending", "in_progress", "completed", "blocked"}
FORBIDDEN_EVIDENCE_FLAGS = (
    "read_heldout4_labels",
    "read_competition_test",
    "read_quarantined_evidence",
)


def load_workflow(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("workflow root must be a mapping")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def validate_workflow(workflow: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    if workflow.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if workflow.get("workflow_id") != "thermal_route_a_generation2":
        errors.append("workflow_id must be thermal_route_a_generation2")

    stages_value = workflow.get("stages")
    stages = stages_value if isinstance(stages_value, list) else []
    stage_ids = tuple(
        stage.get("id") for stage in stages if isinstance(stage, Mapping)
    )
    if stage_ids != REQUIRED_STAGE_ORDER:
        errors.append("route A stages must use the frozen order")

    stage_by_id = {
        str(stage.get("id")): stage
        for stage in stages
        if isinstance(stage, Mapping) and stage.get("id") is not None
    }
    in_progress = 0
    for stage_id in REQUIRED_STAGE_ORDER:
        stage = _mapping(stage_by_id.get(stage_id))
        status = stage.get("status")
        if status not in ALLOWED_STAGE_STATUSES:
            errors.append(f"invalid stage status: {stage_id}")
        if status == "in_progress":
            in_progress += 1

        dependencies = tuple(stage.get("requires", ()))
        if dependencies != REQUIRED_DEPENDENCIES[stage_id]:
            errors.append(f"invalid dependencies: {stage_id}")
        if status in {"in_progress", "completed"}:
            for dependency in dependencies:
                if _mapping(stage_by_id.get(dependency)).get("status") != "completed":
                    errors.append(f"incomplete prerequisite for {stage_id}: {dependency}")

    if in_progress > 1:
        errors.append("at most one stage may be in_progress")

    first_incomplete = next(
        (
            stage_id
            for stage_id in REQUIRED_STAGE_ORDER
            if _mapping(stage_by_id.get(stage_id)).get("status") != "completed"
        ),
        "stop_human_review",
    )
    if workflow.get("current_stage") != first_incomplete:
        errors.append("current_stage must equal the first incomplete stage")

    authorization = _mapping(workflow.get("authorization"))
    for stage_id, authorization_key in TRAINING_AUTHORIZATION.items():
        status = _mapping(stage_by_id.get(stage_id)).get("status")
        if status in {"in_progress", "completed"} and not authorization.get(
            authorization_key, False
        ):
            errors.append(f"{stage_id} requires explicit authorization")
    if authorization.get("plan_approval_is_training_approval") is not False:
        errors.append("plan approval must not authorize training")

    evidence_boundary = _mapping(workflow.get("evidence_boundary"))
    for field in FORBIDDEN_EVIDENCE_FLAGS:
        if evidence_boundary.get(field) is not False:
            errors.append(f"forbidden evidence flag must be false: {field}")
    if evidence_boundary.get("allow_train14_oof") is not False:
        errors.append("train14 OOF must remain unauthorized")
    if evidence_boundary.get("modify_frozen_ir_x3d") is not False:
        errors.append("frozen IR/X3D must remain read-only")

    student_contract = _mapping(workflow.get("student_contract"))
    variants = _mapping(workflow.get("variants"))
    direct = _mapping(variants.get("a_direct"))
    kd = _mapping(variants.get("a_kd"))
    student_id = student_contract.get("id")
    if (
        not student_id
        or direct.get("student_contract_id") != student_id
        or kd.get("student_contract_id") != student_id
    ):
        errors.append("A-direct and A-KD must share one student contract")
    if student_contract.get("initialization") != "random":
        errors.append("route A student initialization must be random")
    if kd.get("only_training_difference") != "teacher_logit_loss":
        errors.append("A-KD may differ only by teacher logit loss")

    teacher_handoff = _mapping(workflow.get("teacher_handoff"))
    if teacher_handoff.get("allow_teacher_ensemble") is not False:
        errors.append("teacher ensemble is forbidden for generation 2")
    if teacher_handoff.get("required_metrics") != {
        "accuracy_min": 0.60,
        "macro_f1_min": 0.45,
        "worst_user_accuracy_min": 0.50,
    }:
        errors.append("teacher metric gates do not match the design spec")

    kd_status = _mapping(stage_by_id.get("a5_kd_training")).get("status")
    if kd_status in {"in_progress", "completed"}:
        if teacher_handoff.get("manifest_status") != "verified":
            errors.append("A-KD requires a verified teacher logits manifest")
        if teacher_handoff.get("passing_teacher_count") != 1:
            errors.append("A-KD requires exactly one passing teacher")

    external_gates = _mapping(workflow.get("external_gates"))
    direct_status = _mapping(stage_by_id.get("a3_direct_training")).get("status")
    route_b_verified = external_gates.get("route_b_report_verified", False) is True
    route_b_waived = external_gates.get("route_b_report_waived_by_user", False) is True
    if route_b_waived and not (
        external_gates.get("route_b_report_waiver_approved_on")
        and external_gates.get("route_b_report_waiver_approval_text")
    ):
        errors.append("Route B waiver requires auditable user approval")
    if direct_status in {"in_progress", "completed"} and not (
        route_b_verified or route_b_waived
    ):
        errors.append("A-direct requires Route B verification or explicit user waiver")

    deployment = _mapping(workflow.get("deployment"))
    if deployment.get("limit_bytes_exclusive") != 95_000_000:
        errors.append("deployment limit must be 95000000 bytes exclusive")
    if deployment.get("include_teacher_assets") is not False:
        errors.append("teacher assets must not enter deployment")

    return errors


def next_action(workflow: Mapping[str, Any]) -> str:
    errors = validate_workflow(workflow)
    if errors:
        raise ValueError("invalid workflow: " + "; ".join(errors))
    stages = workflow["stages"]
    for stage in stages:
        if stage["status"] != "completed":
            return str(stage["id"])
    return "stop_human_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and report the read-only Thermal Route A workflow state."
    )
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workflow = load_workflow(args.workflow.resolve())
    errors = validate_workflow(workflow)
    result = {
        "valid": not errors,
        "workflow_id": workflow.get("workflow_id"),
        "status": workflow.get("status"),
        "next_action": None if errors else next_action(workflow),
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
