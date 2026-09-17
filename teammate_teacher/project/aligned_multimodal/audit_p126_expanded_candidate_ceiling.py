"""Machine-readable ceiling audit for the frozen P126 candidate inventory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p117_transductive_multicandidate_router import (
    load_candidate_splits,
    logits_to_probability,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUTPUT = HERE / "runs/p126_expanded_candidate_ceiling_v1"
CHAMPION = HERE / "runs/p150_repeat_branch_confidence_selector_v1/predictions.npz"


def asset(name: str, path: Path, key: str, probability: bool):
    source = np.load(path)
    return {
        "name": name,
        "source": source,
        "lookup": {
            value: index
            for index, value in enumerate(source["sample_ids"].astype(str))
        },
        "key": key,
        "probability": probability,
    }


def main() -> None:
    bank = load_candidate_splits(
        full_visual_bank=True,
        structured_bank=True,
        legacy_visual_bank=True,
        vjepa_dense_bank=True,
    )
    champion = np.load(CHAMPION)
    p12 = HERE / "runs/p12_complete_oof/complete_oof.npz"
    p122 = HERE / "runs/p122_hand_object_relation_teacher_v1/oof_predictions.npz"
    assets = [
        asset("thermal_candidate", p12, "thermal_candidate_logits", False),
        asset("thermal", p12, "thermal_logits", False),
        asset("p12_imu", p12, "imu_logits", False),
        asset(
            "p90_deep_imu",
            PROJECT
            / "runs/p90_imu_teacher_blend_v1/imu_p90_sensorwise_plus_deep_crossfit_oof.npz",
            "probabilities",
            True,
        ),
        asset(
            "motionbert_front",
            PROJECT
            / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front_linear_oof.npz",
            "probabilities",
            True,
        ),
        asset(
            "motionbert_3view",
            PROJECT
            / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front-side-top_linear_oof.npz",
            "probabilities",
            True,
        ),
        asset(
            "skeleton_invariant",
            HERE / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz",
            "skeleton_logits",
            False,
        ),
        asset("p122_pose", p122, "pose_only_probability", True),
        asset("p122_object", p122, "object_only_probability", True),
        asset("p122_relation", p122, "relations_only_probability", True),
        asset("p122_all", p122, "all_probability", True),
        asset(
            "local_depth",
            HERE / "runs/p16_local_depth_oracle_oof/oof_logits.npz",
            "logits",
            False,
        ),
        asset(
            "p128_hierarchical",
            HERE / "runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz",
            "probabilities",
            True,
        ),
        asset(
            "p130_epic_slowfast",
            HERE / "runs/p130_epic_slowfast_teacher_v1/oof_predictions.npz",
            "all_views_epic_logits_probability",
            True,
        ),
        asset(
            "p131_egovlp",
            HERE / "runs/p131_egovlp_teacher_v1/oof_predictions.npz",
            "all_raw_projected_probability",
            True,
        ),
        asset(
            "p142_token_transformer",
            HERE / "runs/p142_vjepa_token_transformer_three_seed_v2/oof_predictions.npz",
            "probability",
            True,
        ),
        asset(
            "p144_hand_token_transformer",
            HERE / "runs/p144_vjepa_hand_interaction_transformer_three_seed_v1/oof_predictions.npz",
            "probability",
            True,
        ),
        asset(
            "p146_workspace_token_transformer",
            HERE / "runs/p146_vjepa_workspace_transformer_three_seed_v1/oof_predictions.npz",
            "probability",
            True,
        ),
        asset(
            "p147_multimodal_token_transformer",
            HERE / "runs/p147_frozen_multimodal_token_single_seed_v1/oof_predictions.npz",
            "probability",
            True,
        ),
        asset(
            "p149_repeat_consistency_transformer",
            HERE / "runs/p149_vjepa_repeat_consistency_three_seed_v2/oof_predictions.npz",
            "probability",
            True,
        ),
        asset(
            "p151_repeat_embedding_transformer",
            HERE / "runs/p151_vjepa_repeat_embedding_single_seed_v1/oof_predictions.npz",
            "probability",
            True,
        ),
        asset(
            "p153_oof_distillation_transformer",
            HERE / "runs/p153_vjepa_oof_distillation_single_seed_v1/oof_predictions.npz",
            "probability",
            True,
        ),
        asset(
            "p155_lavila_workspace",
            HERE / "runs/p155_lavila_timesformer_teacher_v1/oof_predictions.npz",
            "workspace_raw_probability",
            True,
        ),
        asset(
            "p155_lavila_all_raw",
            HERE / "runs/p155_lavila_timesformer_teacher_v1/oof_predictions.npz",
            "all_raw_probability",
            True,
        ),
        asset(
            "p156_lavila_ek100_all_raw",
            HERE / "runs/p156_lavila_ek100_teacher_v1/oof_predictions.npz",
            "all_raw_probability",
            True,
        ),
        asset(
            "p158_lavila_frame_token_transformer",
            HERE / "runs/p158_lavila_frame_token_transformer_single_seed_v1/oof_predictions.npz",
            "probability",
            True,
        ),
        asset(
            "p160_candidate_set_transformer",
            HERE / "runs/p160_candidate_set_transformer_single_seed_v1/predictions.npz",
            "probability",
            True,
        ),
    ]
    counts = {
        value["name"]: {"safe_rescue": 0, "unique_over_core_bank": 0, "correct": 0}
        for value in assets
    }
    safe_correct = champion_correct = core_oracle = expanded_oracle = 0
    champion_errors_recoverable = 0
    rows = 0
    for split_name, value in bank.items():
        labels = value.split.labels
        safe = value.split.safe_prediction
        champion_prediction = champion[f"{split_name}_prediction"]
        core = safe == labels
        for probability in value.candidates.values():
            core |= probability.argmax(axis=1) == labels
        expanded = core.copy()
        any_extra_correct = np.zeros(len(labels), dtype=bool)
        for item in assets:
            positions = np.asarray(
                [item["lookup"][sample_id] for sample_id in value.split.sample_ids.astype(str)],
                dtype=np.int64,
            )
            raw = np.asarray(item["source"][item["key"]][positions], dtype=np.float64)
            probability = raw if item["probability"] else logits_to_probability(raw)
            prediction = probability.argmax(axis=1)
            correct = prediction == labels
            counts[item["name"]]["safe_rescue"] += int(
                np.sum(correct & (safe != labels))
            )
            counts[item["name"]]["unique_over_core_bank"] += int(
                np.sum(correct & ~core)
            )
            counts[item["name"]]["correct"] += int(correct.sum())
            expanded |= correct
            any_extra_correct |= correct
        safe_correct += int(np.sum(safe == labels))
        champion_correct += int(np.sum(champion_prediction == labels))
        core_oracle += int(core.sum())
        expanded_oracle += int(expanded.sum())
        champion_errors_recoverable += int(
            np.sum((champion_prediction != labels) & expanded)
        )
        rows += len(labels)
    report = {
        "stage": "P126_expanded_candidate_ceiling_v1",
        "status": "complete_analysis_only",
        "protocol": {
            "all_candidates_are_frozen_subject_safe_OOF": True,
            "oracle_used_for_deployment": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "rows": rows,
        "p89_safe": {"correct": safe_correct, "accuracy": safe_correct / rows},
        "current_champion": {
            "correct": champion_correct,
            "accuracy": champion_correct / rows,
            "errors": rows - champion_correct,
        },
        "core_bank_oracle": {
            "correct": core_oracle,
            "accuracy": core_oracle / rows,
        },
        "expanded_bank_oracle": {
            "correct": expanded_oracle,
            "accuracy": expanded_oracle / rows,
            "headroom_over_0.91_correct": expanded_oracle - int(np.ceil(0.91 * rows)),
        },
        "current_champion_errors_recoverable_by_expanded_bank": champion_errors_recoverable,
        "additional_candidates": counts,
        "decision": (
            "Candidate coverage is sufficient for 91%, but no source-safe router has "
            "converted enough of the oracle without cross-subject harm."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
