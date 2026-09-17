"""Replace P238 with P303 and audit the full group + P270 sequence path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p165_deployable_group_teacher import SPLITS, crossfit
from p173_vjepa_augmented_group_teacher import build_train_bank
from p255_repeat_augmented_physical_group import al
from p301_union_pretrained_group_sequence_audit import COHORTS, sequence_crossfit


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p304_union_full_physical_group_sequence_audit_v1"
SOURCES = (
    (HERE / "runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz", "probabilities", "p128_hierarchical_multimodal"),
    (HERE / "runs/p158_lavila_frame_token_transformer_single_seed_v1/oof_predictions.npz", "probability", "p158_lavila_frame_token"),
    (HERE / "runs/p303_union_full_physical_oof_v1/oof_predictions.npz", "probability", "p303_union_full_physical"),
    (HERE / "runs/p231_depth_thermal_ir_oof_v1/oof_predictions.npz", "ir_thermal_probability", "p231_ir_thermal"),
    (HERE / "runs/p253_repeat_physical_transformer_oof_v1/oof_predictions.npz", "probability", "p253_repeat_physical"),
)
P255 = HERE / "runs/p255_repeat_augmented_physical_group_v1/predictions.npz"
P270 = HERE / "runs/p270_fixed_emission065_transition045_v1/predictions.npz"


def main() -> None:
    train, names = build_train_bank()
    for path, key, name in SOURCES:
        artifact = np.load(path)
        for cohort in SPLITS:
            split = train[cohort]
            posterior = al(artifact[key], artifact["sample_ids"], split["ids"])
            split["bank"] = np.concatenate((split["bank"], posterior[:, None, :]), axis=1)
        names.append(name)

    group_report, group_prediction, group_probability, _ = crossfit(train)
    sequence_prediction, sequence_report = sequence_crossfit(
        train, group_prediction, group_probability
    )
    labels = np.concatenate([train[name]["labels"] for name in SPLITS])
    group = np.concatenate([group_prediction[name] for name in COHORTS])
    sequence = np.concatenate([sequence_prediction[name] for name in COHORTS])
    p255 = np.load(P255)
    p270 = np.load(P270)
    p255_prediction = np.concatenate([p255[f"{name}_held_prediction"] for name in COHORTS])
    p270_prediction = np.concatenate([p270[f"{name}_held_prediction"] for name in COHORTS])
    metrics = {
        "p255_correct": int(np.sum(p255_prediction == labels)),
        "group_correct": int(np.sum(group == labels)),
        "p270_correct": int(np.sum(p270_prediction == labels)),
        "sequence_correct": int(np.sum(sequence == labels)),
    }
    report = {
        "stage": "P304_union_full_physical_group_sequence_audit",
        "status": "complete",
        "protocol": {
            "change_vs_p255": "replace P238 posterior with strict P303 posterior",
            "recovered_extra_rows": 10,
            "group_outer_crossfit": True,
            "sequence_recipe_frozen_from_p270": True,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "experts": names,
        "metrics": {
            **metrics,
            "group_net_vs_p255": metrics["group_correct"] - metrics["p255_correct"],
            "sequence_net_vs_p270": metrics["sequence_correct"] - metrics["p270_correct"],
            "group_accuracy": float(np.mean(group == labels)),
            "sequence_accuracy": float(np.mean(sequence == labels)),
        },
        "group_cohorts": group_report,
        "sequence_cohorts": sequence_report,
        "decision": "train_matched_test_head" if metrics["sequence_correct"] > metrics["p270_correct"] else "reject",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "oof_predictions.npz",
        labels=labels,
        group_prediction=group,
        sequence_prediction=sequence,
        **{f"{name}_group_prediction": group_prediction[name] for name in COHORTS},
        **{f"{name}_group_probability": group_probability[name] for name in COHORTS},
        **{f"{name}_sequence_prediction": sequence_prediction[name] for name in COHORTS},
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
