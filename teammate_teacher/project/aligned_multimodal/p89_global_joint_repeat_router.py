from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_quad_consensus_submission as quad
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_global_joint_router_v1"


def align_values(
    source_ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray
) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    return np.asarray(values)[
        np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    ]


def evaluate(
    protocol_value,
    probability: np.ndarray,
    router_prediction: np.ndarray,
    grouping_config: GlobalRepeatConfig,
    evidence_weight: float,
    transition_scale: float,
) -> tuple[dict[str, object], np.ndarray]:
    prediction, grouping = joint_decode(
        probability,
        protocol_value[3],
        protocol_value,
        grouping_config,
        evidence_weight,
        transition_scale,
        initial_prediction=router_prediction,
        grouping_prediction=router_prediction,
    )
    return (
        {
            "configuration": {
                "evidence_weight": evidence_weight,
                "transition_scale": transition_scale,
            },
            "router_base": classification_metrics(
                protocol_value[1], router_prediction
            ),
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_p87": rescue_harm(
                protocol_value[1], protocol_value[3], prediction
            ),
            "rescue_harm_vs_router": rescue_harm(
                protocol_value[1], router_prediction, prediction
            ),
            "grouping": grouping,
        },
        prediction,
    )


def main() -> None:
    global_source = json.loads(
        (PROJECT_DIR / "runs/p89_global_repeat_h1_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    grouping_config = GlobalRepeatConfig(**global_source["selected_config"])
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    routed = np.load(
        PROJECT_DIR
        / "runs/p89_supervised_router_h1_to_h2_v3/confirmation_predictions.npz"
    )
    h1_probability = align_values(
        routed["selection_sample_ids"],
        routed["selection_blended_probability"],
        h1[0],
    )
    h1_router = align_values(
        routed["selection_sample_ids"],
        routed["selection_routed_prediction"],
        h1[0],
    )
    h2_probability = align_values(
        routed["sample_ids"], routed["blended_probability"], h2[0]
    )
    h2_router = align_values(
        routed["sample_ids"], routed["routed_prediction"], h2[0]
    )
    candidates = []
    predictions = []
    for evidence_weight in (0.10, 0.25, 0.50, 1.0, 2.0, 4.0):
        for transition_scale in (0.0, 0.5, 1.0, 1.5, 2.0):
            item, prediction = evaluate(
                h1,
                h1_probability,
                h1_router,
                grouping_config,
                evidence_weight,
                transition_scale,
            )
            candidates.append(item)
            predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["metrics"]["correct"],
            candidates[index]["metrics"]["balanced_accuracy"],
            candidates[index]["rescue_harm_vs_p87"]["net"],
            -candidates[index]["rescue_harm_vs_p87"]["harm"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    h1_prediction = predictions[selected_index]
    configuration = selected["configuration"]
    confirmation, h2_prediction = evaluate(
        h2,
        h2_probability,
        h2_router,
        grouping_config,
        float(configuration["evidence_weight"]),
        float(configuration["transition_scale"]),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=h1_prediction,
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_supervised_router_plus_global_joint_repeat_v1",
        "protocol": (
            "Apply the shared repeated-take latent-path decoder to supervised "
            "router probabilities. Select two joint-decoder scalars on H1 and "
            "transfer unchanged to H2."
        ),
        "grouping_configuration": global_source["selected_config"],
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "all_H1_candidates": [candidates[index] for index in order],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "all_H1_candidates"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
