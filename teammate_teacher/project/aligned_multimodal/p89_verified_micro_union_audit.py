from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import align_metadata, classification_metrics
from p88_train_depth_residual import rescue_harm
from p89_count_regularized_imu_submission import transport
from p89_supported_template_gate import h3_protocol, load_grouping, load_imu


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_verified_micro_union_audit_v1"


def align_prediction(
    target_ids: np.ndarray, source_ids: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids.astype(str))}
    if len(lookup) != len(source_ids):
        raise RuntimeError("duplicate source sample id")
    missing = [sample_id for sample_id in target_ids.astype(str) if sample_id not in lookup]
    if missing:
        raise RuntimeError(f"missing {len(missing)} aligned predictions")
    return prediction[
        np.asarray([lookup[sample_id] for sample_id in target_ids.astype(str)], dtype=np.int64)
    ]


def label_free_union(
    safe: np.ndarray, count_prediction: np.ndarray, template_prediction: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    count_changed = count_prediction != safe
    template_changed = template_prediction != safe
    conflict = count_changed & template_changed & (count_prediction != template_prediction)
    prediction = safe.copy()
    prediction[count_changed & ~conflict] = count_prediction[count_changed & ~conflict]
    prediction[template_changed & ~conflict] = template_prediction[template_changed & ~conflict]
    return prediction, {
        "count_changes": int(count_changed.sum()),
        "template_changes": int(template_changed.sum()),
        "overlap_changes": int((count_changed & template_changed).sum()),
        "agreeing_overlap": int(
            (count_changed & template_changed & (count_prediction == template_prediction)).sum()
        ),
        "conflicting_overlap_reverted_to_safe": int(conflict.sum()),
        "union_changes": int((prediction != safe).sum()),
    }


def per_user_gain(
    labels: np.ndarray,
    safe: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
) -> dict[str, dict[str, int]]:
    output = {}
    for user in sorted(set(users.astype(str).tolist())):
        rows = users.astype(str) == user
        output[user] = {
            "safe_correct": int((safe[rows] == labels[rows]).sum()),
            "candidate_correct": int((prediction[rows] == labels[rows]).sum()),
            "gain": int(
                (prediction[rows] == labels[rows]).sum()
                - (safe[rows] == labels[rows]).sum()
            ),
            "changes": int((prediction[rows] != safe[rows]).sum()),
        }
    return output


def evaluate(
    labels: np.ndarray,
    safe: np.ndarray,
    count_prediction: np.ndarray,
    template_prediction: np.ndarray,
    users: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    prediction, overlap = label_free_union(safe, count_prediction, template_prediction)
    users_audit = per_user_gain(labels, safe, prediction, users)
    return prediction, {
        "safe_metrics": classification_metrics(labels, safe),
        "count_metrics": classification_metrics(labels, count_prediction),
        "count_rescue_harm": rescue_harm(labels, safe, count_prediction),
        "template_metrics": classification_metrics(labels, template_prediction),
        "template_rescue_harm": rescue_harm(labels, safe, template_prediction),
        "union_metrics": classification_metrics(labels, prediction),
        "union_rescue_harm": rescue_harm(labels, safe, prediction),
        "union_overlap": overlap,
        "union_per_user": users_audit,
        "minimum_user_gain": min(value["gain"] for value in users_audit.values()),
    }


def main() -> None:
    count_source = np.load(OUTPUT.parent / "p89_count_preserving_imu_v2/predictions.npz")
    template_source = np.load(
        OUTPUT.parent / "p89_supported_template_gate_v1/validation_predictions.npz"
    )

    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    report: dict[str, object] = {
        "stage": "P89_verified_micro_union_audit_v1",
        "protocol": (
            "A label-free union of the two mechanisms that independently improved both H1 "
            "and H2: cohort count-preserving transport and the frozen supported-template gate. "
            "If both mechanisms modify the same row to different labels, revert that row to the "
            "safe prediction. No Test prediction or leaderboard feedback is used."
        ),
    }
    saved: dict[str, np.ndarray] = {}
    for name, protocol_value in (("H1", h1), ("H2", h2)):
        sample_ids, labels, _, _, metadata = protocol_value[:5]
        prefix = name.lower()
        safe = align_prediction(
            sample_ids,
            template_source[f"{prefix}_sample_ids"],
            template_source[f"{prefix}_safe"],
        )
        count_prediction = align_prediction(
            sample_ids,
            count_source[f"{prefix}_sample_ids"],
            count_source[f"{prefix}_prediction"],
        )
        template_prediction = align_prediction(
            sample_ids,
            template_source[f"{prefix}_sample_ids"],
            template_source[f"{prefix}_prediction"],
        )
        prediction, audit = evaluate(
            labels, safe, count_prediction, template_prediction, metadata.users
        )
        report[name] = audit
        saved[f"{prefix}_sample_ids"] = sample_ids
        saved[f"{prefix}_safe"] = safe
        saved[f"{prefix}_count"] = count_prediction
        saved[f"{prefix}_template"] = template_prediction
        saved[f"{prefix}_union"] = prediction

    imu_ids, imu_logits = load_imu()
    grouping = load_grouping()
    h3_values = h3_protocol(imu_ids, imu_logits, grouping)
    h3_protocol_value, h3_adjusted, h3_safe, all_ids, all_labels, all_metadata, fit = h3_values
    h3_ids, h3_labels, _, _, h3_metadata = h3_protocol_value[:5]
    fit_users = sorted(set(all_metadata.users[fit].astype(str).tolist()))
    observed_counts = np.stack(
        [
            np.bincount(
                all_labels[fit & (all_metadata.users.astype(str) == user)], minlength=40
            )
            for user in fit_users
        ]
    )
    h3_count, h3_transport = transport(
        h3_adjusted,
        h3_safe,
        h3_metadata.users.astype(str),
        observed_counts,
    )
    h3_template = align_prediction(
        h3_ids,
        template_source["h3_sample_ids"],
        template_source["h3_prediction"],
    )
    h3_prediction, h3_audit = evaluate(
        h3_labels, h3_safe, h3_count, h3_template, h3_metadata.users
    )
    h3_audit["count_transport"] = h3_transport
    report["H3"] = h3_audit
    saved.update(
        {
            "h3_sample_ids": h3_ids,
            "h3_safe": h3_safe,
            "h3_count": h3_count,
            "h3_template": h3_template,
            "h3_union": h3_prediction,
        }
    )

    split_gains = {
        split: int(report[split]["union_rescue_harm"]["net"])
        for split in ("H1", "H2", "H3")
    }
    report["decision"] = {
        "split_net_gains": split_gains,
        "all_splits_nonnegative": all(value >= 0 for value in split_gains.values()),
        "all_users_nonnegative": all(
            int(report[split]["minimum_user_gain"]) >= 0
            for split in ("H1", "H2", "H3")
        ),
        "test_artifact_generated": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "validation_predictions.npz", **saved)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
