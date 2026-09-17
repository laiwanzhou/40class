"""Leakage-safe blend of the strongest legacy IMU expert and P90 deep teachers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p90_teacher_common import (
    HERE,
    NUM_CLASSES,
    REPO_ROOT,
    classification_metrics,
    load_protocol,
    save_oof_artifact,
)


LEGACY = HERE / "runs" / "p89_imu_sensor_attention_expert_v1" / "oof_probabilities.npz"
DEEP_RUN = REPO_ROOT / "runs" / "p90_imu_ssl_teacher_v1"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "p90_imu_teacher_blend_v1"


def align_legacy() -> tuple[np.ndarray, np.ndarray]:
    protocol = load_protocol()
    data = np.load(LEGACY, allow_pickle=False)
    sample_ids = data["sample_ids"].astype(str)
    lookup = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    probability = np.full((len(protocol.labels), NUM_CLASSES), 1.0 / NUM_CLASSES)
    available = np.asarray([sample_id in lookup for sample_id in protocol.sample_ids])
    rows = [lookup[sample_id] for sample_id in protocol.sample_ids[available]]
    probability[available] = data["class_log_t1"][rows]
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-deep-weight", type=float, default=0.30)
    parser.add_argument("--weight-step", type=float, default=0.025)
    args = parser.parse_args()
    protocol = load_protocol()
    legacy, legacy_available = align_legacy()
    deep_probabilities = []
    for name in ("imu_hart128_scratch", "imu_hart128_maskedssl"):
        data = np.load(DEEP_RUN / f"{name}_oof.npz", allow_pickle=False)
        if data["sample_ids"].astype(str).tolist() != protocol.sample_ids.tolist():
            raise ValueError(f"{name} is not P90 aligned")
        deep_probabilities.append(np.asarray(data["probabilities"], dtype=np.float64))
    deep = np.mean(deep_probabilities, axis=0)
    grid = np.arange(0.0, args.max_deep_weight + args.weight_step * 0.5, args.weight_step)
    log_legacy = np.log(np.maximum(legacy, 1e-8))
    log_deep = np.log(np.maximum(deep, 1e-8))
    logits = np.zeros_like(log_legacy)
    fold_selection = []
    for fold in range(3):
        selection = (protocol.fold_id != fold) & legacy_available
        validation = protocol.fold_id == fold
        candidates = []
        for weight in grid:
            score = (1.0 - weight) * log_legacy + weight * log_deep
            accuracy = float(
                np.mean(score[selection].argmax(axis=1) == protocol.labels[selection])
            )
            candidates.append((accuracy, -float(weight), float(weight)))
        _, _, selected_weight = max(candidates)
        logits[validation] = (
            (1.0 - selected_weight) * log_legacy[validation]
            + selected_weight * log_deep[validation]
        )
        fold_selection.append(
            {
                "fold": fold,
                "deep_weight": selected_weight,
                "selection_rows": int(selection.sum()),
                "selection_accuracy": max(candidates)[0],
                "held_all_accuracy": float(
                    np.mean(
                        logits[validation].argmax(axis=1)
                        == protocol.labels[validation]
                    )
                ),
                "held_legacy_available_accuracy": float(
                    np.mean(
                        logits[validation & legacy_available].argmax(axis=1)
                        == protocol.labels[validation & legacy_available]
                    )
                ),
            }
        )
    output_dir: Path = args.output_dir
    legacy_logits = np.log(np.maximum(legacy, 1e-8))
    legacy_payload = save_oof_artifact(
        output_dir,
        "imu_p89_sensorwise_aligned_reference",
        legacy_logits,
        protocol,
        metadata={
            "source": str(LEGACY.resolve()),
            "legacy_available": int(legacy_available.sum()),
            "missing_policy": "uniform probability",
            "note": "reused historical P89 OOF; not retrained",
        },
    )
    payload = save_oof_artifact(
        output_dir,
        "imu_p90_sensorwise_plus_deep_crossfit",
        logits,
        protocol,
        metadata={
            "legacy_source": str(LEGACY.resolve()),
            "deep_sources": [
                str((DEEP_RUN / "imu_hart128_scratch_oof.npz").resolve()),
                str((DEEP_RUN / "imu_hart128_maskedssl_oof.npz").resolve()),
            ],
            "formula": "(1-w)*log(P89 class_log_t1) + w*log(mean(P90 scratch, P90 SSL))",
            "weight_grid": grid.tolist(),
            "weight_selection": "for each held fold, maximize accuracy on the other two OOF folds only",
            "fold_selection": fold_selection,
            "legacy_available": int(legacy_available.sum()),
            "legacy_available_metrics": classification_metrics(
                logits[legacy_available], protocol.labels[legacy_available]
            ),
        },
    )
    print(
        json.dumps(
            {
                "legacy": legacy_payload["metrics"],
                "blend": payload["metrics"],
                "fold_selection": fold_selection,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
