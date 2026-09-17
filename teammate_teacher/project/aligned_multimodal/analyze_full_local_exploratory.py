from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from analyze_local_depth_oof_fusion import (
    SMALL_ACTION_IDS,
    late_fuse_cross_fitted,
)
from analyze_thermal_oof_fusion import cluster_bootstrap_delta, fit_temperature


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_LOCAL = (
    PROJECT_DIR / "runs" / "p16_local_depth_oracle_oof" / "oof_logits.npz"
)
DEFAULT_SHARED = (
    PROJECT_DIR
    / "runs"
    / "p16_shared_full_local_oracle_oof"
    / "oof_logits.npz"
)
DEFAULT_HARD = (
    PROJECT_DIR / "data" / "hard_local_v1" / "hard_action_protocol.json"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p16_full_local_exploratory_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exploratory/oracle-assisted Full+Local OOF logits"
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--hard-protocol", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument(
        "--roi-protocol",
        choices=(
            "exploratory_oracle_assisted_all286",
            "strict_fold_pure",
        ),
        default="exploratory_oracle_assisted_all286",
    )
    return parser.parse_args()


def metric_dict(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def summarize(
    labels: np.ndarray,
    folds: np.ndarray,
    logits: np.ndarray,
    hard_ids: np.ndarray,
) -> dict[str, object]:
    predictions = logits.argmax(1)
    small = np.isin(labels, SMALL_ACTION_IDS)
    hard = np.isin(labels, hard_ids)
    return {
        "all": metric_dict(labels, logits),
        "small_actions": {
            "samples": int(small.sum()),
            **metric_dict(labels[small], logits[small]),
        },
        "large_actions": {
            "samples": int((~small).sum()),
            **metric_dict(labels[~small], logits[~small]),
        },
        "hard_classes": {
            "samples": int(hard.sum()),
            **metric_dict(labels[hard], logits[hard]),
        },
        "per_fold": {
            str(fold): metric_dict(
                labels[folds == fold],
                logits[folds == fold],
            )
            for fold in range(3)
        },
        "correct": int(np.sum(predictions == labels)),
    }


def joint_three_expert_cross_fitted(
    labels: np.ndarray,
    folds: np.ndarray,
    skeleton: np.ndarray,
    full: np.ndarray,
    local: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    result = np.zeros_like(skeleton, dtype=np.float64)
    protocols: list[dict[str, object]] = []
    grid = np.arange(0.0, 0.71, 0.05)
    for held_fold in range(3):
        calibration = folds != held_fold
        target = folds == held_fold
        skeleton_temperature = fit_temperature(
            skeleton[calibration],
            labels[calibration],
        )
        full_temperature = fit_temperature(
            full[calibration],
            labels[calibration],
        )
        local_temperature = fit_temperature(
            local[calibration],
            labels[calibration],
        )
        candidates: list[tuple[float, float, float, float, float, float]] = []
        for full_weight in grid:
            for local_weight in grid:
                if full_weight + local_weight > 0.9:
                    continue
                skeleton_weight = 1.0 - full_weight - local_weight
                logits = (
                    skeleton_weight
                    * skeleton[calibration]
                    / skeleton_temperature
                    + full_weight * full[calibration] / full_temperature
                    + local_weight * local[calibration] / local_temperature
                )
                predictions = logits.argmax(1)
                candidates.append(
                    (
                        float(accuracy_score(labels[calibration], predictions)),
                        float(
                            f1_score(
                                labels[calibration],
                                predictions,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        -float(full_weight + local_weight),
                        -float(local_weight),
                        float(full_weight),
                        float(local_weight),
                    )
                )
        (
            calibration_accuracy,
            calibration_macro_f1,
            _,
            _,
            full_weight,
            local_weight,
        ) = max(candidates)
        skeleton_weight = 1.0 - full_weight - local_weight
        result[target] = (
            skeleton_weight * skeleton[target] / skeleton_temperature
            + full_weight * full[target] / full_temperature
            + local_weight * local[target] / local_temperature
        )
        protocols.append(
            {
                "held_fold": held_fold,
                "skeleton_weight": skeleton_weight,
                "full_weight": full_weight,
                "local_weight": local_weight,
                "calibration_accuracy": calibration_accuracy,
                "calibration_macro_f1": calibration_macro_f1,
            }
        )
    return result, protocols


def comparison(
    labels: np.ndarray,
    sample_ids: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    base_predictions = baseline.argmax(1)
    candidate_predictions = candidate.argmax(1)
    return {
        "delta_pp": float(
            100
            * (
                np.mean(candidate_predictions == labels)
                - np.mean(base_predictions == labels)
            )
        ),
        "base_wrong_candidate_right": int(
            np.sum(
                (base_predictions != labels)
                & (candidate_predictions == labels)
            )
        ),
        "base_right_candidate_wrong": int(
            np.sum(
                (base_predictions == labels)
                & (candidate_predictions != labels)
            )
        ),
        "subject_bootstrap": cluster_bootstrap_delta(
            labels,
            candidate_predictions,
            base_predictions,
            sample_ids,
            repeats,
            seed,
        ),
    }


def main() -> None:
    args = parse_args()
    with np.load(args.base.resolve(), allow_pickle=False) as base:
        sample_ids = base["sample_ids"].astype(str)
        labels = base["labels"].astype(np.int64)
        folds = base["folds"].astype(np.int64)
        skeleton = base["skeleton_logits"].astype(np.float32)
        full = base["depth_logits"].astype(np.float32)
        current_sd = base["sd_logits"].astype(np.float32)
    with np.load(args.local.resolve(), allow_pickle=False) as local_file:
        if not np.array_equal(sample_ids, local_file["sample_ids"].astype(str)):
            raise ValueError("Base and oracle Local sample IDs differ")
        if not np.array_equal(labels, local_file["labels"].astype(np.int64)):
            raise ValueError("Base and oracle Local labels differ")
        local = local_file["logits"].astype(np.float32)
    hard_protocol = json.loads(
        args.hard_protocol.resolve().read_text(encoding="utf-8")
    )
    hard_ids = np.asarray(
        hard_protocol["hard_class_ids"],
        dtype=np.int64,
    )
    skeleton_local, skeleton_local_protocols = late_fuse_cross_fitted(
        labels,
        folds,
        skeleton,
        local,
    )
    three_expert, three_expert_protocols = joint_three_expert_cross_fitted(
        labels,
        folds,
        skeleton,
        full,
        local,
    )
    variants: dict[str, np.ndarray] = {
        "skeleton": skeleton,
        "full_depth": full,
        "oracle_local_depth": local,
        "current_skeleton_full": current_sd,
        "skeleton_oracle_local": skeleton_local,
        "skeleton_full_oracle_local": three_expert,
    }
    protocols: dict[str, object] = {
        "skeleton_oracle_local": skeleton_local_protocols,
        "skeleton_full_oracle_local": three_expert_protocols,
    }
    shared_path = args.shared.resolve()
    if shared_path.exists():
        with np.load(shared_path, allow_pickle=False) as shared:
            if not np.array_equal(
                sample_ids,
                shared["sample_ids"].astype(str),
            ):
                raise ValueError("Shared model sample IDs differ")
            shared_visual = shared["fused_logits"].astype(np.float32)
        skeleton_shared, skeleton_shared_protocols = late_fuse_cross_fitted(
            labels,
            folds,
            skeleton,
            shared_visual,
        )
        variants["shared_full_local_visual"] = shared_visual
        variants["skeleton_shared_full_local"] = skeleton_shared
        protocols["skeleton_shared_full_local"] = skeleton_shared_protocols

    metrics = {
        name: summarize(labels, folds, logits, hard_ids)
        for name, logits in variants.items()
    }
    comparisons = {
        "skeleton_oracle_local_vs_current_skeleton_full": comparison(
            labels,
            sample_ids,
            current_sd,
            skeleton_local,
            int(args.bootstrap_repeats),
            20260728,
        ),
        "three_expert_vs_current_skeleton_full": comparison(
            labels,
            sample_ids,
            current_sd,
            three_expert,
            int(args.bootstrap_repeats),
            20260729,
        ),
        "three_expert_vs_skeleton_oracle_local": comparison(
            labels,
            sample_ids,
            skeleton_local,
            three_expert,
            int(args.bootstrap_repeats),
            20260730,
        ),
    }
    if "skeleton_shared_full_local" in variants:
        comparisons["shared_vs_current_skeleton_full"] = comparison(
            labels,
            sample_ids,
            current_sd,
            variants["skeleton_shared_full_local"],
            int(args.bootstrap_repeats),
            20260731,
        )
        comparisons["shared_vs_three_independent_experts"] = comparison(
            labels,
            sample_ids,
            three_expert,
            variants["skeleton_shared_full_local"],
            int(args.bootstrap_repeats),
            20260801,
        )
    strict_fold_pure = str(args.roi_protocol) == "strict_fold_pure"
    report = {
        "status": (
            "strict_fold_pure"
            if strict_fold_pure
            else "exploratory_oracle_assisted"
        ),
        "roi_protocol": str(args.roi_protocol),
        "deployable_oof": strict_fold_pure,
        "wide_local": False,
        "samples": len(labels),
        "warning": (
            "Strict fold-pure ROI generation: the held fold's manual ROI "
            "supervision was excluded from its locator."
            if strict_fold_pure
            else (
                "The Local ROI locator used all available human ROI "
                "supervision. Subject classifier folds remain disjoint, but "
                "ROI generation is oracle-assisted and must be reproduced "
                "fold-pure before deployment."
            )
        ),
        "metrics": metrics,
        "comparisons": comparisons,
        "cross_fitted_protocols": protocols,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
