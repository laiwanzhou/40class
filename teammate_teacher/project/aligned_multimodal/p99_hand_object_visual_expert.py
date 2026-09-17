"""P99-HV0 leakage-safe hand/workspace Visual expert audit.

The VideoMAEv2 feature cache is label-free for this study.  Historical heads,
fold decisions, embedded labels, H2 and H3 are not inputs to this runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from p99_depth_oof_expert import (
    DEFAULT_CONFIG as DEFAULT_D0_CONFIG,
    DEFAULT_DEPTH,
    DEFAULT_E0_BASE,
    DEFAULT_SPLITS,
    Cohort,
    align,
    build_cohort,
    canonical_hash,
    evaluate_recipe,
)
from p99_visual_oof_experts import (
    extended_audit,
    pairwise_expert_audit,
)
from train_p46_videomae_head import l2_normalize


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p99_hand_object_visual_hv0.json"
DEFAULT_V0_SUMMARY = PROJECT / "runs/p99_visual_expert_v0_h1_v1/summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99-HV0 hand/workspace Visual audit")
    parser.add_argument("--stage", choices=("h1", "h2_confirmation"), default="h1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--d0-config", type=Path, default=DEFAULT_D0_CONFIG)
    parser.add_argument("--depth-features", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--split-source", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--e0-base", type=Path, default=DEFAULT_E0_BASE)
    parser.add_argument("--v0-summary", type=Path, default=DEFAULT_V0_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def hand_feature_families(features: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (2, 3, 768):
        raise ValueError(
            "expected full/peak x left/right/interaction x 768 hand features, "
            f"got {values.shape}"
        )
    values = l2_normalize(values)
    mean_windows = l2_normalize(values.mean(axis=1))
    return {
        "hand_all": values.reshape(len(values), -1),
        "mean_windows": mean_windows.reshape(len(values), -1),
        "interaction_only": values[:, :, 2, :].reshape(len(values), -1),
    }


def load_hand_matrices(
    config: dict[str, Any], cohort_ids: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = resolve(config["source"])
    with np.load(cache_path, allow_pickle=False) as source:
        sample_ids = np.asarray(source["sample_ids"]).astype(str)
        features = align(sample_ids, np.asarray(source["features"]), cohort_ids)
    matrices = hand_feature_families(features)
    expected = set(config["experts"])
    if set(matrices) != expected:
        raise RuntimeError(f"HV0 config/code expert mismatch: {expected ^ set(matrices)}")
    return matrices, {
        "cache": str(cache_path),
        "cache_rows": int(len(sample_ids)),
        "cohort_rows": int(len(cohort_ids)),
        "embedded_cache_labels_used": False,
        "feature_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in matrices.items()
        },
        "crop_contract": "full/peak-motion x left-hand/right-hand/two-hand-interaction",
        "crop_before_visual_encoding": True,
    }


def load_v0_control(path: Path) -> dict[str, Any]:
    report = json.loads(path.resolve().read_text(encoding="utf-8"))
    if report.get("stage") != "P99_V0_H1_visual_single_expert_pool":
        raise ValueError("--v0-summary is not the frozen P99-V0 H1 report")
    expert = report["experts"]["videomaev2_early_late"]
    return {
        "name": "videomaev2_early_late",
        "correct": int(expert["metrics"]["correct"]),
        "top5": float(expert["metrics"]["top5"]),
        "rescue": int(expert["vs_anchor"]["rescue"]),
        "harm": int(expert["vs_anchor"]["harm"]),
        "per_user": expert["extended_audit"]["per_user_change"],
    }


def main() -> None:
    args = parse_args()
    if args.stage != "h1":
        raise ValueError("HV0 is H1-only; freeze a candidate before H2")
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    d0_config = json.loads(args.d0_config.resolve().read_text(encoding="utf-8"))
    cohort_base, eval_indices, eval_ids = build_cohort(
        "h1", d0_config, args.depth_features, args.split_source, args.e0_base
    )
    matrices, cache_audit = load_hand_matrices(config, cohort_base.sample_ids)
    cohort = Cohort(
        sample_ids=cohort_base.sample_ids,
        labels=cohort_base.labels,
        users=cohort_base.users,
        anchor_prediction=cohort_base.anchor_prediction,
        features=matrices,
    )
    labels = cohort.labels[eval_indices]
    users = cohort.users[eval_indices]
    anchor = cohort.anchor_prediction[eval_indices]
    results: dict[str, Any] = {}
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    for name, recipe in config["experts"].items():
        result, arrays = evaluate_recipe(
            cohort, eval_indices, recipe, name, "h1", int(config["seed"])
        )
        result["extended_audit"] = extended_audit(
            labels, anchor, arrays["direct_probability"], config["focus_groups"], users
        )
        results[name] = result
        artifacts[name] = arrays
        print(
            f"expert={name} dim={matrices[name].shape[1]} "
            f"correct={result['metrics']['correct']}/{result['metrics']['total']} "
            f"top5={result['metrics']['top5']:.6f} "
            f"rescue={result['vs_anchor']['rescue']} "
            f"harm={result['vs_anchor']['harm']} "
            f"shuffle={result['shuffle_metrics']['correct']}",
            flush=True,
        )

    main_result = results["hand_all"]
    v0_control = load_v0_control(args.v0_summary)
    rescued_users = sum(
        int(value["rescue"] > 0)
        for value in main_result["extended_audit"]["per_user_change"].values()
    )
    selection_checks = {
        "direct_above_zero": main_result["metrics"]["correct"]
        > main_result["zero_metrics"]["correct"],
        "direct_above_shuffle": main_result["metrics"]["correct"]
        > main_result["shuffle_metrics"]["correct"],
        "more_anchor_rescue_than_global_videomae": main_result["vs_anchor"]["rescue"]
        > v0_control["rescue"],
        "fewer_anchor_harm_than_global_videomae": main_result["vs_anchor"]["harm"]
        < v0_control["harm"],
        "at_least_three_users_with_rescue": rescued_users >= 3,
    }
    pairwise: dict[str, Any] = {}
    names = list(config["experts"])
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            pairwise[f"{first}__vs__{second}"] = pairwise_expert_audit(
                labels,
                anchor,
                artifacts[first]["direct_probability"],
                artifacts[second]["direct_probability"],
            )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, np.ndarray] = {
        "sample_ids": eval_ids,
        "labels": labels,
        "users": users,
        "anchor_prediction": anchor,
    }
    for name, arrays in artifacts.items():
        predictions[f"{name}_direct_probability"] = arrays["direct_probability"]
        predictions[f"{name}_zero_logits"] = arrays["zero_logits"]
        predictions[f"{name}_shuffle_logits"] = arrays["shuffle_logits"]
    np.savez_compressed(output / "h1_predictions.npz", **predictions)
    report = {
        "stage": "P99_HV0_H1_hand_workspace_visual_expert",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": canonical_hash(config),
        "protocol": "E0+H1 outer LOUO; label-free hand cache; old head/H2/H3 unused",
        "evaluated_rows": int(len(eval_indices)),
        "anchor": {"correct": int(np.sum(anchor == labels)), "total": int(len(labels))},
        "cache_audit": cache_audit,
        "experts": results,
        "pairwise_ablation_audit": pairwise,
        "global_videomae_control": v0_control,
        "selection_gate": {
            "passed": bool(all(selection_checks.values())),
            "checks": selection_checks,
            "rescued_users": int(rescued_users),
        },
        "selection_rule": config["selection"],
        "leakage_audit": {
            "backbone_features_label_free": True,
            "embedded_cache_labels_used": False,
            "outer_user_disjoint": True,
            "temperature_inner_user_oof": True,
            "old_hand_head_or_fold_selection_used": False,
            "h2_h3_accessed": False,
            "h3_code_path_present": False,
        },
        "h2_policy": config["h2_policy"],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = {
        name: {
            "correct": result["metrics"]["correct"],
            "top5": result["metrics"]["top5"],
            "balanced_accuracy": result["metrics"]["balanced_accuracy"],
            "macro_f1": result["metrics"]["macro_f1"],
            "zero_correct": result["zero_metrics"]["correct"],
            "shuffle_correct": result["shuffle_metrics"]["correct"],
            "vs_anchor": result["vs_anchor"],
            "per_user": result["extended_audit"]["per_user_change"],
        }
        for name, result in results.items()
    }
    print(
        json.dumps(
            {"experts": compact, "selection_gate": report["selection_gate"]},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
