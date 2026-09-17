"""P99-S2 source-safe MotionBERT and HD-GCN single-expert audit."""

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
from p99_visual_oof_experts import extended_audit, pairwise_expert_audit
from train_p46_videomae_head import l2_normalize


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p99_pretrained_skeleton_s2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99-S2 pretrained Skeleton audit")
    parser.add_argument("--stage", choices=("h1", "h2_confirmation"), default="h1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--d0-config", type=Path, default=DEFAULT_D0_CONFIG)
    parser.add_argument("--depth-features", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--split-source", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--e0-base", type=Path, default=DEFAULT_E0_BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def motionbert_matrix(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 9216:
        raise ValueError(f"expected MotionBERT [N,9216], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("MotionBERT features contain non-finite values")
    return values


def hdgcn_matrix(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (6, 256):
        raise ValueError(f"expected HD-GCN [N,6,256], got {values.shape}")
    values = l2_normalize(values)
    if not np.isfinite(values).all():
        raise ValueError("HD-GCN features contain non-finite values")
    return values.reshape(len(values), -1)


def load_matrices(
    config: dict[str, Any], cohort_ids: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    motionbert_path = resolve(config["sources"]["motionbert"])
    hdgcn_path = resolve(config["sources"]["hdgcn"])
    with np.load(motionbert_path, allow_pickle=False) as source:
        motionbert_ids = np.asarray(source["sample_ids"]).astype(str)
        motionbert = motionbert_matrix(
            align(motionbert_ids, np.asarray(source["features"]), cohort_ids)
        )
    with np.load(hdgcn_path, allow_pickle=False) as source:
        hdgcn_ids = np.asarray(source["sample_ids"]).astype(str)
        hdgcn = hdgcn_matrix(
            align(hdgcn_ids, np.asarray(source["features"]), cohort_ids)
        )
    matrices = {
        "motionbert_fullpretrain_front": motionbert,
        "hdgcn_ntu60_sixstream": hdgcn,
    }
    expected = set(config["experts"])
    if set(matrices) != expected:
        raise RuntimeError(f"S2 config/code expert mismatch: {expected ^ set(matrices)}")
    return matrices, {
        "motionbert_cache": str(motionbert_path),
        "hdgcn_cache": str(hdgcn_path),
        "motionbert_cache_rows": int(len(motionbert_ids)),
        "hdgcn_cache_rows": int(len(hdgcn_ids)),
        "cohort_rows": int(len(cohort_ids)),
        "feature_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in matrices.items()
        },
        "embedded_cache_labels_used": False,
        "pretrained_weights_frozen": True,
    }


def selection_gate(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    rule = config["selection_gate"]
    changes = result["extended_audit"]["per_user_change"]
    users_with_rescue = sum(int(value["rescue"] > 0) for value in changes.values())
    checks = {
        "minimum_accuracy": result["metrics"]["accuracy"]
        >= float(rule["minimum_accuracy"]),
        "minimum_anchor_rescue": result["vs_anchor"]["rescue"]
        >= int(rule["minimum_anchor_rescue"]),
        "minimum_users_with_rescue": users_with_rescue
        >= int(rule["minimum_users_with_rescue"]),
        "every_user_rescue": users_with_rescue == len(changes),
        "direct_above_zero": result["metrics"]["correct"]
        > result["zero_metrics"]["correct"],
        "direct_above_shuffle": result["metrics"]["correct"]
        > result["shuffle_metrics"]["correct"],
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "users_with_rescue": int(users_with_rescue),
    }


def main() -> None:
    args = parse_args()
    if args.stage != "h1":
        raise ValueError("S2 is standalone H1-only; freeze a candidate before H2")
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    d0_config = json.loads(args.d0_config.resolve().read_text(encoding="utf-8"))
    base, eval_indices, eval_ids = build_cohort(
        "h1", d0_config, args.depth_features, args.split_source, args.e0_base
    )
    matrices, cache_audit = load_matrices(config, base.sample_ids)
    cohort = Cohort(
        sample_ids=base.sample_ids,
        labels=base.labels,
        users=base.users,
        anchor_prediction=base.anchor_prediction,
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
        result["selection_gate"] = selection_gate(config, result)
        results[name] = result
        artifacts[name] = arrays
        print(
            f"expert={name} correct={result['metrics']['correct']}/{len(labels)} "
            f"top5={result['metrics']['top5']:.6f} "
            f"rescue={result['vs_anchor']['rescue']} harm={result['vs_anchor']['harm']} "
            f"passed={result['selection_gate']['passed']}",
            flush=True,
        )
    names = list(config["experts"])
    pairwise = pairwise_expert_audit(
        labels,
        anchor,
        artifacts[names[0]]["direct_probability"],
        artifacts[names[1]]["direct_probability"],
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "h1_predictions.npz",
        sample_ids=eval_ids,
        labels=labels,
        users=users,
        anchor_prediction=anchor,
        **{
            f"{name}_direct_probability": arrays["direct_probability"].astype(np.float32)
            for name, arrays in artifacts.items()
        },
    )
    report = {
        "stage": "P99_S2_H1_pretrained_skeleton_single_expert_pool",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": canonical_hash(config),
        "protocol": "E0+H1 outer LOUO; frozen external representations; new fixed Ridge heads",
        "cache_audit": cache_audit,
        "anchor": {"correct": int(np.sum(anchor == labels)), "total": int(len(labels))},
        "experts": results,
        "pairwise_expert_audit": pairwise,
        "leakage_audit": {
            "backbone_features_label_free": True,
            "embedded_cache_labels_used": False,
            "old_heads_or_fold_selection_used": False,
            "outer_user_disjoint": True,
            "temperature_inner_user_oof": True,
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
            "selection_gate": result["selection_gate"],
        }
        for name, result in results.items()
    }
    print(json.dumps({"experts": compact, "pairwise": pairwise}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
