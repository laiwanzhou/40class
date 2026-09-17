"""Re-audit frozen Visual experts under the P99 E0+H1 firewall.

Only label-free feature caches are reused.  Old heads and old H2/H3 selection
decisions are not inputs to this run.
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
from train_p46_videomae_head import l2_normalize, row_standardize


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p99_visual_expert_v0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99 Visual source-OOF expert audit")
    parser.add_argument("--stage", choices=("h1", "h2_confirmation"), default="h1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--d0-config", type=Path, default=DEFAULT_D0_CONFIG)
    parser.add_argument("--depth-features", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--split-source", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--e0-base", type=Path, default=DEFAULT_E0_BASE)
    parser.add_argument("--h1-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def videomae_matrix(features: np.ndarray) -> np.ndarray:
    values = l2_normalize(np.asarray(features, dtype=np.float32))
    if values.ndim != 4 or values.shape[1:3] != (2, 3):
        raise ValueError(f"expected early/late x three views, got {values.shape}")
    return values.reshape(len(values), -1)


def internvideo_matrix(features: np.ndarray, action_logits: np.ndarray) -> np.ndarray:
    visual = videomae_matrix(features)
    action = np.asarray(action_logits, dtype=np.float32)
    if action.ndim != 4 or action.shape[1:3] != (2, 3):
        raise ValueError(f"expected InternVideo early/late action views, got {action.shape}")
    action = row_standardize(action.reshape(len(action), -1))
    return np.concatenate((visual, action), axis=1)


def vjepa_dense24_matrix(features: np.ndarray, action_logits: np.ndarray) -> np.ndarray:
    dense = l2_normalize(np.asarray(features, dtype=np.float32))
    action = row_standardize(np.asarray(action_logits, dtype=np.float32))
    if dense.ndim != 3 or dense.shape[1:] != (24, 1024):
        raise ValueError(f"expected V-JEPA dense24 features, got {dense.shape}")
    if action.ndim != 3 or action.shape[1:] != (24, 174):
        raise ValueError(f"expected V-JEPA dense24 SSV2 logits, got {action.shape}")
    grouped_dense = l2_normalize(dense.reshape(len(dense), 8, 3, 1024).mean(axis=2))
    grouped_action = row_standardize(
        action.reshape(len(action), 8, 3, 174).mean(axis=2)
    )
    return np.concatenate(
        (grouped_dense.reshape(len(dense), -1), grouped_action.reshape(len(dense), -1)),
        axis=1,
    )


def load_visual_matrices(
    config: dict[str, Any], cohort_ids: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sources = config["sources"]
    with np.load(resolve(sources["videomaev2_distilled_base"]), allow_pickle=False) as source:
        vmae_ids = source["sample_ids"].astype(str)
        vmae_features = align(vmae_ids, source["features"], cohort_ids)
    with np.load(resolve(sources["internvideo2_l_k400"]), allow_pickle=False) as source:
        iv2_ids = source["sample_ids"].astype(str)
        if not np.array_equal(vmae_ids, iv2_ids):
            raise RuntimeError("VideoMAEv2 and InternVideo2 cache order differs")
        iv2_features = align(iv2_ids, source["features"], cohort_ids)
        iv2_action = align(iv2_ids, source["action_logits"], cohort_ids)

    done = np.asarray(np.load(resolve(sources["vjepa2_dense24_done"]), mmap_mode="r"), dtype=bool)
    if len(done) != len(vmae_ids) or not done.all():
        raise RuntimeError(f"V-JEPA dense24 cache incomplete: {int(done.sum())}/{len(done)}")
    dense_lookup = {value: index for index, value in enumerate(vmae_ids)}
    dense_index = np.asarray([dense_lookup[value] for value in cohort_ids], dtype=np.int64)
    dense_source = np.load(resolve(sources["vjepa2_dense24_features"]), mmap_mode="r")
    action_source = np.load(resolve(sources["vjepa2_dense24_action_logits"]), mmap_mode="r")
    dense = np.asarray(dense_source[dense_index], dtype=np.float32)
    dense_action = np.asarray(action_source[dense_index], dtype=np.float32)
    matrices = {
        "videomaev2_early_late": videomae_matrix(vmae_features),
        "internvideo2_early_late_k400": internvideo_matrix(iv2_features, iv2_action),
        "vjepa2_dense24_group8_ssv2": vjepa_dense24_matrix(dense, dense_action),
    }
    expected = set(config["experts"])
    if set(matrices) != expected:
        raise RuntimeError(f"Visual config/code expert mismatch: {expected ^ set(matrices)}")
    metadata = {
        "master_cache_rows": int(len(vmae_ids)),
        "cohort_rows": int(len(cohort_ids)),
        "embedded_cache_labels_used": False,
        "vjepa_done_rows": int(done.sum()),
        "feature_dimensions": {name: int(value.shape[1]) for name, value in matrices.items()},
    }
    return matrices, metadata


def topk_contains(probability: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    indices = np.argpartition(-probability, kth=k - 1, axis=1)[:, :k]
    return np.any(indices == labels[:, None], axis=1)


def focus_group_audit(
    labels: np.ndarray,
    anchor_prediction: np.ndarray,
    probability: np.ndarray,
    classes: list[int],
) -> dict[str, Any]:
    selected = np.isin(labels, np.asarray(classes, dtype=np.int64))
    prediction = probability.argmax(axis=1)
    top5 = topk_contains(probability, labels, 5)
    if not selected.any():
        return {"classes": classes, "rows": 0}
    anchor_correct = anchor_prediction[selected] == labels[selected]
    candidate_correct = prediction[selected] == labels[selected]
    return {
        "classes": classes,
        "rows": int(selected.sum()),
        "anchor_correct": int(anchor_correct.sum()),
        "expert_correct": int(candidate_correct.sum()),
        "expert_top5_correct": int(top5[selected].sum()),
        "rescue": int(np.sum(~anchor_correct & candidate_correct)),
        "harm": int(np.sum(anchor_correct & ~candidate_correct)),
        "wrong_prediction_inside_group": int(
            np.sum(~candidate_correct & np.isin(prediction[selected], classes))
        ),
    }


def extended_audit(
    labels: np.ndarray,
    anchor_prediction: np.ndarray,
    probability: np.ndarray,
    focus_groups: dict[str, list[int]],
    users: np.ndarray | None = None,
) -> dict[str, Any]:
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    anchor_wrong = anchor_prediction != labels
    low_confidence = confidence < 0.95
    result: dict[str, Any] = {
        "anchor_error_top5_coverage": {
            "rows": int(anchor_wrong.sum()),
            "covered": int(topk_contains(probability, labels, 5)[anchor_wrong].sum()),
        },
        "confidence_lt_0_95": {
            "rows": int(low_confidence.sum()),
            "correct": int(np.sum((prediction == labels) & low_confidence)),
            "accuracy": float(np.mean(prediction[low_confidence] == labels[low_confidence]))
            if low_confidence.any()
            else None,
        },
        "focus_groups": {
            name: focus_group_audit(labels, anchor_prediction, probability, classes)
            for name, classes in focus_groups.items()
        },
    }
    if users is not None:
        result["per_user_change"] = {}
        for user in sorted(set(users.astype(str).tolist())):
            selected = users.astype(str) == user
            anchor_correct = anchor_prediction[selected] == labels[selected]
            expert_correct = prediction[selected] == labels[selected]
            result["per_user_change"][user] = {
                "rows": int(selected.sum()),
                "anchor_correct": int(anchor_correct.sum()),
                "expert_correct": int(expert_correct.sum()),
                "rescue": int(np.sum(~anchor_correct & expert_correct)),
                "harm": int(np.sum(anchor_correct & ~expert_correct)),
                "net": int(expert_correct.sum() - anchor_correct.sum()),
                "changed": int(np.sum(prediction[selected] != anchor_prediction[selected])),
            }
    return result


def pairwise_expert_audit(
    labels: np.ndarray,
    anchor_prediction: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
) -> dict[str, int]:
    first = first_probability.argmax(axis=1)
    second = second_probability.argmax(axis=1)
    first_correct = first == labels
    second_correct = second == labels
    anchor_wrong = anchor_prediction != labels
    first_rescue = anchor_wrong & first_correct
    second_rescue = anchor_wrong & second_correct
    return {
        "disagreements": int(np.sum(first != second)),
        "first_only_correct": int(np.sum(first_correct & ~second_correct)),
        "second_only_correct": int(np.sum(second_correct & ~first_correct)),
        "oracle_union_correct": int(np.sum(first_correct | second_correct)),
        "anchor_rescue_overlap": int(np.sum(first_rescue & second_rescue)),
        "first_unique_anchor_rescue": int(np.sum(first_rescue & ~second_rescue)),
        "second_unique_anchor_rescue": int(np.sum(second_rescue & ~first_rescue)),
        "anchor_rescue_union": int(np.sum(first_rescue | second_rescue)),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    d0_config = json.loads(args.d0_config.resolve().read_text(encoding="utf-8"))
    config_sha256 = canonical_hash(config)
    if args.stage == "h2_confirmation":
        raise ValueError("V0 is an H1-only pool audit; freeze a shortlist before H2")
    if args.h1_summary is not None:
        raise ValueError("--h1-summary is reserved for a later frozen H2 runner")

    base, eval_indices, eval_ids = build_cohort(
        "h1", d0_config, args.depth_features, args.split_source, args.e0_base
    )
    matrices, cache_audit = load_visual_matrices(config, base.sample_ids)
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
        results[name] = result
        artifacts[name] = arrays
        print(
            f"expert={name} dim={matrices[name].shape[1]} "
            f"correct={result['metrics']['correct']}/{result['metrics']['total']} "
            f"top5={result['metrics']['top5']:.6f} rescue={result['vs_anchor']['rescue']} "
            f"harm={result['vs_anchor']['harm']} shuffle={result['shuffle_metrics']['correct']}",
            flush=True,
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
    pairwise: dict[str, Any] = {}
    expert_names = list(config["experts"])
    for first_index, first in enumerate(expert_names):
        for second in expert_names[first_index + 1 :]:
            pairwise[f"{first}__vs__{second}"] = pairwise_expert_audit(
                labels,
                anchor,
                artifacts[first]["direct_probability"],
                artifacts[second]["direct_probability"],
            )
    report = {
        "stage": "P99_V0_H1_visual_single_expert_pool",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": config_sha256,
        "config_path": str(args.config.resolve()),
        "protocol": "E0+H1 outer leave-one-user-out; old heads and old H2/H3 decisions are not used",
        "evaluated_rows": int(len(eval_indices)),
        "anchor": {"correct": int(np.sum(anchor == labels)), "total": int(len(labels))},
        "cache_audit": cache_audit,
        "experts": results,
        "pairwise_expert_audit": pairwise,
        "selection_rule": config["selection"],
        "leakage_audit": {
            "backbone_features_label_free": True,
            "embedded_cache_labels_used": False,
            "outer_user_disjoint": True,
            "temperature_inner_user_oof": True,
            "old_h2_h3_selection_used": False,
            "h2_h3_accessed": False,
        },
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
            "per_user": result["per_user"],
            "anchor_error_top5": result["extended_audit"]["anchor_error_top5_coverage"],
        }
        for name, result in results.items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
