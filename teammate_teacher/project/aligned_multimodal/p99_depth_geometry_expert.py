"""Leakage-safe source-OOF expert over P99-D1 explicit descriptors."""

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
    change_audit,
    evaluate_recipe,
    selection_key,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/p99_depth_geometry_d1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99-D1 source-OOF geometry expert")
    parser.add_argument("--stage", choices=("h1", "h2_confirmation"), default="h1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--d0-config", type=Path, default=DEFAULT_D0_CONFIG)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--depth-features", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--split-source", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--e0-base", type=Path, default=DEFAULT_E0_BASE)
    parser.add_argument("--h1-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_descriptor_groups(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(path.resolve(), allow_pickle=False) as source:
        forbidden = {"label", "labels", "class_id", "class_ids"}
        if forbidden & set(source.files):
            raise RuntimeError("D1 descriptor artifact contains ground truth")
        sample_ids = source["sample_ids"].astype(str)
        users = source["users"].astype(str)
        groups = {
            "geometry": source["geometry"].astype(np.float32),
            "depth_surface": source["depth_surface"].astype(np.float32),
        }
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise RuntimeError("D1 descriptor ids are not unique")
    return sample_ids, users, groups


def recipe_matrices(
    descriptor_ids: np.ndarray,
    groups: dict[str, np.ndarray],
    cohort_ids: np.ndarray,
    recipes: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for name, recipe in recipes.items():
        selected = []
        for group in recipe["groups"]:
            if group not in groups:
                raise KeyError(f"D1 recipe {name} requests unknown group {group}")
            selected.append(align(descriptor_ids, groups[group], cohort_ids))
        matrices[name] = np.concatenate(selected, axis=1).astype(np.float32)
    return matrices


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    d0_config = json.loads(args.d0_config.resolve().read_text(encoding="utf-8"))
    config_sha256 = canonical_hash(config)
    if args.stage == "h2_confirmation":
        if args.h1_summary is None:
            raise ValueError("D1 H2 requires --h1-summary")
        frozen = json.loads(args.h1_summary.resolve().read_text(encoding="utf-8"))
        if frozen.get("stage") != "P99_D1_H1" or frozen.get("config_sha256") != config_sha256:
            raise ValueError("D1 H1 summary/config mismatch")
        recipe_names = [str(frozen["selected_recipe"])]
    else:
        if args.h1_summary is not None:
            raise ValueError("--h1-summary is only valid for H2")
        recipe_names = list(config["recipes"])

    base, eval_indices, eval_ids = build_cohort(
        args.stage, d0_config, args.depth_features, args.split_source, args.e0_base
    )
    descriptor_ids, descriptor_users, groups = load_descriptor_groups(args.descriptors)
    aligned_users = align(descriptor_ids, descriptor_users, base.sample_ids).astype(str)
    if not np.array_equal(aligned_users, base.users):
        raise RuntimeError("D1 descriptor users disagree with the independent cohort")
    matrices = recipe_matrices(
        descriptor_ids,
        groups,
        base.sample_ids,
        {name: config["recipes"][name] for name in recipe_names},
    )
    cohort = Cohort(
        sample_ids=base.sample_ids,
        labels=base.labels,
        users=base.users,
        anchor_prediction=base.anchor_prediction,
        features=matrices,
    )
    results: dict[str, Any] = {}
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    for name in recipe_names:
        result, arrays = evaluate_recipe(
            cohort,
            eval_indices,
            config["recipes"][name],
            name,
            args.stage,
            int(config["seed"]),
        )
        results[name] = result
        artifacts[name] = arrays
        print(
            f"recipe={name} dim={matrices[name].shape[1]} "
            f"correct={result['metrics']['correct']}/{result['metrics']['total']} "
            f"top5={result['metrics']['top5']:.6f} "
            f"zero={result['zero_metrics']['correct']} shuffle={result['shuffle_metrics']['correct']}",
            flush=True,
        )
    selected = recipe_names[0] if args.stage == "h2_confirmation" else max(
        recipe_names, key=lambda name: selection_key(name, results[name])
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = cohort.labels[eval_indices]
    all_predictions: dict[str, np.ndarray] = {
        "sample_ids": eval_ids,
        "labels": labels,
        "users": cohort.users[eval_indices],
        "anchor_prediction": cohort.anchor_prediction[eval_indices],
    }
    for name in recipe_names:
        all_predictions[f"{name}_direct_probability"] = artifacts[name][
            "direct_probability"
        ]
        all_predictions[f"{name}_zero_logits"] = artifacts[name]["zero_logits"]
        all_predictions[f"{name}_shuffle_logits"] = artifacts[name]["shuffle_logits"]
    np.savez_compressed(output / "all_recipe_predictions.npz", **all_predictions)
    np.savez_compressed(
        output / ("h1_predictions.npz" if args.stage == "h1" else "h2_predictions.npz"),
        sample_ids=eval_ids,
        labels=cohort.labels[eval_indices],
        users=cohort.users[eval_indices],
        anchor_prediction=cohort.anchor_prediction[eval_indices],
        selected_recipe=np.asarray(selected),
        **artifacts[selected],
    )
    matched_removal: dict[str, Any] | None = None
    if "geometry_only" in artifacts and "depth_geometry" in artifacts:
        geometry_prediction = artifacts["geometry_only"]["direct_probability"].argmax(axis=1)
        combined_prediction = artifacts["depth_geometry"]["direct_probability"].argmax(axis=1)
        matched_removal = {
            "full": "depth_geometry",
            "without_depth_surface": "geometry_only",
            "change": change_audit(labels, geometry_prediction, combined_prediction),
            "per_user": {},
        }
        for user in sorted(set(cohort.users[eval_indices].tolist())):
            selected_user = cohort.users[eval_indices] == user
            matched_removal["per_user"][user] = {
                "geometry_correct": int(np.sum(geometry_prediction[selected_user] == labels[selected_user])),
                "depth_geometry_correct": int(np.sum(combined_prediction[selected_user] == labels[selected_user])),
                "delta": int(
                    np.sum(combined_prediction[selected_user] == labels[selected_user])
                    - np.sum(geometry_prediction[selected_user] == labels[selected_user])
                ),
            }
    summary = {
        "stage": "P99_D1_H1" if args.stage == "h1" else "P99_D1_H2_confirmation",
        "status": "complete",
        "hypothesis": config["hypothesis"],
        "config_sha256": config_sha256,
        "config_path": str(args.config.resolve()),
        "descriptor_path": str(args.descriptors.resolve()),
        "protocol": (
            "H1 leave-one-exploration-user-out with E0 source-only users"
            if args.stage == "h1"
            else "frozen H1 descriptor recipe trained on H1+E0 and evaluated once on H2"
        ),
        "evaluated_rows": int(len(eval_indices)),
        "selected_recipe": selected,
        "anchor": {
            "correct": int(np.sum(cohort.anchor_prediction[eval_indices] == cohort.labels[eval_indices])),
            "total": int(len(eval_indices)),
        },
        "candidates": results,
        "matched_removal": matched_removal,
        "leakage_audit": {
            "descriptor_artifact_label_free": True,
            "outer_user_disjoint": True,
            "scaler_fit_inside_outer_train": True,
            "temperature_fit_on_inner_user_oof": True,
            "h2_requires_frozen_h1_summary": True,
            "h3_code_path_present": False,
        },
        "next_decision": (
            "Compare depth_geometry against geometry_only and D0 rescue/harm before any Teacher fusion or Student probe."
            if args.stage == "h1"
            else "Audit the frozen Teacher confirmation; H3 remains unavailable."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "selected_recipe": selected,
                "anchor": summary["anchor"],
                "candidates": {
                    name: {
                        "correct": value["metrics"]["correct"],
                        "top5": value["metrics"]["top5"],
                        "balanced_accuracy": value["metrics"]["balanced_accuracy"],
                        "macro_f1": value["metrics"]["macro_f1"],
                        "zero_correct": value["zero_metrics"]["correct"],
                        "shuffle_correct": value["shuffle_metrics"]["correct"],
                        "vs_anchor": value["vs_anchor"],
                        "per_user": value["per_user"],
                    }
                    for name, value in results.items()
                },
                "matched_removal": matched_removal,
                "leakage_audit": summary["leakage_audit"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
