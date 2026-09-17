"""Validate the P109 Visual-first coarse-to-fine hard-class tree."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from a18_full_teacher_data import load_a18_data
from audit_p102_session_closure import load_npz
from audit_p108_class_evidence_atlas import (
    A9_USERS,
    A9_USER_SET,
    BLOCK_DESCRIPTION,
    DEFAULT_A9_OOF,
    DEFAULT_CLASSES,
    DEFAULT_VJEPA,
    EXPECTED_TOP10,
    FULL_OUTER_FOLDS,
    SEMANTIC_RULES,
    build_full_fold_ids,
    candidate_description,
    candidate_modalities,
    load_evidence_blocks,
    read_class_names,
)
from train_p104_modality_specialists_oof import (
    PCA_COMPONENTS,
    Projection,
    cyclic_shuffle_source,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "runs/p109_visual_first_tree_v1"
SEED = 20260824
VISUAL_BLOCKS = ("VLIT", "VHPD", "VWPD")
LEAF_RECIPES = tuple(
    recipe
    for visual in VISUAL_BLOCKS
    for recipe in (
        (visual,),
        (visual, "SWT"),
        (visual, "IAPD"),
        (visual, "SWT", "IAPD"),
    )
)

ROOT_GROUPS = {
    "HAND_OBJECT_ORAL_TABLE": (1, 6, 7, 8, 9, 10, 11, 14, 37),
    "DOCUMENT_HAND": (18, 21, 22, 23),
    "PERSONAL_DEVICE_BODY": (5, 19, 20, 24, 26, 27, 38, 39),
    "POSTURE_TRANSITION": (15, 32, 33, 34, 36),
}
LEAVES = {
    "ORAL_INTAKE": (1, 6, 7, 37),
    "TABLEWARE_MANIPULATION": (8, 9, 10, 11, 14),
    "DOCUMENT_ACTIVITY": (18, 21, 22, 23),
    "PHONE_DEVICE": (19, 20, 24, 26, 27),
    "BODY_CONTACT_DEVICE": (5, 38, 39),
    "POSTURE_MOTION": (32, 33, 34, 36),
    "SURFACE_WIPE_OUTLIER": (15,),
}
BRANCH_LEAVES = {
    "HAND_OBJECT_ORAL_TABLE": ("ORAL_INTAKE", "TABLEWARE_MANIPULATION"),
    "DOCUMENT_HAND": ("DOCUMENT_ACTIVITY",),
    "PERSONAL_DEVICE_BODY": ("PHONE_DEVICE", "BODY_CONTACT_DEVICE"),
    "POSTURE_TRANSITION": ("POSTURE_MOTION", "SURFACE_WIPE_OUTLIER"),
}
ROOT_NAMES = tuple(ROOT_GROUPS)
LEAF_NAMES = tuple(LEAVES)
SUPPORT_CLASSES = tuple(sorted({value for classes in ROOT_GROUPS.values() for value in classes}))
HARD_CLASSES = tuple(EXPECTED_TOP10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a9-oof", type=Path, default=DEFAULT_A9_OOF)
    parser.add_argument("--vjepa-root", type=Path, default=DEFAULT_VJEPA)
    parser.add_argument("--class-mapping", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise RuntimeError(f"refusing to write empty P109 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def validate_tree() -> None:
    root_members = [value for classes in ROOT_GROUPS.values() for value in classes]
    leaf_members = [value for classes in LEAVES.values() for value in classes]
    if len(root_members) != len(set(root_members)):
        raise RuntimeError("P109 root groups overlap")
    if len(leaf_members) != len(set(leaf_members)):
        raise RuntimeError("P109 leaves overlap")
    if set(root_members) != set(leaf_members) or tuple(sorted(root_members)) != SUPPORT_CLASSES:
        raise RuntimeError("P109 root/leaf support differs")
    if not set(HARD_CLASSES) <= set(SUPPORT_CLASSES):
        raise RuntimeError("P109 hard class is outside support")
    for branch, leaf_names in BRANCH_LEAVES.items():
        classes = {value for leaf in leaf_names for value in LEAVES[leaf]}
        if classes != set(ROOT_GROUPS[branch]):
            raise RuntimeError(f"P109 branch leaves differ for {branch}")


def class_to_root() -> dict[int, int]:
    return {
        class_id: root_index
        for root_index, name in enumerate(ROOT_NAMES)
        for class_id in ROOT_GROUPS[name]
    }


def class_to_leaf() -> dict[int, int]:
    return {
        class_id: leaf_index
        for leaf_index, name in enumerate(LEAF_NAMES)
        for class_id in LEAVES[name]
    }


def hard_class_paths() -> dict[int, tuple[str, str]]:
    roots = class_to_root()
    leaves = class_to_leaf()
    return {
        class_id: (ROOT_NAMES[roots[class_id]], LEAF_NAMES[leaves[class_id]])
        for class_id in HARD_CLASSES
    }


def node_metrics(
    labels: np.ndarray, prediction: np.ndarray, expected: Iterable[int]
) -> dict[str, Any]:
    values = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(prediction, dtype=np.int64)
    classes = list(map(int, expected))
    if not len(values):
        return {
            "rows": 0,
            "correct": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "per_class_recall": {},
        }
    recalls = {}
    for class_id in classes:
        selected = values == class_id
        recalls[str(class_id)] = (
            float(np.mean(predicted[selected] == class_id)) if selected.any() else None
        )
    valid = [value for value in recalls.values() if value is not None]
    return {
        "rows": len(values),
        "correct": int(np.sum(values == predicted)),
        "accuracy": float(np.mean(values == predicted)),
        "balanced_accuracy": float(np.mean(valid)),
        "macro_f1": float(
            f1_score(values, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "per_class_recall": recalls,
    }


def fit_node(
    x: np.ndarray, labels: np.ndarray, expected: Iterable[int]
) -> LogisticRegression:
    classes = set(map(int, expected))
    actual = set(map(int, np.unique(labels).tolist()))
    if actual != classes or len(classes) < 2:
        raise RuntimeError(f"P109 node support expected={sorted(classes)} actual={sorted(actual)}")
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=SEED,
    )
    model.fit(np.asarray(x, dtype=np.float32), labels)
    return model


def projected_matrix(
    projected: dict[str, np.ndarray], recipe: tuple[str, ...], rows: np.ndarray
) -> np.ndarray:
    return np.concatenate([projected[key][rows] for key in recipe], axis=1)


def node_selection_key(result: dict[str, Any], order: tuple[Any, ...]) -> tuple[Any, ...]:
    metrics = result["metrics"]
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
        -order.index(result["candidate"]),
    )


def leaf_selection_key(result: dict[str, Any]) -> tuple[Any, ...]:
    recipe = tuple(result["recipe"])
    auxiliary = int("SWT" in recipe) + int("IAPD" in recipe)
    metrics = result["metrics"]
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
        -auxiliary,
        -LEAF_RECIPES.index(recipe),
    )


def first_failure(
    true_root: int,
    predicted_root: int,
    true_leaf: int,
    predicted_leaf: int,
    true_class: int,
    predicted_class: int,
) -> str:
    if predicted_root != true_root:
        return "ROOT_VISUAL_FAIL"
    if predicted_leaf != true_leaf:
        return "FAMILY_VISUAL_FAIL"
    if predicted_class != true_class:
        return "LEAF_CLASS_FAIL"
    return "TREE_CORRECT"


def deployable_entry(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    top5 = np.argsort(values, axis=1)[:, ::-1][:, :5]
    support = set(SUPPORT_CLASSES)
    return np.asarray(
        [
            int(row[0]) in support
            and sum(int(value) in support for value in row.tolist()) >= 2
            for row in top5
        ],
        dtype=bool,
    )


def visual_variant(
    projected: dict[str, dict[str, np.ndarray]], key: str, variant: str
) -> np.ndarray:
    return projected[key][variant]


def recipe_variant_matrix(
    projected: dict[str, dict[str, np.ndarray]],
    recipe: tuple[str, ...],
    rows: np.ndarray,
    variant: str,
) -> np.ndarray:
    selected = []
    for key in recipe:
        block_variant = "aligned"
        if variant == "shuffle_all":
            block_variant = "shuffle"
        elif variant == "zero_all":
            block_variant = "zero"
        elif variant == "shuffle_skeleton" and key == "SWT":
            block_variant = "shuffle"
        elif variant == "zero_skeleton" and key == "SWT":
            block_variant = "zero"
        elif variant == "shuffle_imu" and key == "IAPD":
            block_variant = "shuffle"
        elif variant == "zero_imu" and key == "IAPD":
            block_variant = "zero"
        selected.append(projected[key][block_variant][rows])
    return np.concatenate(selected, axis=1)


def main() -> None:
    args = parse_args()
    validate_tree()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = read_class_names(args.class_mapping.resolve())
    data = load_a18_data()  # Full 18-source label-free tensors only.
    blocks = load_evidence_blocks(data, args.vjepa_root.resolve())
    sample_ids = data.sample_ids.astype(str)
    users = data.users.astype(str)
    labels = data.labels.astype(np.int64)
    full_fold_ids = build_full_fold_ids(users)
    roots = class_to_root()
    leaves = class_to_leaf()
    root_target = np.asarray([roots.get(int(value), -1) for value in labels], dtype=np.int64)
    leaf_target = np.asarray([leaves.get(int(value), -1) for value in labels], dtype=np.int64)
    support_mask = root_target >= 0
    hard_mask = np.isin(labels, np.asarray(HARD_CLASSES, dtype=np.int64))
    if set(labels[support_mask].tolist()) != set(SUPPORT_CLASSES):
        raise RuntimeError("P109 full support rows changed")

    outer_folds = [0] if args.smoke else list(range(3))
    visual_blocks = VISUAL_BLOCKS
    leaf_recipes = LEAF_RECIPES[:4] if args.smoke else LEAF_RECIPES
    components = 8 if args.smoke else PCA_COMPONENTS
    n = len(labels)
    tree_prediction = np.full(n, -1, dtype=np.int64)
    predicted_root_all = np.full(n, -1, dtype=np.int64)
    predicted_leaf_all = np.full(n, -1, dtype=np.int64)
    failure_all = np.full(n, "", dtype=object)
    sample_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    fold_details = []

    for outer_fold in outer_folds:
        source = full_fold_ids != outer_fold
        held = ~source
        source_rows = np.flatnonzero(source).astype(np.int64)
        held_rows = np.flatnonzero(held).astype(np.int64)
        inner_users = sorted(set(users[source].tolist()))
        if args.smoke:
            inner_users = inner_users[:2]
        inner_covered = np.zeros(n, dtype=bool)
        inner_root = {
            key: np.full(n, -1, dtype=np.int16) for key in visual_blocks
        }
        inner_sub = {
            branch: {
                key: np.full(n, -1, dtype=np.int16) for key in visual_blocks
            }
            for branch, branch_leaf_names in BRANCH_LEAVES.items()
            if len(branch_leaf_names) > 1
        }
        inner_leaf = {
            leaf: {
                recipe: np.full(n, -1, dtype=np.int16) for recipe in leaf_recipes
            }
            for leaf, classes in LEAVES.items()
            if len(classes) > 1
        }

        for inner_user in inner_users:
            inner_train_all = np.flatnonzero(source & (users != inner_user)).astype(np.int64)
            inner_held = source & (users == inner_user)
            projected = {}
            for key, block in blocks.items():
                projection = Projection.fit(block.values, inner_train_all, components)
                projected[key] = projection.transform(block.values)

            root_train = np.flatnonzero(source & (users != inner_user) & support_mask)
            root_validation = np.flatnonzero(inner_held & support_mask)
            for key in visual_blocks:
                model = fit_node(
                    projected[key][root_train], root_target[root_train], range(len(ROOT_NAMES))
                )
                inner_root[key][root_validation] = model.predict(
                    projected[key][root_validation]
                )

            for branch, branch_leaf_names in BRANCH_LEAVES.items():
                if len(branch_leaf_names) == 1:
                    continue
                branch_index = ROOT_NAMES.index(branch)
                expected = [LEAF_NAMES.index(value) for value in branch_leaf_names]
                train = np.flatnonzero(
                    source & (users != inner_user) & (root_target == branch_index)
                )
                validation = np.flatnonzero(inner_held & (root_target == branch_index))
                if not len(validation):
                    continue
                for key in visual_blocks:
                    model = fit_node(projected[key][train], leaf_target[train], expected)
                    inner_sub[branch][key][validation] = model.predict(
                        projected[key][validation]
                    )

            for leaf, classes in LEAVES.items():
                if len(classes) == 1:
                    continue
                leaf_index = LEAF_NAMES.index(leaf)
                train = np.flatnonzero(
                    source & (users != inner_user) & (leaf_target == leaf_index)
                )
                validation = np.flatnonzero(inner_held & (leaf_target == leaf_index))
                if not len(validation):
                    continue
                for recipe in leaf_recipes:
                    model = fit_node(
                        projected_matrix(projected, recipe, train), labels[train], classes
                    )
                    inner_leaf[leaf][recipe][validation] = model.predict(
                        projected_matrix(projected, recipe, validation)
                    )
            inner_covered |= inner_held

        source_support_rows = np.flatnonzero(inner_covered & support_mask)
        root_candidates = []
        for key in visual_blocks:
            metrics = node_metrics(
                root_target[source_support_rows],
                inner_root[key][source_support_rows],
                range(len(ROOT_NAMES)),
            )
            root_candidates.append({"candidate": key, "metrics": metrics})
        selected_root = max(
            root_candidates,
            key=lambda value: node_selection_key(value, visual_blocks),
        )

        selected_sub = {}
        sub_candidates = {}
        for branch, branch_leaf_names in BRANCH_LEAVES.items():
            if len(branch_leaf_names) == 1:
                continue
            branch_index = ROOT_NAMES.index(branch)
            expected = [LEAF_NAMES.index(value) for value in branch_leaf_names]
            rows = np.flatnonzero(inner_covered & (root_target == branch_index))
            values = []
            for key in visual_blocks:
                metrics = node_metrics(
                    leaf_target[rows], inner_sub[branch][key][rows], expected
                )
                values.append({"candidate": key, "metrics": metrics})
            selected_sub[branch] = max(
                values, key=lambda value: node_selection_key(value, visual_blocks)
            )
            sub_candidates[branch] = values

        selected_leaf = {}
        leaf_candidates = {}
        for leaf, classes in LEAVES.items():
            if len(classes) == 1:
                continue
            leaf_index = LEAF_NAMES.index(leaf)
            rows = np.flatnonzero(inner_covered & (leaf_target == leaf_index))
            values = []
            for recipe in leaf_recipes:
                metrics = node_metrics(labels[rows], inner_leaf[leaf][recipe][rows], classes)
                values.append(
                    {"candidate": recipe, "recipe": list(recipe), "metrics": metrics}
                )
            selected_leaf[leaf] = max(values, key=leaf_selection_key)
            leaf_candidates[leaf] = values

        shuffle_map = cyclic_shuffle_source(sample_ids, users, held)
        projected_source = {}
        projected_held: dict[str, dict[str, np.ndarray]] = {}
        projection_audit = {}
        for key, block in blocks.items():
            projection = Projection.fit(block.values, source_rows, components)
            projected_source[key] = projection.transform(block.values[source_rows])
            projected_held[key] = {
                "aligned": projection.transform(block.values[held_rows]),
                "shuffle": projection.transform(block.values[shuffle_map[held_rows]]),
                "zero": projection.zero(len(held_rows)),
            }
            projection_audit[key] = {
                "components": projection.components,
                "explained_variance": float(np.sum(projection.pca.explained_variance_ratio_)),
            }

        source_support = support_mask[source_rows]
        root_key = str(selected_root["candidate"])
        root_model = fit_node(
            projected_source[root_key][source_support],
            root_target[source_rows][source_support],
            range(len(ROOT_NAMES)),
        )
        held_predicted_root = root_model.predict(
            visual_variant(projected_held, root_key, "aligned")
        ).astype(np.int64)

        sub_predictions = {}
        for branch, branch_leaf_names in BRANCH_LEAVES.items():
            if len(branch_leaf_names) == 1:
                continue
            branch_index = ROOT_NAMES.index(branch)
            expected = [LEAF_NAMES.index(value) for value in branch_leaf_names]
            source_selected = root_target[source_rows] == branch_index
            key = str(selected_sub[branch]["candidate"])
            model = fit_node(
                projected_source[key][source_selected],
                leaf_target[source_rows][source_selected],
                expected,
            )
            sub_predictions[branch] = model.predict(
                visual_variant(projected_held, key, "aligned")
            ).astype(np.int64)

        leaf_models = {}
        leaf_aligned_predictions = {}
        leaf_variant_audit = {}
        for leaf, classes in LEAVES.items():
            leaf_index = LEAF_NAMES.index(leaf)
            if len(classes) == 1:
                leaf_aligned_predictions[leaf] = np.full(
                    len(held_rows), classes[0], dtype=np.int64
                )
                leaf_variant_audit[leaf] = {
                    "recipe": [],
                    "singleton": True,
                    "variants": {},
                }
                continue
            recipe = tuple(selected_leaf[leaf]["recipe"])
            source_selected = leaf_target[source_rows] == leaf_index
            model = fit_node(
                projected_matrix(
                    projected_source, recipe, np.flatnonzero(source_selected)
                ),
                labels[source_rows][source_selected],
                classes,
            )
            leaf_models[leaf] = model
            leaf_aligned_predictions[leaf] = model.predict(
                recipe_variant_matrix(
                    projected_held, recipe, np.arange(len(held_rows)), "aligned"
                )
            ).astype(np.int64)
            true_leaf_positions = np.flatnonzero(leaf_target[held_rows] == leaf_index)
            variants = {}
            for variant in (
                "aligned",
                "shuffle_all",
                "zero_all",
                "shuffle_skeleton",
                "zero_skeleton",
                "shuffle_imu",
                "zero_imu",
            ):
                prediction = model.predict(
                    recipe_variant_matrix(
                        projected_held, recipe, true_leaf_positions, variant
                    )
                )
                variants[variant] = node_metrics(
                    labels[held_rows][true_leaf_positions], prediction, classes
                )
            leaf_variant_audit[leaf] = {
                "recipe": list(recipe),
                "modalities": list(candidate_modalities(recipe)),
                "singleton": False,
                "variants": variants,
            }

        held_predicted_leaf = np.full(len(held_rows), -1, dtype=np.int64)
        for root_index, branch in enumerate(ROOT_NAMES):
            selected = held_predicted_root == root_index
            branch_leaf_names = BRANCH_LEAVES[branch]
            if len(branch_leaf_names) == 1:
                held_predicted_leaf[selected] = LEAF_NAMES.index(branch_leaf_names[0])
            else:
                held_predicted_leaf[selected] = sub_predictions[branch][selected]
        if np.any(held_predicted_leaf < 0):
            raise RuntimeError("P109 predicted leaf coverage failed")

        held_tree_prediction = np.full(len(held_rows), -1, dtype=np.int64)
        for leaf_index, leaf in enumerate(LEAF_NAMES):
            selected = held_predicted_leaf == leaf_index
            held_tree_prediction[selected] = leaf_aligned_predictions[leaf][selected]
        if np.any(held_tree_prediction < 0):
            raise RuntimeError("P109 tree class coverage failed")

        tree_prediction[held_rows] = held_tree_prediction
        predicted_root_all[held_rows] = held_predicted_root
        predicted_leaf_all[held_rows] = held_predicted_leaf
        held_support_positions = np.flatnonzero(support_mask[held_rows])
        for position in held_support_positions.tolist():
            row = int(held_rows[position])
            failure = first_failure(
                int(root_target[row]),
                int(held_predicted_root[position]),
                int(leaf_target[row]),
                int(held_predicted_leaf[position]),
                int(labels[row]),
                int(held_tree_prediction[position]),
            )
            failure_all[row] = failure
            sample_rows.append(
                {
                    "sample_id": sample_ids[row],
                    "subject": users[row],
                    "outer_fold": outer_fold,
                    "discovery_group": "A9" if users[row] in A9_USER_SET else "NON_A9",
                    "true_class": int(labels[row]),
                    "true_name": names[int(labels[row])],
                    "is_hard_target": int(labels[row] in HARD_CLASSES),
                    "true_root": ROOT_NAMES[int(root_target[row])],
                    "predicted_root": ROOT_NAMES[int(held_predicted_root[position])],
                    "root_correct": int(held_predicted_root[position] == root_target[row]),
                    "true_leaf": LEAF_NAMES[int(leaf_target[row])],
                    "predicted_leaf": LEAF_NAMES[int(held_predicted_leaf[position])],
                    "family_correct": int(
                        held_predicted_root[position] == root_target[row]
                        and held_predicted_leaf[position] == leaf_target[row]
                    ),
                    "tree_prediction": int(held_tree_prediction[position]),
                    "tree_prediction_name": names[int(held_tree_prediction[position])],
                    "tree_correct": int(held_tree_prediction[position] == labels[row]),
                    "first_failure": failure,
                }
            )

        held_support = support_mask[held_rows]
        held_hard = hard_mask[held_rows]
        root_metrics = node_metrics(
            root_target[held_rows][held_support],
            held_predicted_root[held_support],
            range(len(ROOT_NAMES)),
        )
        tree_support_metrics = node_metrics(
            labels[held_rows][held_support],
            held_tree_prediction[held_support],
            SUPPORT_CLASSES,
        )
        tree_hard_metrics = node_metrics(
            labels[held_rows][held_hard],
            held_tree_prediction[held_hard],
            HARD_CLASSES,
        )
        node_rows.append(
            {
                "outer_fold": outer_fold,
                "node_type": "ROOT_VISUAL",
                "node": "ROOT",
                "selected_evidence": root_key,
                "held_rows": root_metrics["rows"],
                "accuracy": root_metrics["accuracy"],
                "balanced_accuracy": root_metrics["balanced_accuracy"],
                "macro_f1": root_metrics["macro_f1"],
            }
        )
        for branch, branch_leaf_names in BRANCH_LEAVES.items():
            if len(branch_leaf_names) == 1:
                continue
            branch_index = ROOT_NAMES.index(branch)
            selected = root_target[held_rows] == branch_index
            expected = [LEAF_NAMES.index(value) for value in branch_leaf_names]
            metrics = node_metrics(
                leaf_target[held_rows][selected], sub_predictions[branch][selected], expected
            )
            node_rows.append(
                {
                    "outer_fold": outer_fold,
                    "node_type": "FAMILY_VISUAL",
                    "node": branch,
                    "selected_evidence": str(selected_sub[branch]["candidate"]),
                    "held_rows": metrics["rows"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
        for leaf, audit in leaf_variant_audit.items():
            if audit["singleton"]:
                continue
            metrics = audit["variants"]["aligned"]
            node_rows.append(
                {
                    "outer_fold": outer_fold,
                    "node_type": "LEAF_CLASS",
                    "node": leaf,
                    "selected_evidence": "+".join(audit["recipe"]),
                    "held_rows": metrics["rows"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
        fold_details.append(
            {
                "outer_fold": outer_fold,
                "held_users": list(FULL_OUTER_FOLDS[outer_fold]),
                "source_users": sorted(set(users[source].tolist())),
                "selected_root": selected_root,
                "root_candidates": root_candidates,
                "selected_subrouters": selected_sub,
                "subrouter_candidates": sub_candidates,
                "selected_leaves": selected_leaf,
                "leaf_candidates": leaf_candidates,
                "leaf_variant_audit": leaf_variant_audit,
                "projection_audit": projection_audit,
                "held_metrics": {
                    "root": root_metrics,
                    "tree_support": tree_support_metrics,
                    "tree_hard": tree_hard_metrics,
                },
            }
        )

    write_csv(output / "node_fold_metrics.csv", node_rows)
    write_csv(output / "support_sample_paths.csv", sample_rows)
    (output / "fold_details.json").write_text(
        json.dumps(fold_details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "tree_structure.json").write_text(
        json.dumps(
            {
                "root_groups": ROOT_GROUPS,
                "branch_leaves": BRANCH_LEAVES,
                "leaves": LEAVES,
                "hard_class_paths": hard_class_paths(),
                "support_classes": SUPPORT_CLASSES,
                "hard_classes": HARD_CLASSES,
                "visual_first": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.smoke:
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "status": "smoke_complete",
                    "folds": outer_folds,
                    "support_rows_written": len(sample_rows),
                    "a18_model_or_prediction_loaded": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"status": "smoke_complete", "output": str(output)}))
        return

    if np.any(tree_prediction < 0) or np.any(predicted_root_all < 0) or np.any(predicted_leaf_all < 0):
        raise RuntimeError("P109 full OOF tree prediction coverage failed")
    if np.any(failure_all[support_mask] == ""):
        raise RuntimeError("P109 support failure localization incomplete")

    hard_class_rows = []
    for target in HARD_CLASSES:
        selected = labels == target
        independent = selected & (~np.isin(users, np.asarray(A9_USERS, dtype=str)))
        root_correct = predicted_root_all[selected] == root_target[selected]
        family_correct = root_correct & (predicted_leaf_all[selected] == leaf_target[selected])
        tree_correct = tree_prediction[selected] == labels[selected]
        leaf_reached = family_correct
        leaf_conditional = float(np.mean(tree_correct[leaf_reached])) if leaf_reached.any() else 0.0
        per_subject = {
            user: float(np.mean(tree_prediction[selected & (users == user)] == target))
            for user in sorted(set(users[selected].tolist()))
        }
        worst_subject, worst_accuracy = min(
            per_subject.items(), key=lambda item: (item[1], item[0])
        )
        failures = Counter(failure_all[selected].tolist())
        path_root, path_leaf = hard_class_paths()[target]
        selected_recipes = []
        for fold in fold_details:
            leaf_record = fold["selected_leaves"].get(path_leaf)
            selected_recipes.append(
                "SINGLETON" if leaf_record is None else "+".join(leaf_record["recipe"])
            )
        recipe_counter = Counter(selected_recipes)
        primary_recipe, primary_count = sorted(
            recipe_counter.items(), key=lambda item: (-item[1], item[0])
        )[0]
        primary_cue, final_rule = SEMANTIC_RULES[target]
        hard_class_rows.append(
            {
                "class_id": target,
                "class_name": names[target],
                "tree_path": f"ROOT/{path_root}/{path_leaf}/{target}",
                "rows": int(np.sum(selected)),
                "root_reach": float(np.mean(root_correct)),
                "family_reach": float(np.mean(family_correct)),
                "leaf_accuracy_conditional": leaf_conditional,
                "tree_accuracy": float(np.mean(tree_correct)),
                "non_a9_rows": int(np.sum(independent)),
                "non_a9_tree_accuracy": float(
                    np.mean(tree_prediction[independent] == labels[independent])
                ),
                "root_visual_fail": int(failures["ROOT_VISUAL_FAIL"]),
                "family_visual_fail": int(failures["FAMILY_VISUAL_FAIL"]),
                "leaf_class_fail": int(failures["LEAF_CLASS_FAIL"]),
                "tree_correct": int(failures["TREE_CORRECT"]),
                "primary_leaf_recipe": primary_recipe,
                "primary_recipe_folds": primary_count,
                "all_leaf_recipes": "|".join(
                    f"f{fold}:{recipe}" for fold, recipe in enumerate(selected_recipes)
                ),
                "worst_subject": worst_subject,
                "worst_subject_accuracy": worst_accuracy,
                "decisive_local_cue": primary_cue,
                "final_confirmation_rule": final_rule,
                "semantic_rule_status": "needs visual confirmation",
            }
        )
    write_csv(output / "hard_class_tree_summary.csv", hard_class_rows)

    a9 = load_npz(args.a9_oof.resolve())
    a9_ids = a9["sample_ids"].astype(str)
    a9_users = a9["users"].astype(str)
    a9_labels = np.asarray(a9["labels"], dtype=np.int64)
    a9_probability = np.asarray(a9["selected_probability"], dtype=np.float64)
    a9_prediction = a9_probability.argmax(axis=1)
    canonical = np.isin(a9_users, np.asarray(A9_USERS, dtype=str))
    full_lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    full_rows = np.asarray([full_lookup[value] for value in a9_ids], dtype=np.int64)
    a9_tree = tree_prediction[full_rows]
    entry = deployable_entry(a9_probability)
    system = a9_prediction.copy()
    system[entry] = a9_tree[entry]
    a_correct = a9_prediction == a9_labels
    system_correct = system == a9_labels
    diagnostic_rows = []
    top5 = np.argsort(a9_probability, axis=1)[:, ::-1][:, :5]
    for row in np.flatnonzero(canonical).tolist():
        full_row = int(full_rows[row])
        diagnostic_rows.append(
            {
                "sample_id": a9_ids[row],
                "subject": a9_users[row],
                "true_class": int(a9_labels[row]),
                "true_name": names[int(a9_labels[row])],
                "a_prediction": int(a9_prediction[row]),
                "a_top5": "|".join(map(str, top5[row].tolist())),
                "tree_entry": int(entry[row]),
                "tree_prediction": int(a9_tree[row]),
                "system_prediction": int(system[row]),
                "a_correct": int(a_correct[row]),
                "system_correct": int(system_correct[row]),
                "rescue": int((not a_correct[row]) and system_correct[row]),
                "harm": int(a_correct[row] and (not system_correct[row])),
                "predicted_root": ROOT_NAMES[int(predicted_root_all[full_row])],
                "predicted_leaf": LEAF_NAMES[int(predicted_leaf_all[full_row])],
                "true_support_class": int(a9_labels[row] in SUPPORT_CLASSES),
                "true_hard_class": int(a9_labels[row] in HARD_CLASSES),
                "tree_internal_failure": (
                    failure_all[full_row] if a9_labels[row] in SUPPORT_CLASSES else "OUTSIDE_SUPPORT"
                ),
            }
        )
    write_csv(output / "a9_deployable_entry_diagnostic.csv", diagnostic_rows)

    support_tree_correct = tree_prediction[support_mask] == labels[support_mask]
    hard_tree_correct = tree_prediction[hard_mask] == labels[hard_mask]
    non_a9_hard = hard_mask & (~np.isin(users, np.asarray(A9_USERS, dtype=str)))
    support_failures = Counter(failure_all[support_mask].tolist())
    hard_failures = Counter(failure_all[hard_mask].tolist())
    canonical_entry = canonical & entry
    summary = {
        "status": "complete",
        "protocol": "P109 Visual-first root/family routing with conditional V/S/I leaves",
        "tree": {
            "support_classes": list(SUPPORT_CLASSES),
            "support_class_count": len(SUPPORT_CLASSES),
            "hard_classes": list(HARD_CLASSES),
            "hard_class_count": len(HARD_CLASSES),
            "root_groups": len(ROOT_GROUPS),
            "leaves": len(LEAVES),
        },
        "oracle_support_entry": {
            "support_rows": int(np.sum(support_mask)),
            "support_correct": int(np.sum(support_tree_correct)),
            "support_accuracy": float(np.mean(support_tree_correct)),
            "support_failures": dict(support_failures),
            "hard_rows": int(np.sum(hard_mask)),
            "hard_correct": int(np.sum(hard_tree_correct)),
            "hard_accuracy": float(np.mean(hard_tree_correct)),
            "hard_failures": dict(hard_failures),
            "non_a9_hard_rows": int(np.sum(non_a9_hard)),
            "non_a9_hard_accuracy": float(
                np.mean(tree_prediction[non_a9_hard] == labels[non_a9_hard])
            ),
        },
        "a9_deployable_entry": {
            "canonical_rows": int(np.sum(canonical)),
            "trigger_rows": int(np.sum(canonical_entry)),
            "a_correct": int(np.sum(a_correct[canonical])),
            "system_correct": int(np.sum(system_correct[canonical])),
            "rescue": int(np.sum(canonical & (~a_correct) & system_correct)),
            "harm": int(np.sum(canonical & a_correct & (~system_correct))),
            "net": int(np.sum(system_correct[canonical]) - np.sum(a_correct[canonical])),
            "rule": "A Top1 in support and at least two support classes in A Top5",
        },
        "visual_first": True,
        "skeleton_or_imu_used_in_root": False,
        "skeleton_or_imu_used_in_family_router": False,
        "a18_model_or_prediction_loaded": False,
        "flat_26_or_40_class_model_trained": False,
        "negative_results_preserved": True,
        "artifacts": {
            "tree": "tree_structure.json",
            "node_metrics": "node_fold_metrics.csv",
            "support_paths": "support_sample_paths.csv",
            "hard_summary": "hard_class_tree_summary.csv",
            "a9_diagnostic": "a9_deployable_entry_diagnostic.csv",
            "fold_details": "fold_details.json",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
