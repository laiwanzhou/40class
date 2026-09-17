from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audit_p86_teacher_mechanisms import (
    FROZEN_ALPHA,
    FROZEN_CLASS_WEIGHT_POWER,
    fixed_model_perturbations,
)
from p46_protocol import HARD_CLASS_IDS
from train_p46_dinov2_head import feature_sets as dinov2_feature_sets
from train_p46_videomae_base_large_joint import build_matrices
from train_p46_videomae_head import (
    aligned_scores,
    feature_sets as videomae_feature_sets,
    make_model,
)
from train_p46_videomae_ir_depth_head import matrices as ir_depth_matrices
from train_p46_videomae_large_weighted import sample_weights
from train_p46_videomae_multiclip_head import matrices as multiclip_matrices
from train_p46_videomae_relation_head import Config, RelationHead, infer
from train_p46_videomae_subject_svm import aligned_scores as svm_aligned_scores
from train_p46_videomae_subject_svm import l2 as svm_l2
from train_p46_videomae_temporal_head import feature_sets as temporal_feature_sets
from train_p85_videomae_full40_head import (
    aligned_scores_40,
    sample_weights as p85_sample_weights,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_multiexpert_test_logits_v1"
HARD_CLASSES = np.asarray(HARD_CLASS_IDS, dtype=np.int64)
P46_DIRECT_RUNS = (
    "p46_dinov2_base_head_v1",
    "p46_videomae_base_large_joint_v1",
    "p46_videomae_depth_head_v1",
    "p46_videomae_head_v1",
    "p46_videomae_ir_depth_head_v1",
    "p46_videomae_large_bagging_v1",
    "p46_videomae_large_head_v1",
    "p46_videomae_large_multiclip_head_v1",
    "p46_videomae_large_weighted_v1",
    "p46_videomae_relation_head_v1",
    "p46_videomae_ssv2_head_v1",
    "p46_videomae_subject_svm_v1",
    "p46_videomae_temporal_head_v2",
    "p46_videomae_thermal_head_v1",
)
P46_MC_KEYS = (
    "full_logits",
    "early_logits",
    "late_logits",
    "window_mean_logits",
    "early_late_logits",
    "full_window_mean_logits",
    "full_temporal_delta_logits",
    "full_early_late_logits",
    "three_clip_kinetics_logits",
)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def summary(run: str) -> dict:
    return json.loads(
        (PROJECT_DIR / "runs" / run / "summary.json").read_text(encoding="utf-8")
    )


def align(reference: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ids = np.asarray(cache["sample_ids"]).astype(str)
    lookup = {value: index for index, value in enumerate(ids)}
    missing = [value for value in reference.astype(str) if value not in lookup]
    if missing:
        raise RuntimeError(f"cache misses {len(missing)} requested rows")
    order = np.asarray([lookup[value] for value in reference.astype(str)], dtype=np.int64)
    return {
        key: value[order] if value.ndim and value.shape[0] == len(ids) else value
        for key, value in cache.items()
    }


def fit_detail_head(
    train_values: np.ndarray,
    test_values: np.ndarray,
    labels21: np.ndarray,
    alpha: float,
    temperature: float,
    weight_power: float = 0.0,
) -> np.ndarray:
    model = make_model(alpha)
    kwargs = {}
    if weight_power:
        kwargs["ridge__sample_weight"] = sample_weights(labels21, weight_power)
    model.fit(train_values, labels21, **kwargs)
    return aligned_scores(model, test_values) / temperature


def build_p46() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    base_train = load(PROJECT_DIR / "runs/p46_videomae_foundation_v1/complete_features.npz")
    ids = base_train["sample_ids"].astype(str)
    labels40 = base_train["labels"].astype(np.int64)
    class_to_index = {value: index for index, value in enumerate(HARD_CLASS_IDS)}
    labels21 = np.asarray([class_to_index[int(value)] for value in labels40], dtype=np.int64)
    base_test = load(PROJECT_DIR / "runs/p89_videomae_base_test_v1/complete_features.npz")
    test_ids = base_test["sample_ids"].astype(str)

    depth_train = align(ids, load(PROJECT_DIR / "runs/p46_videomae_depth_v1/complete_features.npz"))
    thermal_train = align(ids, load(PROJECT_DIR / "runs/p46_videomae_thermal_v1/complete_features.npz"))
    dino_train = align(ids, load(PROJECT_DIR / "runs/p46_dinov2_base_ir_v1/complete_features.npz"))
    temporal_train = align(ids, load(PROJECT_DIR / "runs/p46_videomae_temporal_v2/complete_features.npz"))
    ssv2_train = align(ids, load(PROJECT_DIR / "runs/p46_videomae_ssv2_ir_v1/complete_features.npz"))
    large_train = align(ids, load(PROJECT_DIR / "runs/p46_videomae_large_ir_v1/complete_features.npz"))
    windows_train = align(ids, load(PROJECT_DIR / "runs/p46_videomae_large_multiclip_v1/complete_features.npz"))

    depth_test = align(test_ids, load(PROJECT_DIR / "runs/p89_videomae_depth_test_v1/complete_features.npz"))
    thermal_test = align(test_ids, load(PROJECT_DIR / "runs/p89_videomae_thermal_test_v1/complete_features.npz"))
    dino_test = align(test_ids, load(PROJECT_DIR / "runs/p89_dinov2_base_test_v1/complete_features.npz"))
    temporal_test = align(test_ids, load(PROJECT_DIR / "runs/p89_videomae_temporal_test_v1/complete_features.npz"))
    ssv2_test = align(test_ids, load(PROJECT_DIR / "runs/p89_videomae_ssv2_test_v1/complete_features.npz"))
    large_test = align(test_ids, load(PROJECT_DIR / "runs/p85_videomae_large_fullwindow_test_v1/complete_features.npz"))
    windows_test = align(test_ids, load(PROJECT_DIR / "runs/p46_videomae_large_multiclip_test_v1/complete_features.npz"))

    train_outputs: dict[str, np.ndarray] = {}
    test_outputs: dict[str, np.ndarray] = {}

    def store_refit(
        run: str,
        train_values: np.ndarray,
        test_values: np.ndarray,
        *,
        alpha: float | None = None,
        weight_power: float = 0.0,
    ) -> None:
        info = summary(run)
        selected = info.get("selected_cv", {})
        chosen_alpha = float(alpha if alpha is not None else selected["alpha"])
        temperature = float(
            info.get("temperature_from_train_oof", info.get("temperature_from_oof", 1.0))
        )
        test_outputs[run] = fit_detail_head(
            train_values,
            test_values,
            labels21,
            chosen_alpha,
            temperature,
            weight_power,
        ).astype(np.float32)

    dino_train_sets = dinov2_feature_sets(dino_train["features"])
    dino_test_sets = dinov2_feature_sets(dino_test["features"])
    dino_selected = summary("p46_dinov2_base_head_v1")["selected_cv"]
    store_refit(
        "p46_dinov2_base_head_v1",
        dino_train_sets[dino_selected["feature_set"]],
        dino_test_sets[dino_selected["feature_set"]],
        weight_power=float(dino_selected["class_weight_power"]),
    )

    base_sets_train = videomae_feature_sets(base_train["features"], base_train["kinetics_logits"])
    base_sets_test = videomae_feature_sets(base_test["features"], base_test["kinetics_logits"])
    for run in ("p46_videomae_head_v1",):
        selected = summary(run)["selected_cv"]
        store_refit(run, base_sets_train[selected["feature_set"]], base_sets_test[selected["feature_set"]])

    depth_sets_train = videomae_feature_sets(depth_train["features"], depth_train["kinetics_logits"])
    depth_sets_test = videomae_feature_sets(depth_test["features"], depth_test["kinetics_logits"])
    selected = summary("p46_videomae_depth_head_v1")["selected_cv"]
    store_refit(
        "p46_videomae_depth_head_v1",
        depth_sets_train[selected["feature_set"]],
        depth_sets_test[selected["feature_set"]],
    )

    ir_depth_train = ir_depth_matrices(
        base_train["features"],
        depth_train["features"],
        base_train["kinetics_logits"],
        depth_train["kinetics_logits"],
    )
    ir_depth_test = ir_depth_matrices(
        base_test["features"],
        depth_test["features"],
        base_test["kinetics_logits"],
        depth_test["kinetics_logits"],
    )
    selected = summary("p46_videomae_ir_depth_head_v1")["selected_cv"]
    store_refit(
        "p46_videomae_ir_depth_head_v1",
        ir_depth_train[selected["feature_set"]],
        ir_depth_test[selected["feature_set"]],
    )

    thermal_sets_train = videomae_feature_sets(
        thermal_train["features"], thermal_train["kinetics_logits"]
    )
    thermal_sets_test = videomae_feature_sets(
        thermal_test["features"], thermal_test["kinetics_logits"]
    )
    selected = summary("p46_videomae_thermal_head_v1")["selected_cv"]
    store_refit(
        "p46_videomae_thermal_head_v1",
        thermal_sets_train[selected["feature_set"]],
        thermal_sets_test[selected["feature_set"]],
    )

    ssv2_sets_train = videomae_feature_sets(ssv2_train["features"], ssv2_train["kinetics_logits"])
    ssv2_sets_test = videomae_feature_sets(ssv2_test["features"], ssv2_test["kinetics_logits"])
    selected = summary("p46_videomae_ssv2_head_v1")["selected_cv"]
    store_refit(
        "p46_videomae_ssv2_head_v1",
        ssv2_sets_train[selected["feature_set"]],
        ssv2_sets_test[selected["feature_set"]],
    )

    temporal_sets_train = temporal_feature_sets(
        temporal_train["features"],
        temporal_train["temporal_features"],
        temporal_train["quadrant_features"],
    )
    temporal_sets_test = temporal_feature_sets(
        temporal_test["features"],
        temporal_test["temporal_features"],
        temporal_test["quadrant_features"],
    )
    selected = summary("p46_videomae_temporal_head_v2")["selected_cv"]
    store_refit(
        "p46_videomae_temporal_head_v2",
        temporal_sets_train[selected["feature_set"]],
        temporal_sets_test[selected["feature_set"]],
    )

    large_joint_train = build_matrices(
        base_train["features"],
        base_train["kinetics_logits"],
        large_train["features"],
        large_train["kinetics_logits"],
    )
    large_joint_test = build_matrices(
        base_test["features"],
        base_test["kinetics_logits"],
        large_test["features"],
        large_test["kinetics_logits"],
    )
    selected = summary("p46_videomae_base_large_joint_v1")["selected_cv"]
    store_refit(
        "p46_videomae_base_large_joint_v1",
        large_joint_train[selected["feature_set"]],
        large_joint_test[selected["feature_set"]],
    )

    large_sets_train = videomae_feature_sets(large_train["features"], large_train["kinetics_logits"])
    large_sets_test = videomae_feature_sets(large_test["features"], large_test["kinetics_logits"])
    for run, power in (
        ("p46_videomae_large_head_v1", 0.0),
        ("p46_videomae_large_weighted_v1", 0.75),
    ):
        selected = summary(run)["selected_cv"]
        feature_name = selected.get("feature_set", "concat_views")
        store_refit(
            run,
            large_sets_train[feature_name],
            large_sets_test[feature_name],
            weight_power=power,
        )

    bag = joblib.load(PROJECT_DIR / "runs/p46_videomae_large_bagging_v1/bagged_models.joblib")
    fraction = float(bag["bagged_fraction"])
    values_test = large_sets_test["concat_views"]
    direct = aligned_scores(bag["final_model"], values_test)
    bagged = np.mean([aligned_scores(model, values_test) for model in bag["outer_models"]], axis=0)
    test_outputs["p46_videomae_large_bagging_v1"] = (
        ((1.0 - fraction) * direct + fraction * bagged)
        / float(summary("p46_videomae_large_bagging_v1")["temperature_from_train_oof"])
    ).astype(np.float32)

    mc_train = multiclip_matrices(
        large_train["features"],
        large_train["kinetics_logits"],
        windows_train["features"],
        windows_train["kinetics_logits"],
    )
    mc_test = multiclip_matrices(
        large_test["features"],
        large_test["kinetics_logits"],
        windows_test["features"],
        windows_test["kinetics_logits"],
    )
    mc_summary = summary("p46_videomae_large_multiclip_head_v1")
    for name, values in mc_train.items():
        candidate = mc_summary["candidate_diagnostics"][name]
        selected = candidate["selected_cv"]
        logits = fit_detail_head(
            values,
            mc_test[name],
            labels21,
            float(selected["alpha"]),
            float(candidate["temperature"]),
            float(selected["class_weight_power"]),
        )
        test_outputs[f"p46_mc_{name}"] = logits.astype(np.float32)
    test_outputs["p46_videomae_large_multiclip_head_v1"] = test_outputs[
        "p46_mc_full_window_mean"
    ]

    checkpoint = torch.load(
        PROJECT_DIR / "runs/p46_videomae_relation_head_v1/final_head.pt",
        map_location="cpu",
        weights_only=False,
    )
    relation = RelationHead(Config(**checkpoint["config"]))
    relation.load_state_dict(checkpoint["state_dict"])
    relation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relation = relation.eval().to(relation_device)
    relation_temporal = temporal_test["temporal_features"].astype(np.float32)
    relation_pooled = temporal_test["features"].astype(np.float32)
    relation_temporal /= np.maximum(np.linalg.norm(relation_temporal, axis=-1, keepdims=True), 1e-8)
    relation_pooled /= np.maximum(np.linalg.norm(relation_pooled, axis=-1, keepdims=True), 1e-8)
    relation_logits = infer(
        relation,
        torch.from_numpy(relation_temporal),
        torch.from_numpy(relation_pooled),
        np.arange(len(test_ids), dtype=np.int64),
        64,
        relation_device,
    )
    test_outputs["p46_videomae_relation_head_v1"] = (
        relation_logits
        / float(summary("p46_videomae_relation_head_v1")["temperature_from_train_oof"])
    ).astype(np.float32)

    saved_svm = joblib.load(PROJECT_DIR / "runs/p46_videomae_subject_svm_v1/final_model.joblib")
    svm_values = svm_l2(svm_l2(base_test["features"]).mean(axis=1))
    if float(saved_svm["subject_centering"]):
        svm_values = svm_values - float(saved_svm["subject_centering"]) * (
            svm_values.mean(axis=0, keepdims=True)
            - np.asarray(saved_svm["training_global_mean"])[None]
        )
    test_outputs["p46_videomae_subject_svm_v1"] = (
        svm_aligned_scores(saved_svm["model"], svm_values)
        / float(summary("p46_videomae_subject_svm_v1")["temperature_from_train_oof"])
    ).astype(np.float32)

    for run in P46_DIRECT_RUNS:
        with np.load(PROJECT_DIR / "runs" / run / "crossfit_logits.npz") as data:
            train_outputs[run] = np.asarray(data["logits"], dtype=np.float32)
    with np.load(
        PROJECT_DIR / "runs/p46_videomae_large_multiclip_head_v1/candidate_crossfit_logits.npz"
    ) as data:
        for key in P46_MC_KEYS:
            train_outputs[f"p46_mc_{key.removesuffix('_logits')}"] = np.asarray(
                data[key], dtype=np.float32
            )
    return test_ids, train_outputs, test_outputs


def build_p86(test_ids: np.ndarray) -> dict[str, np.ndarray]:
    train = load(
        PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
    )
    test = align(
        test_ids,
        load(PROJECT_DIR / "runs/p46_videomae_large_multiclip_test_v1/complete_features.npz"),
    )
    labels = train["labels"].astype(np.int64)
    normalized_train = train["features"].astype(np.float32)
    normalized_train /= np.maximum(
        np.linalg.norm(normalized_train, axis=-1, keepdims=True), 1e-8
    )
    normalized_test = test["features"].astype(np.float32)
    normalized_test /= np.maximum(
        np.linalg.norm(normalized_test, axis=-1, keepdims=True), 1e-8
    )
    model = make_model(FROZEN_ALPHA)
    model.fit(
        normalized_train.reshape(len(normalized_train), -1),
        labels,
        ridge__sample_weight=p85_sample_weights(labels, FROZEN_CLASS_WEIGHT_POWER),
    )
    perturbations = fixed_model_perturbations(
        normalized_test, normalized_train.mean(axis=0)
    )
    return {
        f"p86_mechanism_{name}_logits": aligned_scores_40(
            model, values.reshape(len(values), -1)
        ).astype(np.float32)
        for name, values in perturbations.items()
    }


def distill_missing_p46(
    train_outputs: dict[str, np.ndarray], test_outputs: dict[str, np.ndarray]
) -> dict[str, dict]:
    sources = [
        name
        for name in (*P46_DIRECT_RUNS, *(f"p46_mc_{key.removesuffix('_logits')}" for key in P46_MC_KEYS))
        if name in train_outputs and name in test_outputs
    ]
    train_x = np.concatenate([train_outputs[name] for name in sources], axis=1)
    test_x = np.concatenate([test_outputs[name] for name in sources], axis=1)
    diagnostics: dict[str, dict] = {}
    for target_run in ("p46_70_subject_calibrated_v2", "p46_validation70_final_v1"):
        with np.load(PROJECT_DIR / "runs" / target_run / "crossfit_logits.npz") as data:
            target = np.asarray(data["logits"], dtype=np.float32)
        model = Pipeline(
            (("scale", StandardScaler()), ("ridge", Ridge(alpha=100.0)))
        )
        model.fit(train_x, target)
        fitted = model.predict(train_x)
        test_outputs[target_run] = model.predict(test_x).astype(np.float32)
        diagnostics[target_run] = {
            "input_experts": sources,
            "training_rmse": float(np.sqrt(np.mean((fitted - target) ** 2))),
            "argmax_agreement": float(np.mean(fitted.argmax(axis=1) == target.argmax(axis=1))),
        }
        train_outputs[target_run] = target
    return diagnostics


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    test_ids, train_outputs, test_outputs = build_p46()
    distillation = distill_missing_p46(train_outputs, test_outputs)
    p86 = build_p86(test_ids)
    p46_payload = {"sample_ids": test_ids, **test_outputs}
    p86_payload = {"sample_ids": test_ids, **p86}
    np.savez_compressed(OUTPUT / "p46_test_logits.npz", **p46_payload)
    np.savez_compressed(OUTPUT / "p86_mechanism_test_logits.npz", **p86_payload)
    report = {
        "stage": "P89_multiexpert_Test_logits_v1",
        "status": "complete",
        "test_rows": int(len(test_ids)),
        "p46_experts": sorted(test_outputs),
        "p86_mechanisms": sorted(p86),
        "distilled_experts": distillation,
        "notes": (
            "Direct P46 linear heads are refit on all 1384 Detail21 rows with frozen "
            "CV hyperparameters. Relation/bagging/SVM use their frozen deployment "
            "models. Two meta-experts without a complete direct Test graph are "
            "distilled from the 23 deployable P46 OOF experts."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
