"""P416: a small, nested, frozen-family router pilot.

This module deliberately contains no P307/P310/P399 dependency.  The only
inputs used by the real runner are the P90 manifest/folds and the four raw
feature caches named by P238.PATHS.  The public small functions are also used
by the tiny synthetic tests, so protocol mechanics can be checked without
loading the (large) caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .p90_teacher_common import NUM_CLASSES, load_protocol, softmax
from .stable_routing_protocol import (
    ArtifactNode, ProtocolError, assert_prediction_provenance,
    paired_subject_bootstrap, register_experiment,
)

FAMILY_NAMES = ("ir_vmae", "ir_iv2", "depth_vmae", "thermal_vmae")
# Kept byte-for-byte equivalent to P238.PATHS, without importing P238's
# script (which intentionally assumes execution from its own directory).
_ROOT = Path(__file__).resolve().parent.parent
P238_PATHS = (
    _ROOT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz",
    _ROOT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz",
    _ROOT / "runs/p91_videomaev2_depth_fold0_v1/complete_features.npz",
    _ROOT / "runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz",
)
EXCLUDED_USERS = frozenset(("user1", "user2", "user21"))
OUTER_FOLDS = (0, 1, 2)
TEMPERATURE = 1.0
GATE = 0.2
SEED = 20260907


def _as_feature_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < 2:
        raise ValueError("family features must have a sample axis and feature axes")
    if not np.isfinite(x).all():
        raise ProtocolError("nonfinite raw feature cache")
    # token-wise L2, then flatten: this is the sole family transformation.
    if x.ndim >= 3:
        denom = np.linalg.norm(x, axis=-1, keepdims=True)
        x = x / np.maximum(denom, 1e-12)
    return x.reshape(x.shape[0], -1)


def load_frozen_families(protocol: Any | None = None):
    """Load exactly P238's four raw caches and return the P416 subset."""
    protocol = load_protocol() if protocol is None else protocol
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if int(keep.sum()) != 2470 and len(protocol.labels) == 2914:
        raise ProtocolError(f"expected 2470 eligible manifest rows, got {int(keep.sum())}")
    families = []
    for path in P238_PATHS:
        with np.load(path, allow_pickle=False) as z:
            if "features" not in z or "sample_ids" not in z:
                raise ProtocolError(f"raw cache lacks features/sample_ids: {path}")
            if not np.array_equal(z["sample_ids"].astype(str), protocol.sample_ids):
                raise ProtocolError(f"P238 cache order mismatch: {path}")
            families.append(_as_feature_matrix(z["features"])[keep])
    return families, protocol.labels[keep], protocol.users[keep], protocol.fold_id[keep], protocol.sample_ids[keep]


def _assert_fit_disjoint(train_subjects: Sequence[str], held_subjects: Sequence[str]) -> None:
    overlap = set(map(str, train_subjects)) & set(map(str, held_subjects))
    if overlap:
        raise ProtocolError(f"supervised fit intersects held subjects: {sorted(overlap)[:3]}")


def fit_family_head(features: np.ndarray, labels: np.ndarray, train_idx: Sequence[int],
                    held_idx: Sequence[int], subjects: Sequence[str]) -> tuple[Any, Any, np.ndarray]:
    """Fit scaler + Ridge head and return probabilities aligned to 40 classes."""
    features = _as_feature_matrix(features)
    train_idx, held_idx = np.asarray(train_idx, int), np.asarray(held_idx, int)
    _assert_fit_disjoint(np.asarray(subjects)[train_idx], np.asarray(subjects)[held_idx])
    scaler = StandardScaler().fit(features[train_idx])
    y = np.asarray(labels)[train_idx]
    clf = RidgeClassifier(alpha=3000, solver="lsqr", tol=1e-5, max_iter=2000).fit(scaler.transform(features[train_idx]), y)
    raw = clf.decision_function(scaler.transform(features[held_idx]))
    raw = np.asarray(raw, dtype=float)
    if len(clf.classes_) == 1:
        score = np.full((len(held_idx), NUM_CLASSES), -1e9, dtype=float)
        score[:, int(clf.classes_[0])] = 0.0
        return scaler, clf, softmax(score / TEMPERATURE)
    if raw.ndim == 1:  # binary Ridge: map signed score to [-raw,+raw].
        raw = np.column_stack((-raw, raw))
    score = np.full((len(held_idx), NUM_CLASSES), -1e9, dtype=float)
    score[:, np.asarray(clf.classes_, dtype=int)] = raw
    return scaler, clf, softmax(score / TEMPERATURE)


def inner_oof_bank(families: Sequence[np.ndarray], labels: np.ndarray, subjects: Sequence[str],
                   outer_train: Sequence[int], n_splits: int = 3, fit_log: list | None = None) -> np.ndarray:
    """Generate a label-honest inner OOF bank (rows are outer-train rows)."""
    outer_train = np.asarray(outer_train, int)
    local_groups = np.asarray(subjects)[outer_train]
    if len(np.unique(local_groups)) < n_splits:
        raise ProtocolError("not enough subjects for inner GroupKFold")
    bank = np.zeros((len(outer_train), len(families), NUM_CLASSES), dtype=np.float64)
    for inner, (tr_local, va_local) in enumerate(GroupKFold(n_splits=n_splits).split(outer_train, labels[outer_train], local_groups)):
        tr, va = outer_train[tr_local], outer_train[va_local]
        for j, x in enumerate(families):
            started = time.monotonic()
            _, _, bank[va_local, j] = fit_family_head(x, labels, tr, va, subjects)
            if fit_log is not None:
                record = fit_record(f"inner{inner}.family{j}", tr, va, subjects, j)
                record["seconds"] = time.monotonic() - started
                fit_log.append(record)
                print(json.dumps({"event": "inner_head_complete", "inner": inner, "family": j,
                                  "seconds": record["seconds"]}), flush=True)
    return bank


def fit_record(name, train, held, subjects, family):
    users = np.asarray(subjects).astype(str)
    _assert_fit_disjoint(users[train], users[held])
    return {"node_id": name, "family": family, "train_indices": train.tolist(),
            "validation_indices": held.tolist(), "train_subjects": sorted(set(users[train])),
            "validation_subjects": sorted(set(users[held])),
            "preprocessing_fit_subjects": sorted(set(users[train]))}


def run_outer_fold(families: Sequence[np.ndarray], labels: np.ndarray, subjects: Sequence[str],
                   fold_id: Sequence[int], fold: int) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Run one complete nested outer fold; useful for audit and unit tests."""
    folds = np.asarray(fold_id); held = np.flatnonzero(folds == fold); train = np.flatnonzero(folds != fold)
    _assert_fit_disjoint(np.asarray(subjects)[train], np.asarray(subjects)[held])
    if not len(train) or not len(held):
        raise ProtocolError("outer split must contain both training and held rows")
    fit_log = []
    bank = inner_oof_bank(families, labels, subjects, train, fit_log=fit_log)
    scaler, router = fit_router_scorer(bank, np.asarray(labels)[train])
    outer_prob = np.zeros((len(held), len(families), NUM_CLASSES), dtype=float)
    nodes = {f"external.family{j}": ArtifactNode(f"external.family{j}", provenance="frozen_external")
             for j in range(len(families))}
    inner_nodes = []
    for record in fit_log:
        name = record["node_id"]
        nodes[name] = ArtifactNode(name, parents=(f"external.family{record['family']}",),
                                   provenance="supervised", has_task_labels=True,
                                   supervised_train_subjects=frozenset(record["train_subjects"]))
        inner_nodes.append(name)
        for subject in record["validation_subjects"]:
            assert_prediction_provenance(subject, [name], nodes)
    router_id = f"router.fold{fold}"
    nodes[router_id] = ArtifactNode(router_id, parents=tuple(inner_nodes), provenance="supervised",
                                    has_task_labels=True,
                                    supervised_train_subjects=frozenset(map(str, np.asarray(subjects)[train])))
    outer_nodes = []
    for j, x in enumerate(families):
        started = time.monotonic()
        _, _, outer_prob[:, j] = fit_family_head(x, labels, train, held, subjects)
        node_id = f"p416.outer.fold{fold}.family{j}.head"
        nodes[node_id] = ArtifactNode(node_id, parents=(f"external.family{j}",), provenance="supervised",
                                      has_task_labels=True,
                                      supervised_train_subjects=frozenset(map(str, np.asarray(subjects)[train])))
        outer_nodes.append(node_id)
        record = fit_record(node_id, train, held, subjects, j)
        record["seconds"] = time.monotonic() - started
        fit_log.append(record)
        print(json.dumps({"event": "outer_head_complete", "fold": fold, "family": j,
                          "seconds": record["seconds"]}), flush=True)
        pred_id = node_id + ".prediction"; nodes[pred_id] = ArtifactNode(pred_id, parents=(node_id,))
        for s in np.unique(np.asarray(subjects)[held]):
            assert_prediction_provenance(str(s), [pred_id], nodes)
    final_id = f"final.fold{fold}"
    nodes[final_id] = ArtifactNode(final_id, parents=(router_id, *outer_nodes))
    for subject in np.unique(np.asarray(subjects)[held]):
        assert_prediction_provenance(str(subject), [final_id], nodes)
    serialized = {name: {"node_id": value.node_id, "parents": list(value.parents),
                         "provenance": value.provenance, "has_task_labels": value.has_task_labels,
                         "supervised_train_subjects": sorted(value.supervised_train_subjects)}
                  for name, value in nodes.items()}
    return held, route(outer_prob, scaler, router), {"outer_train_indices": train.tolist(),
            "outer_held_indices": held.tolist(), "inner_rows": int(len(train)),
            "fits": fit_log, "artifact_dag": serialized, "provenance_checked": True}


def candidate_set(probabilities: np.ndarray) -> list[np.ndarray]:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 3 or p.shape[1:] != (4, NUM_CLASSES):
        raise ValueError("probabilities must be (rows, 4, 40)")
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("probabilities must be finite and non-negative")
    mean = p.mean(axis=1)
    result = []
    for i in range(len(p)):
        order = np.argsort(-mean[i], kind="stable")
        vals = set(order[:5].tolist())
        vals.update(np.argmax(p[i], axis=1).tolist())
        result.append(np.asarray(sorted(vals), dtype=int))
    return result


def scorer_features(probabilities: np.ndarray, candidates: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed candidate features and return (X, candidate labels, row ids)."""
    p = np.asarray(probabilities, dtype=float); mean = p.mean(axis=1)
    base = mean.argmax(axis=1)
    rows = []; ys = []; rid = []
    for i, cs in enumerate(candidates):
        ranks = np.argsort(np.argsort(-p[i], axis=1, kind="stable"), axis=1, kind="stable")
        for c in cs:
            lp = np.log(np.maximum(p[i, :, c], 1e-300))
            br = np.log(np.maximum(p[i, :, base[i]], 1e-300))
            r = ranks[:, c].astype(float)
            # preregistered mean log-ratio is log(mean p_c)-log(mean p_base).
            mean_ratio = np.log(max(float(mean[i, c]), 1e-300)) - np.log(max(float(mean[i, base[i]]), 1e-300))
            feat = np.r_[lp, lp - br, r, mean_ratio, float(c == base[i]),
                         np.eye(NUM_CLASSES, dtype=float)[c], np.eye(NUM_CLASSES, dtype=float)[base[i]]]
            rows.append(feat); ys.append(int(c)); rid.append(i)
    return np.asarray(rows), np.asarray(ys), np.asarray(rid)


def fit_router_scorer(bank: np.ndarray, labels: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    candidates = candidate_set(bank); x, cs, rid = scorer_features(bank, candidates)
    target = (cs == np.asarray(labels)[rid]).astype(int)
    if len(np.unique(target)) < 2:
        raise ProtocolError("router scorer requires both correct and incorrect candidates")
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=.03, l1_ratio=0, solver="lbfgs", max_iter=2000, class_weight="balanced").fit(scaler.transform(x), target)
    return scaler, clf


def route(probabilities: np.ndarray, scaler: StandardScaler, scorer: LogisticRegression) -> dict[str, np.ndarray]:
    p = np.asarray(probabilities, float); mean = p.mean(axis=1); base = mean.argmax(axis=1)
    candidates = candidate_set(p); x, _, rid = scorer_features(p, candidates)
    score = scorer.predict_proba(scaler.transform(x))[:, 1]; chosen = base.copy()
    for i, cs in enumerate(candidates):
        q = np.flatnonzero(rid == i)
        best = q[np.argmax(score[q])]
        if score[best] - score[q[np.flatnonzero(cs == base[i])[0]]] >= GATE:
            chosen[i] = cs[np.argmax(score[q])]
    family_top = np.argmax(p, axis=2)
    unanimous = np.all(family_top == family_top[:, :1], axis=1)
    selective = chosen
    disagreement = np.where(~unanimous, chosen, base)
    return {"equal_mean": base, "disagreement_only": disagreement, "selective_all": selective}


def evaluate_outputs(outputs: Mapping[str, np.ndarray], labels: np.ndarray, subjects: Sequence[str], fold_id: Sequence[int], base_key: str = "equal_mean") -> dict[str, Any]:
    out: dict[str, Any] = {}
    y = np.asarray(labels); sid = np.asarray(subjects); folds = np.asarray(fold_id)
    for name, pred in outputs.items():
        p = np.asarray(pred); correct = p == y; base = outputs[base_key] == y
        rescue = (correct & ~base); harm = (~correct & base); net = rescue.astype(int) - harm.astype(int)
        by_fold = {str(int(f)): {"rescue": int(rescue[folds == f].sum()), "harm": int(harm[folds == f].sum()), "net": int(net[folds == f].sum())} for f in np.unique(folds)}
        by_user = {str(s): int(net[sid == s].sum()) for s in np.unique(sid)}
        ci_subject = paired_subject_bootstrap(sid, y, p, outputs[base_key], n_bootstrap=10000, seed=SEED, confidence=.99, weighting="subject_mean")
        ci_sample = paired_subject_bootstrap(sid, y, p, outputs[base_key], n_bootstrap=10000, seed=SEED, confidence=.99, weighting="sample_weighted")
        per_user = {str(s): {"rescue": int(rescue[sid == s].sum()), "harm": int(harm[sid == s].sum()), "net": int(net[sid == s].sum())} for s in np.unique(sid)}
        criterion = {"net_rate_ge_005": float(net.mean()) >= .005,
                     "threefold_net_ge_0": len(by_fold) == 3 and all(v["net"] >= 0 for v in by_fold.values()),
                     "both_ci_lower_gt_0": ci_subject.lower > 0 and ci_sample.lower > 0}
        out[name] = {"correct": int(correct.sum()), "accuracy": float(correct.mean()), "rows": int(len(y)), "rescue": int(rescue.sum()), "harm": int(harm.sum()), "net": int(net.sum()), "perfold": by_fold, "peruser": per_user, "subject_clustered_99ci": {"point": ci_subject.point_delta, "lower": ci_subject.lower, "upper": ci_subject.upper}, "sample_weighted_99ci": {"point": ci_sample.point_delta, "lower": ci_sample.lower, "upper": ci_sample.upper}, "criterion": criterion, "mechanism_gate_pass": all(criterion.values()), "target_achieved": False}
    return out


def _spec() -> dict[str, Any]:
    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    prereg = _ROOT / "docs/research/STABLE_093_PREREGISTRATION_2026-09-07.md"
    p90 = _ROOT / "aligned_multimodal/p90_teacher_common.py"
    guard = _ROOT / "aligned_multimodal/stable_routing_protocol.py"
    sources = [p90, guard, prereg, _ROOT / "aligned_multimodal/data/manifest.csv",
               *[_ROOT / f"aligned_multimodal/data/subject_folds/fold_{f}.csv" for f in OUTER_FOLDS],
               *P238_PATHS, *[path.parent / "cache_summary.json" for path in P238_PATHS],
               *[_ROOT / "aligned_multimodal" / name for name in (
                   "p90_videomaev2_distilled_teacher.py", "p90_internvideo2_l_teacher.py",
                   "p91_videomaev2_modality_teacher.py")]]
    cache_provenance = []
    for family, path in zip(FAMILY_NAMES, P238_PATHS, strict=True):
        meta = json.loads((path.parent / "cache_summary.json").read_text(encoding="utf-8"))
        checkpoint = meta.get("checkpoint", meta.get("model_file"))
        if not checkpoint or "huggingface" not in checkpoint or "snapshots" not in checkpoint:
            raise ProtocolError(f"missing frozen checkpoint provenance: {family}")
        cache_provenance.append({"family": family, "checkpoint": checkpoint,
            "model_repo": meta.get("model_repo", "OpenGVLab/VideoMAE2"), "clips": meta["clips"],
            "evidence": "cache_summary plus manually reviewed extraction source: external pretrained load_state_dict, eval(), inference_mode(), no task-label update",
            "task_label_supervised_encoder_fit": False,
            "limitation": "Historical extraction attestation, not independent re-extraction of raw data"})
    return {"experiment": "P416_nested_frozen_family_router", "classes": 40, "temperature": 1.0,
            "gate": {"threshold": .2, "semantic": "candidate_score_minus_base_score_probability"},
            "outer_folds": [0,1,2], "inner": "GroupKFold(3)", "families": list(FAMILY_NAMES),
            "excluded_users": sorted(EXCLUDED_USERS), "forbidden_ancestors": ["P307", "P310", "P399"],
            "seed": SEED, "cpu_threads": 4,
            "ridge": {"alpha": 3000, "solver": "lsqr", "tol": 1e-5, "max_iter": 2000},
            "scorer": {"C": .03, "l1_ratio": 0, "solver": "lbfgs", "max_iter": 2000, "class_weight": "balanced"},
            "preprocessing": "token-wise L2 then flatten; StandardScaler per fit",
            "candidates": "mean top5 union four family top1; stable descending argsort",
            "metrics": {"bootstrap": 10000, "confidence": .99, "weightings": ["subject_mean", "sample_weighted"], "criterion": ["net>=.005", "threefold net>=0", "both CI lower>0"]},
            "primary_output": "selective_all", "promotion_allowed": False,
            "cache_provenance": cache_provenance,
            "source_sha256": {str(path.relative_to(_ROOT)): digest(path) for path in sources}}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(); mode = ap.add_mutually_exclusive_group(required=True); mode.add_argument("--pilot", action="store_true"); mode.add_argument("--run", action="store_true"); ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv); out = Path(args.out_dir)
    if out.exists(): raise FileExistsError(f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True, exist_ok=False); os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
    spec = _spec(); spec["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec["execution_mode"] = "pilot" if args.pilot else "run"
    spec["executed_outer_folds"] = [0] if args.pilot else list(OUTER_FOLDS)
    register_experiment(out / "experiment_registry.json", spec)
    # Preserve executable sources so later edits cannot erase the registered implementation.
    (out / "source_snapshot").mkdir()
    for path in (Path(__file__), Path(__file__).with_name("stable_routing_protocol.py"), Path(__file__).with_name("p90_teacher_common.py")):
        (out / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    started = time.time(); p = load_protocol(); fam, y, users, folds, sample_ids = load_frozen_families(p)
    all_outputs = {}; provenance = {"historical_reused_subjects": True, "target_achieved": False, "note": "exploratory mechanism only", "training_subjects": {}, "fold_indices": {}}
    fold_iter = (0,) if args.pilot else OUTER_FOLDS
    for fold in fold_iter:
        held = np.flatnonzero(folds == fold); train = np.flatnonzero(folds != fold); _assert_fit_disjoint(users[train], users[held])
        provenance["training_subjects"][str(fold)] = sorted(set(users[train].astype(str)))
        provenance["fold_indices"][str(fold)] = {"outer_train": train.tolist(), "outer_held": held.tolist()}
        with threadpool_limits(limits=4):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                _, fold_outputs, fold_log = run_outer_fold(fam, y, users, folds, fold)
        print(json.dumps({"event": "outer_fold_complete", "fold": fold, "elapsed_seconds": time.time() - started, "held_rows": len(held)}), flush=True)
        all_outputs[str(fold)] = fold_outputs
        provenance.setdefault("fold_logs", {})[str(fold)] = fold_log
    complete = {name: np.full(len(y), -1, dtype=int) for name in ("equal_mean", "disagreement_only", "selective_all")}
    for fold, vals in all_outputs.items():
        held = np.asarray(provenance["fold_indices"][fold]["outer_held"], dtype=int)
        for name in complete: complete[name][held] = vals[name]
    np.savez_compressed(out / "predictions.npz", sample_ids=sample_ids, users=users, fold_id=folds, **complete)
    payload = {"mode": "pilot" if args.pilot else "run", "elapsed_seconds": time.time()-started, "shapes": {"families": [list(x.shape) for x in fam]}, "provenance": provenance}
    if args.run:
        if any(np.any(values < 0) for values in complete.values()):
            raise ProtocolError("incomplete outer predictions; evaluation is forbidden")
        payload["evaluation"] = evaluate_outputs(complete, y, users, folds)
        secondary = evaluate_outputs({"disagreement_only": complete["disagreement_only"],
                                      "selective_all": complete["selective_all"]},
                                     y, users, folds, base_key="disagreement_only")["selective_all"]
        secondary.pop("criterion")
        secondary.pop("mechanism_gate_pass")
        payload["secondary_selective_all_minus_disagreement_only"] = secondary
        payload["primary_mechanism_gate_pass"] = payload["evaluation"]["selective_all"]["mechanism_gate_pass"]
        payload["target_achieved"] = False
        payload["independent_confirmation"] = False
        payload["inference_warning"] = "Historical subjects and overlapping fitted folds: CIs describe this experiment, not a fresh external confirmation."
    (out / ("pilot.json" if args.pilot else "summary.json")).write_text(json.dumps(payload, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "mode": payload["mode"], "elapsed_seconds": payload["elapsed_seconds"],
                      "output": str(out), "target_achieved": False}), flush=True)

if __name__ == "__main__": main()
