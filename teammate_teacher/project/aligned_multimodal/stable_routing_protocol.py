"""Small, dependency-light guards for leakage-safe multimodal routing experiments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class ProtocolError(ValueError):
    """Raised when an experiment violates a stability/provenance contract."""


@dataclass(frozen=True)
class ArtifactNode:
    node_id: str
    parents: tuple[str, ...] = ()
    # Subjects used as labels during supervised training at this node.
    supervised_train_subjects: frozenset[str] = frozenset()
    # frozen_external is explicitly allowed only when it carries no task labels.
    provenance: str = "derived"
    has_task_labels: bool = False
    oof_labels_only: bool = False


def assert_prediction_provenance(
    prediction_subject: str,
    prediction_artifacts: Sequence[str],
    nodes: Mapping[str, ArtifactNode],
) -> None:
    """Reject unknown/malformed/cyclic ancestry and train-subject leakage.

    ``oof_labels_only`` never counts as evidence of independence: OOF nodes are
    treated as unknown unless they also have an explicit, complete provenance DAG.
    """
    if not prediction_subject or not prediction_artifacts:
        raise ProtocolError("prediction subject and at least one artifact are required")
    if not nodes:
        raise ProtocolError("artifact ancestry is unknown")
    visiting: set[str] = set()
    visited: set[str] = set()
    train_subjects: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id not in nodes:
            raise ProtocolError(f"missing artifact parent/node: {node_id}")
        if node_id in visiting:
            raise ProtocolError(f"cycle in artifact ancestry at {node_id}")
        if node_id in visited:
            return
        node = nodes[node_id]
        if node.node_id != node_id:
            raise ProtocolError(f"artifact key/node_id mismatch: {node_id}")
        if node.oof_labels_only:
            raise ProtocolError(f"OOF labels are not provenance proof: {node_id}")
        allowed = {"frozen_external", "raw_input", "supervised", "derived", "unknown"}
        if node.provenance not in allowed or node.provenance == "unknown":
            raise ProtocolError(f"unknown artifact ancestry: {node_id}")
        if node.provenance == "frozen_external" and (node.has_task_labels or node.supervised_train_subjects):
            raise ProtocolError(f"frozen_external contains task labels/training subjects: {node_id}")
        if node.provenance == "raw_input" and (node.has_task_labels or node.supervised_train_subjects):
            raise ProtocolError(f"raw_input contains task labels/training subjects: {node_id}")
        if node.has_task_labels and not node.supervised_train_subjects:
            raise ProtocolError(f"labeled artifact lacks explicit training subjects: {node_id}")
        if node.provenance == "supervised" and not node.supervised_train_subjects:
            raise ProtocolError(f"supervised artifact lacks training subjects: {node_id}")
        if node.provenance == "derived" and (not node.parents and not node.supervised_train_subjects):
            raise ProtocolError(f"unsubstantiated derived leaf: {node_id}")
        visiting.add(node_id)
        train_subjects.update(node.supervised_train_subjects)
        for parent in node.parents:
            walk(parent)
        visiting.remove(node_id)
        visited.add(node_id)

    for artifact in prediction_artifacts:
        walk(artifact)
    if prediction_subject in train_subjects:
        raise ProtocolError("prediction subject intersects supervised training ancestry")


@dataclass(frozen=True)
class ExpertContract:
    expert_id: str
    train_model_hash: str
    test_model_hash: str
    train_preprocess_hash: str
    test_preprocess_hash: str
    train_class_order: tuple[str, ...]
    test_class_order: tuple[str, ...]
    train_feature_semantics: str
    test_feature_semantics: str
    # Model hashes identify the fixed encoder; OOF/refit weights may differ.
    train_head_recipe_hash: str | None = None
    test_head_recipe_hash: str | None = None
    evidence_id: str = ""
    alias_of: str | None = None
    fallback: bool = False


def validate_expert_contracts(contracts: Sequence[ExpertContract]) -> None:
    """Validate compatible inputs and provenance labels, not statistical independence."""
    if not contracts:
        raise ProtocolError("at least one expert contract is required")
    evidence: set[str] = set()
    unheaded_fingerprints: set[tuple[Any, ...]] = set()
    for c in contracts:
        hex64 = lambda x: isinstance(x, str) and len(x) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in x)
        if not hex64(c.train_model_hash) or not hex64(c.test_model_hash) or not hex64(c.train_preprocess_hash) or not hex64(c.test_preprocess_hash):
            raise ProtocolError(f"hash must be 64 hexadecimal characters: {c.expert_id!r}")
        values = (c.expert_id, c.train_model_hash, c.test_model_hash,
                  c.train_preprocess_hash, c.test_preprocess_hash,
                  c.train_feature_semantics, c.test_feature_semantics, c.evidence_id)
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ProtocolError(f"incomplete contract: {c.expert_id!r}")
        if not c.train_class_order or not c.test_class_order or any(x in ("", "unknown") for x in c.train_class_order + c.test_class_order):
            raise ProtocolError(f"empty class order: {c.expert_id}")
        if len(set(c.train_class_order)) != len(c.train_class_order) or len(set(c.test_class_order)) != len(c.test_class_order):
            raise ProtocolError(f"class order is not unique: {c.expert_id}")
        if c.train_model_hash != c.test_model_hash:
            raise ProtocolError(f"model hash mismatch: {c.expert_id}")
        if c.train_preprocess_hash != c.test_preprocess_hash:
            raise ProtocolError(f"preprocess hash mismatch: {c.expert_id}")
        if c.train_class_order != c.test_class_order:
            raise ProtocolError(f"class order mismatch: {c.expert_id}")
        if c.train_feature_semantics != c.test_feature_semantics:
            raise ProtocolError(f"feature semantics mismatch: {c.expert_id}")
        if (c.train_head_recipe_hash is None) != (c.test_head_recipe_hash is None):
            raise ProtocolError(f"head recipe must be specified on both sides: {c.expert_id}")
        if c.train_head_recipe_hash is not None and (not hex64(c.train_head_recipe_hash) or c.train_head_recipe_hash != c.test_head_recipe_hash):
            raise ProtocolError(f"head recipe mismatch: {c.expert_id}")
        if c.fallback or c.alias_of:
            raise ProtocolError(f"fallback/alias cannot be independent evidence: {c.expert_id}")
        if not c.evidence_id.strip() or c.evidence_id.strip().lower() == "unknown":
            raise ProtocolError(f"evidence_id must be explicit and known: {c.expert_id}")
        if c.train_head_recipe_hash is None:
            fingerprint = (c.train_model_hash, c.train_preprocess_hash,
                           c.train_class_order, c.train_feature_semantics)
            if fingerprint in unheaded_fingerprints:
                raise ProtocolError(f"duplicate unheaded source fingerprint: {c.expert_id}")
            unheaded_fingerprints.add(fingerprint)
        key = c.evidence_id
        if key in evidence:
            raise ProtocolError(f"duplicate independent evidence: {key}")
        evidence.add(key)


@dataclass(frozen=True)
class BootstrapResult:
    point_delta: float
    lower: float
    upper: float
    confidence: float
    n_subjects: int
    n_bootstrap: int
    weighting: str


def paired_subject_bootstrap(
    subject_ids: Sequence[str], y_true: Sequence[Any], pred_a: Sequence[Any], pred_b: Sequence[Any],
    *, weights: Sequence[float] | None = None, n_bootstrap: int = 2000,
    confidence: float = 0.95, seed: int = 0, weighting: str = "sample_weighted",
) -> BootstrapResult:
    if any(x is None for x in (subject_ids, y_true, pred_a, pred_b)):
        raise ProtocolError("bootstrap inputs cannot be None")
    sid, yt, a, b = map(np.asarray, (subject_ids, y_true, pred_a, pred_b))
    if any(x.ndim != 1 for x in (sid, yt, a, b)):
        raise ProtocolError("bootstrap inputs must be one-dimensional")
    n = len(sid)
    if n == 0 or any(len(x) != n for x in (yt, a, b)):
        raise ProtocolError("non-empty equal-length bootstrap inputs required")
    if weighting not in ("sample_weighted", "subject_mean"):
        raise ProtocolError("weighting must be sample_weighted or subject_mean")
    if not (0 < confidence < 1) or n_bootstrap < 1:
        raise ProtocolError("invalid confidence or bootstrap count")
    if np.any(np.asarray([x is None or (isinstance(x, float) and not np.isfinite(x)) or str(x).strip() in ("", "unknown", "nan") for x in sid], dtype=bool)):
        raise ProtocolError("subject IDs must be non-empty")
    for arr in (yt, a, b):
        for value in arr.tolist():
            if value is None:
                raise ProtocolError("labels/predictions must be finite")
            try:
                if isinstance(value, (float, int, np.number)) and not np.isfinite(value):
                    raise ProtocolError("labels/predictions must be finite")
            except TypeError as exc:
                raise ProtocolError("labels/predictions must be finite") from exc
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != n or np.any(~np.isfinite(w)) or np.any(w < 0) or float(w.sum()) <= 0:
        raise ProtocolError("weights must be finite, non-negative, and nonzero")
    groups = {s: np.flatnonzero(sid == s) for s in np.unique(sid)}
    subjects = list(groups)
    subject_index = {s: i for i, s in enumerate(subjects)}
    if len(subjects) < 2:
        raise ProtocolError("at least two subjects are required for a CI")
    group_weight = np.asarray([w[groups[s]].sum() for s in subjects], dtype=float)
    if np.any(group_weight <= 0):
        raise ProtocolError("each subject must have positive total weight")
    gain = np.asarray([np.sum(w[groups[s]] * ((a[groups[s]] == yt[groups[s]]).astype(float) - (b[groups[s]] == yt[groups[s]]).astype(float))) for s in subjects])
    subject_delta = gain / group_weight
    def delta(chosen: Sequence[str]) -> float:
        if weighting == "subject_mean":
            return float(np.mean(subject_delta[[subject_index[s] for s in chosen]]))
        indexes = np.asarray([subject_index[s] for s in chosen], dtype=int)
        return float(gain[indexes].sum() / group_weight[indexes].sum())
    point = delta(subjects)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for j in range(n_bootstrap):
        draws[j] = delta([subjects[k] for k in rng.integers(0, len(subjects), size=len(subjects))])
    alpha = (1 - confidence) / 2
    lower = float(np.quantile(draws, alpha))
    upper = float(np.quantile(draws, 1-alpha))
    if not (-1.0 <= point <= 1.0 and -1.0 <= lower <= upper <= 1.0):
        raise ProtocolError("accuracy difference outside valid range; aggregation error")
    return BootstrapResult(point, lower, upper, confidence, len(subjects), n_bootstrap, weighting)


def canonical_experiment_json(spec: Mapping[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def experiment_sha256(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_experiment_json(spec).encode("utf-8")).hexdigest()


def register_experiment(registry_path: str | Path, spec: Mapping[str, Any], output_paths: Sequence[str | Path] = ()) -> str:
    """Write a canonical registration; never overwrite an existing output."""
    for output in output_paths:
        if Path(output).exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
    path = Path(registry_path)
    digest = experiment_sha256(spec)
    if path.exists():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ProtocolError("invalid existing experiment registry") from exc
        try:
            stored_spec = prior["spec"]
            recomputed = experiment_sha256(stored_spec)
        except Exception as exc:
            raise ProtocolError("invalid existing experiment registry") from exc
        if prior.get("sha256") != recomputed or recomputed != digest:
            raise FileExistsError("experiment registry already contains a different experiment")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sha256": digest, "spec": json.loads(canonical_experiment_json(spec))}
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    except FileExistsError:
        raise FileExistsError("experiment registry was created concurrently")
    return digest
