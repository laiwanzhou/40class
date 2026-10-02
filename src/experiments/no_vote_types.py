"""Shared, label-independent contracts for the fixed-split experiment."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np


def canonical_data(value):
    if is_dataclass(value):
        return {f.name: canonical_data(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        return os.path.normcase(str(value.resolve())).replace("\\", "/")
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("protocol keys must be strings")
        return {k: canonical_data(value[k]) for k in sorted(value)}
    if isinstance(value, (set, frozenset)):
        items = [canonical_data(v) for v in value]
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    if isinstance(value, (tuple, list)):
        return [canonical_data(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"unsupported or non-finite protocol value: {type(value).__name__}")


def canonical_bytes(value) -> bytes:
    return json.dumps(canonical_data(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_hash(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(v) for v in value)
    return value


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".building")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Partition:
    name: Literal["train12", "development2", "refit14", "final4"]
    users: tuple[str, ...]
    expected_rows: int

    def __post_init__(self):
        users = tuple(self.users)
        if not users or any(not isinstance(u, str) or not u for u in users) or len(set(users)) != len(users):
            raise ValueError("partition users must be nonempty and unique")
        if type(self.expected_rows) is not int or self.expected_rows < 1:
            raise ValueError("expected_rows must be a positive integer")
        object.__setattr__(self, "users", tuple(sorted(users)))


@dataclass(frozen=True)
class ArtifactRef:
    record_path: Path
    sha256: str

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise ValueError("artifact sha256 must contain 64 hex digits")
        object.__setattr__(self, "record_path", Path(self.record_path).resolve())
        object.__setattr__(self, "sha256", self.sha256.lower())


@dataclass(frozen=True)
class RowIndex:
    sample_ids: tuple[str, ...]
    user_ids: tuple[str, ...]
    class_ids: tuple[int, ...]

    def __post_init__(self):
        ids, users, classes = tuple(self.sample_ids), tuple(self.user_ids), tuple(self.class_ids)
        if len(ids) != len(users) or len(set(ids)) != len(ids):
            raise ValueError("sample IDs must be unique and paired with user IDs")
        if any(not isinstance(x, str) or not x for x in (*ids, *users)):
            raise ValueError("sample/user IDs must be nonempty strings")
        if classes != tuple(range(40)):
            raise ValueError("class column order must be exactly 0..39")
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "user_ids", users)
        object.__setattr__(self, "class_ids", classes)


@dataclass(frozen=True)
class Prediction:
    index: RowIndex
    logits: np.ndarray
    probabilities: np.ndarray
    valid: np.ndarray
    artifact: ArtifactRef

    def __post_init__(self):
        shape = (len(self.index.sample_ids), 40)
        if self.logits.shape != shape or self.probabilities.shape != shape:
            raise ValueError("prediction arrays must have shape [N,40]")
        if self.valid.shape != (shape[0],) or self.valid.dtype != np.bool_:
            raise ValueError("prediction valid must be a boolean [N] array")
        if not np.isfinite(self.logits).all() or not np.isfinite(self.probabilities).all():
            raise ValueError("non-finite prediction")
        if np.any(self.probabilities < 0) or np.any(self.probabilities > 1):
            raise ValueError("invalid probability range")
        if not np.allclose(self.probabilities.sum(axis=1), 1., atol=1e-6):
            raise ValueError("probability rows must sum to one; missing rows need a prior")


@dataclass(frozen=True)
class TeacherTargets:
    prediction: Prediction
    features: np.ndarray | None

    def __post_init__(self):
        if self.features is not None:
            if self.features.shape != (len(self.prediction.index.sample_ids), 2, 3, 1024):
                raise ValueError("visual teacher features must be [N,2,3,1024]")
            if not np.isfinite(self.features).all():
                raise ValueError("non-finite visual teacher features")


@dataclass(frozen=True)
class Selection:
    stage: str
    config: Mapping[str, object]
    budget: Mapping[str, int]
    metric: Mapping[str, float]
    fit_artifact: ArtifactRef
    development_ids_sha256: str

    def __post_init__(self):
        canonical_bytes(self)
        for name in ("config", "budget", "metric"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def write(self, path: Path) -> None:
        write_json(path, self)


@dataclass(frozen=True)
class StageInputs:
    partition: Partition
    public_manifest: Path
    labels: Path | None
    parents: Mapping[str, ArtifactRef]


@dataclass(frozen=True)
class AdaptationInputs:
    rows: RowIndex
    base: ArtifactRef
    targets: ArtifactRef
    pixels: ArtifactRef
    sequence: ArtifactRef
    motion: ArtifactRef
    normalization: ArtifactRef
