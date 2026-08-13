from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Mapping

import numpy as np


_ROLES = {"oof_train14", "heldout", "competition_test"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ExpertEvidence:
    role: str
    expert_id: str
    sample_ids: np.ndarray
    user_ids: np.ndarray
    logits: np.ndarray
    availability: np.ndarray
    quality: np.ndarray
    quality_mask: np.ndarray
    fusion_quality_score: np.ndarray
    class_map_hash: str
    model_sha256: str
    config_sha256: str
    deployed_weight_bytes: int
    preprocessing_dependencies: tuple[str, ...]
    quality_mapping: str
    quality_mapping_sha256: str
    labels: np.ndarray | None = None
    embeddings: np.ndarray | None = None
    engineered_summary: np.ndarray | None = None
    diagnostics: Mapping[str, np.ndarray] = field(default_factory=dict)

    def validate(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"Unknown evidence role: {self.role}")
        if not self.expert_id:
            raise ValueError("expert_id must be non-empty")
        sample_ids = np.asarray(self.sample_ids).astype(str)
        rows = len(sample_ids)
        if rows == 0 or np.any(np.char.strip(sample_ids) == ""):
            raise ValueError("Evidence contains unknown sample IDs")
        if len(np.unique(sample_ids)) != rows:
            raise ValueError("Evidence contains duplicate sample IDs")
        self._require_rows("user_ids", self.user_ids, rows)
        if np.any(np.char.strip(np.asarray(self.user_ids).astype(str)) == ""):
            raise ValueError("Evidence contains unknown user IDs")
        if np.asarray(self.logits).shape != (rows, 40):
            raise ValueError("Evidence logits must have shape [N,40]")
        if not np.isfinite(self.logits).all():
            raise ValueError("Evidence logits must be finite")
        if np.asarray(self.availability).shape != (rows, 1):
            raise ValueError("Evidence availability must have shape [N,1]")
        quality = np.asarray(self.quality)
        if quality.ndim != 2 or quality.shape[0] != rows or not np.isfinite(quality).all():
            raise ValueError("Evidence quality must be finite [N,Q]")
        if np.asarray(self.quality_mask).shape != quality.shape:
            raise ValueError("Evidence quality_mask must match quality")
        fusion_quality = np.asarray(self.fusion_quality_score)
        if fusion_quality.shape != (rows, 1) or not np.isfinite(fusion_quality).all():
            raise ValueError("Evidence fusion quality must be finite [N,1]")
        if np.any((fusion_quality < 0) | (fusion_quality > 1)):
            raise ValueError("Evidence fusion quality must lie in [0,1]")
        self._validate_sha("class_map_hash", self.class_map_hash)
        self._validate_sha("model_sha256", self.model_sha256)
        self._validate_sha("config_sha256", self.config_sha256)
        self._validate_sha("quality_mapping_sha256", self.quality_mapping_sha256)
        if self.deployed_weight_bytes <= 0:
            raise ValueError("deployed_weight_bytes must be positive")
        if not self.preprocessing_dependencies or any(
            not value for value in self.preprocessing_dependencies
        ):
            raise ValueError("preprocessing_dependencies must be non-empty")
        if not self.quality_mapping:
            raise ValueError("quality_mapping must be non-empty")
        if self.role == "oof_train14" and self.labels is None:
            raise ValueError("labels required for oof_train14 evidence")
        if self.role in {"heldout", "competition_test"} and self.labels is not None:
            raise ValueError(f"labels forbidden for {self.role} evidence")
        if self.labels is not None:
            labels = np.asarray(self.labels)
            if labels.shape != (rows,) or np.any((labels < 0) | (labels >= 40)):
                raise ValueError("Evidence labels must be [N] in [0,39]")
        for name, value in (
            ("embeddings", self.embeddings),
            ("engineered_summary", self.engineered_summary),
        ):
            if value is not None:
                array = np.asarray(value)
                if array.ndim != 2 or array.shape[0] != rows or not np.isfinite(array).all():
                    raise ValueError(f"Evidence {name} must be finite [N,D]")
        for name, value in self.diagnostics.items():
            self._require_rows(f"diagnostic {name}", value, rows)
            if np.asarray(value).dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError(f"Evidence diagnostic {name} must be finite")

    def save(self, path: Path) -> None:
        self.validate()
        arrays: dict[str, np.ndarray] = {
            "role": np.asarray(self.role),
            "expert_id": np.asarray(self.expert_id),
            "sample_ids": np.asarray(self.sample_ids, dtype=np.str_),
            "user_ids": np.asarray(self.user_ids, dtype=np.str_),
            "logits": np.asarray(self.logits, dtype=np.float32),
            "availability": np.asarray(self.availability, dtype=bool),
            "quality": np.asarray(self.quality, dtype=np.float32),
            "quality_mask": np.asarray(self.quality_mask, dtype=bool),
            "fusion_quality_score": np.asarray(self.fusion_quality_score, dtype=np.float32),
            "class_map_hash": np.asarray(self.class_map_hash),
            "model_sha256": np.asarray(self.model_sha256),
            "config_sha256": np.asarray(self.config_sha256),
            "deployed_weight_bytes": np.asarray(self.deployed_weight_bytes, dtype=np.int64),
            "preprocessing_dependencies": np.asarray(self.preprocessing_dependencies, dtype=np.str_),
            "quality_mapping": np.asarray(self.quality_mapping),
            "quality_mapping_sha256": np.asarray(self.quality_mapping_sha256),
        }
        if self.labels is not None:
            arrays["labels"] = np.asarray(self.labels, dtype=np.int64)
        if self.embeddings is not None:
            arrays["embeddings"] = np.asarray(self.embeddings, dtype=np.float32)
        if self.engineered_summary is not None:
            arrays["engineered_summary"] = np.asarray(
                self.engineered_summary, dtype=np.float32
            )
        for name, value in self.diagnostics.items():
            arrays[f"diagnostic__{name}"] = np.asarray(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> ExpertEvidence:
        with np.load(path, allow_pickle=False) as archive:
            diagnostics = {
                key.removeprefix("diagnostic__"): archive[key]
                for key in archive.files
                if key.startswith("diagnostic__")
            }
            evidence = cls(
                role=str(archive["role"].item()),
                expert_id=str(archive["expert_id"].item()),
                sample_ids=archive["sample_ids"],
                user_ids=archive["user_ids"],
                logits=archive["logits"],
                availability=archive["availability"],
                quality=archive["quality"],
                quality_mask=archive["quality_mask"],
                fusion_quality_score=archive["fusion_quality_score"],
                class_map_hash=str(archive["class_map_hash"].item()),
                model_sha256=str(archive["model_sha256"].item()),
                config_sha256=str(archive["config_sha256"].item()),
                deployed_weight_bytes=int(archive["deployed_weight_bytes"].item()),
                preprocessing_dependencies=tuple(
                    archive["preprocessing_dependencies"].astype(str).tolist()
                ),
                quality_mapping=str(archive["quality_mapping"].item()),
                quality_mapping_sha256=str(archive["quality_mapping_sha256"].item()),
                labels=archive["labels"] if "labels" in archive.files else None,
                embeddings=archive["embeddings"] if "embeddings" in archive.files else None,
                engineered_summary=(
                    archive["engineered_summary"]
                    if "engineered_summary" in archive.files
                    else None
                ),
                diagnostics=diagnostics,
            )
        evidence.validate()
        return evidence

    @staticmethod
    def _require_rows(name: str, value: np.ndarray, rows: int) -> None:
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != rows:
            raise ValueError(f"Evidence {name} rows do not match sample IDs")

    @staticmethod
    def _validate_sha(name: str, value: str) -> None:
        if not _SHA256.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
