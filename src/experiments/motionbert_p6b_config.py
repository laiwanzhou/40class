from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE_COMMIT = "705d3a95354db8bdb696b3492e47a3b5537174ff"
EXPECTED_WEIGHT_REVISION = "370a9196aa3c89198b134c82476143b01c0fb32c"
EXPECTED_WEIGHT_BYTES = 64_099_897
EXPECTED_WEIGHT_SHA256 = (
    "6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461"
)
EXPECTED_TRAIN_USERS = {
    "user1", "user2", "user3", "user5", "user8", "user9", "user16",
    "user18", "user19", "user20", "user21", "user22",
}
EXPECTED_VALIDATION_USERS = {"user6", "user7"}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_motionbert_p6b_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("MotionBERT P6-B config must be a mapping")
    if config.get("schema_version") != 1 or config.get("stage") != "P6-B":
        raise ValueError("MotionBERT P6-B header changed")
    if config.get("candidate") != "motionbert_lite":
        raise ValueError("MotionBERT P6-B candidate changed")
    if config.get("seed") != 20260715 or config.get("class_ids") != list(range(40)):
        raise ValueError("MotionBERT P6-B seed or classes changed")
    population = config.get("population", {})
    if population.get("train_samples") != 2039 or population.get("validation_samples") != 388:
        raise ValueError("MotionBERT P6-B population changed")
    if set(population.get("train_user_ids", [])) != EXPECTED_TRAIN_USERS:
        raise ValueError("MotionBERT P6-B train users changed")
    if set(population.get("validation_user_ids", [])) != EXPECTED_VALIDATION_USERS:
        raise ValueError("MotionBERT P6-B validation users changed")
    if config.get("input") != {
        "frames": 96,
        "joints": 17,
        "channels": ["projected_x", "projected_y", "confidence"],
        "segment_policy": "longest_retained_then_smallest_index",
    }:
        raise ValueError("MotionBERT P6-B input changed")
    upstream = config.get("upstream", {})
    if (
        upstream.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or upstream.get("license") != "Apache-2.0"
        or upstream.get("weight_revision") != EXPECTED_WEIGHT_REVISION
        or upstream.get("weight_path") != "checkpoint/pretrain/MB_lite/latest_epoch.bin"
    ):
        raise ValueError("MotionBERT upstream provenance changed")
    checkpoint = config.get("checkpoint", {})
    if (
        checkpoint.get("bytes") != EXPECTED_WEIGHT_BYTES
        or checkpoint.get("sha256") != EXPECTED_WEIGHT_SHA256
    ):
        raise ValueError("MotionBERT checkpoint provenance changed")
    model = config.get("model", {})
    expected_model = {
        "dim_in": 3, "dim_feat": 256, "dim_rep": 512, "depth": 5,
        "num_heads": 8, "mlp_ratio": 4, "num_joints": 17, "maxlen": 243,
        "att_fuse": True, "dropout": 0.5,
    }
    if model != expected_model:
        raise ValueError("MotionBERT-Lite architecture changed")
    policy = config.get("policy", {})
    required_false = (
        "grouped_cv_allowed", "multiple_seeds_allowed", "p6a_required",
        "fusion_allowed", "distillation_allowed", "competition_test_allowed",
    )
    if any(policy.get(name) is not False for name in required_false):
        raise ValueError("MotionBERT P6-B authorization changed")
    return config
