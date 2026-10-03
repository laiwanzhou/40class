"""A sealed experiment protocol; loading it never reads participant labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .no_vote_types import Partition, canonical_bytes, canonical_hash, freeze
from .teammate_source import SOURCE_FILE_COUNT, SOURCE_MANIFEST_SHA256, verify_teammate_source


TRAIN_USERS = frozenset(("user1","user2","user3","user5","user8","user9","user16","user18","user19","user20","user21","user22"))
DEV_USERS = frozenset(("user6","user7"))
FINAL_USERS = frozenset(("user4","user17","user23","user24"))
PARTITIONS = ("train12", "development2", "refit14", "final4")
FORMAL_ROWS = {"train12":2039,"development2":388,"refit14":2427,"final4":609}


@dataclass(frozen=True)
class NoVoteProtocol:
    run_id: str
    run_root: Path
    source_root: Path
    partitions: Mapping[str, Partition]
    public_manifests: Mapping[str, Path]
    supervised_labels: Mapping[str, Path]
    weights: Mapping[str, Path]
    recipe: Mapping[str, object]

    def __post_init__(self):
        canonical_bytes(self)
        for name in ("partitions", "public_manifests", "supervised_labels", "weights", "recipe"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def identity(self) -> str:
        if not self.recipe.get("asset_bindings", {}).get("verified", False):
            raise ValueError("protocol assets have not been verified; acquire weights first")
        return canonical_hash(self)


def _path(base: Path, value) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("configured paths must be nonempty strings")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _reject_final_label_config(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_") if isinstance(key, str) else ""
            if normalized.startswith("final_label") or normalized in {"final_truth", "test_labels", "private_output", "canonical_labeled_manifest"}:
                raise ValueError("final label locations are forbidden in generation configuration")
            _reject_final_label_config(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _reject_final_label_config(item)


def read_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("expected protocol schema_version 1")
    canonical_bytes(raw)
    _reject_final_label_config(raw)
    base = _path(path.parent, raw.get("project_root", "."))
    return raw, base


def load_protocol(config: str | Path, *, verify_assets: bool = True,
                  source_root: Path | None = None) -> NoVoteProtocol:
    raw, base = read_config(config)
    kind = raw.get("execution_kind")
    if kind not in {"formal", "fixture"}:
        raise ValueError("execution_kind must explicitly be formal or fixture")
    if not isinstance(raw.get("run_id"), str) or not raw["run_id"]:
        raise ValueError("run_id is required")
    parts_raw = raw.get("partitions", {})
    if set(parts_raw) != set(PARTITIONS):
        raise ValueError("all four partitions must be declared")
    parts = {}
    for name in PARTITIONS:
        entry = parts_raw[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("users"), list):
            raise ValueError("partition users must be an explicit list")
        parts[name] = Partition(name, tuple(entry["users"]), entry.get("expected_rows"))
    train, dev, refit, final = (set(parts[n].users) for n in PARTITIONS)
    if train & dev or train & final or dev & final or refit != train | dev:
        raise ValueError("partition overlap or refit ownership mismatch")
    if kind == "formal":
        if train != TRAIN_USERS or dev != DEV_USERS or final != FINAL_USERS:
            raise ValueError("formal user ownership must match the approved fixed split")
        if any(parts[n].expected_rows != FORMAL_ROWS[n] for n in PARTITIONS):
            raise ValueError("formal row counts must match the canonical population")
    if set(raw.get("supervised_labels", {})) != set(PARTITIONS[:-1]):
        raise ValueError("supervised labels may only name train12/development2/refit14")
    if set(raw.get("public_manifests", {})) != set(PARTITIONS):
        raise ValueError("all four public manifests must be declared")
    if set(raw.get("weights", {})) != {"videomae", "mc3", "yolo"}:
        raise ValueError("exactly three public initializers must be declared")
    recipe = raw.get("recipe")
    if not isinstance(recipe, dict) or recipe.get("class_ids") != list(range(40)):
        raise ValueError("recipe must fix class column order to 0..39")
    root = _path(base, raw["run_root"])
    source = raw.get("source", {})
    count, pinned_sha = source.get("expected_files"), source.get("expected_sha256")
    if kind == "formal" and (count != SOURCE_FILE_COUNT or pinned_sha != SOURCE_MANIFEST_SHA256):
        raise ValueError("formal source snapshot pin changed")
    source_path = Path(source_root).resolve() if source_root else _path(base, source["root"])
    manifest_path = source_path.parent / "source_manifest.json" if source_root else _path(base, source["manifest"])
    specs = {name: dict(entry, local_path=str(_path(base, entry["local_path"])))
             for name, entry in raw["weights"].items()}
    recipe = dict(recipe)
    recipe["execution_kind"] = kind
    recipe["weight_specs"] = specs
    recipe["minimum_free_gib"] = raw.get("minimum_free_gib", 20)
    bindings = {"verified": False, "source_manifest_sha256": pinned_sha,
                "weights_manifest_sha256": None}
    if verify_assets:
        from .no_vote_weights import verify_weights_manifest
        report = verify_teammate_source(source_path, manifest_path,
                                        expected_count=count, expected_sha256=pinned_sha, verify_contents=False)
        weights_digest = verify_weights_manifest(root / "protocol/weights_manifest.json", specs,
                                                allow_fixture=(kind == "fixture"), verify_contents=False)
        bindings = {"verified": True, "source_manifest_sha256": report.manifest_sha256,
                    "weights_manifest_sha256": weights_digest}
    recipe["asset_bindings"] = bindings
    return NoVoteProtocol(raw["run_id"], root, source_path, parts,
                          {k:_path(base,v) for k,v in raw["public_manifests"].items()},
                          {k:_path(base,v) for k,v in raw["supervised_labels"].items()},
                          {k:Path(v["local_path"]) for k,v in specs.items()}, recipe)
