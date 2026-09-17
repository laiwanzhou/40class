"""Run-level verifier for the bounded P432 LaViLa pilot.

The verifier derives its expected context from the canonical manifest and the
current frozen cache/probe.  ``expected.json`` is treated as an artifact to
check, never as an authority.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .p90_teacher_common import load_protocol
from .p432_lavila_cache import LaViLaCache
from .p432_lavila_provider import EXCLUDED_USERS, Provider
from .p432_probe_contract import validate_spotcheck
from .p432_rebuild_lavila_bank import (
    CANONICAL, CACHE, PIXEL_ROWS, PROBE, PREREG, _input_files, _key, _source_files,
    array_hash, sha,
)
from .stable_routing_protocol import ProtocolError, experiment_sha256


def _fail(message: str):
    raise ProtocolError(message)


def _derived_expected():
    protocol = load_protocol()
    cache = LaViLaCache(CACHE, PIXEL_ROWS, CANONICAL)
    all_users = dict(zip(protocol.sample_ids.astype(str), protocol.users.astype(str)))
    validate_spotcheck(cache, PROBE, all_users)
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if int(keep.sum()) != 2470:
        _fail("expected 2470 eligible canonical rows")
    ids, labels, users, folds = (protocol.sample_ids[keep], protocol.labels[keep],
                                 protocol.users[keep], protocol.fold_id[keep])
    source = np.flatnonzero(folds != 0); target = np.flatnonzero(folds == 0)
    if (len(source), len(target)) != (1497, 973):
        _fail("expected fold-0 source/target sizes 1497/973")
    # Provider construction performs the exact eligible-subset probe check.
    provider=Provider(cache, ids, users,
             validate_spotcheck(cache, PROBE, dict(zip(ids.astype(str), users.astype(str)))))
    expected = {"context": "fold0.outer", "outer_fold": 0,
                "source_ids": ids[source].tolist(), "source_users": users[source].tolist(),
                "target_ids": ids[target].tolist(), "target_users": users[target].tolist(),
                "source_label_sha256": array_hash(labels[source]),
                "source_class_counts": np.bincount(labels[source], minlength=40).tolist(),
                "cache_provenance": cache.provenance,"probe_evidence":provider.probe_evidence}
    return expected, cache


def _assert_safe_relpath(value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        _fail("artifact path is absolute or malformed")
    path = Path(value)
    if ".." in path.parts or any(part in ("", ".") for part in path.parts):
        _fail("artifact path traversal")
    return path


def verify_run(out) -> dict:
    """Validate one completed P432 run and return its summary payload."""
    out = Path(out)
    if not out.is_dir():
        _fail("P432 output directory missing")
    torch.set_num_threads(4)
    expected, cache = _derived_expected()
    process_path = Path(str(out) + ".process.json")
    if not process_path.is_file():
        _fail("watchdog process report missing")
    process = json.loads(process_path.read_text(encoding="utf-8"))
    command = [sys.executable, "-u", "-m", "aligned_multimodal.p432_rebuild_lavila_bank",
               "--out-dir", str(out)]
    if (process.get("status") != "complete" or process.get("exit_code") != 0
            or not isinstance(process.get("seconds"), (int, float))
            or not 0 <= process["seconds"] <= 960
            or process.get("wall_limit_seconds") != 960
            or process.get("command") != command):
        _fail("invalid P432 watchdog report")

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("mode") != "pilot" or summary.get("contexts_completed") != 1
            or summary.get("target_achieved") is not False
            or summary.get("outer_accuracy_evaluated") is not False
            or summary.get("test_rows_loaded") != 0
            or summary.get("submission_generated") is not False
            or summary.get("promotion_allowed") is not False
            or summary.get("complete_p315") is not False
            or not isinstance(summary.get("elapsed_seconds"), (int, float))
            or not 0 <= summary["elapsed_seconds"] <= 900):
        _fail("invalid P432 pilot summary/scope")

    registry_path = out / "experiment_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    spec = registry.get("spec")
    if not isinstance(spec, dict) or registry.get("sha256") != experiment_sha256(spec):
        _fail("experiment registry hash invalid")
    sources = _source_files(); inputs = _input_files(cache)
    if any(not p.is_file() for p in [*sources, *inputs]):
        _fail("registered source or input missing")
    expected_source = {_key(p): sha(p) for p in sources}
    expected_input = {_key(p): sha(p) for p in inputs}
    expected_spec = {"name": "P432 original LaViLa frame-token expert", "mode": "pilot",
                     "outer_fold": 0, "source_sha256": expected_source,
                     "input_sha256": expected_input,
                     "expert_names": ["p158_lavila_frame_token"],
                     "outer_accuracy_evaluated": False, "promotion_allowed": False,
                     "excluded_users": sorted(EXCLUDED_USERS)}
    if spec != expected_spec:
        _fail("experiment registry spec/inventory differs")

    snap = out / "source_snapshot"
    expected_snap = {"__".join(Path(p).resolve().relative_to(Path(__file__).resolve().parents[1]).parts)
                     for p in sources}
    actual_snap = {p.name for p in snap.iterdir()} if snap.is_dir() else set()
    if actual_snap != expected_snap:
        _fail("source snapshot inventory differs")
    for p in sources:
        name = "__".join(Path(p).resolve().relative_to(Path(__file__).resolve().parents[1]).parts)
        if sha(snap / name) != sha(p):
            _fail("source snapshot changed")

    expected_disk = out / "expected.json"
    if json.loads(expected_disk.read_text(encoding="utf-8")) != expected:
        _fail("disk expected context differs from fresh derivation")

    folder = out / "fold0" / "outer"
    fixed = {"experiment_registry.json", "expected.json", "summary.json",
             "fold0/outer/bank.npz", "fold0/outer/provenance.json",
             "fold0/outer/members/seed15801/checkpoint.pt",
             "fold0/outer/members/seed15801/outputs.npz",
             "fold0/outer/members/seed15801/receipt.json"}
    files = {p.relative_to(out).as_posix() for p in out.rglob("*")
             if p.is_file() and "source_snapshot" not in p.parts}
    if files != fixed:
        _fail("P432 artifact inventory differs")
    for rel in files:
        _assert_safe_relpath(rel)
    artifact_hashes = summary.get("artifact_sha256")
    hashed=fixed-{"summary.json"}
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != hashed:
        _fail("summary artifact inventory differs")
    if any(artifact_hashes[p] != sha(out / p) for p in hashed):
        _fail("artifact hash mismatch")

    verifier = __import__("aligned_multimodal.p432_lavila_verify", fromlist=["verify_context"])
    verifier.verify_context(folder, expected)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify a P432 LaViLa pilot run")
    parser.add_argument("--out-dir", required=True)
    verify_run(parser.parse_args().out_dir)
    print(json.dumps({"status": "verified"}))
