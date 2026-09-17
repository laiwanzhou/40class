"""Run-level contract for the complete twelve-context P433 physical bank."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .p427_foundation_provider import array_hash
from .p433_physical_cache import FAMILY_NAMES, PhysicalTokenCache, _sha
from .p433_physical_provider import BASE_SEEDS, EXCLUDED_USERS, EXPERT_NAME
from .p433_physical_verify import verify_context
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ProtocolError, experiment_sha256


def _fail(message):
    raise ProtocolError(message)


def _runner():
    from . import p433_rebuild_full_physical_bank as runner
    return runner


def _fresh():
    runner = _runner()
    pilot = getattr(runner, "pilot", runner)
    protocol = load_protocol()
    manifest = Path(pilot.CANONICAL)
    # The full-runner records the four original physical paths in its pilot.
    paths = getattr(pilot, "CACHE_PATHS", None)
    if paths is None:
        paths = getattr(pilot, "FEATURE_PATHS", None)
    cache = PhysicalTokenCache(paths, manifest=manifest) if paths is not None else PhysicalTokenCache(manifest=manifest)
    users_all = dict(zip(protocol.sample_ids.astype(str), protocol.users.astype(str)))
    if not np.array_equal(cache.master_ids.astype(str), protocol.sample_ids.astype(str)) or not np.array_equal(cache.master_users.astype(str), protocol.users.astype(str)):
        _fail("P433 cache/protocol identity differs")
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if len(protocol.sample_ids) != 2914 or int(keep.sum()) != 2470:
        _fail("P433 canonical population differs")
    ids = protocol.sample_ids.astype(str)[keep]; labels = protocol.labels[keep]
    users = protocol.users.astype(str)[keep]; folds = protocol.fold_id[keep]
    if len(set(ids)) != 2470 or set(users) & set(EXCLUDED_USERS) or set(folds) != {0, 1, 2}:
        _fail("invalid P433 eligible population")
    return runner, cache, ids, labels, users, folds


def _expected(ids, labels, users, source, target, fold, name, cache):
    return {"context": f"fold{fold}.{name}", "outer_fold": int(fold),
            "source_ids": ids[source].tolist(), "source_users": users[source].tolist(),
            "target_ids": ids[target].tolist(), "target_users": users[target].tolist(),
            "source_label_sha256": array_hash(labels[source]),
            "source_class_counts": np.bincount(labels[source], minlength=40).tolist(),
            "cache_provenance": cache.provenance}


def _key(path, root):
    p = Path(path).resolve()
    try: return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError: return "external/" + p.name


def verify_full(out, require_process=True):
    out = Path(out)
    if not out.is_dir(): _fail("P433 full output directory missing")
    runner, cache, ids, labels, users, folds = _fresh()
    if require_process:
        pp = Path(str(out) + ".process.json")
        if not pp.is_file(): _fail("P433 process report missing")
        process = json.loads(pp.read_text(encoding="utf-8"))
        command = [sys.executable, "-u", "-m", "aligned_multimodal.p433_rebuild_full_physical_bank", "--out-dir", str(out)]
        if (process.get("status") != "complete" or process.get("exit_code") != 0
                or not isinstance(process.get("seconds"), (int, float)) or not 0 <= process["seconds"] <= 1860
                or process.get("wall_limit_seconds") != 1860 or process.get("command") != command):
            _fail("invalid P433 process report")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("mode") != "full" or summary.get("contexts_completed") != 12
            or summary.get("target_achieved") is not False or summary.get("outer_accuracy_evaluated") is not False
            or summary.get("test_rows_loaded") != 0 or summary.get("promotion_allowed") is not False
            or summary.get("submission_generated") is not False
            or summary.get("complete_p315") is not False or not isinstance(summary.get("elapsed_seconds"), (int, float))
            or not 0 <= summary["elapsed_seconds"] <= 1800): _fail("invalid P433 full summary")
    registry = json.loads((out / "experiment_registry.json").read_text(encoding="utf-8")); spec = registry.get("spec")
    if not isinstance(spec, dict) or registry.get("sha256") != experiment_sha256(spec): _fail("invalid P433 registry hash")
    root = Path(getattr(runner, "ROOT", Path(__file__).resolve().parents[1])); sources = list(runner._sources()); inputs = list(runner._inputs(cache))
    key = getattr(runner, "_key", lambda p: _key(p, root)); source_hash = {key(p): _sha(p) for p in sources}; input_hash = {key(p): _sha(p) for p in inputs}
    expected_spec = {"name": "P433 full original physical token expert", "mode": "full", "outer_folds": [0, 1, 2],
                     "source_sha256": source_hash, "input_sha256": input_hash, "expert_names": [EXPERT_NAME],
                     "promotion_allowed": False, "complete_p315": False, "target_achieved": False}
    if spec != expected_spec: _fail("P433 registry spec differs")
    snap = out / "source_snapshot"; expected_snap = {"__".join(Path(p).resolve().relative_to(root.resolve()).parts) for p in sources}
    if ({p.name for p in snap.iterdir()} if snap.is_dir() else set()) != expected_snap: _fail("P433 source snapshot inventory differs")
    for p in sources:
        n = "__".join(Path(p).resolve().relative_to(root.resolve()).parts)
        if _sha(snap / n) != _sha(p): _fail("P433 source snapshot changed")
    fixed = {"experiment_registry.json", "summary.json", *[f"fold{k}_banks.npz" for k in range(3)]}
    for fold in range(3):
        for name in ("outer", "inner0", "inner1", "inner2"):
            base = f"fold{fold}/{name}"; fixed.update(f"{base}/{x}" for x in ("expected.json", "bank.npz", "provenance.json"))
            fixed.update(f"{base}/members/seed{b+1000*fold}/{x}" for b in BASE_SEEDS for x in ("checkpoint.pt", "outputs.npz", "receipt.json"))
    actual = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and "source_snapshot" not in p.parts}
    if actual != fixed: _fail("P433 artifact inventory differs")
    if set(summary.get("artifact_sha256", {})) != fixed - {"summary.json"} or any(_sha(out / p) != d for p, d in summary["artifact_sha256"].items()): _fail("P433 artifact hashes differ")
    for fold in range(3):
        source = np.flatnonzero(folds != fold); target = np.flatnonzero(folds == fold)
        requests = [("outer", source, target, None)] + [(f"inner{k}", source[tr], source[va], va) for k, (tr, va) in enumerate(GroupKFold(3).split(source, groups=users[source]))]
        coverage = np.zeros(len(source), dtype=np.int64)
        with np.load(out / f"fold{fold}_banks.npz", allow_pickle=False) as agg:
            required = {"inner_sample_ids", "outer_sample_ids", "inner_users", "outer_users", "expert_names", "inner_probability_bank", "outer_probability_bank"}
            if set(agg.files) != required or agg["inner_probability_bank"].shape != (len(source), 1, 40) or agg["outer_probability_bank"].shape != (len(target), 1, 40): _fail("P433 aggregate schema differs")
            for arr in (agg["inner_probability_bank"], agg["outer_probability_bank"]):
                if arr.dtype != np.float32 or not np.isfinite(arr).all(): _fail("P433 aggregate dtype differs")
            for k, v in (("inner_sample_ids", ids[source]), ("outer_sample_ids", ids[target]), ("inner_users", users[source]), ("outer_users", users[target]), ("expert_names", np.asarray([EXPERT_NAME]))):
                if not np.array_equal(agg[k], v): _fail("P433 aggregate identity differs")
            for name, tr, va, local in requests:
                expected = _expected(ids, labels, users, tr, va, fold, name, cache); folder = out / f"fold{fold}" / name
                if json.loads((folder / "expected.json").read_text(encoding="utf-8")) != expected: _fail("P433 expected context differs")
                verify_context(folder, expected)
                with np.load(folder / "bank.npz", allow_pickle=False) as bank:
                    combined = agg["outer_probability_bank"] if local is None else agg["inner_probability_bank"][local]
                    if not np.array_equal(bank["probabilities"], combined): _fail("P433 aggregate/context differs")
                if local is not None: coverage[local] += 1
        if not np.all(coverage == 1): _fail("P433 inner coverage differs")
    return {"contexts_verified": 12, "members_verified": 36, "outer_unique_rows": 2470, "outer_accuracy_evaluated": False}
