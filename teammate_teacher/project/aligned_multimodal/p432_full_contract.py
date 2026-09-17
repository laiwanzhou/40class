"""Run-level contract for the complete twelve-context P432 LaViLa bank."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .p427_foundation_provider import array_hash
from .p432_lavila_cache import LaViLaCache
from .p432_lavila_provider import EXCLUDED_USERS
from .p432_lavila_verify import verify_context
from .p432_probe_contract import validate_spotcheck, sha
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ProtocolError, experiment_sha256


def _fail(msg):
    raise ProtocolError(msg)


def _runner():
    """Import the full runner lazily so importing this verifier is harmless."""
    from . import p432_rebuild_full_lavila_bank as runner
    return runner


def _fresh():
    runner = _runner()
    pilot = runner
    if not hasattr(runner, "CACHE"):
        try:
            from . import p432_rebuild_lavila_bank as pilot
        except ImportError as exc:
            raise ProtocolError("P432 pilot constants unavailable") from exc
    protocol = load_protocol()
    pixel_rows = getattr(pilot, "PIXEL_ROWS", getattr(pilot, "PIXEL_TABLE", None))
    cache = LaViLaCache(pilot.CACHE, pixel_rows, pilot.CANONICAL)
    all_users = dict(zip(protocol.sample_ids.astype(str), protocol.users.astype(str)))
    probe = pilot.PROBE
    if probe is None:
        _fail("P432 probe path missing")
    probe_evidence = validate_spotcheck(cache, probe, all_users)
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if len(protocol.sample_ids) != 2914 or int(keep.sum()) != 2470:
        _fail("P432 canonical population differs")
    ids, labels, users, folds = (protocol.sample_ids.astype(str)[keep], protocol.labels[keep],
                                 protocol.users.astype(str)[keep], protocol.fold_id[keep])
    if len(set(ids)) != 2470 or set(users) & set(EXCLUDED_USERS) or set(folds) != {0, 1, 2}:
        _fail("invalid P432 eligible population")
    return runner, cache, probe_evidence, ids, labels, users, folds


def _expected(ids, labels, users, source, target, fold, name, cache, probe_evidence):
    return {"context": f"fold{fold}.{name}", "outer_fold": int(fold),
            "source_ids": ids[source].tolist(), "source_users": users[source].tolist(),
            "target_ids": ids[target].tolist(), "target_users": users[target].tolist(),
            "source_label_sha256": array_hash(labels[source]),
            "source_class_counts": np.bincount(labels[source], minlength=40).tolist(),
            "cache_provenance": cache.provenance, "probe_evidence": probe_evidence}


def _key(path, root):
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return "external/" + p.name


def _process_check(out, require_process):
    if not require_process:
        return
    path = Path(str(out) + ".process.json")
    if not path.is_file():
        _fail("P432 full watchdog report missing")
    process = json.loads(path.read_text(encoding="utf-8"))
    runner = _runner()
    command = [sys.executable, "-u", "-m", "aligned_multimodal.p432_rebuild_full_lavila_bank", "--out-dir", str(out)]
    if (process.get("status") != "complete" or process.get("exit_code") != 0
            or not isinstance(process.get("seconds"), (int, float)) or not 0 <= process["seconds"] <= 1860
            or process.get("wall_limit_seconds")!=1860
            or process.get("command") != command):
        _fail("invalid P432 full watchdog report")


def verify_full(out, require_process=True):
    out = Path(out)
    if not out.is_dir():
        _fail("P432 full output directory missing")
    runner, cache, probe_evidence, ids, labels, users, folds = _fresh()
    _process_check(out, require_process)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("mode") != "full" or summary.get("contexts_completed") != 12
            or summary.get("complete_p315") is not False
            or summary.get("target_achieved") is not False or summary.get("outer_accuracy_evaluated") is not False
            or summary.get("test_rows_loaded") != 0 or summary.get("submission_generated") is not False
            or summary.get("promotion_allowed") is not False or not isinstance(summary.get("elapsed_seconds"), (int, float))
            or not 0 <= summary["elapsed_seconds"] <= 1800):
        _fail("invalid P432 full summary/scope")

    registry = json.loads((out / "experiment_registry.json").read_text(encoding="utf-8"))
    spec = registry.get("spec")
    if not isinstance(spec, dict) or registry.get("sha256") != experiment_sha256(spec):
        _fail("P432 experiment registry hash invalid")
    sources = list(runner._sources()); inputs = list(runner._inputs(cache))
    root = Path(getattr(runner, "ROOT", Path(__file__).resolve().parents[1]))
    key = getattr(runner, "_key", lambda p: _key(p, root))
    source_hashes = {key(p): sha(p) for p in sources}; input_hashes = {key(p): sha(p) for p in inputs}
    expected_spec={"name":"P432 full original LaViLa frame-token expert","mode":"full","outer_folds":[0,1,2],
        "source_sha256":source_hashes,"input_sha256":input_hashes,"expert_names":["p158_lavila_frame_token"],
        "promotion_allowed":False,"complete_p315":False,"target_achieved":False}
    if spec!=expected_spec:
        _fail("P432 full source/input inventory differs")
    snap = out / "source_snapshot"
    expected_snap = {"__".join(Path(p).resolve().relative_to(root.resolve()).parts) for p in sources}
    actual_snap = {p.name for p in snap.iterdir()} if snap.is_dir() else set()
    if actual_snap != expected_snap:
        _fail("P432 full source snapshot inventory differs")
    for p in sources:
        n = "__".join(Path(p).resolve().relative_to(root.resolve()).parts)
        if sha(snap / n) != sha(p):
            _fail("P432 full source snapshot changed")

    fixed = {"experiment_registry.json", "summary.json", "fold0_banks.npz", "fold1_banks.npz", "fold2_banks.npz"}
    for fold in range(3):
        for name in ("outer", "inner0", "inner1", "inner2"):
            prefix = f"fold{fold}/{name}"
            fixed.update(f"{prefix}/{x}" for x in ("expected.json", "bank.npz", "provenance.json",
                *[f"members/seed{15801+1000*fold}/{file}" for file in ("checkpoint.pt","outputs.npz","receipt.json")]))
    actual_files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and "source_snapshot" not in p.parts}
    if actual_files != fixed:
        _fail("P432 full artifact inventory differs")
    if set(summary.get("artifact_sha256", {})) != fixed - {"summary.json"}:
        _fail("P432 full artifact hashes omit/include summary")
    if any(sha(out / p) != d for p, d in summary["artifact_sha256"].items()):
        _fail("P432 full artifact hash mismatch")

    for fold in range(3):
        source = np.flatnonzero(folds != fold); target = np.flatnonzero(folds == fold)
        requests = [("outer", source, target, None)]
        requests += [(f"inner{k}", source[tr], source[va], va)
                     for k, (tr, va) in enumerate(GroupKFold(3).split(source, groups=users[source]))]
        coverage = np.zeros(len(source), dtype=np.int64)
        with np.load(out / f"fold{fold}_banks.npz", allow_pickle=False) as agg:
            if set(agg.files) != {"inner_sample_ids", "outer_sample_ids", "inner_users", "outer_users", "expert_names", "inner_probability_bank", "outer_probability_bank"}:
                _fail("P432 aggregate array inventory differs")
            if agg["inner_probability_bank"].shape != (len(source), 1, 40) or agg["outer_probability_bank"].shape != (len(target), 1, 40):
                _fail("P432 aggregate bank shape differs")
            if any(agg[k].dtype!=np.float32 or not np.isfinite(agg[k]).all() for k in ("inner_probability_bank","outer_probability_bank")):
                _fail("P432 aggregate dtype/finite values differ")
            for k, v in (("inner_sample_ids", ids[source]), ("outer_sample_ids", ids[target]), ("inner_users", users[source]), ("outer_users", users[target]), ("expert_names", np.asarray(["p158_lavila_frame_token"]))):
                if not np.array_equal(agg[k], v): _fail("P432 aggregate identity differs")
            for name, tr, va, local in requests:
                expected = _expected(ids, labels, users, tr, va, fold, name, cache, probe_evidence)
                folder = out / f"fold{fold}" / name
                if json.loads((folder / "expected.json").read_text(encoding="utf-8")) != expected: _fail("P432 context expected differs")
                verify_context(folder, expected)
                with np.load(folder / "bank.npz", allow_pickle=False) as bank:
                    combined = agg["outer_probability_bank"] if local is None else agg["inner_probability_bank"][local]
                    if not np.array_equal(bank["probabilities"], combined): _fail("P432 aggregate/context bank differs")
                if local is not None: coverage[local] += 1
        if not np.all(coverage == 1): _fail("P432 inner coverage differs")
    return {"contexts_verified": 12, "members_verified": 12, "outer_unique_rows": 2470, "outer_accuracy_evaluated": False}
