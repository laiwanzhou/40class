"""Bounded P432 fold-0 LaViLa frame-token pilot.

This runner materialises exactly one source-only outer context.  It does not
load test labels or produce a submission; the resulting bank is an auditable
training artifact for the later verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .p90_teacher_common import load_protocol
from .p432_lavila_cache import LaViLaCache
from .p432_lavila_provider import EXCLUDED_USERS, Provider
from .p432_probe_contract import validate_spotcheck, _expected_probe_sources
from .p427_foundation_provider import array_hash
from .stable_routing_protocol import ProtocolError, register_experiment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CANONICAL = HERE / "data" / "manifest.csv"
CACHE = HERE / "runs" / "p157_lavila_frame_token_cache_v1"
PIXEL_ROWS = HERE / "runs" / "p86_visual_pixel_cache_t16_r160_v12" / "rows.csv"
PROBE = HERE / "runs" / "p432_cpu_spotcheck_v1"
PREREG = ROOT / "docs" / "research" / "STABLE_093_P432_LAVILA_PILOT.md"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _key(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "external/" + Path(path).name


def _source_files() -> list[Path]:
    names = [
        "p432_rebuild_lavila_bank.py", "p432_watchdog.py", "p432_lavila_cache.py",
        "p432_lavila_provider.py", "p432_lavila_training.py", "p432_probe_contract.py",
        "p432_lavila_verify.py", "p432_run_verify.py", "p142_vjepa_token_transformer_oof.py",
        "p155_lavila_teacher.py", "p157_lavila_frame_token_cache.py",
        "p90_teacher_common.py", "p416_nested_frozen_family_router.py",
        "p427_foundation_provider.py", "stable_routing_protocol.py",
        "p427_foundation_kernels.py", "audit_p87_sequence_decoder.py",
    ]
    return [HERE / n for n in names] + [PREREG]


def _input_files(cache: LaViLaCache) -> list[Path]:
    folds = [HERE / "data" / "subject_folds" / f"fold_{k}.csv" for k in range(3)]
    cache_paths = [Path(p) for p in cache.provenance["input_sha256"]]
    probe_paths = [PROBE / "registry.json", PROBE / "summary.json", PROBE / "tokens.npz",
                   Path(str(PROBE) + ".process.json")]
    return list(dict.fromkeys([*folds, *cache_paths, *probe_paths, PREREG,
                              *[Path(p) for p in sorted(_expected_probe_sources())]]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="P432 LaViLa fold-0 outer pilot (one member).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic(); deadline = started + 900
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; P432 real fitting refuses to start")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4 * 1024**3 /
        torch.cuda.get_device_properties(0).total_memory))

    protocol = load_protocol()
    # Validate the durable probe against the complete canonical identity map
    # before narrowing the provider to the eligible 2470-row universe.
    all_users = dict(zip(protocol.sample_ids.astype(str), protocol.users.astype(str)))
    cache = LaViLaCache(CACHE, PIXEL_ROWS, CANONICAL)
    validate_spotcheck(cache, PROBE, all_users)
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if int(keep.sum()) != 2470:
        raise ProtocolError("expected 2470 eligible canonical rows")
    ids, labels, users, folds = (protocol.sample_ids[keep], protocol.labels[keep],
                                 protocol.users[keep], protocol.fold_id[keep])
    source = np.flatnonzero(folds != 0); target = np.flatnonzero(folds == 0)
    if (len(source), len(target)) != (1497, 973):
        raise ProtocolError("expected fold-0 source/target sizes 1497/973")
    provider = Provider(cache, ids, users,
                        validate_spotcheck(cache, PROBE, dict(zip(ids.astype(str), users.astype(str)))))

    out.mkdir(parents=False)
    sources = _source_files()
    if any(not p.exists() for p in sources):
        raise FileNotFoundError(next(p for p in sources if not p.exists()))
    inputs = _input_files(cache)
    if any(not p.exists() for p in inputs):
        raise FileNotFoundError(next(p for p in inputs if not p.exists()))
    spec = {"name": "P432 original LaViLa frame-token expert", "mode": "pilot", "outer_fold": 0,
            "source_sha256": {_key(p): sha(p) for p in sources},
            "input_sha256": {_key(p): sha(p) for p in inputs},
            "expert_names": ["p158_lavila_frame_token"], "outer_accuracy_evaluated": False,
            "promotion_allowed": False, "excluded_users": sorted(EXCLUDED_USERS)}
    register_experiment(out / "experiment_registry.json", spec)
    snap = out / "source_snapshot"; snap.mkdir()
    for p in sources:
        (snap / "__".join(Path(p).resolve().relative_to(ROOT).parts)).write_bytes(p.read_bytes())
    expected = {"context": "fold0.outer", "outer_fold": 0,
                "source_ids": ids[source].tolist(), "source_users": users[source].tolist(),
                "target_ids": ids[target].tolist(), "target_users": users[target].tolist(),
                "source_label_sha256": array_hash(labels[source]),
                "source_class_counts": np.bincount(labels[source], minlength=40).tolist(),
                "cache_provenance": cache.provenance, "probe_evidence": provider.probe_evidence}
    (out / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    folder = out / "fold0" / "outer"; (folder / "members").mkdir(parents=True)

    def callback(seed, logits, state, record):
        member = folder / "members" / f"seed{seed}"; member.mkdir(parents=False, exist_ok=False)
        torch.save(state, member / "checkpoint.pt")
        np.savez_compressed(member / "outputs.npz", logits=logits)
        record=dict(record)
        record.update(checkpoint_sha256=sha(member / "checkpoint.pt"), logits_sha256=sha(member / "outputs.npz"))
        (member / "receipt.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({"event": "lavila_member_saved", "seed": seed,
                          "seconds": time.monotonic() - started}), flush=True)

    probabilities, receipt, extra = provider.fit_predict(
        source, labels[source], target, outer_fold=0, context="fold0.outer",
        deadline=deadline, device="cuda", fit_callback=callback)
    np.savez_compressed(folder / "bank.npz", probabilities=probabilities,
                        logits=extra["logits"], sample_ids=ids[target], users=users[target],
                        expert_names=np.asarray([receipt["expert_name"]]))
    (folder / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    verifier = __import__("aligned_multimodal.p432_lavila_verify", fromlist=["verify_context"])
    verifier.verify_context(folder, expected)
    if {_key(p): sha(p) for p in sources} != spec["source_sha256"]:
        raise ProtocolError("registered source changed during fit")
    if {_key(p): sha(p) for p in inputs} != spec["input_sha256"]:
        raise ProtocolError("registered input changed during fit")
    if time.monotonic() >= deadline:
        raise TimeoutError("P432 reporting exceeded budget")
    report = {"mode": "pilot", "contexts_completed": 1,
              "elapsed_seconds": time.monotonic() - started, "target_achieved": False,
              "outer_accuracy_evaluated": False, "test_rows_loaded": 0,
              "submission_generated": False, "promotion_allowed": False,
              "complete_p315": False,
              "artifact_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*")
                                  if p.is_file() and "source_snapshot" not in p.parts}}
    if time.monotonic()>=deadline:raise TimeoutError("P432 final hashing exceeded budget")
    report["elapsed_seconds"]=time.monotonic()-started
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event": "lavila_pilot_complete", "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
