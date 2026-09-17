"""P430 full nested token-bank rebuild.

This is intentionally a thin orchestration layer over the frozen P430 pilot
provider.  It performs no scoring or submission work; every prediction context
is persisted independently so a failed run cannot be mistaken for a complete
bank.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from . import p430_rebuild_token_bank as pilot
from .p430_token_cache import TokenCache
from .p430_token_provider import TokenProvider,EXPERTS
from .p430_token_verify import verify_context
from .p90_teacher_common import load_protocol
from .p416_nested_frozen_family_router import EXCLUDED_USERS, OUTER_FOLDS
from .p427_foundation_provider import array_hash
from .stable_routing_protocol import ProtocolError, register_experiment

ROOT = pilot.ROOT
HERE = pilot.HERE
PREREG = ROOT / "docs" / "research" / "STABLE_093_P430_FULL_BANK.md"


def _sources() -> list[Path]:
    # Preserve every pilot source, including its original preregistration.
    old = pilot._source_files()
    return old + [HERE / "p430_rebuild_full_token_bank.py", HERE / "p430_full_watchdog.py",
                  HERE / "p430_full_contract.py", PREREG]


def _inputs() -> list[Path]:
    return [HERE / "data" / "subject_folds" / f"fold_{k}.csv" for k in OUTER_FOLDS] + [
        pilot.CACHE / "features.npy", pilot.CACHE / "done.npy", pilot.CACHE / "cache_summary.json",
        pilot.CANONICAL, pilot.EXTRACTION, pilot.PREREG]


def _expected(context, outer, source, target, ids, users, labels, cache):
    return {"context": context, "outer_fold": int(outer), "source_ids": ids[source].tolist(),
            "source_users": users[source].tolist(), "target_ids": ids[target].tolist(),
            "target_users": users[target].tolist(), "source_label_sha256": array_hash(labels[source]),
            "source_class_counts": np.bincount(labels[source], minlength=40).tolist(),
            "cache_provenance": cache.provenance}


def _write_context(out, name, outer, source, target, ids, labels, users, provider, cache, deadline, started):
    context = f"fold{outer}.{name}"
    folder = out / f"fold{outer}" / name
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "members").mkdir()
    expected = _expected(context, outer, source, target, ids, users, labels, cache)
    (folder / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")

    def callback(expert, seed, logits, state, record):
        member = folder / "members" / expert / f"seed{seed}"
        member.mkdir(parents=True, exist_ok=False)
        torch.save(state, member / "checkpoint.pt")
        np.savez_compressed(member / "outputs.npz", logits=logits)
        rec = dict(record, checkpoint_sha256=pilot.sha(member / "checkpoint.pt"),
                   logits_sha256=pilot.sha(member / "outputs.npz"))
        (member / "receipt.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(json.dumps({"event": "token_member_saved", "context": context, "expert": expert,
                          "seed": seed, "seconds": time.monotonic() - started}), flush=True)

    probabilities, receipt, extra = provider.fit_predict(
        source, labels[source], target, outer_fold=outer, context=context,
        deadline=deadline, device="cuda", fit_callback=callback)
    np.savez_compressed(folder / "bank.npz", probabilities=probabilities,
                        sample_ids=ids[target], users=users[target],
                        expert_names=np.asarray(receipt["expert_names"]), **extra)
    (folder / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    verify_context(folder, expected)
    if time.monotonic()>=deadline:raise TimeoutError("full token context exceeded budget")
    return folder, receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description="P430 full nested token bank (three outer folds).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot-dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic(); deadline = started + 1800
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; P430 full fitting refuses to start")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    sources = _sources(); inputs = _inputs()
    if any(not p.exists() for p in [*sources, *inputs]):
        raise FileNotFoundError(next(p for p in [*sources, *inputs] if not p.exists()))
    source_sha = {pilot._key(p): pilot.sha(p) for p in sources}
    input_sha = {pilot._key(p): pilot.sha(p) for p in inputs}
    protocol = load_protocol(); keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if int(keep.sum()) != 2470: raise ProtocolError("expected 2470 eligible canonical rows")
    ids, labels, users, folds = (v[keep] for v in (protocol.sample_ids, protocol.labels, protocol.users, protocol.fold_id))
    cache = TokenCache(pilot.CACHE, pilot.CANONICAL, pilot.EXTRACTION)
    provider = TokenProvider(cache.select(ids), ids, users, cache_provenance=cache.provenance)
    contract = __import__("aligned_multimodal.p430_full_contract", fromlist=["validate_pilot", "compare_first_context", "verify_full"])
    # The contract deliberately receives the authoritative fold-0 context,
    # allowing pilot validation before any model is fitted.
    fold0_source = np.flatnonzero(folds != 0); fold0_target = np.flatnonzero(folds == 0)
    expected_fold0 = _expected("fold0.outer", 0, fold0_source, fold0_target, ids, users, labels, cache)
    receipt = contract.validate_pilot(a.pilot_dir, expected_fold0)
    out.mkdir(parents=False)
    spec = {"name": "P430 full original token experts", "mode": "full", "outer_folds": list(OUTER_FOLDS),
            "source_sha256": source_sha, "input_sha256": input_sha, "expert_names": list(EXPERTS),
            "pilot_receipt": receipt, "promotion_allowed": False, "complete_p315": False, "target_achieved": False}
    register_experiment(out / "experiment_registry.json", spec)
    snap = out / "source_snapshot"; snap.mkdir()
    for p in sources: (snap / "__".join(p.resolve().relative_to(ROOT).parts)).write_bytes(p.read_bytes())
    report = {"mode": "full", "folds": {}, "target_achieved": False, "complete_p315": False,
              "outer_accuracy_evaluated": False, "test_rows_loaded": 0}
    first = True
    for outer in OUTER_FOLDS:
        train = np.flatnonzero(folds != outer); held = np.flatnonzero(folds == outer)
        requests = [("outer", train, held)]
        for inner, (tr, va) in enumerate(GroupKFold(3).split(train, labels[train], users[train])):
            requests.append((f"inner{inner}", train[tr], train[va]))
        inner_bank = np.zeros((len(train), 2, 40), dtype=np.float32); outer_bank = None; covered = np.zeros(len(train), dtype=int)
        contexts = []
        for name, src, tgt in requests:
            folder, r = _write_context(out, name, outer, src, tgt, ids, labels, users, provider, cache, deadline, started)
            if first:
                report["first_context_comparison"]=contract.compare_first_context(folder, a.pilot_dir)
                (out/"first_context_comparison.json").write_text(json.dumps(report["first_context_comparison"],indent=2),encoding="utf-8")
                first = False
            with np.load(folder / "bank.npz", allow_pickle=False) as z: bank = z["probabilities"]
            if name == "outer": outer_bank = bank
            else:
                pos = {sid: i for i, sid in enumerate(ids[train])}
                ix = np.asarray([pos[s] for s in ids[tgt]]); inner_bank[ix] = bank; covered[ix] += 1
            contexts.append({"name": name, "source_rows": len(src), "target_rows": len(tgt), "provenance_checked": True})
            print(json.dumps({"event":"token_context_complete","context":f"fold{outer}.{name}","seconds":time.monotonic()-started}),flush=True)
        if not np.all(covered == 1): raise ProtocolError("inner coverage invalid")
        np.savez_compressed(out / f"fold{outer}_banks.npz", inner_sample_ids=ids[train], outer_sample_ids=ids[held],
                            inner_users=users[train], outer_users=users[held], expert_names=np.asarray(["p142_all_token", "p144_hand_interaction"]),
                            inner_probability_bank=inner_bank, outer_probability_bank=outer_bank)
        report["folds"][str(outer)] = {"contexts": contexts, "contexts_completed": 4, "provenance_checked": True}
    current_sources = {pilot._key(p): pilot.sha(p) for p in sources}; current_inputs = {pilot._key(p): pilot.sha(p) for p in inputs}
    if current_sources != source_sha or current_inputs != input_sha: raise ProtocolError("registered source/input changed during fit")
    report["verification"]=contract.verify_full(out, ids, labels, users, folds, cache.provenance)
    report["contexts_completed"]=report["verification"]["contexts_verified"]
    report["elapsed_seconds"] = time.monotonic() - started; report["artifact_sha256"] = {p.relative_to(out).as_posix(): pilot.sha(p) for p in out.rglob("*") if p.is_file() and "source_snapshot" not in p.parts}
    if time.monotonic()>=deadline:raise TimeoutError("full token final verification exceeded budget")
    report["elapsed_seconds"]=time.monotonic()-started
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event": "token_full_complete", "seconds": report["elapsed_seconds"], "complete_p315": False, "target_achieved": False}), flush=True)

if __name__ == "__main__": main()
