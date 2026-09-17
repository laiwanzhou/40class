"""Bounded full P432 LaViLa nested-bank rebuild (12 source-only contexts)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from . import p432_rebuild_lavila_bank as pilot
from .p432_lavila_cache import LaViLaCache
from .p432_lavila_provider import EXCLUDED_USERS, Provider
from .p427_foundation_provider import array_hash
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ProtocolError, register_experiment

ROOT, HERE = pilot.ROOT, pilot.HERE
PREREG = ROOT / "docs" / "research" / "STABLE_093_P432_FULL_BANK.md"
OUTER_FOLDS = (0, 1, 2)
EXPERT_NAME = "p158_lavila_frame_token"
PILOT_RUN=HERE/'runs/p432_lavila_pilot_v1'


def _sources() -> list[Path]:
    return pilot._source_files() + [HERE / "p432_rebuild_full_lavila_bank.py",
                                    HERE / "p432_full_watchdog.py",
                                    HERE / "p432_full_contract.py", PREREG]


def _inputs(cache: LaViLaCache) -> list[Path]:
    return pilot._input_files(cache) + [PREREG,Path(str(PILOT_RUN)+'.process.json'),
        *[p for p in sorted(PILOT_RUN.rglob('*')) if p.is_file() and 'source_snapshot' not in p.parts]]


def _expected(context, outer, source, target, ids, users, labels, cache, probe_evidence):
    return {"context": context, "outer_fold": int(outer),
            "source_ids": ids[source].tolist(), "source_users": users[source].tolist(),
            "target_ids": ids[target].tolist(), "target_users": users[target].tolist(),
            "source_label_sha256": array_hash(labels[source]),
            "source_class_counts": np.bincount(labels[source], minlength=40).tolist(),
            "cache_provenance": cache.provenance,
            "probe_evidence": probe_evidence}


def _write_context(out, name, outer, source, target, ids, labels, users, provider, cache,
                   deadline, started):
    context = f"fold{outer}.{name}"; folder = out / f"fold{outer}" / name
    folder.mkdir(parents=True, exist_ok=False); (folder / "members").mkdir()
    expected = _expected(context, outer, source, target, ids, users, labels, cache,
                         provider.probe_evidence)
    (folder / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")

    def callback(seed, logits, state, record):
        member = folder / "members" / f"seed{seed}"; member.mkdir(parents=False, exist_ok=False)
        torch.save(state, member / "checkpoint.pt")
        np.savez_compressed(member / "outputs.npz", logits=logits)
        record=dict(record)
        record.update(checkpoint_sha256=pilot.sha(member / "checkpoint.pt"),
                      logits_sha256=pilot.sha(member / "outputs.npz"))
        (member / "receipt.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({"event": "lavila_member_saved", "context": context, "seed": seed,
                          "seconds": time.monotonic() - started}), flush=True)

    probabilities, receipt, extra = provider.fit_predict(
        source, labels[source], target, outer_fold=outer, context=context,
        deadline=deadline, device="cuda", fit_callback=callback)
    np.savez_compressed(folder / "bank.npz", probabilities=probabilities,
                        logits=extra["logits"], sample_ids=ids[target], users=users[target],
                        expert_names=np.asarray([receipt["expert_name"]]))
    (folder / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    verifier = __import__("aligned_multimodal.p432_lavila_verify", fromlist=["verify_context"])
    verifier.verify_context(folder, expected)
    if time.monotonic() >= deadline:
        raise TimeoutError("P432 full context exceeded budget")
    return folder, receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description="P432 full LaViLa nested bank (12 contexts).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic(); deadline = started + 1800
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; P432 full fitting refuses to start")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4 * 1024**3 /
        torch.cuda.get_device_properties(0).total_memory))
    sources = _sources(); protocol = load_protocol()
    if any(not p.exists() for p in sources):
        raise FileNotFoundError(next(p for p in sources if not p.exists()))
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if int(keep.sum()) != 2470:
        raise ProtocolError("expected 2470 eligible canonical rows")
    ids, labels, users, folds = (protocol.sample_ids[keep], protocol.labels[keep],
                                 protocol.users[keep], protocol.fold_id[keep])
    cache = LaViLaCache(pilot.CACHE, pilot.PIXEL_ROWS, pilot.CANONICAL)
    probe_evidence = pilot.validate_spotcheck(
        cache, pilot.PROBE,
        dict(zip(protocol.sample_ids.astype(str), protocol.users.astype(str))),
    )
    inputs = _inputs(cache)
    if any(not p.exists() for p in inputs):
        raise FileNotFoundError(next(p for p in inputs if not p.exists()))
    provider = Provider(cache, ids, users,
                        pilot.validate_spotcheck(cache, pilot.PROBE,
                            dict(zip(ids.astype(str), users.astype(str)))))
    source_sha = {pilot._key(p): pilot.sha(p) for p in sources}
    input_sha = {pilot._key(p): pilot.sha(p) for p in inputs}
    spec = {"name": "P432 full original LaViLa frame-token expert", "mode": "full",
            "outer_folds": list(OUTER_FOLDS), "source_sha256": source_sha,
            "input_sha256": input_sha, "expert_names": [EXPERT_NAME],
            "promotion_allowed": False, "complete_p315": False,
            "target_achieved": False}
    out.mkdir(parents=False); register_experiment(out / "experiment_registry.json", spec)
    snap = out / "source_snapshot"; snap.mkdir()
    for p in sources:
        (snap / "__".join(p.resolve().relative_to(ROOT).parts)).write_bytes(p.read_bytes())
    report = {"mode": "full", "folds": {}, "contexts_completed": 0,
              "target_achieved": False, "complete_p315": False,
              "outer_accuracy_evaluated": False, "promotion_allowed": False,
              "submission_generated": False, "test_rows_loaded": 0}
    for outer in OUTER_FOLDS:
        train = np.flatnonzero(folds != outer); held = np.flatnonzero(folds == outer)
        requests = [("outer", train, held)]
        splitter = GroupKFold(3)
        for inner, (tr, va) in enumerate(splitter.split(train, labels[train], users[train])):
            requests.append((f"inner{inner}", train[tr], train[va]))
        inner_bank = np.zeros((len(train), 1, 40), dtype=np.float32)
        outer_bank = None; covered = np.zeros(len(train), dtype=np.int64); contexts = []
        for name, source, target in requests:
            folder, receipt = _write_context(out, name, outer, source, target, ids, labels, users,
                                             provider, cache, deadline, started)
            with np.load(folder / "bank.npz", allow_pickle=False) as z:
                bank = z["probabilities"]
            if name == "outer":
                outer_bank = bank
            else:
                positions = {sid: i for i, sid in enumerate(ids[train])}
                indices = np.asarray([positions[s] for s in ids[target]], dtype=np.int64)
                inner_bank[indices] = bank; covered[indices] += 1
            contexts.append({"name": name, "source_rows": len(source), "target_rows": len(target),
                             "provenance_checked": True})
            report["contexts_completed"] += 1
            print(json.dumps({"event": "lavila_context_complete", "context": f"fold{outer}.{name}",
                              "seconds": time.monotonic() - started}), flush=True)
        if not np.all(covered == 1) or outer_bank is None:
            raise ProtocolError("P432 inner coverage invalid")
        np.savez_compressed(out / f"fold{outer}_banks.npz", inner_sample_ids=ids[train],
                            outer_sample_ids=ids[held], inner_users=users[train],
                            outer_users=users[held], expert_names=np.asarray([EXPERT_NAME]),
                            inner_probability_bank=inner_bank, outer_probability_bank=outer_bank)
        report["folds"][str(outer)] = {"contexts": contexts, "contexts_completed": 4,
                                       "provenance_checked": True}
    if {pilot._key(p): pilot.sha(p) for p in sources} != source_sha or \
       {pilot._key(p): pilot.sha(p) for p in inputs} != input_sha:
        raise ProtocolError("registered source/input changed during fit")
    report["elapsed_seconds"] = time.monotonic() - started
    report["artifact_sha256"] = {p.relative_to(out).as_posix(): pilot.sha(p) for p in out.rglob("*")
                                  if p.is_file() and "source_snapshot" not in p.parts}
    if time.monotonic() >= deadline:
        raise TimeoutError("P432 full final hashing exceeded budget")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    contract = __import__("aligned_multimodal.p432_full_contract", fromlist=["verify_full"])
    contract.verify_full(out, require_process=False)
    if time.monotonic()>=deadline:raise TimeoutError("P432 full verification exceeded budget")
    report["elapsed_seconds"]=time.monotonic()-started
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event": "lavila_full_complete", "contexts": 12,
                      "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
