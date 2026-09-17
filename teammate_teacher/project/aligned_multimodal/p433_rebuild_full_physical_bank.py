"""Bounded full P433 physical-token nested bank (12 source-only contexts)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from . import p433_rebuild_physical_bank as pilot
from .p433_physical_cache import PhysicalTokenCache
from .p433_physical_provider import EXCLUDED_USERS, Provider
from .p427_foundation_provider import array_hash
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ProtocolError, register_experiment

ROOT, HERE = pilot.ROOT, pilot.HERE
PREREG = ROOT / "docs" / "research" / "STABLE_093_P433_FULL_BANK.md"
PILOT_RUN = HERE / "runs" / "p433_physical_pilot_v1"
OUTER_FOLDS = (0, 1, 2)
EXPERT_NAME = "p238_physical_token"


def _sources() -> list[Path]:
    return pilot._source_files() + [HERE / "p433_rebuild_full_physical_bank.py",
                                    HERE / "p433_full_watchdog.py",
                                    HERE / "p433_full_contract.py", PREREG]


def _pilot_artifacts() -> list[Path]:
    names = ["experiment_registry.json", "expected.json", "summary.json",
             "fold0/outer/bank.npz", "fold0/outer/provenance.json"]
    for seed in (23801, 23817, 23833):
        names.extend([f"fold0/outer/members/seed{seed}/checkpoint.pt",
                      f"fold0/outer/members/seed{seed}/outputs.npz",
                      f"fold0/outer/members/seed{seed}/receipt.json"])
    return [PILOT_RUN / name for name in names] + [
        Path(str(PILOT_RUN) + ".process.json"), Path(str(PILOT_RUN) + ".process.log")]


def _inputs(cache: PhysicalTokenCache) -> list[Path]:
    return pilot._input_files(cache) + [PREREG, *_pilot_artifacts()]


def _expected(context, outer, source, target, ids, users, labels, cache):
    return {"context": context, "outer_fold": int(outer),
            "source_ids": ids[source].tolist(), "source_users": users[source].tolist(),
            "target_ids": ids[target].tolist(), "target_users": users[target].tolist(),
            "source_label_sha256": array_hash(labels[source]),
            "source_class_counts": np.bincount(labels[source], minlength=40).tolist(),
            "cache_provenance": cache.provenance}


def _write_context(out, name, outer, source, target, ids, labels, users, provider, cache,
                   deadline, started):
    context = f"fold{outer}.{name}"; folder = out / f"fold{outer}" / name
    folder.mkdir(parents=True, exist_ok=False); (folder / "members").mkdir()
    expected = _expected(context, outer, source, target, ids, users, labels, cache)
    (folder / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")

    def callback(seed, logits, state, record):
        member = folder / "members" / f"seed{seed}"; member.mkdir(parents=False, exist_ok=False)
        torch.save(state, member / "checkpoint.pt")
        np.savez_compressed(member / "outputs.npz", logits=logits)
        rec = dict(record, checkpoint_sha256=pilot.sha(member / "checkpoint.pt"),
                   logits_sha256=pilot.sha(member / "outputs.npz"))
        (member / "receipt.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(json.dumps({"event": "physical_member_saved", "context": context, "seed": seed,
                          "seconds": time.monotonic() - started}), flush=True)

    probabilities, receipt, extra = provider.fit_predict(
        source, labels[source], target, outer_fold=outer, context=context,
        deadline=deadline, device="cuda", fit_callback=callback)
    np.savez_compressed(folder / "bank.npz", probabilities=probabilities,
                        member_logits=extra["member_logits"], mean_logits=extra["mean_logits"],
                        sample_ids=ids[target], users=users[target],
                        expert_names=np.asarray([receipt["expert_name"]]))
    (folder / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    verifier = __import__("aligned_multimodal.p433_physical_verify", fromlist=["verify_context"])
    verifier.verify_context(folder, expected)
    if time.monotonic() >= deadline:
        raise TimeoutError("P433 full context exceeded budget")
    return folder


def main(argv=None):
    ap = argparse.ArgumentParser(description="P433 full physical nested bank (12 contexts).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic(); deadline = started + 1800
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; P433 full fitting refuses to start")
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
    cache = PhysicalTokenCache(manifest=pilot.CANONICAL)
    inputs = _inputs(cache)
    if any(not p.exists() for p in inputs):
        raise FileNotFoundError(next(p for p in inputs if not p.exists()))
    provider = Provider(cache, ids, users)
    source_sha = {pilot._key(p): pilot.sha(p) for p in sources}
    input_sha = {pilot._key(p): pilot.sha(p) for p in inputs}
    spec = {"name": "P433 full original physical token expert", "mode": "full",
            "outer_folds": list(OUTER_FOLDS), "source_sha256": source_sha,
            "input_sha256": input_sha, "expert_names": [EXPERT_NAME],
            "promotion_allowed": False, "complete_p315": False,
            "target_achieved": False}
    out.mkdir(parents=False); register_experiment(out / "experiment_registry.json", spec)
    snapshot = out / "source_snapshot"; snapshot.mkdir()
    for path in sources:
        (snapshot / "__".join(path.resolve().relative_to(ROOT).parts)).write_bytes(path.read_bytes())
    report = {"mode": "full", "folds": {}, "contexts_completed": 0,
              "target_achieved": False, "complete_p315": False,
              "outer_accuracy_evaluated": False, "promotion_allowed": False,
              "submission_generated": False, "test_rows_loaded": 0}
    for outer in OUTER_FOLDS:
        train = np.flatnonzero(folds != outer); held = np.flatnonzero(folds == outer)
        requests = [("outer", train, held)]
        for inner, (tr, va) in enumerate(GroupKFold(3).split(train, labels[train], users[train])):
            requests.append((f"inner{inner}", train[tr], train[va]))
        inner_bank = np.zeros((len(train), 1, 40), dtype=np.float32)
        outer_bank = None; covered = np.zeros(len(train), dtype=np.int64); contexts = []
        for name, source, target in requests:
            folder = _write_context(out, name, outer, source, target, ids, labels, users,
                                    provider, cache, deadline, started)
            with np.load(folder / "bank.npz", allow_pickle=False) as archive:
                bank = archive["probabilities"]
            if name == "outer":
                outer_bank = bank
            else:
                positions = {sid: i for i, sid in enumerate(ids[train])}
                indices = np.asarray([positions[sid] for sid in ids[target]], dtype=np.int64)
                inner_bank[indices] = bank; covered[indices] += 1
            report["contexts_completed"] += 1
            contexts.append({"name": name, "source_rows": len(source), "target_rows": len(target),
                             "provenance_checked": True})
            print(json.dumps({"event": "physical_context_complete", "context": f"fold{outer}.{name}",
                              "seconds": time.monotonic() - started}), flush=True)
        if not np.all(covered == 1) or outer_bank is None:
            raise ProtocolError("P433 inner coverage invalid")
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
        raise TimeoutError("P433 full final hashing exceeded budget")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    contract = __import__("aligned_multimodal.p433_full_contract", fromlist=["verify_full"])
    contract.verify_full(out, require_process=False)
    report["elapsed_seconds"] = time.monotonic() - started
    if time.monotonic() >= deadline:
        raise TimeoutError("P433 full post-verification exceeded budget")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event": "physical_full_complete", "contexts": 12,
                      "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
