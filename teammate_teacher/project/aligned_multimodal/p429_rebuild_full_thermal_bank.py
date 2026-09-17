"""Build the complete, subject-disjoint P429 thermal probability bank.

The fold-0 pilot is a verified, bit-exact seed artifact.  Every other context
is fitted independently; in particular, no pilot epoch/checkpoint is reused.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from . import p429_rebuild_thermal_bank as pilot
from .p416_nested_frozen_family_router import EXCLUDED_USERS
from .p427_foundation_provider import array_hash
from .p429_thermal_population import ThermalPopulation
from .p429_thermal_provider import ThermalProvider
from .p429_full_contract import verify_context
from .p90_teacher_common import load_protocol
from .p419_vjepa_repeat_group_bridge import sha
from .stable_routing_protocol import ProtocolError, register_experiment

ROOT, HERE = pilot.ROOT, pilot.HERE
PREREG = ROOT / "docs" / "research" / "STABLE_093_P429_FULL_BANK.md"
THERMAL_MANIFEST = pilot.THERMAL_FOLD
CANONICAL_MANIFEST = pilot.CANONICAL_MANIFEST
FOLDS = tuple(range(3))
TOTAL_BUDGET = 43200.0
CONTEXT_BUDGET = 7200.0


def _active_context(out,name):
    temporary=out/"active_context.tmp"
    temporary.write_text(json.dumps({"context":name}),encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(out/"active_context.json")
            return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.05)


def _sources() -> list[Path]:
    """Include the immutable pilot inventory plus this runner/watchdog/contract."""
    old = list(pilot._sources())
    extras = [HERE / "p429_rebuild_full_thermal_bank.py",
              HERE / "p429_full_watchdog.py", HERE / "p429_full_contract.py", PREREG]
    return old + extras


def _inputs() -> list[Path]:
    return [CANONICAL_MANIFEST, THERMAL_MANIFEST, pilot.THERMAL_MANIFEST,
            *[HERE / "data" / "subject_folds" / f"fold_{k}.csv" for k in FOLDS],
            pilot.CONFIG, Path(pilot._WEIGHTS)]


def _expected(context: str, source_i, target_i, population: ThermalPopulation) -> dict:
    return {
        "source_ids": population.ids[source_i].tolist(),
        "source_users": population.users[source_i].tolist(),
        "target_ids": population.canonical_ids[target_i].tolist(),
        "target_users": population.canonical_users[target_i].tolist(),
        "thermal_present": population.present[target_i].tolist(),
        "source_label_sha256": array_hash(population.labels[source_i]),
    }


def _raw_inventory(out: Path, population: ThermalPopulation) -> dict[str, str]:
    """Hash every thermal frame belonging to the 2513-row research universe."""
    train_root = (ROOT / "Training").resolve()
    research = np.isin(population.users.astype(str), np.unique(population.canonical_users))
    raw: dict[str, str] = {}
    for sid in population.ids[research]:
        path = Path(population.paths[str(sid)]).resolve()
        try:
            path.relative_to(train_root)
        except ValueError as exc:
            raise ProtocolError(f"raw path outside Training: {path}") from exc
        if not path.is_dir():
            raise ProtocolError(f"thermal sample path is not a directory: {path}")
        for frame in path.iterdir():
            if frame.is_file() and frame.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                resolved = frame.resolve()
                try:
                    resolved.relative_to(train_root)
                except ValueError as exc:
                    raise ProtocolError(f"raw frame outside Training: {resolved}") from exc
                raw[str(resolved)] = sha(resolved)
    (out / "raw_inventory.json").write_text(
        json.dumps({"thermal_rows": int(research.sum()), "raw_frame_sha256": raw}, indent=2),
        encoding="utf-8")
    return raw


def _check_raw(raw: dict[str, str], population) -> None:
    research=np.isin(population.users.astype(str),np.unique(population.canonical_users))
    current={str(frame.resolve()) for sid in population.ids[research]
             for frame in Path(population.paths[str(sid)]).iterdir()
             if frame.is_file() and frame.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}}
    if current!=set(raw) or {p: sha(p) for p in raw} != raw:
        raise ProtocolError("thermal raw frame changed during fit")


def _write_context(out, name, outer, source_c, target_c, protocol, population, provider,
                   deadline, started):
    context = f"fold{outer}.{name}"
    source_i, source_y, source_u, target_ids, target_u = population.context(source_c, target_c)
    folder = out / f"fold{outer}" / name
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "members").mkdir()
    expected = _expected(context, source_i, target_c, population)
    (folder / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")

    def callback(member, logits, state, diagnostics, train_ids, prediction_ids):
        mf = folder / "members" / member
        mf.mkdir(exist_ok=False)
        torch.save(state, mf / "checkpoint.pt")
        np.savez_compressed(mf / "outputs.npz", logits=np.asarray(logits, dtype=np.float32))
        receipt = {"train_ids": np.asarray(train_ids).astype(str).tolist(),
                   "prediction_ids": np.asarray(prediction_ids).astype(str).tolist(),
                   "diagnostics": diagnostics,
                   "checkpoint_sha256": sha(mf / "checkpoint.pt"),
                   "logits_sha256": sha(mf / "outputs.npz")}
        (mf / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps({"event": "thermal_member_saved", "context": context,
                          "member": member, "seconds": time.monotonic() - started}), flush=True)

    probabilities, receipt, arrays, state = provider.fit_predict(
        source_i, source_y, source_u, target_ids, target_u, context=context,
        deadline=deadline, device="cuda", fit_callback=callback)
    np.savez_compressed(folder / "bank.npz", probabilities=probabilities,
                        sample_ids=target_ids, users=target_u,
                        expert_names=np.asarray(["p12_thermal"]), **arrays)
    torch.save(state, folder / "checkpoint.pt")
    receipt["checkpoint_sha256"] = sha(folder / "checkpoint.pt")
    (folder / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    verify_context(folder, expected)
    if time.monotonic()>=deadline:raise TimeoutError("thermal full context exceeded budget")
    return folder


def main(argv=None):
    ap = argparse.ArgumentParser(description="P429 full thermal bank")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot-dir", required=True)
    args = ap.parse_args(argv)
    out, pilot_dir = Path(args.out_dir), Path(args.pilot_dir)
    if out.exists():
        raise FileExistsError(out)
    if not pilot_dir.exists():
        raise FileNotFoundError(pilot_dir)
    started = time.monotonic(); total_deadline = started + TOTAL_BUDGET
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; P429 full fitting refuses to start")
    torch.cuda.set_per_process_memory_fraction(min(1., 8 * 1024**3 /
                                                    torch.cuda.get_device_properties(0).total_memory))
    protocol = load_protocol()
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if int(keep.sum()) != 2470:
        raise ProtocolError("expected 2470 eligible canonical rows")
    ids, labels, users, folds = (v[keep] for v in
                                  (protocol.sample_ids, protocol.labels, protocol.users, protocol.fold_id))
    population = ThermalPopulation(THERMAL_MANIFEST, ids, labels, users)
    provider = ThermalProvider(population.paths, population.ids)
    # Root-owned contract validates the completed pilot before any output is created.
    contract = __import__("aligned_multimodal.p429_full_contract",
                          fromlist=["validate_pilot", "verify_full"])
    f0_source, f0_target = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    # Re-derive the pilot population from the pinned Train manifests, not from
    # the receipt we are about to validate.
    expected_f0 = contract.expected_context(population,f0_source,f0_target)
    pilot_receipt = contract.validate_pilot(pilot_dir, expected_f0)
    print(json.dumps({"event":"thermal_pilot_revalidated"}),flush=True)
    out.mkdir(parents=False)
    sources, inputs = _sources(), _inputs()
    if any(not p.exists() for p in [*sources, *inputs]):
        raise FileNotFoundError(next(p for p in [*sources, *inputs] if not p.exists()))
    source_sha = {pilot._key(p): sha(p) for p in sources}
    input_sha = {pilot._key(p): sha(p) for p in inputs}
    register_experiment(out / "experiment_registry.json", {
        "name": "P429 full original thermal expert", "mode": "full", "outer_folds": list(FOLDS),
        "source_sha256": source_sha, "input_sha256": input_sha,
        "expert_names": ["p12_thermal"], "pilot_receipt": pilot_receipt,
        "promotion_allowed": False, "complete_p315": False, "target_achieved": False,
        "excluded_users": sorted(EXCLUDED_USERS)})
    snap = out / "source_snapshot"; snap.mkdir()
    for p in sources:
        (snap / "__".join(p.resolve().relative_to(ROOT).parts)).write_bytes(p.read_bytes())
    raw = _raw_inventory(out, population)
    print(json.dumps({"event":"full_raw_inventory_ready","frames":len(raw)}),flush=True)
    # Copy only the verified pilot context.  All eleven other contexts are fresh fits.
    copied = out / "fold0" / "outer"
    shutil.copytree(pilot_dir / "fold0" / "outer", copied)
    (copied / "expected.json").write_text(json.dumps(expected_f0, indent=2), encoding="utf-8")
    contract.verify_reused_context(copied,pilot_receipt,expected_f0)
    report = {"mode": "full", "folds": {}, "contexts_completed": 1,
              "target_achieved": False, "complete_p315": False,
              "outer_accuracy_evaluated": False, "test_rows_loaded": 0}
    for outer in FOLDS:
        train = np.flatnonzero(folds != outer); held = np.flatnonzero(folds == outer)
        requests = [("outer", train, held)]
        requests.extend((f"inner{k}", train[tr], train[va]) for k, (tr, va) in
                         enumerate(GroupKFold(3).split(train, labels[train], groups=users[train])))
        inner_bank = np.zeros((len(train), 1, 40), np.float32)
        inner_present = np.zeros(len(train), dtype=bool); outer_bank = None; outer_present = None
        coverage=np.zeros(len(train),int)
        contexts = []
        for name, src, tgt in requests:
            if outer == 0 and name == "outer":
                folder = copied
            else:
                context_started = time.monotonic()
                deadline = min(total_deadline, context_started + CONTEXT_BUDGET)
                _active_context(out,f"fold{outer}.{name}")
                folder = _write_context(out, name, outer, src, tgt, protocol, population,
                                        provider, deadline, started)
                _active_context(out,None)
            with np.load(folder / "bank.npz", allow_pickle=False) as z:
                bank, present = z["probabilities"], z["thermal_present"].astype(bool)
            if name == "outer": outer_bank, outer_present = bank, present
            else:
                pos = {sid: i for i, sid in enumerate(ids[train])}
                ix = np.asarray([pos[s] for s in ids[tgt]])
                inner_bank[ix], inner_present[ix] = bank, present
                coverage[ix]+=1
            contexts.append({"name": name, "source_rows": len(json.loads((folder/"expected.json").read_text())["source_ids"]),
                             "target_rows": len(tgt), "provenance_checked": True})
            report["contexts_completed"] += (0 if (outer == 0 and name == "outer") else 1)
            print(json.dumps({"event": "thermal_context_complete", "context": f"fold{outer}.{name}",
                              "seconds": time.monotonic() - started}), flush=True)
        if not np.all(coverage==1) or not np.array_equal(inner_present,population.present[train]):
            raise ProtocolError("inner thermal coverage invalid")
        np.savez_compressed(out / f"fold{outer}_banks.npz", inner_sample_ids=ids[train],
                            outer_sample_ids=ids[held], inner_users=users[train], outer_users=users[held],
                            expert_names=np.asarray(["p12_thermal"]), inner_probability_bank=inner_bank,
                            outer_probability_bank=outer_bank, inner_thermal_present=inner_present,
                            outer_thermal_present=outer_present)
        report["folds"][str(outer)] = {"contexts": contexts, "contexts_completed": 4,
                                       "provenance_checked": True}
    _check_raw(raw,population)
    if {pilot._key(p): sha(p) for p in sources} != source_sha or {pilot._key(p): sha(p) for p in inputs} != input_sha:
        raise ProtocolError("registered source/input changed during fit")
    report["verification"] = contract.verify_full(out, population, ids, users, folds)
    contract.verify_reused_context(copied,pilot_receipt,expected_f0)
    report["elapsed_seconds"] = time.monotonic() - started
    report["artifact_sha256"] = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*")
                                  if p.is_file() and "source_snapshot" not in p.parts}
    if time.monotonic() >= total_deadline:
        raise TimeoutError("P429 full final verification exceeded budget")
    report["elapsed_seconds"]=time.monotonic()-started
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event": "thermal_full_complete", "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
