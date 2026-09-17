"""Fixed P87 beam recipe, source-only transitions, and P419 held emissions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .audit_p87_sequence_decoder import fit_transition_model, decode_unique_beam
from .stable_routing_structure import build_safe_sessions, _timestamp_tie_mask
from .stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance, register_experiment
from .p418_nested_repeat_group_bridge import load_recording_metadata, _subset_metadata
from .p416_nested_frozen_family_router import evaluate_outputs, EXCLUDED_USERS, OUTER_FOLDS
from .p419_vjepa_repeat_group_bridge import sha
from .p90_teacher_common import load_protocol

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "aligned_multimodal"
SOURCE = HERE / "runs/p419_vjepa_repeat_group_bridge_v1"
PREREG = ROOT / "docs/research/STABLE_093_P420_PREREGISTRATION_2026-09-07.md"
METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
SOURCE_HASHES = {
    "predictions.npz": "1974ef827ca64a3a54e67bb28bf60026d14a85acc04dad471f348e3e8505fb17",
    "experiment_registry.json": "9f2c1c405a741c62ee11e542c897bdcf80420ea0b0e8e506ddb2587407a1e7d0",
}


def eligible_sessions(metadata):
    sessions, audit = build_safe_sessions(metadata, 30.0)
    tied = _timestamp_tie_mask(metadata)
    eligible = [s for s in sessions if 2 <= len(s) <= 40 and not tied[s].any()]
    audit.update(eligible_sessions=len(eligible), eligible_rows=sum(map(len, eligible)),
                 overlength_sessions=sum(len(s) > 40 for s in sessions))
    return eligible, audit


def run_outer_fold(probability, labels, subjects, folds, fold, metadata, upstream_nodes):
    """Only labels[train] are passed to transition fitting; held labels are inert."""
    train = np.flatnonzero(np.asarray(folds) != fold)
    held = np.flatnonzero(np.asarray(folds) == fold)
    users = np.asarray(subjects).astype(str)
    if not len(train) or not len(held) or set(users[train]) & set(users[held]):
        raise ProtocolError("invalid subject-disjoint sequence split")
    p = np.asarray(probability, float)[held]
    if p.shape != (len(held), 40) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ProtocolError("invalid sequence emission probability")
    src_meta = _subset_metadata(metadata, train)
    dst_meta = _subset_metadata(metadata, held)
    source_sessions, source_audit = eligible_sessions(src_meta)
    target_sessions, target_audit = eligible_sessions(dst_meta)
    transition = fit_transition_model(np.asarray(labels)[train], source_sessions, 40, 1.0, alpha=.25)
    base = p.argmax(1)
    mixture = .65 * p + .35 * np.eye(40)[base]
    mixture /= mixture.sum(1, keepdims=True)
    emission = np.log(np.clip(mixture, 1e-9, 1))
    outputs = {"p419_repeat": base, "unique_only": base.copy(), "source_sequence": base.copy()}
    for session in target_sessions:
        for name, weight in (("unique_only", 0.0), ("source_sequence", .45)):
            outputs[name][session] = decode_unique_beam(emission[session], transition, weight, 50)
    nodes = dict(upstream_nodes)
    parent = f"repeat_group.fold{fold}.head.prediction"
    for subject in np.unique(users[held]):
        assert_prediction_provenance(str(subject), [parent], nodes)
    trans_id = f"source_transition.fold{fold}"
    nodes[trans_id] = ArtifactNode(trans_id, provenance="supervised", has_task_labels=True,
                                   supervised_train_subjects=frozenset(users[train]))
    for name in ("unique_only", "source_sequence"):
        pid = f"{name}.fold{fold}.prediction"
        nodes[pid] = ArtifactNode(pid, parents=(parent, trans_id))
        for subject in np.unique(users[held]):
            assert_prediction_provenance(str(subject), [pid], nodes)
    log = {"outer_train_ids": np.asarray(metadata.sample_ids)[train].astype(str).tolist(),
           "outer_held_ids": np.asarray(metadata.sample_ids)[held].astype(str).tolist(),
           "train_subjects": sorted(set(users[train])), "held_subjects": sorted(set(users[held])),
           "source_geometry": source_audit, "target_geometry": target_audit,
           "source_sessions_local_indices": [s.tolist() for s in source_sessions],
           "target_sessions_local_indices": [s.tolist() for s in target_sessions],
           "provenance_checked": True, "artifact_dag": {
               k: {"node_id": k, "parents": list(v.parents), "provenance": v.provenance,
                   "has_task_labels": v.has_task_labels,
                   "supervised_train_subjects": sorted(v.supervised_train_subjects)} for k, v in nodes.items()}}
    return held, outputs, log


def main(argv=None):
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, expected in SOURCE_HASHES.items():
        if sha(SOURCE / name) != expected: raise ProtocolError("P419 frozen source mismatch")
    sources = [Path(__file__), PREREG, *[HERE / name for name in (
        "audit_p87_sequence_decoder.py", "stable_routing_structure.py", "stable_routing_protocol.py",
        "p418_nested_repeat_group_bridge.py", "p416_nested_frozen_family_router.py",
        "p419_vjepa_repeat_group_bridge.py", "p90_teacher_common.py")]]
    inputs = [METADATA, HERE / "data/manifest.csv", SOURCE / "predictions.npz", SOURCE / "summary.json",
              SOURCE / "experiment_registry.json", *[HERE / f"data/subject_folds/fold_{f}.csv" for f in OUTER_FOLDS],
              *[SOURCE / f"fold{f}_banks.npz" for f in OUTER_FOLDS],
              *[SOURCE / f"fold{f}_provenance.json" for f in OUTER_FOLDS]]
    spec = {"experiment": "P420_source_only_session_bridge", "mode": "pilot" if args.pilot else "run",
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
            "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
            "recipe": {"emission": .65, "own": .35, "transition": .45, "beam": 50,
                       "backoff": 1.0, "alpha": .25, "session_gap": 30.0, "eligible_length": [2, 40]},
            "primary": "source_sequence_minus_p419_repeat", "cpu_threads": 4, "promotion_allowed": False}
    register_experiment(out / "experiment_registry.json", spec)
    snap = out / "source_snapshot"; snap.mkdir()
    for source in sources: (snap / source.name).write_bytes(source.read_bytes())
    started = time.monotonic()
    protocol = load_protocol()
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    ids, y, users, folds = (v[keep] for v in (protocol.sample_ids, protocol.labels, protocol.users, protocol.fold_id))
    metadata = load_recording_metadata(METADATA, ids)
    with np.load(SOURCE / "predictions.npz", allow_pickle=False) as z:
        for key, value in (("sample_ids", ids), ("users", users), ("fold_id", folds)):
            if not np.array_equal(z[key], value): raise ProtocolError("P419 input order mismatch")
        base = z["repeat_group"].copy()
    probability = np.zeros((len(ids), 40))
    graphs = {}
    for fold in OUTER_FOLDS:
        held = np.flatnonzero(folds == fold)
        with np.load(SOURCE / f"fold{fold}_banks.npz", allow_pickle=False) as z:
            if not np.array_equal(z["outer_sample_ids"], ids[held]): raise ProtocolError("P419 bank ID mismatch")
            probability[held] = z["repeat_probability"]
        log = json.loads((SOURCE / f"fold{fold}_provenance.json").read_text(encoding="utf-8"))
        graphs[fold] = {k: ArtifactNode(k, parents=tuple(v["parents"]), provenance=v["provenance"],
            has_task_labels=v["has_task_labels"], supervised_train_subjects=frozenset(v["supervised_train_subjects"]))
            for k, v in log["artifact_dag"].items()}
    if not np.array_equal(probability.argmax(1), base): raise ProtocolError("P419 probability/base mismatch")
    complete = {key: np.full(len(ids), -1, int) for key in ("p419_repeat", "unique_only", "source_sequence")}
    logs = {}
    for fold in ((0,) if args.pilot else OUTER_FOLDS):
        with threadpool_limits(limits=4):
            held, output, logs[str(fold)] = run_outer_fold(probability, y, users, folds, fold, metadata, graphs[fold])
        for key in complete: complete[key][held] = output[key]
        print(json.dumps({"event": "fold_complete", "fold": fold, "elapsed_seconds": time.monotonic()-started}), flush=True)
    np.savez_compressed(out / "predictions.npz", sample_ids=ids, users=users, fold_id=folds, **complete)
    result = {"mode": spec["mode"], "folds": logs, "target_achieved": False,
              "independent_confirmation": False, "test_rows_loaded": 0}
    if args.run:
        if any((v < 0).any() for v in complete.values()): raise ProtocolError("incomplete sequence outputs")
        result["evaluation"] = evaluate_outputs(complete, y, users, folds, base_key="p419_repeat")
        for key in ("p419_repeat", "unique_only"):
            result["evaluation"][key].pop("criterion", None)
            result["evaluation"][key].pop("mechanism_gate_pass", None)
        result["primary_mechanism_gate_pass"] = result["evaluation"]["source_sequence"]["mechanism_gate_pass"]
    result["elapsed_seconds"] = time.monotonic()-started
    (out / ("pilot.json" if args.pilot else "summary.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "notes.txt").write_text("Fixed source-only P87 beam on P419; no Test. Results are exploratory.\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "mode": spec["mode"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__": main()
