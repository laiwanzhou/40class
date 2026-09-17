"""Versioned score-blind P12 skeleton bank reconstruction."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from .p428_skeleton_provider import SkeletonProvider, select_epoch
from .p427_foundation_provider import array_hash
from .p419_vjepa_repeat_group_bridge import sha
from .p416_nested_frozen_family_router import EXCLUDED_USERS, OUTER_FOLDS
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance, register_experiment

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "aligned_multimodal"
CACHE = HERE / "cache/aligned_192x144"
PREREG = ROOT / "docs/research/STABLE_093_P428_SKELETON_REBUILD.md"


def verify_context(folder):
    folder = Path(folder)
    r = json.loads((folder / "provenance.json").read_text())
    if sha(folder / "checkpoint.pt") != r["checkpoint_sha256"]:
        raise ProtocolError("checkpoint hash mismatch")
    nodes = {k: ArtifactNode(k, tuple(v["parents"]), frozenset(v["supervised_train_subjects"]),
                v["provenance"], v["has_task_labels"], v.get("oof_labels_only", False)) for k, v in r["artifact_dag"].items()}
    for diagnostics, epochs in [(f["diagnostics"], 15) for f in r["selection_fits"]] + [(r["refit_diagnostics"], r["selected_epoch"])]:
        if (diagnostics["epochs_completed"] != epochs or len(diagnostics["losses"]) != epochs
                or not np.isfinite(diagnostics["losses"]).all()
                or len(diagnostics["successful_amp_steps"]) != epochs
                or min(diagnostics["successful_amp_steps"]) <= 0):
            raise ProtocolError("incomplete training diagnostics")
    state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    if not state or not all(torch.isfinite(v).all() for v in state.values()):
        raise ProtocolError("nonfinite checkpoint")
    for subject in r["target_users"]:
        assert_prediction_provenance(subject, r["prediction_nodes"], nodes)
    source_ids, source_users = np.asarray(r["source_ids"]), np.asarray(r["source_users"])
    if len(set(source_ids)) != len(source_ids) or len(r["selection_fits"]) != 3:
        raise ProtocolError("invalid selection inventory")
    for fit, (tr, va) in zip(r["selection_fits"], GroupKFold(3).split(source_ids, groups=source_users)):
        if (fit["train_ids"] != source_ids[tr].tolist() or fit["prediction_ids"] != source_ids[va].tolist()
                or fit["train_users"] != source_users[tr].tolist() or fit["prediction_users"] != source_users[va].tolist()
                or nodes[fit["node"]].supervised_train_subjects != frozenset(source_users[tr])):
            raise ProtocolError("source selection partition mismatch")
        for subject in set(source_users[va]):
            assert_prediction_provenance(subject, [fit["node"]], nodes)
    with np.load(folder / "bank.npz", allow_pickle=False) as z:
        p = z["probabilities"]
        if (p.shape != (len(r["target_ids"]), 1, 40) or not np.isfinite(p).all()
                or z["expert_names"].astype(str).tolist() != ["p12_skeleton"]
                or (p < 0).any() or not np.allclose(p.sum(2), 1)
                or not np.array_equal(z["sample_ids"], r["target_ids"])
                or not np.array_equal(z["users"], r["target_users"])
                or not np.array_equal(z["selection_ids"], r["source_ids"])
                or set(r["source_ids"]) & set(r["target_ids"])
                or set(r["source_users"]) & set(r["target_users"])):
            raise ProtocolError("bank IDs/probabilities invalid")
        if (array_hash(z["selection_logits"]) != r["selection_logit_sha256"]
                or array_hash(z["selection_labels"]) != r["source_label_sha256"]):
            raise ProtocolError("selection arrays changed")
        epoch, counts = select_epoch(z["selection_logits"], z["selection_labels"])
        if epoch != r["selected_epoch"] or counts.tolist() != r["source_epoch_correct"]:
            raise ProtocolError("epoch selection differs")
        logits = z["logits"]; expected = np.exp(logits - logits.max(1, keepdims=True))
        expected /= expected.sum(1, keepdims=True)
        if not np.allclose(expected, p[:, 0], atol=1e-7):
            raise ProtocolError("logits/probability mismatch")
    return r


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true"); mode.add_argument("--run", action="store_true")
    parser.add_argument("--out-dir", required=True); parser.add_argument("--pilot-dir")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    sources = [Path(__file__), PREREG, *[HERE / n for n in (
        "p428_skeleton_provider.py", "p428_skeleton_training.py", "p428_skeleton_data.py", "p428_watchdog.py",
        "aligned_data.py", "aligned_model.py", "train.py", "p90_teacher_common.py",
        "p416_nested_frozen_family_router.py", "p427_foundation_provider.py", "stable_routing_protocol.py")]]
    inputs = [CACHE / "metadata.json", CACHE / "skeleton_float32.npy", HERE / "data/manifest.csv",
              *[HERE / f"data/subject_folds/fold_{k}.csv" for k in OUTER_FOLDS],
              HERE / "runs/p0_tracking/fold_0_skeleton_first/config_used.json"]
    spec = {"name": "P428 source-only original skeleton", "mode": "pilot" if args.pilot else "full",
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        "expert_names": ["p12_skeleton"], "outer_accuracy_evaluated": False, "promotion_allowed": False}
    if args.run:
        if not args.pilot_dir:
            raise ProtocolError("full requires pilot")
        pilot = Path(args.pilot_dir)
        pr = json.loads((pilot / "summary.json").read_text())
        ps = json.loads((pilot / "experiment_registry.json").read_text())["spec"]
        if pr["mode"] != "pilot" or pr["elapsed_seconds"] > 900 or pr["contexts_completed"] != 1:
            raise ProtocolError("invalid pilot")
        for key in ("source_sha256", "input_sha256", "expert_names"):
            if ps[key] != spec[key]:
                raise ProtocolError("pilot recipe mismatch")
        for path, digest in pr["artifact_sha256"].items():
            if sha(pilot / path) != digest:
                raise ProtocolError("pilot artifacts changed")
        verify_context(pilot / "fold0/outer")
        spec["pilot_summary_sha256"] = sha(pilot / "summary.json")
    out.mkdir(parents=True)
    register_experiment(out / "experiment_registry.json", spec)
    snapshot = out / "source_snapshot"; snapshot.mkdir()
    for p in sources:
        (snapshot / p.name).write_bytes(p.read_bytes())
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; stop before real training")
    torch.cuda.set_per_process_memory_fraction(min(1., 4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory))
    started = time.monotonic(); deadline = started + (900 if args.pilot else 10800)
    protocol = load_protocol(); keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    ids, labels, users, folds = (v[keep] for v in (protocol.sample_ids, protocol.labels, protocol.users, protocol.fold_id))
    provider = SkeletonProvider(CACHE, ids)
    contexts = []
    print("P428 reconstructs the original skeleton slot; no held accuracy or Test evaluation.", flush=True)
    for fold in ((0,) if args.pilot else OUTER_FOLDS):
        source, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        requests = [("outer", source, held)]
        if not args.pilot:
            requests.extend((f"inner{k}", source[tr], source[va]) for k, (tr, va) in
                            enumerate(GroupKFold(3).split(source, groups=users[source])))
        for name, tr, va in requests:
            context = f"fold{fold}.{name}"; folder = out / f"fold{fold}" / name; folder.mkdir(parents=True)
            p, receipt, arrays, state = provider.fit_predict(tr, labels[tr], users[tr], va, users[va], context=context, deadline=deadline)
            torch.save(state, folder / "checkpoint.pt")
            receipt["checkpoint_sha256"] = sha(folder / "checkpoint.pt")
            np.savez_compressed(folder / "bank.npz", probabilities=p, sample_ids=ids[va], users=users[va],
                                expert_names=np.asarray(["p12_skeleton"]), **arrays)
            (folder / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            verify_context(folder)
            if args.run and context == "fold0.outer":
                with np.load(Path(args.pilot_dir) / "fold0/outer/bank.npz", allow_pickle=False) as prior:
                    for key, value in {"probabilities": p, **arrays}.items():
                        if not np.array_equal(value, prior[key]):
                            raise ProtocolError(f"pilot/full deterministic mismatch: {key}")
                prior_state = torch.load(Path(args.pilot_dir) / "fold0/outer/checkpoint.pt", map_location="cpu", weights_only=True)
                if state.keys() != prior_state.keys() or not all(torch.equal(state[k], prior_state[k]) for k in state):
                    raise ProtocolError("pilot/full weights differ")
            contexts.append({"context": context, "selected_epoch": receipt["selected_epoch"], "source_rows": len(tr), "target_rows": len(va)})
            if time.monotonic() >= deadline:
                raise TimeoutError("P428 context exceeded budget")
            print(json.dumps({"event": "skeleton_context_complete", **contexts[-1], "seconds": time.monotonic()-started}), flush=True)
    report = {"mode": spec["mode"], "contexts": contexts, "contexts_completed": len(contexts),
        "elapsed_seconds": time.monotonic()-started, "complete_p315": False, "target_achieved": False,
        "held_accuracy_evaluated": False, "test_rows_loaded": 0,
        "artifact_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()
                            and "source_snapshot" not in p.parts and p.name != "experiment_registry.json"}}
    if time.monotonic() >= deadline:
        raise TimeoutError("P428 reporting exceeded budget")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "notes.txt").write_text("Actual skeleton slot only; source-internal selection. No held score or promotion.\n", encoding="utf-8")
    print(json.dumps({"event": "skeleton_complete", "seconds": report["elapsed_seconds"], "contexts": len(contexts)}), flush=True)


if __name__ == "__main__":
    main()
