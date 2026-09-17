"""Rebuild the original thermal expert bank (the bounded fold-0 pilot)."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from .p90_teacher_common import load_protocol
from .p416_nested_frozen_family_router import EXCLUDED_USERS
from .p429_thermal_population import ThermalPopulation
from .p429_thermal_provider import ThermalProvider
from .p429_thermal_training import _WEIGHTS
from .p427_foundation_provider import array_hash
from .p419_vjepa_repeat_group_bridge import sha
from .stable_routing_protocol import ProtocolError, register_experiment

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "aligned_multimodal"
THERMAL_FOLD = ROOT / "thermal_baseline/data/subject_folds/fold_0.csv"
THERMAL_MANIFEST = ROOT / "thermal_baseline/data/manifest.csv"
CANONICAL_MANIFEST = HERE / "data/manifest.csv"
PREREG = ROOT / "docs/research/STABLE_093_P429_THERMAL_REBUILD.md"
CONFIG = ROOT / "thermal_baseline/runs/p11_thermal_imagenet/fold_0/config_used.json"

def _key(path):
    p = Path(path)
    try: return p.relative_to(ROOT).as_posix()
    except ValueError: return "external/" + p.name

def _sources():
    names = ["p429_rebuild_thermal_bank.py", "p429_watchdog.py", "p429_thermal_provider.py", "p429_thermal_training.py",
             "p429_thermal_data.py", "p429_thermal_population.py", "p90_teacher_common.py",
             "p416_nested_frozen_family_router.py", "p427_foundation_provider.py", "p419_vjepa_repeat_group_bridge.py",
             "stable_routing_protocol.py"]
    paths = [HERE / n for n in names] + [ROOT / "thermal_baseline/thermal_oof_data.py",
             ROOT / "thermal_baseline/thermal_tsm_model.py", ROOT / "thermal_baseline/train_imagenet_oof.py",
             PREREG, THERMAL_FOLD, THERMAL_MANIFEST, CONFIG]
    verifier = HERE / "p429_thermal_verify.py"
    if verifier.exists(): paths.append(verifier)
    return paths

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv); out = Path(a.out_dir)
    started = time.monotonic(); deadline = started + 7200
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; stop before real training")
    torch.cuda.set_per_process_memory_fraction(min(1., 8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory))
    protocol = load_protocol()
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if int(keep.sum()) != 2470: raise ProtocolError("expected 2470 eligible canonical rows")
    ids, labels, users, folds = protocol.sample_ids[keep], protocol.labels[keep], protocol.users[keep], protocol.fold_id[keep]
    population = ThermalPopulation(THERMAL_FOLD, ids, labels, users)
    source = np.flatnonzero(folds != 0); target = np.flatnonzero(folds == 0)
    source_i, source_y, source_u, target_ids, target_u = population.context(source, target)
    provider = ThermalProvider(population.paths, population.ids)

    out.mkdir(parents=False)
    sources = _sources()
    fold_inputs = [HERE / f"data/subject_folds/fold_{k}.csv" for k in range(3)]
    spec = {"name": "P429 original thermal expert", "mode": "pilot", "outer_fold": 0,
            "source_sha256": {_key(p): sha(p) for p in sources},
            "input_sha256": {_key(p): sha(p) for p in (CANONICAL_MANIFEST, *fold_inputs, THERMAL_FOLD, THERMAL_MANIFEST, CONFIG, Path(_WEIGHTS))},
            "expert_names": ["p12_thermal"], "outer_accuracy_evaluated": False,
            "promotion_allowed": False, "excluded_users": sorted(EXCLUDED_USERS)}
    register_experiment(out / "experiment_registry.json", spec)
    snap = out / "source_snapshot"; snap.mkdir()
    for p in sources:
        safe = "__".join(p.relative_to(ROOT).parts)
        (snap / safe).write_bytes(p.read_bytes())

    # Inventory is intentionally written before any supervised fitting.
    from thermal_baseline.thermal_oof_data import IMAGE_EXTENSIONS
    train_root = (ROOT / "Training").resolve(); raw = {}
    # Target IDs are canonical and may not be in the thermal manifest; use paths directly.
    requested = list(population.ids[source_i]) + [str(x) for x in target_ids if str(x) in population.paths]
    for sid in requested:
        trial = Path(population.paths[sid]).resolve()
        try: trial.relative_to(train_root)
        except ValueError: raise ProtocolError(f"raw path outside Training: {trial}")
        for frame in trial.iterdir():
            if frame.is_file() and frame.suffix.lower() in IMAGE_EXTENSIONS:
                if not frame.resolve().is_relative_to(train_root):
                    raise ProtocolError("raw frame resolves outside Training")
                raw[str(frame)] = sha(frame)
    (out / "raw_inventory.json").write_text(json.dumps({"raw_frame_sha256": raw}, indent=2), encoding="utf-8")
    print(json.dumps({"event":"raw_inventory_ready", "raw_files":len(raw)}), flush=True)
    expected = {"source_ids": population.ids[source_i].tolist(), "source_users": population.users[source_i].tolist(),
                "target_ids": target_ids.tolist(), "target_users": target_u.tolist(),
                "thermal_present": [str(x) in population.paths for x in target_ids],
                "source_label_sha256": array_hash(source_y)}
    (out / "context_expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    folder = out / "fold0" / "outer"; folder.mkdir(parents=True)
    members = folder / "members"; members.mkdir()
    def callback(name, logits, state, diagnostics, train_ids, prediction_ids):
        folder = members / name; folder.mkdir(exist_ok=False)
        torch.save(state, folder / "checkpoint.pt")
        np.savez_compressed(folder / "outputs.npz", logits=np.asarray(logits, dtype=np.float32))
        receipt = {"train_ids": np.asarray(train_ids).astype(str).tolist(), "prediction_ids": np.asarray(prediction_ids).astype(str).tolist(),
                   "diagnostics": diagnostics, "checkpoint_sha256": sha(folder / "checkpoint.pt"),
                   "logits_sha256": sha(folder / "outputs.npz")}
        (folder / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps({"event":"thermal_member_saved","member":name,"seconds":time.monotonic()-started}),flush=True)
    p, receipt, arrays, state = provider.fit_predict(source_i, source_y, population.users[source_i], target_ids, target_u,
        context="fold0.outer", deadline=deadline, fit_callback=callback)
    torch.save(state, folder / "checkpoint.pt"); receipt["checkpoint_sha256"] = sha(folder / "checkpoint.pt")
    np.savez_compressed(folder / "bank.npz", probabilities=p, sample_ids=target_ids, users=target_u,
                        expert_names=np.asarray(["p12_thermal"]), **arrays)
    (folder / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    verifier = __import__("aligned_multimodal.p429_thermal_verify", fromlist=["verify_context"])
    verifier.verify_context(folder, expected)
    # Recheck all registered code/inputs and every raw frame before publishing.
    current = {_key(p): sha(p) for p in sources}
    if current != spec["source_sha256"]:
        raise ProtocolError("registered source changed during fit")
    current_inputs = {_key(p): sha(p) for p in (CANONICAL_MANIFEST, *fold_inputs, THERMAL_FOLD, THERMAL_MANIFEST, CONFIG, Path(_WEIGHTS))}
    if current_inputs != spec["input_sha256"]:
        raise ProtocolError("registered input changed during fit")
    current_paths={str(frame) for sid in requested for frame in Path(population.paths[sid]).resolve().iterdir()
                   if frame.is_file() and frame.suffix.lower() in IMAGE_EXTENSIONS}
    if current_paths!=set(raw) or {k: sha(k) for k in raw} != raw:
        raise ProtocolError("raw frame changed during fit")
    if time.monotonic() >= deadline: raise TimeoutError("P429 reporting exceeded budget")
    report = {"mode":"pilot", "contexts_completed":1, "elapsed_seconds":time.monotonic()-started,
              "complete_p315":False, "target_achieved":False, "held_accuracy_evaluated":False, "test_rows_loaded":0,
              "artifact_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "experiment_registry.json" and "source_snapshot" not in p.parts}}
    if time.monotonic()>=deadline:raise TimeoutError("P429 final inventory exceeded budget")
    report["elapsed_seconds"]=time.monotonic()-started
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event":"thermal_complete", "seconds":report["elapsed_seconds"]}), flush=True)

if __name__ == "__main__": main()
