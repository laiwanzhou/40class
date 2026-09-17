"""Bounded P430 fold-0 rebuild using the original frozen VJEPA token cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# This must precede importing torch (including transitive imports).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .p90_teacher_common import load_protocol
from .p416_nested_frozen_family_router import EXCLUDED_USERS
from .p427_foundation_provider import array_hash
from .p430_token_cache import TokenCache
from .p430_token_provider import TokenProvider
from .stable_routing_protocol import ProtocolError, register_experiment

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "aligned_multimodal"
CANONICAL = HERE / "data" / "manifest.csv"
EXTRACTION = HERE / "data" / "p46_single_split.csv"
CACHE = ROOT / "runs" / "p96_vjepa2_vitl_ssv2_dense24_fold0_v1"
PREREG = ROOT / "docs" / "research" / "STABLE_093_P430_TOKEN_REBUILD.md"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "external/" + Path(path).name


def _source_files() -> list[Path]:
    names = ["p430_rebuild_token_bank.py", "p430_watchdog.py", "p430_token_cache.py",
             "p430_token_provider.py", "p430_token_training.py", "p142_vjepa_token_transformer_oof.py",
             "p96_vjepa2_dense24_extractor.py", "p90_videomae_lora_teacher.py", "p430_token_verify.py",
             "p90_teacher_common.py", "p416_nested_frozen_family_router.py",
             "p427_foundation_provider.py", "p419_vjepa_repeat_group_bridge.py", "stable_routing_protocol.py"]
    files = [HERE / n for n in names]
    # P90's loader supplies row order; no LoRA checkpoint enters VJEPA tokens.
    files += [PREREG]
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description="P430 fold-0 outer pilot (two real token experts, six members).")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic(); deadline = started + 900
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; P430 real fitting refuses to start")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4 * 1024**3 /
        torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    protocol = load_protocol()
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if int(keep.sum()) != 2470: raise ProtocolError("expected 2470 eligible canonical rows")
    ids, labels, users, folds = (protocol.sample_ids[keep], protocol.labels[keep],
                                  protocol.users[keep], protocol.fold_id[keep])
    source = np.flatnonzero(folds != 0); target = np.flatnonzero(folds == 0)
    if (len(source), len(target)) != (1497, 973):
        raise ProtocolError("expected fold-0 source/target sizes 1497/973")
    cache = TokenCache(CACHE, CANONICAL, EXTRACTION)
    provider = TokenProvider(cache.select(ids), ids, users, cache_provenance=cache.provenance)
    out.mkdir(parents=False)
    sources = _source_files()
    if any(not p.exists() for p in sources):
        raise FileNotFoundError(next(p for p in sources if not p.exists()))
    folds_csv = [HERE / "data" / "subject_folds" / f"fold_{k}.csv" for k in range(3)]
    cache_inputs = [CACHE/"features.npy", CACHE/"done.npy", CACHE/"cache_summary.json", CANONICAL, EXTRACTION]
    spec = {"name":"P430 original token experts", "mode":"pilot", "outer_fold":0,
            "source_sha256":{_key(p):sha(p) for p in sources},
            "input_sha256":{_key(p):sha(p) for p in [*folds_csv, *cache_inputs, PREREG]},
            "expert_names":["p142_all_token","p144_hand_interaction"], "outer_accuracy_evaluated":False,
            "promotion_allowed":False, "excluded_users":sorted(EXCLUDED_USERS)}
    register_experiment(out / "experiment_registry.json", spec)
    snap = out / "source_snapshot"; snap.mkdir()
    for p in sources:
        (snap / "__".join(Path(p).resolve().relative_to(ROOT).parts)).write_bytes(Path(p).read_bytes())
    expected = {"context":"fold0.outer", "outer_fold":0, "source_ids":ids[source].tolist(),
                "source_users":users[source].tolist(), "target_ids":ids[target].tolist(),
                "target_users":users[target].tolist(), "source_label_sha256":array_hash(labels[source]),
                "source_class_counts":np.bincount(labels[source], minlength=40).tolist(),
                "cache_provenance":cache.provenance}
    (out / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    folder = out / "fold0" / "outer"; folder.mkdir(parents=True); (folder / "members").mkdir()
    def callback(expert, seed, logits, state, record):
        member = folder / "members" / expert / f"seed{seed}"; member.mkdir(parents=True, exist_ok=False)
        torch.save(state, member / "checkpoint.pt"); np.savez_compressed(member / "outputs.npz", logits=logits)
        record = dict(record); record.update(checkpoint_sha256=sha(member/"checkpoint.pt"), logits_sha256=sha(member/"outputs.npz"))
        (member / "receipt.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({"event":"token_member_saved","expert":expert,"seed":seed,"seconds":time.monotonic()-started}),flush=True)
    probabilities, receipt, extra = provider.fit_predict(source, labels[source], target,
        outer_fold=0, context="fold0.outer", deadline=deadline, device="cuda", fit_callback=callback)
    np.savez_compressed(folder / "bank.npz", probabilities=probabilities, sample_ids=ids[target],
                        users=users[target], expert_names=np.asarray(receipt["expert_names"]), **extra)
    (folder / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    verifier = __import__("aligned_multimodal.p430_token_verify", fromlist=["verify_context"])
    verifier.verify_context(folder, expected)
    if {_key(p):sha(p) for p in sources}!=spec["source_sha256"]:
        raise ProtocolError("registered source changed during fit")
    current_inputs = {_key(p):sha(p) for p in [*folds_csv, *cache_inputs, PREREG]}
    if current_inputs != spec["input_sha256"]: raise ProtocolError("registered input changed during fit")
    if time.monotonic() >= deadline: raise TimeoutError("P430 reporting exceeded budget")
    report = {"mode":"pilot", "contexts_completed":1, "elapsed_seconds":time.monotonic()-started,
              "target_achieved":False,
              "complete_p315":False, "outer_accuracy_evaluated":False, "test_rows_loaded":0,
              "promotion_allowed":False, "artifact_sha256":{p.relative_to(out).as_posix():sha(p)
              for p in out.rglob("*") if p.is_file() and "source_snapshot" not in p.parts}}
    if time.monotonic()>=deadline:raise TimeoutError("P430 final hashing exceeded budget")
    report["elapsed_seconds"]=time.monotonic()-started
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event":"token_pilot_complete","seconds":report["elapsed_seconds"]}),flush=True)

if __name__ == "__main__": main()
