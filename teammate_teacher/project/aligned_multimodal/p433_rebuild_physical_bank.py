"""Bounded P433/P238 physical-token fold-0 source-only pilot."""
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
from .p433_physical_cache import PhysicalTokenCache
from .p433_physical_provider import EXCLUDED_USERS, Provider
from .p427_foundation_provider import array_hash
from .stable_routing_protocol import ProtocolError, register_experiment

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "aligned_multimodal"
CANONICAL = HERE / "data" / "manifest.csv"
PREREG = ROOT / "docs" / "research" / "STABLE_093_P433_PHYSICAL_PILOT.md"
PILOT_NAME = "p238_physical_token"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _key(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "external/" + Path(path).name


def _source_files() -> list[Path]:
    names = [
        "p433_rebuild_physical_bank.py", "p433_watchdog.py", "p433_physical_cache.py",
        "p433_physical_provider.py", "p433_physical_training.py", "p433_physical_verify.py",
        "p433_run_verify.py", "p238_physical_token_transformer_oof.py",
        "p142_vjepa_token_transformer_oof.py", "p90_teacher_common.py",
        "audit_p87_sequence_decoder.py", "p427_foundation_provider.py",
        "p427_foundation_kernels.py", "p90_videomaev2_distilled_teacher.py",
        "p90_internvideo2_l_teacher.py", "p91_videomaev2_modality_teacher.py",
        "stable_routing_protocol.py",
    ]
    return [HERE / name for name in names] + [PREREG]


def _input_files(cache: PhysicalTokenCache) -> list[Path]:
    folds = [HERE / "data" / "subject_folds" / f"fold_{k}.csv" for k in range(3)]
    summaries = [path.with_name("cache_summary.json") for path in cache.paths]
    return [*cache.paths, *summaries, CANONICAL, *folds, PREREG]


def main(argv=None):
    ap = argparse.ArgumentParser(description="P433 physical-token fold-0 outer pilot (three members).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic(); deadline = started + 900
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; P433 real fitting refuses to start")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4 * 1024**3 /
        torch.cuda.get_device_properties(0).total_memory))

    protocol = load_protocol()
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    if int(keep.sum()) != 2470:
        raise ProtocolError("expected 2470 eligible canonical rows")
    ids, labels, users, folds = (protocol.sample_ids[keep], protocol.labels[keep],
                                 protocol.users[keep], protocol.fold_id[keep])
    source = np.flatnonzero(folds != 0); target = np.flatnonzero(folds == 0)
    if (len(source), len(target)) != (1497, 973):
        raise ProtocolError("expected fold-0 source/target sizes 1497/973")
    cache = PhysicalTokenCache(manifest=CANONICAL)
    provider = Provider(cache, ids, users)
    out.mkdir(parents=False)
    sources = _source_files(); inputs = _input_files(cache)
    if any(not p.exists() for p in [*sources, *inputs]):
        raise FileNotFoundError(next(p for p in [*sources, *inputs] if not p.exists()))
    spec = {"name": "P433 original physical token expert", "mode": "pilot", "outer_fold": 0,
            "source_sha256": {_key(p): sha(p) for p in sources},
            "input_sha256": {_key(p): sha(p) for p in inputs},
            "expert_names": [PILOT_NAME], "outer_accuracy_evaluated": False,
            "promotion_allowed": False, "excluded_users": sorted(EXCLUDED_USERS)}
    register_experiment(out / "experiment_registry.json", spec)
    snapshot = out / "source_snapshot"; snapshot.mkdir()
    for path in sources:
        (snapshot / "__".join(path.resolve().relative_to(ROOT).parts)).write_bytes(path.read_bytes())
    expected = {"context": "fold0.outer", "outer_fold": 0,
                "source_ids": ids[source].tolist(), "source_users": users[source].tolist(),
                "target_ids": ids[target].tolist(), "target_users": users[target].tolist(),
                "source_label_sha256": array_hash(labels[source]),
                "source_class_counts": np.bincount(labels[source], minlength=40).tolist(),
                "cache_provenance": cache.provenance}
    (out / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    folder = out / "fold0" / "outer"; (folder / "members").mkdir(parents=True)

    def callback(seed, logits, state, record):
        member = folder / "members" / f"seed{seed}"; member.mkdir(parents=False, exist_ok=False)
        torch.save(state, member / "checkpoint.pt")
        np.savez_compressed(member / "outputs.npz", logits=logits)
        rec = dict(record, checkpoint_sha256=sha(member / "checkpoint.pt"),
                   logits_sha256=sha(member / "outputs.npz"))
        (member / "receipt.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(json.dumps({"event": "physical_member_saved", "seed": seed,
                          "seconds": time.monotonic() - started}), flush=True)

    probabilities, receipt, extra = provider.fit_predict(
        source, labels[source], target, outer_fold=0, context="fold0.outer",
        deadline=deadline, device="cuda", fit_callback=callback)
    np.savez_compressed(folder / "bank.npz", probabilities=probabilities,
                        member_logits=extra["member_logits"], mean_logits=extra["mean_logits"],
                        sample_ids=ids[target], users=users[target],
                        expert_names=np.asarray([receipt["expert_name"]]))
    (folder / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    verifier = __import__("aligned_multimodal.p433_physical_verify", fromlist=["verify_context"])
    verifier.verify_context(folder, expected)
    if {_key(p): sha(p) for p in sources} != spec["source_sha256"]:
        raise ProtocolError("registered source changed during fit")
    if {_key(p): sha(p) for p in inputs} != spec["input_sha256"]:
        raise ProtocolError("registered input changed during fit")
    if time.monotonic() >= deadline:
        raise TimeoutError("P433 reporting exceeded budget")
    report = {"mode": "pilot", "contexts_completed": 1,
              "elapsed_seconds": time.monotonic() - started, "target_achieved": False,
              "complete_p315": False, "outer_accuracy_evaluated": False,
              "promotion_allowed": False, "submission_generated": False,
              "test_rows_loaded": 0,
              "artifact_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*")
                                  if p.is_file() and "source_snapshot" not in p.parts}}
    if time.monotonic()>=deadline:raise TimeoutError('P433 final hashing exceeded budget')
    report['elapsed_seconds']=time.monotonic()-started
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event": "physical_pilot_complete", "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
