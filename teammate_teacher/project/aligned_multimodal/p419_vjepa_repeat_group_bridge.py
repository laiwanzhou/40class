"""One frozen V-JEPA addition to the verified P418 nested repeat bridge."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .p416_nested_frozen_family_router import (
    EXCLUDED_USERS, OUTER_FOLDS, _spec as family_spec, evaluate_outputs, load_frozen_families,
)
from .p418_nested_repeat_group_bridge import load_recording_metadata, run_outer_fold
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ProtocolError, register_experiment

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "aligned_multimodal"
VJEPA_CACHE = ROOT / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1/features.npy"
VJEPA_DONE = VJEPA_CACHE.with_name("done.npy")
VJEPA_SUMMARY = VJEPA_CACHE.with_name("cache_summary.json")
EXTRACTION_ROWS = HERE / "data/p46_single_split.csv"
METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
REFERENCE = HERE / "runs/p418_nested_repeat_group_bridge_v1"
PREREG = ROOT / "docs/research/STABLE_093_P419_PREREGISTRATION_2026-09-07.md"
MODEL_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
OUTPUT_KEYS = ("own_group", "repeat_group", "equal_mean", "selective_all")
REFERENCE_HASHES = {
    "predictions.npz": "00fbeeddd6c42891cf919151e03ca0488c4ad8c1c80a1c83ea688610a0c396f3",
    "experiment_registry.json": "52eed265622a26bd503947cccee26a13aa7a2897097ae95f4cd878d5a4991ffb",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def vjepa_dense24(features):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (24, 1024) or not np.isfinite(x).all():
        raise ProtocolError("V-JEPA cache shape/finite check failed")
    x = x / np.maximum(np.linalg.norm(x, axis=2, keepdims=True), 1e-12)
    x = x.reshape(len(x), 8, 3, 1024).mean(2)
    x = x / np.maximum(np.linalg.norm(x, axis=2, keepdims=True), 1e-12)
    return x.reshape(len(x), 8192)


def load_vjepa(cache=VJEPA_CACHE, done=VJEPA_DONE, summary=VJEPA_SUMMARY):
    if not all(Path(p).exists() for p in (cache, done, summary)):
        raise ProtocolError("incomplete V-JEPA cache artifacts")
    flags = np.load(done, allow_pickle=False)
    if flags.shape != (2914,) or flags.dtype != np.bool_ or not flags.all():
        raise ProtocolError("V-JEPA done.npy is incomplete or malformed")
    meta = json.loads(Path(summary).read_text(encoding="utf-8"))
    if (meta.get("model_repo") != MODEL_REPO or meta.get("label_free_extraction") is not True
            or meta.get("complete") is not True or meta.get("completed_samples") != 2914
            or meta.get("total_samples") != 2914 or meta.get("view_count") != 24):
        raise ProtocolError("V-JEPA frozen cache provenance/completion mismatch")
    x = np.load(cache, mmap_mode="r", allow_pickle=False)
    if x.shape != (2914, 24, 1024):
        raise ProtocolError(f"unexpected V-JEPA shape {x.shape}")
    return vjepa_dense24(x)


def assert_extraction_alignment(path, master_ids):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        ids = [row["sample_id"] for row in csv.DictReader(stream)]
    master = np.asarray(master_ids).astype(str)
    if len(ids) != len(set(ids)) or len(master) != len(set(master)) or set(ids) != set(master):
        raise ProtocolError("extraction row IDs do not uniquely match master")
    return {"method": "historical_source_attestation", "rows": len(ids),
            "loader_order": "p90_videomae_lora_teacher.read_aligned_rows reorders rows to master IDs",
            "limitation": "raw .npy has no embedded IDs; source/manifest attestation is not re-extraction"}


def load_reference(path, sample_ids, users, folds):
    with np.load(path, allow_pickle=False) as z:
        ids = z["sample_ids"].astype(str)
        target = np.asarray(sample_ids).astype(str)
        if len(set(ids)) != len(ids) or len(set(target)) != len(target) or set(ids) != set(target):
            raise ProtocolError("reference sample IDs are duplicated or mismatched")
        lookup = {sid: i for i, sid in enumerate(ids)}
        order = np.asarray([lookup[sid] for sid in target])
        if not np.array_equal(z["users"][order].astype(str), np.asarray(users).astype(str)):
            raise ProtocolError("reference subject alignment mismatch")
        if not np.array_equal(z["fold_id"][order], folds):
            raise ProtocolError("reference fold alignment mismatch")
        prediction = z["repeat_group"][order]
        if prediction.shape != (len(target),) or not np.issubdtype(prediction.dtype, np.integer):
            raise ProtocolError("reference prediction schema mismatch")
        if np.any((prediction < 0) | (prediction >= 40)):
            raise ProtocolError("reference prediction out of range")
        return prediction.copy()


def run_outer_fold5(families, labels, subjects, fold_id, fold, metadata):
    if len(families) != 5:
        raise ProtocolError("P419 requires exactly five registered families")
    return run_outer_fold(families, labels, subjects, fold_id, fold, metadata)


def reference_receipt():
    for name, expected_hash in REFERENCE_HASHES.items():
        if sha(REFERENCE / name) != expected_hash:
            raise ProtocolError("P418 frozen reference hash mismatch")
    registry = json.loads((REFERENCE / "experiment_registry.json").read_text(encoding="utf-8"))
    register_experiment(REFERENCE / "experiment_registry.json", registry["spec"])
    expected = registry["spec"]["source_sha256"]
    expected_names = {Path(key).name for key in expected if Path(key).suffix in (".py", ".md")}
    actual_names = {source.name for source in (REFERENCE / "source_snapshot").iterdir()}
    if actual_names != expected_names:
        raise ProtocolError("P418 frozen source snapshot set mismatch")
    for source in (REFERENCE / "source_snapshot").iterdir():
        keys = [key for key in expected if Path(key).name == source.name]
        if len(keys) != 1 or sha(source) != expected[keys[0]]:
            raise ProtocolError("P418 frozen source snapshot mismatch")
    return {"registry_sha256": sha(REFERENCE / "experiment_registry.json"),
            "prediction_sha256": sha(REFERENCE / "predictions.npz"),
            "summary_sha256": sha(REFERENCE / "summary.json"), "snapshot_verified": True}


def main(argv=None):
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite: {out}")
    out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), *[HERE / name for name in (
        "p418_nested_repeat_group_bridge.py", "p416_nested_frozen_family_router.py",
        "stable_routing_structure.py", "stable_routing_protocol.py", "p90_teacher_common.py",
        "p96_vjepa2_dense24_extractor.py", "p90_videomae_lora_teacher.py")], PREREG]
    inputs = [VJEPA_CACHE, VJEPA_DONE, VJEPA_SUMMARY, EXTRACTION_ROWS, METADATA]
    receipt = reference_receipt()
    spec = {"experiment": "P419_vjepa_repeat_group_bridge", "base_family_spec": family_spec(),
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
            "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
            "reference": receipt, "primary": "repeat_group_minus_frozen_p418_repeat",
            "vjepa_recipe": "view L2; mean groups of 3; group L2; flatten8192; P416 Ridge recipe",
            "group_head": {"C": .03, "solver": "lbfgs", "max_iter": 2000, "class_weight": None},
            "mode": "pilot" if args.pilot else "run", "cpu_threads": 4,
            "executed_outer_folds": [0] if args.pilot else list(OUTER_FOLDS), "promotion_allowed": False}
    register_experiment(out / "experiment_registry.json", spec)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for path in source_paths:
        (snapshot / path.name).write_bytes(path.read_bytes())
    started = time.monotonic()
    protocol = load_protocol()
    attestation = assert_extraction_alignment(EXTRACTION_ROWS, protocol.sample_ids)
    families, y, users, folds, ids = load_frozen_families(protocol)
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    families.append(load_vjepa()[keep])
    metadata = load_recording_metadata(METADATA, ids)
    baseline = load_reference(REFERENCE / "predictions.npz", ids, users, folds)
    complete = {key: np.full(len(ids), -1, dtype=int) for key in OUTPUT_KEYS}
    logs = {}
    for fold in ((0,) if args.pilot else OUTER_FOLDS):
        with threadpool_limits(limits=4), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            held, predictions, audit = run_outer_fold5(families, y, users, folds, fold, metadata)
        for key in complete:
            complete[key][held] = predictions[key]
        arrays = {key: audit.pop(key) for key in ("inner_probability_bank", "outer_probability_bank",
                                                  "own_probability", "repeat_probability")}
        np.savez_compressed(out / f"fold{fold}_banks.npz", **arrays,
                            inner_sample_ids=np.asarray(audit["outer_train_ids"]),
                            outer_sample_ids=np.asarray(audit["outer_held_ids"]))
        (out / f"fold{fold}_provenance.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        logs[str(fold)] = audit
        print(json.dumps({"event": "outer_fold_complete", "fold": fold,
                          "elapsed_seconds": time.monotonic()-started}), flush=True)
    np.savez_compressed(out / "predictions.npz", sample_ids=ids, users=users, fold_id=folds,
                        p418_repeat=baseline, **complete)
    payload = {"mode": spec["mode"], "folds": logs, "attestation": attestation,
               "target_achieved": False, "independent_confirmation": False, "test_rows_loaded": 0,
               "warning": "Historical-subject frozen-feature bridge; not P315 reconstruction or Test confirmation."}
    if args.run:
        if any(np.any(values < 0) for values in complete.values()):
            raise ProtocolError("incomplete predictions")
        result = evaluate_outputs({"p418_repeat": baseline, **complete}, y, users, folds, base_key="p418_repeat")
        for name, metrics in result.items():
            if name != "repeat_group":
                metrics.pop("mechanism_gate_pass", None)
                metrics.pop("criterion", None)
        secondary = evaluate_outputs({"own_group": complete["own_group"], "repeat_group": complete["repeat_group"]},
                                     y, users, folds, base_key="own_group")["repeat_group"]
        secondary.pop("mechanism_gate_pass", None)
        secondary.pop("criterion", None)
        payload.update(evaluation=result, secondary_repeat_minus_own=secondary,
                       primary_contrast=spec["primary"], primary_mechanism_gate_pass=result["repeat_group"]["mechanism_gate_pass"])
    payload["elapsed_seconds"] = time.monotonic()-started
    (out / ("pilot.json" if args.pilot else "summary.json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "notes.txt").write_text("Frozen five-family repeat versus P418 four-family repeat. No Test. "
                                  "Full metrics are exploratory; pilot does not score.\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "mode": spec["mode"],
                      "elapsed_seconds": payload["elapsed_seconds"], "target_achieved": False}), flush=True)


if __name__ == "__main__":
    main()
