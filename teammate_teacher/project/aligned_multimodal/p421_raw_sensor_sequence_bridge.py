"""Fixed raw IMU/skeleton expert addition to the nested P420 pipeline."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .p416_nested_frozen_family_router import (
    EXCLUDED_USERS, OUTER_FOLDS, _spec as family_spec, fit_family_head,
    load_frozen_families, evaluate_outputs,
)
from .p418_nested_repeat_group_bridge import run_outer_fold as group_fold, load_recording_metadata
from .p419_vjepa_repeat_group_bridge import (
    sha, load_vjepa, assert_extraction_alignment, EXTRACTION_ROWS,
    VJEPA_CACHE, VJEPA_DONE, VJEPA_SUMMARY,
)
from .p420_source_only_session_bridge import run_outer_fold as sequence_fold
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ArtifactNode, ProtocolError, register_experiment

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "aligned_multimodal"
PREREG = ROOT / "docs/research/STABLE_093_P421_PREREGISTRATION_2026-09-07.md"
META = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
IMU_FEATURES = HERE / "runs/p89_imu_orientation_expert_v1/features.npy"
IMU_INDEX = HERE / "cache/imu_32/index.csv"
SKEL_FEATURES = HERE / "runs/p89_skeleton_invariant_expert_v1/train_features.npy"
SKEL_INDEX = HERE / "runs/p86_motion_window_cache_t16_v1/rows.csv"
REFERENCE = HERE / "runs/p420_source_only_session_bridge_v1"
REFERENCE_HASHES = {
    "predictions.npz": "cf2ea49374399c90d20af62a2ba13d6c94afd77a67929285d8e8ee774f4e2243",
    "experiment_registry.json": "74564ac8a52c9ea1e95163affc4b88e9ca8880e49d8d121389941c1500e77db5",
}
FITTER_RECIPE = {
    "imu": {"n_estimators": 600, "max_depth": 24},
    "skeleton": {"n_estimators": 700, "max_depth": 28},
}


def load_descriptor(path, index_path, target_ids, width, required_split=None):
    """Join by CSV position: cached descriptor extraction iterated CSV rows."""
    with Path(index_path).open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    source_ids = [r["sample_id"] for r in records]
    ids = np.asarray(target_ids).astype(str)
    if len(set(source_ids)) != len(source_ids) or len(set(ids)) != len(ids):
        raise ProtocolError("duplicate descriptor IDs")
    lookup = {sid: i for i, sid in enumerate(source_ids)}
    if any(sid not in lookup for sid in ids): raise ProtocolError("missing descriptor ID")
    positions = np.asarray([lookup[sid] for sid in ids], dtype=int)
    if required_split and any(records[i].get("split") != required_split for i in positions):
        raise ProtocolError("descriptor split mismatch")
    cache = np.load(path, mmap_mode="r", allow_pickle=False)
    if cache.shape != (len(records), width): raise ProtocolError("descriptor cache/index shape mismatch")
    features = np.asarray(cache[positions], dtype=np.float32)
    if not np.isfinite(features).all(): raise ProtocolError("nonfinite raw descriptors")
    audit = {"rows": len(ids), "width": width, "row_alignment": "CSV ordinal joined by sample_id",
             "unusable_rows_retained": sum(records[i].get("usable", "1") != "1" for i in positions),
             "label_or_user_columns_used_as_features": False, "cache_provenance": "historical source/manifest attestation"}
    return features, audit


def fit_sensor_head(features, labels, train_idx, held_idx, subjects, *, kind):
    tr, va = np.asarray(train_idx, int), np.asarray(held_idx, int)
    users = np.asarray(subjects).astype(str)
    if not len(tr) or not len(va) or set(users[tr]) & set(users[va]):
        raise ProtocolError("invalid sensor subject split")
    if kind not in FITTER_RECIPE: raise ProtocolError("unknown raw-sensor head")
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ProtocolError("sensor features must be finite 2D")
    head = ExtraTreesClassifier(**FITTER_RECIPE[kind], min_samples_leaf=2, max_features="sqrt",
                               class_weight="balanced", random_state=20260816, n_jobs=4)
    started = time.monotonic()
    print(json.dumps({"event": "sensor_fit_start", "kind": kind, "rows": len(tr)}), flush=True)
    head.fit(x[tr], np.asarray(labels)[tr])
    probability = np.zeros((len(va), 40))
    probability[:, np.asarray(head.classes_, int)] = head.predict_proba(x[va])
    print(json.dumps({"event": "sensor_fit_complete", "kind": kind,
                      "seconds": time.monotonic()-started}), flush=True)
    return None, head, probability


def fit_imu(features, labels, train_idx, held_idx, subjects):
    return fit_sensor_head(features, labels, train_idx, held_idx, subjects, kind="imu")


def fit_skeleton(features, labels, train_idx, held_idx, subjects):
    return fit_sensor_head(features, labels, train_idx, held_idx, subjects, kind="skeleton")


def nodes_from_log(log):
    return {key: ArtifactNode(key, parents=tuple(v["parents"]), provenance=v["provenance"],
                has_task_labels=v["has_task_labels"], supervised_train_subjects=frozenset(v["supervised_train_subjects"]))
            for key, v in log["artifact_dag"].items()}


def validate_pilot(path, current_spec):
    if path is None: raise ProtocolError("--run requires --pilot-dir from a successful score-blind pilot")
    path = Path(path)
    info = json.loads((path/"pilot.json").read_text(encoding="utf-8"))
    registry = json.loads((path/"experiment_registry.json").read_text(encoding="utf-8"))
    register_experiment(path/"experiment_registry.json", registry["spec"])
    seconds = info.get("elapsed_seconds",float("inf"))
    if info.get("mode") != "pilot" or "evaluation" in info or not np.isfinite(seconds) or not 0 <= seconds <= 600:
        raise ProtocolError("pilot is not score-blind/complete or exceeds 600 seconds")
    for key in ("source_sha256", "input_sha256", "sensor_heads", "sensor_common"):
        if registry["spec"].get(key) != current_spec[key]: raise ProtocolError("pilot recipe/input mismatch")
    if set(info.get("folds",{})) != {"0"} or not info["folds"]["0"].get("provenance_checked"):
        raise ProtocolError("pilot fold/provenance incomplete")
    return {"pilot_registry_sha256":sha(path/"experiment_registry.json"),
            "pilot_summary_sha256":sha(path/"pilot.json"),"seconds":seconds}


def run_fold(families, labels, users, folds, fold, metadata):
    if len(families) != 7: raise ProtocolError("P421 requires exactly seven families")
    held, outputs, log = group_fold(families, labels, users, folds, fold, metadata,
        family_fitters=[fit_family_head]*5+[fit_imu, fit_skeleton],
        family_provenances=["frozen_external"]*5+["raw_input", "raw_input"])
    probability = np.zeros((len(labels), 40))
    probability[held] = log["repeat_probability"]
    held_seq, seq, seq_log = sequence_fold(probability, labels, users, folds, fold, metadata, nodes_from_log(log))
    if not np.array_equal(held, held_seq) or not np.array_equal(outputs["repeat_group"], seq["p419_repeat"]):
        raise ProtocolError("group/sequence alignment mismatch")
    outputs.update(unique_only=seq["unique_only"], source_sequence=seq["source_sequence"])
    log["sequence"] = seq_log
    return held, outputs, log


def main(argv=None):
    ap = argparse.ArgumentParser()
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--pilot", action="store_true"); modes.add_argument("--run", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot-dir", help="Required for --run; enforces frozen recipe and 600-second pilot budget")
    args = ap.parse_args(argv); out = Path(args.out_dir)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, expected in REFERENCE_HASHES.items():
        if sha(REFERENCE/name) != expected: raise ProtocolError("P420 reference hash mismatch")
    source_paths = [Path(__file__), PREREG, *[HERE/name for name in (
        "p416_nested_frozen_family_router.py", "p418_nested_repeat_group_bridge.py",
        "p419_vjepa_repeat_group_bridge.py", "p420_source_only_session_bridge.py",
        "p90_teacher_common.py", "stable_routing_protocol.py", "stable_routing_structure.py",
        "audit_p87_sequence_decoder.py", "p89_imu_orientation_expert.py", "p89_skeleton_invariant_expert.py",
        "build_p86_motion_window_cache.py", "p31_skeleton_imu_preprocessing.py", "p46_event_preprocessing.py",
        "imu_data.py", "p96_vjepa2_dense24_extractor.py", "p90_videomae_lora_teacher.py")]]
    inputs = [IMU_FEATURES, IMU_INDEX, SKEL_FEATURES, SKEL_INDEX, META,
              VJEPA_CACHE, VJEPA_DONE, VJEPA_SUMMARY, EXTRACTION_ROWS,
              REFERENCE/"predictions.npz", REFERENCE/"experiment_registry.json"]
    spec = {"experiment": "P421_raw_sensor_sequence_bridge", "mode": "pilot" if args.pilot else "run",
            "base_family_spec": family_spec(), "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
            "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
            "sensor_heads": FITTER_RECIPE, "sensor_common": {"min_samples_leaf":2,"max_features":"sqrt",
                "class_weight":"balanced","random_state":20260816,"n_jobs":4},
            "primary": "source_sequence_minus_frozen_p420", "cpu_threads":4,"promotion_allowed":False}
    if args.run: spec["pilot_receipt"] = validate_pilot(args.pilot_dir, spec)
    register_experiment(out/"experiment_registry.json", spec)
    snapshot = out/"source_snapshot"; snapshot.mkdir()
    for path in source_paths: (snapshot/path.name).write_bytes(path.read_bytes())
    started = time.monotonic()
    protocol = load_protocol()
    families, y, users, folds, ids = load_frozen_families(protocol)
    keep = ~np.isin(protocol.users.astype(str), list(EXCLUDED_USERS))
    assert_extraction_alignment(EXTRACTION_ROWS, protocol.sample_ids)
    families.append(load_vjepa()[keep])
    imu, imu_audit = load_descriptor(IMU_FEATURES, IMU_INDEX, ids, 5222, "train")
    skel, skel_audit = load_descriptor(SKEL_FEATURES, SKEL_INDEX, ids, 9103)
    families.extend((imu, skel)); metadata = load_recording_metadata(META, ids)
    with np.load(REFERENCE/"predictions.npz", allow_pickle=False) as z:
        for key, value in (("sample_ids",ids),("users",users),("fold_id",folds)):
            if not np.array_equal(z[key],value): raise ProtocolError("P420 reference alignment mismatch")
        baseline = z["source_sequence"].copy()
    complete = {key: np.full(len(ids),-1,int) for key in ("own_group","repeat_group","equal_mean","selective_all","unique_only","source_sequence")}
    logs = {}
    for fold in ((0,) if args.pilot else OUTER_FOLDS):
        with threadpool_limits(limits=4), warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            held, output, log = run_fold(families,y,users,folds,fold,metadata)
        for key in complete: complete[key][held] = output[key]
        arrays = {key: log.pop(key) for key in ("inner_probability_bank","outer_probability_bank","own_probability","repeat_probability")}
        np.savez_compressed(out/f"fold{fold}_banks.npz",**arrays,
            inner_sample_ids=np.asarray(log["outer_train_ids"]),outer_sample_ids=np.asarray(log["outer_held_ids"]))
        (out/f"fold{fold}_provenance.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
        logs[str(fold)] = log
        print(json.dumps({"event":"fold_complete","fold":fold,"elapsed_seconds":time.monotonic()-started}),flush=True)
    np.savez_compressed(out/"predictions.npz",sample_ids=ids,users=users,fold_id=folds,p420_sequence=baseline,**complete)
    result = {"mode":spec["mode"],"folds":logs,"raw_descriptors":{"imu":imu_audit,"skeleton":skel_audit},
              "target_achieved":False,"independent_confirmation":False,"test_features_used":False,
              "elapsed_seconds_before_metrics":time.monotonic()-started}
    if args.run:
        if any((p<0).any() for p in complete.values()): raise ProtocolError("incomplete outputs")
        result["evaluation"] = evaluate_outputs({"p420_sequence":baseline,**complete},y,users,folds,base_key="p420_sequence")
        for key, value in result["evaluation"].items():
            if key != "source_sequence": value.pop("criterion",None); value.pop("mechanism_gate_pass",None)
        result["primary_mechanism_gate_pass"] = result["evaluation"]["source_sequence"]["mechanism_gate_pass"]
    result["elapsed_seconds"] = time.monotonic()-started
    (out/("pilot.json" if args.pilot else "summary.json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"notes.txt").write_text("Fixed raw IMU+skeleton addition to P420. No Test features/labels used. Historical development only.\n",encoding="utf-8")
    print(json.dumps({"event":"complete","mode":spec["mode"],"elapsed_seconds":result["elapsed_seconds"]}),flush=True)


if __name__ == "__main__": main()
