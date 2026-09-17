"""Reuse proven nested P421 banks; retain all classifiers, protect five-family geometry."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from .p416_nested_frozen_family_router import EXCLUDED_USERS, OUTER_FOLDS, evaluate_outputs
from .p418_nested_repeat_group_bridge import fit_group_head, _align_scores, _subset_metadata, load_recording_metadata
from .p419_vjepa_repeat_group_bridge import sha
from .p420_source_only_session_bridge import run_outer_fold as sequence_fold
from .p421_raw_sensor_sequence_bridge import nodes_from_log, REFERENCE, REFERENCE_HASHES
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance, register_experiment
from .stable_routing_structure import build_group_features

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "aligned_multimodal"
SOURCE = HERE / "runs/p421_raw_sensor_sequence_bridge_v1"
META = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
PREREG = ROOT / "docs/research/STABLE_093_P422_PREREGISTRATION_2026-09-07.md"


def protected_features(bank, metadata, peers, classifier_family_count=7, bank_family_count=7):
    p = np.asarray(bank)
    if bank_family_count not in (6,7) or p.ndim != 3 or p.shape[1:] != (bank_family_count, 40) or not np.isfinite(p).all() or np.any(p < 0):
        raise ProtocolError("expected finite declared-family probability bank")
    if not np.allclose(p.sum(2), 1): raise ProtocolError("unnormalized probability bank")
    if classifier_family_count not in (5, bank_family_count): raise ProtocolError("classifier family count violates bank contract")
    return build_group_features(p[:, :classifier_family_count], metadata, include_peers=peers, grouping_bank=p[:, :5])


def run_cached_fold(inner_bank, outer_bank, source_log, labels, users, folds, fold, metadata, classifier_family_count=7, bank_family_count=7):
    if bank_family_count not in (6,7) or classifier_family_count not in (5, bank_family_count): raise ProtocolError("classifier family count violates bank contract")
    train = np.flatnonzero(np.asarray(folds) != fold)
    held = np.flatnonzero(np.asarray(folds) == fold)
    subjects = np.asarray(users).astype(str)
    ids = np.asarray(metadata.sample_ids).astype(str)
    if set(subjects[train]) & set(subjects[held]): raise ProtocolError("outer subject overlap")
    for key, value in (("outer_train_indices",train), ("outer_held_indices",held),
                       ("outer_train_ids",ids[train]), ("outer_held_ids",ids[held])):
        if not np.array_equal(source_log[key],value): raise ProtocolError("cached source split/ID mismatch")
    if len(inner_bank) != len(train) or len(outer_bank) != len(held):
        raise ProtocolError("cached bank row mismatch")
    nodes = nodes_from_log(source_log)
    for subject in np.unique(subjects[held]):
        assert_prediction_provenance(str(subject),[f"repeat_group.fold{fold}.head.prediction"],nodes)
    partitions = [r for r in source_log["fits"] if "peer_ancestors" in r]
    expected = {frozenset(train[va].tolist()) for _,va in GroupKFold(3).split(train, np.asarray(labels)[train],subjects[train])}
    actual = [frozenset(r["partition_indices"]) for r in partitions]
    if len(actual) != 3 or set(actual) != expected: raise ProtocolError("cached inner partitions differ from GroupKFold3")
    if sum(len(r["partition_indices"]) for r in partitions) != len(train):
        raise ProtocolError("cached inner partitions contain duplicate rows")
    local_index = {int(row): i for i,row in enumerate(train)}
    width = classifier_family_count*40*2+41
    own_x = np.zeros((len(train),width),np.float32); repeat_x = own_x.copy()
    feature_nodes = []; partition_audits = []
    for index, record in enumerate(partitions):
        va = np.asarray(record["partition_indices"],int)
        if len(va) != len(np.unique(va)): raise ProtocolError("duplicate inner validation index")
        local = np.asarray([local_index[int(row)] for row in va])
        if not np.array_equal(record["partition_ids"],ids[va]): raise ProtocolError("inner partition ID mismatch")
        parents = tuple(record["peer_ancestors"])
        if len(parents) != bank_family_count or len(set(parents)) != bank_family_count: raise ProtocolError("incomplete family ancestry")
        parents = parents[:classifier_family_count]
        for subject in np.unique(subjects[va]): assert_prediction_provenance(str(subject),parents,nodes)
        own_x[local], own_audit = protected_features(inner_bank[local],_subset_metadata(metadata,va),False,classifier_family_count,bank_family_count)
        repeat_x[local], repeat_audit = protected_features(inner_bank[local],_subset_metadata(metadata,va),True,classifier_family_count,bank_family_count)
        node_id = f"p422.inner{index}.features"
        nodes[node_id] = ArtifactNode(node_id,parents=parents); feature_nodes.append(node_id)
        partition_audits.append({"global_indices":va.tolist(),"sample_ids":ids[va].tolist(),
                                 "own":own_audit,"repeat":repeat_audit,"parents":list(parents)})
    own_scaler, own_head, _ = fit_group_head(own_x,np.asarray(labels)[train],np.arange(len(train)),[],subjects[train])
    rep_scaler, rep_head, _ = fit_group_head(repeat_x,np.asarray(labels)[train],np.arange(len(train)),[],subjects[train])
    own_h, own_audit = protected_features(outer_bank,_subset_metadata(metadata,held),False,classifier_family_count,bank_family_count)
    rep_h, rep_audit = protected_features(outer_bank,_subset_metadata(metadata,held),True,classifier_family_count,bank_family_count)
    own_p = _align_scores(own_head.decision_function(own_scaler.transform(own_h)),own_head.classes_,len(held))
    rep_p = _align_scores(rep_head.decision_function(rep_scaler.transform(rep_h)),rep_head.classes_,len(held))
    outer_nodes = tuple(f"p418.outer.fold{fold}.family{j}.head" for j in range(classifier_family_count))
    for name in ("own_group","repeat_group"):
        nid = f"p422.{name}.fold{fold}.head"
        nodes[nid] = ArtifactNode(nid,parents=tuple(feature_nodes),provenance="supervised",has_task_labels=True,
                                  supervised_train_subjects=frozenset(subjects[train]))
        # The legacy alias is needed by the already checked sequence kernel;
        # it now points to the new head, never to the rejected P421 group head.
        pid = f"{name}.fold{fold}.head.prediction"
        nodes[pid] = ArtifactNode(pid,parents=(nid,*outer_nodes))
        for subject in np.unique(subjects[held]): assert_prediction_provenance(str(subject),[pid],nodes)
    full_probability = np.zeros((len(labels),40)); full_probability[held] = rep_p
    seq_held, seq, seq_log = sequence_fold(full_probability,labels,users,folds,fold,metadata,nodes)
    if not np.array_equal(held,seq_held) or not np.array_equal(rep_p.argmax(1),seq["p419_repeat"]):
        raise ProtocolError("sequence/feature alignment mismatch")
    outputs = {"own_group":own_p.argmax(1),"repeat_group":rep_p.argmax(1),
               "unique_only":seq["unique_only"],"source_sequence":seq["source_sequence"]}
    log = {"outer_train_ids":ids[train].tolist(),"outer_held_ids":ids[held].tolist(),
           "partitions":partition_audits,"outer_geometry":{"own":own_audit,"repeat":rep_audit},
           "sequence":seq_log,"provenance_checked":True,
           "own_probability":own_p,"repeat_probability":rep_p}
    return held,outputs,log


def verify_source_snapshots():
    reg = json.loads((SOURCE/"experiment_registry.json").read_text(encoding="utf-8"))
    register_experiment(SOURCE/"experiment_registry.json",reg["spec"])
    expected = reg["spec"]["source_sha256"]
    files = list((SOURCE/"source_snapshot").iterdir())
    if {p.name for p in files} != {Path(key).name for key in expected}: raise ProtocolError("source snapshot set mismatch")
    for path in files:
        keys = [key for key in expected if Path(key).name == path.name]
        if len(keys) != 1 or sha(path) != expected[keys[0]]: raise ProtocolError("source snapshot hash mismatch")
    registered_inputs = {key.replace("\\", "/"): value for key,value in {
        **reg["spec"]["base_family_spec"]["source_sha256"], **reg["spec"]["input_sha256"]}.items()}
    protocol_files = [META,HERE/"data/manifest.csv",*[HERE/f"data/subject_folds/fold_{f}.csv" for f in OUTER_FOLDS]]
    for path in protocol_files:
        relative = path.relative_to(ROOT).as_posix()
        if registered_inputs.get(relative) != sha(path):
            raise ProtocolError("current labels/splits/metadata differ from borrowed P421 fitting inputs")


def main(argv=None):
    ap = argparse.ArgumentParser(); modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--pilot",action="store_true"); modes.add_argument("--run",action="store_true")
    ap.add_argument("--out-dir",required=True); args = ap.parse_args(argv); out = Path(args.out_dir)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=False); verify_source_snapshots()
    for name,expected in REFERENCE_HASHES.items():
        if sha(REFERENCE/name) != expected: raise ProtocolError("P420 reference changed")
    sources = [Path(__file__),PREREG,*[HERE/name for name in (
        "p416_nested_frozen_family_router.py","p418_nested_repeat_group_bridge.py",
        "p419_vjepa_repeat_group_bridge.py","p420_source_only_session_bridge.py","p421_raw_sensor_sequence_bridge.py",
        "p90_teacher_common.py","stable_routing_structure.py","stable_routing_protocol.py","audit_p87_sequence_decoder.py")]]
    inputs = [META,HERE/"data/manifest.csv",REFERENCE/"predictions.npz",SOURCE/"predictions.npz",SOURCE/"experiment_registry.json",
              *[HERE/f"data/subject_folds/fold_{f}.csv" for f in OUTER_FOLDS],
              *[SOURCE/f"fold{f}_banks.npz" for f in OUTER_FOLDS],*[SOURCE/f"fold{f}_provenance.json" for f in OUTER_FOLDS]]
    spec = {"experiment":"P422_protected_geometry_bridge","mode":"pilot" if args.pilot else "run",
            "source_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            "input_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
            "geometry_families":[0,1,2,3,4],"classifier_families":list(range(7)),
            "primary":"source_sequence_minus_frozen_p420","base_head_refits":0,"promotion_allowed":False}
    register_experiment(out/"experiment_registry.json",spec); snap = out/"source_snapshot"; snap.mkdir()
    for path in sources: (snap/path.name).write_bytes(path.read_bytes())
    started = time.monotonic(); protocol = load_protocol()
    keep = ~np.isin(protocol.users.astype(str),list(EXCLUDED_USERS))
    ids,y,users,folds = (v[keep] for v in (protocol.sample_ids,protocol.labels,protocol.users,protocol.fold_id))
    metadata = load_recording_metadata(META,ids)
    with np.load(REFERENCE/"predictions.npz",allow_pickle=False) as z:
        for key,value in (("sample_ids",ids),("users",users),("fold_id",folds)):
            if not np.array_equal(z[key],value): raise ProtocolError("reference ID/subject/fold mismatch")
        baseline = z["source_sequence"].copy()
    with np.load(SOURCE/"predictions.npz",allow_pickle=False) as z:
        for key,value in (("sample_ids",ids),("users",users),("fold_id",folds)):
            if not np.array_equal(z[key],value): raise ProtocolError("P421 diagnostic reference mismatch")
        rejected_reference = z["source_sequence"].copy()
    complete = {key:np.full(len(ids),-1,int) for key in ("own_group","repeat_group","unique_only","source_sequence")}; logs = {}
    for fold in ((0,) if args.pilot else OUTER_FOLDS):
        source_log = json.loads((SOURCE/f"fold{fold}_provenance.json").read_text(encoding="utf-8"))
        with np.load(SOURCE/f"fold{fold}_banks.npz",allow_pickle=False) as z:
            if not np.array_equal(z["inner_sample_ids"],source_log["outer_train_ids"]) or not np.array_equal(z["outer_sample_ids"],source_log["outer_held_ids"]):
                raise ProtocolError("bank IDs do not match proven source log")
            inner,outer = z["inner_probability_bank"].copy(),z["outer_probability_bank"].copy()
        with threadpool_limits(limits=4),warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            held,outputs,log = run_cached_fold(inner,outer,source_log,y,users,folds,fold,metadata)
        for key in complete: complete[key][held] = outputs[key]
        np.savez_compressed(out/f"fold{fold}_group_probability.npz",sample_ids=ids[held],
                            own_probability=log.pop("own_probability"),repeat_probability=log.pop("repeat_probability"))
        logs[str(fold)] = log
        print(json.dumps({"event":"fold_complete","fold":fold,"elapsed_seconds":time.monotonic()-started}),flush=True)
    np.savez_compressed(out/"predictions.npz",sample_ids=ids,users=users,fold_id=folds,p420_sequence=baseline,**complete)
    report = {"mode":spec["mode"],"folds":logs,"target_achieved":False,"independent_confirmation":False,"test_rows_loaded":0}
    if args.run:
        if any((p<0).any() for p in complete.values()): raise ProtocolError("incomplete predictions")
        report["evaluation"] = evaluate_outputs({"p420_sequence":baseline,**complete},y,users,folds,base_key="p420_sequence")
        for key,value in report["evaluation"].items():
            if key != "source_sequence": value.pop("criterion",None);value.pop("mechanism_gate_pass",None)
        report["primary_mechanism_gate_pass"] = report["evaluation"]["source_sequence"]["mechanism_gate_pass"]
        secondary = evaluate_outputs({"p421_sequence":rejected_reference,"source_sequence":complete["source_sequence"]},
                                     y,users,folds,base_key="p421_sequence")["source_sequence"]
        secondary.pop("criterion",None); secondary.pop("mechanism_gate_pass",None)
        report["secondary_vs_rejected_p421"] = secondary
    report["elapsed_seconds"] = time.monotonic()-started
    (out/("pilot.json" if args.pilot else "summary.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out/"notes.txt").write_text("All7 classifier families; first5 geometry only. Compare to frozenP420. No Test or base-head refits.\n",encoding="utf-8")
    print(json.dumps({"event":"complete","mode":spec["mode"],"elapsed_seconds":report["elapsed_seconds"]}),flush=True)


if __name__ == "__main__": main()
