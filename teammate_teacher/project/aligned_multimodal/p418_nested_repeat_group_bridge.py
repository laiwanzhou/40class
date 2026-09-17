"""P418 nested repeat-group bridge.

The public kernels are deliberately small and synthetic-testable.  Group
features for every inner partition are constructed only on that partition;
there is no supervised OOF bank or cross-partition peer route.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time, warnings
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .stable_routing_structure import RecordingMetadata, build_group_features
from .p416_nested_frozen_family_router import (NUM_CLASSES, P238_PATHS, FAMILY_NAMES,
    EXCLUDED_USERS, OUTER_FOLDS, SEED, _as_feature_matrix, fit_family_head,
    fit_router_scorer, route, evaluate_outputs, load_frozen_families)
from .p416_nested_frozen_family_router import _spec as family_spec
from .p90_teacher_common import load_protocol, softmax
from .stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance, register_experiment

GROUP_FEATURE_NAMES = ("own_group", "repeat_group")

def _align_scores(raw, classes, n):
    raw = np.asarray(raw, float)
    if len(classes) == 1:
        out = np.full((n, NUM_CLASSES), -1e9); out[:, int(classes[0])] = 0
    else:
        # Binary LogisticRegression uses sigmoid(raw), not sigmoid(2*raw).
        if raw.ndim == 1: raw = np.column_stack((-raw / 2, raw / 2))
        out = np.full((n, NUM_CLASSES), -1e9); out[:, np.asarray(classes, int)] = raw
    return softmax(out)

def fit_group_head(features, labels, train_idx, held_idx, subjects):
    """Fit the fixed StandardScaler + LogisticRegression group head."""
    tr, va = np.asarray(train_idx, int), np.asarray(held_idx, int)
    s = np.asarray(subjects).astype(str)
    overlap = set(s[tr]) & set(s[va])
    if overlap: raise ProtocolError(f"group fit intersects held subjects: {sorted(overlap)[:3]}")
    y = np.asarray(labels, int)[tr]
    if len(np.unique(y)) < 2: raise ProtocolError("group head requires at least two classes")
    scaler = StandardScaler().fit(np.asarray(features, float)[tr])
    clf = LogisticRegression(C=.03, solver="lbfgs", max_iter=2000,
                             class_weight=None).fit(scaler.transform(np.asarray(features)[tr]), y)
    if len(va) == 0:
        return scaler, clf, np.empty((0, NUM_CLASSES), float)
    return scaler, clf, _align_scores(clf.decision_function(scaler.transform(np.asarray(features)[va])), clf.classes_, len(va))

def _subset_metadata(meta: RecordingMetadata, rows):
    rows = np.asarray(rows, int)
    return RecordingMetadata(np.asarray(meta.sample_ids)[rows], np.asarray(meta.users)[rows],
                             np.asarray(meta.dates)[rows], np.asarray(meta.starts)[rows])

def _group_features_partition(bank, meta, include_peers):
    return build_group_features(bank, meta, include_peers=include_peers)

def nested_group_bank(families, labels, subjects, outer_train, metadata, n_splits=3, fit_log=None, family_fitters=None):
    """Return OOF base bank and two partition-local group feature matrices."""
    outer_train = np.asarray(outer_train, int); groups = np.asarray(subjects)[outer_train]
    fitters = [fit_family_head] * len(families) if family_fitters is None else list(family_fitters)
    if len(fitters) != len(families) or not all(callable(f) for f in fitters):
        raise ProtocolError("family fitter contract mismatch")
    if len(np.unique(groups)) < n_splits: raise ProtocolError("not enough subjects for inner GroupKFold")
    width = len(families) * NUM_CLASSES * 2 + NUM_CLASSES + 1
    bank = np.zeros((len(outer_train), len(families), NUM_CLASSES)); own = np.zeros((len(outer_train), width), np.float32); rep = own.copy()
    for inner, (trloc, valoc) in enumerate(GroupKFold(n_splits=n_splits).split(outer_train, labels[outer_train], groups)):
        tr, va = outer_train[trloc], outer_train[valoc]
        for j, x in enumerate(families):
            started = time.monotonic()
            _, _, bank[valoc, j] = fitters[j](x, labels, tr, va, subjects)
            if fit_log is not None: fit_log.append({"node_id":f"inner{inner}.family{j}","family":j,"train_indices":tr.tolist(),"validation_indices":va.tolist(),"train_subjects":sorted(set(np.asarray(subjects)[tr].astype(str))),"validation_subjects":sorted(set(np.asarray(subjects)[va].astype(str))),"preprocessing_fit_subjects":sorted(set(np.asarray(subjects)[tr].astype(str)))})
            if fit_log is not None:
                fit_log[-1].update(train_ids=np.asarray(metadata.sample_ids)[tr].astype(str).tolist(),
                                   validation_ids=np.asarray(metadata.sample_ids)[va].astype(str).tolist(),
                                   seconds=time.monotonic()-started)
            print(json.dumps({"event":"inner_head_complete", "inner":inner, "family":j}), flush=True)
        # Critical: peer search sees this validation partition only.
        local = _subset_metadata(metadata, va)
        own[valoc], own_audit = _group_features_partition(bank[valoc], local, False)
        rep[valoc], rep_audit = _group_features_partition(bank[valoc], local, True)
        if fit_log is not None:
            fit_log.append({"node_id": f"inner{inner}.group_features", "partition_indices": va.tolist(),
                            "partition_ids": np.asarray(metadata.sample_ids)[va].astype(str).tolist(),
                            "peer_coverage_own": own_audit, "peer_coverage_repeat": rep_audit,
                            "peer_ancestors": [f"inner{inner}.family{j}" for j in range(len(families))]})
    return bank, own, rep

def run_outer_fold(families, labels, subjects, fold_id, fold, metadata, family_fitters=None, family_provenances=None):
    folds=np.asarray(fold_id); held=np.flatnonzero(folds==fold); train=np.flatnonzero(folds!=fold)
    s=np.asarray(subjects).astype(str)
    if not len(train) or not len(held): raise ProtocolError("empty outer train or held partition")
    if set(s[train]) & set(s[held]): raise ProtocolError("outer subject overlap")
    fitters = [fit_family_head] * len(families) if family_fitters is None else list(family_fitters)
    provenance = ["frozen_external"] * len(families) if family_provenances is None else list(family_provenances)
    if len(fitters) != len(families) or not all(callable(f) for f in fitters):
        raise ProtocolError("family fitter contract mismatch")
    if len(provenance) != len(families) or any(p not in ("raw_input", "frozen_external") for p in provenance):
        raise ProtocolError("family root provenance contract mismatch")
    logs=[]; inner_bank, own_x, rep_x = nested_group_bank(families, labels, subjects, train, metadata, fit_log=logs, family_fitters=fitters)
    # Fit matched group heads on partition-local OOF features.
    # Empty held set is intentional: obtain fitted estimators for outer-held
    # prediction while keeping the fit API's disjointness assertion active.
    group_started=time.monotonic()
    own_scaler, own_head, _ = fit_group_head(own_x, labels[train], np.arange(len(train)), np.array([], int), s[train])
    rep_scaler, rep_head, _ = fit_group_head(rep_x, labels[train], np.arange(len(train)), np.array([], int), s[train])
    # Outer base heads are fitted on outer train and produce held-only bank.
    outer_bank=np.zeros((len(held),len(families),NUM_CLASSES))
    outer_nodes=[]
    for j,x in enumerate(families):
        started=time.monotonic()
        _,_,outer_bank[:,j]=fitters[j](x,labels,train,held,subjects)
        outer_nodes.append(f"p418.outer.fold{fold}.family{j}.head")
        print(json.dumps({"event":"outer_head_complete", "fold":fold, "family":j,
                          "seconds":time.monotonic()-started}), flush=True)
    held_meta=_subset_metadata(metadata,held)
    own_h, own_audit = _group_features_partition(outer_bank,held_meta,False); rep_h, rep_audit = _group_features_partition(outer_bank,held_meta,True)
    own_p=_align_scores(own_head.decision_function(own_scaler.transform(own_h)),own_head.classes_,len(held))
    rep_p=_align_scores(rep_head.decision_function(rep_scaler.transform(rep_h)),rep_head.classes_,len(held))
    # Descriptive P416 comparator; its scorer is trained on the same inner bank.
    # P416 is always the original four-family descriptive comparator; extra
    # families in a separately registered extension do not redefine it.
    rs, rc=fit_router_scorer(inner_bank[:, :4], labels[train]); desc=route(outer_bank[:, :4],rs,rc)
    nodes={f"external.family{j}":ArtifactNode(f"external.family{j}",provenance=provenance[j]) for j in range(len(families))}
    inner_nodes=[]
    for rec in logs:
        nm=rec["node_id"]
        if "family" not in rec: continue
        nodes[nm]=ArtifactNode(nm,parents=(f"external.family{rec['family']}",),provenance="supervised",has_task_labels=True,supervised_train_subjects=frozenset(rec["train_subjects"])); inner_nodes.append(nm)
        for subject in rec["validation_subjects"]:
            assert_prediction_provenance(str(subject), [nm], nodes)
    feature_nodes=[]
    for rec in logs:
        if "peer_ancestors" not in rec: continue
        nm=rec["node_id"]
        nodes[nm]=ArtifactNode(nm, parents=tuple(rec["peer_ancestors"]))
        feature_nodes.append(nm)
        for subject in np.unique(s[rec["partition_indices"]]):
            assert_prediction_provenance(str(subject), [nm], nodes)
    for name in GROUP_FEATURE_NAMES:
        gid=f"{name}.fold{fold}.head"; nodes[gid]=ArtifactNode(gid,parents=tuple(feature_nodes),provenance="supervised",has_task_labels=True,supervised_train_subjects=frozenset(s[train]))
    for j, oid in enumerate(outer_nodes):
        nodes[oid] = ArtifactNode(oid, parents=(f"external.family{j}",), provenance="supervised",
                                  has_task_labels=True, supervised_train_subjects=frozenset(s[train]))
    scorer_id=f"p416_scorer.fold{fold}"
    descriptive_inner = tuple(rec["node_id"] for rec in logs if rec.get("family", 99) < 4)
    nodes[scorer_id]=ArtifactNode(scorer_id, parents=descriptive_inner, provenance="supervised",
        has_task_labels=True, supervised_train_subjects=frozenset(s[train]))
    nodes["equal_mean.prediction"]=ArtifactNode("equal_mean.prediction",parents=tuple(outer_nodes[:4]))
    nodes["selective_all.prediction"]=ArtifactNode("selective_all.prediction",parents=(scorer_id,*outer_nodes[:4]))
    # Explicit ancestry checks: group heads may depend on inner heads, but held
    # predictions only depend on outer heads and held-input geometry.
    for subj in np.unique(s[held]):
        assert_prediction_provenance(str(subj), ["equal_mean.prediction", "selective_all.prediction"], nodes)
        pred_nodes=[]
        for name in GROUP_FEATURE_NAMES:
            gid=f"{name}.fold{fold}.head"; pid=gid+".prediction"; nodes[pid]=ArtifactNode(pid,parents=(gid,*outer_nodes)); pred_nodes.append(pid)
            assert_prediction_provenance(str(subj), [pid], nodes)
    return held,{"own_group":own_p.argmax(1),"repeat_group":rep_p.argmax(1),"equal_mean":desc["equal_mean"],"selective_all":desc["selective_all"]},{"outer_train_indices":train.tolist(),"outer_train_ids":np.asarray(metadata.sample_ids)[train].astype(str).tolist(),"outer_train_subjects":sorted(set(s[train])),"outer_held_subjects":sorted(set(s[held])),"outer_held_indices":held.tolist(),"outer_held_ids":np.asarray(metadata.sample_ids)[held].astype(str).tolist(),"fits":logs,"artifact_dag":{k:{"node_id":k,"parents":list(v.parents),"provenance":v.provenance,"has_task_labels":v.has_task_labels,"supervised_train_subjects":sorted(v.supervised_train_subjects)} for k,v in nodes.items()},"provenance_checked":True,"group_feature_rows":{"own":int(len(own_x)),"repeat":int(len(rep_x))},"outer_peer_coverage":{"own":own_audit,"repeat":rep_audit},"inner_probability_bank":inner_bank,"outer_probability_bank":outer_bank,"own_probability":own_p,"repeat_probability":rep_p}

def load_recording_metadata(path, sample_ids):
    import csv
    rows={}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            sid=str(r["sample_id"])
            if sid in rows: raise ProtocolError("duplicate metadata sample_id")
            rows[sid] = r
    out=[]
    for sid in np.asarray(sample_ids).astype(str):
        # Missing rows are intentionally retained with invalid geometry; the
        # portable structure then yields own-only features for those rows.
        r=rows.get(sid, {})
        try: start=float(r.get("start_seconds", r.get("start", "nan")))
        except (ValueError, TypeError): start=float("nan")
        out.append((r.get("recording_date", r.get("date", "")), start))
    return RecordingMetadata(np.asarray(sample_ids), np.asarray([rows.get(str(x), {}).get("user_id", rows.get(str(x), {}).get("user", "")) for x in sample_ids]), np.asarray([x[0] for x in out]), np.asarray([x[1] for x in out]))

def main(argv=None):
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(required=True); g.add_argument("--pilot",action="store_true"); g.add_argument("--run",action="store_true"); ap.add_argument("--out-dir",required=True); a=ap.parse_args(argv)
    out=Path(a.out_dir)
    if out.exists(): raise FileExistsError(f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True); os.environ.setdefault("OMP_NUM_THREADS","4"); os.environ.setdefault("MKL_NUM_THREADS","4")
    # Freeze protocol, code, preregistration, and available input hashes before
    # touching the manifest or loading feature data.
    root=Path(__file__).resolve().parent.parent
    prereg=root/"docs/research/STABLE_093_P418_PREREGISTRATION_2026-09-07.md"
    def sha(p):
        h=hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
        return h.hexdigest()
    inherited=family_spec()
    metadata_path=root/"aligned_multimodal/data/p85_recording_metadata/train_recording_metadata.csv"
    snapshots=(Path(__file__),Path(__file__).with_name("stable_routing_structure.py"),
        Path(__file__).with_name("stable_routing_protocol.py"),
        Path(__file__).with_name("p416_nested_frozen_family_router.py"),Path(__file__).with_name("p90_teacher_common.py"))
    spec={"experiment":"P418_nested_repeat_group_bridge","preregistration_sha256":sha(prereg),
        "source_sha256":{p.relative_to(root).as_posix():sha(p) for p in (*snapshots,prereg,metadata_path)},
        "frozen_family_provenance":inherited,
        "group_head":{"C":.03,"solver":"lbfgs","max_iter":2000,"class_weight":None},
        "execution_mode":"pilot" if a.pilot else "run","executed_outer_folds":[0] if a.pilot else list(OUTER_FOLDS)}
    spec.update({"outer_folds":list(OUTER_FOLDS),"inner":"GroupKFold(3)","primary":"repeat_group_minus_own_group","promotion_allowed":False,"cpu_threads":4})
    register_experiment(out/"experiment_registry.json",spec)
    snap=out/"source_snapshot"; snap.mkdir()
    for pth in snapshots:
        (snap/pth.name).write_bytes(pth.read_bytes())
    (snap/prereg.name).write_bytes(prereg.read_bytes())
    started=time.monotonic()
    p=load_protocol(); fam,y,users,folds,sids=load_frozen_families(p); meta=load_recording_metadata(Path(__file__).parent/"data/p85_recording_metadata/train_recording_metadata.csv",sids)
    all_out={}; prov={"mode":"pilot" if a.pilot else "run","folds":{},"target_achieved":False,
        "independent_confirmation":False,"test_rows_loaded":0,"test_labels_read":False,
        "warning":"Exploratory historical-subject bridge, not the P315 topology or a competition candidate."}
    for f in ((0,) if a.pilot else OUTER_FOLDS):
        with threadpool_limits(limits=4):
            with warnings.catch_warnings():
                warnings.simplefilter("error",ConvergenceWarning)
                _,o,log=run_outer_fold(fam,y,users,folds,f,meta)
        all_out[str(f)]=o; prov["folds"][str(f)]=log
        arrays={key:log.pop(key) for key in ("inner_probability_bank","outer_probability_bank","own_probability","repeat_probability")}
        np.savez_compressed(out/f"fold{f}_banks.npz",**arrays,
            inner_sample_ids=np.asarray(log["outer_train_ids"]),outer_sample_ids=np.asarray(log["outer_held_ids"]))
        (out/f"fold{f}_provenance.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
        print(json.dumps({"event":"outer_fold_complete","fold":f,"elapsed_seconds":time.monotonic()-started}),flush=True)
    complete={k:np.full(len(y),-1,int) for k in ("own_group","repeat_group","equal_mean","selective_all")}
    for f,o in all_out.items():
        ix=prov["folds"][f]["outer_held_indices"]
        for k in complete: complete[k][ix]=o[k]
    np.savez_compressed(out/"predictions.npz",sample_ids=sids,users=users,fold_id=folds,**complete)
    if a.run:
        if any(np.any(values<0) for values in complete.values()): raise ProtocolError("incomplete outer predictions")
        prov["evaluation"]=evaluate_outputs(complete,y,users,folds,base_key="own_group")
        for key in ("own_group","equal_mean","selective_all"):
            prov["evaluation"][key].pop("mechanism_gate_pass",None)
            prov["evaluation"][key].pop("criterion",None)
        prov["primary_mechanism_gate_pass"] = prov["evaluation"]["repeat_group"]["mechanism_gate_pass"]
        prov["primary_contrast"] = "repeat_group_minus_own_group"
    prov["elapsed_seconds"]=time.monotonic()-started
    (out/("pilot.json" if a.pilot else "summary.json")).write_text(json.dumps(prov,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x,indent=2),encoding="utf-8")
    (out/"notes.txt").write_text("P418 repeat-group context versus own-only group head. No Test inputs. "
        "Pilot has no accuracy evaluation. Full results are exploratory, not P315 or 0.93 confirmation.\n",encoding="utf-8")
    print(json.dumps({"event":"complete","mode":prov["mode"],"elapsed_seconds":prov["elapsed_seconds"],"target_achieved":False}),flush=True)


if __name__ == "__main__": main()
