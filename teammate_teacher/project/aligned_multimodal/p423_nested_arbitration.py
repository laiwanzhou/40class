"""Three-level source-only selective arbitration; no global-OOF gate training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .p416_nested_frozen_family_router import (
    EXCLUDED_USERS, OUTER_FOLDS, fit_family_head, load_frozen_families, evaluate_outputs,
)
from .p418_nested_repeat_group_bridge import run_outer_fold as expert_fold, _subset_metadata, load_recording_metadata
from .p419_vjepa_repeat_group_bridge import load_vjepa, assert_extraction_alignment, EXTRACTION_ROWS, sha
from .p421_raw_sensor_sequence_bridge import (
    fit_imu, fit_skeleton, load_descriptor, IMU_FEATURES, IMU_INDEX, SKEL_FEATURES, SKEL_INDEX, META,
    nodes_from_log,
)
from .p422_protected_geometry_bridge import run_cached_fold, verify_source_snapshots
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ArtifactNode, ProtocolError, register_experiment, assert_prediction_provenance

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "aligned_multimodal"
PREREG = ROOT / "docs/research/STABLE_093_P423_PREREGISTRATION_2026-09-07.md"
P419 = HERE / "runs/p419_vjepa_repeat_group_bridge_v1"
P420 = HERE / "runs/p420_source_only_session_bridge_v1"
P421 = HERE / "runs/p421_raw_sensor_sequence_bridge_v1"
P422 = HERE / "runs/p422_protected_geometry_bridge_v1"
FEATURE_DIM = 31


def comparative_features(bank, group5, group7, base, alt):
    p = np.asarray(bank,float); base = np.asarray(base); alt = np.asarray(alt)
    n = len(base)
    if p.shape != (n,7,40) or np.asarray(group5).shape != (n,40) or np.asarray(group7).shape != (n,40):
        raise ProtocolError("comparison probability shapes differ")
    if base.shape != (n,) or alt.shape != (n,) or not all(np.issubdtype(v.dtype,np.integer) for v in (base,alt)):
        raise ProtocolError("comparison classes must be integer vectors")
    if any(np.any((v<0)|(v>=40)) for v in (base,alt)): raise ProtocolError("comparison class out of range")
    sources = np.concatenate((p,np.asarray(group5)[:,None],np.asarray(group7)[:,None]),axis=1)
    if not np.isfinite(sources).all() or np.any(sources<0) or not np.allclose(sources.sum(2),1):
        raise ProtocolError("invalid comparison probabilities")
    rows = np.arange(n)[:,None]; experts = np.arange(9)[None]
    pb = sources[rows,experts,base[:,None]]; pa = sources[rows,experts,alt[:,None]]
    log_ratio = np.clip(np.log(np.maximum(pa,1e-9))-np.log(np.maximum(pb,1e-9)),-20,20)
    def rank(values):
        return (sources>values[:,:,None]).sum(2)+.5*((sources==values[:,:,None]).sum(2)-1)
    rank_gap = (rank(pb)-rank(pa))/39
    contrasts = np.stack((log_ratio,pa-pb,rank_gap),axis=2).reshape(n,27)
    extra = []
    for group in (group5,group7):
        ordered = np.sort(group,axis=1)
        extra.extend((ordered[:,-1],ordered[:,-1]-ordered[:,-2]))
    result = np.column_stack((contrasts,*extra))
    if result.shape != (n,FEATURE_DIM) or not np.isfinite(result).all(): raise ProtocolError("comparison feature failure")
    return result


def namespace_nodes(prefix, graph):
    return {f"{prefix}/{key}":ArtifactNode(f"{prefix}/{key}",parents=tuple(f"{prefix}/{p}" for p in node.parents),
            provenance=node.provenance,has_task_labels=node.has_task_labels,
            supervised_train_subjects=node.supervised_train_subjects,oof_labels_only=node.oof_labels_only)
            for key,node in graph.items()}


def serialize_nodes(nodes):
    return {k:{"node_id":k,"parents":list(v.parents),"provenance":v.provenance,
               "has_task_labels":v.has_task_labels,"supervised_train_subjects":sorted(v.supervised_train_subjects)}
            for k,v in nodes.items()}


def source_policy_data(families, labels, users, metadata, fitters=None, deadline=None, artifact_dir=None):
    """Caller passes only outer-train rows: excluded outer-held arrays are absent."""
    n = len(labels); ids = np.asarray(metadata.sample_ids).astype(str); subjects = np.asarray(users).astype(str)
    if len(families)!=7 or any(len(x)!=n for x in families) or len(ids)!=n:
        raise ProtocolError("source pool shape mismatch")
    use_fitters = [fit_family_head]*5+[fit_imu,fit_skeleton] if fitters is None else fitters
    x = np.zeros((n,FEATURE_DIM)); base = np.full(n,-1,int); alt = base.copy()
    nodes = {}; feature_nodes = []; audit = []; covered = np.zeros(n,int)
    for inner,(train,held) in enumerate(GroupKFold(3).split(np.arange(n),labels,subjects)):
        if deadline is not None and time.monotonic()>deadline: raise TimeoutError("P423 pilot exceeded600seconds")
        print(json.dumps({"event":"policy_partition_start","inner":inner,"source_pool_rows":n}),flush=True)
        fold_id = np.ones(n,int); fold_id[held] = 0
        got,_,raw = expert_fold(families,labels,users,fold_id,0,metadata,
            family_fitters=use_fitters,family_provenances=["frozen_external"]*5+["raw_input"]*2)
        if not np.array_equal(got,held): raise ProtocolError("policy split changed in expert pipeline")
        branches = []
        for count in (5,7):
            got,pred,log = run_cached_fold(raw["inner_probability_bank"],raw["outer_probability_bank"],
                raw,labels,users,fold_id,0,metadata,classifier_family_count=count)
            if not np.array_equal(got,held): raise ProtocolError("policy branch changed held rows")
            branches.append((pred,log))
            prefix = f"policy{inner}.family{count}"
            nodes.update(namespace_nodes(prefix,nodes_from_log(log["sequence"])))
        pred5,log5 = branches[0]; pred7,log7 = branches[1]
        base[held] = pred5["source_sequence"]; alt[held] = pred7["source_sequence"]
        x[held] = comparative_features(raw["outer_probability_bank"],log5["repeat_probability"],
                                      log7["repeat_probability"],base[held],alt[held])
        feature_id = f"policy{inner}.comparison_features"
        nodes[feature_id] = ArtifactNode(feature_id,parents=(f"policy{inner}.family5/source_sequence.fold0.prediction",
                                                             f"policy{inner}.family7/source_sequence.fold0.prediction"))
        for user in np.unique(subjects[held]): assert_prediction_provenance(str(user),[feature_id],nodes)
        feature_nodes.append(feature_id); covered[held] += 1
        branch_audits = {str(count):{key:value for key,value in branch[1].items()
                                   if key not in ("own_probability","repeat_probability")}
                         for count,branch in zip((5,7),branches)}
        partition_record = {"inner":inner,"train_indices":train.tolist(),"held_indices":held.tolist(),
                      "train_ids":ids[train].tolist(),"held_ids":ids[held].tolist(),
                      "train_subjects":sorted(set(subjects[train])),"held_subjects":sorted(set(subjects[held])),
                      "expert_fit_log":raw["fits"],"feature_node":feature_id,
                      "outer_expert_fits":{"head_nodes":[f"p418.outer.fold0.family{j}.head" for j in range(7)],
                          "train_indices":train.tolist(),"held_indices":held.tolist(),"train_ids":ids[train].tolist(),
                          "held_ids":ids[held].tolist(),"train_subjects":sorted(set(subjects[train]))},
                      "branches":branch_audits}
        audit.append(partition_record)
        if artifact_dir is not None:
            folder=Path(artifact_dir);folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f"policy{inner}_arrays.npz",sample_ids=ids[held],features=x[held],base=base[held],alt=alt[held],
                                expert_bank=raw["outer_probability_bank"],group5=log5["repeat_probability"],group7=log7["repeat_probability"])
            (folder/f"policy{inner}_provenance.json").write_text(json.dumps(partition_record,indent=2),encoding="utf-8")
        print(json.dumps({"event":"policy_partition_complete","inner":inner}),flush=True)
        if deadline is not None and time.monotonic()>deadline: raise TimeoutError("P423 pilot exceeded600seconds")
    if not np.all(covered==1) or np.any(base<0) or np.any(alt<0): raise ProtocolError("incomplete source policy predictions")
    return {"features":x,"base":base,"alt":alt,"nodes":nodes,"feature_nodes":feature_nodes,"partitions":audit}


def fit_policy(features, base, alt, labels):
    gain = (np.asarray(alt)==labels).astype(int)-(np.asarray(base)==labels).astype(int)
    informative = (np.asarray(alt)!=base)&(gain!=0)
    target = (gain[informative]>0).astype(int)
    info = {"positive":int(target.sum()),"negative":int((target==0).sum()),
            "neutral_disagreements":int(np.sum((np.asarray(alt)!=base)&(gain==0))),
            "fit_rows":np.flatnonzero(informative).tolist()}
    if len(np.unique(target))<2:
        info["abstain"] = True
        return None,None,info
    scaler = StandardScaler().fit(np.asarray(features)[informative])
    model = LogisticRegression(C=.03,solver="lbfgs",max_iter=2000,class_weight=None).fit(
        scaler.transform(np.asarray(features)[informative]),target)
    info["abstain"] = False
    info["coefficient"] = model.coef_[0].tolist(); info["intercept"] = model.intercept_.tolist()
    info["scaler_mean"] = scaler.mean_.tolist(); info["scaler_scale"] = scaler.scale_.tolist()
    return scaler,model,info


def apply_policy(features, base, alt, scaler, model):
    base,alt = np.asarray(base),np.asarray(alt)
    probability = np.zeros(len(base)) if model is None else model.predict_proba(scaler.transform(features))[:,1]
    route = (alt!=base)&(probability>=.5)
    return np.where(route,alt,base),probability,route


def read_aligned(path, ids, users, folds, key):
    with np.load(path,allow_pickle=False) as z:
        for name,value in (("sample_ids",ids),("users",users),("fold_id",folds)):
            if not np.array_equal(z[name],value): raise ProtocolError("cached outer application alignment mismatch")
        return z[key].copy()


def held_application(fold, ids, users, folds):
    held = np.flatnonzero(folds==fold)
    base = read_aligned(P420/"predictions.npz",ids,users,folds,"source_sequence")[held]
    alt = read_aligned(P422/"predictions.npz",ids,users,folds,"source_sequence")[held]
    with np.load(P421/f"fold{fold}_banks.npz",allow_pickle=False) as z:
        if not np.array_equal(z["outer_sample_ids"],ids[held]): raise ProtocolError("outer expert IDs differ")
        bank = z["outer_probability_bank"].copy()
    with np.load(P419/f"fold{fold}_banks.npz",allow_pickle=False) as z:
        if not np.array_equal(z["outer_sample_ids"],ids[held]): raise ProtocolError("group5 IDs differ")
        group5 = z["repeat_probability"].copy()
    with np.load(P422/f"fold{fold}_group_probability.npz",allow_pickle=False) as z:
        if not np.array_equal(z["sample_ids"],ids[held]): raise ProtocolError("group7 IDs differ")
        group7 = z["repeat_probability"].copy()
    summary5 = json.loads((P420/"summary.json").read_text(encoding="utf-8"))["folds"][str(fold)]
    summary7 = json.loads((P422/"summary.json").read_text(encoding="utf-8"))["folds"][str(fold)]["sequence"]
    graph = namespace_nodes("application5",nodes_from_log(summary5))
    graph.update(namespace_nodes("application7",nodes_from_log(summary7)))
    feature_node = "application.comparison_features"
    graph[feature_node] = ArtifactNode(feature_node,parents=(f"application5/source_sequence.fold{fold}.prediction",
                                                           f"application7/source_sequence.fold{fold}.prediction"))
    for user in np.unique(users[held]): assert_prediction_provenance(str(user),[feature_node],graph)
    return {"features":comparative_features(bank,group5,group7,base,alt),"base":base,"alt":alt,
            "nodes":graph,"feature_node":feature_node}


def run_outer(families, labels, users, folds, fold, metadata, application, fitters=None, deadline=None, artifact_dir=None):
    train = np.flatnonzero(folds!=fold); held = np.flatnonzero(folds==fold)
    if set(users[train])&set(users[held]): raise ProtocolError("outer subjects overlap")
    # Physical subsetting is essential; no enclosing outer-held arrays enter nested fitting.
    options = {}
    if deadline is not None: options["deadline"] = deadline
    if artifact_dir is not None: options["artifact_dir"] = artifact_dir
    source = source_policy_data([np.asarray(x)[train] for x in families],np.asarray(labels)[train],
                                np.asarray(users)[train],_subset_metadata(metadata,train),fitters,**options)
    scaler,model,fit = fit_policy(source["features"],source["base"],source["alt"],np.asarray(labels)[train])
    prediction,score,route = apply_policy(application["features"],application["base"],application["alt"],scaler,model)
    nodes = {**source["nodes"],**application["nodes"]}
    nodes["gate.head"] = ArtifactNode("gate.head",parents=tuple(source["feature_nodes"]),provenance="supervised",
                                     has_task_labels=True,supervised_train_subjects=frozenset(map(str,users[train])))
    nodes["gate.prediction"] = ArtifactNode("gate.prediction",parents=("gate.head",application["feature_node"]))
    for user in np.unique(users[held]): assert_prediction_provenance(str(user),["gate.prediction"],nodes)
    log = {"outer_train_ids":np.asarray(metadata.sample_ids)[train].astype(str).tolist(),
           "outer_held_ids":np.asarray(metadata.sample_ids)[held].astype(str).tolist(),
           "outer_train_subjects":sorted(set(map(str,users[train]))),"outer_held_subjects":sorted(set(map(str,users[held]))),
           "source_partitions":source["partitions"],"gate_fit":fit,"artifact_dag":serialize_nodes(nodes),"provenance_checked":True}
    arrays = {"source_features":source["features"],"source_base":source["base"],"source_alt":source["alt"],
              "source_ids":np.asarray(metadata.sample_ids)[train],"held_features":application["features"],
              "held_ids":np.asarray(metadata.sample_ids)[held],"base":application["base"],"alt":application["alt"],
              "prediction":prediction,"gate_probability":score,"route":route}
    return held,prediction,log,arrays


def validate_pilot_receipt(pilot, registered_spec, current_spec):
    seconds = pilot.get("elapsed_seconds",float("inf"))
    if pilot.get("mode")!="pilot" or "evaluation" in pilot or not np.isfinite(seconds) or not 0<=seconds<=600:
        raise ProtocolError("pilot failed budget/score-blind contract")
    if set(pilot.get("folds",{}))!={"0"} or not pilot["folds"]["0"].get("provenance_checked"):
        raise ProtocolError("pilot fold/provenance incomplete")
    if len(pilot["folds"]["0"].get("source_partitions",[]))!=3:
        raise ProtocolError("pilot source partitions incomplete")
    for key in ("source_sha256","input_sha256","recipe"):
        if current_spec[key]!=registered_spec[key]: raise ProtocolError("pilot recipe/input changed")


def main(argv=None):
    ap = argparse.ArgumentParser(); modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--pilot",action="store_true");modes.add_argument("--run",action="store_true")
    ap.add_argument("--out-dir",required=True);ap.add_argument("--pilot-dir")
    args = ap.parse_args(argv);out = Path(args.out_dir)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=False);verify_source_snapshots()
    parent_reg = json.loads((P421/"experiment_registry.json").read_text(encoding="utf-8"))["spec"]
    sources = [Path(__file__),PREREG,*[HERE/name for name in (
        "p416_nested_frozen_family_router.py","p418_nested_repeat_group_bridge.py","p419_vjepa_repeat_group_bridge.py",
        "p420_source_only_session_bridge.py","p421_raw_sensor_sequence_bridge.py","p422_protected_geometry_bridge.py",
        "p90_teacher_common.py","stable_routing_structure.py","stable_routing_protocol.py","audit_p87_sequence_decoder.py")]]
    application_paths = [META,P420/"predictions.npz",P420/"summary.json",P422/"predictions.npz",P422/"summary.json",
        *[P421/f"fold{f}_banks.npz" for f in OUTER_FOLDS],*[P419/f"fold{f}_banks.npz" for f in OUTER_FOLDS],
        *[P422/f"fold{f}_group_probability.npz" for f in OUTER_FOLDS]]
    inputs = {path.relative_to(ROOT).as_posix():sha(path) for path in application_paths}
    # Bind numerical confidence arrays to the recorded inputs of the sequence
    # artifacts whose DAGs will be used for outer application.
    for parent in (P420,P422):
        registry=json.loads((parent/"experiment_registry.json").read_text(encoding="utf-8"))
        register_experiment(parent/"experiment_registry.json",registry["spec"])
        recorded={key.replace("\\","/"):value for key,value in registry["spec"]["input_sha256"].items()}
        for relative,actual in inputs.items():
            if relative in recorded and actual!=recorded[relative]: raise ProtocolError("outer application confidence lineage changed")
    raw_paths = {**parent_reg["base_family_spec"]["source_sha256"],**parent_reg["input_sha256"]}
    for relative,expected in raw_paths.items():
        actual = sha(ROOT/relative)
        if actual!=expected: raise ProtocolError("raw source/descriptor inputs changed since P421")
        inputs[relative.replace("\\","/")] = actual
    spec = {"experiment":"P423_nested_arbitration","mode":"pilot" if args.pilot else "run",
            "source_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},"input_sha256":inputs,
            "recipe":{"features":31,"C":.03,"threshold":.5,"policy_folds":3,"group_folds":3,"cpu_threads":4},
            "primary":"gated_minus_p420","promotion_allowed":False}
    if args.run:
        if not args.pilot_dir: raise ProtocolError("full run requires frozen pilot")
        pilot_dir = Path(args.pilot_dir)
        pilot = json.loads((pilot_dir/"pilot.json").read_text(encoding="utf-8"))
        registered = json.loads((pilot_dir/"experiment_registry.json").read_text(encoding="utf-8"))
        register_experiment(pilot_dir/"experiment_registry.json",registered["spec"])
        validate_pilot_receipt(pilot,registered["spec"],spec)
        spec["pilot_sha256"] = sha(pilot_dir/"pilot.json")
    register_experiment(out/"experiment_registry.json",spec);snap = out/"source_snapshot";snap.mkdir()
    for path in sources:(snap/path.name).write_bytes(path.read_bytes())
    started = time.monotonic();protocol = load_protocol()
    families,y,users,folds,ids = load_frozen_families(protocol)
    keep = ~np.isin(protocol.users.astype(str),list(EXCLUDED_USERS));assert_extraction_alignment(EXTRACTION_ROWS,protocol.sample_ids)
    families.append(load_vjepa()[keep])
    families.extend((load_descriptor(IMU_FEATURES,IMU_INDEX,ids,5222,"train")[0],
                     load_descriptor(SKEL_FEATURES,SKEL_INDEX,ids,9103)[0]))
    metadata = load_recording_metadata(META,ids)
    baseline = read_aligned(P420/"predictions.npz",ids,users,folds,"source_sequence")
    donor = read_aligned(P422/"predictions.npz",ids,users,folds,"source_sequence")
    predictions = np.full(len(ids),-1,int);logs = {}
    for fold in ((0,) if args.pilot else OUTER_FOLDS):
        application = held_application(fold,ids,users,folds)
        with threadpool_limits(limits=4),warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            held,prediction,log,arrays = run_outer(families,y,users,folds,fold,metadata,application,
                deadline=started+600 if args.pilot else None,artifact_dir=out/f"fold{fold}_source_policy")
        predictions[held] = prediction;logs[str(fold)] = log
        np.savez_compressed(out/f"fold{fold}_gate_arrays.npz",**arrays)
        (out/f"fold{fold}_provenance.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
        print(json.dumps({"event":"outer_fold_complete","fold":fold,"elapsed_seconds":time.monotonic()-started}),flush=True)
    np.savez_compressed(out/"predictions.npz",sample_ids=ids,users=users,fold_id=folds,p420_sequence=baseline,p422_sequence=donor,gated=predictions)
    result = {"mode":spec["mode"],"folds":logs,"target_achieved":False,"independent_confirmation":False,"test_rows_loaded":0}
    if args.run:
        if (predictions<0).any(): raise ProtocolError("incomplete gate predictions")
        result["evaluation"] = evaluate_outputs({"p420_sequence":baseline,"p422_sequence":donor,"gated":predictions},
                                                y,users,folds,base_key="p420_sequence")
        for key in ("p420_sequence","p422_sequence"):
            result["evaluation"][key].pop("criterion",None);result["evaluation"][key].pop("mechanism_gate_pass",None)
        result["primary_mechanism_gate_pass"] = result["evaluation"]["gated"]["mechanism_gate_pass"]
        changed=predictions!=baseline; disagreement=donor!=baseline
        result["routing"]={"candidate_disagreements":int(disagreement.sum()),"changed":int(changed.sum()),
            "neutral_changes":int(np.sum(changed&(predictions!=y)&(baseline!=y))),
            "abstained_disagreements":int(np.sum(disagreement&~changed)),
            "no_model_folds":[f for f,log in logs.items() if log["gate_fit"]["abstain"]],
            "per_subject":{str(user):{"changed":int(np.sum(changed&(users==user))),
                "neutral_changes":int(np.sum(changed&(users==user)&(predictions!=y)&(baseline!=y))),
                "abstained_disagreements":int(np.sum(disagreement&~changed&(users==user)))} for user in np.unique(users)}}
    result["elapsed_seconds"] = time.monotonic()-started
    if args.pilot and result["elapsed_seconds"]>600: raise TimeoutError("P423 pilot exceeded600seconds; no valid receipt")
    (out/("pilot.json" if args.pilot else "summary.json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"notes.txt").write_text("Fully nested source policy predictions; fixed31D gate. No Test or champion mutation.\n",encoding="utf-8")
    print(json.dumps({"event":"complete","mode":spec["mode"],"elapsed_seconds":result["elapsed_seconds"]}),flush=True)


if __name__ == "__main__":main()
