"""Keep standalone V-JEPA; add the stronger joint expert without moving geometry."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .p416_nested_frozen_family_router import EXCLUDED_USERS,OUTER_FOLDS,evaluate_outputs
from .p418_nested_repeat_group_bridge import load_recording_metadata
from .p419_vjepa_repeat_group_bridge import sha
from .p421_raw_sensor_sequence_bridge import nodes_from_log,REFERENCE,REFERENCE_HASHES,META
from .p422_protected_geometry_bridge import run_cached_fold
from .p423_nested_arbitration import namespace_nodes,serialize_nodes
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ArtifactNode,ProtocolError,register_experiment

ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/"aligned_multimodal"
BASE=HERE/"runs/p419_vjepa_repeat_group_bridge_v1";JOINT=HERE/"runs/p424_joint_visual_sequence_bridge_v1"
PREREG=ROOT/"docs/research/STABLE_093_P425_PREREGISTRATION_2026-09-07.md"


def combine_banks(base_inner,base_outer,base_log,joint_inner,joint_outer,joint_log,fold):
    for key in ("outer_train_indices","outer_held_indices","outer_train_ids","outer_held_ids"):
        if not np.array_equal(base_log[key],joint_log[key]):raise ProtocolError("combined expert source IDs/splits differ")
    pairs=((base_inner,joint_inner),(base_outer,joint_outer))
    for original,joint in pairs:
        if original.shape!=joint.shape or original.ndim!=3 or original.shape[1:]!=(5,40):raise ProtocolError("combined expert bank shape mismatch")
        if not np.array_equal(original[:,:4],joint[:,:4]):raise ProtocolError("shared four expert probabilities differ")
    log=copy.deepcopy(base_log);nodes=nodes_from_log(base_log)
    nodes.update(namespace_nodes("joint_source",nodes_from_log(joint_log)))
    nodes["external.family5"]=ArtifactNode("external.family5",parents=("joint_source/external.family4",))
    joint_records={r["node_id"]:r for r in joint_log["fits"] if r.get("family")==4}
    extra=[]
    for record in log["fits"]:
        if "peer_ancestors" not in record:continue
        original_joint_id=record["peer_ancestors"][-1]
        if original_joint_id not in joint_records:raise ProtocolError("joint inner head receipt missing")
        joint_record=joint_records[original_joint_id]
        if not np.array_equal(joint_record["validation_indices"],record["partition_indices"]):raise ProtocolError("joint inner validation indices differ")
        base_record=next(r for r in base_log["fits"] if r["node_id"]==original_joint_id)
        for key in ("train_indices","validation_indices","train_ids","validation_ids"):
            if not np.array_equal(base_record[key],joint_record[key]):raise ProtocolError("joint inner training context mismatch")
        alias=original_joint_id.rsplit("family",1)[0]+"family5"
        nodes[alias]=ArtifactNode(alias,parents=(f"joint_source/{original_joint_id}",))
        new_record=copy.deepcopy(joint_record);new_record.update(node_id=alias,family=5)
        extra.append(new_record);record["peer_ancestors"].append(alias)
        record["feature_plan_not_historical_fit"]=True
    if len(extra)!=3:raise ProtocolError("joint inner plan incomplete")
    outer_alias=f"p418.outer.fold{fold}.family5.head"
    nodes[outer_alias]=ArtifactNode(outer_alias,parents=(f"joint_source/p418.outer.fold{fold}.family4.head",))
    log["fits"].extend(extra);log["artifact_dag"]=serialize_nodes(nodes)
    log["combination"]="five P419 experts plus P424 joint expert; only feature plan is extended"
    return np.concatenate((base_inner,joint_inner[:,4:5]),1),np.concatenate((base_outer,joint_outer[:,4:5]),1),log


def verify_source(folder):
    reg=json.loads((folder/"experiment_registry.json").read_text(encoding="utf-8"));register_experiment(folder/"experiment_registry.json",reg["spec"])
    expected=reg["spec"]["source_sha256"];files=list((folder/"source_snapshot").iterdir())
    if {p.name for p in files}!={Path(k).name for k in expected}:raise ProtocolError("source snapshot set changed")
    for path in files:
        keys=[k for k in expected if Path(k).name==path.name]
        if len(keys)!=1 or sha(path)!=expected[keys[0]]:raise ProtocolError("source snapshot changed")
    data={k.replace("\\","/"):v for k,v in {**reg["spec"]["base_family_spec"]["source_sha256"],**reg["spec"]["input_sha256"]}.items()}
    for path in (META,HERE/"data/manifest.csv",*[HERE/f"data/subject_folds/fold_{f}.csv" for f in OUTER_FOLDS]):
        if data.get(path.relative_to(ROOT).as_posix())!=sha(path):raise ProtocolError("labels/splits/metadata changed since expert fitting")


def read_fold(folder,fold):
    log=json.loads((folder/f"fold{fold}_provenance.json").read_text(encoding="utf-8"))
    with np.load(folder/f"fold{fold}_banks.npz",allow_pickle=False) as z:
        if not np.array_equal(z["inner_sample_ids"],log["outer_train_ids"]) or not np.array_equal(z["outer_sample_ids"],log["outer_held_ids"]):raise ProtocolError("bank/source ID mismatch")
        return z["inner_probability_bank"].copy(),z["outer_probability_bank"].copy(),log


def pilot_receipt(path,spec):
    if path is None:raise ProtocolError("full run requires --pilot-dir")
    folder=Path(path);pilot=json.loads((folder/"pilot.json").read_text(encoding="utf-8"))
    reg=json.loads((folder/"experiment_registry.json").read_text(encoding="utf-8"));register_experiment(folder/"experiment_registry.json",reg["spec"])
    if pilot.get("mode")!="pilot" or "evaluation" in pilot or set(pilot.get("folds",{}))!={"0"} or not pilot["folds"]["0"].get("provenance_checked"):
        raise ProtocolError("pilot incomplete or not score-blind")
    for key in ("source_sha256","input_sha256","classifier_families","geometry_families"):
        if spec[key]!=reg["spec"][key]:raise ProtocolError("pilot inputs/recipe changed")
    return {"summary_sha256":sha(folder/"pilot.json"),"registry_sha256":sha(folder/"experiment_registry.json")}


def main(argv=None):
    parser=argparse.ArgumentParser();m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument("--pilot",action="store_true");m.add_argument("--run",action="store_true")
    parser.add_argument("--out-dir",required=True);parser.add_argument("--pilot-dir");args=parser.parse_args(argv);out=Path(args.out_dir)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=False);verify_source(BASE);verify_source(JOINT)
    for name,expected in REFERENCE_HASHES.items():
        if sha(REFERENCE/name)!=expected:raise ProtocolError("frozen P420 changed")
    sources=[Path(__file__),PREREG,*[HERE/name for name in (
        "p416_nested_frozen_family_router.py","p418_nested_repeat_group_bridge.py","p419_vjepa_repeat_group_bridge.py",
        "p420_source_only_session_bridge.py","p421_raw_sensor_sequence_bridge.py","p422_protected_geometry_bridge.py",
        "p423_nested_arbitration.py","p424_joint_visual_sequence_bridge.py","p90_teacher_common.py",
        "stable_routing_structure.py","stable_routing_protocol.py","audit_p87_sequence_decoder.py")]]
    inputs=[META,HERE/"data/manifest.csv",REFERENCE/"predictions.npz",
        *[HERE/f"data/subject_folds/fold_{f}.csv" for f in OUTER_FOLDS],
        *[folder/name for folder in (BASE,JOINT) for name in ("experiment_registry.json","predictions.npz","summary.json")],
        *[folder/f"fold{f}_{kind}" for folder in (BASE,JOINT) for f in OUTER_FOLDS for kind in ("banks.npz","provenance.json")]]
    spec={"experiment":"P425_complementary_joint_bridge","mode":"pilot" if args.pilot else "run",
        "source_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},"input_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
        "classifier_families":6,"geometry_families":5,"base_head_refits":0,"primary":"source_sequence_minus_frozen_p420","promotion_allowed":False}
    if args.run:spec["pilot_receipt"]=pilot_receipt(args.pilot_dir,spec)
    register_experiment(out/"experiment_registry.json",spec);snap=out/"source_snapshot";snap.mkdir()
    for path in sources:(snap/path.name).write_bytes(path.read_bytes())
    started=time.monotonic();protocol=load_protocol();keep=~np.isin(protocol.users.astype(str),list(EXCLUDED_USERS))
    ids,y,users,folds=(v[keep] for v in (protocol.sample_ids,protocol.labels,protocol.users,protocol.fold_id));metadata=load_recording_metadata(META,ids)
    with np.load(REFERENCE/"predictions.npz",allow_pickle=False) as z:
        for k,v in (("sample_ids",ids),("users",users),("fold_id",folds)):
            if not np.array_equal(z[k],v):raise ProtocolError("P420 reference misaligned")
        baseline=z["source_sequence"].copy()
    outputs={k:np.full(len(ids),-1,int) for k in ("own_group","repeat_group","unique_only","source_sequence")};logs={}
    for fold in ((0,) if args.pilot else OUTER_FOLDS):
        inner,outer,source_log=combine_banks(*read_fold(BASE,fold),*read_fold(JOINT,fold),fold)
        with threadpool_limits(limits=4),warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            held,pred,log=run_cached_fold(inner,outer,source_log,y,users,folds,fold,metadata,classifier_family_count=6,bank_family_count=6)
        for key in outputs:outputs[key][held]=pred[key]
        np.savez_compressed(out/f"fold{fold}_group_probability.npz",sample_ids=ids[held],own_probability=log.pop("own_probability"),repeat_probability=log.pop("repeat_probability"))
        logs[str(fold)]=log;(out/f"fold{fold}_provenance.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
        print(json.dumps({"event":"fold_complete","fold":fold,"elapsed_seconds":time.monotonic()-started}),flush=True)
    np.savez_compressed(out/"predictions.npz",sample_ids=ids,users=users,fold_id=folds,p420_sequence=baseline,**outputs)
    report={"mode":spec["mode"],"folds":logs,"target_achieved":False,"independent_confirmation":False,"test_rows_loaded":0}
    if args.run:
        if any((v<0).any() for v in outputs.values()):raise ProtocolError("incomplete predictions")
        report["evaluation"]=evaluate_outputs({"p420_sequence":baseline,**outputs},y,users,folds,base_key="p420_sequence")
        for key,value in report["evaluation"].items():
            if key!="source_sequence":value.pop("criterion",None);value.pop("mechanism_gate_pass",None)
        report["primary_mechanism_gate_pass"]=report["evaluation"]["source_sequence"]["mechanism_gate_pass"]
    report["elapsed_seconds"]=time.monotonic()-started
    (out/("pilot.json" if args.pilot else "summary.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out/"notes.txt").write_text("Keep five complementary experts; add joint as classifier-only sixth. No Test/head refits.\n",encoding="utf-8")
    print(json.dumps({"event":"complete","mode":spec["mode"],"elapsed_seconds":report["elapsed_seconds"]}),flush=True)


if __name__=="__main__":main()
