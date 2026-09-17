"""Fixed joint frozen-visual feature head, nested group and source sequence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .p416_nested_frozen_family_router import (
    EXCLUDED_USERS,OUTER_FOLDS,load_frozen_families,fit_family_head,evaluate_outputs,_spec as family_spec,
)
from .p418_nested_repeat_group_bridge import run_outer_fold as group_fold,load_recording_metadata
from .p419_vjepa_repeat_group_bridge import sha,load_vjepa,assert_extraction_alignment,EXTRACTION_ROWS,VJEPA_CACHE,VJEPA_DONE,VJEPA_SUMMARY
from .p420_source_only_session_bridge import run_outer_fold as sequence_fold
from .p421_raw_sensor_sequence_bridge import nodes_from_log,REFERENCE,REFERENCE_HASHES,META
from .p90_teacher_common import load_protocol,softmax
from .stable_routing_protocol import ProtocolError,register_experiment

ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/"aligned_multimodal"
PREREG=ROOT/"docs/research/STABLE_093_P424_PREREGISTRATION_2026-09-07.md"
ACTION_LOGITS=VJEPA_CACHE.with_name("ssv2_logits.npy")


def standardize_rows(values):
    x=np.asarray(values,np.float32)
    centered=x-x.mean(axis=-1,keepdims=True)
    return centered/np.maximum(centered.std(axis=-1,keepdims=True),1e-6)


def joint_features(ir_vmae,ir_iv2,dense_group,action_logits):
    arrays=[np.asarray(v,np.float32) for v in (ir_vmae,ir_iv2,dense_group,action_logits)]
    n=len(arrays[0])
    expected=((n,4608),(n,4608),(n,8192),(n,24,174))
    if any(a.shape!=s for a,s in zip(arrays,expected)) or any(not np.isfinite(a).all() for a in arrays):
        raise ProtocolError("joint visual input shapes/finite values mismatch")
    action=standardize_rows(arrays[3]);action=standardize_rows(action.reshape(n,8,3,174).mean(axis=2)).reshape(n,1392)
    return np.concatenate((*arrays[:3],action),axis=1)


def source_weights(labels):
    y=np.asarray(labels)
    if y.ndim!=1 or not len(y) or not np.issubdtype(y.dtype,np.integer) or np.any((y<0)|(y>=40)):
        raise ProtocolError("invalid joint-head source labels")
    counts=np.bincount(y,minlength=40).astype(float);reference=counts[counts>0].mean()
    weights=(reference/counts[y])**.75
    return weights/weights.mean()


def fit_joint(features,labels,train_idx,held_idx,subjects):
    train=np.asarray(train_idx,int);held=np.asarray(held_idx,int);users=np.asarray(subjects).astype(str)
    if not len(train) or not len(held) or set(users[train])&set(users[held]): raise ProtocolError("joint-head subject split invalid")
    x=np.asarray(features,np.float32);y=np.asarray(labels)[train]
    if x.ndim!=2 or not np.isfinite(x).all(): raise ProtocolError("invalid joint matrix")
    weights=source_weights(y);scaler=StandardScaler().fit(x[train])
    head=RidgeClassifier(alpha=9000,solver="lsqr",tol=1e-5,max_iter=5000).fit(scaler.transform(x[train]),y,sample_weight=weights)
    raw=np.asarray(head.decision_function(scaler.transform(x[held])),float)
    score=np.full((len(held),40),-1e9)
    if len(head.classes_)==1: score[:,int(head.classes_[0])]=0
    else:
        if raw.ndim==1:raw=np.column_stack((-raw,raw))
        score[:,head.classes_.astype(int)]=raw
    print(json.dumps({"event":"joint_head_complete","rows":len(train),"width":x.shape[1],
        "class_count":len(head.classes_),"weight_min":float(weights.min()),"weight_max":float(weights.max())}),flush=True)
    return scaler,head,softmax(score)


def run_fold(families,labels,users,folds,fold,metadata):
    if len(families)!=5:raise ProtocolError("joint bridge requires five families")
    held,pred,log=group_fold(families,labels,users,folds,fold,metadata,
        family_fitters=[fit_family_head]*4+[fit_joint])
    probability=np.zeros((len(labels),40));probability[held]=log["repeat_probability"]
    hs,seq,seq_log=sequence_fold(probability,labels,users,folds,fold,metadata,nodes_from_log(log))
    if not np.array_equal(hs,held) or not np.array_equal(seq["p419_repeat"],pred["repeat_group"]):raise ProtocolError("joint sequence alignment failure")
    pred.update(unique_only=seq["unique_only"],source_sequence=seq["source_sequence"])
    log["sequence"]=seq_log
    for record in log["fits"]:
        if record.get("family")==4:
            w=source_weights(np.asarray(labels)[record["train_indices"]])
            record["joint_head"]={"alpha":9000,"weight_power":.75,"weight_min":float(w.min()),"weight_max":float(w.max()),"weight_mean":float(w.mean())}
    outer_weights=source_weights(np.asarray(labels)[log["outer_train_indices"]])
    log["outer_joint_head"]={"alpha":9000,"weight_power":.75,"train_ids":log["outer_train_ids"],
        "held_ids":log["outer_held_ids"],"weight_min":float(outer_weights.min()),
        "weight_max":float(outer_weights.max()),"weight_mean":float(outer_weights.mean()),
        "input_sources":["IR VideoMAEv2","IR InternVideo2","VJEPA dense+public SSV2 logits"],
        "not_independent_of_existing_ir_families":True}
    return held,pred,log


def main(argv=None):
    parser=argparse.ArgumentParser();m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument("--pilot",action="store_true");m.add_argument("--run",action="store_true")
    parser.add_argument("--out-dir",required=True);parser.add_argument("--pilot-dir")
    args=parser.parse_args(argv);out=Path(args.out_dir)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=False)
    for name,expected in REFERENCE_HASHES.items():
        if sha(REFERENCE/name)!=expected:raise ProtocolError("frozen P420 reference changed")
    sources=[Path(__file__),PREREG,*[HERE/name for name in (
        "p416_nested_frozen_family_router.py","p418_nested_repeat_group_bridge.py","p419_vjepa_repeat_group_bridge.py",
        "p420_source_only_session_bridge.py","p421_raw_sensor_sequence_bridge.py","p90_teacher_common.py",
        "stable_routing_structure.py","stable_routing_protocol.py","audit_p87_sequence_decoder.py",
        "p123_vjepa_dense24_full_oof.py","p96_vjepa2_dense24_teacher_h1h2.py","train_p46_videomae_head.py",
        "p90_videomaev2_distilled_teacher.py","p96_vjepa2_dense24_extractor.py","p90_videomae_lora_teacher.py")]]
    inputs=[ACTION_LOGITS,VJEPA_CACHE,VJEPA_DONE,VJEPA_SUMMARY,EXTRACTION_ROWS,META,REFERENCE/"predictions.npz",REFERENCE/"experiment_registry.json"]
    spec={"experiment":"P424_joint_visual_sequence_bridge","mode":"pilot" if args.pilot else "run",
        "base_family_spec":family_spec(),"source_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        "input_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
        "joint_recipe":{"width":18800,"alpha":9000,"weight_power":.75,"solver":"lsqr","tol":1e-5,"max_iter":5000},
        "primary":"source_sequence_minus_frozen_p420","promotion_allowed":False,"cpu_threads":4}
    if args.run:
        if not args.pilot_dir:raise ProtocolError("full run requires successful pilot")
        pd=Path(args.pilot_dir);pilot=json.loads((pd/"pilot.json").read_text());reg=json.loads((pd/"experiment_registry.json").read_text())
        register_experiment(pd/"experiment_registry.json",reg["spec"])
        if pilot.get("mode")!="pilot" or "evaluation" in pilot or not 0<=pilot["elapsed_seconds"]<=600 or set(pilot["folds"])!={"0"} or not pilot["folds"]["0"]["provenance_checked"]:
            raise ProtocolError("pilot incomplete/overbudget/not score-blind")
        for key in ("source_sha256","input_sha256","joint_recipe","base_family_spec"):
            if reg["spec"][key]!=spec[key]:raise ProtocolError("pilot input/recipe mismatch")
        spec["pilot_summary_sha256"]=sha(pd/"pilot.json")
    register_experiment(out/"experiment_registry.json",spec);snap=out/"source_snapshot";snap.mkdir()
    for source in sources:(snap/source.name).write_bytes(source.read_bytes())
    started=time.monotonic();protocol=load_protocol();families,y,users,folds,ids=load_frozen_families(protocol)
    keep=~np.isin(protocol.users.astype(str),list(EXCLUDED_USERS));assert_extraction_alignment(EXTRACTION_ROWS,protocol.sample_ids)
    dense=load_vjepa()[keep];action=np.load(ACTION_LOGITS,mmap_mode="r",allow_pickle=False)
    if action.shape!=(2914,24,174):raise ProtocolError("unexpected external SSV2 logits schema")
    families.append(joint_features(families[0],families[1],dense,action[keep]))
    metadata=load_recording_metadata(META,ids)
    with np.load(REFERENCE/"predictions.npz",allow_pickle=False) as z:
        for key,value in (("sample_ids",ids),("users",users),("fold_id",folds)):
            if not np.array_equal(z[key],value):raise ProtocolError("reference alignment mismatch")
        base=z["source_sequence"].copy()
    outputs={k:np.full(len(ids),-1,int) for k in ("own_group","repeat_group","equal_mean","selective_all","unique_only","source_sequence")};logs={}
    for fold in ((0,) if args.pilot else OUTER_FOLDS):
        with threadpool_limits(limits=4),warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            held,pred,log=run_fold(families,y,users,folds,fold,metadata)
        for key in outputs:outputs[key][held]=pred[key]
        arrays={key:log.pop(key) for key in ("inner_probability_bank","outer_probability_bank","own_probability","repeat_probability")}
        np.savez_compressed(out/f"fold{fold}_banks.npz",**arrays,inner_sample_ids=np.asarray(log["outer_train_ids"]),outer_sample_ids=ids[held])
        logs[str(fold)]=log;(out/f"fold{fold}_provenance.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
        print(json.dumps({"event":"fold_complete","fold":fold,"elapsed_seconds":time.monotonic()-started}),flush=True)
    np.savez_compressed(out/"predictions.npz",sample_ids=ids,users=users,fold_id=folds,p420_sequence=base,**outputs)
    result={"mode":spec["mode"],"folds":logs,"target_achieved":False,"independent_confirmation":False,"test_rows_loaded":0}
    if args.run:
        if any((p<0).any() for p in outputs.values()):raise ProtocolError("incomplete joint predictions")
        result["evaluation"]=evaluate_outputs({"p420_sequence":base,**outputs},y,users,folds,base_key="p420_sequence")
        for key,metrics in result["evaluation"].items():
            if key!="source_sequence":metrics.pop("criterion",None);metrics.pop("mechanism_gate_pass",None)
        result["primary_mechanism_gate_pass"]=result["evaluation"]["source_sequence"]["mechanism_gate_pass"]
    result["elapsed_seconds"]=time.monotonic()-started
    if args.pilot and result["elapsed_seconds"]>600:raise TimeoutError("joint pilot exceeded600seconds; no valid receipt")
    (out/("pilot.json" if args.pilot else "summary.json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"notes.txt").write_text("One joint visual family replacement; fixed historical weighted Ridge. No Test, new encoder or champion mutation.\n",encoding="utf-8")
    print(json.dumps({"event":"complete","mode":spec["mode"],"elapsed_seconds":result["elapsed_seconds"]}),flush=True)


if __name__=="__main__":main()
