"""Append a fixed small attention expert; retain frozen five-family geometry."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import copy
import json
from pathlib import Path
import time
import warnings
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from . import p426_attention_head as attention
from .p416_nested_frozen_family_router import EXCLUDED_USERS,OUTER_FOLDS,evaluate_outputs
from .p418_nested_repeat_group_bridge import load_recording_metadata
from .p419_vjepa_repeat_group_bridge import sha,VJEPA_CACHE,VJEPA_DONE,VJEPA_SUMMARY,EXTRACTION_ROWS,assert_extraction_alignment
from .p421_raw_sensor_sequence_bridge import nodes_from_log,REFERENCE,REFERENCE_HASHES,META
from .p422_protected_geometry_bridge import run_cached_fold
from .p423_nested_arbitration import serialize_nodes
from .p425_complementary_joint_bridge import verify_source,read_fold,BASE
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ArtifactNode,ProtocolError,register_experiment

ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/"aligned_multimodal"
PREREG=ROOT/"docs/research/STABLE_093_P426_PREREGISTRATION_2026-09-07.md"
MEMORY_BUDGET=4*1024**3


def configure_cuda():
    if not torch.cuda.is_available():raise RuntimeError("P426 real experiment requires CUDA")
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(MEMORY_BUDGET/total,1.0),device=0)
    return {"torch":str(torch.__version__),"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(0),
            "pytorch_memory_budget_bytes":MEMORY_BUDGET,"deterministic_algorithms":True,"sdpa":"math",
            "cublas_workspace":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"tf32":bool(torch.backends.cuda.matmul.allow_tf32)}


def validate_base_plan(log,ids,users,folds,fold):
    train=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
    for key,value in (("outer_train_indices",train),("outer_held_indices",held),("outer_train_ids",ids[train]),("outer_held_ids",ids[held])):
        if not np.array_equal(log[key],value):raise ProtocolError("cached outer attention plan differs from current protocol")
    expected={frozenset(train[va].tolist()) for _,va in GroupKFold(3).split(train,groups=users[train])}
    plans=[r for r in log["fits"] if "peer_ancestors" in r]
    if len(plans)!=3 or {frozenset(r["partition_indices"]) for r in plans}!=expected:raise ProtocolError("cached inner attention plan differs")
    for plan in plans:
        va=np.asarray(plan["partition_indices"],int);tr=np.setdiff1d(train,va)
        record=next(r for r in log["fits"] if r["node_id"]==plan["peer_ancestors"][-1])
        if not np.array_equal(record["train_indices"],tr) or not np.array_equal(record["validation_indices"],va):
            raise ProtocolError("cached attention training rows differ from GroupKFold")
        if not np.array_equal(record["train_ids"],ids[tr]) or not np.array_equal(record["validation_ids"],ids[va]):
            raise ProtocolError("cached attention training IDs differ")


def attach_attention(inner,outer,source_log,tokens,labels,users,ids,fold,folder,deadline=None,fitter=None):
    log=copy.deepcopy(source_log);nodes=nodes_from_log(log)
    if inner.shape[1:]!=(5,40) or outer.shape[1:]!=(5,40):raise ProtocolError("P426 base bank must have five experts")
    outer_train=np.asarray(log["outer_train_indices"],int);outer_held=np.asarray(log["outer_held_indices"],int)
    if not np.array_equal(log["outer_train_ids"],ids[outer_train]) or not np.array_equal(log["outer_held_ids"],ids[outer_held]):
        raise ProtocolError("P426 base bank IDs changed")
    use_fit=attention.fit_attention if fitter is None else fitter
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    nodes["external.family5"]=ArtifactNode("external.family5",provenance="frozen_external")
    inner_new=np.zeros((len(outer_train),40));covered=np.zeros(len(outer_train),int)
    lookup={int(row):i for i,row in enumerate(outer_train)};new_records=[];fit_audits=[]

    def run_head(name,train,held):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError("attention pilot exceeded900seconds")
        if set(map(str,users[train]))&set(map(str,users[outer_held])):raise ProtocolError("attention train intersects outer held subjects")
        attention.reset_fit_receipts()
        if fitter is None:torch.cuda.reset_peak_memory_stats()
        print(json.dumps({"event":"attention_fit_start","fold":fold,"fit":name,"train_rows":len(train)}),flush=True)
        _,_,probability=use_fit(tokens,labels,train,held,users)
        receipts=copy.deepcopy(attention.FIT_RECEIPTS);members=attention.last_member_logits()
        if [r["seed"] for r in receipts]!=list(attention.SEEDS) or members.shape!=(3,len(held),40):
            raise ProtocolError("attention seed/ensemble receipt incomplete")
        mean=members.astype(float).mean(0);expected=np.exp(mean-mean.max(1,keepdims=True));expected/=expected.sum(1,keepdims=True)
        if not np.allclose(probability,expected,atol=1e-7):raise ProtocolError("attention aggregation differs from mean logits")
        peak=int(torch.cuda.max_memory_allocated()) if fitter is None else 0
        if peak>MEMORY_BUDGET:raise MemoryError("attention pilot exceeded4GiB allocated memory")
        seed_nodes=[]
        for receipt in receipts:
            if receipt["train_indices"]!=train.tolist() or receipt["held_indices"]!=held.tolist():raise ProtocolError("attention receipt row mismatch")
            if receipt["train_subjects"]!=sorted(set(map(str,users[train]))) or receipt["held_subjects"]!=sorted(set(map(str,users[held]))):
                raise ProtocolError("attention receipt subjects differ")
            node=f"attention.{name}.seed{receipt['seed']}"
            nodes[node]=ArtifactNode(node,parents=("external.family5",),provenance="supervised",has_task_labels=True,
                                     supervised_train_subjects=frozenset(receipt["train_subjects"]))
            seed_nodes.append(node)
        np.savez_compressed(folder/f"{name}_logits.npz",train_ids=ids[train],held_ids=ids[held],
                            member_logits=members,probability=probability)
        audit={"name":name,"train_ids":ids[train].tolist(),"held_ids":ids[held].tolist(),
               "receipts":receipts,"peak_allocated_bytes":peak,"seed_nodes":seed_nodes}
        (folder/f"{name}_receipt.json").write_text(json.dumps(audit,indent=2),encoding="utf-8");fit_audits.append(audit)
        if fitter is None:torch.cuda.empty_cache()
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError("attention pilot exceeded900seconds")
        return probability,seed_nodes

    for plan in log["fits"]:
        if "peer_ancestors" not in plan:continue
        base_id=plan["peer_ancestors"][-1]
        record=next(r for r in source_log["fits"] if r["node_id"]==base_id)
        train=np.asarray(record["train_indices"],int);held=np.asarray(record["validation_indices"],int)
        if not np.array_equal(held,plan["partition_indices"]):raise ProtocolError("attention inner partition differs")
        name=base_id.split(".")[0]
        probability,parents=run_head(name,train,held)
        positions=np.asarray([lookup[int(row)] for row in held]);inner_new[positions]=probability;covered[positions]+=1
        alias=f"{name}.family5";nodes[alias]=ArtifactNode(alias,parents=tuple(parents))
        plan["peer_ancestors"].append(alias);plan["feature_plan_not_historical_fit"]=True
        new_record=copy.deepcopy(record);new_record.update(node_id=alias,family=5,attention_seeds=list(attention.SEEDS))
        new_records.append(new_record)
    if not np.all(covered==1) or len(new_records)!=3:raise ProtocolError("attention inner coverage incomplete")
    outer_new,parents=run_head("outer",outer_train,outer_held)
    nodes[f"p418.outer.fold{fold}.family5.head"]=ArtifactNode(f"p418.outer.fold{fold}.family5.head",parents=tuple(parents))
    log["fits"].extend(new_records);log["artifact_dag"]=serialize_nodes(nodes);log["attention_fits"]=fit_audits
    return np.concatenate((inner,inner_new[:,None]),1),np.concatenate((outer,outer_new[:,None]),1),log


def main(argv=None):
    parser=argparse.ArgumentParser();m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument("--pilot",action="store_true");m.add_argument("--run",action="store_true")
    parser.add_argument("--out-dir",required=True);parser.add_argument("--pilot-dir")
    args=parser.parse_args(argv);out=Path(args.out_dir)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=False);runtime=configure_cuda();verify_source(BASE)
    base_spec=json.loads((BASE/"experiment_registry.json").read_text(encoding="utf-8"))["spec"]
    base_inputs={key.replace("\\","/"):value for key,value in base_spec["input_sha256"].items()}
    for path in (VJEPA_CACHE,VJEPA_DONE,VJEPA_SUMMARY,EXTRACTION_ROWS):
        if base_inputs.get(path.relative_to(ROOT).as_posix())!=sha(path):raise ProtocolError("frozen VJEPA inputs changed since P419")
    for name,expected in REFERENCE_HASHES.items():
        if sha(REFERENCE/name)!=expected:raise ProtocolError("P420 reference changed")
    sources=[Path(__file__),PREREG,*[HERE/name for name in (
        "p426_attention_head.py","p142_vjepa_token_transformer_oof.py","p416_nested_frozen_family_router.py",
        "p418_nested_repeat_group_bridge.py","p419_vjepa_repeat_group_bridge.py","p420_source_only_session_bridge.py",
        "p421_raw_sensor_sequence_bridge.py","p422_protected_geometry_bridge.py","p423_nested_arbitration.py",
        "p425_complementary_joint_bridge.py","p90_teacher_common.py","stable_routing_structure.py",
        "stable_routing_protocol.py","audit_p87_sequence_decoder.py","p96_vjepa2_dense24_extractor.py","p90_videomae_lora_teacher.py")]]
    inputs=[VJEPA_CACHE,VJEPA_DONE,VJEPA_SUMMARY,EXTRACTION_ROWS,META,HERE/"data/manifest.csv",REFERENCE/"predictions.npz",
        *[HERE/f"data/subject_folds/fold_{f}.csv" for f in OUTER_FOLDS],BASE/"experiment_registry.json",
        *[BASE/f"fold{f}_{kind}" for f in OUTER_FOLDS for kind in ("banks.npz","provenance.json")]]
    spec={"experiment":"P426_attention_sequence_bridge","mode":"pilot" if args.pilot else "run","runtime":runtime,
        "source_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},"input_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
        "attention_recipe":vars(attention._args()),"seeds":list(attention.SEEDS),"head_parameters":953832,
        "primary":"source_sequence_minus_frozen_p420","base_ridge_refits":0,"promotion_allowed":False}
    if args.run:
        if not args.pilot_dir:raise ProtocolError("full run requires pilot")
        pd=Path(args.pilot_dir);pilot=json.loads((pd/"pilot.json").read_text());reg=json.loads((pd/"experiment_registry.json").read_text())
        register_experiment(pd/"experiment_registry.json",reg["spec"])
        if pilot.get("mode")!="pilot" or "evaluation" in pilot or not 0<=pilot["elapsed_seconds"]<=900 or set(pilot.get("folds",{}))!={"0"} or not pilot["folds"]["0"]["provenance_checked"]:
            raise ProtocolError("pilot incomplete/overbudget/not score-blind")
        for key in ("source_sha256","input_sha256","attention_recipe","seeds","runtime"):
            if spec[key]!=reg["spec"][key]:raise ProtocolError("pilot input/recipe/runtime changed")
        spec["pilot_summary_sha256"]=sha(pd/"pilot.json")
    register_experiment(out/"experiment_registry.json",spec);snap=out/"source_snapshot";snap.mkdir()
    for path in sources:(snap/path.name).write_bytes(path.read_bytes())
    started=time.monotonic();protocol=load_protocol();keep=~np.isin(protocol.users.astype(str),list(EXCLUDED_USERS))
    ids,y,users,folds=(v[keep] for v in (protocol.sample_ids,protocol.labels,protocol.users,protocol.fold_id))
    assert_extraction_alignment(EXTRACTION_ROWS,protocol.sample_ids)
    flags=np.load(VJEPA_DONE,allow_pickle=False);meta=json.loads(VJEPA_SUMMARY.read_text())
    if flags.shape!=(2914,) or flags.dtype!=np.bool_ or not flags.all() or not meta.get("label_free_extraction") or not meta.get("complete") or meta.get("view_count")!=24 or meta.get("model_repo")!="facebook/vjepa2-vitl-fpc16-256-ssv2":
        raise ProtocolError("raw attention cache is not complete/frozen")
    raw=np.load(VJEPA_CACHE,mmap_mode="r",allow_pickle=False)
    if raw.shape!=(2914,24,1024):raise ProtocolError("raw token shape mismatch")
    tokens=np.asarray(raw[keep]);metadata=load_recording_metadata(META,ids)
    with np.load(REFERENCE/"predictions.npz",allow_pickle=False) as z:
        for key,value in (("sample_ids",ids),("users",users),("fold_id",folds)):
            if not np.array_equal(z[key],value):raise ProtocolError("reference IDs/subjects/folds mismatch")
        baseline=z["source_sequence"].copy()
    outputs={k:np.full(len(ids),-1,int) for k in ("own_group","repeat_group","unique_only","source_sequence")};logs={}
    for fold in ((0,) if args.pilot else OUTER_FOLDS):
        inner,outer,base_log=read_fold(BASE,fold)
        validate_base_plan(base_log,ids,users,folds,fold)
        with threadpool_limits(limits=4),warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            inner,outer,source=attach_attention(inner,outer,base_log,tokens,y,users,ids,fold,out/f"fold{fold}_attention",started+900 if args.pilot else None)
            held,pred,log=run_cached_fold(inner,outer,source,y,users,folds,fold,metadata,classifier_family_count=6,bank_family_count=6)
        for key in outputs:outputs[key][held]=pred[key]
        np.savez_compressed(out/f"fold{fold}_banks.npz",inner_probability_bank=inner,outer_probability_bank=outer,
            inner_sample_ids=np.asarray(source["outer_train_ids"]),outer_sample_ids=ids[held],
            own_probability=log.pop("own_probability"),repeat_probability=log.pop("repeat_probability"))
        log["attention_fits"]=source["attention_fits"];logs[str(fold)]=log
        (out/f"fold{fold}_provenance.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
        print(json.dumps({"event":"fold_complete","fold":fold,"elapsed_seconds":time.monotonic()-started}),flush=True)
    np.savez_compressed(out/"predictions.npz",sample_ids=ids,users=users,fold_id=folds,p420_sequence=baseline,**outputs)
    result={"mode":spec["mode"],"folds":logs,"runtime":runtime,"target_achieved":False,"independent_confirmation":False,"test_rows_loaded":0}
    if args.run:
        if any((v<0).any() for v in outputs.values()):raise ProtocolError("incomplete attention predictions")
        result["evaluation"]=evaluate_outputs({"p420_sequence":baseline,**outputs},y,users,folds,base_key="p420_sequence")
        for key,value in result["evaluation"].items():
            if key!="source_sequence":value.pop("criterion",None);value.pop("mechanism_gate_pass",None)
        result["primary_mechanism_gate_pass"]=result["evaluation"]["source_sequence"]["mechanism_gate_pass"]
    result["elapsed_seconds"]=time.monotonic()-started
    if args.pilot and result["elapsed_seconds"]>900:raise TimeoutError("attention pilot exceeded900seconds")
    (out/("pilot.json" if args.pilot else "summary.json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"notes.txt").write_text("Three fixed small attention heads on frozen tokens; original5geometry. No bigencoder/Test/champion mutation.\n",encoding="utf-8")
    print(json.dumps({"event":"complete","mode":spec["mode"],"elapsed_seconds":result["elapsed_seconds"]}),flush=True)


if __name__=="__main__":main()
