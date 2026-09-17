"""Persist nested, source-calibrated P315 foundation banks; no partial champion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from .p427_foundation_provider import FoundationProvider,EXPERT_NAMES,array_hash
from .p419_vjepa_repeat_group_bridge import sha
from .p416_nested_frozen_family_router import EXCLUDED_USERS,OUTER_FOLDS
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import ArtifactNode,ProtocolError,register_experiment,assert_prediction_provenance

ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/"aligned_multimodal"
CACHE=HERE/"runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
CACHE_SUMMARY=CACHE.with_name("cache_summary.json")
CHAMPION_BANK=HERE/"runs/p307_union_repeat_group_sequence_audit_v1/summary.json"
PREREG=ROOT/"docs/research/STABLE_093_P427_FOUNDATION_REBUILD.md"


def load_raw(ids):
    meta=json.loads(CACHE_SUMMARY.read_text(encoding="utf-8"))
    if meta.get("model")!="MCG-NJU/videomae-large-finetuned-kinetics" or meta.get("trials")!=2914:
        raise ProtocolError("P85 external cache provenance/schema changed")
    with np.load(CACHE,allow_pickle=False) as archive:
        cache_ids=archive["sample_ids"].astype(str)
        if len(cache_ids)!=2914 or len(set(cache_ids))!=len(cache_ids):raise ProtocolError("P85 cache IDs not unique")
        if archive["window_names"].astype(str).tolist()!=["early","late"] or archive["view_names"].astype(str).tolist()!=["scene","person","workspace"]:
            raise ProtocolError("P85 view/window order changed")
        positions={sid:i for i,sid in enumerate(cache_ids)}
        if len(set(ids))!=len(ids) or any(sid not in positions for sid in ids):raise ProtocolError("P85/master ID join failed")
        order=np.asarray([positions[sid] for sid in ids],int)
        return archive["features"][order],archive["kinetics_logits"][order],{
            "cache_rows":len(cache_ids),"selected_rows":len(ids),"order":"explicit unique-ID join",
            "cache_label_fields_read":False,"model":meta["model"],"checkpoint_snapshot":meta.get("snapshot"),
            "provenance":"historical extraction source attestation; external model frozen, no task-label fitting"}


def verify_context(bank_path,receipt_path):
    receipt=json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    nodes={key:ArtifactNode(key,parents=tuple(v["parents"]),provenance=v["provenance"],has_task_labels=v["has_task_labels"],
                           supervised_train_subjects=frozenset(v["supervised_train_subjects"])) for key,v in receipt["artifact_dag"].items()}
    with np.load(bank_path,allow_pickle=False) as z:
        ids=z["sample_ids"].astype(str);p=z["probabilities"]
        if len(set(ids))!=len(ids) or not np.array_equal(ids,receipt["target_ids"]):raise ProtocolError("context bank IDs invalid")
        if p.shape!=(len(ids),len(EXPERT_NAMES),40) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(2),1):raise ProtocolError("context bank probabilities invalid")
        if z["expert_names"].astype(str).tolist()!=list(EXPERT_NAMES) or receipt["expert_names"]!=list(EXPERT_NAMES):raise ProtocolError("context expert order changed")
        if set(z["users"].astype(str))!=set(receipt["target_subjects"]) or set(receipt["source_ids"])&set(ids):raise ProtocolError("context source/target metadata invalid")
        for subject in receipt["target_subjects"]:assert_prediction_provenance(subject,receipt["prediction_nodes"],nodes)
        for number,node in receipt["calibration_array_nodes"].items():
            record=receipt["fit_receipts"][node]
            if not np.array_equal(z[f"calibration{number}_ids"],record["fit_ids"]):raise ProtocolError("calibration IDs mismatch")
            if set(z[f"calibration{number}_ids"].astype(str))&set(ids):raise ProtocolError("target in calibration data")
            if array_hash(z[f"calibration{number}_scores"])!=record["oof_score_sha256"] or array_hash(z[f"calibration{number}_labels"])!=record["source_label_sha256"]:
                raise ProtocolError("calibration payload changed")
    return receipt


def validate_pilot(path,spec):
    if path is None:raise ProtocolError("full generation requires successful --pilot-dir")
    folder=Path(path);pilot=json.loads((folder/"pilot.json").read_text());reg=json.loads((folder/"experiment_registry.json").read_text())
    register_experiment(folder/"experiment_registry.json",reg["spec"])
    if pilot.get("mode")!="pilot" or not 0<=pilot["elapsed_seconds"]<=600 or set(pilot.get("folds",{}))!={"0"}:
        raise ProtocolError("pilot incomplete or overbudget")
    if pilot["folds"]["0"].get("contexts_completed")!=4 or not pilot["folds"]["0"].get("provenance_checked"):
        raise ProtocolError("pilot contexts/provenance incomplete")
    for key in ("source_sha256","input_sha256","expert_names"):
        if reg["spec"][key]!=spec[key]:raise ProtocolError("pilot configuration/input mismatch")
    expected_files={"fold0_banks.npz",*[f"fold0/{name}_{kind}" for name in ("outer","inner0","inner1","inner2") for kind in ("bank.npz","provenance.json")]}
    if set(pilot.get("artifact_sha256",{}))!=expected_files:raise ProtocolError("pilot artifact inventory incomplete")
    for relative,expected in pilot["artifact_sha256"].items():
        if sha(folder/relative)!=expected:raise ProtocolError("pilot artifact hash changed")
    receipts=[verify_context(folder/f"fold0/{name}_bank.npz",folder/f"fold0/{name}_provenance.json") for name in ("outer","inner0","inner1","inner2")]
    with np.load(folder/"fold0_banks.npz",allow_pickle=False) as z:
        if z["expert_names"].astype(str).tolist()!=list(EXPERT_NAMES):raise ProtocolError("pilot combined expert names changed")
        if not np.array_equal(z["inner_sample_ids"],receipts[0]["source_ids"]) or not np.array_equal(z["outer_sample_ids"],receipts[0]["target_ids"]):raise ProtocolError("pilot combined IDs changed")
        inner_ids=z["inner_sample_ids"].astype(str);positions={sid:i for i,sid in enumerate(inner_ids)};covered=np.zeros(len(inner_ids),int)
        for name,receipt in zip(("inner0","inner1","inner2"),receipts[1:]):
            ix=np.asarray([positions[sid] for sid in receipt["target_ids"]]);covered[ix]+=1
            with np.load(folder/f"fold0/{name}_bank.npz",allow_pickle=False) as part:
                if not np.array_equal(part["probabilities"],z["inner_probability_bank"][ix]):raise ProtocolError("pilot inner bank assembly differs")
        if not np.all(covered==1):raise ProtocolError("pilot inner coverage invalid")
        with np.load(folder/"fold0/outer_bank.npz",allow_pickle=False) as part:
            if not np.array_equal(part["probabilities"],z["outer_probability_bank"]):raise ProtocolError("pilot outer bank assembly differs")
    return {"pilot_summary_sha256":sha(folder/"pilot.json"),"pilot_registry_sha256":sha(folder/"experiment_registry.json")}


def main(argv=None):
    parser=argparse.ArgumentParser();m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument("--pilot",action="store_true");m.add_argument("--run",action="store_true")
    parser.add_argument("--out-dir",required=True);parser.add_argument("--pilot-dir")
    args=parser.parse_args(argv);out=Path(args.out_dir)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),PREREG,*[HERE/name for name in (
        "p427_foundation_provider.py","p427_foundation_kernels.py","p90_teacher_common.py","stable_routing_protocol.py","p416_nested_frozen_family_router.py",
        "build_p85_videomae_full40_multiclip_cache.py","build_p46_videomae_cache.py","build_p46_videomae_multiclip_cache.py",
        "train_p85_videomae_full40_head.py","audit_p86_teacher_mechanisms.py","train_p46_videomae_head.py")]]
    inputs=[CACHE,CACHE_SUMMARY,CHAMPION_BANK,HERE/"data/manifest.csv",*[HERE/f"data/subject_folds/fold_{f}.csv" for f in OUTER_FOLDS]]
    required=json.loads(CHAMPION_BANK.read_text(encoding="utf-8"))["experts"]
    if len(required)!=30 or not set(EXPERT_NAMES).issubset(required):raise ProtocolError("foundation names do not match actual champion bank")
    spec={"experiment":"P427_actual_foundation_rebuild","mode":"pilot" if args.pilot else "run",
        "source_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        "input_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},"expert_names":list(EXPERT_NAMES),
        "required_champion_names":required,"cpu_threads":4,"partial_bank_only":True,"promotion_allowed":False}
    if args.run:spec["pilot_receipt"]=validate_pilot(args.pilot_dir,spec)
    register_experiment(out/"experiment_registry.json",spec);snap=out/"source_snapshot";snap.mkdir()
    for p in sources:(snap/p.name).write_bytes(p.read_bytes())
    started=time.monotonic();deadline=started+600 if args.pilot else None
    protocol=load_protocol();keep=~np.isin(protocol.users.astype(str),list(EXCLUDED_USERS))
    ids,y,users,folds=(v[keep] for v in (protocol.sample_ids,protocol.labels,protocol.users,protocol.fold_id))
    raw,kinetics,cache_receipt=load_raw(ids);provider=FoundationProvider(raw,kinetics,ids)
    report={"mode":spec["mode"],"cache":cache_receipt,"folds":{},"target_achieved":False,"complete_p315":False,
        "missing_champion_experts":[name for name in required if name not in EXPERT_NAMES],"test_rows_loaded":0,"held_accuracy_evaluated":False}
    for fold in ((0,) if args.pilot else OUTER_FOLDS):
        train=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
        inner_bank=np.zeros((len(train),16,40));coverage=np.zeros(len(train),int)
        folder=out/f"fold{fold}";folder.mkdir()
        contexts=[]
        # Full-source outer prediction first permits exact head-cache reuse when
        # the same calibration folds occur in later inner prediction contexts.
        requests=[("outer",train,held,None)]
        for inner,(tr,va) in enumerate(GroupKFold(3).split(train,y[train],users[train])):
            requests.append((f"inner{inner}",train[tr],train[va],va))
        for name,source,target,local in requests:
            context=f"fold{fold}.{name}"
            with threadpool_limits(limits=4):
                bank,receipt,aux=provider.fit_predict(source,y[source],users[source],target,users[target],context=context,deadline=deadline)
            np.savez_compressed(folder/f"{name}_bank.npz",sample_ids=ids[target],users=users[target],
                expert_names=np.asarray(EXPERT_NAMES),probabilities=bank.astype(np.float32),**aux)
            (folder/f"{name}_provenance.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
            contexts.append({"name":name,"target_rows":len(target),"source_rows":len(source),"new_ridge_fits":receipt["new_ridge_fits"],
                             "seconds":receipt["seconds"],"provenance_checked":receipt["provenance_checked"]})
            if local is None:outer_bank=bank
            else:inner_bank[local]=bank;coverage[local]+=1
            print(json.dumps({"event":"foundation_context_complete","context":context,"seconds":time.monotonic()-started}),flush=True)
        if not np.all(coverage==1):raise ProtocolError("inner foundation bank not covered exactly once")
        np.savez_compressed(out/f"fold{fold}_banks.npz",inner_sample_ids=ids[train],outer_sample_ids=ids[held],
            inner_users=users[train],outer_users=users[held],expert_names=np.asarray(EXPERT_NAMES),
            inner_probability_bank=inner_bank.astype(np.float32),outer_probability_bank=outer_bank.astype(np.float32))
        report["folds"][str(fold)]={"contexts":contexts,"contexts_completed":len(contexts),"provenance_checked":all(c["provenance_checked"] for c in contexts),
                                    "outer_train_ids":ids[train].tolist(),"outer_held_ids":ids[held].tolist()}
    report["artifact_sha256"]={p.relative_to(out).as_posix():sha(p) for p in out.rglob("*") if p.is_file() and "source_snapshot" not in p.parts and p.name!="experiment_registry.json"}
    report["unique_ridge_fits"]=provider.model_fit_count;report["temperature_fits"]=len(provider.calibrators)
    report["elapsed_seconds"]=time.monotonic()-started
    if args.pilot and report["elapsed_seconds"]>600:raise TimeoutError("foundation pilot exceeded600seconds")
    (out/("pilot.json" if args.pilot else "summary.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out/"notes.txt").write_text("Actual16P315foundation slots rebuilt with contextual temperatures. Partialbank only; no held accuracy,Test or promotion.\n",encoding="utf-8")
    print(json.dumps({"event":"foundation_complete","mode":spec["mode"],"unique_fits":provider.model_fit_count,
                      "seconds":report["elapsed_seconds"],"complete_p315":False,"target_achieved":False}),flush=True)


if __name__=="__main__":main()
