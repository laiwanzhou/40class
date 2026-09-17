"""Full actual-token bank gates; no target-accuracy evaluation."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from .p430_token_verify import verify_context
from .p430_token_provider import EXPERTS,BASE_SEEDS
from .p430_rebuild_token_bank import ROOT,sha
from .p427_foundation_provider import array_hash
from .stable_routing_protocol import ProtocolError,register_experiment


def _expected_pilot_keys():
    from . import p430_rebuild_token_bank as pilot
    sources={pilot._key(p) for p in pilot._source_files()}
    inputs={pilot._key(p) for p in [*[pilot.HERE/f"data/subject_folds/fold_{k}.csv" for k in range(3)],
        pilot.CACHE/"features.npy",pilot.CACHE/"done.npy",pilot.CACHE/"cache_summary.json",
        pilot.CANONICAL,pilot.EXTRACTION,pilot.PREREG]}
    return sources,inputs


def expected_context(ids,labels,users,source,target,fold,name,cache_provenance):
    return {"context":f"fold{fold}.{name}","outer_fold":int(fold),
        "source_ids":ids[source].tolist(),"source_users":users[source].tolist(),
        "target_ids":ids[target].tolist(),"target_users":users[target].tolist(),
        "source_label_sha256":array_hash(labels[source]),
        "source_class_counts":np.bincount(labels[source],minlength=40).tolist(),"cache_provenance":cache_provenance}


def validate_pilot(pilot_dir,expected_fold0):
    p=Path(pilot_dir);s=json.loads((p/"summary.json").read_text());reg=json.loads((p/"experiment_registry.json").read_text())
    register_experiment(p/"experiment_registry.json",reg["spec"])
    source_keys,input_keys=_expected_pilot_keys()
    if set(reg["spec"].get("source_sha256",{}))!=source_keys or set(reg["spec"].get("input_sha256",{}))!=input_keys:
        raise ProtocolError("pilot source/input inventory incomplete")
    process=json.loads(Path(str(p)+".process.json").read_text())
    if (s.get("mode")!="pilot" or s.get("contexts_completed")!=1 or not 0<=s.get("elapsed_seconds",-1)<=900
        or s.get("target_achieved") is not False or s.get("complete_p315") is not False
        or s.get("outer_accuracy_evaluated") is not False or s.get("test_rows_loaded")!=0
        or process.get("status")!="complete" or process.get("exit_code")!=0 or not 0<=process.get("seconds",-1)<=960):
        raise ProtocolError("pilot did not complete its registered scope/budget")
    if reg["spec"].get("expert_names")!=list(EXPERTS):raise ProtocolError("pilot expert order differs")
    # Exact mandatory artifact inventory: expected+registry+bank+provenance+6x3.
    inventory={"expected.json","experiment_registry.json","fold0/outer/bank.npz","fold0/outer/provenance.json"}
    for expert,bases in zip(EXPERTS,BASE_SEEDS):
        for seed in bases:
            inventory.update(f"fold0/outer/members/{expert}/seed{seed}/{f}" for f in ("checkpoint.pt","outputs.npz","receipt.json"))
    if set(s.get("artifact_sha256",{}))!=inventory:raise ProtocolError("pilot artifact inventory differs")
    for rel,digest in s["artifact_sha256"].items():
        if sha(p/rel)!=digest:raise ProtocolError("pilot artifact changed")
    for rel,digest in reg["spec"]["source_sha256"].items():
        if sha(ROOT/rel)!=digest or sha(p/"source_snapshot"/"__".join(Path(rel).parts))!=digest:
            raise ProtocolError("pilot core/source snapshot changed")
    for rel,digest in reg["spec"]["input_sha256"].items():
        if sha(ROOT/rel)!=digest:raise ProtocolError("pilot input changed")
    if json.loads((p/"expected.json").read_text())!=expected_fold0:raise ProtocolError("pilot canonical context differs")
    verify_context(p/"fold0/outer",expected_fold0)
    return {"summary_sha256":sha(p/"summary.json"),"registry_sha256":sha(p/"experiment_registry.json"),
            "process_sha256":sha(Path(str(p)+".process.json"))}


def compare_first_context(folder,pilot_dir):
    folder=Path(folder);pilot=Path(pilot_dir)/"fold0/outer"
    with np.load(folder/"bank.npz",allow_pickle=False) as actual,np.load(pilot/"bank.npz",allow_pickle=False) as prior:
        if set(actual.files)!=set(prior.files) or any(not np.array_equal(actual[k],prior[k]) for k in actual.files):
            raise ProtocolError("first full bank differs from pilot")
    for expert,seeds in zip(EXPERTS,BASE_SEEDS):
        for seed in seeds:
            rel=f"members/{expert}/seed{seed}/checkpoint.pt"
            a=torch.load(folder/rel,map_location="cpu",weights_only=True);b=torch.load(pilot/rel,map_location="cpu",weights_only=True)
            if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):raise ProtocolError("first full weights differ from pilot")
    return {"bank_bit_exact":True,"six_checkpoints_bit_exact":True}


def verify_full(out,ids,labels,users,folds,cache_provenance):
    out=Path(out);ids=np.asarray(ids).astype(str);labels=np.asarray(labels);users=np.asarray(users).astype(str);folds=np.asarray(folds)
    if (len(ids)!=2470 or len(set(ids))!=2470 or set(folds)!={0,1,2}
        or set(users)&{"user1","user2","user21"}):raise ProtocolError("invalid full research population")
    outer_ids=[];checked=0
    for fold in range(3):
        source=np.flatnonzero(folds!=fold);target=np.flatnonzero(folds==fold)
        requests=[("outer",source,target,None)]
        requests.extend((f"inner{k}",source[tr],source[va],va) for k,(tr,va) in enumerate(GroupKFold(3).split(source,groups=users[source])))
        coverage=np.zeros(len(source),int)
        with np.load(out/f"fold{fold}_banks.npz",allow_pickle=False) as aggregate:
            if (aggregate["inner_probability_bank"].shape!=(len(source),2,40)
                or aggregate["outer_probability_bank"].shape!=(len(target),2,40)):
                raise ProtocolError("aggregate bank shape differs")
            for key,value in (("inner_sample_ids",ids[source]),("outer_sample_ids",ids[target]),
                              ("inner_users",users[source]),("outer_users",users[target]),("expert_names",np.asarray(EXPERTS))):
                if not np.array_equal(aggregate[key],value):raise ProtocolError("aggregate identities differ")
            for name,tr,va,local in requests:
                expected=expected_context(ids,labels,users,tr,va,fold,name,cache_provenance)
                folder=out/f"fold{fold}/{name}"
                if json.loads((folder/"expected.json").read_text())!=expected:raise ProtocolError("full context expected population changed")
                verify_context(folder,expected);checked+=1
                with np.load(folder/"bank.npz",allow_pickle=False) as bank:
                    combined=aggregate["outer_probability_bank"] if local is None else aggregate["inner_probability_bank"][local]
                    if not np.array_equal(bank["probabilities"],combined):raise ProtocolError("aggregate/context bank differs")
                if local is None:outer_ids.extend(ids[va].tolist())
                else:coverage[local]+=1
        if not np.all(coverage==1):raise ProtocolError("inner population not covered exactly once")
    if len(outer_ids)!=len(set(outer_ids)) or set(outer_ids)!=set(ids):raise ProtocolError("outer coverage differs")
    return {"contexts_verified":checked,"members_verified":checked*6,"outer_unique_rows":len(outer_ids),"outer_accuracy_evaluated":False}
