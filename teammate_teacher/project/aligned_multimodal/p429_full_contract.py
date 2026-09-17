"""Source-pure full thermal bank and verified-pilot reuse contracts."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from thermal_baseline.thermal_tsm_model import ThermalResNetTSM
from thermal_baseline.thermal_oof_data import IMAGE_EXTENSIONS
from . import p429_rebuild_thermal_bank as pilot
from .p429_thermal_verify import verify_context as verify_core
from .p427_foundation_provider import array_hash
from .p429_thermal_data import load_path_map
from .stable_routing_protocol import ProtocolError,register_experiment

ROOT=pilot.ROOT


@lru_cache(maxsize=1)
def state_schema():
    with torch.device("meta"):model=ThermalResNetTSM(imagenet_pretrained=False)
    return {k:(tuple(v.shape),v.dtype) for k,v in model.state_dict().items()}


def verify_context(folder,expected):
    folder=Path(folder);r=verify_core(folder,expected);schema=state_schema()
    for path in [folder/"checkpoint.pt",*[folder/f"members/{name}/checkpoint.pt" for name in ("selection0","selection1","selection2","refit")]]:
        state=torch.load(path,map_location="cpu",weights_only=True)
        if {k:(tuple(v.shape),v.dtype) for k,v in state.items()}!=schema:
            raise ProtocolError("full thermal checkpoint schema mismatch")
    return r


def expected_context(population,source,target):
    ix,labels,users,ids,target_users=population.context(source,target)
    return {"source_ids":population.ids[ix].tolist(),"source_users":users.tolist(),
        "target_ids":ids.tolist(),"target_users":target_users.tolist(),
        "thermal_present":[s in population.paths for s in ids],"source_label_sha256":array_hash(labels)}


def pilot_inputs():
    return [pilot.CANONICAL_MANIFEST,*[pilot.HERE/f"data/subject_folds/fold_{k}.csv" for k in range(3)],
            pilot.THERMAL_FOLD,pilot.THERMAL_MANIFEST,pilot.CONFIG,Path(pilot._WEIGHTS)]


def _pilot_raw_paths(expected):
    ids=[*expected["source_ids"],*[s for s,present in zip(expected["target_ids"],expected["thermal_present"]) if present]]
    paths=load_path_map(pilot.THERMAL_FOLD,ids);result=set()
    for sid in ids:
        if sid not in paths:raise ProtocolError("pilot present ID missing raw thermal path")
        for path in paths[sid].iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                path=path.resolve()
                if not path.is_relative_to(ROOT/"Training"):raise ProtocolError("pilot raw path outside Training")
                result.add(str(path))
    return result


def validate_pilot(path,expected):
    path=Path(path);s=json.loads((path/"summary.json").read_text());reg=json.loads((path/"experiment_registry.json").read_text())
    register_experiment(path/"experiment_registry.json",reg["spec"])
    process=json.loads(Path(str(path)+".process.json").read_text())
    if (s.get("mode")!="pilot" or s.get("contexts_completed")!=1 or not 0<=s.get("elapsed_seconds",-1)<=7200
        or s.get("held_accuracy_evaluated") is not False or s.get("test_rows_loaded")!=0
        or s.get("complete_p315") is not False or s.get("target_achieved") is not False
        or process.get("status")!="complete" or process.get("exit_code")!=0 or not 0<=process.get("seconds",-1)<=7260):
        raise ProtocolError("thermal pilot incomplete or overbudget")
    source={pilot._key(p):p for p in pilot._sources()};inputs={pilot._key(p):p for p in pilot_inputs()}
    if (set(source)!=set(reg["spec"]["source_sha256"]) or set(inputs)!=set(reg["spec"]["input_sha256"])
        or reg["spec"].get("expert_names")!=["p12_thermal"]):raise ProtocolError("thermal pilot inventory incomplete")
    for key,p in source.items():
        h=reg["spec"]["source_sha256"][key]
        if pilot.sha(p)!=h or pilot.sha(path/"source_snapshot"/"__".join(Path(key).parts))!=h:
            raise ProtocolError("thermal pilot source changed")
    for key,p in inputs.items():
        if pilot.sha(p)!=reg["spec"]["input_sha256"][key]:raise ProtocolError("thermal pilot input changed")
    inventory={"raw_inventory.json","context_expected.json","fold0/outer/bank.npz","fold0/outer/checkpoint.pt","fold0/outer/provenance.json"}
    for name in ("selection0","selection1","selection2","refit"):
        inventory.update(f"fold0/outer/members/{name}/{f}" for f in ("checkpoint.pt","outputs.npz","receipt.json"))
    if set(s["artifact_sha256"])!=inventory:raise ProtocolError("thermal pilot output inventory incomplete")
    for rel,h in s["artifact_sha256"].items():
        if pilot.sha(path/rel)!=h:raise ProtocolError("thermal pilot artifact changed")
    if json.loads((path/"context_expected.json").read_text())!=expected:raise ProtocolError("thermal pilot population differs")
    raw=json.loads((path/"raw_inventory.json").read_text())["raw_frame_sha256"]
    if set(raw)!=_pilot_raw_paths(expected):raise ProtocolError("thermal pilot raw inventory incomplete")
    for p,h in raw.items():
        if not Path(p).resolve().is_relative_to(ROOT/"Training") or pilot.sha(p)!=h:
            raise ProtocolError("thermal pilot raw input differs")
    r=verify_context(path/"fold0/outer",expected)
    if r["context"]!="fold0.outer":raise ProtocolError("thermal pilot is wrong context")
    return {"summary_sha256":pilot.sha(path/"summary.json"),"registry_sha256":pilot.sha(path/"experiment_registry.json"),
        "process_sha256":pilot.sha(Path(str(path)+".process.json")),"selected_epoch":r["selected_epoch"],
        "context_file_sha256":{str(p.relative_to(path/"fold0/outer")).replace("\\","/"):pilot.sha(p) for p in (path/"fold0/outer").rglob("*") if p.is_file()}}


def verify_reused_context(folder,pilot_receipt,expected):
    folder=Path(folder)
    for rel,h in pilot_receipt["context_file_sha256"].items():
        if pilot.sha(folder/rel)!=h:raise ProtocolError("reused thermal context bytes differ")
    r=verify_context(folder,expected)
    if r["selected_epoch"]!=pilot_receipt["selected_epoch"] or r["context"]!="fold0.outer":
        raise ProtocolError("reused thermal context recipe differs")
    return r


def verify_full(out,population,canonical_ids,canonical_users,canonical_folds):
    out=Path(out);ids=np.asarray(canonical_ids).astype(str);users=np.asarray(canonical_users).astype(str);folds=np.asarray(canonical_folds)
    if (len(ids)!=2470 or len(set(ids))!=2470 or set(folds)!={0,1,2}
        or not np.array_equal(ids,population.canonical_ids) or not np.array_equal(users,population.canonical_users)
        or set(users)&{"user1","user2","user21"}):raise ProtocolError("invalid full thermal research population")
    seen=[];count=0
    for fold in range(3):
        source=np.flatnonzero(folds!=fold);target=np.flatnonzero(folds==fold)
        requests=[("outer",source,target,None)]
        requests.extend((f"inner{k}",source[tr],source[va],va) for k,(tr,va) in enumerate(GroupKFold(3).split(source,groups=users[source])))
        covered=np.zeros(len(source),int)
        with np.load(out/f"fold{fold}_banks.npz",allow_pickle=False) as aggregate:
            required={"inner_sample_ids","outer_sample_ids","inner_users","outer_users","expert_names",
                "inner_thermal_present","outer_thermal_present","inner_probability_bank","outer_probability_bank"}
            if set(aggregate.files)!=required:raise ProtocolError("unexpected thermal aggregate fields")
            for key,value in (("inner_sample_ids",ids[source]),("outer_sample_ids",ids[target]),("inner_users",users[source]),
                ("outer_users",users[target]),("expert_names",["p12_thermal"]),
                ("inner_thermal_present",population.present[source]),("outer_thermal_present",population.present[target])):
                if not np.array_equal(aggregate[key],value):raise ProtocolError("thermal aggregate identities/masks differ")
            if aggregate["inner_probability_bank"].shape!=(len(source),1,40) or aggregate["outer_probability_bank"].shape!=(len(target),1,40):
                raise ProtocolError("thermal aggregate shape differs")
            for name,tr,va,local in requests:
                expected=expected_context(population,tr,va);folder=out/f"fold{fold}/{name}"
                if json.loads((folder/"expected.json").read_text())!=expected:raise ProtocolError("thermal full expected context differs")
                r=verify_context(folder,expected)
                if r["context"]!=f"fold{fold}.{name}":raise ProtocolError("thermal context name differs")
                with np.load(folder/"bank.npz",allow_pickle=False) as bank:
                    combined=aggregate["outer_probability_bank"] if local is None else aggregate["inner_probability_bank"][local]
                    if not np.array_equal(bank["probabilities"],combined):raise ProtocolError("thermal aggregate/context differs")
                count+=1
                if local is None:seen.extend(ids[va].tolist())
                else:covered[local]+=1
        if not np.all(covered==1):raise ProtocolError("thermal inner coverage differs")
    if len(seen)!=len(set(seen)) or set(seen)!=set(ids):raise ProtocolError("thermal outer coverage differs")
    return {"contexts_verified":count,"members_verified":count*4,"outer_unique_rows":len(seen),"outer_accuracy_evaluated":False}
