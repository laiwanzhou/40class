"""Reloaded actual P142/P144 pilot verification; never accuracy evaluation."""
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .p430_token_provider import EXPERTS,TOKEN_INDICES,BASE_SEEDS
from .p430_token_training import FIXED_RECIPE,TokenHead
from .p419_vjepa_repeat_group_bridge import sha
from .stable_routing_protocol import ArtifactNode,ProtocolError,assert_prediction_provenance


@lru_cache(maxsize=2)
def state_schema(tokens):
    if tokens not in (4,24):raise ProtocolError("invalid token count")
    # Shape-only construction; no model allocation on GPU or random weight fit.
    with torch.device("meta"):
        model=TokenHead(input_dim=1024,hidden_dim=192,heads=6,layers=2,dropout=.2,view_dropout=.15,num_tokens=tokens)
    return {k:(tuple(v.shape),v.dtype) for k,v in model.state_dict().items()}


def verify_diagnostics(d,rows,seed,tokens):
    planned=35*math.ceil(rows/128)
    values=np.asarray([d["planned_steps"],d["optimizer_steps"],d["amp_skips"]],dtype=float)
    if not np.isfinite(values).all() or np.any(values!=np.floor(values)):
        raise ProtocolError("noninteger token update counters")
    total,good,bad=values
    lr=3e-4*.05+(3e-4-3e-4*.05)*(1+np.cos(np.pi*good/planned))/2
    if (total!=planned or good+bad!=planned or good<.9*planned or bad<0
        or d["epochs"]!=35 or d["seed"]!=seed or d["token_count"]!=tokens or d["recipe"]!=FIXED_RECIPE
        or not np.isfinite(d["final_lr"]) or not np.isclose(d["final_lr"],lr,rtol=1e-8,atol=1e-12)):
        raise ProtocolError("token training recipe/progress mismatch")
    telemetry=d["telemetry"]
    if [x["epoch"] for x in telemetry]!=[1,10,20,30,35]:raise ProtocolError("token epoch telemetry incomplete")
    prior=0
    for x in telemetry:
        counts=np.asarray([x["optimizer_steps"],x["amp_skips"]],float)
        if (not np.isfinite(counts).all() or np.any(counts!=np.floor(counts)) or np.any(counts<0)
            or counts.sum()!=x["epoch"]*math.ceil(rows/128) or counts[0]<prior or not np.isfinite(x["train_loss"])):
            raise ProtocolError("invalid token epoch telemetry")
        prior=counts[0]
    if telemetry[-1]["optimizer_steps"]!=good or telemetry[-1]["amp_skips"]!=bad:
        raise ProtocolError("token final counter mismatch")


def verify_context(folder,expected):
    folder=Path(folder);r=json.loads((folder/"provenance.json").read_text(encoding="utf-8"))
    for key in ("context","outer_fold","source_ids","source_users","target_ids","target_users",
                "source_label_sha256","source_class_counts","cache_provenance"):
        if r[key]!=expected[key]:raise ProtocolError(f"token authoritative context differs: {key}")
    if (r["expert_names"]!=list(EXPERTS) or r["aggregation"]!="softmax(mean_three_logits)"
        or r["target_labels_received"] or r["checkpoint_selection"] or r["injected_test_fitter"]
        or not r["member_callback_used"]):raise ProtocolError("token recipe/fit mode invalid")
    source,target=r["source_ids"],r["target_ids"];su=frozenset(r["source_users"]);tu=set(r["target_users"])
    if (len(set(source))!=len(source) or len(set(target))!=len(target) or set(source)&set(target)
        or su&tu or (su|tu)&{"user1","user2","user21"}):raise ProtocolError("token identity/subject exclusion failed")
    cache=r["cache_provenance"]
    if (cache.get("type")!="frozen_external" or cache.get("labels_loaded") is not False
        or cache.get("model_repo")!="facebook/vjepa2-vitl-fpc16-256-ssv2" or len(cache.get("input_sha256",{}))!=5):
        raise ProtocolError("token cache ancestry missing")
    for path,digest in cache["input_sha256"].items():
        if sha(path)!=digest:raise ProtocolError("token cache input changed")
    context=r["context"];external=context+".external_vjepa"
    nodes={external:ArtifactNode(external,provenance="frozen_external")};roots=[];fits=r["fits"]
    if len(fits)!=6:raise ProtocolError("token member count wrong")
    with np.load(folder/"bank.npz",allow_pickle=False) as z:
        p=z["probabilities"];members=z["member_logits"]
        if (p.shape!=(len(target),2,40) or members.shape!=(2,3,len(target),40)
            or members.dtype!=np.float32 or not np.isfinite(p).all() or not np.isfinite(members).all()
            or not np.array_equal(z["sample_ids"],target) or not np.array_equal(z["users"],r["target_users"])
            or z["expert_names"].astype(str).tolist()!=list(EXPERTS)):
            raise ProtocolError("token bank shape/identity invalid")
        for i,(expert,columns,bases) in enumerate(zip(EXPERTS,TOKEN_INDICES,BASE_SEEDS)):
            parents=[]
            for j,base in enumerate(bases):
                seed=int(base+1000*r["outer_fold"]);fit=fits[i*3+j];node=context+f".{expert}.seed{seed}"
                if (fit["node"]!=node or fit["expert"]!=expert or fit["seed"]!=seed or fit["token_indices"]!=list(columns)
                    or fit["source_ids"]!=source or fit["target_ids"]!=target):raise ProtocolError("token member recipe/IDs wrong")
                verify_diagnostics(fit["diagnostics"],len(source),seed,len(columns))
                member=folder/f"members/{expert}/seed{seed}";mr=json.loads((member/"receipt.json").read_text(encoding="utf-8"))
                if any(mr[k]!=v for k,v in fit.items()):raise ProtocolError("member/context receipts disagree")
                if sha(member/"checkpoint.pt")!=mr["checkpoint_sha256"] or sha(member/"outputs.npz")!=mr["logits_sha256"]:
                    raise ProtocolError("token member artifact hash mismatch")
                state=torch.load(member/"checkpoint.pt",map_location="cpu",weights_only=True)
                if not state or not all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in state.values()):
                    raise ProtocolError("token member state nonfinite or empty")
                schema=state_schema(len(columns))
                if set(state)!=set(schema) or any((tuple(state[k].shape),state[k].dtype)!=v for k,v in schema.items()):
                    raise ProtocolError("checkpoint is not the exact TokenHead state schema")
                with np.load(member/"outputs.npz",allow_pickle=False) as mz:
                    if not np.array_equal(mz["logits"],members[i,j]):raise ProtocolError("member/aggregate logits disagree")
                nodes[node]=ArtifactNode(node,(external,),su,"supervised",True);parents.append(node)
            root=context+f".{expert}.mean_logits";nodes[root]=ArtifactNode(root,tuple(parents));roots.append(root)
            mean=members[i].mean(0);soft=np.exp(mean-mean.max(1,keepdims=True));soft/=soft.sum(1,keepdims=True)
            if not np.array_equal(soft,p[:,i]):raise ProtocolError("not original float32 mean-logit aggregation")
    actual={k:ArtifactNode(k,tuple(v["parents"]),frozenset(v["supervised_train_subjects"]),v["provenance"],
              v["has_task_labels"],v.get("oof_labels_only",False)) for k,v in r["artifact_dag"].items()}
    if actual!=nodes or r["prediction_nodes"]!=roots or any(k!=v["node_id"] for k,v in r["artifact_dag"].items()):
        raise ProtocolError("token provenance topology differs")
    for u in tu:assert_prediction_provenance(u,roots,actual)
    return r
