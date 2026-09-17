"""Reload and verify original thermal context/member artifacts without scoring."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from .p429_thermal_provider import select_epoch
from .p427_foundation_provider import array_hash
from .p419_vjepa_repeat_group_bridge import sha
from .stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance


def verify_diagnostics(d, epochs):
    good=np.asarray(d["successful_amp_steps"]); bad=np.asarray(d["skipped_steps"])
    lr=.0007*(1+np.cos(np.pi*np.arange(1,epochs+1)/15))/2
    if (d["epochs_completed"]!=epochs or np.asarray(d["losses"]).shape!=(epochs,)
        or not np.isfinite(d["losses"]).all() or good.shape!=(epochs,) or bad.shape!=(epochs,)
        or not np.isfinite(good).all() or not np.isfinite(bad).all()
        or np.any(good!=np.floor(good)) or np.any(bad!=np.floor(bad))
        or np.any(good<=0) or np.any(bad<0) or np.any(good/(good+bad)<.9)
        or np.asarray(d["learning_rates"]).shape!=(epochs,)
        or not np.allclose(d["learning_rates"],lr,rtol=1e-8,atol=1e-12)
        or d["batch_size"]!=32 or not d["drop_last"] or d["num_workers"]!=0):
        raise ProtocolError("invalid thermal training diagnostics")


def checkpoint(path, digest):
    if sha(path)!=digest:raise ProtocolError("thermal checkpoint hash mismatch")
    state=torch.load(path,map_location="cpu",weights_only=True)
    if not state or not all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in state.values()):
        raise ProtocolError("nonfinite or empty thermal checkpoint")
    return state


def verify_member(folder, train_ids, prediction_ids, expected_logits, diagnostics, epochs):
    r=json.loads((folder/"receipt.json").read_text(encoding="utf-8"))
    if r["train_ids"]!=list(train_ids) or r["prediction_ids"]!=list(prediction_ids) or r["diagnostics"]!=diagnostics:
        raise ProtocolError("thermal member receipt differs")
    verify_diagnostics(r["diagnostics"],epochs)
    state=checkpoint(folder/"checkpoint.pt",r["checkpoint_sha256"])
    if sha(folder/"outputs.npz")!=r["logits_sha256"]:raise ProtocolError("member output hash mismatch")
    with np.load(folder/"outputs.npz",allow_pickle=False) as z:
        if not np.array_equal(z["logits"],expected_logits):raise ProtocolError("member/context logits disagree")
    return state


def verify_context(folder, expected):
    folder=Path(folder);r=json.loads((folder/"provenance.json").read_text(encoding="utf-8"))
    for key in ("source_ids","source_users","target_ids","target_users","source_label_sha256"):
        if r[key]!=expected[key]:raise ProtocolError(f"context does not match authoritative population: {key}")
    source=np.asarray(r["source_ids"]);users=np.asarray(r["source_users"]);target=np.asarray(r["target_ids"])
    present=np.asarray(expected["thermal_present"])
    if (present.dtype.kind!="b" or present.shape!=target.shape or len(set(source))!=len(source)
        or len(set(target))!=len(target) or set(source)&set(target) or set(users)&set(r["target_users"])
        or (set(users)|set(r["target_users"]))&{"user1","user2","user21"}):
        raise ProtocolError("context identity/exclusion invalid")
    if r["target_labels_received"] or r["historical_task_checkpoint_loaded"]:
        raise ProtocolError("forbidden target labels or historical checkpoint")
    if r["missing_logits"]!="zeros -> uniform40":
        raise ProtocolError("thermal missingness semantics changed")
    nodes={k:ArtifactNode(k,tuple(v["parents"]),frozenset(v["supervised_train_subjects"]),v["provenance"],
                          v["has_task_labels"],v.get("oof_labels_only",False)) for k,v in r["artifact_dag"].items()}
    context=r["context"];raw=context+".raw";choice=context+".epoch_choice";refit=context+".refit"
    expected_nodes={raw:ArtifactNode(raw,provenance="raw_input")}
    selection_nodes=[]
    for k,(tr,_) in enumerate(GroupKFold(3).split(source,groups=users)):
        node=context+f".selection{k}";selection_nodes.append(node)
        expected_nodes[node]=ArtifactNode(node,(raw,),frozenset(users[tr]),"supervised",True)
    expected_nodes[choice]=ArtifactNode(choice,tuple(selection_nodes),frozenset(users),"supervised",True)
    expected_nodes[refit]=ArtifactNode(refit,(raw,choice),frozenset(users),"supervised",True)
    if (nodes!=expected_nodes or r["prediction_nodes"]!=[refit]
        or any(v["node_id"]!=k for k,v in r["artifact_dag"].items())):
        raise ProtocolError("thermal provenance topology differs from registered recipe")
    for u in set(r["target_users"]):assert_prediction_provenance(u,r["prediction_nodes"],nodes)
    final_state=checkpoint(folder/"checkpoint.pt",r["checkpoint_sha256"])
    with np.load(folder/"bank.npz",allow_pickle=False) as z:
        p=z["probabilities"];logits=z["logits"];oof=z["selection_logits"];labels=z["selection_labels"]
        if (p.shape!=(len(target),1,40) or logits.shape!=(len(target),40) or not np.isfinite(p).all()
            or not np.isfinite(logits).all() or (p<0).any() or not np.allclose(p.sum(2),1)
            or z["expert_names"].astype(str).tolist()!=["p12_thermal"]
            or not np.array_equal(z["sample_ids"],target) or not np.array_equal(z["users"],r["target_users"])
            or not np.array_equal(z["selection_ids"],source) or not np.array_equal(z["thermal_present"],present)
            or np.any(logits[~present]!=0) or not np.allclose(p[~present],1/40)):
            raise ProtocolError("thermal bank shape/identity/missingness invalid")
        if array_hash(labels)!=r["source_label_sha256"] or array_hash(oof)!=r["selection_logit_sha256"]:
            raise ProtocolError("selection arrays changed")
        epoch,correct=select_epoch(oof,labels)
        if epoch!=r["selected_epoch"] or correct.tolist()!=r["source_epoch_correct"]:
            raise ProtocolError("thermal epoch choice mismatch")
        softmax=np.exp(logits-logits.max(1,keepdims=True));softmax/=softmax.sum(1,keepdims=True)
        if not np.allclose(softmax,p[:,0],atol=1e-7):raise ProtocolError("thermal softmax mismatch")
        if len(r["selection_fits"])!=3:raise ProtocolError("wrong selection fit count")
        for k,(fit,(tr,va)) in enumerate(zip(r["selection_fits"],GroupKFold(3).split(source,groups=users))):
            if (fit["train_ids"]!=source[tr].tolist() or fit["prediction_ids"]!=source[va].tolist()
                or fit["train_users"]!=users[tr].tolist() or fit["prediction_users"]!=users[va].tolist()
                or nodes[fit["node"]].supervised_train_subjects!=frozenset(users[tr])):
                raise ProtocolError("thermal selection partition mismatch")
            for u in set(users[va]):assert_prediction_provenance(u,[fit["node"]],nodes)
            verify_member(folder/f"members/selection{k}",source[tr],source[va],oof[:,va],fit["diagnostics"],15)
        state=verify_member(folder/"members/refit",source,target[present],logits[None,present],r["refit_diagnostics"],epoch)
        if final_state.keys()!=state.keys() or not all(torch.equal(state[k],final_state[k]) for k in state):
            raise ProtocolError("final/member checkpoint differs")
    return r
