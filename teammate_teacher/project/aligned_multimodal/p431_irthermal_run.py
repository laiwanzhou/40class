"""CPU-only original P231 expert rebuild; score-blind pilot then full bank."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import sklearn
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from .p431_irthermal_provider import Provider,load_raw,align_scores,IR1_PATH,IR2_PATH,THERMAL_PATH,SUMMARY_PATHS
from .p90_teacher_common import load_protocol
from .p427_foundation_provider import array_hash
from .p431_irthermal_provider import _file_hash as sha
from .stable_routing_protocol import ArtifactNode,ProtocolError,assert_prediction_provenance,register_experiment

ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/"aligned_multimodal"
PREREG=ROOT/"docs/research/STABLE_093_P431_IRTHERMAL.md"
RECIPE={"alpha":3000.,"solver":"lsqr","tol":1e-5,"max_iter":5000,"power":.75,"temperature":1.,"cpu_threads":1,"features":11520,"class_alignment":"source_score_floor_v1"}


def expected_context(ids,y,users,tr,va,context,provenance):
    return {"context":context,"source_ids":ids[tr].tolist(),"source_users":users[tr].tolist(),
        "target_ids":ids[va].tolist(),"target_users":users[va].tolist(),"source_label_sha256":array_hash(y[tr]),
        "source_class_counts":np.bincount(y[tr],minlength=40).tolist(),"provenance":provenance}


def verify_context(folder,expected,target_x,target_mask):
    folder=Path(folder);r=json.loads((folder/"receipt.json").read_text())
    source_classes=np.flatnonzero(np.asarray(expected["source_class_counts"])>0)
    if (any(r[k]!=v for k,v in expected.items()) or r["recipe"]!=RECIPE or r["target_labels_received"] or r["calibration"] or r["selection"]
        or r["classes"]!=source_classes.tolist() or r["missing_classes"]!=sorted(set(range(40))-set(source_classes))):
        raise ProtocolError("P431 source context/recipe mismatch")
    if sha(folder/"model.npz")!=r["model_sha256"]:raise ProtocolError("P431 model hash mismatch")
    node=r["context"]+".ridge";source=frozenset(r["source_users"])
    nodes={k:ArtifactNode(k,tuple(v["parents"]),frozenset(v["supervised_train_subjects"]),v["provenance"],v["has_task_labels"],v.get("oof_labels_only",False)) for k,v in r["artifact_dag"].items()}
    wanted={"raw.external":ArtifactNode("raw.external",provenance="frozen_external"),node:ArtifactNode(node,("raw.external",),source,"supervised",True)}
    if nodes!=wanted or r["prediction_nodes"]!=[node] or any(k!=v["node_id"] for k,v in r["artifact_dag"].items()):raise ProtocolError("P431 DAG differs")
    for user in set(r["target_users"]):assert_prediction_provenance(user,[node],nodes)
    with np.load(folder/"model.npz",allow_pickle=False) as m:
        ncoef=1 if len(source_classes)==2 else len(source_classes)
        shapes={"mean":(11520,),"scale":(11520,),"var":(11520,),"coef":(ncoef,11520),"intercept":(ncoef,),"classes":(len(source_classes),),"n_samples_seen":(),"logits":(len(target_x),40)}
        if set(m.files)!=set(shapes) or any(m[k].shape!=v or not np.isfinite(m[k]).all() for k,v in shapes.items()):raise ProtocolError("P431 model schema invalid")
        if (np.any(m["scale"]<=0) or np.any(m["var"]<0) or m["n_samples_seen"].item()!=len(r["source_ids"])
            or not np.array_equal(m["classes"],source_classes)):raise ProtocolError("P431 scaler/classes invalid")
        with threadpool_limits(limits=1):
            scaler=StandardScaler();scaler.mean_=m["mean"];scaler.scale_=m["scale"];scaler.var_=m["var"]
            scaler.n_samples_seen_=m["n_samples_seen"];scaler.n_features_in_=11520
            x=scaler.transform(np.asarray(target_x,np.float32))
            replay=align_scores(x@m["coef"].T+m["intercept"],m["classes"])
        if not np.array_equal(replay,m["logits"]):raise ProtocolError("P431 saved model does not replay logits")
        prob=np.exp(replay-replay.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    with np.load(folder/"bank.npz",allow_pickle=False) as z:
        if (set(z.files)!={"sample_ids","users","expert_names","thermal_available","probabilities"}
            or not np.array_equal(z["sample_ids"],r["target_ids"]) or not np.array_equal(z["users"],r["target_users"])
            or z["expert_names"].tolist()!=["p231_ir_thermal"] or not np.array_equal(z["thermal_available"],target_mask)
            or not np.array_equal(z["probabilities"],prob.astype(np.float32)[:,None,:])):raise ProtocolError("P431 bank replay/IDs differ")
    return r


def main():
    ap=argparse.ArgumentParser();m=ap.add_mutually_exclusive_group(required=True);m.add_argument("--pilot",action="store_true");m.add_argument("--run",action="store_true")
    ap.add_argument("--out-dir",required=True);ap.add_argument("--pilot-dir");a=ap.parse_args();out=Path(a.out_dir)
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();deadline=started+(300 if a.pilot else 1200)
    sources=[Path(__file__),HERE/"p431_irthermal_provider.py",HERE/"p431_watchdog.py",PREREG,
        *[HERE/n for n in ("p231_depth_thermal_ir_oof.py","train_p46_videomae_head.py","p90_videomaev2_distilled_teacher.py",
            "p90_internvideo2_l_teacher.py","p91_videomaev2_modality_teacher.py","p90_teacher_common.py","p427_foundation_provider.py","stable_routing_protocol.py")]]
    inputs=[IR1_PATH,IR2_PATH,THERMAL_PATH,*SUMMARY_PATHS.values(),HERE/"data/manifest.csv",*[HERE/f"data/subject_folds/fold_{k}.csv" for k in range(3)]]
    spec={"mode":"pilot" if a.pilot else "full","source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in sources},
        "input_sha256":{str(p.relative_to(ROOT)):sha(p) for p in inputs},"recipe":RECIPE,
        "runtime":{"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__}}
    p=load_protocol();raw,mask,provenance=load_raw(p.sample_ids);keep=~np.isin(p.users,["user1","user2","user21"])
    ids,y,users,folds=(v[keep] for v in (p.sample_ids,p.labels,p.users,p.fold_id));x=raw[keep];mask=mask[keep]
    provider=Provider(x,ids,users,mask,provenance)
    if a.run:
        if not a.pilot_dir:raise ProtocolError("full requires pilot")
        prior=Path(a.pilot_dir);ps=json.loads((prior/"summary.json").read_text());pr=json.loads((prior/"registry.json").read_text())
        register_experiment(prior/"registry.json",pr["spec"])
        process=json.loads(Path(str(prior)+".process.json").read_text())
        if (ps["mode"]!="pilot" or pr["spec"]["mode"]!="pilot" or ps["contexts"]!=1 or not 0<=ps["seconds"]<=300
            or ps["outer_accuracy_evaluated"] or ps["test_rows_loaded"]!=0 or ps["target_achieved"]
            or process["status"]!="complete" or process["exit_code"]!=0 or not 0<=process["seconds"]<=330):raise ProtocolError("bad P431 pilot")
        for key in ("source_sha256","input_sha256","recipe","runtime"):
            if spec[key]!=pr["spec"][key]:raise ProtocolError("pilot/core changed")
        for rel,h in ps["artifacts"].items():
            if sha(prior/rel)!=h:raise ProtocolError("pilot artifact changed")
        for path in sources:
            if sha(prior/"source_snapshot"/path.name)!=spec["source_sha256"][str(path.relative_to(ROOT))]:raise ProtocolError("pilot source snapshot changed")
        tr=np.flatnonzero(folds!=0);va=np.flatnonzero(folds==0)
        verify_context(prior/"fold0/outer",expected_context(ids,y,users,tr,va,"fold0.outer",provenance),x[va],mask[va])
    out.mkdir(parents=True,exist_ok=False);register_experiment(out/"registry.json",spec)
    snap=out/"source_snapshot";snap.mkdir()
    for path in sources:(snap/path.name).write_bytes(path.read_bytes())
    count=0;outer_ids=[];plans=[];coverage={k:np.zeros(len(ids),int) for k in range(3)}
    print("P431 fixed original IR+thermal Ridge; CPU1, no held scoring.",flush=True)
    for fold in ([0] if a.pilot else range(3)):
        tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold);requests=[("outer",tr,va)]
        if a.run:requests.extend((f"inner{k}",tr[t],tr[v]) for k,(t,v) in enumerate(GroupKFold(3).split(tr,groups=users[tr])))
        for name,source,target in requests:
            if time.monotonic()>=deadline:raise TimeoutError("P431 budget")
            context=f"fold{fold}.{name}";folder=out/f"fold{fold}"/name;folder.mkdir(parents=True)
            plans.append((folder,source.copy(),target.copy(),context))
            probability,r,payload=provider.fit_predict(source,y[source],target,context=context)
            np.savez_compressed(folder/"model.npz",**payload);r["model_sha256"]=sha(folder/"model.npz")
            (folder/"receipt.json").write_text(json.dumps(r,indent=2),encoding="utf-8")
            np.savez_compressed(folder/"bank.npz",sample_ids=ids[target],users=users[target],expert_names=np.array(["p231_ir_thermal"]),thermal_available=mask[target],probabilities=probability)
            verify_context(folder,expected_context(ids,y,users,source,target,context,provenance),x[target],mask[target])
            if a.run and fold==0 and name=="outer":
                for filename in ("model.npz","bank.npz"):
                    with np.load(folder/filename,allow_pickle=False) as z,np.load(Path(a.pilot_dir)/"fold0/outer"/filename,allow_pickle=False) as old:
                        if set(z.files)!=set(old.files) or any(not np.array_equal(z[k],old[k]) for k in z.files):raise ProtocolError("P431 first context not bit-exact")
            count+=1
            if name=="outer":outer_ids.extend(ids[target])
            else:coverage[fold][target]+=1
            print(json.dumps({"context":context,"seconds":time.monotonic()-started,"verified":True}),flush=True)
    if a.run and (count!=12 or len(outer_ids)!=2470 or len(set(outer_ids))!=2470):raise ProtocolError("P431 coverage")
    if a.run:
        for fold in range(3):
            if not np.array_equal(coverage[fold],(folds!=fold).astype(int)):raise ProtocolError("P431 inner coverage")
    for folder,source,target,context in plans:
        verify_context(folder,expected_context(ids,y,users,source,target,context,provenance),x[target],mask[target])
    for path in sources+inputs:
        key=str(path.relative_to(ROOT));expected=spec["source_sha256"].get(key,spec["input_sha256"].get(key))
        if sha(path)!=expected:raise ProtocolError("P431 inputs/source changed")
    report={"mode":spec["mode"],"contexts":count,"seconds":time.monotonic()-started,"outer_accuracy_evaluated":False,"test_rows_loaded":0,"target_achieved":False,"complete_p315":False,
        "artifacts":{str(path.relative_to(out)):sha(path) for path in out.rglob("*") if path.is_file() and "source_snapshot" not in path.parts}}
    if time.monotonic()>=deadline:raise TimeoutError("P431 final budget")
    (out/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":main()
