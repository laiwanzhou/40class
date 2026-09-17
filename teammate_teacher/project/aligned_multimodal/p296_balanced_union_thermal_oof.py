"""Strict OOF Thermal scene model with deterministic balanced extra-row cap."""
from __future__ import annotations
import collections,json
from pathlib import Path
import numpy as np,torch
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import train_fold
from p295_union_thermal_scene_oof_test import cfg,SEEDS
H=Path(__file__).resolve().parent;O=H/"runs/p296_balanced_union_thermal_oof_v1";CACHE=H/"runs/p294_thermal_scene_union_v1/train_features.npz";CONTROL=H/"runs/p295_union_thermal_scene_v1/predictions.npz"
def balanced(ids,users,labels,available):
 out=[];uc=collections.Counter();uu=collections.Counter()
 for i in np.flatnonzero(np.char.startswith(ids,"extra__")&available):
  key=(users[i],int(labels[i]))
  if uc[key]<3 and uu[users[i]]<15:out.append(i);uc[key]+=1;uu[users[i]]+=1
 return np.asarray(out,int)
def main():
 p=load_protocol();z=np.load(CACHE);ids=z["sample_ids"].astype(str);users=z["users"].astype(str);y=z["labels"].astype(int);av=z["available"].astype(bool);x=z["features"].astype(np.float16);pos={q:i for i,q in enumerate(ids)};main=np.asarray([pos[q] for q in p.sample_ids]);extra=balanced(ids,users,y,av);domain=np.zeros(len(ids),int);out=np.full((len(p.labels),40),1/40,np.float32);folds=[];device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 for f in range(3):
  held=p.val_indices(f);hc=main[held];base=main[(p.fold_id!=f)&av[main]];held_users=set(p.users[held].tolist());add=extra[~np.isin(users[extra],list(held_users))];tr=np.concatenate((base,add));members=[train_fold(x,y,tr,hc,domain,s+f*1000,cfg(),device,None,None) for s in SEEDS];log=np.mean(members,0);prob=np.exp(log-log.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);valid=av[hc];out[held[valid]]=prob[valid];folds.append({"fold":f,"train_rows":len(tr),"extra_rows":len(add),"correct":int(np.sum(out[held].argmax(1)==p.labels[held])),"rows":len(held)})
 control=np.load(CONTROL)["control_probability"];report={"stage":"P296_balanced_union_Thermal_OOF","status":"complete","protocol":{"extra_available":int(np.sum(np.char.startswith(ids,"extra__")&av)),"balanced_extra":len(extra),"cap_user_class":3,"cap_user":15,"held_subject_excluded":True,"paired_control":"P295 same architecture/seeds"},"control":{"correct":int(np.sum(control.argmax(1)==p.labels))},"balanced":{"correct":int(np.sum(out.argmax(1)==p.labels)),"accuracy":float(np.mean(out.argmax(1)==p.labels)),"folds":folds}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",sample_ids=p.sample_ids,labels=p.labels,probability=out);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
