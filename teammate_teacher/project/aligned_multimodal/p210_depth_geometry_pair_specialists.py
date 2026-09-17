"""Source-user-safe physical pair specialists from P207 Depth geometry posteriors."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from p90_crossuser_visual_router import load_splits
H=Path(__file__).resolve().parent;O=H/"runs/p210_depth_geometry_pair_specialists_v1";D=H/"runs/p207_depth_geometry_full_test_v1/predictions.npz";P205=H/"runs/p205_fixed_p203_p150_residual_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0");PAIRS=((24,26),(23,27),(0,4),(21,22),(8,10),(6,37))
def feat(d,ids):
 x=np.concatenate((d["geometry_probability"],d["depth_surface_probability"],d["depth_geometry_probability"]),1);lookup={v:i for i,v in enumerate(d["sample_ids"].astype(str))};return x[np.asarray([lookup[v] for v in ids])]
def clf():return make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",solver="liblinear",max_iter=1000))
def loso(x,labels,users,pair):
 pred=np.full(len(labels),-1);conf=np.zeros(len(labels))
 for u in sorted(set(users.tolist())):
  te=users==u;train=(users!=u)&np.isin(labels,pair)
  if len(set(labels[train]))<2:continue
  m=clf();m.fit(x[train],labels[train]);p=m.predict_proba(x[te]);idx=np.argmax(p,1);pred[te]=m[-1].classes_[idx];conf[te]=np.abs(p[:,1]-p[:,0])
 return pred,conf
def threshold(base,pred,conf,labels,users,pair):
 eligible=np.isin(base,pair)&(pred>=0)&(pred!=base);gain=(pred==labels).astype(int)-(base==labels).astype(int);best=None
 for t in np.unique(np.concatenate(([np.inf,-np.inf],np.linspace(0,1,101),np.quantile(conf[eligible],[.25,.5,.75]) if eligible.any() else [np.inf]))):
  q=eligible&(conf>=t);per={u:int(gain[q&(users==u)].sum()) for u in sorted(set(users.tolist()))};r=int(np.sum(q&(gain>0)));h=int(np.sum(q&(gain<0)));row={"threshold":float(t),"selected":int(q.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(per.values()),"per_user":per};key=(row["minimum_user_gain"]>=0,row["net"],r,-h,-row["selected"])
  if best is None or key>best[0]:best=(key,row)
 return best[1]
def main():
 d=np.load(D);p=np.load(P205);splits=load_splits();mp={v:(int(y),int(b)) for v,y,b in zip(p["sample_ids"].astype(str),p["labels"],p["prediction"])};data={n:{"ids":splits[n].sample_ids.astype(str),"labels":np.asarray([mp[v][0] for v in splits[n].sample_ids.astype(str)]),"base":np.asarray([mp[v][1] for v in splits[n].sample_ids.astype(str)]),"users":splits[n].users.astype(str),"x":feat(d,splits[n].sample_ids.astype(str))} for n in S};report={"stage":"P210_depth_geometry_pair_specialists","status":"complete","protocol":{"pairs":[list(x) for x in PAIRS],"feature":"strict OOF P207 posteriors","source_user_LOSO_threshold":True,"held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];x=np.concatenate([data[n]["x"] for n in src]);labels=np.concatenate([data[n]["labels"] for n in src]);base=np.concatenate([data[n]["base"] for n in src]);users=np.concatenate([data[n]["users"] for n in src]);out=data[held]["base"].copy();aud=[]
  for pair in PAIRS:
   pred,conf=loso(x,labels,users,pair);sel=threshold(base,pred,conf,labels,users,pair);train=np.isin(labels,pair)
   if len(set(labels[train]))<2:continue
   m=clf();m.fit(x[train],labels[train]);pp=m.predict_proba(data[held]["x"]);idx=pp.argmax(1);hp=m[-1].classes_[idx];hc=np.abs(pp[:,1]-pp[:,0]);q=np.isin(data[held]["base"],pair)&(hp!=data[held]["base"])&(hc>=sel["threshold"]);eligible=sel["net"]>0 and sel["minimum_user_gain"]>=0
   if eligible:out[q]=hp[q]
   aud.append({"pair":list(pair),"source":sel,"eligible":eligible,"held_changes":int(q.sum()) if eligible else 0})
  outs[held]=out;bc=int(np.sum(data[held]["base"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"pairs":aud,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(np.sum(out!=data[held]["base"]))}}
 labels=np.concatenate([data[n]["labels"] for n in S]);base=np.concatenate([data[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(base==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p205":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
