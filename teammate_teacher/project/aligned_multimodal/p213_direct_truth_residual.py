"""Direct truth-trained deployable residual over P205 with strict outer cross-fit."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p190_inductive_p150_meta_distillation import S,cat,P89T
from p191_source_truth_calibrated_p150_distillation import model,route_score,choose
from p204_p203_calibrated_p150_residual import enrich,testx

H=Path(__file__).resolve().parent;O=H/"runs/p213_direct_truth_residual_v1"
P205=H/"runs/p205_fixed_p203_p150_residual_v1/oof_predictions.npz"
P205T=H/"runs/p205_fixed_p203_p150_residual_v1/submission_p205_fixed_residual.csv"
CS=(.0003,.001,.003,.01,.03,.1,.3)
def cp(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def data_load():
 data=enrich();p=np.load(P205);mp={v:int(x) for v,x in zip(p["sample_ids"].astype(str),p["prediction"])}
 for n in S:
  data[n]["base"]=np.asarray([mp[v] for v in data[n]["ids"]]);data[n]["target"]=data[n]["labels"]
 return data
def tx_load():
 ids,x,_=testx();return ids,x,cp(P205T)
def inner(data,src,c):
 out={}
 for tr,te in ((src[0],src[1]),(src[1],src[0])):
  m=model(c);m.fit(data[tr]["x"],data[tr]["labels"]);p=m.predict_proba(data[te]["x"]);f=np.zeros((len(p),40));f[:,m.classes_.astype(int)]=p;out[te]=f
 return np.concatenate([out[n] for n in src])
def main():
 data=data_load();report={"stage":"P213_direct_truth_residual","status":"complete","protocol":{"base":"P205","target":"source ground truth","source_crossfit_threshold":True,"held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];labels=cat([data[n] for n in src],"labels");base=cat([data[n] for n in src],"base");users=cat([data[n] for n in src],"users");coh=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in src]);cands=[]
  for c in CS:
   prob=inner(data,src,c);pred=prob.argmax(1);gain=(pred==labels).astype(int)-(base==labels).astype(int)
   for kind in ("confidence","gap_base","margin"):
    sel=choose(route_score(prob,base,pred,kind),gain,pred!=base,users,coh);cands.append((sel["minimum_cohort_gain"]>=0,sel["minimum_user_gain"]>=0,sel["net"],-sel["harm"],-sel["changed"],c,kind,sel))
  best=max(cands);c,kind,sel=best[5],best[6],best[7];m=model(c);m.fit(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"labels"));p=m.predict_proba(data[held]["x"]);f=np.zeros((len(p),40));f[:,m.classes_.astype(int)]=p;pred=f.argmax(1);mask=(pred!=data[held]["base"])&(route_score(f,data[held]["base"],pred,kind)>=sel["threshold"]);out=data[held]["base"].copy();out[mask]=pred[mask];outs[held]=out;bc=int(np.sum(data[held]["base"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"C":c,"score":kind,"source_selection":sel,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(mask.sum())}}
  print(json.dumps({"held":held,**report["cohorts"][held]["held"]}),flush=True)
 labels=cat([data[n] for n in S],"labels");base=cat([data[n] for n in S],"base");out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(base==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p205":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S],"gap_to_0.91":int(np.ceil(.91*len(labels))-cor)}
 candidates=[];users=cat([data[n] for n in S],"users");coh=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in S])
 for c in CS:
  probs=[]
  for held in S:
   src=[n for n in S if n!=held];m=model(c);m.fit(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"labels"));p=m.predict_proba(data[held]["x"]);f=np.zeros((len(p),40));f[:,m.classes_.astype(int)]=p;probs.append(f)
  prob=np.concatenate(probs);pred=prob.argmax(1);gain=(pred==labels).astype(int)-(base==labels).astype(int)
  for kind in ("confidence","gap_base","margin"):
   sel=choose(route_score(prob,base,pred,kind),gain,pred!=base,users,coh);candidates.append((sel["minimum_cohort_gain"]>=0,sel["minimum_user_gain"]>=0,sel["net"],-sel["harm"],-sel["changed"],c,kind,sel))
 best=max(candidates);c,kind,sel=best[5],best[6],best[7];ids,tx,tbase=tx_load();m=model(c);m.fit(cat([data[n] for n in S],"x"),labels);p=m.predict_proba(tx);f=np.zeros((len(p),40));f[:,m.classes_.astype(int)]=p;pred=f.argmax(1);mask=(pred!=tbase)&(route_score(f,tbase,pred,kind)>=sel["threshold"]);tout=tbase.copy();tout[mask]=pred[mask];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p213_direct_truth_residual.csv";io.write_submission(sub,io.read_rows(P89T),tout);report["test"]={"C":c,"score":kind,"oof_selection":sel,"changes_vs_p205":int(mask.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=np.concatenate([data[n]["ids"] for n in S]),labels=labels,base_prediction=base,prediction=out,test_sample_ids=ids,test_base_prediction=tbase,test_prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps({"aggregate":report["aggregate"],"test":report["test"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
