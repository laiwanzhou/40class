"""Source-truth calibrated P150 residual on top of P203."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p117_transductive_multicandidate_router import one_hot
from p190_inductive_p150_meta_distillation import S,cat,load_oof,test_features,P89T
from p191_source_truth_calibrated_p150_distillation import CS,model,route_score,choose,inner_prob
H=Path(__file__).resolve().parent;O=H/"runs/p204_p203_calibrated_p150_residual_v1";P203=H/"runs/p203_current_champion_v1/oof_predictions.npz";P203T=H/"runs/p203_current_champion_v1/submission_p203_current_champion.csv"
def cp(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def enrich():
 data=load_oof();p=np.load(P203);mp={v:int(x) for v,x in zip(p["sample_ids"].astype(str),p["prediction"])}
 for n in S:
  old=data[n]["base"].copy();new=np.asarray([mp[v] for v in data[n]["ids"]]);data[n]["x"]=np.concatenate((data[n]["x"],one_hot(new),np.column_stack((new==old,new!=old)).astype(np.float32)),axis=1);data[n]["base"]=new
 return data
def testx():
 ids,x,old=test_features();new=cp(P203T);return ids,np.concatenate((x,one_hot(new),np.column_stack((new==old,new!=old)).astype(np.float32)),axis=1),new
def main():
 data=enrich();report={"stage":"P204_P203_calibrated_P150_residual","status":"complete","protocol":{"base":"P203","meta_target":"P150 OOF","route_selection":"source cross-prediction truth only","held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];labels=cat([data[n] for n in src],"labels");base=cat([data[n] for n in src],"base");users=cat([data[n] for n in src],"users");coh=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in src]);cand=[]
  for c in CS:
   prob=inner_prob(data,src,c);pred=prob.argmax(1);gain=(pred==labels).astype(int)-(base==labels).astype(int)
   for kind in ("confidence","gap_base","margin"):
    sel=choose(route_score(prob,base,pred,kind),gain,pred!=base,users,coh);cand.append((sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],-sel["harm"],-sel["changed"],c,kind,sel))
  best=max(cand);c,kind,sel=best[5],best[6],best[7];m=model(c);m.fit(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"target"));p=m.predict_proba(data[held]["x"]);full=np.zeros((len(p),40));full[:,m.classes_.astype(int)]=p;pred=full.argmax(1);mask=(pred!=data[held]["base"])&(route_score(full,data[held]["base"],pred,kind)>=sel["threshold"]);out=data[held]["base"].copy();out[mask]=pred[mask];outs[held]=out;bc=int(np.sum(data[held]["base"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"C":c,"score":kind,"selection":sel,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(mask.sum())}}
 labels=cat([data[n] for n in S],"labels");base=cat([data[n] for n in S],"base");out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(base==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p203":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]}
 # global source-safe OOF selection
 candidates=[]
 for c in CS:
  probs=[]
  for held in S:
   src=[n for n in S if n!=held];m=model(c);m.fit(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"target"));p=m.predict_proba(data[held]["x"]);f=np.zeros((len(p),40));f[:,m.classes_.astype(int)]=p;probs.append(f)
  prob=np.concatenate(probs);pred=prob.argmax(1);gain=(pred==labels).astype(int)-(base==labels).astype(int);users=cat([data[n] for n in S],"users");coh=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in S])
  for kind in ("confidence","gap_base","margin"):
   sel=choose(route_score(prob,base,pred,kind),gain,pred!=base,users,coh);candidates.append((sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],-sel["harm"],-sel["changed"],c,kind,sel))
 best=max(candidates);c,kind,sel=best[5],best[6],best[7];ids,tx,tbase=testx();m=model(c);m.fit(cat([data[n] for n in S],"x"),cat([data[n] for n in S],"target"));p=m.predict_proba(tx);f=np.zeros((len(p),40));f[:,m.classes_.astype(int)]=p;pred=f.argmax(1);mask=(pred!=tbase)&(route_score(f,tbase,pred,kind)>=sel["threshold"]);tout=tbase.copy();tout[mask]=pred[mask];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p204_p203_residual.csv";io.write_submission(sub,io.read_rows(P89T),tout);report["test"]={"C":c,"score":kind,"oof_selection":sel,"changes_vs_p203":int(mask.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=ids,base_prediction=tbase,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
