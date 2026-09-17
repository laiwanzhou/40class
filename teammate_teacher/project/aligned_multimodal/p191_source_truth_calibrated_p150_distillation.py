"""Source-truth calibrated routing for inductive P150 meta-distillation."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import p89_build_dual_consensus_submission as io
from p190_inductive_p150_meta_distillation import S,cat,load_oof,test_features,P89T

H=Path(__file__).resolve().parent;O=H/"runs/p191_source_truth_calibrated_p150_v1"
CS=(.003,.01,.03,.1)
def model(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1400,solver="lbfgs"))
def route_score(prob,base,pred,kind):
 rows=np.arange(len(base));
 if kind=="confidence":return prob[rows,pred]
 if kind=="gap_base":return prob[rows,pred]-prob[rows,base]
 q=np.sort(prob,axis=1)[:,-2:];return q[:,1]-q[:,0]
def choose(score,gain,disagree,users,cohorts):
 vals=np.unique(np.concatenate(([-np.inf,np.inf],np.linspace(-.5,1,301),np.quantile(score[disagree],np.linspace(.05,.95,19)) if disagree.any() else [np.inf])));best=None
 for t in vals:
  m=disagree&(score>=t);peru={u:int(gain[m&(users==u)].sum()) for u in sorted(set(users.tolist()))};perc={c:int(gain[m&(cohorts==c)].sum()) for c in sorted(set(cohorts.tolist()))};row={"threshold":float(t),"changed":int(m.sum()),"rescue":int(np.sum(m&(gain>0))),"harm":int(np.sum(m&(gain<0))),"net":int(gain[m].sum()),"minimum_user_gain":min(peru.values()),"minimum_cohort_gain":min(perc.values()),"per_user":peru,"per_cohort":perc}
  key=(row["minimum_user_gain"]>=0,row["minimum_cohort_gain"]>=0,row["net"],-row["harm"],-row["changed"])
  if best is None or key>best[0]:best=(key,row)
 return best[1]
def inner_prob(data,src,c):
 a,b=src;out={}
 for tr,te in ((a,b),(b,a)):
  m=model(c);m.fit(data[tr]["x"],data[tr]["target"]);p=m.predict_proba(data[te]["x"]);full=np.zeros((len(p),40));full[:,m.classes_.astype(int)]=p;out[te]=full
 return np.concatenate([out[n] for n in src])
def main():
 data=load_oof();report={"stage":"P191_source_truth_calibrated_P150_distillation","status":"complete","protocol":{"meta_training_target":"P150 OOF","route_selection":"source cross-prediction ground truth only","held_targets_or_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];labels=cat([data[n] for n in src],"labels");base=cat([data[n] for n in src],"base");users=cat([data[n] for n in src],"users");coh=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in src]);cands=[]
  for c in CS:
   prob=inner_prob(data,src,c);pred=prob.argmax(1);gain=(pred==labels).astype(int)-(base==labels).astype(int)
   for kind in ("confidence","gap_base","margin"):
    sel=choose(route_score(prob,base,pred,kind),gain,pred!=base,users,coh);cands.append((sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],-sel["harm"],-sel["changed"],c,kind,sel))
  best=max(cands);c,kind,sel=best[5],best[6],best[7];m=model(c);m.fit(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"target"));p=m.predict_proba(data[held]["x"]);full=np.zeros((len(p),40));full[:,m.classes_.astype(int)]=p;pred=full.argmax(1);mask=(pred!=data[held]["base"])&(route_score(full,data[held]["base"],pred,kind)>=sel["threshold"]);out=data[held]["base"].copy();out[mask]=pred[mask];outs[held]=out;bc=int(np.sum(data[held]["base"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"C":c,"score":kind,"source_selection":sel,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(mask.sum())}}
 labels=cat([data[n] for n in S],"labels");base=cat([data[n] for n in S],"base");pred=np.concatenate([outs[n] for n in S]);correct=int(np.sum(pred==labels));bc=int(np.sum(base==labels));folds=[report["cohorts"][n]["held"]["net"] for n in S];report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":correct,"accuracy":correct/len(labels),"net_vs_p180":correct-bc,"fold_nets":folds}
 # final Test config chosen from three-cohort OOF truth with cohort stability
 candidates=[]
 for c in CS:
  probs=[]
  for held in S:
   src=[n for n in S if n!=held];m=model(c);m.fit(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"target"));p=m.predict_proba(data[held]["x"]);f=np.zeros((len(p),40));f[:,m.classes_.astype(int)]=p;probs.append(f)
  prob=np.concatenate(probs);pp=prob.argmax(1);gain=(pp==labels).astype(int)-(base==labels).astype(int);users=cat([data[n] for n in S],"users");coh=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in S])
  for kind in ("confidence","gap_base","margin"):
   sel=choose(route_score(prob,base,pp,kind),gain,pp!=base,users,coh);candidates.append((sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],-sel["harm"],-sel["changed"],c,kind,sel))
 best=max(candidates);c,kind,sel=best[5],best[6],best[7];ids,tx,tbase=test_features();m=model(c);m.fit(cat([data[n] for n in S],"x"),cat([data[n] for n in S],"target"));p=m.predict_proba(tx);full=np.zeros((len(p),40));full[:,m.classes_.astype(int)]=p;pp=full.argmax(1);mask=(pp!=tbase)&(route_score(full,tbase,pp,kind)>=sel["threshold"]);tout=tbase.copy();tout[mask]=pp[mask];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p191_calibrated_meta.csv";io.write_submission(sub,io.read_rows(P89T),tout);report["test"]={"C":c,"score":kind,"oof_selection":sel,"changes_vs_p180_base":int(mask.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=ids,base_prediction=tbase,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
