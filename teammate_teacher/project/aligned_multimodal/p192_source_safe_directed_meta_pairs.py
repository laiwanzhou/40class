"""Source-safe directed pair selector for the P150 meta student."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p191_source_truth_calibrated_p150_distillation import S, cat, model
from p190_inductive_p150_meta_distillation import load_oof,test_features,P89T
import p89_build_dual_consensus_submission as io

H=Path(__file__).resolve().parent;O=H/"runs/p192_source_safe_directed_meta_pairs_v1"
def prob_model(trainx,target,testx):
 m=model(.03);m.fit(trainx,target);p=m.predict_proba(testx);f=np.zeros((len(p),40));f[:,m.classes_.astype(int)]=p;return f
def inner(data,names):
 out={}
 for tr,te in ((names[0],names[1]),(names[1],names[0])):out[te]=prob_model(data[tr]["x"],data[tr]["target"],data[te]["x"])
 return np.concatenate([out[n] for n in names])
def select_pairs(base,pred,prob,labels,users):
 rows=np.arange(len(base));margin=np.sort(prob,axis=1)[:,-1]-np.sort(prob,axis=1)[:,-2];rules={};audit=[]
 for a,b in sorted(set(zip(base[pred!=base].tolist(),pred[pred!=base].tolist()))):
  pair=(base==a)&(pred==b);gain=(pred==labels).astype(int)-(base==labels).astype(int);values=np.unique(np.concatenate(([np.inf,-np.inf],np.linspace(0,1,101),np.quantile(margin[pair],[.25,.5,.75]) if pair.any() else [np.inf])));best=None
  for t in values:
   m=pair&(margin>=t);per={u:int(gain[m&(users==u)].sum()) for u in sorted(set(users.tolist()))};r=int(np.sum(m&(gain>0)));h=int(np.sum(m&(gain<0)));row={"base":int(a),"candidate":int(b),"threshold":float(t),"selected":int(m.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(per.values()),"per_user":per};key=(row["minimum_user_gain"]>=0,row["net"],r,-h,-row["selected"])
   if best is None or key>best[0]:best=(key,row)
  row=best[1];row["eligible"]=row["minimum_user_gain"]>=0 and row["rescue"]>=2 and row["net"]>0
  audit.append(row)
  if row["eligible"]:rules[(a,b)]=row["threshold"]
 return rules,audit
def apply(base,prob,rules):
 pred=prob.argmax(1);margin=np.sort(prob,axis=1)[:,-1]-np.sort(prob,axis=1)[:,-2];out=base.copy();mask=np.zeros(len(base),bool)
 for (a,b),t in rules.items():m=(base==a)&(pred==b)&(margin>=t);out[m]=b;mask|=m
 return out,mask
def main():
 data=load_oof();report={"stage":"P192_source_safe_directed_meta_pairs","status":"complete","protocol":{"meta_target":"P150 OOF","pair_and_threshold_source_only":True,"held_labels_or_targets_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];p=inner(data,src);base=cat([data[n] for n in src],"base");labels=cat([data[n] for n in src],"labels");users=cat([data[n] for n in src],"users");rules,audit=select_pairs(base,p.argmax(1),p,labels,users);hp=prob_model(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"target"),data[held]["x"]);out,mask=apply(data[held]["base"],hp,rules);outs[held]=out;bc=int(np.sum(data[held]["base"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"eligible_rules":[{"base":a,"candidate":b,"threshold":t} for (a,b),t in rules.items()],"source_audit":audit,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(mask.sum())}}
 labels=cat([data[n] for n in S],"labels");base=cat([data[n] for n in S],"base");pred=np.concatenate([outs[n] for n in S]);cor=int(np.sum(pred==labels));bc=int(np.sum(base==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p180":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]}
 probs=[]
 for held in S:
  src=[n for n in S if n!=held];probs.append(prob_model(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"target"),data[held]["x"]))
 p=np.concatenate(probs);users=cat([data[n] for n in S],"users");rules,audit=select_pairs(base,p.argmax(1),p,labels,users);ids,tx,tbase=test_features();tp=prob_model(cat([data[n] for n in S],"x"),cat([data[n] for n in S],"target"),tx);tout,mask=apply(tbase,tp,rules);O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p192_directed_meta.csv";io.write_submission(sub,io.read_rows(P89T),tout);report["test"]={"eligible_rules":[{"base":a,"candidate":b,"threshold":t} for (a,b),t in rules.items()],"oof_rule_audit":audit,"changes":int(mask.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=ids,base_prediction=tbase,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
