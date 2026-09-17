"""Strict outer-cross-fit pair specialists for P245 Top-K errors.

The model never tries to solve all 40 classes.  A source error authorizes an
unordered pair only when the true class is already in the deployable P244
posterior Top-K.  Pair models consume only deployable expert posterior support
for the two classes.  Inner cross-cohort predictions select thresholds and
rules; held labels are used only for the final audit.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p165_deployable_group_teacher import SPLITS
from p173_vjepa_augmented_group_teacher import build_train_bank
from p255_repeat_augmented_physical_group import al
from p307_union_repeat_group_sequence_audit import SOURCES

H=Path(__file__).resolve().parent
O=H/"runs/p311_p245_topk_pair_reranker_oof_v3"
P244=H/"runs/p244_dual_physical_group_v1/predictions.npz"
P245=H/"runs/p245_p244_soft_sequence_gate_v1/predictions.npz"
KS=(3,5)
CS=(.01,.03,.1)

def build():
 tr,names=build_train_bank()
 for path,key,name in SOURCES:
  z=np.load(path)
  for cohort in SPLITS:
   q=tr[cohort];q["bank"]=np.concatenate((q["bank"],al(z[key],z["sample_ids"],q["ids"])[:,None,:]),1)
  names.append(name)
 g=np.load(P244);s=np.load(P245);out={}
 for cohort in SPLITS:
  q=tr[cohort];p=g[f"{cohort}_held_probability"].astype(np.float32);base=s[f"{cohort}_held_prediction"].astype(int);order=np.argsort(-p,axis=1,kind="stable");out[cohort]={**q,"posterior":p,"base":base,"order":order}
 return out,names

def pair_features(part,a,b):
 bank=np.asarray(part["bank"],np.float32);p=np.asarray(part["posterior"],np.float32);pa=bank[:,:,a];pb=bank[:,:,b];eps=1e-6;vote=bank.argmax(2);rank=np.argsort(np.argsort(-p,axis=1,kind="stable"),axis=1)+1
 scalar=np.stack((pa.mean(1),pb.mean(1),pa.max(1),pb.max(1),pa.std(1),pb.std(1),(vote==a).mean(1),(vote==b).mean(1),p[:,a],p[:,b],p[:,b]-p[:,a],np.log(p[:,b]+eps)-np.log(p[:,a]+eps),rank[:,a]/40.,rank[:,b]/40.),1)
 return np.concatenate((np.sqrt(np.clip(pa,0,1)),np.sqrt(np.clip(pb,0,1)),pb-pa,scalar),1).astype(np.float32)

def model(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight="balanced",solver="liblinear",max_iter=1500))

def fit_predict(train,held,a,b,c):
 y=train["labels"].astype(int);m=(y==a)|(y==b)
 if np.sum(y[m]==a)<3 or np.sum(y[m]==b)<3:return None
 clf=model(c);clf.fit(pair_features(train,a,b)[m],(y[m]==b).astype(int));return clf.predict_proba(pair_features(held,a,b))[:,1]

def eligible(part,a,b,k):
 base=part["base"];top=part["order"][:,:k];other=np.where(base==a,b,a);return ((base==a)|(base==b))&np.any(top==other[:,None],axis=1)

def choose(score,proposal,part,a,b,k):
 base=part["base"];labels=part["labels"];q=eligible(part,a,b,k)&(proposal!=base);gain=(proposal==labels).astype(int)-(base==labels).astype(int);vals=np.unique(np.concatenate(([-np.inf,np.inf],np.linspace(-1,1,201),np.quantile(score[q],np.linspace(.1,.9,9)) if q.any() else [np.inf])));best=None
 for t in vals:
  m=q&(score>=t);users=part["users"];coh=part["cohort"];peru={u:int(gain[m&(users==u)].sum()) for u in sorted(set(users.tolist()))};perc={c:int(gain[m&(coh==c)].sum()) for c in sorted(set(coh.tolist()))};r=int(np.sum(m&(gain>0)));h=int(np.sum(m&(gain<0)));row={"pair":[a,b],"k":k,"threshold":float(t),"changed":int(m.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(peru.values()),"minimum_cohort_gain":min(perc.values()),"positive_users":int(sum(v>0 for v in peru.values())),"per_user":peru,"per_cohort":perc};key=(h==0,row["minimum_cohort_gain"]>=1,row["positive_users"]>=2,row["net"],r,-row["changed"])
  if best is None or key>best[0]:best=(key,row)
 return best[1]

def cat(parts):
 return {key:np.concatenate([p[key] for p in parts]) for key in ("ids","labels","users","base","bank","posterior","order","cohort")}

def source_pairs(part,k):
 pairs={};y=part["labels"];base=part["base"]
 for i in np.flatnonzero((base!=y)&np.any(part["order"][:,:k]==y[:,None],axis=1)):
  pair=tuple(sorted((int(base[i]),int(y[i]))));pairs[pair]=pairs.get(pair,0)+1
 return {pair for pair,count in pairs.items() if count>=2}

def main():
 data,names=build()
 for cohort in SPLITS:data[cohort]["cohort"]=np.full(len(data[cohort]["labels"]),cohort,object)
 report={"stage":"P311_P245_Top3_Top5_pair_reranker_OOF_v3","status":"complete","protocol":{"base":"P245","expert_count":len(names),"candidate_k":list(KS),"k_selected_on_source_only":True,"equal_source_evidence_prefers_smaller_k":True,"C":list(CS),"pair_requires_source_topk_error_count":2,"rule_gate":"zero harm, both source cohorts positive, at least two positive subjects","inner_cross_cohort_rule_selection":True,"held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in SPLITS:
  src=[n for n in SPLITS if n!=held];best_rules=None;best_key=None;audits=[]
  for k in KS:
   source=cat([data[n] for n in src]);pairs=source_pairs(source,k)
   for c in CS:
    proposals=np.tile(source["base"][:,None],(1,len(pairs))) if pairs else np.empty((len(source["base"]),0),int);scores=np.full(proposals.shape,-np.inf,float);rules=[]
    offsets={src[0]:(0,len(data[src[0]]["labels"])),src[1]:(len(data[src[0]]["labels"]),len(source["labels"]))}
    for j,(a,b) in enumerate(sorted(pairs)):
     ok=True
     for trn,ten in ((src[0],src[1]),(src[1],src[0])):
      pr=fit_predict(data[trn],data[ten],a,b,c)
      if pr is None:ok=False;break
      lo,hi=offsets[ten];proposals[lo:hi,j]=np.where(pr>=.5,b,a);scores[lo:hi,j]=np.abs(pr-.5)*2
     if not ok:continue
     row=choose(scores[:,j],proposals[:,j],source,a,b,k);row["C"]=c;audits.append(row)
     if row["rescue"]>=2 and row["harm"]==0 and row["net"]>0 and row["minimum_user_gain"]>=0 and row["minimum_cohort_gain"]>=1 and row["positive_users"]>=2:rules.append(row)
    key=(sum(r["net"] for r in rules),sum(r["rescue"] for r in rules),-sum(r["harm"] for r in rules),-len(rules),-k,-c)
    if best_key is None or key>best_key:best_key=key;best_rules=rules
  out=data[held]["base"].copy();best_score=np.full(len(out),-np.inf);applied=np.zeros(len(out),bool)
  for rule in best_rules or []:
   a,b=rule["pair"];pr=fit_predict(cat([data[n] for n in src]),data[held],a,b,rule["C"])
   if pr is None:continue
   proposal=np.where(pr>=.5,b,a);score=np.abs(pr-.5)*2;q=eligible(data[held],a,b,rule["k"])&(proposal!=data[held]["base"])&(score>=rule["threshold"])&(score>best_score);out[q]=proposal[q];best_score[q]=score[q];applied[q]=True
  outs[held]=out;y=data[held]["labels"];base=data[held]["base"];report["cohorts"][held]={"source":src,"rules":best_rules or [],"source_audit_count":len(audits),"held":{"rows":len(y),"base_correct":int(np.sum(base==y)),"correct":int(np.sum(out==y)),"net":int(np.sum(out==y)-np.sum(base==y)),"changed":int(np.sum(out!=base)),"rescue":int(np.sum((out!=base)&(base!=y)&(out==y))),"harm":int(np.sum((out!=base)&(base==y)&(out!=y)))}}
 labels=np.concatenate([data[n]["labels"] for n in SPLITS]);base=np.concatenate([data[n]["base"] for n in SPLITS]);out=np.concatenate([outs[n] for n in SPLITS]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p245":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in SPLITS]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in SPLITS});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps({"aggregate":report["aggregate"],"cohorts":{n:report["cohorts"][n]["held"] for n in SPLITS},"rules":{n:report["cohorts"][n]["rules"] for n in SPLITS}},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
