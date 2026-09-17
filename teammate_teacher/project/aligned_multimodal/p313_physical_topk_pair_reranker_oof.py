"""Physical-token pair specialists restricted to P244 Top-3/Top-5."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier
from p90_teacher_common import load_protocol
from p238_physical_token_transformer_oof import PATHS
from p311_p245_topk_pair_reranker_oof import SPLITS,build,choose,eligible,source_pairs
H=Path(__file__).resolve().parent;O=H/"runs/p313_physical_topk_pair_reranker_oof_v1";KS=(3,5);ALPHAS=(300.,1000.,3000.)
def norm(x):
 x=np.asarray(x,np.float32);return x/np.clip(np.linalg.norm(x,axis=1,keepdims=True),1e-6,None)
def physical():
 p=load_protocol();parts=[]
 for path in PATHS:
  z=np.load(path)
  if not np.array_equal(z["sample_ids"].astype(str),p.sample_ids):raise RuntimeError("physical order")
  v=z["features"].astype(np.float32).reshape(len(p.labels),-1,768);parts.extend((norm(v.mean(1)),norm(v.std(1))))
 return p.sample_ids,np.concatenate(parts,1).astype(np.float32)
def cat(parts):return {key:np.concatenate([p[key] for p in parts]) for key in ("ids","labels","users","base","posterior","order","cohort","physical")}
def fit_predict(train,held,a,b,alpha):
 y=train["labels"].astype(int);m=(y==a)|(y==b)
 if np.sum(y[m]==a)<3 or np.sum(y[m]==b)<3:return None
 clf=RidgeClassifier(alpha=alpha,class_weight="balanced",solver="lsqr");clf.fit(train["physical"][m],y[m]);score=np.asarray(clf.decision_function(held["physical"]),float);return score
def main():
 data,names=build();ids,x=physical();pos={q:i for i,q in enumerate(ids.astype(str))}
 for cohort in SPLITS:
  data[cohort]["cohort"]=np.full(len(data[cohort]["labels"]),cohort,object);data[cohort]["physical"]=x[[pos[q] for q in data[cohort]["ids"]]]
 report={"stage":"P313_physical_TopK_pair_reranker_OOF","status":"complete","protocol":{"base":"P245","candidate_k":list(KS),"alphas":list(ALPHAS),"physical_features":"per-backbone token mean+std, L2 normalized","inner_cross_cohort_selection":True,"rule_gate":"zero harm, both source cohorts positive, at least two positive subjects","held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in SPLITS:
  src=[n for n in SPLITS if n!=held];source=cat([data[n] for n in src]);best=None
  for k in KS:
   pairs=source_pairs(source,k)
   for alpha in ALPHAS:
    rules=[];audit=[];offsets={src[0]:(0,len(data[src[0]]["labels"])),src[1]:(len(data[src[0]]["labels"]),len(source["labels"]))}
    for a,b in sorted(pairs):
     proposal=source["base"].copy();score=np.full(len(proposal),-np.inf);ok=True
     for trn,ten in ((src[0],src[1]),(src[1],src[0])):
      decision=fit_predict(data[trn],data[ten],a,b,alpha)
      if decision is None:ok=False;break
      lo,hi=offsets[ten];proposal[lo:hi]=np.where(decision>=0,b,a);score[lo:hi]=np.abs(decision)
     if not ok:continue
     row=choose(score,proposal,source,a,b,k);row["alpha"]=alpha;audit.append(row)
     if row["rescue"]>=2 and row["harm"]==0 and row["minimum_user_gain"]>=0 and row["minimum_cohort_gain"]>=1 and row["positive_users"]>=2:rules.append(row)
    key=(sum(r["net"] for r in rules),sum(r["rescue"] for r in rules),-sum(r["harm"] for r in rules),-len(rules),-k,-alpha)
    if best is None or key>best[0]:best=(key,rules,len(audit))
  rules=best[1];out=data[held]["base"].copy();best_score=np.full(len(out),-np.inf)
  for rule in rules:
   a,b=rule["pair"];decision=fit_predict(source,data[held],a,b,rule["alpha"])
   if decision is None:continue
   proposal=np.where(decision>=0,b,a);score=np.abs(decision);q=eligible(data[held],a,b,rule["k"])&(proposal!=data[held]["base"])&(score>=rule["threshold"])&(score>best_score);out[q]=proposal[q];best_score[q]=score[q]
  outs[held]=out;y=data[held]["labels"];base=data[held]["base"];report["cohorts"][held]={"source":src,"rules":rules,"source_audit_count":best[2],"held":{"rows":len(y),"base_correct":int(np.sum(base==y)),"correct":int(np.sum(out==y)),"net":int(np.sum(out==y)-np.sum(base==y)),"changed":int(np.sum(out!=base)),"rescue":int(np.sum((out!=base)&(base!=y)&(out==y))),"harm":int(np.sum((out!=base)&(base==y)&(out!=y)))}}
 labels=np.concatenate([data[n]["labels"] for n in SPLITS]);base=np.concatenate([data[n]["base"] for n in SPLITS]);out=np.concatenate([outs[n] for n in SPLITS]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p245":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in SPLITS]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in SPLITS});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps({"aggregate":report["aggregate"],"cohorts":{n:report["cohorts"][n]["held"] for n in SPLITS},"rules":{n:report["cohorts"][n]["rules"] for n in SPLITS}},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
