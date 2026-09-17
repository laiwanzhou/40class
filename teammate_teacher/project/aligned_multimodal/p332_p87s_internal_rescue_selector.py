"""Source-cross-predicted P87-S rescue selector with internal embeddings."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from p117_transductive_multicandidate_router import load_candidate_splits
from p191_source_truth_calibrated_p150_distillation import choose
from p329_p87s_rescue_selector_oof import P328,P244,P307,P255,P306,S,part,cat
H=Path(__file__).resolve().parent;O=H/"runs/p332_p87s_internal_rescue_selector_v1";INTERNAL=H/"runs/p331_p87s_internal_oof_v1/internal_features.npz"
def make_parts():
 data=load_candidate_splits();a=np.load(P328);b=np.load(P244);c=np.load(P307);d=np.load(P255);e=np.load(P306);f=np.load(INTERNAL);parts={}
 for n in S:
  q=part(data,n,a,b,c,d,e);q["embed"]=np.concatenate([f[f"{n}_visual_embedding"],f[f"{n}_window_state"],f[f"{n}_motion_semantic"],f[f"{n}_gates"]],1).astype(np.float32);parts[n]=q
 return parts
def fit_score(config,train,test):
 dim,kind,param=config;q=train["disagree"];target=(train["gain"][q]>0).astype(int)
 if len(np.unique(target))<2:return np.zeros(len(test["labels"]))
 nc=min(dim,int(q.sum())-2,train["embed"].shape[1]);pca=PCA(n_components=nc,whiten=True,svd_solver="randomized",random_state=332);pca.fit(train["embed"][q]);tx=np.concatenate((train["x"][q],pca.transform(train["embed"][q])),1);vx=np.concatenate((test["x"],pca.transform(test["embed"])),1)
 if kind=="logistic":m=make_pipeline(StandardScaler(),LogisticRegression(C=param,class_weight="balanced",solver="liblinear",max_iter=1800))
 else:m=ExtraTreesClassifier(n_estimators=600,max_depth=int(param),min_samples_leaf=4,max_features="sqrt",class_weight="balanced",random_state=332,n_jobs=-1)
 m.fit(tx,target);return m.predict_proba(vx)[:,1]
def main():
 print("P332 tests whether source-only PCA of terminal P87-S internal embeddings predicts rescue.",flush=True);parts=make_parts();configs=[(d,"logistic",c) for d in (32,64) for c in (.01,.1)]+[(d,"trees",dep) for d in (32,64) for dep in (4,6)];report={"stage":"P332_P87S_internal_rescue_selector","status":"complete","protocol":{"base":"P310","proposal":"terminal P87-S","internal":["visual embedding","window early/late/delta","motion semantic","reliability gates"],"PCA_fit_source_only":[32,64],"source_inner_cross_prediction":True,"held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];source=cat([parts[n] for n in src]);source["embed"]=np.concatenate([parts[n]["embed"] for n in src]);coh=np.concatenate([np.full(len(parts[n]["labels"]),n,object) for n in src]);best=None
  for cfg in configs:
   scores={}
   for trn,ten in ((src[0],src[1]),(src[1],src[0])):scores[ten]=fit_score(cfg,parts[trn],parts[ten])
   sc=np.concatenate([scores[n] for n in src]);sel=choose(sc,source["gain"],source["disagree"],source["users"],coh);key=(sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],sel["rescue"],-sel["harm"],-sel["changed"],cfg[1]=="logistic",-cfg[0],-float(cfg[2]));cand=(key,cfg,sel)
   if best is None or cand[0]>best[0]:best=cand
  cfg,sel=best[1:];hs=fit_score(cfg,source,parts[held]);route=parts[held]["disagree"]&(hs>=sel["threshold"]);out=parts[held]["base"].copy();out[route]=parts[held]["alt"][route];outs[held]=out;y=parts[held]["labels"];base=parts[held]["base"];report["cohorts"][held]={"source":src,"config":{"pca":cfg[0],"model":cfg[1],"parameter":cfg[2]},"source_selection":sel,"held":{"rows":len(y),"base_correct":int(np.sum(base==y)),"correct":int(np.sum(out==y)),"net":int(np.sum(out==y)-np.sum(base==y)),"changed":int(route.sum()),"rescue":int(np.sum(route&(base!=y)&(out==y))),"harm":int(np.sum(route&(base==y)&(out!=y)))}}
 labels=np.concatenate([parts[n]["labels"] for n in S]);base=np.concatenate([parts[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p310":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: source-only PCA internal P87-S rescue selector.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
