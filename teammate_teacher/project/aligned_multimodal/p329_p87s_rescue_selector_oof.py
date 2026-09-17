"""Outer-safe rescue selector between P310 and the terminal P87-S OOF."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from p117_transductive_multicandidate_router import load_candidate_splits
from p191_source_truth_calibrated_p150_distillation import choose
H=Path(__file__).resolve().parent;O=H/"runs/p329_p87s_rescue_selector_oof_v2";P328=H/"runs/p328_p87s_threefold_oof_audit_v1/oof_predictions.npz";P244=H/"runs/p244_dual_physical_group_v1/predictions.npz";P307=H/"runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz";P255=H/"runs/p255_repeat_augmented_physical_group_v1/predictions.npz";P306=H/"runs/p306_union_full_repeat_physical_oof_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0");MOTION={n:H/f"runs/p87s_mobind_holdout{i}_v1/subject_holdout_predictions.csv" for i,n in enumerate(S,1)}
def align(z,key,ids):d={x:i for i,x in enumerate(z["sample_ids"].astype(str))};return z[key][[d[x] for x in ids]]
def scalars(p,base,alt):
 r=np.arange(len(base));s=np.sort(p,axis=1);return np.stack((p[r,alt],p[r,base],p[r,alt]-p[r,base],p.max(1),s[:,-1]-s[:,-2],-np.sum(p*np.log(np.clip(p,1e-8,1)),1)/np.log(40)),1)
def motion(ids,n):
 with MOTION[n].open("r",encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
 lookup={x["sample_id"]:x for x in rows};q=[lookup[x] for x in ids];return np.asarray([int(x["skeleton_prediction"]) for x in q]),np.asarray([float(x["skeleton_confidence"]) for x in q]),np.asarray([int(x["imu_prediction"]) for x in q]),np.asarray([float(x["imu_confidence"]) for x in q])
def part(data,n,a,b,c,d,e):
 q=data[n].split;ids=q.sample_ids.astype(str);p=a[f"{n}_probability"].astype(float);v=a[f"{n}_visual_probability"].astype(float);p244=b[f"{n}_held_probability"].astype(float);p307=c[f"{n}_group_probability"].astype(float);p255=d[f"{n}_held_probability"].astype(float);p306=align(e,"probability",ids).astype(float);base=a["base_prediction"][sum(len(data[x].split.labels) for x in S[:S.index(n)]):sum(len(data[x].split.labels) for x in S[:S.index(n)+1])].astype(int);alt=p.argmax(1);sk,skc,imu,imuc=motion(ids,n);probs=(p,v,p244,p307,p255,p306);x=np.concatenate([*(np.log(np.clip(z,1e-7,1)) for z in probs),*(scalars(z,base,alt) for z in probs),np.stack([v.argmax(1)==alt,v.argmax(1)==base,p244.argmax(1)==alt,p307.argmax(1)==alt,p255.argmax(1)==alt,p306.argmax(1)==alt,sk==alt,sk==base,imu==alt,imu==base,skc,imuc],1),np.eye(40)[base],np.eye(40)[alt],np.eye(40)[sk],np.eye(40)[imu]],1).astype(np.float32);labels=q.labels.astype(int);gain=(alt==labels).astype(int)-(base==labels).astype(int);return {"x":x,"labels":labels,"users":q.users.astype(str),"base":base,"alt":alt,"gain":gain,"disagree":alt!=base}
def make(kind,param):
 if kind=="logistic":return make_pipeline(StandardScaler(),LogisticRegression(C=param,class_weight="balanced",solver="liblinear",max_iter=1600))
 return ExtraTreesClassifier(n_estimators=500,max_depth=int(param),min_samples_leaf=4,max_features="sqrt",class_weight="balanced",random_state=329,n_jobs=-1)
def fit_score(kind,param,train,test):
 q=train["disagree"];target=(train["gain"][q]>0).astype(int)
 if len(np.unique(target))<2:return np.zeros(len(test["labels"]))
 m=make(kind,param);m.fit(train["x"][q],target);return m.predict_proba(test["x"])[:,1]
def cat(ps):return {k:np.concatenate([p[k] for p in ps]) for k in ps[0]}
def main():
 print("P329 v2 adds source-pure Skeleton and IMU branch evidence to the P87-S rescue selector.",flush=True);data=load_candidate_splits();a=np.load(P328);b=np.load(P244);c=np.load(P307);d=np.load(P255);e=np.load(P306);parts={n:part(data,n,a,b,c,d,e) for n in S};configs=[("logistic",x) for x in (.003,.01,.03,.1)]+[("trees",x) for x in (4,6,8)];report={"stage":"P329_P87S_rescue_selector_OOF_v2","status":"complete","protocol":{"base":"P310","proposal":"exact terminal P87-S subject OOF","feature_posteriors":["P87S final","P87S visual","P244","P307","P255","P306"],"motion_gate_features":["Skeleton prediction/confidence","IMU prediction/confidence","candidate/base agreements"],"source_inner_cross_prediction":True,"held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];source=cat([parts[n] for n in src]);coh=np.concatenate([np.full(len(parts[n]["labels"]),n,object) for n in src]);best=None
  for kind,param in configs:
   scores={}
   for trn,ten in ((src[0],src[1]),(src[1],src[0])):scores[ten]=fit_score(kind,param,parts[trn],parts[ten])
   sc=np.concatenate([scores[n] for n in src]);sel=choose(sc,source["gain"],source["disagree"],source["users"],coh);key=(sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],sel["rescue"],-sel["harm"],-sel["changed"],kind=="logistic",-float(param));cand=(key,kind,param,sel)
   if best is None or cand[0]>best[0]:best=cand
  kind,param,sel=best[1:];hs=fit_score(kind,param,source,parts[held]);route=parts[held]["disagree"]&(hs>=sel["threshold"]);out=parts[held]["base"].copy();out[route]=parts[held]["alt"][route];outs[held]=out;y=parts[held]["labels"];base=parts[held]["base"];report["cohorts"][held]={"source":src,"model":kind,"parameter":param,"source_selection":sel,"held":{"rows":len(y),"base_correct":int(np.sum(base==y)),"correct":int(np.sum(out==y)),"net":int(np.sum(out==y)-np.sum(base==y)),"changed":int(route.sum()),"rescue":int(np.sum(route&(base!=y)&(out==y))),"harm":int(np.sum(route&(base==y)&(out!=y)))}}
 labels=np.concatenate([parts[n]["labels"] for n in S]);base=np.concatenate([parts[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p310":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: source-cross-predicted rescue selector for exact terminal P87-S OOF.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
