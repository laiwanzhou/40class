"""Outer-safe rescue selector for common-space SigLIP2 workspace embeddings."""
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
H=Path(__file__).resolve().parent;O=H/"runs/p338_siglip2_rescue_selector_oof_v2";CACHE=H/"runs/p335_siglip2_workspace_state_cache_v1/features.npy";P336=H/"runs/p336_siglip2_workspace_state_ridge_v1/oof_predictions.npz";P244=H/"runs/p244_dual_physical_group_v1/predictions.npz";P307=H/"runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def scalar(p,b,a):r=np.arange(len(b));s=np.sort(p,1);return np.stack((p[r,a],p[r,b],p[r,a]-p[r,b],p.max(1),s[:,-1]-s[:,-2]),1)
def part(data,n,v,z336,z244,z307,z310,pos,offset):
 q=data[n].split;ids=q.sample_ids.astype(str);ix=np.asarray([pos[x] for x in ids]);p=z336["all_frames_a3000_probability"][ix].astype(float);p244=z244[f"{n}_held_probability"].astype(float);p307=z307[f"{n}_group_probability"].astype(float);base=z310["prediction"][offset:offset+len(ids)].astype(int);alt=p.argmax(1);raw=v[ix].reshape(len(ix),-1).astype(np.float32);meta=np.concatenate((np.log(np.clip(p,1e-7,1)),np.log(np.clip(p244,1e-7,1)),np.log(np.clip(p307,1e-7,1)),scalar(p,base,alt),scalar(p244,base,alt),scalar(p307,base,alt),np.eye(40)[base],np.eye(40)[alt]),1).astype(np.float32);labels=q.labels.astype(int);gain=(alt==labels).astype(int)-(base==labels).astype(int);return {"raw":raw,"meta":meta,"labels":labels,"users":q.users.astype(str),"base":base,"alt":alt,"gain":gain,"disagree":alt!=base}
def cat(ps):return {k:np.concatenate([p[k] for p in ps]) for k in ps[0]}
def fit_score(cfg,tr,te):
 dim,kind,param=cfg;q=tr["disagree"];target=(tr["gain"][q]>0).astype(int)
 if len(np.unique(target))<2:return np.zeros(len(te["labels"]))
 nc=min(dim,int(q.sum())-2);pca=PCA(n_components=nc,whiten=True,svd_solver="randomized",random_state=338);pca.fit(tr["raw"][q]);x=np.concatenate((tr["meta"][q],pca.transform(tr["raw"][q])),1);tx=np.concatenate((te["meta"],pca.transform(te["raw"])),1)
 if kind=="logistic":m=make_pipeline(StandardScaler(),LogisticRegression(C=param,class_weight="balanced",solver="liblinear",max_iter=1800))
 else:m=ExtraTreesClassifier(n_estimators=600,max_depth=int(param),min_samples_leaf=4,max_features="sqrt",class_weight="balanced",random_state=338,n_jobs=-1)
 m.fit(x,target);return m.predict_proba(tx)[:,1]
def percentile(score,mask):
 out=np.full(len(score),-np.inf);idx=np.flatnonzero(mask)
 if len(idx):
  order=np.argsort(np.argsort(score[idx],kind="stable"),kind="stable");out[idx]=(order+1)/len(idx)
 return out
def main():
 print("P338 trains a source-cross-predicted rescue selector in the common frozen SigLIP2 space.",flush=True);data=load_candidate_splits();v=np.asarray(np.load(CACHE,mmap_mode="r"),np.float32);a=np.load(P336);b=np.load(P244);c=np.load(P307);d=np.load(P310);pos={x:i for i,x in enumerate(a["sample_ids"].astype(str))};parts={};off=0
 for n in S:parts[n]=part(data,n,v,a,b,c,d,pos,off);off+=len(parts[n]["labels"])
 configs=[(x,"logistic",c) for x in (32,64) for c in (.01,.1)]+[(x,"trees",dep) for x in (32,64) for dep in (4,6)];report={"stage":"P338_SigLIP2_rescue_selector_OOF_v2","status":"complete","protocol":{"base":"P310","proposal":"P336 all_frames_a3000","common_embedding":"P335 frozen SigLIP2 8 workspace frames","source_only_PCA":[32,64],"score_calibration":"within-cohort disagreement percentile","source_inner_cross_prediction":True,"held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];source=cat([parts[n] for n in src]);coh=np.concatenate([np.full(len(parts[n]["labels"]),n,object) for n in src]);best=None
  for cfg in configs:
   scores={}
   for trn,ten in ((src[0],src[1]),(src[1],src[0])):scores[ten]=percentile(fit_score(cfg,parts[trn],parts[ten]),parts[ten]["disagree"])
   sc=np.concatenate([scores[n] for n in src]);sel=choose(sc,source["gain"],source["disagree"],source["users"],coh);key=(sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],sel["rescue"],-sel["harm"],-sel["changed"],cfg[1]=="logistic",-cfg[0],-float(cfg[2]));cand=(key,cfg,sel)
   if best is None or cand[0]>best[0]:best=cand
  cfg,sel=best[1:];hs=percentile(fit_score(cfg,source,parts[held]),parts[held]["disagree"]);route=parts[held]["disagree"]&(hs>=sel["threshold"]);out=parts[held]["base"].copy();out[route]=parts[held]["alt"][route];outs[held]=out;y=parts[held]["labels"];base=parts[held]["base"];report["cohorts"][held]={"source":src,"config":{"pca":cfg[0],"model":cfg[1],"parameter":cfg[2]},"source_selection":sel,"held":{"rows":len(y),"base_correct":int(np.sum(base==y)),"correct":int(np.sum(out==y)),"net":int(np.sum(out==y)-np.sum(base==y)),"changed":int(route.sum()),"rescue":int(np.sum(route&(base!=y)&(out==y))),"harm":int(np.sum(route&(base==y)&(out!=y)))}}
 labels=np.concatenate([parts[n]["labels"] for n in S]);base=np.concatenate([parts[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p310":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: source-only PCA rescue selector in common SigLIP2 space.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
