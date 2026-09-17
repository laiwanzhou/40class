"""Distributionally robust two-head rescue/harm gate over complementary candidates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from p117_transductive_multicandidate_router import load_candidate_splits
from p191_source_truth_calibrated_p150_distillation import choose
H=Path(__file__).resolve().parent;O=H/"runs/p346_distributionally_robust_rescue_harm_gate_v1";S=("H1_selection","H2_confirmation","H3_independent_fold0");P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";P244=H/"runs/p244_dual_physical_group_v1/predictions.npz";P307=H/"runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz";P255=H/"runs/p255_repeat_augmented_physical_group_v1/predictions.npz";P306=H/"runs/p306_union_full_repeat_physical_oof_v1/oof_predictions.npz";P328=H/"runs/p328_p87s_threefold_oof_audit_v1/oof_predictions.npz";P336=H/"runs/p336_siglip2_workspace_state_ridge_v1/oof_predictions.npz";P344=H/"runs/p344_siglip2_threeview_temporal_oof_v1/oof_predictions.npz";P279=H/"runs/p279_p278_fixed_emission07_transition04_v1/predictions.npz";P278=H/"runs/p278_centroid_augmented_group_v1/predictions.npz";NAMES=("p87s","sig_ridge","sig_temporal","p279","p307")
def align(z,key,ids):d={x:i for i,x in enumerate(z["sample_ids"].astype(str))};return z[key][[d[x] for x in ids]]
def scalar(p,b,a):r=np.arange(len(b));s=np.sort(p,1);return np.stack((p[r,a],p[r,b],p[r,a]-p[r,b],p.max(1),s[:,-1]-s[:,-2],-np.sum(p*np.log(np.clip(p,1e-8,1)),1)/np.log(40)),1)
def part(data,n,z):
 q=data[n].split;ids=q.sample_ids.astype(str);m=len(ids);offset=sum(len(data[x].split.labels) for x in S[:S.index(n)]);base=z["p310"]["prediction"][offset:offset+m].astype(int);base_probs=(z["p244"][f"{n}_held_probability"].astype(float),z["p307"][f"{n}_group_probability"].astype(float),z["p255"][f"{n}_held_probability"].astype(float),align(z["p306"],"probability",ids).astype(float));cand_probs=(z["p328"][f"{n}_probability"].astype(float),align(z["p336"],"all_frames_a3000_probability",ids).astype(float),align(z["p344"],"probability",ids).astype(float),z["p278"][f"{n}_held_probability"].astype(float),z["p307"][f"{n}_group_probability"].astype(float));cand_pred=np.stack((cand_probs[0].argmax(1),cand_probs[1].argmax(1),cand_probs[2].argmax(1),z["p279"][f"{n}_held_prediction"].astype(int),z["p307"][f"{n}_group_prediction"].astype(int)),1);rows=[]
 for k,p in enumerate(cand_probs):
  a=cand_pred[:,k];vote=(cand_pred==a[:,None]).mean(1);x=np.concatenate([*(np.log(np.clip(v,1e-7,1)) for v in (*base_probs,p)),*(scalar(v,base,a) for v in (*base_probs,p)),vote[:,None],np.eye(len(NAMES))[np.full(m,k)],np.eye(40)[base],np.eye(40)[a]],1);rows.append(x)
 x=np.stack(rows,1).astype(np.float32);y=q.labels.astype(int);dis=cand_pred!=base[:,None];rescue=(base[:,None]!=y[:,None])&(cand_pred==y[:,None]);harm=(base[:,None]==y[:,None])&(cand_pred!=y[:,None]);gain=rescue.astype(int)-harm.astype(int);return {"x":x,"labels":y,"users":q.users.astype(str),"base":base,"candidate":cand_pred,"disagree":dis,"rescue":rescue,"harm":harm,"gain":gain}
def cat(ps):return {k:np.concatenate([p[k] for p in ps]) for k in ps[0]}
def fit_head(x,target,c):
 if len(np.unique(target))<2:return float(target[0]) if len(target) else 0.
 m=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight="balanced",solver="liblinear",max_iter=1600));m.fit(x,target.astype(int));return m
def predict(m,x):return np.full(len(x),m,float) if isinstance(m,float) else m.predict_proba(x)[:,1]
def train_pair(part,c):
 q=part["disagree"].reshape(-1);x=part["x"].reshape(-1,part["x"].shape[-1])[q];return fit_head(x,part["rescue"].reshape(-1)[q],c),fit_head(x,part["harm"].reshape(-1)[q],c)
def score(models,part):
 x=part["x"].reshape(-1,part["x"].shape[-1]);rs=[];hs=[]
 for rm,hm in models:rs.append(predict(rm,x).reshape(part["candidate"].shape));hs.append(predict(hm,x).reshape(part["candidate"].shape))
 robust=np.min(np.stack(rs),0)-np.max(np.stack(hs),0);robust[~part["disagree"]]=-np.inf;idx=robust.argmax(1);return part["candidate"][np.arange(len(idx)),idx],robust[np.arange(len(idx)),idx]
def main():
 print("P346 tests separate rescue/harm heads with conservative cross-domain aggregation.",flush=True);data=load_candidate_splits();z={"p310":np.load(P310),"p244":np.load(P244),"p307":np.load(P307),"p255":np.load(P255),"p306":np.load(P306),"p328":np.load(P328),"p336":np.load(P336),"p344":np.load(P344),"p279":np.load(P279),"p278":np.load(P278)};parts={n:part(data,n,z) for n in S};report={"stage":"P346_distributionally_robust_rescue_harm_gate","status":"complete","protocol":{"base":"P310","candidates":list(NAMES),"heads":["P(rescue)","P(harm)"],"held_score":"min source P(rescue) - max source P(harm)","source_inner_cross_prediction":True,"held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];source=cat([parts[n] for n in src]);coh=np.concatenate([np.full(len(parts[n]["labels"]),n,object) for n in src]);best=None
  for c in (.003,.01,.03,.1):
   proposals={};scores={}
   for trn,ten in ((src[0],src[1]),(src[1],src[0])):proposals[ten],scores[ten]=score([train_pair(parts[trn],c)],parts[ten])
   prop=np.concatenate([proposals[n] for n in src]);sc=np.concatenate([scores[n] for n in src]);gain=(prop==source["labels"]).astype(int)-(source["base"]==source["labels"]).astype(int);sel=choose(sc,gain,prop!=source["base"],source["users"],coh);key=(sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],sel["rescue"],-sel["harm"],-sel["changed"],-c);cand=(key,c,sel)
   if best is None or cand[0]>best[0]:best=cand
  c,sel=best[1:];models=[train_pair(parts[n],c) for n in src];prop,hs=score(models,parts[held]);route=(prop!=parts[held]["base"])&(hs>=sel["threshold"]);out=parts[held]["base"].copy();out[route]=prop[route];outs[held]=out;y=parts[held]["labels"];base=parts[held]["base"];report["cohorts"][held]={"source":src,"C":c,"source_selection":sel,"held":{"rows":len(y),"base_correct":int(np.sum(base==y)),"correct":int(np.sum(out==y)),"net":int(np.sum(out==y)-np.sum(base==y)),"changed":int(route.sum()),"rescue":int(np.sum(route&(base!=y)&(out==y))),"harm":int(np.sum(route&(base==y)&(out!=y)))}}
 labels=np.concatenate([parts[n]["labels"] for n in S]);base=np.concatenate([parts[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p310":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: distributionally robust separate rescue/harm heads.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
