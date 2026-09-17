"""Strict source-crossfit pair specialists on raw frozen V-JEPA tokens."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
import p89_build_dual_consensus_submission as io
from p90_crossuser_visual_router import load_splits
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p226_raw_vjepa_pair_specialists_v1";TRAIN=R/"runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1/features.npy";TEST=R/"runs/p171_vjepa2_dense24_test_v1";P205=H/"runs/p205_fixed_p203_p150_residual_v1/oof_predictions.npz";P205T=H/"runs/p205_fixed_p203_p150_residual_v1/submission_p205_fixed_residual.csv";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";S=("H1_selection","H2_confirmation","H3_independent_fold0");PAIRS=((24,26),(8,10),(23,27),(0,4),(20,39),(25,27),(11,26),(7,37),(32,34),(6,37),(21,22),(17,38));ALPHAS=(10.,100.,1000.,10000.)
SUBSETS={"workspace":np.asarray((2,5,8,11)),"hand":np.asarray((14,17,20,23)),"workspace_hand":np.asarray((2,5,8,11,14,17,20,23))}
def cp(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def unit(v):v=np.asarray(v,np.float32);return v/np.maximum(np.linalg.norm(v,axis=2,keepdims=True),1e-8)
def model(a):return make_pipeline(StandardScaler(),RidgeClassifier(alpha=a,class_weight="balanced",solver="lsqr",tol=1e-4))
def infer(m,x):return m.predict(x).astype(int),np.abs(np.asarray(m.decision_function(x),float))
def threshold(base,pred,margin,labels,users,cohort,pair):
 eligible=np.isin(base,pair)&(pred!=base);gain=(pred==labels).astype(int)-(base==labels).astype(int);best=None
 if not eligible.any():return {"threshold":float("inf"),"selected":0,"rescue":0,"harm":0,"net":0,"minimum_cohort_gain":0,"minimum_user_gain":0}
 for t in np.unique(np.r_[np.inf,np.quantile(margin[eligible],np.linspace(0,1,31)),0.]):
  q=eligible&(margin>=t);pc={c:int(gain[q&(cohort==c)].sum()) for c in sorted(set(cohort.tolist()))};pu={u:int(gain[q&(users==u)].sum()) for u in sorted(set(users.tolist()))};r=int(np.sum(q&(gain>0)));h=int(np.sum(q&(gain<0)));row={"threshold":float(t),"selected":int(q.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_cohort_gain":min(pc.values()),"minimum_user_gain":min(pu.values()),"per_cohort":pc};key=(row["minimum_cohort_gain"]>=0,row["net"],r,-h,row["minimum_user_gain"],-row["selected"])
  if best is None or key>best[0]:best=(key,row)
 return best[1]
def cross(data,names,pair,variant,alpha):
 preds=[];margins=[]
 for target in names:
  tr=[n for n in names if n!=target];x=np.concatenate([data[n]["x"][variant] for n in tr]);y=np.concatenate([data[n]["labels"] for n in tr]);fit=np.isin(y,pair)
  if len(np.unique(y[fit]))<2:pred=np.full(len(data[target]["labels"]),-1);mar=np.zeros(len(pred))
  else:m=model(alpha);m.fit(x[fit],y[fit]);pred,mar=infer(m,data[target]["x"][variant])
  preds.append(pred);margins.append(mar)
 return np.concatenate(preds),np.concatenate(margins)
def select(data,names):
 base=np.concatenate([data[n]["base"] for n in names]);labels=np.concatenate([data[n]["labels"] for n in names]);users=np.concatenate([data[n]["users"] for n in names]);co=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in names]);audit=[]
 for pair in PAIRS:
  best=None
  for v in SUBSETS:
   for a in ALPHAS:
    pred,mar=cross(data,names,pair,v,a);sel=threshold(base,pred,mar,labels,users,co,pair);row={"pair":list(pair),"variant":v,"alpha":a,**sel};key=(row["minimum_cohort_gain"]>=0,row["net"],row["rescue"],-row["harm"],row["minimum_user_gain"],-row["selected"])
    if best is None or key>best[0]:best=(key,row)
  row=best[1];row["eligible"]=bool(row["net"]>=2 and row["rescue"]>=3 and row["minimum_cohort_gain"]>=0 and row["harm"]<=row["rescue"]//2);audit.append(row)
 eligible=sorted([r for r in audit if r["eligible"]],key=lambda r:(r["net"],r["rescue"],-r["harm"]),reverse=True);rules=[];used=set()
 for r in eligible:
  if not used.intersection(r["pair"]):rules.append(r);used.update(r["pair"])
 return rules,audit
def apply(data,names,target,rules):
 out=target["base"].copy();changed=np.zeros(len(out),bool);details=[]
 for r in rules:
  pair=tuple(r["pair"]);x=np.concatenate([data[n]["x"][r["variant"]] for n in names]);y=np.concatenate([data[n]["labels"] for n in names]);fit=np.isin(y,pair);m=model(r["alpha"]);m.fit(x[fit],y[fit]);pred,mar=infer(m,target["x"][r["variant"]]);q=np.isin(target["base"],pair)&(pred!=target["base"])&(mar>=r["threshold"]);out[q]=pred[q];changed|=q;details.append({**r,"target_changes":int(q.sum())})
 return out,changed,details
def main():
 protocol=np.load(P205);mp={v:(int(y),int(x)) for v,y,x in zip(protocol["sample_ids"].astype(str),protocol["labels"],protocol["prediction"])};splits=load_splits();allids=np.load(R/"runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1/sample_ids.npy").astype(str) if (R/"runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1/sample_ids.npy").exists() else None;raw=np.load(TRAIN,mmap_mode="r");# Train cache follows canonical protocol order.
 from p90_teacher_common import load_protocol
 tids=load_protocol().sample_ids.astype(str);pos={v:i for i,v in enumerate(tids)};data={}
 for n in S:
  ids=splits[n].sample_ids.astype(str);rows=np.asarray([pos[v] for v in ids]);data[n]={"ids":ids,"labels":np.asarray([mp[v][0] for v in ids]),"base":np.asarray([mp[v][1] for v in ids]),"users":splits[n].users.astype(str),"x":{v:unit(raw[rows][:,ix]).reshape(len(rows),-1) for v,ix in SUBSETS.items()}}
 report={"stage":"P226_raw_VJEPA_pair_specialists","status":"complete","protocol":{"base":"P205","raw_frozen_tokens":True,"source_crossfit_recipe":True,"held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];rules,audit=select(data,src);out,m,details=apply(data,src,data[held],rules);outs[held]=out;bc=int(np.sum(data[held]["base"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"rules":details,"source_audit":audit,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(m.sum()),"rescue":int(np.sum(m&(data[held]["base"]!=data[held]["labels"])&(out==data[held]["labels"]))),"harm":int(np.sum(m&(data[held]["base"]==data[held]["labels"])&(out!=data[held]["labels"])))}};print(json.dumps({"held":held,**report["cohorts"][held]["held"]}),flush=True)
 labels=np.concatenate([data[n]["labels"] for n in S]);base=np.concatenate([data[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(base==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p205":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S],"gap_to_0.91":int(np.ceil(.91*len(labels))-cor)}
 testids=np.load(TEST/"sample_ids.npy").astype(str);traw=np.load(TEST/"features.npy",mmap_mode="r");official=np.load(H/"runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz")["sample_ids"].astype(str);opos={v:i for i,v in enumerate(official)};avail=np.zeros(len(official),bool);x={}
 for v,ix in SUBSETS.items():
  z=np.zeros((len(official),len(ix)*1024),np.float32);rows=np.asarray([opos[q] for q in testids]);z[rows]=unit(traw[:,ix]).reshape(len(testids),-1);x[v]=z;avail[rows]=True
 target={"base":cp(P205T),"x":x};rules,audit=select(data,list(S));tout,m,details=apply(data,list(S),target,rules);m&=avail;tout[~avail]=target["base"][~avail];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p226_raw_vjepa_pairs.csv";io.write_submission(sub,io.read_rows(P89),tout);report["test"]={"rules":details,"changes_vs_p205":int(np.sum(tout!=target["base"])),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=np.concatenate([data[n]["ids"] for n in S]),labels=labels,base_prediction=base,prediction=out,test_sample_ids=official,test_base_prediction=target["base"],test_prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps({"aggregate":report["aggregate"],"test":report["test"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
