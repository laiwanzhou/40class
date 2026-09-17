"""Outer-safe multi-posterior Dirichlet-style Top-5 calibrator over P310."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from p117_transductive_multicandidate_router import load_candidate_splits
from p191_source_truth_calibrated_p150_distillation import choose
H=Path(__file__).resolve().parent;O=H/"runs/p327_multiview_dirichlet_topk_calibrator_v1";P244=H/"runs/p244_dual_physical_group_v1/predictions.npz";P307=H/"runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz";P255=H/"runs/p255_repeat_augmented_physical_group_v1/predictions.npz";P306=H/"runs/p306_union_full_repeat_physical_oof_v1/oof_predictions.npz";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0");CS=(.003,.01,.03,.1)
def model(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1600,solver="lbfgs",class_weight="balanced"))
def fullprob(m,x):
 p=m.predict_proba(x);out=np.zeros((len(x),40));out[:,m.classes_.astype(int)]=p;return out
def part(data,n,z244,z307,z255,z306,z310,pos):
 q=data[n].split;ids=q.sample_ids.astype(str);p244=z244[f"{n}_held_probability"].astype(float);p307=z307[f"{n}_group_probability"].astype(float);p255=z255[f"{n}_held_probability"].astype(float);p306=z306["probability"][[pos[x] for x in ids]].astype(float);x=np.concatenate([np.log(np.clip(p244,1e-7,1)),np.log(np.clip(p307,1e-7,1)),np.log(np.clip(p255,1e-7,1)),np.log(np.clip(p306,1e-7,1))],1).astype(np.float32);return {"x":x,"labels":q.labels.astype(int),"users":q.users.astype(str),"base":z310[f"{n}_held_prediction"].astype(int),"top5":np.argsort(-p244,axis=1)[:,:5]}
def score(p,base,alt,kind):
 r=np.arange(len(base));s=np.sort(p,axis=1)
 if kind=="gap_base":return p[r,alt]-p[r,base]
 if kind=="confidence":return p[r,alt]
 return s[:,-1]-s[:,-2]
def main():
 print("P327 tests a small Dirichlet-style calibration matrix over four strict OOF posterior views.",flush=True);data=load_candidate_splits();a=np.load(P244);b=np.load(P307);c=np.load(P255);d=np.load(P306);e=np.load(P310);pos={x:i for i,x in enumerate(d["sample_ids"].astype(str))};parts={n:part(data,n,a,b,c,d,e,pos) for n in S};report={"stage":"P327_multiview_Dirichlet_TopK_calibrator","status":"complete","protocol":{"base":"P310","inputs":["P244","P307 group","P255 group","P306 physical"],"candidate_restricted_to_P244_top5":True,"source_inner_cross_prediction":True,"held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];sx=np.concatenate([parts[n]["x"] for n in src]);sy=np.concatenate([parts[n]["labels"] for n in src]);sb=np.concatenate([parts[n]["base"] for n in src]);su=np.concatenate([parts[n]["users"] for n in src]);sc=np.concatenate([np.full(len(parts[n]["labels"]),n,object) for n in src]);stop=np.concatenate([parts[n]["top5"] for n in src]);best=None
  for cv in CS:
   predprob={}
   for trn,ten in ((src[0],src[1]),(src[1],src[0])):predprob[ten]=fullprob(model(cv).fit(parts[trn]["x"],parts[trn]["labels"]),parts[ten]["x"])
   p=np.concatenate([predprob[n] for n in src]);masked=np.full_like(p,-np.inf);masked[np.arange(len(p))[:,None],stop]=p[np.arange(len(p))[:,None],stop];alt=masked.argmax(1);gain=(alt==sy).astype(int)-(sb==sy).astype(int)
   for kind in ("gap_base","confidence","margin"):
    sel=choose(score(p,sb,alt,kind),gain,alt!=sb,su,sc);key=(sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],sel["rescue"],-sel["harm"],-sel["changed"],-cv);cand=(key,cv,kind,sel)
    if best is None or cand[0]>best[0]:best=cand
  cv,kind,sel=best[1:];m=model(cv).fit(sx,sy);hp=fullprob(m,parts[held]["x"]);masked=np.full_like(hp,-np.inf);masked[np.arange(len(hp))[:,None],parts[held]["top5"]]=hp[np.arange(len(hp))[:,None],parts[held]["top5"]];alt=masked.argmax(1);route=(alt!=parts[held]["base"])&(score(hp,parts[held]["base"],alt,kind)>=sel["threshold"]);out=parts[held]["base"].copy();out[route]=alt[route];outs[held]=out;y=parts[held]["labels"];base=parts[held]["base"];report["cohorts"][held]={"source":src,"C":cv,"score":kind,"source_selection":sel,"held":{"rows":len(y),"base_correct":int(np.sum(base==y)),"correct":int(np.sum(out==y)),"net":int(np.sum(out==y)-np.sum(base==y)),"changed":int(route.sum()),"rescue":int(np.sum(route&(base!=y)&(out==y))),"harm":int(np.sum(route&(base==y)&(out!=y)))}}
 labels=np.concatenate([parts[n]["labels"] for n in S]);base=np.concatenate([parts[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p310":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: Top-5 Dirichlet-style calibration of four strict OOF posterior views.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
