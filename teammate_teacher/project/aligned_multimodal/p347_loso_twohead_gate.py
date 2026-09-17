"""Nested source-LOSO two-head rescue/harm gate over P346 candidates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p117_transductive_multicandidate_router import load_candidate_splits
from p191_source_truth_calibrated_p150_distillation import choose
from p346_distributionally_robust_rescue_harm_gate import H,S,part,cat,train_pair,score,P310,P244,P307,P255,P306,P328,P336,P344,P279,P278
O=H/"runs/p347_loso_twohead_gate_v1"
def subset(p,m):return {k:v[m] for k,v in p.items()}
def main():
 print("P347 tests nested source leave-one-subject-out rescue/harm gating.",flush=True);data=load_candidate_splits();z={"p310":np.load(P310),"p244":np.load(P244),"p307":np.load(P307),"p255":np.load(P255),"p306":np.load(P306),"p328":np.load(P328),"p336":np.load(P336),"p344":np.load(P344),"p279":np.load(P279),"p278":np.load(P278)};parts={n:part(data,n,z) for n in S};report={"stage":"P347_nested_LOSO_twohead_gate","status":"complete","protocol":{"base":"P310","candidate_pool":"P346 five candidates","heads":["rescue","harm"],"source_validation":"leave-one-subject-out over both source cohorts","held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];source=cat([parts[n] for n in src]);coh=np.concatenate([np.full(len(parts[n]["labels"]),n,object) for n in src]);best=None
  for c in (.003,.01,.03,.1):
   prop=np.zeros(len(source["labels"]),int);sc=np.full(len(prop),-np.inf)
   for u in sorted(set(source["users"].tolist())):
    te=source["users"]==u;tr=~te;pr,ss=score([train_pair(subset(source,tr),c)],subset(source,te));prop[te]=pr;sc[te]=ss
   gain=(prop==source["labels"]).astype(int)-(source["base"]==source["labels"]).astype(int);sel=choose(sc,gain,prop!=source["base"],source["users"],coh);key=(sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],sel["rescue"],-sel["harm"],-sel["changed"],-c);cand=(key,c,sel)
   if best is None or cand[0]>best[0]:best=cand
  c,sel=best[1:];prop,hs=score([train_pair(source,c)],parts[held]);route=(prop!=parts[held]["base"])&(hs>=sel["threshold"]);out=parts[held]["base"].copy();out[route]=prop[route];outs[held]=out;y=parts[held]["labels"];base=parts[held]["base"];report["cohorts"][held]={"source":src,"C":c,"source_selection":sel,"held":{"rows":len(y),"base_correct":int(np.sum(base==y)),"correct":int(np.sum(out==y)),"net":int(np.sum(out==y)-np.sum(base==y)),"changed":int(route.sum()),"rescue":int(np.sum(route&(base!=y)&(out==y))),"harm":int(np.sum(route&(base==y)&(out!=y)))}}
 labels=np.concatenate([parts[n]["labels"] for n in S]);base=np.concatenate([parts[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p310":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: nested LOSO two-head gate over P346 candidate pool.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
