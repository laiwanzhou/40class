"""Strict Top-K local transition repair over the P245 final prediction."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from audit_p87_sequence_decoder import fit_transition_model
from p117_transductive_multicandidate_router import load_candidate_splits
from p139_soft_sequence_gate import sessions_for
from p191_source_truth_calibrated_p150_distillation import choose
H=Path(__file__).resolve().parent;O=H/"runs/p314_topk_local_transition_repair_v1";P244=H/"runs/p244_dual_physical_group_v1/predictions.npz";P245=H/"runs/p245_p244_soft_sequence_gate_v1/predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0");KS=(3,5);WEIGHTS=(.1,.2,.3,.5,1.)
def part(data,g,s,n):
 q=data[n].split;return {"ids":q.sample_ids.astype(str),"labels":q.labels.astype(int),"users":q.users.astype(str),"base":s[f"{n}_held_prediction"].astype(int),"prob":g[f"{n}_held_probability"].astype(float)}
def propose(p,base,sessions,tr,k,w):
 n=len(base);score=np.log(np.clip(p,1e-9,1));local=score.copy()
 for session in sessions:
  for j,row in enumerate(session):
   left=tr.start_log_probability if j==0 else tr.bigram_log_probability[base[session[j-1]]]
   right=tr.end_log_probability if j+1==len(session) else tr.bigram_log_probability[:,base[session[j+1]]]
   local[row]+=w*(left+right)
 order=np.argsort(-p,axis=1,kind="stable")[:,:k];cand=local[np.arange(n)[:,None],order];ix=cand.argmax(1);proposal=order[np.arange(n),ix];gap=local[np.arange(n),proposal]-local[np.arange(n),base];return proposal,gap
def cross_source(data,parts,src,k,w):
 out={}
 for train,target in ((src[0],src[1]),(src[1],src[0])):
  ts=sessions_for(data,parts[train]["ids"],[train]);tr=fit_transition_model(parts[train]["labels"],ts,40,1.);hs=sessions_for(data,parts[target]["ids"],[target]);out[target]=propose(parts[target]["prob"],parts[target]["base"],hs,tr,k,w)
 return np.concatenate([out[n][0] for n in src]),np.concatenate([out[n][1] for n in src])
def main():
 data=load_candidate_splits();g=np.load(P244);s=np.load(P245);parts={n:part(data,g,s,n) for n in S};report={"stage":"P314_TopK_local_transition_repair","status":"complete","protocol":{"base":"P245","candidate_k":list(KS),"weights":list(WEIGHTS),"neighbor_labels":"P245 predictions only","transition_fit":"source labels only","held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];labels=np.concatenate([parts[n]["labels"] for n in src]);base=np.concatenate([parts[n]["base"] for n in src]);users=np.concatenate([parts[n]["users"] for n in src]);coh=np.concatenate([np.full(len(parts[n]["labels"]),n,object) for n in src]);best=None
  for k in KS:
   for w in WEIGHTS:
    proposal,gap=cross_source(data,parts,src,k,w);gain=(proposal==labels).astype(int)-(base==labels).astype(int);sel=choose(gap,gain,proposal!=base,users,coh);key=(sel["minimum_user_gain"]>=0,sel["minimum_cohort_gain"]>=0,sel["net"],sel["rescue"],-sel["harm"],-sel["changed"],-k,-w)
    if best is None or key>best[0]:best=(key,k,w,sel)
  k,w,sel=best[1:];ids=np.concatenate([parts[n]["ids"] for n in src]);y=np.concatenate([parts[n]["labels"] for n in src]);sessions=sessions_for(data,ids,src);tr=fit_transition_model(y,sessions,40,1.);hs=sessions_for(data,parts[held]["ids"],[held]);proposal,gap=propose(parts[held]["prob"],parts[held]["base"],hs,tr,k,w);route=(proposal!=parts[held]["base"])&(gap>=sel["threshold"]);out=parts[held]["base"].copy();out[route]=proposal[route];outs[held]=out;hy=parts[held]["labels"];hb=parts[held]["base"];report["cohorts"][held]={"source":src,"k":k,"weight":w,"source_selection":sel,"held":{"rows":len(hy),"base_correct":int(np.sum(hb==hy)),"correct":int(np.sum(out==hy)),"net":int(np.sum(out==hy)-np.sum(hb==hy)),"changed":int(route.sum()),"rescue":int(np.sum(route&(hb!=hy)&(out==hy))),"harm":int(np.sum(route&(hb==hy)&(out!=hy)))}}
 labels=np.concatenate([parts[n]["labels"] for n in S]);base=np.concatenate([parts[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p245":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
