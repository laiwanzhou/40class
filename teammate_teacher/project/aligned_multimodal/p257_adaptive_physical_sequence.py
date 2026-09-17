"""Outer-safe decoder/emission configuration selection over P255 posterior."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from audit_p87_sequence_decoder import DecoderConfig,decode_sessions,fit_transition_model,align_metadata,build_sessions
from p117_transductive_multicandidate_router import load_candidate_splits
from p139_soft_sequence_gate import sessions_for,gate_features,select_gate
H=Path(__file__).resolve().parent;O=H/"runs/p257_adaptive_physical_sequence_v1";SOURCE=H/"runs/p255_repeat_augmented_physical_group_v1/predictions.npz";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";S=("H1_selection","H2_confirmation","H3_independent_fold0");CONFIGS=tuple((e,t) for e in (.5,.75,1.) for t in (.2,.3,.4))
def dec(t):return DecoderConfig(30.,t,1.,50)
def emission(p,b,w):
 x=w*np.asarray(p,float)
 if w<1:x[np.arange(len(b)),b]+=1-w
 x/=x.sum(1,keepdims=True);return np.log(np.clip(x,1e-9,1))
def source_recipe(data,parts,names,cfg):
 e,t=cfg;ids=np.concatenate([parts[n]["ids"] for n in names]);y=np.concatenate([parts[n]["labels"] for n in names]);b=np.concatenate([parts[n]["base"] for n in names]);p=np.concatenate([parts[n]["prob"] for n in names]);sessions=sessions_for(data,ids,names);tr=fit_transition_model(y,sessions,40,1.);seq=decode_sessions(emission(p,b,e),sessions,tr,dec(t));f=gate_features(p,b,seq);g=select_gate(b,seq,f,y);q=(seq!=b)&(f[:,g["score_index"]]>=g["threshold"]);out=b.copy();out[q]=seq[q];per=[];off=0
 for n in names:
  z=slice(off,off+len(parts[n]["ids"]));per.append(int(np.sum(out[z]==y[z])-np.sum(b[z]==y[z])));off+=len(parts[n]["ids"])
 return tr,g,per
def apply(data,part,name,cfg,tr,g):
 e,t=cfg;sessions=sessions_for(data,part["ids"],[name]);seq=decode_sessions(emission(part["prob"],part["base"],e),sessions,tr,dec(t));f=gate_features(part["prob"],part["base"],seq);q=(seq!=part["base"])&(f[:,g["score_index"]]>=g["threshold"]);out=part["base"].copy();out[q]=seq[q];return out,q
def main():
 data=load_candidate_splits();z=np.load(SOURCE);parts={n:{"ids":data[n].split.sample_ids.astype(str),"labels":data[n].split.labels.astype(int),"base":z[f"{n}_held_prediction"].astype(int),"prob":z[f"{n}_held_probability"].astype(float)} for n in S};report={"stage":"P257_outer_safe_adaptive_physical_sequence","status":"complete","protocol":{"source":"P255","config_grid":[list(x) for x in CONFIGS],"held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={};fixed={cfg:{} for cfg in CONFIGS}
 for held in S:
  src=[n for n in S if n!=held];cand=[]
  for cfg in CONFIGS:
   tr,g,per=source_recipe(data,parts,src,cfg);out,q=apply(data,parts[held],held,cfg,tr,g);net=int(np.sum(out==parts[held]["labels"])-np.sum(parts[held]["base"]==parts[held]["labels"]));fixed[cfg][held]=(out,net,g);cand.append((min(per),sum(per),g["net"],-g["harm"],cfg,tr,g,per,out,q))
  best=max(cand);cfg,tr,g,per,out,q=best[4:10];outs[held]=out;bc=int(np.sum(parts[held]["base"]==parts[held]["labels"]));cor=int(np.sum(out==parts[held]["labels"]));report["cohorts"][held]={"source":src,"config":list(cfg),"source_cohort_nets":per,"source_gate":g,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(q.sum())}};print(json.dumps({"held":held,"config":cfg,"source":per,"result":report["cohorts"][held]["held"]}),flush=True)
 labels=np.concatenate([parts[n]["labels"] for n in S]);base=np.concatenate([parts[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(base==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p255":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S],"p256_fixed_correct":2204}
 ranked=[]
 for cfg in CONFIGS:
  nets=[fixed[cfg][n][1] for n in S];ranked.append((min(nets),sum(nets),cfg,nets))
 best=max(ranked);cfg,nets=best[2],best[3];ids=np.concatenate([parts[n]["ids"] for n in S]);sessions=sessions_for(data,ids,list(S));tr=fit_transition_model(labels,sessions,40,1.);seq=decode_sessions(emission(np.concatenate([parts[n]["prob"] for n in S]),base,cfg[0]),sessions,tr,dec(cfg[1]));f=gate_features(np.concatenate([parts[n]["prob"] for n in S]),base,seq);g=select_gate(base,seq,f,labels);tids=z["sample_ids"].astype(str);tb=z["prediction"].astype(int);tp=z["probability"].astype(float);meta=align_metadata(H/"data/p85_recording_metadata/test_recording_metadata.csv",tids);ts=build_sessions(np.arange(len(tids)),meta,30.,"anonymous_date");tseq=decode_sessions(emission(tp,tb,cfg[0]),ts,tr,dec(cfg[1]));tf=gate_features(tp,tb,tseq);q=(tseq!=tb)&(tf[:,g["score_index"]]>=g["threshold"]);tout=tb.copy();tout[q]=tseq[q];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p257_adaptive_sequence.csv";io.write_submission(sub,io.read_rows(P89),tout);report["final_config"]={"config":list(cfg),"outer_fold_nets":nets,"gate":g};report["test"]={"changes_vs_p255":int(q.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=tids,base_prediction=z["base_prediction"],group_prediction=tb,probability=tp.astype(np.float32),prediction=tout,route=q,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps({"aggregate":report["aggregate"],"final":report["final_config"],"test":report["test"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
