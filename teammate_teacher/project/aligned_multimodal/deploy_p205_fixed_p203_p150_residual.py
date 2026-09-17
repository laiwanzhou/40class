"""Freeze fixed C=0.1 margin P150 residual over P203."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p204_p203_calibrated_p150_residual import enrich,testx
from p191_source_truth_calibrated_p150_distillation import model,route_score,choose
from p190_inductive_p150_meta_distillation import S,cat,P89T
H=Path(__file__).resolve().parent;O=H/"runs/p205_fixed_p203_p150_residual_v1"
def inner(data,src):
 out={}
 for tr,te in ((src[0],src[1]),(src[1],src[0])):
  m=model(.1);m.fit(data[tr]["x"],data[tr]["target"]);p=m.predict_proba(data[te]["x"]);f=np.zeros((len(p),40));f[:,m.classes_.astype(int)]=p;out[te]=f
 return np.concatenate([out[n] for n in src])
def main():
 data=enrich();report={"stage":"P205_fixed_P203_P150_residual","status":"complete","protocol":{"base":"P203","model":"Logistic C=0.1","score":"margin","outer_source_threshold":True,"held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];p=inner(data,src);labels=cat([data[n] for n in src],"labels");base=cat([data[n] for n in src],"base");users=cat([data[n] for n in src],"users");coh=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in src]);pr=p.argmax(1);gain=(pr==labels).astype(int)-(base==labels).astype(int);sel=choose(route_score(p,base,pr,"margin"),gain,pr!=base,users,coh);m=model(.1);m.fit(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"target"));hp=m.predict_proba(data[held]["x"]);f=np.zeros((len(hp),40));f[:,m.classes_.astype(int)]=hp;pr=f.argmax(1);mask=(pr!=data[held]["base"])&(route_score(f,data[held]["base"],pr,"margin")>=sel["threshold"]);out=data[held]["base"].copy();out[mask]=pr[mask];outs[held]=out;bc=int(np.sum(data[held]["base"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"source_threshold":sel,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(mask.sum())}}
 labels=cat([data[n] for n in S],"labels");base=cat([data[n] for n in S],"base");out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(base==labels));assert cor==2197,cor;report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p203":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S],"gap_to_0.91":2248-cor}
 probs=[]
 for held in S:
  src=[n for n in S if n!=held];m=model(.1);m.fit(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"target"));p=m.predict_proba(data[held]["x"]);f=np.zeros((len(p),40));f[:,m.classes_.astype(int)]=p;probs.append(f)
 p=np.concatenate(probs);pr=p.argmax(1);gain=(pr==labels).astype(int)-(base==labels).astype(int);users=cat([data[n] for n in S],"users");coh=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in S]);sel=choose(route_score(p,base,pr,"margin"),gain,pr!=base,users,coh);ids,tx,tbase=testx();m=model(.1);m.fit(cat([data[n] for n in S],"x"),cat([data[n] for n in S],"target"));tp=m.predict_proba(tx);f=np.zeros((len(tp),40));f[:,m.classes_.astype(int)]=tp;pr=f.argmax(1);mask=(pr!=tbase)&(route_score(f,tbase,pr,"margin")>=sel["threshold"]);tout=tbase.copy();tout[mask]=pr[mask];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p205_fixed_residual.csv";io.write_submission(sub,io.read_rows(P89T),tout);prob=np.full((405,40),.0005,np.float32);prob[np.arange(405),tout]=.9805;targets=O/"student_test_targets.npz";np.savez_compressed(targets,sample_ids=ids,target_mask=np.ones(405,bool),emission_probability=prob,structured_distillation_probability=prob,structured_confidence=np.full(405,.9805,np.float32),emission_prediction=tout,structured_distillation_prediction=tout);np.savez_compressed(O/"oof_predictions.npz",sample_ids=np.concatenate([data[n]["ids"] for n in S]),labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});report["test"]={"oof_threshold":sel,"changes_vs_p203":int(mask.sum()),"submission":str(sub.resolve()),"targets":str(targets.resolve()),"test_labels_read":False};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
