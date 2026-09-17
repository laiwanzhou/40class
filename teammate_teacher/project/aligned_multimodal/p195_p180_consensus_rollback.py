"""Outer-cross-fit consensus rollback from P180 to the original P89 safe label."""
from __future__ import annotations
import itertools,json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p193_deployable_candidate_ranker import bank
from p173_vjepa_augmented_group_teacher import build_test_bank

H=Path(__file__).resolve().parent;O=H/"runs/p195_p180_consensus_rollback_v1";P128=H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz";P128T=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz";P177=H/"runs/p177_p128_vjepa_group_teacher_v1/predictions.npz";P180=H/"runs/p180_sequence_micro_teacher_v1/oof_predictions.npz";P180T=H/"runs/p180_sequence_micro_teacher_v1/submission_p180_sequence_micro.csv";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def al(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def csvpred(p):
 import csv
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def feat(prob,safe,current,group):
 hard=prob.argmax(2);rows=np.arange(len(safe));idx=np.arange(prob.shape[1]);mean=prob.mean(1);mx=prob.max(1);return np.column_stack((np.mean(hard==safe[:,None],1),np.mean(hard==current[:,None],1),mean[rows,safe]-mean[rows,current],mx[rows,safe]-mx[rows,current],prob[:,0,:][rows,safe],group[rows,safe]-group[rows,current])).astype(float)
RULES=((0,),(2,),(5,),(0,2),(0,5),(0,4))
def grid(v,dim):
 if dim==1:return np.unique(v)
 return np.unique(np.quantile(v,np.linspace(0,1,25)))
def select(x,safe,current,labels,users):
 disagreement=safe!=current;gain=(safe==labels).astype(int)-(current==labels).astype(int);best=None
 for rule in RULES:
  grids=[np.concatenate(([np.inf],grid(x[disagreement,j],len(rule))[::-1],[-np.inf])) for j in rule]
  for values in itertools.product(*grids):
   m=disagreement.copy()
   for j,t in zip(rule,values):m&=x[:,j]>=t
   per={u:int(gain[m&(users==u)].sum()) for u in sorted(set(users.tolist()))};r=int(np.sum(m&(gain>0)));h=int(np.sum(m&(gain<0)));row={"rule":list(rule),"thresholds":[float(t) for t in values],"changed":int(m.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(per.values()),"per_user":per};key=(row["minimum_user_gain"]>=0,row["net"],r,-h,-row["changed"])
   if best is None or key>best[0]:best=(key,row)
 return best[1]
def apply(x,safe,current,sel):
 m=safe!=current
 for j,t in zip(sel["rule"],sel["thresholds"]):m&=x[:,j]>=t
 out=current.copy();out[m]=safe[m];return out,m
def main():
 tr,names=bank();p180=np.load(P180);pm={v:(int(y),int(p)) for v,y,p in zip(p180["sample_ids"].astype(str),p180["labels"],p180["prediction"])};p177=np.load(P177);data={}
 for n in S:
  q=tr[n];safe=q["base"].astype(int);cur=np.asarray([pm[v][1] for v in q["ids"]]);labels=np.asarray([pm[v][0] for v in q["ids"]]);group=p177[f"{n}_held_probability"].astype(float);data[n]={**q,"safe":safe,"current":cur,"labels":labels,"x":feat(q["bank"],safe,cur,group)}
 report={"stage":"P195_P180_consensus_rollback","status":"complete","protocol":{"class_agnostic":True,"alternative":"P89 safe","source_only_threshold":True,"held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];x=np.concatenate([data[n]["x"] for n in src]);safe=np.concatenate([data[n]["safe"] for n in src]);cur=np.concatenate([data[n]["current"] for n in src]);labels=np.concatenate([data[n]["labels"] for n in src]);users=np.concatenate([data[n]["users"] for n in src]);sel=select(x,safe,cur,labels,users);out,m=apply(data[held]["x"],data[held]["safe"],data[held]["current"],sel);outs[held]=out;bc=int(np.sum(data[held]["current"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"selected":sel,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(m.sum()),"rescue":int(np.sum(m&(data[held]["current"]!=data[held]["labels"])&(out==data[held]["labels"]))),"harm":int(np.sum(m&(data[held]["current"]==data[held]["labels"])&(out!=data[held]["labels"])))}}
 labels=np.concatenate([data[n]["labels"] for n in S]);cur=np.concatenate([data[n]["current"] for n in S]);out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(cur==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p180":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]}
 # final selection on all OOF, apply Test
 x=np.concatenate([data[n]["x"] for n in S]);safe=np.concatenate([data[n]["safe"] for n in S]);users=np.concatenate([data[n]["users"] for n in S]);sel=select(x,safe,cur,labels,users);test=build_test_bank(names[:21],names[:24]);pt=np.load(P128T);test["bank"]=np.concatenate((test["bank"],al(pt["probabilities"],pt["sample_ids"],test["ids"])[:,None,:]),axis=1);tsafe=csvpred(P89);tcur=csvpred(P180T);tx=feat(test["bank"],tsafe,tcur,p177["probability"].astype(float));tout,m=apply(tx,tsafe,tcur,sel);O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p195_consensus_rollback.csv";io.write_submission(sub,io.read_rows(P89),tout);report["test"]={"oof_selection":sel,"changes_vs_p180":int(m.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=test["ids"],base_prediction=tcur,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
