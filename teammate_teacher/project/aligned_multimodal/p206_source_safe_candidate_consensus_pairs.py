"""Source-safe directed candidate-consensus rules on top of P205."""
from __future__ import annotations
import csv,itertools,json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p193_deployable_candidate_ranker import bank
from p173_vjepa_augmented_group_teacher import build_test_bank
H=Path(__file__).resolve().parent;O=H/"runs/p206_source_safe_candidate_consensus_pairs_v1";P128=H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz";P128T=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz";P205=H/"runs/p205_fixed_p203_p150_residual_v1/oof_predictions.npz";P205T=H/"runs/p205_fixed_p203_p150_residual_v1/submission_p205_fixed_residual.csv";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def al(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def cp(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def descriptors(prob,base):
 hard=prob.argmax(2);vote=np.stack([(hard==c).mean(1) for c in range(40)],1);mean=prob.mean(1);proposal=vote.argmax(1);r=np.arange(len(base));x=np.column_stack((vote[r,proposal],vote[r,proposal]-vote[r,base],mean[r,proposal]-mean[r,base],mean[r,proposal],mean[r,base]));return proposal,x
RULES=((0,),(1,),(2,),(0,1),(0,2))
def select(base,proposal,x,labels,users):
 gain=(proposal==labels).astype(int)-(base==labels).astype(int);out=[];rules={}
 for a,b in sorted(set(zip(base[proposal!=base].tolist(),proposal[proposal!=base].tolist()))):
  pair=(base==a)&(proposal==b);best=None
  for rule in RULES:
   grids=[]
   for j in rule:grids.append(np.concatenate(([np.inf],np.unique(np.quantile(x[pair,j],np.linspace(0,1,21)))[::-1],[-np.inf])))
   for ts in itertools.product(*grids):
    m=pair.copy()
    for j,t in zip(rule,ts):m&=x[:,j]>=t
    r=int(np.sum(m&(gain>0)));h=int(np.sum(m&(gain<0)));per={u:int(gain[m&(users==u)].sum()) for u in sorted(set(users.tolist()))};row={"base":int(a),"candidate":int(b),"rule":list(rule),"thresholds":[float(t) for t in ts],"selected":int(m.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(per.values()),"per_user":per};key=(h==0,row["minimum_user_gain"]>=0,row["net"],r,-row["selected"])
    if best is None or key>best[0]:best=(key,row)
  row=best[1];row["eligible"]=row["rescue"]>=2 and row["harm"]==0 and row["minimum_user_gain"]>=0;out.append(row)
  if row["eligible"]:rules[(a,b)]=row
 return rules,out
def apply(base,proposal,x,rules):
 m=np.zeros(len(base),bool)
 for (a,b),r in rules.items():
  q=(base==a)&(proposal==b)
  for j,t in zip(r["rule"],r["thresholds"]):q&=x[:,j]>=t
  m|=q
 out=base.copy();out[m]=proposal[m];return out,m
def main():
 tr,names=bank();p=np.load(P205);mp={v:(int(y),int(x)) for v,y,x in zip(p["sample_ids"].astype(str),p["labels"],p["prediction"])};data={}
 for n in S:
  q=tr[n];base=np.asarray([mp[v][1] for v in q["ids"]]);labels=np.asarray([mp[v][0] for v in q["ids"]]);proposal,x=descriptors(q["bank"],base);data[n]={**q,"base":base,"labels":labels,"proposal":proposal,"x":x}
 report={"stage":"P206_source_safe_candidate_consensus_pairs","status":"complete","protocol":{"base":"P205","directed_pair_source_only":True,"eligibility":"rescue>=2,harm=0,min_user>=0","held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];rules,audit=select(np.concatenate([data[n]["base"] for n in src]),np.concatenate([data[n]["proposal"] for n in src]),np.concatenate([data[n]["x"] for n in src]),np.concatenate([data[n]["labels"] for n in src]),np.concatenate([data[n]["users"] for n in src]));out,m=apply(data[held]["base"],data[held]["proposal"],data[held]["x"],rules);outs[held]=out;bc=int(np.sum(data[held]["base"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"eligible_rules":list(rules.values()),"source_audit":audit,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(m.sum()),"rescue":int(np.sum(m&(data[held]["base"]!=data[held]["labels"])&(out==data[held]["labels"]))),"harm":int(np.sum(m&(data[held]["base"]==data[held]["labels"])&(out!=data[held]["labels"])))}}
 labels=np.concatenate([data[n]["labels"] for n in S]);base=np.concatenate([data[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(base==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p205":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]}
 rules,audit=select(base,np.concatenate([data[n]["proposal"] for n in S]),np.concatenate([data[n]["x"] for n in S]),labels,np.concatenate([data[n]["users"] for n in S]));test=build_test_bank(names[:21],names[:24]);pt=np.load(P128T);test["bank"]=np.concatenate((test["bank"],al(pt["probabilities"],pt["sample_ids"],test["ids"])[:,None,:]),axis=1);tb=cp(P205T);proposal,x=descriptors(test["bank"],tb);tout,m=apply(tb,proposal,x,rules);O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p206_candidate_pairs.csv";io.write_submission(sub,io.read_rows(P89),tout);report["test"]={"eligible_rules":list(rules.values()),"oof_audit":audit,"changes_vs_p205":int(m.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=test["ids"],base_prediction=tb,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
