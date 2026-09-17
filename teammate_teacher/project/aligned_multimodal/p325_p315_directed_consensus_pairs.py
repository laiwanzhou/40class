"""Directed Top-5 consensus pairs over P315/P310."""
from __future__ import annotations
import itertools,json
from pathlib import Path
import numpy as np
from p316_topk_consensus_uncertainty_gate import build,S
H=Path(__file__).resolve().parent;O=H/"runs/p325_p315_directed_consensus_pairs_v1";RULES=((0,),(1,),(2,),(0,1),(0,2),(1,2))
def features(p):return np.column_stack((p["votes"]/7.,p["gap"],-p["base_conf"]))
def cat(ps):return {k:np.concatenate([p[k] for p in ps]) for k in ps[0]}
def select(p):
 x=features(p);gain=(p["alt"]==p["labels"]).astype(int)-(p["base"]==p["labels"]).astype(int);rules={};audit=[]
 for a,b in sorted(set(zip(p["base"][p["in_top5"]&(p["alt"]!=p["base"])].tolist(),p["alt"][p["in_top5"]&(p["alt"]!=p["base"])].tolist()))):
  pair=p["in_top5"]&(p["base"]==a)&(p["alt"]==b);best=None
  for rule in RULES:
   grids=[np.concatenate(([np.inf],np.unique(np.quantile(x[pair,j],np.linspace(0,1,11)))[::-1],[-np.inf])) for j in rule]
   for ts in itertools.product(*grids):
    m=pair.copy()
    for j,t in zip(rule,ts):m&=x[:,j]>=t
    r=int(np.sum(m&(gain>0)));h=int(np.sum(m&(gain<0)));peru={u:int(gain[m&(p["users"]==u)].sum()) for u in sorted(set(p["users"].tolist()))};perc={c:int(gain[m&(p["cohort"]==c)].sum()) for c in sorted(set(p["cohort"].tolist()))};row={"base":int(a),"candidate":int(b),"rule":list(rule),"thresholds":[float(t) for t in ts],"selected":int(m.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(peru.values()),"minimum_cohort_gain":min(perc.values()),"positive_users":int(sum(v>0 for v in peru.values())),"per_user":peru,"per_cohort":perc};key=(h==0,row["minimum_user_gain"]>=0,row["minimum_cohort_gain"]>=0,row["positive_users"]>=2,row["net"],r,-row["selected"])
    if best is None or key>best[0]:best=(key,row)
  row=best[1];row["eligible"]=row["rescue"]>=2 and row["harm"]==0 and row["minimum_user_gain"]>=0 and row["minimum_cohort_gain"]>=0 and row["positive_users"]>=2;audit.append(row)
  if row["eligible"]:rules[(a,b)]=row
 return rules,audit
def apply(p,rules):
 x=features(p);m=np.zeros(len(p["base"]),bool)
 for (a,b),r in rules.items():
  q=p["in_top5"]&(p["base"]==a)&(p["alt"]==b)
  for j,t in zip(r["rule"],r["thresholds"]):q&=x[:,j]>=t
  m|=q
 out=p["base"].copy();out[m]=p["alt"][m];return out,m
def main():
 print("P325 tests source-safe directed consensus pairs for the P315/P310 base.",flush=True);parts=build()
 for n in S:parts[n]["cohort"]=np.full(len(parts[n]["labels"]),n,object)
 report={"stage":"P325_P315_directed_consensus_pairs","status":"complete","protocol":{"base":"P310/P315 OOF","proposal":"seven-expert plurality restricted to P244 Top-5","source_pair_thresholds":True,"eligibility":"rescue>=2,harm=0,min_user>=0,min_cohort>=0,positive_users>=2","held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];rules,audit=select(cat([parts[n] for n in src]));out,m=apply(parts[held],rules);outs[held]=out;y=parts[held]["labels"];b=parts[held]["base"];report["cohorts"][held]={"source":src,"eligible_rules":list(rules.values()),"source_audit_count":len(audit),"held":{"rows":len(y),"base_correct":int(np.sum(b==y)),"correct":int(np.sum(out==y)),"net":int(np.sum(out==y)-np.sum(b==y)),"changed":int(m.sum()),"rescue":int(np.sum(m&(b!=y)&(out==y))),"harm":int(np.sum(m&(b==y)&(out!=y)))}}
 labels=np.concatenate([parts[n]["labels"] for n in S]);base=np.concatenate([parts[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p310":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: directed Top-5 consensus pairs over P310.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps({"aggregate":report["aggregate"],"cohorts":{n:report["cohorts"][n]["held"] for n in S},"rules":{n:report["cohorts"][n]["eligible_rules"] for n in S}},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
