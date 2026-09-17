"""Source-safe directed rollback from P180 to P89 safe."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p193_deployable_candidate_ranker import bank

H=Path(__file__).resolve().parent;O=H/"runs/p196_source_safe_directed_rollback_v1";P180=H/"runs/p180_sequence_micro_teacher_v1/oof_predictions.npz";P180T=H/"runs/p180_sequence_micro_teacher_v1/submission_p180_sequence_micro.csv";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def cp(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def rules(cur,safe,labels,users):
 gain=(safe==labels).astype(int)-(cur==labels).astype(int);out=[]
 for a,b in sorted(set(zip(cur[cur!=safe].tolist(),safe[cur!=safe].tolist()))):
  m=(cur==a)&(safe==b);r=int(np.sum(m&(gain>0)));h=int(np.sum(m&(gain<0)));per={u:int(gain[m&(users==u)].sum()) for u in sorted(set(users.tolist()))};row={"current":int(a),"safe":int(b),"selected":int(m.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(per.values()),"per_user":per,"eligible":r>=2 and h==0 and min(per.values())>=0};out.append(row)
 return {(r["current"],r["safe"]) for r in out if r["eligible"]},out
def apply(cur,safe,selected):
 m=np.asarray([(int(a),int(b)) in selected for a,b in zip(cur,safe)]);out=cur.copy();out[m]=safe[m];return out,m
def main():
 tr,_=bank();p=np.load(P180);mp={v:(int(y),int(x)) for v,y,x in zip(p["sample_ids"].astype(str),p["labels"],p["prediction"])};data={n:{"ids":tr[n]["ids"],"safe":tr[n]["base"].astype(int),"current":np.asarray([mp[v][1] for v in tr[n]["ids"]]),"labels":np.asarray([mp[v][0] for v in tr[n]["ids"]]),"users":tr[n]["users"]} for n in S};report={"stage":"P196_source_safe_directed_rollback","status":"complete","protocol":{"alternative":"P89 safe","directed_pairs_source_only":True,"held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];sel,audit=rules(np.concatenate([data[n]["current"] for n in src]),np.concatenate([data[n]["safe"] for n in src]),np.concatenate([data[n]["labels"] for n in src]),np.concatenate([data[n]["users"] for n in src]));out,m=apply(data[held]["current"],data[held]["safe"],sel);outs[held]=out;bc=int(np.sum(data[held]["current"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"eligible_pairs":[list(x) for x in sorted(sel)],"source_audit":audit,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(m.sum()),"rescue":int(np.sum(m&(data[held]["current"]!=data[held]["labels"])&(out==data[held]["labels"]))),"harm":int(np.sum(m&(data[held]["current"]==data[held]["labels"])&(out!=data[held]["labels"])))}}
 labels=np.concatenate([data[n]["labels"] for n in S]);cur=np.concatenate([data[n]["current"] for n in S]);out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(cur==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p180":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};sel,audit=rules(cur,np.concatenate([data[n]["safe"] for n in S]),labels,np.concatenate([data[n]["users"] for n in S]));tb=cp(P180T);ts=cp(P89);tout,m=apply(tb,ts,sel);O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p196_directed_rollback.csv";io.write_submission(sub,io.read_rows(P89),tout);report["test"]={"eligible_pairs":[list(x) for x in sorted(sel)],"oof_audit":audit,"changes":int(m.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",base_prediction=tb,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
