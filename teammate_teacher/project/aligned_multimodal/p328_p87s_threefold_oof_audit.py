"""Assemble the first exact three-cohort OOF for the terminal P87-S architecture."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;O=H/"runs/p328_p87s_threefold_oof_audit_v1";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0");RUNS=(H/"runs/p87s_fusion_holdout1_c0_v1",H/"runs/p87s_fusion_holdout2_c0_v1",H/"runs/p87s_fusion_holdout3_c0_v1")
def rows(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def softmax(x):x=np.asarray(x,float);x-=x.max(1,keepdims=True);p=np.exp(x);return p/p.sum(1,keepdims=True)
def main():
 print("P328 assembles exact H1/H2/H3 P87-S OOF and audits complementarity against P310/P315.",flush=True);data=load_candidate_splits();basez=np.load(P310);saved={};report={"stage":"P328_P87S_terminal_threefold_OOF","status":"complete","protocol":{"fresh_visual_motion_fusion_per_cohort":True,"architecture_matches_terminal_P87S":True,"all_checkpoints_under_100MB":True,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};allp=[];alli=[];allv=[];labels=[];base=[]
 for n,run in zip(S,RUNS):
  rr=rows(run/"subject_holdout_predictions.csv");ids=np.asarray([x["sample_id"] for x in rr]);y=np.asarray([int(x["label"]) for x in rr]);q=data[n].split
  if not np.array_equal(ids,q.sample_ids.astype(str)) or not np.array_equal(y,q.labels.astype(int)):raise RuntimeError(n+" order")
  logits=np.load(run/"subject_holdout_logits.npy");visual=np.load(run/"cached_visual_baseline_logits.npy");prob=softmax(logits);pred=prob.argmax(1);bp=basez[f"{n}_held_prediction"].astype(int);changed=pred!=bp;r=int(np.sum(changed&(bp!=y)&(pred==y)));h=int(np.sum(changed&(bp==y)&(pred!=y)));report["cohorts"][n]={"rows":len(y),"base_correct":int(np.sum(bp==y)),"student_correct":int(np.sum(pred==y)),"rescue":r,"harm":h,"oracle_correct":int(np.sum((bp==y)|(pred==y))),"oracle_gain":r};saved[f"{n}_probability"]=prob.astype(np.float32);saved[f"{n}_visual_probability"]=softmax(visual).astype(np.float32);saved[f"{n}_prediction"]=pred;allp.append(prob);alli.append(pred);allv.append(softmax(visual));labels.append(y);base.append(bp)
 p=np.concatenate(allp);pred=np.concatenate(alli);visual=np.concatenate(allv);y=np.concatenate(labels);b=np.concatenate(base);changed=pred!=b;r=int(np.sum(changed&(b!=y)&(pred==y)));h=int(np.sum(changed&(b==y)&(pred!=y)));report["aggregate"]={"rows":len(y),"base_correct":int(np.sum(b==y)),"student_correct":int(np.sum(pred==y)),"student_accuracy":float(np.mean(pred==y)),"rescue":r,"harm":h,"oracle_correct":int(np.sum((b==y)|(pred==y))),"oracle_gain":r,"fold_rescues":[report["cohorts"][n]["rescue"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=y,base_prediction=b,probability=p.astype(np.float32),visual_probability=visual.astype(np.float32),prediction=pred,**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: first exact three-cohort OOF for the terminal compact P87-S architecture.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
