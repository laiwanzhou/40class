"""P270 sequence with P307 group precedence on P307's own group routes."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;O=H/"runs/p310_union_repeat_precedence_teacher_v1";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";P255=H/"runs/p255_repeat_augmented_physical_group_v1/predictions.npz";P270=H/"runs/p270_fixed_emission065_transition045_v1/predictions.npz";P307=H/"runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz";P309=H/"runs/p309_union_repeat_group_test_v1/predictions.npz";P246=H/"runs/p246_final_kaggle_candidate_v1/submission_p246_compact_student_raw.csv";P280=H/"runs/p280_final_kaggle_candidate_v1/submission_p280_compact_student_raw.csv";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def cp(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def main():
 data=load_candidate_splits();a=np.load(P255);b=np.load(P270);c=np.load(P307);outs={};folds=[]
 for name in S:
  old=a[f"{name}_held_prediction"].astype(int);seq=b[f"{name}_held_prediction"].astype(int);new=c[f"{name}_group_prediction"].astype(int);route=new!=old;out=seq.copy();out[route]=new[route];outs[name]=out;y=data[name].split.labels.astype(int);folds.append({"cohort":name,"rows":len(y),"p270_correct":int(np.sum(seq==y)),"correct":int(np.sum(out==y)),"net_vs_p270":int(np.sum(out==y)-np.sum(seq==y)),"p307_precedence_rows":int(route.sum()),"changes_vs_p270":int(np.sum(out!=seq))})
 labels=np.concatenate([data[n].split.labels.astype(int) for n in S]);pred=np.concatenate([outs[n] for n in S]);base=np.concatenate([b[f"{n}_held_prediction"].astype(int) for n in S]);correct=int(np.sum(pred==labels));bc=int(np.sum(base==labels));ta=np.load(P255);tb=np.load(P270);tc=np.load(P309);ids=tb["sample_ids"].astype(str)
 if not np.array_equal(ids,ta["sample_ids"].astype(str)) or not np.array_equal(ids,tc["sample_ids"].astype(str)):raise RuntimeError("Test order mismatch")
 old=ta["prediction"].astype(int);seq=tb["prediction"].astype(int);new=tc["prediction"].astype(int);route=new!=old;tout=seq.copy();tout[route]=new[route];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p310_union_repeat_precedence.csv";io.write_submission(sub,io.read_rows(P89),tout);prob=np.full((len(tout),40),.0005,np.float32);prob[np.arange(len(tout)),tout]=.9805;targets=O/"student_test_targets.npz";np.savez_compressed(targets,sample_ids=ids,target_mask=np.ones(len(ids),bool),emission_probability=prob,structured_distillation_probability=prob,structured_confidence=np.full(len(ids),.9805,np.float32),emission_prediction=tout,structured_distillation_prediction=tout);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=pred,**{f"{n}_held_prediction":outs[n] for n in S});report={"stage":"P310_union_repeat_precedence_teacher","status":"complete","protocol":{"base":"P270","precedence":"where P307 group differs from P255 group, use P307 group; otherwise keep P270","rule_uses_labels":False,"test_labels_read":False,"user_id_used_as_feature":False},"validation":{"rows":len(labels),"p270_correct":bc,"correct":correct,"accuracy":correct/len(labels),"net_vs_p270":correct-bc,"folds":folds},"test":{"rows":len(ids),"p307_precedence_rows":int(route.sum()),"changes_vs_p270":int(np.sum(tout!=seq)),"changes_vs_p246":int(np.sum(tout!=cp(P246))),"changes_vs_p280":int(np.sum(tout!=cp(P280))),"changed_rows_vs_p270_zero_based":np.flatnonzero(tout!=seq).tolist(),"submission":str(sub.resolve()),"submission_sha256":sha(sub),"targets":str(targets.resolve()),"targets_sha256":sha(targets),"test_labels_read":False}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
