"""Deploy the P311 Top-3/Top-5 pair salvage as an overlay on P310."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p173_vjepa_augmented_group_teacher import build_test_bank
from p255_repeat_augmented_physical_group import al
from p309_union_repeat_group_test import SOURCES as DEPLOY_SOURCES
from p311_p245_topk_pair_reranker_oof import SPLITS,KS,CS,build,cat,source_pairs,fit_predict,choose,eligible
H=Path(__file__).resolve().parent;O=H/"runs/p312_topk_salvage_teacher_v2";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";P244=H/"runs/p244_dual_physical_group_v1/predictions.npz";P245=H/"runs/p245_p244_soft_sequence_gate_v1/predictions.npz";P310=H/"runs/p310_union_repeat_precedence_teacher_v1";P311=H/"runs/p311_p245_topk_pair_reranker_oof_v3/oof_predictions.npz";OFF=H.parent/"Testing/test.csv"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def cp(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def full_rules(data):
 source=cat([data[n] for n in SPLITS]);best=None
 for k in KS:
  pairs=source_pairs(source,k)
  for c in CS:
   rules=[];audit=[]
   for a,b in sorted(pairs):
    proposal=source["base"].copy();score=np.full(len(proposal),-np.inf);ok=True;off=0
    for held in SPLITS:
     src=[n for n in SPLITS if n!=held];pr=fit_predict(cat([data[n] for n in src]),data[held],a,b,c)
     if pr is None:ok=False;break
     n=len(data[held]["labels"]);proposal[off:off+n]=np.where(pr>=.5,b,a);score[off:off+n]=np.abs(pr-.5)*2;off+=n
    if not ok:continue
    row=choose(score,proposal,source,a,b,k);row["C"]=c;audit.append(row)
    positive_cohorts=sum(v>0 for v in row["per_cohort"].values())
    if row["rescue"]>=3 and row["harm"]==0 and row["minimum_user_gain"]>=0 and row["minimum_cohort_gain"]>=0 and positive_cohorts>=2 and row["positive_users"]>=2:rules.append(row)
   key=(sum(r["net"] for r in rules),sum(r["rescue"] for r in rules),-sum(r["harm"] for r in rules),-len(rules),-k,-c)
   if best is None or key>best[0]:best=(key,k,c,rules,audit)
 return best[1:]
def test_part(names):
 test=build_test_bank(names[:21],names[:24]);blocks=[]
 for op,ok,tp,tk,name in DEPLOY_SOURCES:
  z=np.load(tp);ids=z["test_sample_ids"] if "test_sample_ids" in z.files and len(z[tk])==len(z["test_sample_ids"]) else z["sample_ids"];p=al(z[tk],ids,test["ids"]);av=al(z["available"],ids,test["ids"]).astype(bool) if "available" in z.files else np.ones(len(p),bool);p[~av]=test["bank"][~av,0,:];blocks.append(p[:,None,:])
 test["bank"]=np.concatenate((test["bank"],*blocks),1);g=np.load(P244);s=np.load(P245);test["posterior"]=g["probability"].astype(np.float32);test["base"]=s["prediction"].astype(int);test["order"]=np.argsort(-test["posterior"],axis=1,kind="stable");return test
def main():
 data,names=build()
 for n in SPLITS:data[n]["cohort"]=np.full(len(data[n]["labels"]),n,object)
 k,c,rules,audit=full_rules(data);test=test_part(names);reranked=test["base"].copy();best_score=np.full(len(reranked),-np.inf);applied=np.zeros(len(reranked),bool);source=cat([data[n] for n in SPLITS])
 for rule in rules:
  a,b=rule["pair"];pr=fit_predict(source,test,a,b,rule["C"])
  if pr is None:continue
  proposal=np.where(pr>=.5,b,a);score=np.abs(pr-.5)*2;q=eligible(test,a,b,rule["k"])&(proposal!=test["base"])&(score>=rule["threshold"])&(score>best_score);reranked[q]=proposal[q];best_score[q]=score[q];applied[q]=True
 p310=cp(P310/"submission_p310_union_repeat_precedence.csv");route=reranked!=test["base"];tout=p310.copy();tout[route]=reranked[route];o310=np.load(P310/"oof_predictions.npz");o311=np.load(P311);oout=o310["prediction"].copy();oroute=o311["prediction"]!=o311["base_prediction"];oout[oroute]=o311["prediction"][oroute];labels=o310["labels"];correct=int(np.sum(oout==labels));bc=int(np.sum(o310["prediction"]==labels));O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p312_topk_salvage.csv";io.write_submission(sub,io.read_rows(P89),tout);prob=np.full((len(tout),40),.0005,np.float32);prob[np.arange(len(tout)),tout]=.9805;targets=O/"student_test_targets.npz";np.savez_compressed(targets,sample_ids=test["ids"],target_mask=np.ones(len(tout),bool),emission_probability=prob,structured_distillation_probability=prob,structured_confidence=np.full(len(tout),.9805,np.float32),emission_prediction=tout,structured_distillation_prediction=tout);report={"stage":"P312_TopK_salvage_teacher_v2","status":"complete","protocol":{"base":"P310","candidate_set":"P244 Top-3/Top-5 selected on outer-cross-predicted OOF only","selected_k":k,"selected_C":c,"rule_gate":"at least 3 rescues, zero harm, no cohort regression, at least two positive cohorts and two positive subjects","held_or_test_labels_used_for_deployment":False,"test_labels_read":False,"user_id_used_as_feature":False},"rules":rules,"rule_audit_count":len(audit),"validation":{"rows":len(labels),"p310_correct":bc,"correct":correct,"accuracy":correct/len(labels),"net_vs_p310":correct-bc},"test":{"rows":len(tout),"pair_reranker_changes_vs_p245":int(route.sum()),"changes_vs_p310":int(np.sum(tout!=p310)),"changed_rows_vs_p310_zero_based":np.flatnonzero(tout!=p310).tolist(),"submission":str(sub.resolve()),"submission_sha256":sha(sub),"targets":str(targets.resolve()),"targets_sha256":sha(targets),"test_labels_read":False}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
