"""Audit/package the compact P310 union-repeat precedence Student."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;OFF=R/"Testing/test.csv";T=H/"runs/p310_union_repeat_precedence_teacher_v1/submission_p310_union_repeat_precedence.csv";RAW=H/"runs/p310_student_test_predictions_v1/submission_p87s_student_raw.csv";DEC=H/"runs/p310_student_test_predictions_v1/submission_p87s_student_decoded.csv";CK=H/"runs/p310_student_test_adapt_e40_v1/unified_student.pt";P199=H/"runs/p199_final_kaggle_candidate_v1/submission_p199_compact_student_raw.csv";P246=H/"runs/p246_final_kaggle_candidate_v1/submission_p246_compact_student_raw.csv";P280=H/"runs/p280_final_kaggle_candidate_v1/submission_p280_compact_student_raw.csv";O=H/"runs/p315_final_kaggle_candidate_v1"
def read(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def pred(x):return np.asarray([int(r["prediction"]) for r in x])
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 off=read(OFF);paths=[r["path"] for r in off];rows={n:read(p) for n,p in (("teacher",T),("raw",RAW),("decoded",DEC),("p199",P199),("p246",P246),("p280",P280))}
 for n,r in rows.items():
  if len(r)!=405 or [x["path"] for x in r]!=paths:raise RuntimeError(n)
 if not np.array_equal(pred(rows["teacher"]),pred(rows["raw"])):raise RuntimeError("P315 Student differs from P310")
 size=CK.stat().st_size
 if size>=100000000:raise RuntimeError(size)
 O.mkdir(parents=True,exist_ok=True);out=O/"submission_p315_compact_student_raw.csv"
 with out.open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=("path","prediction"));w.writeheader();w.writerows(rows["raw"])
 report={"stage":"P315_compact_union_repeat_precedence_candidate","status":"complete_awaiting_kaggle_score","validation":{"p310_correct":2211,"rows":2470,"accuracy":2211/2470,"p270_correct":2210,"net_vs_p270":1,"fold_nets_vs_p270":[0,0,1]},"student":{"checkpoint":str(CK.resolve()),"bytes":size,"mb_decimal":size/1e6,"sha256":sha(CK),"under_100MB":True,"teacher_agreement":1.0,"decoder_rejected":False,"decoder_changes":int(np.sum(pred(rows["decoded"])!=pred(rows["raw"])))},"submission":{"path":str(out.resolve()),"sha256":sha(out),"rows":405,"changes_vs_p199":int(np.sum(pred(rows["raw"])!=pred(rows["p199"]))),"changes_vs_p246":int(np.sum(pred(rows["raw"])!=pred(rows["p246"]))),"changes_vs_p280":int(np.sum(pred(rows["raw"])!=pred(rows["p280"]))),"changed_rows_vs_p280_zero_based":np.flatnonzero(pred(rows["raw"])!=pred(rows["p280"])).tolist(),"test_labels_read":False}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
