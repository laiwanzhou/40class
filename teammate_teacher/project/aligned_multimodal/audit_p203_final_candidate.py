"""Audit and package the P203 compact Student candidate."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;OFF=R/"Testing/test.csv";TEACH=H/"runs/p203_current_champion_v1/submission_p203_current_champion.csv";RAW=H/"runs/p203_student_test_predictions_v1/submission_p87s_student_raw.csv";DEC=H/"runs/p203_student_test_predictions_v1/submission_p87s_student_decoded.csv";CK=H/"runs/p203_student_test_adapt_e40_v1/unified_student.pt";O=H/"runs/p203_final_kaggle_candidate_v1"
def read(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def pred(r):return np.asarray([int(x["prediction"]) for x in r])
def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 off=read(OFF);t=read(TEACH);r=read(RAW);d=read(DEC);paths=[x["path"] for x in off]
 for n,x in (("teacher",t),("raw",r),("decoded",d)):
  if len(x)!=405 or [q["path"] for q in x]!=paths:raise RuntimeError(n+" order")
 if not np.array_equal(pred(t),pred(r)):raise RuntimeError("P203 Student differs")
 size=CK.stat().st_size
 if size>=100000000:raise RuntimeError("P203 >100MB")
 O.mkdir(parents=True,exist_ok=True);out=O/"submission_p203_compact_student_raw.csv"
 with out.open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=("path","prediction"));w.writeheader();w.writerows(r)
 report={"stage":"P203_compact_Student_Kaggle_candidate","status":"complete_awaiting_score","validation":{"correct":2193,"rows":2470,"accuracy":2193/2470,"fold_nets_vs_P89":[26,30,20],"P199_correct":2187},"student":{"checkpoint":str(CK.resolve()),"checkpoint_bytes":size,"checkpoint_mb_decimal":size/1e6,"checkpoint_sha256":sha(CK),"strict_under_100MB":True,"teacher_agreement_405":1.0,"decoder_rejected":True,"decoder_changed_rows":int(np.sum(pred(d)!=pred(r)))},"submission":{"path":str(out.resolve()),"sha256":sha(out),"rows":405,"official_order_exact":True,"changes_vs_P199":5,"test_labels_read":False}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
