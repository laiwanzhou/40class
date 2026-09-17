"""Audit and package the P205 compact Student."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;OFF=R/"Testing/test.csv";T=H/"runs/p205_fixed_p203_p150_residual_v1/submission_p205_fixed_residual.csv";RAW=H/"runs/p205_student_test_predictions_v1/submission_p87s_student_raw.csv";DEC=H/"runs/p205_student_test_predictions_v1/submission_p87s_student_decoded.csv";CK=H/"runs/p205_student_test_adapt_e40_v1/unified_student.pt";P199=H/"runs/p199_final_kaggle_candidate_v1/submission_p199_compact_student_raw.csv";O=H/"runs/p205_final_kaggle_candidate_v1"
def read(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def pr(r):return np.asarray([int(x["prediction"]) for x in r])
def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 off=read(OFF);t=read(T);r=read(RAW);d=read(DEC);old=read(P199);paths=[x["path"] for x in off]
 for n,x in (("teacher",t),("raw",r),("decoded",d),("P199",old)):
  if len(x)!=405 or [q["path"] for q in x]!=paths:raise RuntimeError(n)
 if not np.array_equal(pr(t),pr(r)):raise RuntimeError("student mismatch")
 size=CK.stat().st_size
 if size>=100000000:raise RuntimeError("size")
 O.mkdir(parents=True,exist_ok=True);out=O/"submission_p205_compact_student_raw.csv"
 with out.open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=("path","prediction"));w.writeheader();w.writerows(r)
 report={"stage":"P205_compact_Student_Kaggle_candidate","status":"complete_awaiting_score","validation":{"correct":2197,"rows":2470,"accuracy":2197/2470,"fold_nets_vs_P203":[0,1,3],"P199_correct":2187},"student":{"checkpoint":str(CK.resolve()),"bytes":size,"mb_decimal":size/1e6,"sha256":sha(CK),"under_100MB":True,"teacher_agreement":1.0,"decoder_rejected":True,"decoder_changes":int(np.sum(pr(d)!=pr(r)))},"submission":{"path":str(out.resolve()),"sha256":sha(out),"rows":405,"changes_vs_P199":int(np.sum(pr(r)!=pr(old))),"test_labels_read":False}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
