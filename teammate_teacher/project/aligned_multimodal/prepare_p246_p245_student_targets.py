"""Build compact-Student targets from the exact P241 physical sequence teacher."""
from __future__ import annotations
import csv,json,re,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;SRC=H/"runs/p245_p244_soft_sequence_gate_v1/submission_soft_sequence.csv";OFF=R/"Testing/test.csv";O=H/"runs/p246_p245_student_targets_v1"
def read(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 off=read(OFF);src=read(SRC);paths=[r["path"] for r in off]
 if len(src)!=405 or [r["path"] for r in src]!=paths:raise RuntimeError("P246 order mismatch")
 pred=np.asarray([int(r["prediction"]) for r in src]);ids=[]
 for p in paths:
  m=re.search(r"(SM_test_\d{4})",p)
  if not m:raise RuntimeError(p)
  ids.append(m.group(1))
 prob=np.full((405,40),.0005,np.float32);prob[np.arange(405),pred]=.9805;O.mkdir(parents=True,exist_ok=True);out=O/"student_test_targets.npz";np.savez_compressed(out,sample_ids=np.asarray(ids),target_mask=np.ones(405,bool),emission_probability=prob,structured_distillation_probability=prob,structured_confidence=np.full(405,.9805,np.float32),emission_prediction=pred,structured_distillation_prediction=pred);report={"stage":"P246_P245_student_targets","status":"complete","source":str(SRC.resolve()),"source_sha256":sha(SRC),"targets":str(out.resolve()),"targets_sha256":sha(out),"rows":405,"test_labels_read":False};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

