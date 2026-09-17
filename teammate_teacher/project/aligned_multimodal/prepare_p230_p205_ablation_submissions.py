"""Prepare P205 Test ablations anchored bit-for-bit to scored P199."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;P199=H/"runs/p199_final_kaggle_candidate_v1/submission_p199_compact_student_raw.csv";P205=H/"runs/p205_final_kaggle_candidate_v1/submission_p205_compact_student_raw.csv";O=H/"runs/p230_p205_ablation_submissions_v1"
SEQUENCE_ROWS=(179,237,245,350);RESIDUAL_ROWS=(99,264)
def read(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def write(path,rows,pred):
 with path.open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=("path","prediction"));w.writeheader()
  for r,v in zip(rows,pred,strict=True):w.writerow({"path":r["path"],"prediction":int(v)})
def main():
 a=read(P199);b=read(P205);pa=np.asarray([int(r["prediction"]) for r in a]);pb=np.asarray([int(r["prediction"]) for r in b]);O.mkdir(parents=True,exist_ok=True);report={"stage":"P230_P205_ablation_submissions","status":"prepared_not_yet_compact_student_packaged","anchor":{"path":str(P199.resolve()),"kaggle_score":.89552,"sha256":sha(P199)},"variants":{}}
 for name,rows in (("sequence4",SEQUENCE_ROWS),("residual2",RESIDUAL_ROWS)):
  p=pa.copy();p[list(rows)]=pb[list(rows)];out=O/f"submission_p230_{name}.csv";write(out,a,p);report["variants"][name]={"path":str(out.resolve()),"sha256":sha(out),"changes_vs_p199":int(np.sum(p!=pa)),"rows_zero_based":list(rows),"predictions":[{"p199":int(pa[i]),"variant":int(p[i])} for i in rows],"test_labels_read":False}
 (O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
