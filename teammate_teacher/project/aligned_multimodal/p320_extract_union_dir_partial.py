"""Extract matched P238 tokens for seven recoverable Depth+IR rows missing Thermal."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from p302_extract_union_full_physical import videomaev2_features,internvideo_features
H=Path(__file__).resolve().parent;UNION=H/"data/six_modality_audit/train_union_manifest.csv";MAIN=H/"data/manifest.csv";O=H/"runs/p320_union_dir_partial_features_v1"
def read(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def rows():
 main={(r["class_name"],r["user_id"],r["trial_id"]) for r in read(MAIN)};out=[]
 for s in read(UNION):
  key=(s["class_name"],s["user_id"],s["trial_id"])
  if key in main or s["usable_pattern"] not in {"110100","110110"}:continue
  out.append({"sample_id":s["sample_id"],"source_id":s["sample_id"],"user_id":s["user_id"],"class_id":s["class_id"],"ir_dir":s["ir_path"],"depth_dir":s["depth_color_path"]})
 out.sort(key=lambda r:r["sample_id"])
 if len(out)!=7:raise RuntimeError(len(out))
 return out
def main():
 selected=rows();y=np.asarray([int(r["class_id"]) for r in selected],int);ids=np.asarray([r["sample_id"] for r in selected]);users=np.asarray([r["user_id"] for r in selected]);a=videomaev2_features(selected,y,"ir");b=internvideo_features(selected,y);d=videomaev2_features(selected,y,"depth");t=np.zeros((len(y),3,768),np.float16);O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"features.npz",sample_ids=ids,labels=y,users=users,ir_videomaev2=a,ir_internvideo2=b,depth_videomaev2=d,thermal_videomaev2=t,thermal_available=np.zeros(len(y),bool));report={"stage":"P320_union_DIR_partial_features","status":"complete","rows":len(y),"classes":sorted(set(y.tolist())),"users":sorted(set(users.tolist())),"tokens":{"ir_videomaev2":list(a.shape),"ir_internvideo2":list(b.shape),"depth_videomaev2":list(d.shape),"thermal_videomaev2":list(t.shape)},"missing_modality":"thermal","test_rows_loaded":0,"test_labels_read":False};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
