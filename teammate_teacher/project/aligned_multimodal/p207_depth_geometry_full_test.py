"""Complete P99 Depth geometry descriptors, strict OOF heads and Test inference."""
from __future__ import annotations
import csv,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from audit_yolo11_pose_skeleton import safe_name
from p99_depth_geometry_descriptor import extract_trial,canonical_sample_id
from p90_teacher_common import load_protocol
H=Path(__file__).resolve().parent;O=H/"runs/p207_depth_geometry_full_test_v1";TRAIN_MAN=H/"data/six_modality_audit/train_union_manifest.csv";TEST_MAN=H/"data/p46_test_union_manifest.csv";TRAIN_ROI=H/"runs/p29_dir_multiscale_roi_full/trial_roi_cache";TEST_ROI=H/"runs/p29_dir_multiscale_roi_test/trial_roi_cache";EXIST=H/"runs/p99_depth_geometry_d1_descriptor_h1_v1/descriptors.npz";REGIONS=("full_body","left_hand","right_hand","hand_workspace")
def read(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def worker(item):
 row,roi=item
 try:v=extract_trial(row,roi,REGIONS);return v["geometry"],v["depth_surface"],None
 except Exception as e:return None,None,str(e)
def fit_head(x,y,fold,test):
 logits=np.zeros((len(y),40));reports=[]
 for k in range(3):
  held=fold==k;m=make_pipeline(StandardScaler(),RidgeClassifier(alpha=1000,class_weight="balanced",solver="lsqr",tol=1e-4));m.fit(x[~held],y[~held]);s=m.decision_function(x[held]);logits[np.flatnonzero(held)[:,None],m[-1].classes_.astype(int)[None,:]]=s;reports.append({"fold":k,"correct":int(np.sum(logits[held].argmax(1)==y[held])),"rows":int(held.sum())})
 m=make_pipeline(StandardScaler(),RidgeClassifier(alpha=1000,class_weight="balanced",solver="lsqr",tol=1e-4));m.fit(x,y);s=m.decision_function(test);tl=np.zeros((len(test),40));tl[:,m[-1].classes_.astype(int)]=s;return logits,tl,reports
def main():
 protocol=load_protocol();train_rows=[r for r in read(TRAIN_MAN) if r["depth_color_usable"]=="1"];lookup={canonical_sample_id(r):r for r in train_rows};order=protocol.sample_ids.astype(str);existing=np.load(EXIST);emap={v:i for i,v in enumerate(existing["sample_ids"].astype(str))};gd=existing["geometry"].shape[1];dd=existing["depth_surface"].shape[1];geom=np.zeros((len(order),gd),np.float32);depth=np.zeros((len(order),dd),np.float32);pending=[]
 for i,v in enumerate(order):
  if v in emap:geom[i]=existing["geometry"][emap[v]];depth[i]=existing["depth_surface"][emap[v]]
  else:
   row=lookup[v];pending.append((i,row,TRAIN_ROI/safe_name(row["sample_id"]).with_suffix(".npz")))
 print(json.dumps({"stage":"P207_train_extract","cached":len(order)-len(pending),"pending":len(pending)}),flush=True)
 with ThreadPoolExecutor(max_workers=8) as ex:
  for j,(item,res) in enumerate(zip(pending,ex.map(worker,[(r,p) for _,r,p in pending])),1):
   i,_,_=item;g,d,e=res
   if e:raise RuntimeError(e)
   geom[i]=g;depth[i]=d
   if j%100==0:print(json.dumps({"train_done":j,"train_pending":len(pending)}),flush=True)
 test_rows=read(TEST_MAN);tids=np.asarray([r["official_sample_id"] for r in test_rows]);tg=np.zeros((len(tids),gd),np.float32);td=np.zeros((len(tids),dd),np.float32);available=np.zeros(len(tids),bool);items=[]
 for i,r in enumerate(test_rows):
  roi=TEST_ROI/safe_name(r["sample_id"]).with_suffix(".npz")
  if r["depth_color_usable"]=="1" and roi.is_file():items.append((i,r,roi))
 with ThreadPoolExecutor(max_workers=8) as ex:
  for j,(item,res) in enumerate(zip(items,ex.map(worker,[(r,p) for _,r,p in items])),1):
   i,_,_=item;g,d,e=res
   if e:raise RuntimeError(e)
   tg[i]=g;td[i]=d;available[i]=True
   if j%100==0:print(json.dumps({"test_done":j,"test_total":len(items)}),flush=True)
 variants={"geometry":(geom,tg),"depth_surface":(depth,td),"depth_geometry":(np.concatenate((geom,depth),1),np.concatenate((tg,td),1))};saved={"sample_ids":order,"labels":protocol.labels,"fold_id":protocol.fold_id,"test_sample_ids":tids,"test_available":available};report={"stage":"P207_depth_geometry_full_OOF_Test","status":"complete","protocol":{"alpha":1000,"strict_subject_folds":True,"test_labels_read":False},"variants":{}}
 for n,(x,tx) in variants.items():
  ol,tl,fr=fit_head(x,protocol.labels,protocol.fold_id,tx);op=np.exp(ol-ol.max(1,keepdims=True));op/=op.sum(1,keepdims=True);tp=np.exp(tl-tl.max(1,keepdims=True));tp/=tp.sum(1,keepdims=True);saved[n+"_probability"]=op.astype(np.float32);saved["test_"+n+"_probability"]=tp.astype(np.float32);report["variants"][n]={"dim":x.shape[1],"correct":int(np.sum(op.argmax(1)==protocol.labels)),"accuracy":float(np.mean(op.argmax(1)==protocol.labels)),"folds":fr}
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"predictions.npz",**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
