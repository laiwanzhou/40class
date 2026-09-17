"""Build exact P122 hand-object relation features and heads on readable Test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from ultralytics import YOLO

import p122_hand_object_relation_teacher as p122


HERE=Path(__file__).resolve().parent
OUTPUT=HERE/"runs/p186_p122_relation_test_v1"
PIXELS=HERE/"runs/p87s_test_pixel_cache_t16_r160_v1"
P29=HERE/"runs/p29_dir_multiscale_roi_test/trial_roi_cache"
P28=HERE/"runs/p28_adaptive_ir_pose_skeleton_test/trial_cache"
OOF=HERE/"runs/p122_hand_object_relation_teacher_v1/oof_predictions.npz"


def rows():
    with (PIXELS/"rows.csv").open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))


def main():
    source_rows=rows(); images=np.load(PIXELS/"images.npy",mmap_mode="r"); source_indices=np.load(PIXELS/"source_frame_indices.npy",mmap_mode="r")
    valid=[]
    for i,row in enumerate(source_rows):
        rel=Path(row["source_id"]).with_suffix(".npz")
        if (P29/rel).is_file() and (P28/rel).is_file():valid.append(i)
    if len(valid)!=401:raise RuntimeError(f"P186 readable rows changed: {len(valid)}")
    p122.P29=P29; p122.P28=P28
    keypoints=np.zeros((len(valid),2,len(p122.FRAME_POSITIONS),17,3),np.float32)
    for out,row_index in enumerate(valid):keypoints[out],_=p122.trial_geometry(source_rows[row_index]["source_id"],source_indices[row_index])
    raw=np.zeros((len(valid),2,len(p122.FRAME_POSITIONS),p122.FRAME_DIM),np.float16)
    model=YOLO("assets/models/yolo11n.pt")
    records=[(out,row_index,w,t) for out,row_index in enumerate(valid) for w in range(2) for t in range(len(p122.FRAME_POSITIONS))]
    for start in range(0,len(records),64):
        batch_records=records[start:start+64]; batch=[]
        for out,row_index,w,t in batch_records:
            image=images[row_index,w,p122.FRAME_POSITIONS[t],2];batch.append(np.repeat(image[:,:,None],3,axis=2))
        results=model.predict(source=batch,imgsz=320,conf=.03,iou=.70,device=0,half=True,verbose=False,batch=64)
        for record,result in zip(batch_records,results):
            out,_,w,t=record;raw[out,w,t]=p122.frame_features(result,keypoints[out,w,t]).astype(np.float16)
        if start%(64*10)==0:print(json.dumps({"stage":"P186_cache","encoded":min(start+len(batch_records),len(records)),"total":len(records)}),flush=True)
    variants={"pose_only":p122.descriptor(raw[...,:p122.POSE_DIM].astype(np.float32)),"object_only":p122.descriptor(raw[...,p122.POSE_DIM:p122.POSE_DIM+p122.OBJECT_DIM].astype(np.float32)),"relations_only":p122.descriptor(raw[...,p122.POSE_DIM+p122.OBJECT_DIM:].astype(np.float32)),"all":p122.descriptor(raw.astype(np.float32))}
    train=np.load(OOF,allow_pickle=False); labels=train["labels"].astype(np.int64); saved={"sample_ids":np.asarray([source_rows[i]["sample_id"] for i in valid])}
    for name,test_feature in variants.items():
        train_feature=train[f"{name}_features"].astype(np.float32)
        estimator=ExtraTreesClassifier(n_estimators=600,max_depth=14,min_samples_leaf=3,max_features="sqrt",class_weight="balanced",random_state=12200,n_jobs=-1)
        estimator.fit(train_feature,labels); probability=np.zeros((len(test_feature),40),np.float64); probability[:,estimator.classes_.astype(int)]=estimator.predict_proba(test_feature)
        saved[f"{name}_features"]=test_feature.astype(np.float16);saved[f"{name}_probability"]=probability.astype(np.float32)
    OUTPUT.mkdir(parents=True,exist_ok=True);np.save(OUTPUT/"frame_relation_features.npy",raw);np.savez_compressed(OUTPUT/"test_predictions.npz",**saved)
    report={"stage":"P186_P122_relation_all2914_to_Test","status":"complete","rows":len(valid),"variants":list(variants),"test_labels_read":False}
    (OUTPUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
