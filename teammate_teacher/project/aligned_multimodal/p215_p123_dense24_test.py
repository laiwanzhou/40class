"""Refit the frozen P123 dense24 Ridge heads on all Train and predict Test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p123_vjepa_dense24_full_oof import load_features
from p96_vjepa2_dense24_teacher_h1h2 import fit_scores
from p90_teacher_common import load_protocol,softmax
from train_p46_videomae_head import l2_normalize,row_standardize

H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p215_p123_dense24_test_v1";T=R/"runs/p171_vjepa2_dense24_test_v1";VB=R/"runs/p90_videomaev2_distilled_test_v1/complete_features.npz";VI=R/"runs/p90_internvideo2_l_k400_test_v1/complete_features.npz"
def main():
 protocol=load_protocol();train=load_features();ids=np.load(T/"sample_ids.npy").astype(str);dense=l2_normalize(np.asarray(np.load(T/"features.npy",mmap_mode="r"),np.float32));act=row_standardize(np.asarray(np.load(T/"ssv2_logits.npy",mmap_mode="r"),np.float32));dg=l2_normalize(dense.reshape(len(dense),8,3,1024).mean(2));ag=row_standardize(act.reshape(len(act),8,3,174).mean(2));df=dg.reshape(len(dense),-1);da=np.concatenate((df,ag.reshape(len(dense),-1)),1).astype(np.float32)
 with np.load(VB) as v, np.load(VI) as i:
  if not np.array_equal(ids,v["sample_ids"].astype(str)) or not np.array_equal(ids,i["sample_ids"].astype(str)):raise RuntimeError("P215 Test order mismatch")
  old=np.concatenate((l2_normalize(v["features"].astype(np.float32)).reshape(len(ids),-1),l2_normalize(i["features"].astype(np.float32)).reshape(len(ids),-1)),1).astype(np.float32)
 test={"dense24_group8":df,"dense24_group8_ssv2":da,"old_ir_plus_dense24_group8_ssv2":np.concatenate((old,da),1).astype(np.float32)};saved={"sample_ids":ids};report={"stage":"P215_P123_dense24_allTrain_to_Test","status":"complete","test_labels_read":False,"variants":{}}
 for n,(x,a) in train.items():
  logits=fit_scores(x,protocol.labels,test[n],a);saved[n+"_logits"]=logits.astype(np.float32);saved[n+"_probability"]=softmax(logits).astype(np.float32);report["variants"][n]={"alpha":a,"train_rows":len(x),"test_rows":len(logits),"feature_dim":x.shape[1]}
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"test_predictions.npz",**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
