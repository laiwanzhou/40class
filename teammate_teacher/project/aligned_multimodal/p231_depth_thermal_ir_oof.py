"""Strict three-fold OOF for frozen IR + Depth/Thermal VideoMAEv2 representations."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p90_teacher_common import load_protocol,softmax
from p90_videomaev2_distilled_teacher import class_sample_weights
from train_p46_videomae_head import l2_normalize,make_model
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p231_depth_thermal_ir_oof_v1";IR1=R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz";IR2=R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz";D=R/"runs/p91_videomaev2_depth_fold0_v1/complete_features.npz";T=R/"runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz"
def flat(z):return l2_normalize(z["features"].astype(np.float32)).reshape(len(z["features"]),-1)
def fit(x,y,tr,va,alpha=3000.,power=.75):
 m=make_model(alpha);m.fit(x[tr],y[tr],ridge__sample_weight=class_sample_weights(y[tr],power));return m.decision_function(x[va])
def main():
 p=load_protocol();a=np.load(IR1);b=np.load(IR2);d=np.load(D);t=np.load(T)
 for z in (a,b,d,t):
  if not np.array_equal(z["sample_ids"].astype(str),p.sample_ids):raise RuntimeError("P231 order mismatch")
 ir=np.concatenate((flat(a),flat(b)),1);df=flat(d);tf=flat(t);variants={"ir_depth":np.concatenate((ir,df),1),"ir_thermal":np.concatenate((ir,tf),1),"ir_depth_thermal":np.concatenate((ir,df,tf),1)};saved={"sample_ids":p.sample_ids,"labels":p.labels,"users":p.users,"fold_id":p.fold_id};report={"stage":"P231_frozen_IR_depth_thermal_OOF","status":"complete","protocol":{"strict_subject_folds":True,"features_frozen":True,"alpha":3000.,"class_weight_power":.75,"held_fold_used_for_selection":False,"test_rows_loaded":0},"variants":{}}
 for n,x in variants.items():
  logits=np.zeros((len(p.labels),40),np.float64);folds=[]
  for k in range(3):
   tr=p.train_indices(k);va=p.val_indices(k);logits[va]=fit(x,p.labels,tr,va);folds.append({"fold":k,"rows":len(va),"correct":int(np.sum(logits[va].argmax(1)==p.labels[va]))})
  prob=softmax(logits);saved[n+"_probability"]=prob.astype(np.float32);saved[n+"_logits"]=logits.astype(np.float32);report["variants"][n]={"dimensions":x.shape[1],"correct":int(np.sum(prob.argmax(1)==p.labels)),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds};print(json.dumps({n:report["variants"][n]}),flush=True)
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
