"""Strict OOF and all-Train/Test Ridge for early/late Thermal representation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p90_teacher_common import load_protocol,softmax
from p90_videomaev2_distilled_teacher import class_sample_weights
from train_p46_videomae_head import l2_normalize,make_model
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p272_thermal_early_late_ridge_v1";IR=(R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz");IRT=(R/"runs/p90_videomaev2_distilled_test_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_test_v1/complete_features.npz");TH=H/"runs/p271_thermal_early_late_v1/train_features.npz";THT=H/"runs/p271_thermal_early_late_v1/test_features.npz";IDS=H/"runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
def flat(z):return l2_normalize(z["features"].astype(np.float32)).reshape(len(z["features"]),-1)
def align(v,s,ids):
 out=np.zeros((len(ids),v.shape[1]),np.float32);pos={q:i for i,q in enumerate(ids)};rows=np.asarray([pos[q] for q in s.astype(str)]);out[rows]=v;mask=np.zeros(len(ids),bool);mask[rows]=True;return out,mask
def main():
 p=load_protocol();a=np.load(IR[0]);b=np.load(IR[1]);th=np.load(TH);x=np.concatenate((flat(a),flat(b),flat(th)),1);logits=np.zeros((len(p.labels),40),float);folds=[]
 for k in range(3):
  tr=p.train_indices(k);va=p.val_indices(k);m=make_model(3000.);m.fit(x[tr],p.labels[tr],ridge__sample_weight=class_sample_weights(p.labels[tr],.75));logits[va]=m.decision_function(x[va]);folds.append({"fold":k,"correct":int(np.sum(logits[va].argmax(1)==p.labels[va])),"rows":len(va)})
 prob=softmax(logits);ids=np.load(IDS)["sample_ids"].astype(str);at,ma=align(flat(np.load(IRT[0])),np.load(IRT[0])["sample_ids"],ids);bt,mb=align(flat(np.load(IRT[1])),np.load(IRT[1])["sample_ids"],ids);tt,mt=align(flat(np.load(THT)),np.load(THT)["sample_ids"],ids);tx=np.concatenate((at,bt,tt),1);m=make_model(3000.);m.fit(x,p.labels,ridge__sample_weight=class_sample_weights(p.labels,.75));tl=m.decision_function(tx);tp=softmax(tl);available=ma&mb&np.load(THT)["available"].astype(bool);report={"stage":"P272_thermal_early_late_Ridge","status":"complete","protocol":{"alpha":3000.,"class_weight_power":.75,"thermal_windows":2,"views":3,"strict_subject_folds":True,"test_labels_read":False},"oof":{"correct":int(np.sum(prob.argmax(1)==p.labels)),"rows":len(p.labels),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds},"test":{"rows":len(ids),"available":int(available.sum())}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"predictions.npz",sample_ids=p.sample_ids,labels=p.labels,probability=prob.astype(np.float32),test_sample_ids=ids,test_probability=tp.astype(np.float32),test_available=available);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
