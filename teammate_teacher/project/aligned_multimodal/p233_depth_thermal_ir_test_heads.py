"""Refit P231 frozen multimodal Ridge heads on all Train and predict Test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p90_teacher_common import load_protocol,softmax
from p90_videomaev2_distilled_teacher import class_sample_weights
from train_p46_videomae_head import l2_normalize,make_model
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p233_depth_thermal_ir_test_heads_v1";IR1=R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz";IR2=R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz";D=R/"runs/p91_videomaev2_depth_fold0_v1/complete_features.npz";T=R/"runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz";IR1T=R/"runs/p90_videomaev2_distilled_test_v1/complete_features.npz";IR2T=R/"runs/p90_internvideo2_l_k400_test_v1/complete_features.npz";DT=H/"runs/p232_depth_thermal_test_features_v1/depth_features.npz";TT=H/"runs/p232_depth_thermal_test_features_v1/thermal_features.npz";IDS=H/"runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
def flat(z):return l2_normalize(z["features"].astype(np.float32)).reshape(len(z["features"]),-1)
def align(values,sids,ids):
 out=np.zeros((len(ids),values.shape[1]),np.float32);pos={v:i for i,v in enumerate(ids)};rows=np.asarray([pos[v] for v in sids.astype(str)]);out[rows]=values;mask=np.zeros(len(ids),bool);mask[rows]=True;return out,mask
def main():
 p=load_protocol();a=np.load(IR1);b=np.load(IR2);d=np.load(D);t=np.load(T);at=np.load(IR1T);bt=np.load(IR2T);dt=np.load(DT);tt=np.load(TT);ids=np.load(IDS)["sample_ids"].astype(str);ir=np.concatenate((flat(a),flat(b)),1);df=flat(d);tf=flat(t);ait,ia=align(flat(at),at["sample_ids"],ids);bit,ib=align(flat(bt),bt["sample_ids"],ids);irt=np.concatenate((ait,bit),1);dft,_=align(flat(dt),dt["sample_ids"],ids);tft,_=align(flat(tt),tt["sample_ids"],ids);dav=dt["modality_available"].astype(bool);tav=tt["modality_available"].astype(bool);variants={"ir_depth":(np.concatenate((ir,df),1),np.concatenate((irt,dft),1),ia&ib&dav),"ir_thermal":(np.concatenate((ir,tf),1),np.concatenate((irt,tft),1),ia&ib&tav),"ir_depth_thermal":(np.concatenate((ir,df,tf),1),np.concatenate((irt,dft,tft),1),ia&ib&dav&tav)};saved={"sample_ids":ids};report={"stage":"P233_P231_allTrain_to_Test","status":"complete","protocol":{"alpha":3000.,"class_weight_power":.75,"test_labels_read":False},"variants":{}}
 for n,(x,tx,av) in variants.items():
  m=make_model(3000.);m.fit(x,p.labels,ridge__sample_weight=class_sample_weights(p.labels,.75));logits=m.decision_function(tx);prob=softmax(logits);saved[n+"_probability"]=prob.astype(np.float32);saved[n+"_available"]=av;report["variants"][n]={"train_rows":len(x),"test_rows":len(tx),"available":int(av.sum()),"dimensions":x.shape[1]}
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"test_predictions.npz",**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
