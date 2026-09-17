"""All-Train/Test counterpart of P266 modality-specific physical model."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from p90_teacher_common import load_protocol
from p266_modality_specific_physical_oof import train,SEEDS
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p267_modality_specific_physical_test_v1";TR=(R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz",R/"runs/p91_videomaev2_depth_fold0_v1/complete_features.npz",R/"runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz");TE=(R/"runs/p90_videomaev2_distilled_test_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_test_v1/complete_features.npz",H/"runs/p232_depth_thermal_test_features_v1/depth_features.npz",H/"runs/p232_depth_thermal_test_features_v1/thermal_features.npz");IDS=H/"runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
def align(z,ids):
 v=z["features"].astype(np.float16).reshape(len(z["features"]),-1,768);out=np.zeros((len(ids),v.shape[1],768),np.float16);pos={q:i for i,q in enumerate(ids)};rows=np.asarray([pos[q] for q in z["sample_ids"].astype(str)]);out[rows]=v;mask=np.zeros(len(ids),bool);mask[rows]=True;return out,mask
def main():
 p=load_protocol();tr=[np.load(q) for q in TR];x=np.concatenate([q["features"].astype(np.float16).reshape(len(p.labels),-1,768) for q in tr],1);ids=np.load(IDS)["sample_ids"].astype(str);blocks=[];masks=[]
 for path in TE:
  v,m=align(np.load(path),ids);blocks.append(v);masks.append(m)
 tx=np.concatenate(blocks,1);available=masks[0]&masks[1]&masks[2];values=np.concatenate((x,tx));labels=np.concatenate((p.labels,np.zeros(len(ids),np.int64)));ti=np.arange(len(p.labels));vi=np.arange(len(p.labels),len(labels));device=torch.device("cuda" if torch.cuda.is_available() else "cpu");members=[train(values,labels,ti,vi,s,device) for s in SEEDS];logits=np.mean(members,0);prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"test_predictions.npz",sample_ids=ids,logits=logits.astype(np.float32),probability=prob.astype(np.float32),available=available);report={"stage":"P267_P266_allTrain_to_Test","status":"complete","protocol":{"separate_projections":True,"token_lengths":[6,6,3,3],"seeds":list(SEEDS),"test_labels_read":False},"test":{"rows":len(ids),"available":int(available.sum()),"mean_confidence":float(prob[available].max(1).mean())}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
