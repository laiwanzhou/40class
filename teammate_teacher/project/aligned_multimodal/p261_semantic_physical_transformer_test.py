"""All-Train/Test counterpart of P260 semantic physical Transformer."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import train_fold
from p260_semantic_physical_transformer_oof import std
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p261_semantic_physical_transformer_test_v1";TR=(R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz",R/"runs/p91_videomaev2_depth_fold0_v1/complete_features.npz",R/"runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz");TE=(R/"runs/p90_videomaev2_distilled_test_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_test_v1/complete_features.npz",H/"runs/p232_depth_thermal_test_features_v1/depth_features.npz",H/"runs/p232_depth_thermal_test_features_v1/thermal_features.npz");IDS=H/"runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz";SEEDS=(26001,26017,26033)
def cfg():return SimpleNamespace(hidden_dim=192,heads=6,layers=2,dropout=.2,view_dropout=.15,epochs=35,batch_size=128,learning_rate=3e-4,weight_decay=.05,mixup_alpha=.2,repeat_consistency_weight=0.,repeat_embedding_weight=0.,repeat_same_label_only=False,class_triplet_weight=0.,triplet_margin=.2,teacher_weight=0.,domain_adversarial_weight=0.)
def tok(z,kind):
 f=std(z["features"].reshape(len(z["features"]),-1,768));a=std(z["action_logits"].reshape(len(f),len(f[0]),-1));out=np.zeros((len(f),f.shape[1],1878),np.float16);out[:,:,:768]=f
 if kind=="k400":out[:,:,1478:]=a
 else:out[:,:,768:1478]=a
 return out
def align(v,s,ids):
 out=np.zeros((len(ids),v.shape[1],v.shape[2]),v.dtype);pos={q:i for i,q in enumerate(ids)};rows=np.asarray([pos[q] for q in s.astype(str)]);out[rows]=v;mask=np.zeros(len(ids),bool);mask[rows]=True;return out,mask
def main():
 p=load_protocol();tr=[np.load(x) for x in TR];te=[np.load(x) for x in TE];x=np.concatenate((tok(tr[0],"k710"),tok(tr[1],"k400"),tok(tr[2],"k710"),tok(tr[3],"k710")),1);ids=np.load(IDS)["sample_ids"].astype(str);blocks=[];masks=[]
 for z,k in zip(te,("k710","k400","k710","k710")):
  v,m=align(tok(z,k),z["sample_ids"],ids);blocks.append(v);masks.append(m)
 tx=np.concatenate(blocks,1);available=masks[0]&masks[1]&masks[2];values=np.concatenate((x,tx));labels=np.concatenate((p.labels,np.zeros(len(ids),np.int64)));domains=np.concatenate((p.fold_id,np.zeros(len(ids),np.int64)));ti=np.arange(len(p.labels));vi=np.arange(len(p.labels),len(labels));device=torch.device("cuda" if torch.cuda.is_available() else "cpu");members=[train_fold(values,labels,ti,vi,domains,s,cfg(),device,None,None) for s in SEEDS];logits=np.mean(np.stack(members),0);prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"test_predictions.npz",sample_ids=ids,logits=logits.astype(np.float32),probability=prob.astype(np.float32),available=available);report={"stage":"P261_P260_allTrain_to_Test","status":"complete","protocol":{"tokens":18,"token_dim":1878,"seeds":list(SEEDS),"epochs":35,"test_labels_read":False},"test":{"rows":len(ids),"available":int(available.sum()),"mean_confidence":float(prob[available].max(1).mean())}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
