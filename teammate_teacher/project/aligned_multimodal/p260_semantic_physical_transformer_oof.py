"""Strict OOF physical Transformer using visual features plus K710/K400 semantics."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import train_fold
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p260_semantic_physical_transformer_oof_v1";PATHS=(R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz",R/"runs/p91_videomaev2_depth_fold0_v1/complete_features.npz",R/"runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz");SEEDS=(26001,26017,26033)
def cfg():return SimpleNamespace(hidden_dim=192,heads=6,layers=2,dropout=.2,view_dropout=.15,epochs=35,batch_size=128,learning_rate=3e-4,weight_decay=.05,mixup_alpha=.2,repeat_consistency_weight=0.,repeat_embedding_weight=0.,repeat_same_label_only=False,class_triplet_weight=0.,triplet_margin=.2,teacher_weight=0.,domain_adversarial_weight=0.)
def std(x):
 x=np.asarray(x,np.float32);return ((x-x.mean(-1,keepdims=True))/np.maximum(x.std(-1,keepdims=True),1e-6)).astype(np.float16)
def tokens(z,kind):
 f=std(z["features"].reshape(len(z["features"]),-1,768));a=std(z["action_logits"].reshape(len(f),len(f[0]),-1));out=np.zeros((len(f),f.shape[1],1878),np.float16);out[:,:,:768]=f
 if kind=="k400":out[:,:,1478:1878]=a
 else:out[:,:,768:1478]=a
 return out
def main():
 p=load_protocol();z=[np.load(x) for x in PATHS]
 for q in z:
  if not np.array_equal(q["sample_ids"].astype(str),p.sample_ids):raise RuntimeError("P260 order mismatch")
 x=np.concatenate((tokens(z[0],"k710"),tokens(z[1],"k400"),tokens(z[2],"k710"),tokens(z[3],"k710")),1);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");logits=np.zeros((len(p.labels),40),float);folds=[]
 for k in range(3):
  tr=p.train_indices(k);va=p.val_indices(k);members=[train_fold(x,p.labels,tr,va,p.fold_id,s+k*1000,cfg(),device,None,None) for s in SEEDS];logits[va]=np.mean(np.stack(members),0);folds.append({"fold":k,"rows":len(va),"correct":int(np.sum(logits[va].argmax(1)==p.labels[va]))})
 prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);report={"stage":"P260_semantic_physical_transformer_OOF","status":"complete","protocol":{"features_frozen":True,"tokens":18,"token_dim":1878,"layout":"standardized feature768 + disjoint K710710/K400400 channels","seeds":list(SEEDS),"epochs":35,"strict_subject_folds":True,"test_rows_loaded":0},"metrics":{"correct":int(np.sum(prob.argmax(1)==p.labels)),"rows":len(p.labels),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",sample_ids=p.sample_ids,labels=p.labels,users=p.users,fold_id=p.fold_id,logits=logits.astype(np.float32),probability=prob.astype(np.float32));(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
