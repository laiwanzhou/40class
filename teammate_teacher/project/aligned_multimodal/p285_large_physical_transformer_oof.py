"""Strict three-seed OOF for a 12-token IR/Depth/Thermal Transformer."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import train_fold
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p285_large_physical_transformer_oof_v1";PATHS=(R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz",R/"runs/p91_videomaev2_depth_fold0_v1/complete_features.npz",R/"runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz");SEEDS=(28501,28517,28533)
def cfg():return SimpleNamespace(hidden_dim=384,heads=8,layers=3,dropout=.30,view_dropout=.20,epochs=45,batch_size=96,learning_rate=2e-4,weight_decay=.08,mixup_alpha=.30,repeat_consistency_weight=0.,repeat_embedding_weight=0.,repeat_same_label_only=False,class_triplet_weight=0.,triplet_margin=.2,teacher_weight=0.,domain_adversarial_weight=0.)
def main():
 p=load_protocol();sources=[np.load(x) for x in PATHS]
 for z in sources:
  if not np.array_equal(z["sample_ids"].astype(str),p.sample_ids):raise RuntimeError("P238 order mismatch")
 x=np.concatenate([z["features"].astype(np.float16).reshape(len(p.labels),-1,768) for z in sources],1);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");logits=np.zeros((len(p.labels),40),np.float64);folds=[]
 for k in range(3):
  tr=p.train_indices(k);va=p.val_indices(k);members=[]
  for seed in SEEDS:members.append(train_fold(x,p.labels,tr,va,p.fold_id,seed+k*1000,cfg(),device,None,None))
  logits[va]=np.mean(np.stack(members),0);folds.append({"fold":k,"rows":len(va),"correct":int(np.sum(logits[va].argmax(1)==p.labels[va]))})
 prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);report={"stage":"P285_large_physical_transformer_OOF","status":"complete","protocol":{"features_frozen":True,"modalities":["ir_vmae","ir_iv2","depth_vmae","thermal_vmae"],"tokens":18,"hidden":384,"layers":3,"heads":8,"seeds":list(SEEDS),"epochs":45,"strict_subject_folds":True,"test_rows_loaded":0},"metrics":{"correct":int(np.sum(prob.argmax(1)==p.labels)),"rows":len(p.labels),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",sample_ids=p.sample_ids,labels=p.labels,users=p.users,fold_id=p.fold_id,logits=logits.astype(np.float32),probability=prob.astype(np.float32));(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

