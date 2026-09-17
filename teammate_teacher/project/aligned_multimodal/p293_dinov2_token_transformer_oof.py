"""Strict three-seed OOF 24-token DINOv2 appearance Transformer."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import train_fold
H=Path(__file__).resolve().parent;O=H/"runs/p293_dinov2_token_transformer_oof_v1";CACHE=H/"runs/p292_full40_dinov2_cache_v1/complete_features.npz";SEEDS=(29301,29317,29333)
def cfg():return SimpleNamespace(hidden_dim=192,heads=6,layers=2,dropout=.25,view_dropout=.15,epochs=35,batch_size=128,learning_rate=3e-4,weight_decay=.05,mixup_alpha=.2,repeat_consistency_weight=0.,repeat_embedding_weight=0.,repeat_same_label_only=False,class_triplet_weight=0.,triplet_margin=.2,teacher_weight=0.,domain_adversarial_weight=0.)
def main():
 p=load_protocol();z=np.load(CACHE);d={q:i for i,q in enumerate(z["sample_ids"].astype(str))};rows=np.asarray([d[q] for q in p.sample_ids]);x=z["features"][rows].astype(np.float16).reshape(len(rows),24,768);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");logits=np.zeros((len(p.labels),40),float);folds=[]
 for f in range(3):
  tr=p.train_indices(f);va=p.val_indices(f);members=[train_fold(x,p.labels,tr,va,p.fold_id,s+f*1000,cfg(),device,None,None) for s in SEEDS];logits[va]=np.mean(members,0);folds.append({"fold":f,"rows":len(va),"correct":int(np.sum(logits[va].argmax(1)==p.labels[va]))})
 prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);report={"stage":"P293_DINOv2_token_transformer_OOF","status":"complete","protocol":{"backbone":"frozen facebook/dinov2-base","tokens":24,"views":3,"frames":8,"seeds":list(SEEDS),"strict_subject_folds":True,"test_rows_loaded":0},"metrics":{"correct":int(np.sum(prob.argmax(1)==p.labels)),"rows":len(p.labels),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",sample_ids=p.sample_ids,labels=p.labels,users=p.users,fold_id=p.fold_id,logits=logits.astype(np.float32),probability=prob.astype(np.float32));(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
