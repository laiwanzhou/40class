"""Refit the frozen P146 workspace-token head on all Train and infer Test."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from p142_vjepa_token_transformer_oof import train_fold
from p90_teacher_common import load_protocol
H=Path(__file__).resolve().parent;R=H.parent;TRAIN=R/"runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1/features.npy";TEST=R/"runs/p171_vjepa2_dense24_test_v1";O=H/"runs/p221_p146_workspace_test_v1";SEEDS=(14601,14617,14633);SUB=np.asarray((2,5,8,11))
def config():return SimpleNamespace(hidden_dim=192,heads=6,layers=2,dropout=.20,view_dropout=.15,epochs=35,batch_size=128,learning_rate=3e-4,weight_decay=.05,mixup_alpha=.20,repeat_consistency_weight=0.,repeat_embedding_weight=0.,repeat_same_label_only=False,class_triplet_weight=0.,triplet_margin=.20,teacher_weight=0.,domain_adversarial_weight=0.,token_subset="workspace")
def main():
 p=load_protocol();tr=np.load(TRAIN,mmap_mode="r");te=np.load(TEST/"features.npy",mmap_mode="r");ids=np.load(TEST/"sample_ids.npy").astype(str);done=np.load(TEST/"done.npy")
 if tr.shape!=(2914,24,1024) or te.shape!=(401,24,1024) or not done.all():raise RuntimeError("P221 cache mismatch")
 values=np.concatenate((np.asarray(tr[:,SUB],np.float16),np.asarray(te[:,SUB],np.float16)));labels=np.concatenate((p.labels,np.zeros(len(ids),np.int64)));domains=np.concatenate((p.fold_id,np.zeros(len(ids),np.int64)));ti=np.arange(2914);vi=np.arange(2914,len(labels));device=torch.device("cuda" if torch.cuda.is_available() else "cpu");members=[]
 for seed in SEEDS:
  members.append(train_fold(values,labels,ti,vi,domains,seed,config(),device,None,None));print(json.dumps({"seed":seed,"complete":True}),flush=True)
 logits=np.mean(np.stack(members),0).astype(np.float64);prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"test_predictions.npz",sample_ids=ids,logits=logits.astype(np.float32),probability=prob.astype(np.float32));report={"stage":"P221_P146_workspace_allTrain_to_Test","status":"complete","protocol":{"backbone_frozen":True,"token_subset":"workspace","token_indices":SUB.tolist(),"seeds":list(SEEDS),"epochs":35,"test_labels_read":False},"test":{"rows":len(ids),"mean_confidence":float(prob.max(1).mean())}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
