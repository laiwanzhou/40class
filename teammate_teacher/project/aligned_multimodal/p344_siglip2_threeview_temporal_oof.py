"""Strict 3-seed temporal Transformer over three-view SigLIP2 states."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import train_fold
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;O=H/"runs/p344_siglip2_threeview_temporal_oof_v1";CACHE=H/"runs/p340_siglip2_threeview_state_cache_v1/features.npy";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0");SEEDS=(34401,34417,34433)
def cfg():return SimpleNamespace(hidden_dim=192,heads=6,layers=2,dropout=.2,view_dropout=.15,epochs=35,batch_size=128,learning_rate=3e-4,weight_decay=.05,mixup_alpha=.2,repeat_consistency_weight=0.,repeat_embedding_weight=0.,repeat_same_label_only=False,class_triplet_weight=0.,triplet_margin=.2,teacher_weight=0.,domain_adversarial_weight=0.)
def main():
 print("P344 tests 24 three-view SigLIP2 frame tokens plus 12 explicit state tokens.",flush=True);p=load_protocol();v=np.asarray(np.load(CACHE,mmap_mode="r"),np.float16);states=[]
 for j in range(3):
  e=v[:,j,0].astype(np.float32).mean(1);l=v[:,j,1].astype(np.float32).mean(1);states.extend((e,l,l-e,np.abs(l-e)))
 x=np.concatenate((v.reshape(len(v),24,768),np.stack(states,1).astype(np.float16)),1);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");logits=np.zeros((len(p.labels),40),float);folds=[]
 for f in range(3):
  tr=p.train_indices(f);va=p.val_indices(f);members=[train_fold(x,p.labels,tr,va,p.fold_id,s+f*1000,cfg(),device,None,None) for s in SEEDS];logits[va]=np.mean(members,0);folds.append({"fold":f,"correct":int(np.sum(logits[va].argmax(1)==p.labels[va])),"rows":len(va)})
 prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);splits=load_candidate_splits();basez=np.load(P310);base=basez["prediction"];labels=basez["labels"];pos={q:i for i,q in enumerate(p.sample_ids)};pred=np.concatenate([prob[[pos[q] for q in splits[n].split.sample_ids.astype(str)]].argmax(1) for n in S]);q=pred!=base;r=int(np.sum(q&(base!=labels)&(pred==labels)));h=int(np.sum(q&(base==labels)&(pred!=labels)));report={"stage":"P344_SigLIP2_threeview_temporal_OOF","status":"complete","protocol":{"tokens":36,"views":["scene","person","workspace"],"seeds":list(SEEDS),"epochs":35,"strict_subject_folds":True,"test_rows_loaded":0,"test_labels_read":False},"metrics":{"correct":int(np.sum(prob.argmax(1)==p.labels)),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds},"vs_p310":{"rescue":r,"harm":h,"oracle_correct":int(np.sum((base==labels)|(pred==labels)))},"gate":{"pass":r>=8}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",sample_ids=p.sample_ids,labels=p.labels,probability=prob.astype(np.float32),logits=logits.astype(np.float32));(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: fixed 3-seed SigLIP2 three-view temporal-state Transformer.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
