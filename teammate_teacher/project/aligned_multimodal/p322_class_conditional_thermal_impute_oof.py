"""P321 with source-only class-conditional Thermal token imputation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import build_repeat_pairs,train_fold
from p253_repeat_physical_transformer_oof import PATHS,SEEDS,cfg
from p321_union_full_partial_repeat_oof import FULL,PART,CONTROL,tokens
H=Path(__file__).resolve().parent;O=H/"runs/p322_class_conditional_thermal_impute_oof_v1"
def main():
 print("P322 tests source-only class-conditional Thermal token imputation for the seven P320 training rows.",flush=True);p=load_protocol();sources=[np.load(q) for q in PATHS];main=np.concatenate([q["features"].astype(np.float16).reshape(len(p.labels),-1,768) for q in sources],1);a=np.load(FULL);b=np.load(PART);base_x=np.concatenate((main,tokens(a),tokens(b)));labels=np.concatenate((p.labels,a["labels"].astype(int),b["labels"].astype(int)));users=np.concatenate((p.users,a["users"].astype(str),b["users"].astype(str)));domain=np.zeros(len(labels),int);pairs=build_repeat_pairs(p);extra=np.arange(len(p.labels),len(labels));partial=np.arange(len(p.labels)+len(a["labels"]),len(labels));logits=np.zeros((len(p.labels),40),float);folds=[];device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 for f in range(3):
  held=p.val_indices(f);hu=set(p.users[held].tolist());add=extra[~np.isin(users[extra],list(hu))];trmain=p.train_indices(f);x=base_x.copy()
  for idx in partial:
   cls=labels[idx];pool=trmain[p.labels[trmain]==cls]
   if len(pool):x[idx,15:18]=main[pool,15:18].astype(np.float32).mean(0).astype(np.float16)
  tr=np.concatenate((trmain,add));members=[train_fold(x,labels,tr,held,domain,s+f*1000,cfg(),device,pairs,None) for s in SEEDS];logits[held]=np.mean(members,0);folds.append({"fold":f,"extra_train_rows":len(add),"imputation_source":"same-class main source-fold mean","correct":int(np.sum(logits[held].argmax(1)==p.labels[held])),"rows":len(held)})
 prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);old=np.load(CONTROL)["probability"].argmax(1);new=prob.argmax(1);changed=old!=new;report={"stage":"P322_class_conditional_Thermal_impute_OOF","status":"complete","protocol":{"paired_control":"P306","partial_rows":len(partial),"imputation":"same-class mean Thermal tokens from main source-fold training rows only","held_rows_used_for_imputation":False,"same_architecture_seeds_config":True,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"control_correct":int(np.sum(old==p.labels)),"correct":int(np.sum(new==p.labels)),"accuracy":float(np.mean(new==p.labels)),"net_vs_p306":int(np.sum(new==p.labels)-np.sum(old==p.labels)),"changed":int(changed.sum()),"rescued":int(np.sum(changed&(old!=p.labels)&(new==p.labels))),"harmed":int(np.sum(changed&(old==p.labels)&(new!=p.labels))),"folds":folds};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",sample_ids=p.sample_ids,labels=p.labels,probability=prob.astype(np.float32),logits=logits.astype(np.float32));(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: source-only class-conditional Thermal token imputation for seven aligned Depth+IR extras.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
