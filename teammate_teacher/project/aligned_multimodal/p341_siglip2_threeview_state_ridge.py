"""Strict three-view SigLIP2 state-change Ridge ablation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p90_teacher_common import load_protocol,softmax
from p90_videomaev2_distilled_teacher import class_sample_weights
from train_p46_videomae_head import l2_normalize,make_model
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;O=H/"runs/p341_siglip2_threeview_state_ridge_v1";CACHE=H/"runs/p340_siglip2_threeview_state_cache_v1/features.npy";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def state(v,j):e=v[:,j,0].mean(1);l=v[:,j,1].mean(1);return np.concatenate([l2_normalize(x) for x in (e,l,l-e,np.abs(l-e))],1)
def fit(x,y,tr,va):m=make_model(3000.);m.fit(x[tr],y[tr],ridge__sample_weight=class_sample_weights(y[tr],.75));return m.decision_function(x[va])
def main():
 print("P341 tests fixed-alpha scene/person/workspace SigLIP2 state combinations.",flush=True);p=load_protocol();v=np.asarray(np.load(CACHE,mmap_mode="r"),np.float32);blocks=[state(v,j) for j in range(3)];variants={"scene_state":blocks[0],"person_state":blocks[1],"scene_person_state":np.concatenate(blocks[:2],1),"threeview_state":np.concatenate(blocks,1)};splits=load_candidate_splits();basez=np.load(P310);base=basez["prediction"];labels=basez["labels"];saved={"sample_ids":p.sample_ids,"labels":p.labels};report={"stage":"P341_SigLIP2_threeview_state_Ridge","status":"complete","protocol":{"cache":"P340 frozen SigLIP2 threeview","alpha":3000.,"strict_subject_folds":True,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"variants":{}}
 for name,x in variants.items():
  logits=np.zeros((len(p.labels),40),float);folds=[]
  for f in range(3):
   tr=p.train_indices(f);va=p.val_indices(f);logits[va]=fit(x,p.labels,tr,va);folds.append({"fold":f,"correct":int(np.sum(logits[va].argmax(1)==p.labels[va])),"rows":len(va)})
  prob=softmax(logits);saved[name+"_probability"]=prob.astype(np.float32);pos={q:i for i,q in enumerate(p.sample_ids)};pred=np.concatenate([prob[[pos[q] for q in splits[s].split.sample_ids.astype(str)]].argmax(1) for s in S]);q=pred!=base;r=int(np.sum(q&(base!=labels)&(pred==labels)));h=int(np.sum(q&(base==labels)&(pred!=labels)));report["variants"][name]={"dimensions":x.shape[1],"correct":int(np.sum(prob.argmax(1)==p.labels)),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds,"vs_p310":{"rescue":r,"harm":h,"oracle_correct":int(np.sum((base==labels)|(pred==labels)))}}
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",**saved);best=max(variants,key=lambda n:(report["variants"][n]["correct"],report["variants"][n]["vs_p310"]["rescue"]));report["selection"]={"best":best};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: fixed-alpha SigLIP2 three-view state ablation.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
