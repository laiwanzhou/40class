"""DINOv2 early/late object-state-change Ridge experts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p90_teacher_common import load_protocol,softmax
from p90_videomaev2_distilled_teacher import class_sample_weights
from train_p46_videomae_head import l2_normalize,make_model
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;O=H/"runs/p318_dinov2_state_change_ridge_v1";CACHE=H/"runs/p292_full40_dinov2_cache_v1/complete_features.npz";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def block(v,views):
 parts=[]
 for j in views:
  early=v[:,j,:4].mean(1);late=v[:,j,4:].mean(1);parts.extend((l2_normalize(early),l2_normalize(late),l2_normalize(late-early),l2_normalize(np.abs(late-early))))
 return np.concatenate(parts,1).astype(np.float32)
def fit(x,y,tr,va,alpha):
 m=make_model(alpha);m.fit(x[tr],y[tr],ridge__sample_weight=class_sample_weights(y[tr],.75));return m.decision_function(x[va])
def main():
 print("P318 tests whether explicit DINOv2 early/late state-change descriptors add deployable rescue beyond P310.",flush=True);p=load_protocol();z=np.load(CACHE)
 if not np.array_equal(z["sample_ids"].astype(str),p.sample_ids):raise RuntimeError("cache order")
 v=z["features"].astype(np.float32);variants={"workspace_state":block(v,[2]),"person_workspace_state":block(v,[1,2]),"all_view_state":block(v,[0,1,2])};saved={"sample_ids":p.sample_ids,"labels":p.labels};report={"stage":"P318_DINOv2_state_change_Ridge","status":"complete","protocol":{"frozen_cache":"P292 DINOv2-base 3 views x 8 frames","descriptors":["early_mean","late_mean","signed_delta","absolute_delta"],"alphas":[1000.,3000.],"strict_subject_folds":True,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"variants":{}};splits=load_candidate_splits();basez=np.load(P310);base=basez["prediction"];labels=basez["labels"]
 for name,x in variants.items():
  for alpha in (1000.,3000.):
   logits=np.zeros((len(p.labels),40),float);folds=[]
   for f in range(3):
    tr=p.train_indices(f);va=p.val_indices(f);logits[va]=fit(x,p.labels,tr,va,alpha);folds.append({"fold":f,"correct":int(np.sum(logits[va].argmax(1)==p.labels[va])),"rows":len(va)})
   prob=softmax(logits);key=f"{name}_a{int(alpha)}";saved[key+"_probability"]=prob.astype(np.float32);pos={q:i for i,q in enumerate(p.sample_ids)};pred=np.concatenate([prob[[pos[q] for q in splits[n].split.sample_ids.astype(str)]].argmax(1) for n in S]);q=pred!=base;rescue=int(np.sum(q&(base!=labels)&(pred==labels)));harm=int(np.sum(q&(base==labels)&(pred!=labels)));report["variants"][key]={"dimensions":x.shape[1],"correct":int(np.sum(prob.argmax(1)==p.labels)),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds,"vs_p310":{"changed":int(q.sum()),"rescue":rescue,"harm":harm,"oracle_gain":rescue,"direct_net":rescue-harm}}
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",**saved);best=max(report["variants"],key=lambda k:(report["variants"][k]["correct"],report["variants"][k]["vs_p310"]["rescue"]));report["selection"]={"best_standalone":best,"gate_oracle_rescue_at_least_8":report["variants"][best]["vs_p310"]["rescue"]>=8};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: frozen DINOv2 state-change Ridge over workspace, person+workspace, and all-view descriptors. Two pre-registered alphas. Primary evidence: strict OOF accuracy and unique rescue against P310.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
