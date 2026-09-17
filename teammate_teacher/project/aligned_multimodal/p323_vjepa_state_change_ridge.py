"""Explicit V-JEPA2 hand/workspace early-late-motion state-change Ridge."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p90_teacher_common import load_protocol,softmax
from p90_videomaev2_distilled_teacher import class_sample_weights
from train_p46_videomae_head import l2_normalize,make_model
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;O=H/"runs/p323_vjepa_state_change_ridge_v1";CACHE=H.parent/"runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def n(x):return l2_normalize(np.asarray(x,np.float32))
def descriptor(v,kind):
 parts=[]
 if kind in {"workspace_interaction","combined"}:
  for e,l in ((5,11),(17,20)):
   parts.extend((n(v[:,e]),n(v[:,l]),n(v[:,l]-v[:,e]),n(np.abs(v[:,l]-v[:,e]))))
  parts.extend((n(v[:,14]),n(v[:,23]),n(v[:,23]-v[:,14]),n(np.abs(v[:,23]-v[:,14]))))
 if kind in {"left_right_hands","combined"}:
  for e,l in ((15,18),(16,19)):
   parts.extend((n(v[:,e]),n(v[:,l]),n(v[:,l]-v[:,e]),n(np.abs(v[:,l]-v[:,e]))))
 return np.concatenate(parts,1).astype(np.float32)
def fit(x,y,tr,va):
 m=make_model(3000.);m.fit(x[tr],y[tr],ridge__sample_weight=class_sample_weights(y[tr],.75));return m.decision_function(x[va])
def main():
 print("P323 tests explicit V-JEPA2 workspace/hand state changes before authorizing new-modality extraction.",flush=True);p=load_protocol();done=np.load(CACHE/"done.npy",mmap_mode="r")
 if not np.asarray(done).all():raise RuntimeError("P96 incomplete")
 v=np.asarray(np.load(CACHE/"features.npy",mmap_mode="r"),np.float32);splits=load_candidate_splits();basez=np.load(P310);base=basez["prediction"];labels=basez["labels"];saved={"sample_ids":p.sample_ids,"labels":p.labels};report={"stage":"P323_VJEPA2_state_change_Ridge","status":"complete","protocol":{"cache":"P96 dense24 frozen V-JEPA2 SSV2","variants":["workspace_interaction","left_right_hands","combined"],"alpha":3000.,"strict_subject_folds":True,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"variants":{}}
 for kind in ("workspace_interaction","left_right_hands","combined"):
  x=descriptor(v,kind);logits=np.zeros((len(p.labels),40),float);folds=[]
  for f in range(3):
   tr=p.train_indices(f);va=p.val_indices(f);logits[va]=fit(x,p.labels,tr,va);folds.append({"fold":f,"correct":int(np.sum(logits[va].argmax(1)==p.labels[va])),"rows":len(va)})
  prob=softmax(logits);saved[kind+"_probability"]=prob.astype(np.float32);pos={q:i for i,q in enumerate(p.sample_ids)};pred=np.concatenate([prob[[pos[q] for q in splits[s].split.sample_ids.astype(str)]].argmax(1) for s in S]);q=pred!=base;r=int(np.sum(q&(base!=labels)&(pred==labels)));h=int(np.sum(q&(base==labels)&(pred!=labels)));report["variants"][kind]={"dimensions":x.shape[1],"correct":int(np.sum(prob.argmax(1)==p.labels)),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds,"vs_p310":{"changed":int(q.sum()),"rescue":r,"harm":h,"oracle_gain":r,"direct_net":r-h}}
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",**saved);best=max(report["variants"],key=lambda k:(report["variants"][k]["correct"],report["variants"][k]["vs_p310"]["rescue"]));report["selection"]={"best":best,"authorize_new_modality_extraction":report["variants"][best]["vs_p310"]["rescue"]>=8};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: explicit V-JEPA2 early/late/motion-peak state-change Ridge from the existing P96 cache.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
