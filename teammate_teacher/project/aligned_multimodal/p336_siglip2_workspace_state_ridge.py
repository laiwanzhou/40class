"""Strict SigLIP2 hand-workspace state-change Ridge experts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p90_teacher_common import load_protocol,softmax
from p90_videomaev2_distilled_teacher import class_sample_weights
from train_p46_videomae_head import l2_normalize,make_model
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;O=H/"runs/p336_siglip2_workspace_state_ridge_v1";CACHE=H/"runs/p335_siglip2_workspace_state_cache_v1/features.npy";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def n(x):return l2_normalize(np.asarray(x,np.float32))
def variants(v):
 early=v[:,0].mean(1);late=v[:,1].mean(1);state=np.concatenate((n(early),n(late),n(late-early),n(np.abs(late-early))),1);frames=np.concatenate([n(v[:,w,t]) for w in range(2) for t in range(4)],1);temporal=np.concatenate((frames,n(late-early),n(np.abs(late-early))),1);return {"window_state":state,"all_frames":frames,"frames_plus_delta":temporal}
def fit(x,y,tr,va,a):m=make_model(a);m.fit(x[tr],y[tr],ridge__sample_weight=class_sample_weights(y[tr],.75));return m.decision_function(x[va])
def main():
 print("P336 tests frozen SigLIP2 hand-workspace early/late state readouts under strict subject folds.",flush=True);p=load_protocol();v=np.asarray(np.load(CACHE,mmap_mode="r"),np.float32);splits=load_candidate_splits();basez=np.load(P310);base=basez["prediction"];labels=basez["labels"];saved={"sample_ids":p.sample_ids,"labels":p.labels};report={"stage":"P336_SigLIP2_workspace_state_Ridge","status":"complete","protocol":{"cache":"P335 SigLIP2-B16 2 windows x 4 frames","variants":["window_state","all_frames","frames_plus_delta"],"alphas":[1000.,3000.],"strict_subject_folds":True,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"variants":{}}
 for name,x in variants(v).items():
  for a in (1000.,3000.):
   logits=np.zeros((len(p.labels),40),float);folds=[]
   for f in range(3):
    tr=p.train_indices(f);va=p.val_indices(f);logits[va]=fit(x,p.labels,tr,va,a);folds.append({"fold":f,"correct":int(np.sum(logits[va].argmax(1)==p.labels[va])),"rows":len(va)})
   prob=softmax(logits);key=f"{name}_a{int(a)}";saved[key+"_probability"]=prob.astype(np.float32);pos={q:i for i,q in enumerate(p.sample_ids)};pred=np.concatenate([prob[[pos[q] for q in splits[s].split.sample_ids.astype(str)]].argmax(1) for s in S]);q=pred!=base;r=int(np.sum(q&(base!=labels)&(pred==labels)));h=int(np.sum(q&(base==labels)&(pred!=labels)));report["variants"][key]={"dimensions":x.shape[1],"correct":int(np.sum(prob.argmax(1)==p.labels)),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds,"vs_p310":{"changed":int(q.sum()),"rescue":r,"harm":h,"oracle_correct":int(np.sum((base==labels)|(pred==labels))),"direct_net":r-h}}
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",**saved);best=max(report["variants"],key=lambda k:(report["variants"][k]["correct"],report["variants"][k]["vs_p310"]["rescue"]));report["selection"]={"best":best,"authorize_group_audit":report["variants"][best]["vs_p310"]["rescue"]>=8};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: frozen SigLIP2 workspace state Ridge.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
