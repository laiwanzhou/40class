"""Source-only centroid/Ridge readouts in each outer P87-S embedding space."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier
from p90_teacher_common import load_protocol
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;O=H/"runs/p334_p87s_embedding_readout_oof_v1";E=H/"runs/p333_p87s_outer_embedding_spaces_v1/embedding_spaces.npz";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def softmax(x):x=np.asarray(x,float);x-=x.max(1,keepdims=True);p=np.exp(x);return p/p.sum(1,keepdims=True)
def centroid(x,y,tr,va):
 c=np.stack([x[tr][y[tr]==k].mean(0) for k in range(40)]);c/=np.clip(np.linalg.norm(c,axis=1,keepdims=True),1e-6,None);return x[va]@c.T
def ridge(x,y,tr,va,a):m=RidgeClassifier(alpha=a,class_weight="balanced",solver="lsqr");m.fit(x[tr],y[tr]);return m.decision_function(x[va])
def main():
 print("P334 tests source-only centroid and Ridge readouts on strict outer P87-S embeddings.",flush=True);p=load_protocol();data=load_candidate_splits();e=np.load(E);basez=np.load(P310);variants=("centroid","ridge100","ridge1000","ridge3000");report={"stage":"P334_P87S_embedding_readout_OOF","status":"complete","protocol":{"embedding":"P333 source-trained outer P87-S visual_embedding","readouts":list(variants),"fit_rows":"all non-held subjects in the same outer space","held_labels_used_for_fit_or_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"variants":{v:{"cohorts":{}} for v in variants}};saved={};totals={v:[] for v in variants};labels=[];bases=[]
 for n in S:
  ids=e[f"{n}_sample_ids"].astype(str);x=e[f"{n}_embedding"].astype(np.float32);pos={q:i for i,q in enumerate(ids)};held=np.asarray([pos[q] for q in data[n].split.sample_ids.astype(str)]);held_users=set(data[n].split.users.astype(str).tolist());tr=np.flatnonzero(~np.isin(p.users,list(held_users)));y=p.labels;labels.append(y[held]);base=basez[f"{n}_held_prediction"].astype(int);bases.append(base)
  outputs={"centroid":centroid(x,y,tr,held),"ridge100":ridge(x,y,tr,held,100.),"ridge1000":ridge(x,y,tr,held,1000.),"ridge3000":ridge(x,y,tr,held,3000.)}
  for name,logits in outputs.items():
   prob=softmax(logits);pred=prob.argmax(1);q=pred!=base;r=int(np.sum(q&(base!=y[held])&(pred==y[held])));h=int(np.sum(q&(base==y[held])&(pred!=y[held])));report["variants"][name]["cohorts"][n]={"rows":len(held),"correct":int(np.sum(pred==y[held])),"accuracy":float(np.mean(pred==y[held])),"vs_p310_rescue":r,"vs_p310_harm":h};saved[f"{n}_{name}_probability"]=prob.astype(np.float32);totals[name].append(pred)
 labels=np.concatenate(labels);base=np.concatenate(bases)
 for name in variants:
  pred=np.concatenate(totals[name]);q=pred!=base;r=int(np.sum(q&(base!=labels)&(pred==labels)));h=int(np.sum(q&(base==labels)&(pred!=labels)));report["variants"][name]["aggregate"]={"rows":len(labels),"correct":int(np.sum(pred==labels)),"accuracy":float(np.mean(pred==labels)),"vs_p310_rescue":r,"vs_p310_harm":h,"oracle_correct":int(np.sum((base==labels)|(pred==labels)))}
 best=max(variants,key=lambda n:(report["variants"][n]["aggregate"]["correct"],report["variants"][n]["aggregate"]["vs_p310_rescue"]));report["selection"]={"best_standalone":best};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: source-only fixed readouts in each outer P87-S embedding space.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
