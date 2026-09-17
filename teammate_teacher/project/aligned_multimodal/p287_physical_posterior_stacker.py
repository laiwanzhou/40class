"""Strict OOF/Test physical-posterior stackers with two frozen regularizations."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
H=Path(__file__).resolve().parent;O=H/"runs/p287_physical_posterior_stacker_v1";SOURCES=((H/"runs/p231_depth_thermal_ir_oof_v1/oof_predictions.npz","ir_thermal_probability",H/"runs/p233_depth_thermal_ir_test_heads_v1/test_predictions.npz","ir_thermal_probability","ir_thermal_available"),(H/"runs/p238_physical_token_transformer_oof_v1/oof_predictions.npz","probability",H/"runs/p239_physical_token_transformer_test_v1/test_predictions.npz","probability","available"),(H/"runs/p253_repeat_physical_transformer_oof_v1/oof_predictions.npz","probability",H/"runs/p254_repeat_physical_transformer_test_v1/test_predictions.npz","probability","available"),(H/"runs/p260_semantic_physical_transformer_oof_v1/oof_predictions.npz","probability",H/"runs/p261_semantic_physical_transformer_test_v1/test_predictions.npz","probability","available"),(H/"runs/p266_modality_specific_physical_oof_v1/oof_predictions.npz","probability",H/"runs/p267_modality_specific_physical_test_v1/test_predictions.npz","probability","available"),(H/"runs/p277_physical_centroid_expert_v1/predictions.npz","probability",H/"runs/p277_physical_centroid_expert_v1/predictions.npz","test_probability","test_available"));CS=(.001,.03)
def al(v,s,t):d={q:i for i,q in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[q] for q in t.astype(str)])]
def feat(ps):return np.concatenate([v for p in ps for v in (np.sqrt(np.clip(p,0,1)),np.log(np.clip(p,1e-6,1)))],1).astype(np.float32)
def model(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1200,solver="lbfgs",class_weight="balanced"))
def fullprob(m,x):
 p=m.predict_proba(x);out=np.zeros((len(x),40),np.float32);out[:,m[-1].classes_.astype(int)]=p;return out
def main():
 first=np.load(SOURCES[0][0]);ids=first["sample_ids"].astype(str);y=first["labels"].astype(int);fold=first["fold_id"].astype(int);tfirst=np.load(SOURCES[0][2]);tids=tfirst["sample_ids"].astype(str);train=[];test=[];av=[]
 for op,ok,tp,tk,ak in SOURCES:
  a=np.load(op);b=np.load(tp);sid=b["test_sample_ids"] if "test_sample_ids" in b.files and len(b[tk])==len(b["test_sample_ids"]) else b["sample_ids"];train.append(al(a[ok],a["sample_ids"],ids));test.append(al(b[tk],sid,tids));av.append(al(b[ak],sid,tids).astype(bool))
 x=feat(train);tx=feat(test);available=np.logical_and.reduce(av);saved={"sample_ids":ids,"labels":y,"fold_id":fold,"test_sample_ids":tids,"test_available":available};report={"stage":"P287_physical_posterior_stacker","status":"complete","protocol":{"source_count":len(SOURCES),"features":"sqrt+log posterior","regularizations":list(CS),"strict_subject_folds":True,"test_labels_read":False},"variants":{}}
 for c in CS:
  oof=np.zeros((len(y),40),np.float32);fs=[]
  for f in range(3):
   tr=fold!=f;va=~tr;m=model(c);m.fit(x[tr],y[tr]);oof[va]=fullprob(m,x[va]);fs.append({"fold":f,"rows":int(va.sum()),"correct":int(np.sum(oof[va].argmax(1)==y[va]))})
  m=model(c);m.fit(x,y);tp=fullprob(m,tx);key=f"c{str(c).replace('.','p')}";saved[key+"_probability"]=oof;saved["test_"+key+"_probability"]=tp;report["variants"][key]={"correct":int(np.sum(oof.argmax(1)==y)),"accuracy":float(np.mean(oof.argmax(1)==y)),"folds":fs}
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"predictions.npz",**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
