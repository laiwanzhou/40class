"""Strict OOF/Test PCA-cosine class-centroid expert on physical embeddings."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p277_physical_centroid_expert_v1";TR=(R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz",R/"runs/p91_videomaev2_depth_fold0_v1/complete_features.npz",H/"runs/p271_thermal_early_late_v1/train_features.npz");TE=(R/"runs/p90_videomaev2_distilled_test_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_test_v1/complete_features.npz",H/"runs/p232_depth_thermal_test_features_v1/depth_features.npz",H/"runs/p271_thermal_early_late_v1/test_features.npz");IDS=H/"runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
def flatten(z):
 v=z["features"].astype(np.float32).reshape(len(z["features"]),-1,768);v/=np.maximum(np.linalg.norm(v,axis=2,keepdims=True),1e-8);return v.reshape(len(v),-1)
def align(v,s,ids):
 out=np.zeros((len(ids),v.shape[1]),np.float32);pos={q:i for i,q in enumerate(ids)};rows=np.asarray([pos[q] for q in s.astype(str)]);out[rows]=v;mask=np.zeros(len(ids),bool);mask[rows]=True;return out,mask
def probability(xt,y,xe):
 cent=np.stack([normalize(xt[y==c].mean(0,keepdims=True))[0] for c in range(40)]);score=xe@cent.T/.1;score-=score.max(1,keepdims=True);p=np.exp(score);return p/p.sum(1,keepdims=True)
def main():
 z=[np.load(p) for p in TR];ids=z[0]["sample_ids"].astype(str);y=z[0]["labels"].astype(int);fold=z[0]["fold_id"].astype(int);x=np.concatenate([flatten(q) for q in z],1);oof=np.zeros((len(y),40),np.float32);folds=[]
 for f in range(3):
  tr=fold!=f;va=~tr;pca=PCA(256,whiten=True,random_state=277,svd_solver="randomized");xt=normalize(pca.fit_transform(x[tr]));xv=normalize(pca.transform(x[va]));oof[va]=probability(xt,y[tr],xv);folds.append({"fold":f,"rows":int(va.sum()),"correct":int(np.sum(oof[va].argmax(1)==y[va]))})
 tids=np.load(IDS)["sample_ids"].astype(str);tb=[];masks=[]
 for path in TE:
  q=np.load(path);v,m=align(flatten(q),q["sample_ids"],tids);tb.append(v);masks.append(m)
 tx=np.concatenate(tb,1);pca=PCA(256,whiten=True,random_state=277,svd_solver="randomized");xt=normalize(pca.fit_transform(x));te=normalize(pca.transform(tx));tp=probability(xt,y,te);available=masks[0]&masks[1]&masks[2];report={"stage":"P277_physical_centroid_expert","status":"complete","protocol":{"feature":"tokenwise L2 IR/Depth/Thermal-early-late","projection":"source-only PCA256 whiten","classifier":"cosine class centroid","temperature":.1,"strict_subject_folds":True,"test_labels_read":False},"oof":{"rows":len(y),"correct":int(np.sum(oof.argmax(1)==y)),"accuracy":float(np.mean(oof.argmax(1)==y)),"folds":folds},"test":{"rows":len(tids),"available":int(available.sum())}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"predictions.npz",sample_ids=ids,labels=y,probability=oof,test_sample_ids=tids,test_probability=tp.astype(np.float32),test_available=available);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
