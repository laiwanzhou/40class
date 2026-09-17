"""Resumable hand-workspace optical-flow cache and strict OOF expert."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from p90_teacher_common import load_protocol
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;PIX=H/"runs/p86_visual_pixel_cache_t16_r160_v12";O=H/"runs/p326_workspace_optical_flow_expert_v1";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0");STEP=48;DIM=1344
def rows():
 with (PIX/"rows.csv").open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def flow_step(a,b):
 a=cv2.resize(a,(80,80),interpolation=cv2.INTER_AREA);b=cv2.resize(b,(80,80),interpolation=cv2.INTER_AREA);flow=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0);fx,fy=flow[...,0],flow[...,1];mag=np.sqrt(fx*fx+fy*fy);ang=(np.arctan2(fy,fx)+2*np.pi)%(2*np.pi);edges=np.linspace(0,2*np.pi,9);parts=[]
 h=np.histogram(ang,bins=edges,weights=mag)[0];parts.extend((h/(h.sum()+1e-6)).tolist())
 for yy in (slice(0,40),slice(40,80)):
  for xx in (slice(0,40),slice(40,80)):
   q=np.histogram(ang[yy,xx],bins=edges,weights=mag[yy,xx])[0];parts.extend((q/(q.sum()+1e-6)).tolist())
 parts.extend((float(mag.mean()),float(mag.std()),float(np.quantile(mag,.9)),float((mag>1.).mean())));div=np.gradient(fx,axis=1)+np.gradient(fy,axis=0);curl=np.gradient(fy,axis=1)-np.gradient(fx,axis=0);parts.extend((float(np.abs(div).mean()),float(div.std()),float(np.abs(curl).mean()),float(curl.std())));out=np.asarray(parts,np.float32)
 if out.shape!=(STEP,):raise RuntimeError(out.shape)
 return out
def window(v):
 seq=np.stack([flow_step(v[i],v[i+1]) for i in range(len(v)-1)]);e=seq[:7].mean(0);l=seq[7:].mean(0);return np.concatenate((seq.mean(0),seq.std(0),seq.max(0),e,l,l-e,np.abs(l-e))).astype(np.float32)
def trial(images):
 a=window(images[0]);b=window(images[1]);return np.concatenate((a,b,b-a,np.abs(b-a))).astype(np.float32)
def cache():
 O.mkdir(parents=True,exist_ok=True);images=np.load(PIX/"images.npy",mmap_mode="r");fp=O/"features.npy";dp=O/"done.npy";new_done=not dp.exists();x=np.lib.format.open_memmap(fp,mode="r+" if fp.exists() else "w+",dtype=np.float16,shape=(len(images),DIM));done=np.lib.format.open_memmap(dp,mode="r+" if dp.exists() else "w+",dtype=np.bool_,shape=(len(images),))
 if new_done:done[:]=False
 for i in np.flatnonzero(~np.asarray(done)):
  x[i]=trial(np.asarray(images[i,:,:,2],np.uint8)).astype(np.float16);done[i]=True
  if (i+1)%100==0:x.flush();done.flush();print(json.dumps({"stage":"flow_cache","done":int(np.sum(done)),"total":len(done)}),flush=True)
 x.flush();done.flush();return x
def oof(x):
 p=load_protocol();r=rows();pixel_ids=np.asarray([q["sample_id"] for q in r]);lookup={q:i for i,q in enumerate(pixel_ids)}
 if set(pixel_ids.tolist())!=set(p.sample_ids.tolist()) or len(lookup)!=len(pixel_ids):raise RuntimeError("pixel universe")
 positions=np.asarray([lookup[q] for q in p.sample_ids],int);x=np.asarray(x[positions]);ids=p.sample_ids;labels=p.labels
 prob=np.zeros((len(labels),40),float);folds=[]
 for f in range(3):
  tr=p.train_indices(f);va=p.val_indices(f);m=ExtraTreesClassifier(n_estimators=700,max_depth=18,min_samples_leaf=2,max_features="sqrt",class_weight="balanced",random_state=32600+f,n_jobs=-1);m.fit(np.asarray(x[tr],np.float32),labels[tr]);prob[va]=m.predict_proba(np.asarray(x[va],np.float32));folds.append({"fold":f,"correct":int(np.sum(prob[va].argmax(1)==labels[va])),"rows":len(va)})
 splits=load_candidate_splits();basez=np.load(P310);base=basez["prediction"];y=basez["labels"];pos={q:i for i,q in enumerate(ids)};pred=np.concatenate([prob[[pos[q] for q in splits[n].split.sample_ids.astype(str)]].argmax(1) for n in S]);q=pred!=base;rescue=int(np.sum(q&(base!=y)&(pred==y)));harm=int(np.sum(q&(base==y)&(pred!=y)));report={"stage":"P326_workspace_optical_flow_expert","status":"complete","protocol":{"input":"P86 hand-workspace grayscale 2x16x160x160","flow":"Farneback on 80x80","descriptor":"orientation, 2x2 spatial flow, magnitude, divergence, curl, early/late delta","dimensions":DIM,"classifier":"ExtraTrees 700 depth18 leaf2 sqrt balanced","strict_subject_folds":True,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"metrics":{"correct":int(np.sum(prob.argmax(1)==labels)),"accuracy":float(np.mean(prob.argmax(1)==labels)),"folds":folds},"vs_p310":{"changed":int(q.sum()),"rescue":rescue,"harm":harm,"oracle_gain":rescue,"direct_net":rescue-harm},"gate":{"required_oracle_rescue":8,"pass":rescue>=8}};np.savez_compressed(O/"oof_predictions.npz",sample_ids=ids,labels=labels,probability=prob.astype(np.float32));(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: first verified workspace optical-flow descriptor in the project.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
def main():
 print("P326 tests whether explicit dense hand-workspace flow adds motion evidence beyond P310.",flush=True);x=cache();oof(x)
if __name__=="__main__":main()
