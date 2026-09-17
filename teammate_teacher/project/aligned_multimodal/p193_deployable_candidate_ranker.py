"""Candidate-conditioned reliability ranker over the 25 deployable experts."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import p89_build_dual_consensus_submission as io
from p173_vjepa_augmented_group_teacher import build_train_bank,build_test_bank

H=Path(__file__).resolve().parent;O=H/"runs/p193_deployable_candidate_ranker_v1";P128=H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz";P128T=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz";P180=H/"runs/p180_sequence_micro_teacher_v1/oof_predictions.npz";P180T=H/"runs/p180_sequence_micro_teacher_v1/submission_p180_sequence_micro.csv";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def al(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def csvpred(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def bank():
 tr,names=build_train_bank();p=np.load(P128)
 for n in S:
  q=tr[n];q["bank"]=np.concatenate((q["bank"],al(p["probabilities"],p["sample_ids"],q["ids"])[:,None,:]),axis=1)
 return tr,[*names,"p128_hierarchical_multimodal"]
def feature_matrix(prob,base):
 n,k,_=prob.shape;pred=prob.argmax(2);rows=np.arange(n)[:,None];idx=np.arange(k)[None,:];top=np.sort(prob,axis=2)[:,:,-2:];vote=np.stack([(pred==pred[:,j:j+1]).mean(1) for j in range(k)],1);basevote=np.mean(pred==base[:,None],axis=1);scalar=np.stack((prob[rows,idx,pred],top[:,:,1]-top[:,:,0],-np.sum(prob*np.log(np.clip(prob,1e-8,1)),axis=2)/np.log(40),prob[rows,idx,base[:,None]],prob[rows,idx,pred]-prob[rows,idx,base[:,None]],vote,np.broadcast_to(basevote[:,None],(n,k)),pred!=base[:,None]),axis=2)
 ident=np.eye(k,dtype=np.float32)[None].repeat(n,0);cl=np.eye(40,dtype=np.float32)[pred];bc=np.eye(40,dtype=np.float32)[base][:,None].repeat(k,1);return np.concatenate((scalar,ident,cl,bc),axis=2).astype(np.float32),pred
def add_current(prob,base):
 p=np.full((len(base),1,40),.0005,np.float32);p[np.arange(len(base)),0,base]=.9805;return np.concatenate((prob,p),axis=1)
def fit(x,target):
 m=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=800,solver="liblinear",class_weight="balanced"));m.fit(x.reshape(-1,x.shape[-1]),target.reshape(-1));return m
def score(m,x):return m.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(x.shape[:2])
def choose_threshold(alt,base,labels,gap,users):
 gain=(alt==labels).astype(int)-(base==labels).astype(int);d=alt!=base;vals=np.unique(np.concatenate(([-np.inf,np.inf],np.linspace(-.5,.5,201),np.quantile(gap[d],np.linspace(.1,.9,9)) if d.any() else [np.inf])));best=None
 for t in vals:
  q=d&(gap>=t);per={u:int(gain[q&(users==u)].sum()) for u in sorted(set(users.tolist()))};r=int(np.sum(q&(gain>0)));h=int(np.sum(q&(gain<0)));row={"threshold":float(t),"changed":int(q.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(per.values()),"per_user":per};key=(row["minimum_user_gain"]>=0,row["net"],r,-h,-row["changed"])
  if best is None or key>best[0]:best=(key,row)
 return best[1]
def main():
 tr,names=bank();p180=np.load(P180);mp={v:int(x) for v,x in zip(p180["sample_ids"].astype(str),p180["prediction"])}
 data={}
 for n in S:
  q=tr[n];base=np.asarray([mp[v] for v in q["ids"]]);prob=add_current(q["bank"],base);x,pred=feature_matrix(prob,base);target=pred==q["labels"][:,None];data[n]={**q,"base":base,"prob":prob,"x":x,"pred":pred,"target":target}
 report={"stage":"P193_deployable_candidate_ranker","status":"complete","protocol":{"candidate_count":len(names)+1,"candidate_names":[*names,"P180_current"],"source_LOSO_threshold":True,"held_labels_used_for_training":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];sx=np.concatenate([data[n]["x"] for n in src]);st=np.concatenate([data[n]["target"] for n in src]);susers=np.concatenate([data[n]["users"] for n in src]);slabels=np.concatenate([data[n]["labels"] for n in src]);sbase=np.concatenate([data[n]["base"] for n in src]);spred=np.concatenate([data[n]["pred"] for n in src]);nested=np.zeros(spred.shape)
  for u in sorted(set(susers.tolist())):
   keep=susers!=u;m=fit(sx[keep],st[keep]);nested[susers==u]=score(m,sx[susers==u])
  bi=spred.shape[1]-1;altidx=np.argmax(nested[:,:bi],axis=1);alt=spred[np.arange(len(spred)),altidx];gap=nested[np.arange(len(spred)),altidx]-nested[:,bi];sel=choose_threshold(alt,sbase,slabels,gap,susers);m=fit(sx,st);hs=score(m,data[held]["x"]);hi=np.argmax(hs[:,:bi],1);ha=data[held]["pred"][np.arange(len(hi)),hi];hg=hs[np.arange(len(hi)),hi]-hs[:,bi];mask=(ha!=data[held]["base"])&(hg>=sel["threshold"]);out=data[held]["base"].copy();out[mask]=ha[mask];outs[held]=out;bc=int(np.sum(data[held]["base"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"source_threshold":sel,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(mask.sum())}}
 labels=np.concatenate([data[n]["labels"] for n in S]);base=np.concatenate([data[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(base==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p180":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]}
 # final OOF LOSO threshold and full refit Test
 allx=np.concatenate([data[n]["x"] for n in S]);allt=np.concatenate([data[n]["target"] for n in S]);users=np.concatenate([data[n]["users"] for n in S]);pred=np.concatenate([data[n]["pred"] for n in S]);nested=np.zeros(pred.shape)
 for u in sorted(set(users.tolist())):
  keep=users!=u;m=fit(allx[keep],allt[keep]);nested[users==u]=score(m,allx[users==u])
 bi=pred.shape[1]-1;ai=np.argmax(nested[:,:bi],1);alt=pred[np.arange(len(pred)),ai];gap=nested[np.arange(len(pred)),ai]-nested[:,bi];sel=choose_threshold(alt,base,labels,gap,users)
 test=build_test_bank(names[:21],names[:24]);pt=np.load(P128T);test["bank"]=np.concatenate((test["bank"],al(pt["probabilities"],pt["sample_ids"],test["ids"])[:,None,:]),axis=1);tb=csvpred(P180T);tp=add_current(test["bank"],tb);tx,tpr=feature_matrix(tp,tb);m=fit(allx,allt);ts=score(m,tx);ti=np.argmax(ts[:,:bi],1);ta=tpr[np.arange(len(ti)),ti];tg=ts[np.arange(len(ti)),ti]-ts[:,bi];mask=(ta!=tb)&(tg>=sel["threshold"]);tout=tb.copy();tout[mask]=ta[mask];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p193_candidate_ranker.csv";io.write_submission(sub,io.read_rows(P89),tout);report["test"]={"oof_threshold":sel,"changes_vs_p180":int(mask.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=test["ids"],base_prediction=tb,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
