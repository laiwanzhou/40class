"""Strict outer-cross-fit arbitrator between P205 and native P150."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import p89_build_dual_consensus_submission as io
from p190_inductive_p150_meta_distillation import S
from p204_p203_calibrated_p150_residual import enrich,testx
from p191_source_truth_calibrated_p150_distillation import choose
from p117_transductive_multicandidate_router import load_candidate_splits,one_hot
from p134_frozen_repeat_consensus import probability_lookup
from p216_native_p150_test import test_lookup
from p214_p128_meta_test import read_pred

H=Path(__file__).resolve().parent;O=H/"runs/p217_p205_p150_arbitrator_v1";P205=H/"runs/p205_fixed_p203_p150_residual_v1/oof_predictions.npz";P205T=H/"runs/p205_fixed_p203_p150_residual_v1/submission_p205_fixed_residual.csv";P150=H/"runs/p150_repeat_branch_confidence_selector_v1/predictions.npz";P216=H/"runs/p216_native_p150_test_v1/predictions.npz";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
CS=(.0003,.001,.003,.01,.03,.1,.3)
def align(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def scalars(bank,a,b):
 n,k,c=bank.shape;r=np.arange(n)[:,None];j=np.arange(k)[None,:];hard=bank.argmax(2);ordered=np.sort(bank,2)[:,:,-2:];mx=ordered[:,:,1];margin=ordered[:,:,1]-ordered[:,:,0];ent=-np.sum(bank*np.log(np.clip(bank,1e-8,1)),2)/np.log(40);pa=bank[r,j,a[:,None]];pb=bank[r,j,b[:,None]]
 return np.concatenate((pa,pb,pb-pa,(hard==a[:,None]).astype(np.float32),(hard==b[:,None]).astype(np.float32),mx,margin,ent,one_hot(a),one_hot(b),one_hot(a)-one_hot(b),np.column_stack((a==b,a!=b))),1).astype(np.float32)
def factories():
 d={f"log_{c:g}":(lambda c=c:make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight="balanced",solver="liblinear",max_iter=1500))) for c in CS}
 d.update({"extra_d4":lambda:ExtraTreesClassifier(n_estimators=600,max_depth=4,min_samples_leaf=3,max_features="sqrt",class_weight="balanced",random_state=21701,n_jobs=-1),"extra_d7":lambda:ExtraTreesClassifier(n_estimators=600,max_depth=7,min_samples_leaf=3,max_features="sqrt",class_weight="balanced",random_state=21702,n_jobs=-1)})
 return d
def prob(m,x):return m.predict_proba(x)[:,list(m.classes_).index(1)]
def cross_prob(data,names,factory):
 out=[]
 for target in names:
  train=[n for n in names if n!=target];x=np.concatenate([data[n]["x"] for n in train]);y=np.concatenate([data[n]["target"] for n in train]);m=factory();m.fit(x,y);out.append(prob(m,data[target]["x"]))
 return np.concatenate(out)
def select_config(data,names):
 labels=np.concatenate([data[n]["labels"] for n in names]);a=np.concatenate([data[n]["a"] for n in names]);b=np.concatenate([data[n]["b"] for n in names]);users=np.concatenate([data[n]["users"] for n in names]);coh=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in names]);gain=(b==labels).astype(int)-(a==labels).astype(int);dis=a!=b;best=None
 for name,factory in factories().items():
  score=cross_prob(data,names,factory);sel=choose(score,gain,dis,users,coh);key=(sel["minimum_cohort_gain"]>=0,sel["net"],sel["rescue"],-sel["harm"],sel["minimum_user_gain"],-sel["changed"])
  if best is None or key>best[0]:best=(key,name,sel,score)
 return best
def main():
 train=load_candidate_splits(full_visual_bank=True,structured_bank=True,legacy_visual_bank=True,hand_object_bank=True,vjepa_dense_bank=True,nonvisual_bank=True,hierarchical_bank=True,epic_bank=True,expanded_bank=True);lk=probability_lookup(train);meta=enrich();p205=np.load(P205);mp={v:(int(y),int(x)) for v,y,x in zip(p205["sample_ids"].astype(str),p205["labels"],p205["prediction"])};p150=np.load(P150);data={}
 for n in S:
  ids=meta[n]["ids"];a=np.asarray([mp[v][1] for v in ids]);b=p150[f"{n}_prediction"].astype(int);labels=np.asarray([mp[v][0] for v in ids]);bank=np.stack([lk[v] for v in ids]);x=np.concatenate((meta[n]["x"],scalars(bank,a,b)),1);dec=a!=b;target=(b[dec]==labels[dec]).astype(int);data[n]={"ids":ids,"labels":labels,"users":meta[n]["users"],"a":a,"b":b,"x_all":x,"x":x[dec],"target":target,"dec":dec}
 # Models train only on decisive disagreements; score arrays are expanded to all rows for thresholding.
 def packed(names,factory):
  scores=[]
  for target in names:
   tr=[n for n in names if n!=target];x=np.concatenate([data[n]["x"] for n in tr]);y=np.concatenate([data[n]["target"] for n in tr]);m=factory();m.fit(x,y);s=np.zeros(len(data[target]["labels"]));s[data[target]["dec"]]=prob(m,data[target]["x"]);scores.append(s)
  return np.concatenate(scores)
 def pick(names):
  labels=np.concatenate([data[n]["labels"] for n in names]);a=np.concatenate([data[n]["a"] for n in names]);b=np.concatenate([data[n]["b"] for n in names]);users=np.concatenate([data[n]["users"] for n in names]);coh=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in names]);gain=(b==labels).astype(int)-(a==labels).astype(int);best=None
  for name,f in factories().items():
   s=packed(names,f);sel=choose(s,gain,a!=b,users,coh);key=(sel["minimum_cohort_gain"]>=0,sel["net"],sel["rescue"],-sel["harm"],sel["minimum_user_gain"],-sel["changed"])
   if best is None or key>best[0]:best=(key,name,sel,s)
  return best
 report={"stage":"P217_P205_P150_arbitrator","status":"complete","protocol":{"alternatives":["P205","native P150"],"model_training":"decisive source disagreements only","threshold":"source cross-prediction truth","held_labels_used_for_selection":False,"user_id_used_as_feature":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];_,name,sel,_=pick(src);x=np.concatenate([data[n]["x"] for n in src]);y=np.concatenate([data[n]["target"] for n in src]);m=factories()[name]();m.fit(x,y);score=np.zeros(len(data[held]["labels"]));score[data[held]["dec"]]=prob(m,data[held]["x"]);q=data[held]["dec"]&(score>=sel["threshold"]);out=data[held]["a"].copy();out[q]=data[held]["b"][q];outs[held]=out;bc=int(np.sum(data[held]["a"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"model":name,"source_selection":sel,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(q.sum()),"rescue":int(np.sum(q&(data[held]["a"]!=data[held]["labels"])&(out==data[held]["labels"]))),"harm":int(np.sum(q&(data[held]["a"]==data[held]["labels"])&(out!=data[held]["labels"])))}}
  print(json.dumps({"held":held,"model":name,**report["cohorts"][held]["held"]}),flush=True)
 labels=np.concatenate([data[n]["labels"] for n in S]);a=np.concatenate([data[n]["a"] for n in S]);b=np.concatenate([data[n]["b"] for n in S]);out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(a==labels));report["aggregate"]={"rows":len(labels),"p205_correct":bc,"p150_correct":int(np.sum(b==labels)),"oracle_correct":int(np.sum((a==labels)|(b==labels))),"correct":cor,"accuracy":cor/len(labels),"net_vs_p205":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S],"gap_to_0.91":int(np.ceil(.91*len(labels))-cor)}
 _,name,sel,_=pick(list(S));x=np.concatenate([data[n]["x"] for n in S]);y=np.concatenate([data[n]["target"] for n in S]);m=factories()[name]();m.fit(x,y);ids,meta_tx,_=testx();p216=np.load(P216);ta=read_pred(P205T);tb=align(p216["prediction"],p216["sample_ids"],ids);_,tlk,_=test_lookup(train,list(next(iter(train.values())).candidates));bank=np.stack([tlk[v] for v in ids]);tx=np.concatenate((meta_tx,scalars(bank,ta,tb)),1);dec=ta!=tb;score=np.zeros(len(ids));score[dec]=prob(m,tx[dec]);q=dec&(score>=sel["threshold"]);tout=ta.copy();tout[q]=tb[q];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p217_p205_p150_arbitrated.csv";io.write_submission(sub,io.read_rows(P89),tout);report["test"]={"model":name,"oof_selection":sel,"p205_p150_disagreements":int(dec.sum()),"selected_p150":int(q.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=np.concatenate([data[n]["ids"] for n in S]),labels=labels,p205_prediction=a,p150_prediction=b,prediction=out,test_sample_ids=ids,test_p205_prediction=ta,test_p150_prediction=tb,test_score=score,test_prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps({"aggregate":report["aggregate"],"test":report["test"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
