"""Candidate-conditioned source-safe gain ranker over deployable physical experts."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import p89_build_dual_consensus_submission as io
from p90_crossuser_visual_router import load_splits
from p191_source_truth_calibrated_p150_distillation import choose
H=Path(__file__).resolve().parent;O=H/"runs/p291_physical_gain_ranker_v1";BASE=H/"runs/p270_fixed_emission065_transition045_v1/predictions.npz";BASET=H/"runs/p270_fixed_emission065_transition045_v1/submission_p270_fixed_sequence.csv";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";S=("H1_selection","H2_confirmation","H3_independent_fold0");SRC=((H/"runs/p231_depth_thermal_ir_oof_v1/oof_predictions.npz","ir_thermal_probability",H/"runs/p233_depth_thermal_ir_test_heads_v1/test_predictions.npz","ir_thermal_probability"),(H/"runs/p238_physical_token_transformer_oof_v1/oof_predictions.npz","probability",H/"runs/p239_physical_token_transformer_test_v1/test_predictions.npz","probability"),(H/"runs/p253_repeat_physical_transformer_oof_v1/oof_predictions.npz","probability",H/"runs/p254_repeat_physical_transformer_test_v1/test_predictions.npz","probability"),(H/"runs/p260_semantic_physical_transformer_oof_v1/oof_predictions.npz","probability",H/"runs/p261_semantic_physical_transformer_test_v1/test_predictions.npz","probability"),(H/"runs/p266_modality_specific_physical_oof_v1/oof_predictions.npz","probability",H/"runs/p267_modality_specific_physical_test_v1/test_predictions.npz","probability"),(H/"runs/p277_physical_centroid_expert_v1/predictions.npz","probability",H/"runs/p277_physical_centroid_expert_v1/predictions.npz","test_probability"),(H/"runs/p287_physical_posterior_stacker_v1/predictions.npz","c0p001_probability",H/"runs/p287_physical_posterior_stacker_v1/predictions.npz","test_c0p001_probability"))
def cp(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def al(v,s,t):d={q:i for i,q in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[q] for q in t.astype(str)])]
def feat(bank,base):
 n,k,_=bank.shape;pred=bank.argmax(2);r=np.arange(n)[:,None];j=np.arange(k)[None,:];top=np.sort(bank,2)[:,:,-2:];vote=np.stack([(pred==pred[:,i:i+1]).mean(1) for i in range(k)],1);mean=bank.mean(1);sc=np.stack((bank[r,j,pred],top[:,:,1]-top[:,:,0],-np.sum(bank*np.log(np.clip(bank,1e-8,1)),2)/np.log(40),bank[r,j,base[:,None]],bank[r,j,pred]-bank[r,j,base[:,None]],vote,mean[r,pred]-mean[np.arange(n),base][:,None]),2);ident=np.eye(k,dtype=np.float32)[None].repeat(n,0);pc=np.eye(40,dtype=np.float32)[pred];bc=np.eye(40,dtype=np.float32)[base][:,None].repeat(k,1);return np.concatenate((sc,ident,pc,bc),2).astype(np.float32),pred
def models():return (make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1200,solver="liblinear",class_weight="balanced")),ExtraTreesClassifier(n_estimators=500,max_depth=7,min_samples_leaf=4,max_features="sqrt",class_weight="balanced",random_state=291,n_jobs=-1))
def fit(x,target):
 ms=models()
 for m in ms:m.fit(x,target)
 return ms
def score(ms,x):return np.mean([m.predict_proba(x)[:,1] for m in ms],0)
def nested(data,names):
 x=np.concatenate([data[n]["x"] for n in names]);gain=np.concatenate([data[n]["gain"] for n in names]);users=np.concatenate([np.repeat(data[n]["users"],data[n]["x"].shape[1]) for n in names]);flat=x.reshape(-1,x.shape[-1]);g=gain.reshape(-1);out=np.zeros(len(g));dec=g!=0
 for u in sorted(set(users.tolist())):
  tr=dec&(users!=u);te=users==u
  if len(np.unique((g[tr]>0).astype(int)))==2:out[te]=score(fit(flat[tr],(g[tr]>0).astype(int)),flat[te])
 return out.reshape(x.shape[:2])
def choose_rows(scores,pred,base):
 ix=scores.argmax(1);alt=pred[np.arange(len(ix)),ix];s=scores[np.arange(len(ix)),ix];return alt,s
def main():
 splits=load_splits();b=np.load(BASE);train_src=[np.load(x[0]) for x in SRC];data={}
 for n in S:
  ids=splits[n].sample_ids.astype(str);base=b[f"{n}_held_prediction"].astype(int);labels=splits[n].labels.astype(int);bank=np.stack([al(z[k],z["sample_ids"],ids) for z,(_,k,_,_) in zip(train_src,SRC)],1);x,pred=feat(bank,base);gain=(pred==labels[:,None]).astype(np.int8)-(base==labels).astype(np.int8)[:,None];data[n]={"ids":ids,"labels":labels,"base":base,"users":splits[n].users.astype(str),"bank":bank,"x":x,"pred":pred,"gain":gain}
 report={"stage":"P291_physical_gain_ranker","status":"complete","protocol":{"candidate_count":len(SRC),"training":"decisive rescue-vs-harm instances","source_user_LOSO_threshold":True,"held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];ns=nested(data,src);sp=np.concatenate([data[n]["pred"] for n in src]);sb=np.concatenate([data[n]["base"] for n in src]);sl=np.concatenate([data[n]["labels"] for n in src]);su=np.concatenate([data[n]["users"] for n in src]);co=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in src]);alt,ss=choose_rows(ns,sp,sb);gain=(alt==sl).astype(int)-(sb==sl).astype(int);sel=choose(ss,gain,alt!=sb,su,co);x=np.concatenate([data[n]["x"] for n in src]);g=np.concatenate([data[n]["gain"] for n in src]);flat=x.reshape(-1,x.shape[-1]);gf=g.reshape(-1);decisive=gf!=0;ms=fit(flat[decisive],(gf[decisive]>0).astype(int));hs=score(ms,data[held]["x"].reshape(-1,x.shape[-1])).reshape(data[held]["x"].shape[:2]);ha,hscore=choose_rows(hs,data[held]["pred"],data[held]["base"]);q=(ha!=data[held]["base"])&(hscore>=sel["threshold"]);out=data[held]["base"].copy();out[q]=ha[q];outs[held]=out;bc=int(np.sum(data[held]["base"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"source_selection":sel,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(q.sum()),"rescue":int(np.sum(q&(data[held]["base"]!=data[held]["labels"])&(out==data[held]["labels"]))),"harm":int(np.sum(q&(data[held]["base"]==data[held]["labels"])&(out!=data[held]["labels"])))}}
 labels=np.concatenate([data[n]["labels"] for n in S]);base=np.concatenate([data[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p270":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]}
 ns=nested(data,list(S));pred=np.concatenate([data[n]["pred"] for n in S]);alt,ss=choose_rows(ns,pred,base);gain=(alt==labels).astype(int)-(base==labels).astype(int);users=np.concatenate([data[n]["users"] for n in S]);co=np.concatenate([np.full(len(data[n]["labels"]),n,dtype=object) for n in S]);sel=choose(ss,gain,alt!=base,users,co);x=np.concatenate([data[n]["x"] for n in S]);g=np.concatenate([data[n]["gain"] for n in S]);flat=x.reshape(-1,x.shape[-1]);gf=g.reshape(-1);decisive=gf!=0;ms=fit(flat[decisive],(gf[decisive]>0).astype(int));tids=b["sample_ids"].astype(str);tb=cp(BASET);test_src=[np.load(x[2]) for x in SRC];tbank=[]
 for z,(_,_,_,k) in zip(test_src,SRC):sid=z["test_sample_ids"] if "test_sample_ids" in z.files and len(z[k])==len(z["test_sample_ids"]) else z["sample_ids"];tbank.append(al(z[k],sid,tids))
 tbank=np.stack(tbank,1);tx,tpred=feat(tbank,tb);ts=score(ms,tx.reshape(-1,tx.shape[-1])).reshape(tx.shape[:2]);ta,tss=choose_rows(ts,tpred,tb);q=(ta!=tb)&(tss>=sel["threshold"]);tout=tb.copy();tout[q]=ta[q];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p291_physical_ranker.csv";io.write_submission(sub,io.read_rows(P89),tout);report["test"]={"oof_selection":sel,"changes_vs_p270":int(q.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=tids,base_prediction=tb,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
