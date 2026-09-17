"""Inductively distill P150 decisions from deployable candidate probabilities."""

from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import p89_build_dual_consensus_submission as io
from p90_crossuser_visual_router import load_splits
from deploy_p168_historical_micro_union import deploy_micro

H=Path(__file__).resolve().parent;O=H/"runs/p190_inductive_p150_meta_distillation_v1";S=("H1_selection","H2_confirmation","H3_independent_fold0")
P150=H/"runs/p150_repeat_branch_confidence_selector_v1/predictions.npz";A18=H.parent/"runs/a18_subject_safe_revalidation/oof_predictions.npz";P142=H/"runs/p142_vjepa_token_transformer_three_seed_v2/oof_predictions.npz";P144=H/"runs/p144_vjepa_hand_interaction_transformer_three_seed_v1/oof_predictions.npz";P149=H/"runs/p149_vjepa_repeat_consistency_three_seed_v2/oof_predictions.npz";P128=H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz";P177=H/"runs/p177_p128_vjepa_group_teacher_v1/predictions.npz";P179=H/"runs/p179_p177_soft_sequence_gate_v1/predictions.npz";MIC=H/"runs/p89_verified_micro_union_audit_v1/validation_predictions.npz";P180=H/"runs/p180_sequence_micro_teacher_v1/oof_predictions.npz"
P89T=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";P89P=H/"runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz";IMUT=H/"runs/p3_sd_imu_rf_full18/test_logits.npz";A18T=H/"runs/a18_full_teacher_test_v1/test_predictions.npz";P172=H/"runs/p172_vjepa_token_heads_test_v1/test_predictions.npz";P128T=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz";P177T=P177;P179T=P179;P180T=H/"runs/p180_sequence_micro_teacher_v1/submission_p180_sequence_micro.csv"

def sm(v):v=np.asarray(v,float);v-=v.max(1,keepdims=True);p=np.exp(v);return p/p.sum(1,keepdims=True)
def norm(v):v=np.clip(np.asarray(v,float),1e-10,None);return v/v.sum(1,keepdims=True)
def align(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def predcsv(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def scalars(p):
 q=np.sort(p,axis=1)[:,-2:];e=-np.sum(p*np.log(np.clip(p,1e-10,1)),axis=1)/np.log(40);return np.column_stack((p.max(1),q[:,1]-q[:,0],e))
def features(probabilities,predictions):
 mats=[]
 for p in probabilities:mats.extend((np.sqrt(np.clip(p,0,1)),np.log(np.clip(p,1e-6,1)),scalars(p)))
 for v in predictions:
  x=np.zeros((len(v),40),np.float32);x[np.arange(len(v)),v]=1;mats.append(x)
 return np.concatenate(mats,axis=1).astype(np.float32)
def models():return {
 "log_c003":lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1200,solver="lbfgs")),
 "log_c01":lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1200,solver="lbfgs")),
 "extra":lambda:ExtraTreesClassifier(n_estimators=500,max_depth=14,min_samples_leaf=4,max_features="sqrt",class_weight="balanced",random_state=19001,n_jobs=-1),
 "hist":lambda:HistGradientBoostingClassifier(learning_rate=.05,max_iter=180,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=8,random_state=19002),}
def load_oof():
 splits=load_splits();a=np.load(A18);p142=np.load(P142);p144=np.load(P144);p149=np.load(P149);p128=np.load(P128);p177=np.load(P177);p179=np.load(P179);mic=np.load(MIC);p180=np.load(P180);p150=np.load(P150); sources=[]
 for z,k in ((a,"best_session_probability"),(p142,"probability"),(p144,"probability"),(p149,"probability"),(p128,"probabilities")):sources.append((z["sample_ids"].astype(str),z[k]))
 p180map={v:int(x) for v,x in zip(p180["sample_ids"].astype(str),p180["prediction"])};out={};prefix={S[0]:"h1",S[1]:"h2",S[2]:"h3"}
 for n in S:
  q=splits[n];ids=q.sample_ids.astype(str);probs=[norm(q.safe_probability),*[norm(align(v,s,ids)) for s,v in sources],norm(p177[f"{n}_held_probability"])];preds=[q.safe_prediction.astype(int),p177[f"{n}_held_prediction"].astype(int),p179[f"{n}_held_prediction"].astype(int),mic[f"{prefix[n]}_union"].astype(int),np.asarray([p180map[v] for v in ids])];target=p150[f"{n}_prediction"].astype(int)
  out[n]={"ids":ids,"labels":q.labels.astype(int),"users":q.users.astype(str),"x":features(probs,preds),"target":target,"base":preds[-1]}
 return out
def cat(parts,key):return np.concatenate([p[key] for p in parts])
def select(source_names,data):
 scores={}
 for name,factory in models().items():
  vals=[]
  for train_name,test_name in ((source_names[0],source_names[1]),(source_names[1],source_names[0])):
   m=factory();m.fit(data[train_name]["x"],data[train_name]["target"]);pred=m.predict(data[test_name]["x"]);vals.append(float(np.mean(pred==data[test_name]["target"])))
  scores[name]={"agreements":vals,"minimum":min(vals),"mean":float(np.mean(vals))}
 return max(scores,key=lambda n:(scores[n]["minimum"],scores[n]["mean"])),scores
def test_features():
 p89=np.load(P89P);ids=p89["sample_ids"].astype(str);basep=norm(p89["base_probability"]);imu=np.load(IMUT);ip=sm(align(imu["imu_logits"],imu["sample_ids"],ids)/3);safeprob=norm(.95*basep+.05*ip);a=np.load(A18T);v172=np.load(P172);p128=np.load(P128T);p177=np.load(P177T);p179=np.load(P179T);_,base,_,_,micro,_=deploy_micro();probs=[safeprob,norm(align(a["selected_probability"],a["sample_ids"],ids))]
 for k in ("p142_all_probability","p144_hand_interaction_probability","p149_repeat_consistency_probability"):
  full=safeprob.copy();rid=v172["sample_ids"].astype(str);pos={v:i for i,v in enumerate(ids)};rows=np.asarray([pos[v] for v in rid]);full[rows]=v172[k];probs.append(norm(full))
 probs.extend((norm(align(p128["probabilities"],p128["sample_ids"],ids)),norm(p177["probability"])))
 preds=[base,p177["prediction"].astype(int),p179["prediction"].astype(int),micro,predcsv(P180T)];return ids,features(probs,preds),preds[-1]
def main():
 data=load_oof();report={"stage":"P190_inductive_P150_meta_distillation","status":"complete","protocol":{"model_selected_by_cross_source_P150_agreement":True,"held_P150_targets_used_for_training":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];chosen,scores=select(src,data);m=models()[chosen]();m.fit(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"target"));pred=m.predict(data[held]["x"]);outs[held]=pred;report["cohorts"][held]={"source":src,"selected_model":chosen,"source_agreement":scores,"held":{"base_correct":int(np.sum(data[held]["base"]==data[held]["labels"])),"correct":int(np.sum(pred==data[held]["labels"])),"teacher_agreement":float(np.mean(pred==data[held]["target"])),"p150_correct":int(np.sum(data[held]["target"]==data[held]["labels"]))}}
 labels=cat([data[n] for n in S],"labels");base=cat([data[n] for n in S],"base");pred=cat([{"p":outs[n]} for n in S],"p");correct=int(np.sum(pred==labels));bc=int(np.sum(base==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":correct,"accuracy":correct/len(labels),"net_vs_p180":correct-bc,"fold_nets":[report["cohorts"][n]["held"]["correct"]-report["cohorts"][n]["held"]["base_correct"] for n in S],"p150_correct":2210}
 chosen,scores=select(list(S),data) if False else (None,None)
 # global model family by 3-way leave-one-cohort target agreement
 global_scores={}
 for name,factory in models().items():
  vals=[]
  for held in S:
   src=[n for n in S if n!=held];m=factory();m.fit(cat([data[n] for n in src],"x"),cat([data[n] for n in src],"target"));vals.append(float(np.mean(m.predict(data[held]["x"])==data[held]["target"])))
  global_scores[name]={"agreements":vals,"minimum":min(vals),"mean":float(np.mean(vals))}
 chosen=max(global_scores,key=lambda n:(global_scores[n]["minimum"],global_scores[n]["mean"]));ids,tx,tbase=test_features();m=models()[chosen]();m.fit(cat([data[n] for n in S],"x"),cat([data[n] for n in S],"target"));tout=m.predict(tx).astype(int);O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p190_meta_distilled.csv";io.write_submission(sub,io.read_rows(P89T),tout);report["test"]={"selected_model":chosen,"selection_scores":global_scores,"changes_vs_p89":int(np.sum(tout!=tbase)),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=ids,base_prediction=tbase,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
