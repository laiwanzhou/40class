"""Pair-specific source-user-safe rollback rankers from P205 to P89 safe."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import p89_build_dual_consensus_submission as io
from p193_deployable_candidate_ranker import bank
from p173_vjepa_augmented_group_teacher import build_test_bank
H=Path(__file__).resolve().parent;O=H/"runs/p211_pair_specific_rollback_ranker_v1";P128T=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz";D=H/"runs/p207_depth_geometry_full_test_v1/predictions.npz";P205=H/"runs/p205_fixed_p203_p150_residual_v1/oof_predictions.npz";P205T=H/"runs/p205_fixed_p203_p150_residual_v1/submission_p205_fixed_residual.csv";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def al(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def cp(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def feature(prob,safe,current,depth):
 r=np.arange(len(safe));s=prob[r,:,safe];c=prob[r,:,current];hard=prob.argmax(2);ds=np.concatenate([depth[k][r,safe][:,None] for k in range(3)],1);dc=np.concatenate([depth[k][r,current][:,None] for k in range(3)],1);return np.concatenate((s,c,s-c,np.column_stack((np.mean(hard==safe[:,None],1),np.mean(hard==current[:,None],1),np.mean(s,1)-np.mean(c,1))),ds,dc,ds-dc),1).astype(np.float32)
def model():return make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",solver="liblinear",max_iter=1000))
def loso(x,gain,users,pairmask):
 score=np.zeros(len(gain))
 for u in sorted(set(users.tolist())):
  te=(users==u)&pairmask;tr=(users!=u)&pairmask&(gain!=0)
  if te.any() and len(set((gain[tr]>0).tolist()))==2:
   m=model();m.fit(x[tr],(gain[tr]>0).astype(int));score[te]=m.predict_proba(x[te])[:,1]
 return score
def threshold(score,gain,users,pairmask):
 vals=np.unique(np.concatenate(([np.inf,-np.inf],np.linspace(0,1,101),np.quantile(score[pairmask],[.25,.5,.75]) if pairmask.any() else [np.inf])));best=None
 for t in vals:
  q=pairmask&(score>=t);per={u:int(gain[q&(users==u)].sum()) for u in sorted(set(users.tolist()))};r=int(np.sum(q&(gain>0)));h=int(np.sum(q&(gain<0)));row={"threshold":float(t),"selected":int(q.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(per.values()),"per_user":per};key=(row["minimum_user_gain"]>=0,row["net"],r,-h,-row["selected"])
  if best is None or key>best[0]:best=(key,row)
 return best[1]
def select(source):
 cur=source["current"];safe=source["safe"];gain=(safe==source["labels"]).astype(int)-(cur==source["labels"]).astype(int);rules={};audit=[]
 for a,b in sorted(set(zip(cur[cur!=safe].tolist(),safe[cur!=safe].tolist()))):
  pair=(cur==a)&(safe==b)
  if np.sum(pair&(gain>0))<2 or np.sum(pair&(gain<0))<2:continue
  sc=loso(source["x"],gain,source["users"],pair);sel=threshold(sc,gain,source["users"],pair);sel.update({"current":int(a),"safe":int(b)});sel["eligible"]=sel["net"]>0 and sel["rescue"]>=2 and sel["minimum_user_gain"]>=0;audit.append(sel)
  if sel["eligible"]:
   tr=pair&(gain!=0);m=model();m.fit(source["x"][tr],(gain[tr]>0).astype(int));rules[(a,b)]=(m,sel["threshold"])
 return rules,audit
def apply(data,rules):
 out=data["current"].copy();mask=np.zeros(len(out),bool)
 for (a,b),(m,t) in rules.items():
  q=(data["current"]==a)&(data["safe"]==b)
  if q.any():
   s=m.predict_proba(data["x"][q])[:,1];rows=np.flatnonzero(q)[s>=t];out[rows]=b;mask[rows]=True
 return out,mask
def main():
 tr,names=bank();d=np.load(D);p=np.load(P205);mp={v:(int(y),int(x)) for v,y,x in zip(p["sample_ids"].astype(str),p["labels"],p["prediction"])};data={}
 for n in S:
  q=tr[n];ids=q["ids"];safe=q["base"].astype(int);cur=np.asarray([mp[v][1] for v in ids]);labels=np.asarray([mp[v][0] for v in ids]);depth=[al(d[k+"_probability"],d["sample_ids"],ids) for k in ("geometry","depth_surface","depth_geometry")];data[n]={**q,"safe":safe,"current":cur,"labels":labels,"x":feature(q["bank"],safe,cur,depth)}
 report={"stage":"P211_pair_specific_rollback_ranker","status":"complete","protocol":{"base":"P205","alternative":"P89 safe","pair_specific_models":True,"source_user_LOSO_threshold":True,"held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];source={k:np.concatenate([data[n][k] for n in src]) for k in ("current","safe","labels","users","x")};rules,audit=select(source);out,m=apply(data[held],rules);outs[held]=out;bc=int(np.sum(data[held]["current"]==data[held]["labels"]));cor=int(np.sum(out==data[held]["labels"]));report["cohorts"][held]={"source":src,"eligible_rules":[{"current":a,"safe":b,"threshold":t} for (a,b),(_,t) in rules.items()],"source_audit":audit,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(m.sum()),"rescue":int(np.sum(m&(data[held]["current"]!=data[held]["labels"])&(out==data[held]["labels"]))),"harm":int(np.sum(m&(data[held]["current"]==data[held]["labels"])&(out!=data[held]["labels"])))}}
 labels=np.concatenate([data[n]["labels"] for n in S]);cur=np.concatenate([data[n]["current"] for n in S]);out=np.concatenate([outs[n] for n in S]);cor=int(np.sum(out==labels));bc=int(np.sum(cur==labels));report["aggregate"]={"rows":len(labels),"base_correct":bc,"correct":cor,"accuracy":cor/len(labels),"net_vs_p205":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]}
 source={k:np.concatenate([data[n][k] for n in S]) for k in ("current","safe","labels","users","x")};rules,audit=select(source);test=build_test_bank(names[:21],names[:24]);pt=np.load(P128T);test["bank"]=np.concatenate((test["bank"],al(pt["probabilities"],pt["sample_ids"],test["ids"])[:,None,:]),axis=1);safe=cp(P89);current=cp(P205T);depth=[]
 for k in ("geometry","depth_surface","depth_geometry"):
  v=test["bank"][:,0,:].copy();v[d["test_available"]]=d["test_"+k+"_probability"][d["test_available"]];depth.append(v)
 td={"current":current,"safe":safe,"x":feature(test["bank"],safe,current,depth)};tout,m=apply(td,rules);O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p211_pair_rollback.csv";io.write_submission(sub,io.read_rows(P89),tout);report["test"]={"eligible_rules":[{"current":a,"safe":b,"threshold":t} for (a,b),(_,t) in rules.items()],"oof_audit":audit,"changes_vs_p205":int(m.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=test["ids"],base_prediction=current,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
