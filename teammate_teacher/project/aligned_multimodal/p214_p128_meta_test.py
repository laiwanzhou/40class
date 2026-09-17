"""Rebuild the native P118 -> P128 candidate -> outer-meta path on Test.

All model fits use labelled Train OOF rows. Thresholds are selected from strict
held-cohort OOF outputs. Test labels are never present or read.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

from audit_p87_sequence_decoder import align_metadata, build_sessions
from p117_transductive_multicandidate_router import CandidateSplit, one_hot, shared_features, candidate_features
from p118_candidate_conditioned_router import feature_bank, stack_instances, router_fit_score, score_to_row_choice, select_threshold
from p120_router_meta_selector import meta_features, fit_score as meta_fit_score, threshold_report
from p90_crossuser_visual_router import SplitData, load_splits
from p90_build_routed_test_targets import visual_test_probabilities, test_safe_contract, test_quality, TEST_METADATA
from p165_deployable_group_teacher import normalise

H=Path(__file__).resolve().parent; O=H/"runs/p214_p128_meta_test_v1"
BASE=H/"runs/p118_candidate_conditioned_structured_maxnet_v2"
HIER=H/"runs/p128_hierarchical_candidate_router_v1"
META=H/"runs/p128_base_hierarchical_meta_selector_v1"
P128T=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz"
P88=H/"runs/p88_final_test_predictions_v1/submission_p88_repeat.csv"
P89P=H/"runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
P87=H/"runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
S=("H1_selection","H2_confirmation","H3_independent_fold0")

def read_pred(path):
 with path.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def hard(p):
 q=np.full((len(p),40),(1-.95)/39,np.float64);q[np.arange(len(p)),p]=.95;return q
def al(v,s,t):
 d={x:i for i,x in enumerate(np.asarray(s).astype(str))};return np.asarray(v)[np.asarray([d[x] for x in np.asarray(t).astype(str)])]

def test_candidate_split():
 p89=np.load(P89P);ids=p89["sample_ids"].astype(str);safe=normalise(p89["base_probability"])
 # test_safe_contract reproduces the exact deployed P89 safe probability/prediction on readable rows.
 rid,vis=visual_test_probabilities();rsafe,rpred,rp87=test_safe_contract(rid)
 pos={v:i for i,v in enumerate(ids)};rows=np.asarray([pos[v] for v in rid]);safe[rows]=rsafe
 deployed=read_pred(H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv")
 p87=read_pred(P87);repeat=read_pred(P88)
 visual={}
 for name,value in vis.items():
  full=safe.copy();full[rows]=value;visual[name]=full
 quality,quality_names=test_quality(ids);metadata=align_metadata(TEST_METADATA,ids);sessions=build_sessions(np.arange(len(ids)),metadata,30.,"anonymous_date")
 candidates={
  "a18_best_session":None,
  "p90_visual_equal":visual["videomaev2_base_plus_internvideo2_l_equal"],
  "p90_videomaev2_distilled_base":visual["videomaev2_distilled_base"],
  "p90_internvideo2_l_early_late":visual["internvideo2_l_early_late"],
  "p90_internvideo2_l_early_late_plus_k400":visual["internvideo2_l_early_late_plus_k400"],
  "p87_sequence":hard(p87),"p88_repeat":hard(repeat),
  # The original latent-prefix branch has no frozen Test artifact. Its conservative
  # deployment fallback is the already validated anonymous repeat branch.
  "p88_latent_prefix":hard(repeat),
 }
 a=np.load(H/"runs/a18_full_teacher_test_v1/test_predictions.npz");candidates["a18_best_session"]=normalise(al(a["selected_probability"],a["sample_ids"],ids))
 split=SplitData("official_test",ids,np.full(len(ids),-1),np.full(len(ids),"anonymous"),safe,deployed,p87,sessions,visual,quality,quality_names)
 return CandidateSplit(split,candidates)

def test_features(value,names,hier=False):
 if list(value.candidates)!=names:raise RuntimeError("Test candidate order differs")
 shared=shared_features(value)
 if hier:
  p=np.load(P128T);prob=al(p["probabilities"],p["sample_ids"],value.split.sample_ids);rel=1/(1+np.exp(-np.clip(al(p["reliability_logits"],p["sample_ids"],value.split.sample_ids),-20,20)));shared=np.concatenate((shared,prob.astype(np.float32),rel[:,None].astype(np.float32)),1)
 out={}
 for i,n in enumerate(names):
  base=candidate_features(value,shared,n,include_session_context=False);ident=np.zeros((len(base),len(names)),np.float32);ident[:,i]=1;out[n]=np.concatenate((base,ident),1).astype(np.float32)
 return out

def global_oof_threshold(run,data,names):
 saved=np.load(run/"predictions.npz");scores=[];candidates=[]
 for n in S:scores.append(saved[f"{n}_route_score"]);candidates.append(saved[f"{n}_candidate_index"])
 return select_threshold(list(S),data,names,np.concatenate(scores),np.concatenate(candidates),"max_net")[0]

def final_router(data,features,test,test_feat,run):
 names=list(next(iter(data.values())).candidates);train=stack_instances(list(S),data,features,names);x,gain,disagreement,users,sk,ck=train
 tx=[];td=[];ts=[];tc=[]
 rows=len(test.split.sample_ids)
 for i,n in enumerate(names):
  tx.append(test_feat[n]);td.append(test.candidates[n].argmax(1)!=test.split.safe_prediction);ts.append(np.arange(rows));tc.append(np.full(rows,i))
 tx=np.concatenate(tx);td=np.concatenate(td);ts=np.concatenate(ts);tc=np.concatenate(tc)
 score=router_fit_score(x,gain,tx,users,"default");best_score,best_candidate=score_to_row_choice(score,td,ts,tc,rows);sel=global_oof_threshold(run,data,names);out=test.split.safe_prediction.copy();route=(best_candidate>=0)&(best_score>=sel["threshold"])
 for i,n in enumerate(names):
  q=route&(best_candidate==i);out[q]=test.candidates[n].argmax(1)[q]
 return out,best_score,best_candidate,sel

def main():
 core=load_splits();base_data=__import__('p117_transductive_multicandidate_router').load_candidate_splits(full_visual_bank=True,structured_bank=True);base_feat,base_names=feature_bank(base_data)
 hier_data=__import__('p117_transductive_multicandidate_router').load_candidate_splits(full_visual_bank=True,structured_bank=True,hierarchical_bank=True);hier_feat,hier_names=feature_bank(hier_data,include_hierarchical_features=True)
 test=test_candidate_split();bt=test_features(test,base_names,False)
 p=np.load(P128T);hp=normalise(al(p["probabilities"],p["sample_ids"],test.split.sample_ids));hc=dict(test.candidates);hc["p128_hierarchical_multimodal"]=hp;htest=CandidateSplit(test.split,hc);ht=test_features(htest,hier_names,True)
 bout,bs,bi,bsel=final_router(base_data,base_feat,test,bt,BASE);hout,hs,hi,hsel=final_router(hier_data,hier_feat,htest,ht,HIER)
 # Fit the same outer-meta family on all strict OOF decisive disagreements.
 bsum=json.load(open(BASE/"summary.json",encoding="utf-8"));hsum=json.load(open(HIER/"summary.json",encoding="utf-8"));bp=np.load(BASE/"predictions.npz");hpz=np.load(HIER/"predictions.npz")
 xs=[];ys=[];us=[];labs=[];bo=[];ho=[]
 for n in S:
  safe=bp[f"{n}_safe_prediction"];left=bp[f"{n}_router_prediction"];right=hpz[f"{n}_router_prediction"];y=bp[f"{n}_labels"];u=core[n].users.astype(str);bl=float(bsum["cohorts"][n]["selected_threshold"]["threshold"]);hl=float(hsum["cohorts"][n]["selected_threshold"]["threshold"]);x=meta_features(safe,left,right,bp[f"{n}_route_score"],hpz[f"{n}_route_score"],bl,hl,bp[f"{n}_candidate_index"],hpz[f"{n}_candidate_index"],max(len(base_names),len(hier_names)));xs.append(x);ys.append(y);us.append(u);bo.append(left);ho.append(right)
 x=np.concatenate(xs);y=np.concatenate(ys);users=np.concatenate(us);left=np.concatenate(bo);right=np.concatenate(ho);dis=(left!=right);dec=dis&((left==y)!=(right==y));nested=np.zeros(len(y))
 for u in sorted(set(users.tolist())):
  te=(users==u)&dis;tr=dec&(users!=u)
  if te.any() and len(np.unique((right[tr]==y[tr]).astype(int)))==2:nested[te]=meta_fit_score(x[tr],(right[tr]==y[tr]).astype(int),x[te])
 grid=[threshold_report(nested,y,users,left,right,t) for t in np.arange(.2,.851,.025)];meta_sel=max(grid,key=lambda r:(r["net_vs_base"],r["minimum_user_gain"],r["positive_users"],-r["switches"]))
 tx=meta_features(test.split.safe_prediction,bout,hout,bs,hs,float(bsel["threshold"]),float(hsel["threshold"]),bi,hi,max(len(base_names),len(hier_names)));target=(right[dec]==y[dec]).astype(int);ts=meta_fit_score(x[dec],target,tx);m=(bout!=hout)&(ts>=meta_sel["threshold"]);out=bout.copy();out[m]=hout[m]
 # Audit the historic strict OOF output; this is the authoritative native P128-meta score.
 native=np.load(META/"predictions.npz");oof=np.concatenate([native[f"{n}_prediction"] for n in S]);labels=np.concatenate([native[f"{n}_labels"] for n in S])
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"test_predictions.npz",sample_ids=test.split.sample_ids,base_router_prediction=bout,hierarchical_router_prediction=hout,meta_score=ts,prediction=out)
 report={"stage":"P214_native_P128_meta_Test","status":"complete","protocol":{"base_and_hier_models":"all labelled strict OOF rows","thresholds":"strict held-cohort OOF","meta":"LOSO threshold then all decisive OOF refit","latent_prefix_test_fallback":"P88 anonymous repeat","user_id_used_as_feature":False,"test_labels_read":False},"oof":{"native_correct":int(np.sum(oof==labels)),"rows":len(labels),"accuracy":float(np.mean(oof==labels))},"test":{"base_threshold":bsel,"hier_threshold":hsel,"meta_threshold":meta_sel,"base_changes_vs_p89":int(np.sum(bout!=test.split.safe_prediction)),"hier_changes_vs_p89":int(np.sum(hout!=test.split.safe_prediction)),"meta_switches":int(m.sum()),"meta_changes_vs_p89":int(np.sum(out!=test.split.safe_prediction))}}
 (O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
