"""Native all-Train -> Test deployment of the P140/P143/P145/P150 path."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from audit_p87_sequence_decoder import decode_sessions,fit_transition_model
from p117_transductive_multicandidate_router import load_candidate_splits
from p134_frozen_repeat_consensus import probability_lookup
from p136_peer_support_repeat_gate import peer_candidate,select_rule
from p137_group_classifier_selector import group_features,fit_probability,choose_threshold
from p139_soft_sequence_gate import DECODER,emission,gate_features,select_gate
from p140_source_stable_expanded_sequence import source_candidate,sessions_for
from p143_p142_source_user_safe_selector import build_features
from p117_transductive_multicandidate_router import loso_scores,select_threshold,fit_score
from p214_p128_meta_test import test_candidate_split,al,hard,read_pred
from p165_deployable_group_teacher import build_train_bank as p165_train,build_test_bank as p165_test,normalise
import p89_build_dual_consensus_submission as io

H=Path(__file__).resolve().parent;O=H/"runs/p216_native_p150_test_v1";S=("H1_selection","H2_confirmation","H3_independent_fold0")
P128META=H/"runs/p128_base_hierarchical_meta_selector_v1/predictions.npz";P214=H/"runs/p214_p128_meta_test_v1/test_predictions.npz";OBJRUN=H/"runs/p137_expanded_plus_object_v23/predictions.npz";P140=H/"runs/p140_source_stable_expanded_sequence_v1/predictions.npz";P142=H/"runs/p142_vjepa_token_transformer_three_seed_v2/oof_predictions.npz";P144=H/"runs/p144_vjepa_hand_interaction_transformer_three_seed_v1/oof_predictions.npz";P149=H/"runs/p149_vjepa_repeat_consistency_three_seed_v2/oof_predictions.npz";P143=H/"runs/p143_p142_source_user_safe_selector_v1/predictions.npz";P149H=H/"runs/p149_repeat_source_user_safe_selector_v3/predictions.npz";P145=H/"runs/p145_dual_token_agreement_selector_v1/predictions.npz";P150=H/"runs/p150_repeat_branch_confidence_selector_v1/predictions.npz";P172=H/"runs/p172_vjepa_token_heads_test_v1/test_predictions.npz";P186=H/"runs/p186_p122_relation_test_v1/test_predictions.npz";P215=H/"runs/p215_p123_dense24_test_v1/test_predictions.npz";P176=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";P205=H/"runs/p205_fixed_p203_p150_residual_v1/submission_p205_fixed_residual.csv"

def align_full(values,source_ids,ids,safe):
 out=safe.copy();pos={v:i for i,v in enumerate(ids)};rows=np.asarray([pos[v] for v in source_ids.astype(str)]);out[rows]=values;return normalise(out)
def test_lookup(expanded,names):
 t=test_candidate_split();ids=t.split.sample_ids;safe=t.split.safe_probability;values={n:safe.copy() for n in names};values.update({n:normalise(v) for n,v in t.candidates.items() if n in values})
 tr,n165=p165_train();tt=p165_test(n165);bank=tt["bank"]
 map165={"p85_window_mean":4,"p85_early":2,"p86_drop_person":10,"p12_thermal_candidate":19}
 for n,i in map165.items():values[n]=normalise(bank[:,i])
 r=np.load(P186);p=np.load(P215);h=np.load(P176);v=np.load(P172)
 for n,k in (("p122_hand_object_all","all_probability"),("expanded_p122_pose","pose_only_probability"),("expanded_p122_object","object_only_probability"),("expanded_p122_relation","relations_only_probability")):
  values[n]=align_full(r[k],r["sample_ids"],ids,safe)
 for n,k in (("p123_dense24_group8","dense24_group8_probability"),("p123_dense24_group8_ssv2","dense24_group8_ssv2_probability"),("expanded_p123_old_ir_dense","old_ir_plus_dense24_group8_ssv2_probability")):
  values[n]=align_full(p[k],p["sample_ids"],ids,safe)
 values["p128_hierarchical_multimodal"]=normalise(al(h["probabilities"],h["sample_ids"],ids))
 for n,k in (("expanded_p142_token","p142_all_probability"),("expanded_p144_hand_token","p144_hand_interaction_probability")):
  values[n]=align_full(v[k],v["sample_ids"],ids,safe)
 values["expanded_thermal"]=values["p12_thermal_candidate"].copy()
 # Unknown/unmatched experts remain the safe posterior, preserving availability semantics.
 lookup={sid:np.stack([safe[i],*[values[n][i] for n in names]]).astype(np.float32) for i,sid in enumerate(ids)}
 return t,lookup,values

def p140_test(train,core,test,tlk):
 meta=np.load(P128META);mp={v:int(x) for n in S for v,x in zip(meta[f"{n}_sample_ids"].astype(str),meta[f"{n}_prediction"])};parts=[]
 for n in S:
  q=train[n];parts.append({"ids":q.split.sample_ids.astype(str),"labels":q.split.labels.astype(int),"base":np.asarray([mp[v] for v in q.split.sample_ids.astype(str)])})
 ids=np.concatenate([p["ids"] for p in parts]);labels=np.concatenate([p["labels"] for p in parts]);base=np.concatenate([p["base"] for p in parts]);elk=probability_lookup(train);clk=probability_lookup(core)
 x=group_features(ids,base,elk,posterior_feature_mode="sqrt",group_feature_layout="full",teacher_subset="base_plus_object",grouping_lookup=clk)
 pt=np.load(P214);tbase=al(pt["prediction"],pt["sample_ids"],test.split.sample_ids);tx=group_features(test.split.sample_ids,tbase,tlk,posterior_feature_mode="sqrt",group_feature_layout="full",teacher_subset="base_plus_object",grouping_lookup={k:v[:20] for k,v in tlk.items()},metadata_path=H/"data/p85_recording_metadata/test_recording_metadata.csv")
 tp=fit_probability(x,labels,tx,.03,2.,False,0.);proposal=tp.argmax(1)
 # Global peer rule and strict-OOF group threshold.
 pp,ps,pc,_=peer_candidate(ids,base,clk);rule=select_rule(base,labels,pp,ps,pc);tpp,tps,tpc,_=peer_candidate(test.split.sample_ids,tbase,{k:v[:20] for k,v in tlk.items()},metadata_path=H/"data/p85_recording_metadata/test_recording_metadata.csv");peer=tbase.copy();q=(tpp!=tbase)&(tpc>0)&(tps[:,int(rule["score_index"]) ]>=float(rule["threshold"]));peer[q]=tpp[q]
 run=np.load(OBJRUN);ob=np.concatenate([run[f"{n}_p136_prediction"] for n in S]);op=np.concatenate([run[f"{n}_group_probability"] for n in S]);og=op.argmax(1);gsel=choose_threshold(ob,og,op,labels);score=tp[np.arange(len(tp)),proposal]-tp[np.arange(len(tp)),peer];q=(proposal!=peer)&(score>=gsel["threshold"]);group=peer.copy();group[q]=proposal[q]
 # Strict outer sequence candidates select one global gate; Test transition refits all Train.
 seqs=[];feats=[];bases=[]
 for n in S:
  src=source_candidate(core,n,run);bid=run[f"{n}_sample_ids"].astype(str);bb=run[f"{n}_prediction"];prob=run[f"{n}_group_probability"];sess=sessions_for(core,bid,[n]);seq=decode_sessions(emission(prob,bb),sess,src["transition"],DECODER);seqs.append(seq);feats.append(gate_features(prob,bb,seq));bases.append(bb)
 obase=np.concatenate(bases);oseq=np.concatenate(seqs);ofeat=np.concatenate(feats);gate=select_gate(obase,oseq,ofeat,labels);trans=fit_transition_model(labels,sessions_for(core,ids,list(S)),40,DECODER.trigram_backoff);tseq=decode_sessions(emission(tp,group),sessions_for({"T":type('X',(object,),{'split':test.split})()},test.split.sample_ids,["T"]),trans,DECODER);tf=gate_features(tp,group,tseq);q=(tseq!=group)&(tf[:,int(gate["score_index"]) ]>=float(gate["threshold"]));out=group.copy();out[q]=tseq[q]
 return out,{"peer_rule":rule,"group_threshold":gsel,"sequence_gate":gate,"peer_changes":int(np.sum(peer!=tbase)),"group_changes":int(np.sum(group!=peer)),"sequence_changes":int(np.sum(out!=group))}

def branch(name,path,key,base_oof,base_test,ids,users,labels,lookup,tlk,tprob):
 z=np.load(path);prob=al(z[key],z["sample_ids"],ids);alt=prob.argmax(1);x=build_features(ids,base_oof,alt,prob,lookup);gain=(alt==labels).astype(np.int8)-(base_oof==labels).astype(np.int8);score=loso_scores(x,gain,users);sel=select_threshold(score,gain,alt!=base_oof,users);tx=build_features(tprob["ids"],base_test,tprob[name].argmax(1),tprob[name],tlk);ts=fit_score(x,gain,tx);out=base_test.copy();q=(tprob[name].argmax(1)!=base_test)&(ts>=sel["threshold"])&(sel["net"]>0)&(sel["minimum_user_gain"]>=0);out[q]=tprob[name].argmax(1)[q];return out,sel

def main():
 train=load_candidate_splits(full_visual_bank=True,structured_bank=True,legacy_visual_bank=True,hand_object_bank=True,vjepa_dense_bank=True,nonvisual_bank=True,hierarchical_bank=True,epic_bank=True,expanded_bank=True);core=load_candidate_splits(full_visual_bank=True,structured_bank=True,legacy_visual_bank=True,hand_object_bank=True,vjepa_dense_bank=True,nonvisual_bank=True,hierarchical_bank=True,epic_bank=True);names=list(next(iter(train.values())).candidates);test,tlk,tvals=test_lookup(train,names);p140t,audit=p140_test(train,core,test,tlk)
 p140=np.load(P140);ids=np.concatenate([p140[f"{n}_sample_ids"].astype(str) for n in S]);labels=np.concatenate([p140[f"{n}_labels"] for n in S]);users=np.concatenate([train[n].split.users.astype(str) for n in S]);base=np.concatenate([p140[f"{n}_prediction"] for n in S]);lookup=probability_lookup(train);v=np.load(P172);tpids=test.split.sample_ids;tp={"ids":tpids,"p142":align_full(v["p142_all_probability"],v["sample_ids"],tpids,test.split.safe_probability),"p144":align_full(v["p144_hand_interaction_probability"],v["sample_ids"],tpids,test.split.safe_probability),"p149":align_full(v["p149_repeat_consistency_probability"],v["sample_ids"],tpids,test.split.safe_probability)}
 allb,asel=branch("p142",P142,"probability",base,p140t,ids,users,labels,lookup,tlk,tp);repb,rsel=branch("p149",P149,"probability",base,p140t,ids,users,labels,lookup,tlk,tp)
 # P145 agreement on strict P143 OOF.
 p143=np.load(P143);strict_all=np.concatenate([p143[f"{n}_prediction"] for n in S]);ap=al(np.load(P142)["probability"],np.load(P142)["sample_ids"],ids);hp=al(np.load(P144)["probability"],np.load(P144)["sample_ids"],ids);aa=ap.argmax(1);ha=hp.argmax(1);conf=np.minimum(ap[np.arange(len(ids)),aa],hp[np.arange(len(ids)),aa]);agree=(aa==ha)&(aa!=strict_all);best=None
 for t in np.unique(np.concatenate((np.linspace(0,1,201),np.quantile(conf[agree],[.1,.2,.4,.6,.8,.9,.95]) if agree.any() else [2.]))):
  q=agree&(conf>=t);gain=(aa==labels).astype(int)-(strict_all==labels).astype(int);per={u:int(gain[q&(users==u)].sum()) for u in sorted(set(users))};r=int(np.sum(q&(gain>0)));h=int(np.sum(q&(gain<0)));row={"threshold":float(t),"changed":int(q.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(per.values())};key=(row["minimum_user_gain"]>=0,row["harm"]==0,row["net"],r,-row["changed"]);best=(key,row) if best is None or key>best[0] else best
 p145sel=best[1];ta=tp["p142"].argmax(1);th=tp["p144"].argmax(1);tc=np.minimum(tp["p142"][np.arange(len(tpids)),ta],tp["p144"][np.arange(len(tpids)),ta]);q=(ta==th)&(ta!=allb)&(tc>=p145sel["threshold"])&(p145sel["rescue"]>=2)&(p145sel["harm"]==0)&(p145sel["minimum_user_gain"]>=0);p145t=allb.copy();p145t[q]=ta[q]
 # P150 confidence selector between strict P145 and strict repeat branches.
 p145=np.load(P145);ph=np.load(P149H);oa=np.concatenate([p145[f"{n}_prediction"] for n in S]);orr=np.concatenate([ph[f"{n}_prediction"] for n in S]);rp=al(np.load(P149)["probability"],np.load(P149)["sample_ids"],ids);rc=rp[np.arange(len(ids)),orr];dis=oa!=orr;best=None
 for t in np.unique(np.concatenate((np.linspace(0,1,201),np.quantile(rc[dis],[.1,.2,.4,.6,.8,.9,.95]) if dis.any() else [2.]))):
  q=dis&(rc>=t);gain=(orr==labels).astype(int)-(oa==labels).astype(int);per={u:int(gain[q&(users==u)].sum()) for u in sorted(set(users))};r=int(np.sum(q&(gain>0)));h=int(np.sum(q&(gain<0)));row={"threshold":float(t),"changed":int(q.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(per.values())};key=(row["minimum_user_gain"]>=0,row["net"],r,-h,-row["changed"]);best=(key,row) if best is None or key>best[0] else best
 p150sel=best[1];trc=tp["p149"][np.arange(len(tpids)),repb];q=(repb!=p145t)&(trc>=p150sel["threshold"])&(p150sel["net"]>0)&(p150sel["minimum_user_gain"]>=0);out=p145t.copy();out[q]=repb[q]
 O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p216_native_p150.csv";io.write_submission(sub,io.read_rows(P89),out);p205=read_pred(P205);native=np.load(P150);ny=np.concatenate([native[f"{n}_labels"] for n in S]);no=np.concatenate([native[f"{n}_prediction"] for n in S]);report={"stage":"P216_native_P150_Test","status":"complete","protocol":{"P128_meta_test":"P214","P140_test":"object expanded group + sequence, all-Train refit","P143_P145_P150":"native selector families refit from strict OOF","user_id_used_as_feature":False,"test_labels_read":False},"oof":{"native_p150_correct":int(np.sum(no==ny)),"rows":len(ny),"accuracy":float(np.mean(no==ny))},"p140_test":audit,"selectors":{"p142":asel,"p149":rsel,"p145":p145sel,"p150":p150sel},"test":{"rows":len(out),"changes_vs_p89":int(np.sum(out!=test.split.safe_prediction)),"changes_vs_p205":int(np.sum(out!=p205)),"submission":str(sub.resolve()),"test_labels_read":False}};np.savez_compressed(O/"predictions.npz",sample_ids=tpids,p140_prediction=p140t,p143_prediction=allb,p149_prediction=repb,p145_prediction=p145t,prediction=out,p205_prediction=p205);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
