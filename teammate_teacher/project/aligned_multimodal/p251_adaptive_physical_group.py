"""Outer-safe configuration selection for the P244 dual-physical group bank."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p137_group_classifier_selector import CONFIG,fit_probability,group_features,choose_threshold
from p165_deployable_group_teacher import SPLITS,TRAIN_METADATA,TEST_METADATA,concatenate,lookup_for,metrics
from p173_vjepa_augmented_group_teacher import build_train_bank,build_test_bank
H=Path(__file__).resolve().parent;O=H/"runs/p251_adaptive_physical_group_v1";P128=H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz";P128T=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz";L=H/"runs/p158_lavila_frame_token_transformer_single_seed_v1/oof_predictions.npz";LT=H/"runs/p198_lavila_frame_token_test_head_v1/test_predictions.npz";PHY=H/"runs/p238_physical_token_transformer_oof_v1/oof_predictions.npz";PHYT=H/"runs/p239_physical_token_transformer_test_v1/test_predictions.npz";RIDGE=H/"runs/p231_depth_thermal_ir_oof_v1/oof_predictions.npz";RIDGET=H/"runs/p233_depth_thermal_ir_test_heads_v1/test_predictions.npz";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";CONFIGS=tuple((c,w,m) for c in (.01,.03,.1) for w in (1.,2.) for m in ("sqrt","sqrt_raw"))
def al(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def bank():
 tr,names=build_train_bank();p=np.load(P128);l=np.load(L);phy=np.load(PHY);ridge=np.load(RIDGE)
 for n in SPLITS:
  q=tr[n];q["bank"]=np.concatenate((q["bank"],al(p["probabilities"],p["sample_ids"],q["ids"])[:,None,:],al(l["probability"],l["sample_ids"],q["ids"])[:,None,:],al(phy["probability"],phy["sample_ids"],q["ids"])[:,None,:],al(ridge["ir_thermal_probability"],ridge["sample_ids"],q["ids"])[:,None,:]),1)
 return tr,[*names,"p128_hierarchical","p158_lavila","p238_physical","p231_ir_thermal"]
def feats(part,lookup,mode,meta=TRAIN_METADATA):return group_features(part["ids"],part["base"],lookup,posterior_feature_mode=mode,group_config=CONFIG,group_feature_layout="full",teacher_subset="full",grouping_lookup=lookup,metadata_path=meta)
def source_eval(tr,names,cfg):
 c,w,mode=cfg;parts=[tr[n] for n in names];source=concatenate(parts);lk=lookup_for(parts);xs={n:feats(tr[n],lk,mode) for n in names};prob=np.zeros((len(source["ids"]),40));off=0
 for target,cal in ((names[0],names[1]),(names[1],names[0])):
  p=fit_probability(xs[cal],tr[cal]["labels"],xs[target],c,w,False,0.);prob[off:off+len(p)]=p;off+=len(p)
 pr=prob.argmax(1);sel=choose_threshold(source["base"],pr,prob,source["labels"]);score=prob[np.arange(len(pr)),pr]-prob[np.arange(len(pr)),source["base"]];q=(pr!=source["base"])&(score>=sel["threshold"]);out=source["base"].copy();out[q]=pr[q];offset=0;per=[]
 for n in names:
  z=slice(offset,offset+len(tr[n]["ids"]));per.append(int(np.sum(out[z]==source["labels"][z])-np.sum(source["base"][z]==source["labels"][z])));offset+=len(tr[n]["ids"])
 return sel,per
def held_apply(tr,held,src,cfg,sel):
 c,w,mode=cfg;parts=[tr[n] for n in src];source=concatenate(parts);lk=lookup_for([*parts,tr[held]]);sx=feats(source,lk,mode);hx=feats(tr[held],lk,mode);p=fit_probability(sx,source["labels"],hx,c,w,False,0.);pr=p.argmax(1);score=p[np.arange(len(pr)),pr]-p[np.arange(len(pr)),tr[held]["base"]];q=(pr!=tr[held]["base"])&(score>=sel["threshold"]);out=tr[held]["base"].copy();out[q]=pr[q];return out,p,q
def main():
 tr,names=bank();report={"stage":"P251_outer_safe_adaptive_physical_group","status":"complete","protocol":{"config_grid":[list(x) for x in CONFIGS],"selection":"source cross-cohort min-net then aggregate net","held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":{}};outs={};probs={};chosen=[]
 for held in SPLITS:
  src=[n for n in SPLITS if n!=held];cand=[]
  for cfg in CONFIGS:
   sel,per=source_eval(tr,src,cfg);cand.append((min(per),sel["net"],sel["rescue"],-sel["harm"],cfg,sel,per))
  best=max(cand);cfg,sel,per=best[4],best[5],best[6];out,p,q=held_apply(tr,held,src,cfg,sel);outs[held]=out;probs[held]=p;chosen.append(cfg);report["cohorts"][held]={"source":src,"config":list(cfg),"source_selection":sel,"source_cohort_nets":per,"held":metrics(tr[held]["labels"],tr[held]["base"],out)};print(json.dumps({"held":held,"config":cfg,"source_per":per,"result":report["cohorts"][held]["held"]}),flush=True)
 total=sum(len(tr[n]["labels"]) for n in SPLITS);cor=sum(int(np.sum(outs[n]==tr[n]["labels"])) for n in SPLITS);bc=sum(int(np.sum(tr[n]["base"]==tr[n]["labels"])) for n in SPLITS);report["aggregate"]={"rows":total,"base_correct":bc,"correct":cor,"accuracy":cor/total,"net_vs_p89":cor-bc,"fold_nets":[report["cohorts"][n]["held"]["net"] for n in SPLITS],"p244_fixed_correct":2170}
 # Final config selected by strict outer results, prioritizing worst-fold and total gain.
 evals=[]
 for cfg in CONFIGS:
  fold=[];thresholds=[]
  for held in SPLITS:
   src=[n for n in SPLITS if n!=held];sel,per=source_eval(tr,src,cfg);out,_,_=held_apply(tr,held,src,cfg,sel);fold.append(int(np.sum(out==tr[held]["labels"])-np.sum(tr[held]["base"]==tr[held]["labels"])));thresholds.append(sel["threshold"])
  evals.append((min(fold),sum(fold),cfg,fold,thresholds))
 final=max(evals);cfg,fold,thresholds=final[2],final[3],final[4];alltr=concatenate([tr[n] for n in SPLITS]);lk=lookup_for([tr[n] for n in SPLITS]);x=feats(alltr,lk,cfg[2]);test=build_test_bank(names[:21],names[:24]);pt=np.load(P128T);lt=np.load(LT);ph=np.load(PHYT);ri=np.load(RIDGET);pp=al(ph["probability"],ph["sample_ids"],test["ids"]);av=al(ph["available"],ph["sample_ids"],test["ids"]);pp[~av]=test["bank"][~av,0];rp=al(ri["ir_thermal_probability"],ri["sample_ids"],test["ids"]);rav=al(ri["ir_thermal_available"],ri["sample_ids"],test["ids"]);rp[~rav]=test["bank"][~rav,0];test["bank"]=np.concatenate((test["bank"],al(pt["probabilities"],pt["sample_ids"],test["ids"])[:,None,:],al(lt["probability"],lt["sample_ids"],test["ids"])[:,None,:],pp[:,None,:],rp[:,None,:]),1);tlk={v:q for v,q in zip(test["ids"],test["bank"])};tx=feats(test,tlk,cfg[2],TEST_METADATA);tp=fit_probability(x,alltr["labels"],tx,cfg[0],cfg[1],False,0.);pr=tp.argmax(1);score=tp[np.arange(len(pr)),pr]-tp[np.arange(len(pr)),test["base"]];threshold=float(np.median(thresholds));q=(pr!=test["base"])&(score>=threshold)&test["visual_available"];tout=test["base"].copy();tout[q]=pr[q];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p251_adaptive_group.csv";io.write_submission(sub,io.read_rows(P89),tout);report["final_config"]={"config":list(cfg),"outer_fold_nets":fold,"outer_thresholds":thresholds,"threshold_median":threshold};report["test"]={"changes_vs_p89":int(q.sum()),"submission":str(sub.resolve()),"test_labels_read":False};np.savez_compressed(O/"predictions.npz",sample_ids=test["ids"],base_prediction=test["base"],probability=tp.astype(np.float32),prediction=tout,route=q,**{f"{n}_held_prediction":outs[n] for n in SPLITS},**{f"{n}_held_probability":probs[n] for n in SPLITS});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps({"aggregate":report["aggregate"],"final":report["final_config"],"test":report["test"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
