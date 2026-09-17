"""Deploy P177 plus the positive P122 relations-only candidate."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p137_group_classifier_selector import CONFIG,fit_probability,group_features
from p165_deployable_group_teacher import SPLITS,TEST_METADATA,TRAIN_METADATA,concatenate,crossfit,features,lookup_for
from p173_vjepa_augmented_group_teacher import build_train_bank,build_test_bank

H=Path(__file__).resolve().parent; O=H/"runs/p188_relation_vjepa_group_teacher_v1"; P128=H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz";P128T=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz";REL=H/"runs/p122_hand_object_relation_teacher_v1/oof_predictions.npz";RELT=H/"runs/p186_p122_relation_test_v1/test_predictions.npz";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
def al(v,s,t):
 d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def main():
 train,names=build_train_bank();p=np.load(P128);r=np.load(REL)
 for n in SPLITS:
  q=train[n];q["bank"]=np.concatenate((q["bank"],al(p["probabilities"],p["sample_ids"],q["ids"])[:,None,:],al(r["relations_only_probability"],r["sample_ids"],q["ids"])[:,None,:]),axis=1)
 names=[*names,"p128_hierarchical_multimodal","p122_relations_only"]
 rep,pred,prob,thr=crossfit(train);alltrain=concatenate([train[n] for n in SPLITS]);lk=lookup_for([train[n] for n in SPLITS]);x=features(alltrain,lk,TRAIN_METADATA)
 test=build_test_bank(names[:21],names[:24]);pt=np.load(P128T);rt=np.load(RELT);safe=test["bank"][:,0,:];pfull=al(pt["probabilities"],pt["sample_ids"],test["ids"]);rfull=safe.copy();rid=rt["sample_ids"].astype(str);pos={v:i for i,v in enumerate(test["ids"])};rows=np.asarray([pos[v] for v in rid]);rfull[rows]=rt["relations_only_probability"]
 test["bank"]=np.concatenate((test["bank"],pfull[:,None,:],rfull[:,None,:]),axis=1);lk={v:x for v,x in zip(test["ids"],test["bank"])};tx=group_features(test["ids"],test["base"],lk,posterior_feature_mode="sqrt",group_config=CONFIG,group_feature_layout="full",teacher_subset="full",grouping_lookup=lk,metadata_path=TEST_METADATA);tp=fit_probability(x,alltrain["labels"],tx,.03,2.,False,0.);proposal=tp.argmax(1);i=np.arange(len(proposal));score=tp[i,proposal]-tp[i,test["base"]];threshold=float(np.median(thr));route=(proposal!=test["base"])&(score>=threshold)&test["visual_available"];out=test["base"].copy();out[route]=proposal[route]
 O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p188_relation_group.csv";io.write_submission(sub,io.read_rows(P89),out);np.savez_compressed(O/"predictions.npz",sample_ids=test["ids"],base_prediction=test["base"],probability=tp.astype(np.float32),prediction=out,route=route,**{f"{n}_held_prediction":pred[n] for n in SPLITS},**{f"{n}_held_probability":prob[n] for n in SPLITS});total=sum(len(train[n]["labels"]) for n in SPLITS);correct=sum(int(np.sum(pred[n]==train[n]["labels"])) for n in SPLITS);bc=sum(int(np.sum(train[n]["base"]==train[n]["labels"])) for n in SPLITS);report={"stage":"P188_relation_VJEPA_group_teacher","status":"complete","protocol":{"expert_names":names,"test_labels_read":False},"cohorts":rep,"aggregate":{"rows":total,"base_correct":bc,"correct":correct,"accuracy":correct/total,"net_vs_p89":correct-bc},"test":{"threshold":threshold,"changes_vs_p89":int(route.sum()),"submission":str(sub.resolve()),"test_labels_read":False}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
