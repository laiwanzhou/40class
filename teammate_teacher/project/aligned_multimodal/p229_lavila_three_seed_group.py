"""Add the matched LaViLa frame-token expert to the P177 deployable bank."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p137_group_classifier_selector import CONFIG,fit_probability,group_features
from p165_deployable_group_teacher import SPLITS,TEST_METADATA,TRAIN_METADATA,concatenate,crossfit,features,lookup_for
from p173_vjepa_augmented_group_teacher import build_train_bank,build_test_bank
H=Path(__file__).resolve().parent;O=H/"runs/p229_lavila_three_seed_group_v1";P128=H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz";P128T=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz";L=H/"runs/p227_lavila_frame_token_three_seed_v1/oof_predictions.npz";LT=H/"runs/p228_lavila_three_seed_test_v1/test_predictions.npz";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
def al(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def main():
 tr,names=build_train_bank();p=np.load(P128);l=np.load(L)
 for n in SPLITS:
  q=tr[n];q["bank"]=np.concatenate((q["bank"],al(p["probabilities"],p["sample_ids"],q["ids"])[:,None,:],al(l["probability"],l["sample_ids"],q["ids"])[:,None,:]),axis=1)
 names=[*names,"p128_hierarchical_multimodal","p227_lavila_frame_token_three_seed"]
 rep,pred,prob,thr=crossfit(tr);alltr=concatenate([tr[n] for n in SPLITS]);lk=lookup_for([tr[n] for n in SPLITS]);x=features(alltr,lk,TRAIN_METADATA)
 test=build_test_bank(names[:21],names[:24]);pt=np.load(P128T);lt=np.load(LT);test["bank"]=np.concatenate((test["bank"],al(pt["probabilities"],pt["sample_ids"],test["ids"])[:,None,:],al(lt["probability"],lt["sample_ids"],test["ids"])[:,None,:]),axis=1);lk={v:q for v,q in zip(test["ids"],test["bank"])};tx=group_features(test["ids"],test["base"],lk,posterior_feature_mode="sqrt",group_config=CONFIG,group_feature_layout="full",teacher_subset="full",grouping_lookup=lk,metadata_path=TEST_METADATA);tp=fit_probability(x,alltr["labels"],tx,.03,2.,False,0.);pr=tp.argmax(1);i=np.arange(len(pr));score=tp[i,pr]-tp[i,test["base"]];t=float(np.median(thr));route=(pr!=test["base"])&(score>=t)&test["visual_available"];out=test["base"].copy();out[route]=pr[route];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p229_lavila_three_seed_group.csv";io.write_submission(sub,io.read_rows(P89),out);np.savez_compressed(O/"predictions.npz",sample_ids=test["ids"],base_prediction=test["base"],probability=tp.astype(np.float32),prediction=out,route=route,**{f"{n}_held_prediction":pred[n] for n in SPLITS},**{f"{n}_held_probability":prob[n] for n in SPLITS});total=sum(len(tr[n]["labels"]) for n in SPLITS);cor=sum(int(np.sum(pred[n]==tr[n]["labels"])) for n in SPLITS);bc=sum(int(np.sum(tr[n]["base"]==tr[n]["labels"])) for n in SPLITS);report={"stage":"P229_LaViLa_three_seed_P128_VJEPA_group_teacher","status":"complete","protocol":{"expert_names":names,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":rep,"aggregate":{"rows":total,"base_correct":bc,"correct":cor,"accuracy":cor/total,"net_vs_p89":cor-bc,"p177_reference_correct":2154},"test":{"threshold":t,"changes_vs_p89":int(route.sum()),"submission":str(sub.resolve()),"test_labels_read":False}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

