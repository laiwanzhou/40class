"""Deployable Test counterpart of P307 (P306 added beside P253)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p137_group_classifier_selector import CONFIG,fit_probability,group_features
from p165_deployable_group_teacher import SPLITS,TEST_METADATA,TRAIN_METADATA,concatenate,crossfit,features,lookup_for
from p173_vjepa_augmented_group_teacher import build_train_bank,build_test_bank
from p255_repeat_augmented_physical_group import al
H=Path(__file__).resolve().parent;O=H/"runs/p309_union_repeat_group_test_v1";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
SOURCES=((H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz","probabilities",H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz","probabilities","p128_hierarchical_multimodal"),(H/"runs/p158_lavila_frame_token_transformer_single_seed_v1/oof_predictions.npz","probability",H/"runs/p198_lavila_frame_token_test_head_v1/test_predictions.npz","probability","p158_lavila_frame_token"),(H/"runs/p238_physical_token_transformer_oof_v1/oof_predictions.npz","probability",H/"runs/p239_physical_token_transformer_test_v1/test_predictions.npz","probability","p238_physical_token"),(H/"runs/p231_depth_thermal_ir_oof_v1/oof_predictions.npz","ir_thermal_probability",H/"runs/p233_depth_thermal_ir_test_heads_v1/test_predictions.npz","ir_thermal_probability","p231_ir_thermal"),(H/"runs/p253_repeat_physical_transformer_oof_v1/oof_predictions.npz","probability",H/"runs/p254_repeat_physical_transformer_test_v1/test_predictions.npz","probability","p253_repeat_physical"),(H/"runs/p306_union_full_repeat_physical_oof_v1/oof_predictions.npz","probability",H/"runs/p308_union_repeat_physical_test_v1/test_predictions.npz","probability","p306_union_repeat_physical"))
def main():
 tr,names=build_train_bank()
 for op,ok,tp,tk,name in SOURCES:
  z=np.load(op)
  for cohort in SPLITS:
   q=tr[cohort];q["bank"]=np.concatenate((q["bank"],al(z[ok],z["sample_ids"],q["ids"])[:,None,:]),1)
  names.append(name)
 rep,pred,prob,thr=crossfit(tr);alltr=concatenate([tr[n] for n in SPLITS]);lk=lookup_for([tr[n] for n in SPLITS]);x=features(alltr,lk,TRAIN_METADATA);test=build_test_bank(names[:21],names[:24]);blocks=[]
 for op,ok,tp,tk,name in SOURCES:
  z=np.load(tp);ids=z["test_sample_ids"] if "test_sample_ids" in z.files and len(z[tk])==len(z["test_sample_ids"]) else z["sample_ids"];p=al(z[tk],ids,test["ids"]);available=al(z["available"],ids,test["ids"]).astype(bool) if "available" in z.files else np.ones(len(p),bool);p[~available]=test["bank"][~available,0,:];blocks.append(p[:,None,:])
 test["bank"]=np.concatenate((test["bank"],*blocks),axis=1);tlk={v:q for v,q in zip(test["ids"],test["bank"])};tx=group_features(test["ids"],test["base"],tlk,posterior_feature_mode="sqrt",group_config=CONFIG,group_feature_layout="full",teacher_subset="full",grouping_lookup=tlk,metadata_path=TEST_METADATA);tp=fit_probability(x,alltr["labels"],tx,.03,2.,False,0.);proposal=tp.argmax(1);rows=np.arange(len(proposal));score=tp[rows,proposal]-tp[rows,test["base"]];threshold=float(np.median(thr));route=(proposal!=test["base"])&(score>=threshold)&test["visual_available"];out=test["base"].copy();out[route]=proposal[route];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p309_union_repeat_group.csv";io.write_submission(sub,io.read_rows(P89),out);total=sum(len(tr[n]["labels"]) for n in SPLITS);correct=sum(int(np.sum(pred[n]==tr[n]["labels"])) for n in SPLITS);np.savez_compressed(O/"predictions.npz",sample_ids=test["ids"],base_prediction=test["base"],probability=tp.astype(np.float32),prediction=out,route=route,**{f"{n}_held_prediction":pred[n] for n in SPLITS},**{f"{n}_held_probability":prob[n] for n in SPLITS});report={"stage":"P309_union_repeat_group_Test","status":"complete","protocol":{"source":"P307 exact 30-expert bank","recovered_extra_rows":10,"test_labels_read":False,"user_id_used_as_feature":False},"validation":{"correct":correct,"rows":total,"accuracy":correct/total,"p255_reference_correct":2172,"net_vs_p255":correct-2172},"test":{"threshold":threshold,"changes_vs_p89":int(route.sum()),"submission":str(sub.resolve()),"test_labels_read":False}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
