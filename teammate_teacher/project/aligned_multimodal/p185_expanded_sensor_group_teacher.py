"""Add matched P12 IMU/Thermal, Skeleton invariant and MotionBERT to P177."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io
from p137_group_classifier_selector import CONFIG, fit_probability, group_features
from p165_deployable_group_teacher import SPLITS, TEST_METADATA, TRAIN_METADATA, concatenate, crossfit, features, lookup_for
from p173_vjepa_augmented_group_teacher import build_test_bank as build_p173_test, build_train_bank as build_p173_train


HERE=Path(__file__).resolve().parent
OUTPUT=HERE/"runs/p185_expanded_sensor_group_teacher_v1"
P128_OOF=HERE/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz"
P128_TEST=HERE/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz"
P12_OOF=HERE/"runs/p12_complete_oof/complete_oof.npz"
P12_TEST=HERE/"runs/p11_final_package/test_candidate/test_logits.npz"
IMU_TEST=HERE/"runs/p3_sd_imu_rf_full18/test_logits.npz"
SKELETON_OOF=HERE/"runs/p89_skeleton_invariant_expert_v1/oof_logits.npz"
SKELETON_TEST=HERE/"runs/p89_skeleton_invariant_expert_v1/test_logits.npz"
MOTION_OOF=HERE.parent/"runs/p90_motionbert_teacher_v1/motionbert_pretrain_front_linear_oof.npz"
MOTION_TEST=HERE/"runs/p184_motionbert_front_test_v1/test_predictions.npz"
P89=HERE/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"


def softmax(values):
    values=np.asarray(values,dtype=np.float64); values-=values.max(axis=1,keepdims=True); p=np.exp(values); return p/p.sum(axis=1,keepdims=True)


def align(values,source_ids,target_ids):
    lookup={v:i for i,v in enumerate(source_ids.astype(str))}; return np.asarray(values)[np.asarray([lookup[v] for v in target_ids.astype(str)],dtype=np.int64)]


def append(train,ids,probability):
    for name in SPLITS:
        part=train[name]; part["bank"]=np.concatenate((part["bank"],align(probability,ids,part["ids"])[:,None,:]),axis=1)


def main():
    train,names=build_p173_train()
    p128=np.load(P128_OOF,allow_pickle=False); append(train,p128["sample_ids"],p128["probabilities"])
    p12=np.load(P12_OOF,allow_pickle=False)
    append(train,p12["sample_ids"],softmax(p12["imu_logits"]))
    append(train,p12["sample_ids"],softmax(p12["thermal_logits"]))
    sk=np.load(SKELETON_OOF,allow_pickle=False); append(train,sk["sample_ids"],softmax(sk["skeleton_logits"]))
    mo=np.load(MOTION_OOF,allow_pickle=False); append(train,mo["sample_ids"],mo["probabilities"])
    expert_names=[*names,"p128_hierarchical_multimodal","p12_imu","p12_thermal","skeleton_invariant","motionbert_front"]
    reports,held_predictions,held_probabilities,thresholds=crossfit(train)
    all_train=concatenate([train[n] for n in SPLITS]); all_lookup=lookup_for([train[n] for n in SPLITS]); all_x=features(all_train,all_lookup,TRAIN_METADATA)
    test=build_p173_test(names[:21],names)
    extras=[]
    p128t=np.load(P128_TEST,allow_pickle=False); extras.append(align(p128t["probabilities"],p128t["sample_ids"],test["ids"]))
    imut=np.load(IMU_TEST,allow_pickle=False); extras.append(softmax(align(imut["imu_logits"],imut["sample_ids"],test["ids"])))
    p12t=np.load(P12_TEST,allow_pickle=False); extras.append(softmax(align(p12t["thermal_logits"],p12t["sample_ids"],test["ids"])))
    skt=np.load(SKELETON_TEST,allow_pickle=False); extras.append(softmax(align(skt["skeleton_logits"],skt["sample_ids"],test["ids"])))
    mot=np.load(MOTION_TEST,allow_pickle=False); extras.append(align(mot["probabilities"],mot["sample_ids"],test["ids"]))
    test["bank"]=np.concatenate((test["bank"],*[x[:,None,:] for x in extras]),axis=1).astype(np.float32)
    lookup={v:x for v,x in zip(test["ids"],test["bank"])}
    test_x=group_features(test["ids"],test["base"],lookup,posterior_feature_mode="sqrt",group_config=CONFIG,group_feature_layout="full",teacher_subset="full",grouping_lookup=lookup,metadata_path=TEST_METADATA)
    probability=fit_probability(all_x,all_train["labels"],test_x,.03,2.,False,0.)
    proposal=probability.argmax(1); rows=np.arange(len(proposal)); score=probability[rows,proposal]-probability[rows,test["base"]]; threshold=float(np.median(thresholds)); route=(proposal!=test["base"])&(score>=threshold)&test["visual_available"]
    prediction=test["base"].copy(); prediction[route]=proposal[route]
    OUTPUT.mkdir(parents=True,exist_ok=True); submission=OUTPUT/"submission_p185_expanded_sensor_group.csv"; submission_io.write_submission(submission,submission_io.read_rows(P89),prediction)
    np.savez_compressed(OUTPUT/"predictions.npz",sample_ids=test["ids"],base_prediction=test["base"],probability=probability.astype(np.float32),prediction=prediction,route=route,**{f"{n}_held_prediction":held_predictions[n] for n in SPLITS},**{f"{n}_held_probability":held_probabilities[n] for n in SPLITS})
    total=sum(len(train[n]["labels"]) for n in SPLITS); correct=sum(int(np.sum(held_predictions[n]==train[n]["labels"])) for n in SPLITS); base_correct=sum(int(np.sum(train[n]["base"]==train[n]["labels"])) for n in SPLITS)
    report={"stage":"P185_expanded_sensor_group_teacher","status":"complete","protocol":{"expert_names":expert_names,"expert_count":len(expert_names),"new_experts":expert_names[-4:],"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":reports,"aggregate":{"rows":total,"base_correct":base_correct,"correct":correct,"accuracy":correct/total,"net_vs_p89":correct-base_correct,"p177_reference_correct":2154},"test":{"threshold":threshold,"changes_vs_p89":int(route.sum()),"submission":str(submission.resolve()),"test_labels_read":False}}
    (OUTPUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
