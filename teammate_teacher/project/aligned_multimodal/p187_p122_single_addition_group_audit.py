"""Evaluate each P122 variant as one addition to the P177 deployable bank."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p165_deployable_group_teacher import SPLITS,crossfit
from p173_vjepa_augmented_group_teacher import build_train_bank as build_p173_train


HERE=Path(__file__).resolve().parent
OUTPUT=HERE/"runs/p187_p122_single_addition_group_audit_v1"
P128=HERE/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz"
P122=HERE/"runs/p122_hand_object_relation_teacher_v1/oof_predictions.npz"
VARIANTS=("pose_only","object_only","relations_only","all")


def align(values,source_ids,target_ids):
    lookup={v:i for i,v in enumerate(source_ids.astype(str))};return np.asarray(values)[np.asarray([lookup[v] for v in target_ids.astype(str)],dtype=np.int64)]


def base_bank():
    train,names=build_p173_train(); p=np.load(P128,allow_pickle=False)
    for n in SPLITS:
        part=train[n]; part["bank"]=np.concatenate((part["bank"],align(p["probabilities"],p["sample_ids"],part["ids"])[:,None,:]),axis=1)
    return train,[*names,"p128_hierarchical_multimodal"]


def main():
    source=np.load(P122,allow_pickle=False); report={"stage":"P187_P122_single_addition_group_audit","status":"complete","protocol":{"base":"P177","one_candidate_at_a_time":True,"held_labels_used_for_selection":False,"test_labels_read":False},"variants":{}};payload={}
    for variant in VARIANTS:
        train,names=base_bank()
        for n in SPLITS:
            part=train[n];extra=align(source[f"{variant}_probability"],source["sample_ids"],part["ids"]);part["bank"]=np.concatenate((part["bank"],extra[:,None,:]),axis=1)
        cohorts,predictions,probabilities,thresholds=crossfit(train);total=sum(len(train[n]["labels"]) for n in SPLITS);correct=sum(int(np.sum(predictions[n]==train[n]["labels"])) for n in SPLITS)
        report["variants"][variant]={"cohorts":cohorts,"aggregate":{"rows":total,"correct":correct,"accuracy":correct/total,"p177_reference_correct":2154,"delta_vs_p177":correct-2154,"fold_nets_vs_p89":[int(cohorts[n]["held"]["net"]) for n in SPLITS]},"thresholds":thresholds}
        for n in SPLITS:payload[f"{variant}_{n}_prediction"]=predictions[n];payload[f"{variant}_{n}_probability"]=probabilities[n]
    OUTPUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUTPUT/"predictions.npz",**payload);(OUTPUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
