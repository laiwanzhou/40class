"""Re-run frozen repeat consensus with the current deployable bank and P180 base."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import p89_build_dual_consensus_submission as io
from p134_frozen_repeat_consensus import repeat_proposal,select_rule
from p136_peer_support_repeat_gate import peer_candidate,select_rule as select_peer
from p193_deployable_candidate_ranker import bank
from p173_vjepa_augmented_group_teacher import build_test_bank

H=Path(__file__).resolve().parent;O=H/"runs/p194_current_bank_repeat_consensus_v1";P128=H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz";P128T=H/"runs/p176_p128_hierarchical_test_v1/test_predictions.npz";P180=H/"runs/p180_sequence_micro_teacher_v1/oof_predictions.npz";P180T=H/"runs/p180_sequence_micro_teacher_v1/submission_p180_sequence_micro.csv";P89=H/"runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv";TRAIN_META=H/"data/p85_recording_metadata/train_recording_metadata.csv";TEST_META=H/"data/p85_recording_metadata/test_recording_metadata.csv";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def al(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def csvpred(p):
 import csv
 with p.open("r",encoding="utf-8-sig",newline="") as f:return np.asarray([int(r["prediction"]) for r in csv.DictReader(f)])
def main():
 tr,names=bank();lookup={v:x for n in S for v,x in zip(tr[n]["ids"],tr[n]["bank"])};p180=np.load(P180);bmap={v:int(x) for v,x in zip(p180["sample_ids"].astype(str),p180["prediction"])};lmap={v:int(x) for v,x in zip(p180["sample_ids"].astype(str),p180["labels"])};reports={};outs={}
 for held in S:
  src=[n for n in S if n!=held];ids=np.concatenate([tr[n]["ids"] for n in src]);base=np.asarray([bmap[v] for v in ids]);labels=np.asarray([lmap[v] for v in ids]);cons,cache=select_rule(ids,base,labels,lookup);pp,sc,pc,_=peer_candidate(ids,base,lookup,TRAIN_META);peer=select_peer(base,labels,pp,sc,pc);kind="consensus" if cons["net"]>=peer["net"] else "peer";hids=tr[held]["ids"];hb=np.asarray([bmap[v] for v in hids]);hl=np.asarray([lmap[v] for v in hids])
  if kind=="consensus":proposal,adv,peers,grp=repeat_proposal(hids,hb,lookup,str(cons["mode"]),TRAIN_META);route=(proposal!=hb)&(peers>0)&(adv>=cons["threshold"]);sel=cons
  else:proposal,scores,peers,grp=peer_candidate(hids,hb,lookup,TRAIN_META);route=(proposal!=hb)&(peers>0)&(scores[:,peer["score_index"]]>=peer["threshold"]);sel=peer
  out=hb.copy();out[route]=proposal[route];outs[held]=out;bc=int(np.sum(hb==hl));cor=int(np.sum(out==hl));reports[held]={"source":src,"selected_kind":kind,"source_consensus":cons,"source_peer":peer,"held":{"base_correct":bc,"correct":cor,"net":cor-bc,"changed":int(route.sum()),"grouping":grp}}
 labels=np.concatenate([np.asarray([lmap[v] for v in tr[n]["ids"]]) for n in S]);base=np.concatenate([np.asarray([bmap[v] for v in tr[n]["ids"]]) for n in S]);out=np.concatenate([outs[n] for n in S]);correct=int(np.sum(out==labels));bc=int(np.sum(base==labels))
 # Test bank and full-source rule
 ids=np.concatenate([tr[n]["ids"] for n in S]);cons,_=select_rule(ids,base,labels,lookup);pp,sc,pc,_=peer_candidate(ids,base,lookup,TRAIN_META);peer=select_peer(base,labels,pp,sc,pc);kind="consensus" if cons["net"]>=peer["net"] else "peer";test=build_test_bank(names[:21],names[:24]);pt=np.load(P128T);test["bank"]=np.concatenate((test["bank"],al(pt["probabilities"],pt["sample_ids"],test["ids"])[:,None,:]),axis=1);tl={v:x for v,x in zip(test["ids"],test["bank"])};tb=csvpred(P180T)
 if kind=="consensus":proposal,adv,peers,grp=repeat_proposal(test["ids"],tb,tl,str(cons["mode"]),TEST_META);route=(proposal!=tb)&(peers>0)&(adv>=cons["threshold"]);sel=cons
 else:proposal,scores,peers,grp=peer_candidate(test["ids"],tb,tl,TEST_META);route=(proposal!=tb)&(peers>0)&(scores[:,peer["score_index"]]>=peer["threshold"]);sel=peer
 tout=tb.copy();tout[route]=proposal[route];O.mkdir(parents=True,exist_ok=True);sub=O/"submission_p194_repeat.csv";io.write_submission(sub,io.read_rows(P89),tout);report={"stage":"P194_current_bank_repeat_consensus","status":"complete","protocol":{"frozen_repeat_geometry":True,"current_bank_experts":len(names),"held_labels_used_for_selection":False,"test_labels_read":False},"cohorts":reports,"aggregate":{"rows":len(labels),"base_correct":bc,"correct":correct,"accuracy":correct/len(labels),"net_vs_p180":correct-bc,"fold_nets":[reports[n]["held"]["net"] for n in S]},"test":{"selected_kind":kind,"source_rule":sel,"changes_vs_p180":int(route.sum()),"grouping":grp,"submission":str(sub.resolve()),"test_labels_read":False}};np.savez_compressed(O/"predictions.npz",sample_ids=test["ids"],base_prediction=tb,prediction=tout,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
