"""Add P306 beside P253 and audit the frozen group/sequence stack."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p165_deployable_group_teacher import SPLITS,crossfit
from p173_vjepa_augmented_group_teacher import build_train_bank
from p255_repeat_augmented_physical_group import al
from p301_union_pretrained_group_sequence_audit import COHORTS,sequence_crossfit
H=Path(__file__).resolve().parent;O=H/"runs/p307_union_repeat_group_sequence_audit_v1"
SOURCES=((H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz","probabilities","p128_hierarchical_multimodal"),(H/"runs/p158_lavila_frame_token_transformer_single_seed_v1/oof_predictions.npz","probability","p158_lavila_frame_token"),(H/"runs/p238_physical_token_transformer_oof_v1/oof_predictions.npz","probability","p238_physical_token"),(H/"runs/p231_depth_thermal_ir_oof_v1/oof_predictions.npz","ir_thermal_probability","p231_ir_thermal"),(H/"runs/p253_repeat_physical_transformer_oof_v1/oof_predictions.npz","probability","p253_repeat_physical"),(H/"runs/p306_union_full_repeat_physical_oof_v1/oof_predictions.npz","probability","p306_union_repeat_physical"))
P255=H/"runs/p255_repeat_augmented_physical_group_v1/predictions.npz";P270=H/"runs/p270_fixed_emission065_transition045_v1/predictions.npz"
def main():
 tr,names=build_train_bank()
 for path,key,name in SOURCES:
  z=np.load(path)
  for cohort in SPLITS:
   q=tr[cohort];q["bank"]=np.concatenate((q["bank"],al(z[key],z["sample_ids"],q["ids"])[:,None,:]),1)
  names.append(name)
 group_report,gp,gprob,_=crossfit(tr);sp,sequence_report=sequence_crossfit(tr,gp,gprob);labels=np.concatenate([tr[n]["labels"] for n in SPLITS]);group=np.concatenate([gp[n] for n in COHORTS]);sequence=np.concatenate([sp[n] for n in COHORTS]);a=np.load(P255);b=np.load(P270);p255=np.concatenate([a[f"{n}_held_prediction"] for n in COHORTS]);p270=np.concatenate([b[f"{n}_held_prediction"] for n in COHORTS]);gc=int(np.sum(group==labels));sc=int(np.sum(sequence==labels));report={"stage":"P307_union_repeat_group_sequence_audit","status":"complete","protocol":{"change_vs_p255":"add P306 beside P253","recovered_extra_rows":10,"group_outer_crossfit":True,"sequence_recipe_frozen_from_p270":True,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"experts":names,"group":{"correct":gc,"accuracy":float(np.mean(group==labels)),"p255_reference_correct":int(np.sum(p255==labels)),"net_vs_p255":gc-int(np.sum(p255==labels)),"cohorts":group_report},"sequence":{"correct":sc,"accuracy":float(np.mean(sequence==labels)),"p270_reference_correct":int(np.sum(p270==labels)),"net_vs_p270":sc-int(np.sum(p270==labels)),"fold_nets_vs_group":[sequence_report[n]["held"]["net"] for n in COHORTS],"cohorts":sequence_report},"decision":"train_matched_test_head" if sc>int(np.sum(p270==labels)) else "reject"};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,group_prediction=group,sequence_prediction=sequence,**{f"{n}_group_prediction":gp[n] for n in COHORTS},**{f"{n}_group_probability":gprob[n] for n in COHORTS},**{f"{n}_sequence_prediction":sp[n] for n in COHORTS});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
