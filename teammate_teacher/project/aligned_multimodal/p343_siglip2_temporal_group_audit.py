"""Add P342 SigLIP2 temporal posterior to frozen P307/P310."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p165_deployable_group_teacher import SPLITS,crossfit
from p173_vjepa_augmented_group_teacher import build_train_bank
from p255_repeat_augmented_physical_group import al
from p301_union_pretrained_group_sequence_audit import COHORTS,sequence_crossfit
from p307_union_repeat_group_sequence_audit import SOURCES,P255,P270
H=Path(__file__).resolve().parent;O=H/"runs/p343_siglip2_temporal_group_audit_v1";SIG=H/"runs/p342_siglip2_temporal_state_transformer_oof_v1/oof_predictions.npz";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz"
def main():
 print("P343 tests only the P342 SigLIP2 temporal posterior in frozen P307/P310.",flush=True);tr,names=build_train_bank()
 for path,key,name in (*SOURCES,(SIG,"probability","p342_siglip2_temporal")):
  z=np.load(path)
  for c in SPLITS:
   q=tr[c];q["bank"]=np.concatenate((q["bank"],al(z[key],z["sample_ids"],q["ids"])[:,None,:]),1)
  names.append(name)
 rep,gp,gprob,_=crossfit(tr);sp,srep=sequence_crossfit(tr,gp,gprob);labels=np.concatenate([tr[n]["labels"] for n in SPLITS]);group=np.concatenate([gp[n] for n in COHORTS]);sequence=np.concatenate([sp[n] for n in COHORTS]);p255=np.load(P255);p270=np.load(P270);old=np.concatenate([p255[f"{n}_held_prediction"] for n in COHORTS]);seq=np.concatenate([p270[f"{n}_held_prediction"] for n in COHORTS]);route=group!=old;precedence=seq.copy();precedence[route]=group[route];p310=np.load(P310)["prediction"];report={"stage":"P343_SigLIP2_temporal_group_audit","status":"complete","protocol":{"only_change":"add P342 temporal posterior","frozen_stack":True,"test_rows_loaded":0,"test_labels_read":False},"group":{"correct":int(np.sum(group==labels)),"net_vs_p307":int(np.sum(group==labels)-2178),"cohorts":rep},"sequence":{"correct":int(np.sum(sequence==labels)),"fold_nets_vs_group":[srep[n]["held"]["net"] for n in COHORTS]},"precedence":{"correct":int(np.sum(precedence==labels)),"p310_reference":2211,"net_vs_p310":int(np.sum(precedence==labels)-2211),"changed_vs_p310":int(np.sum(precedence!=p310))},"gate":{"pass":int(np.sum(precedence==labels)-2211)>=1}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,group_prediction=group,precedence_prediction=precedence,**{f"{n}_group_probability":gprob[n] for n in COHORTS});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
