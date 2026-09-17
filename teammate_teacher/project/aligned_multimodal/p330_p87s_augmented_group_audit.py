"""Add exact terminal P87-S OOF posterior to P307/P310."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p165_deployable_group_teacher import SPLITS,crossfit
from p173_vjepa_augmented_group_teacher import build_train_bank
from p255_repeat_augmented_physical_group import al
from p301_union_pretrained_group_sequence_audit import COHORTS,sequence_crossfit
from p307_union_repeat_group_sequence_audit import SOURCES,P255,P270
H=Path(__file__).resolve().parent;O=H/"runs/p330_p87s_augmented_group_audit_v1";P328=H/"runs/p328_p87s_threefold_oof_audit_v1/oof_predictions.npz";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz"
def main():
 print("P330 tests the isolated exact terminal P87-S posterior in the frozen P307/P310 stack.",flush=True);tr,names=build_train_bank();p87=np.load(P328)
 for path,key,name in SOURCES:
  z=np.load(path)
  for cohort in SPLITS:
   q=tr[cohort];q["bank"]=np.concatenate((q["bank"],al(z[key],z["sample_ids"],q["ids"])[:,None,:]),1)
  names.append(name)
 for cohort in SPLITS:
  q=tr[cohort];q["bank"]=np.concatenate((q["bank"],p87[f"{cohort}_probability"][:,None,:]),1)
 names.append("p328_terminal_p87s")
 rep,gp,gprob,_=crossfit(tr);sp,srep=sequence_crossfit(tr,gp,gprob);labels=np.concatenate([tr[n]["labels"] for n in SPLITS]);group=np.concatenate([gp[n] for n in COHORTS]);sequence=np.concatenate([sp[n] for n in COHORTS]);p255=np.load(P255);p270=np.load(P270);old=np.concatenate([p255[f"{n}_held_prediction"] for n in COHORTS]);seq=np.concatenate([p270[f"{n}_held_prediction"] for n in COHORTS]);route=group!=old;precedence=seq.copy();precedence[route]=group[route];p310=np.load(P310)["prediction"];report={"stage":"P330_P87S_augmented_group_audit","status":"complete","protocol":{"only_change":"add exact terminal P87-S strict OOF posterior","group_sequence_precedence_frozen":True,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"group":{"correct":int(np.sum(group==labels)),"p307_reference":2178,"net_vs_p307":int(np.sum(group==labels)-2178),"cohorts":rep},"sequence":{"correct":int(np.sum(sequence==labels)),"p307_sequence_reference":2200,"fold_nets_vs_group":[srep[n]["held"]["net"] for n in COHORTS]},"precedence":{"correct":int(np.sum(precedence==labels)),"p310_reference":int(np.sum(p310==labels)),"net_vs_p310":int(np.sum(precedence==labels)-np.sum(p310==labels)),"changed_vs_p310":int(np.sum(precedence!=p310))},"gate":{"required_precedence_net":2,"pass":int(np.sum(precedence==labels)-np.sum(p310==labels))>=2}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,group_prediction=group,sequence_prediction=sequence,precedence_prediction=precedence,**{f"{n}_group_prediction":gp[n] for n in COHORTS},**{f"{n}_group_probability":gprob[n] for n in COHORTS});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: exact terminal P87-S posterior added to frozen P307/P310.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
