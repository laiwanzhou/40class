"""Add P231 physical candidates one-at-a-time to the deployable P200 group bank."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p165_deployable_group_teacher import SPLITS,crossfit
from p173_vjepa_augmented_group_teacher import build_train_bank
H=Path(__file__).resolve().parent;O=H/"runs/p234_physical_group_audit_v1";P128=H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz";L=H/"runs/p158_lavila_frame_token_transformer_single_seed_v1/oof_predictions.npz";PHY=H/"runs/p231_depth_thermal_ir_oof_v1/oof_predictions.npz";V=("ir_depth","ir_thermal","ir_depth_thermal")
def al(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def main():
 phy=np.load(PHY);p=np.load(P128);l=np.load(L);report={"stage":"P234_physical_group_audit","status":"complete","protocol":{"base":"P200 bank","one_candidate_at_a_time":True,"held_labels_used_for_selection":False,"test_labels_read":False},"variants":{}};saved={}
 for v in V:
  tr,names=build_train_bank()
  for n in SPLITS:
   q=tr[n];q["bank"]=np.concatenate((q["bank"],al(p["probabilities"],p["sample_ids"],q["ids"])[:,None,:],al(l["probability"],l["sample_ids"],q["ids"])[:,None,:],al(phy[v+"_probability"],phy["sample_ids"],q["ids"])[:,None,:]),1)
  rep,pred,prob,thr=crossfit(tr);cor=sum(int(np.sum(pred[n]==tr[n]["labels"])) for n in SPLITS);report["variants"][v]={"correct":cor,"accuracy":cor/sum(len(tr[n]["labels"]) for n in SPLITS),"delta_vs_p200":cor-2157,"fold_nets_vs_p89":[rep[n]["held"]["net"] for n in SPLITS],"thresholds":thr,"cohorts":rep}
  for n in SPLITS:saved[f"{v}_{n}_prediction"]=pred[n];saved[f"{v}_{n}_probability"]=prob[n]
  print(json.dumps({v:report["variants"][v]}),flush=True)
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"predictions.npz",**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
