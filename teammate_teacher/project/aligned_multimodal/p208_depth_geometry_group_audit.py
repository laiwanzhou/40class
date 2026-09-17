"""Audit P207 Depth geometry variants as single additions to P177."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p165_deployable_group_teacher import SPLITS,crossfit
from p173_vjepa_augmented_group_teacher import build_train_bank
H=Path(__file__).resolve().parent;O=H/"runs/p208_depth_geometry_group_audit_v1";P128=H/"runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz";D=H/"runs/p207_depth_geometry_full_test_v1/predictions.npz";V=("geometry","depth_surface","depth_geometry")
def al(v,s,t):d={x:i for i,x in enumerate(s.astype(str))};return np.asarray(v)[np.asarray([d[x] for x in t.astype(str)])]
def base():
 tr,n=build_train_bank();p=np.load(P128)
 for k in SPLITS:q=tr[k];q["bank"]=np.concatenate((q["bank"],al(p["probabilities"],p["sample_ids"],q["ids"])[:,None,:]),axis=1)
 return tr
def main():
 d=np.load(D);report={"stage":"P208_depth_geometry_group_audit","status":"complete","protocol":{"base":"P177","single_addition":True,"test_labels_read":False},"variants":{}}
 for v in V:
  tr=base()
  for n in SPLITS:q=tr[n];q["bank"]=np.concatenate((q["bank"],al(d[v+"_probability"],d["sample_ids"],q["ids"])[:,None,:]),axis=1)
  rep,p,pr,t=crossfit(tr);total=sum(len(tr[n]["labels"]) for n in SPLITS);cor=sum(int(np.sum(p[n]==tr[n]["labels"])) for n in SPLITS);report["variants"][v]={"cohorts":rep,"aggregate":{"rows":total,"correct":cor,"accuracy":cor/total,"delta_vs_P177":cor-2154,"fold_nets":[int(rep[n]["held"]["net"]) for n in SPLITS]}}
 O.mkdir(parents=True,exist_ok=True);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
