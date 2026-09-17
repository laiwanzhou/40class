"""Outer-safe Top-5 consensus gate over seven heterogeneous experts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;O=H/"runs/p316_topk_consensus_uncertainty_gate_v1";P244=H/"runs/p244_dual_physical_group_v1/predictions.npz";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0");NAMES=("expanded_p146_workspace_token","expanded_p142_token","p90_internvideo2_l_early_late","p128_hierarchical_multimodal","a18_best_session","expanded_p123_old_ir_dense","p90_visual_equal")
def build():
 data=load_candidate_splits(full_visual_bank=True,structured_bank=True,legacy_visual_bank=True,hand_object_bank=True,vjepa_dense_bank=True,nonvisual_bank=True,hierarchical_bank=True,epic_bank=True,expanded_bank=True);g=np.load(P244);z=np.load(P310);parts={};off=0
 for n in S:
  q=data[n].split;m=len(q.labels);base=z["prediction"][off:off+m].astype(int);p=g[f"{n}_held_probability"].astype(float);bank=np.stack([data[n].candidates[x] for x in NAMES],1);pred=bank.argmax(2);counts=np.stack([(pred==c).sum(1) for c in range(40)],1);counts[np.arange(m),base]=-1;alt=counts.argmax(1);votes=counts[np.arange(m),alt];mean=bank.mean(1);rows=np.arange(m);parts[n]={"labels":q.labels.astype(int),"users":q.users.astype(str),"base":base,"alt":alt,"votes":votes,"gap":mean[rows,alt]-mean[rows,base],"base_conf":p[rows,base],"in_top5":np.any(np.argsort(-p,axis=1)[:,:5]==alt[:,None],1)};off+=m
 return parts
def select(part):
 gain=(part["alt"]==part["labels"]).astype(int)-(part["base"]==part["labels"]).astype(int);best=None
 for v in range(2,8):
  for gap in np.linspace(-.1,.5,49):
   for bc in (.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,1.01):
    m=part["in_top5"]&(part["alt"]!=part["base"])&(part["votes"]>=v)&(part["gap"]>=gap)&(part["base_conf"]<=bc);peru={u:int(gain[m&(part["users"]==u)].sum()) for u in sorted(set(part["users"].tolist()))};perc={c:int(gain[m&(part["cohort"]==c)].sum()) for c in sorted(set(part["cohort"].tolist()))};r=int(np.sum(m&(gain>0)));h=int(np.sum(m&(gain<0)));row={"votes":v,"gap":float(gap),"base_confidence":bc,"changed":int(m.sum()),"rescue":r,"harm":h,"net":r-h,"minimum_user_gain":min(peru.values()),"minimum_cohort_gain":min(perc.values()),"per_user":peru,"per_cohort":perc};key=(row["minimum_user_gain"]>=0,row["minimum_cohort_gain"]>=0,row["net"],r,-h,-row["changed"],v,gap,-bc)
    if best is None or key>best[0]:best=(key,row)
 return best[1]
def apply(part,r):
 m=part["in_top5"]&(part["alt"]!=part["base"])&(part["votes"]>=r["votes"])&(part["gap"]>=r["gap"])&(part["base_conf"]<=r["base_confidence"]);out=part["base"].copy();out[m]=part["alt"][m];return out,m
def cat(ps):return {k:np.concatenate([p[k] for p in ps]) for k in ps[0]}
def main():
 parts=build()
 for n in S:parts[n]["cohort"]=np.full(len(parts[n]["labels"]),n,object)
 report={"stage":"P316_Top5_consensus_uncertainty_gate","status":"complete","protocol":{"base":"P310","experts":list(NAMES),"candidate":"plurality alternative restricted to P244 Top-5","thresholds_selected_on_source_only":True,"held_labels_used_for_selection":False,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};outs={}
 for held in S:
  src=[n for n in S if n!=held];rule=select(cat([parts[n] for n in src]));out,m=apply(parts[held],rule);outs[held]=out;y=parts[held]["labels"];b=parts[held]["base"];report["cohorts"][held]={"source":src,"source_rule":rule,"held":{"rows":len(y),"base_correct":int(np.sum(b==y)),"correct":int(np.sum(out==y)),"net":int(np.sum(out==y)-np.sum(b==y)),"changed":int(m.sum()),"rescue":int(np.sum(m&(b!=y)&(out==y))),"harm":int(np.sum(m&(b==y)&(out!=y)))}}
 labels=np.concatenate([parts[n]["labels"] for n in S]);base=np.concatenate([parts[n]["base"] for n in S]);out=np.concatenate([outs[n] for n in S]);report["aggregate"]={"rows":len(labels),"base_correct":int(np.sum(base==labels)),"correct":int(np.sum(out==labels)),"accuracy":float(np.mean(out==labels)),"net_vs_p310":int(np.sum(out==labels)-np.sum(base==labels)),"fold_nets":[report["cohorts"][n]["held"]["net"] for n in S]};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",labels=labels,base_prediction=base,prediction=out,**{f"{n}_held_prediction":outs[n] for n in S});(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
