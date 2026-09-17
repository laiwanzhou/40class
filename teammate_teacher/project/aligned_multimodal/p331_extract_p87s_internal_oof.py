"""Extract terminal P87-S internal representations from three strict holdouts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from p117_transductive_multicandidate_router import load_candidate_splits
from p87s_deploy_model import deployment_model_config,build_p87s_deploy_model
from p86_cached_motion_data import P86CachedSequenceMotionDataset,collate_p86_cached_motion
from train_p86_mobind_fusion_proxy import model_forward
H=Path(__file__).resolve().parent;O=H/"runs/p331_p87s_internal_oof_v1";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def norm(x):return torch.nn.functional.normalize(x.float(),dim=-1)
def main():
 print("P331 extracts strict held-cohort P87-S visual/window/motion embeddings and reliability gates.",flush=True);data=load_candidate_splits();O.mkdir(parents=True,exist_ok=True);saved={};report={"stage":"P331_extract_P87S_internal_OOF","status":"complete","protocol":{"fresh_checkpoint_per_cohort":True,"cached_sequence_inference":True,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"cohorts":{}};device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 for i,n in enumerate(S,1):
  run=H/f"runs/p87s_fusion_holdout{i}_c0_v1";seq=H/f"runs/p87s_mc3_sequence_holdout{i}_v1";ck=torch.load(run/"unified_student.pt",map_location="cpu",weights_only=False);summary=json.loads((run/"summary.json").read_text());cfg=deployment_model_config(ck["visual_config"],ck["pretrain_config"],ck["modality"],summary["config"]);model=build_p87s_deploy_model(cfg);model.load_state_dict(ck["model_state"],strict=True);model.eval().to(device);full=P86CachedSequenceMotionDataset(seq,H/"runs/p86_motion_window_cache_t16_v1",H/"runs/p86_visual_pixel_cache_t16_r160_v12",H/"runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz",H/"runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz");ids=data[n].split.sample_ids.astype(str);idx=np.asarray([full.index_lookup[x] for x in ids]);ds=P86CachedSequenceMotionDataset(seq,H/"runs/p86_motion_window_cache_t16_v1",H/"runs/p86_visual_pixel_cache_t16_r160_v12",H/"runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz",H/"runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz",indices=idx);loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0,collate_fn=collate_p86_cached_motion);vals={k:[] for k in ("logits","visual_embedding","window_state","motion_semantic","motion_logits","gates")};order=[]
  with torch.inference_mode():
   for batch in loader:
    order.extend(batch["sample_id"]);batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):out=model_forward(model,batch)
    window=out["window_embeddings"].float();window_state=torch.cat((norm(window[:,0]),norm(window[:,1]),norm(window[:,1]-window[:,0])),1);gates=torch.cat((out["view_weight"].float().flatten(1),out["motion_reliability"].float().flatten(1),out["motion_global_reliability"].float().flatten(1),out["motion_available"].float().flatten(1)),1);vals["logits"].append(out["logits"].float().cpu().numpy());vals["visual_embedding"].append(norm(out["visual_embedding"]).cpu().numpy());vals["window_state"].append(window_state.cpu().numpy());vals["motion_semantic"].append(norm(out["motion_semantic_embedding"]).cpu().numpy());vals["motion_logits"].append(out["motion_logits"].float().cpu().numpy());vals["gates"].append(gates.cpu().numpy())
  if order!=ids.tolist():raise RuntimeError(n+" order")
  for k,v in vals.items():saved[f"{n}_{k}"]=np.concatenate(v).astype(np.float16)
  saved[f"{n}_sample_ids"]=ids;saved[f"{n}_labels"]=data[n].split.labels.astype(int);report["cohorts"][n]={"rows":len(ids),"dimensions":{k:int(np.concatenate(v).shape[1]) for k,v in vals.items()}};model.to("cpu");torch.cuda.empty_cache()
 np.savez_compressed(O/"internal_features.npz",**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: strict held-cohort internal representation extraction for terminal P87-S.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
