"""Extract all 2914 rows in each strict outer P87-S embedding space."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from p87s_deploy_model import deployment_model_config,build_p87s_deploy_model
from p86_cached_motion_data import P86CachedSequenceMotionDataset,collate_p86_cached_motion
from train_p86_mobind_fusion_proxy import model_forward
H=Path(__file__).resolve().parent;O=H/"runs/p333_p87s_outer_embedding_spaces_v1";S=("H1_selection","H2_confirmation","H3_independent_fold0")
def main():
 print("P333 extracts all Train rows in each source-trained outer P87-S embedding space.",flush=True);O.mkdir(parents=True,exist_ok=True);saved={};report={"stage":"P333_extract_P87S_outer_embedding_spaces","status":"complete","protocol":{"each_space_uses_its_source_trained_checkpoint":True,"rows_per_space":2914,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"spaces":{}};device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 for i,n in enumerate(S,1):
  run=H/f"runs/p87s_fusion_holdout{i}_c0_v1";seq=H/f"runs/p87s_mc3_sequence_holdout{i}_v1";ck=torch.load(run/"unified_student.pt",map_location="cpu",weights_only=False);summary=json.loads((run/"summary.json").read_text());cfg=deployment_model_config(ck["visual_config"],ck["pretrain_config"],ck["modality"],summary["config"]);model=build_p87s_deploy_model(cfg);model.load_state_dict(ck["model_state"],strict=True);model.eval().to(device);ds=P86CachedSequenceMotionDataset(seq,H/"runs/p86_motion_window_cache_t16_v1",H/"runs/p86_visual_pixel_cache_t16_r160_v12",H/"runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz",H/"runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz");loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0,collate_fn=collate_p86_cached_motion);emb=[];log=[];motion=[];ids=[]
  with torch.inference_mode():
   for batch in loader:
    ids.extend(batch["sample_id"]);batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):out=model_forward(model,batch)
    emb.append(torch.nn.functional.normalize(out["visual_embedding"].float(),dim=1).cpu().numpy());log.append(out["logits"].float().cpu().numpy());motion.append(out["motion_logits"].float().cpu().numpy())
  if ids!=[r["sample_id"] for r in ds.rows]:raise RuntimeError(n+" order")
  saved[f"{n}_sample_ids"]=np.asarray(ids);saved[f"{n}_embedding"]=np.concatenate(emb).astype(np.float16);saved[f"{n}_logits"]=np.concatenate(log).astype(np.float16);saved[f"{n}_motion_logits"]=np.concatenate(motion).astype(np.float16);report["spaces"][n]={"rows":len(ids),"embedding_dim":512};model.to("cpu");torch.cuda.empty_cache()
 np.savez_compressed(O/"embedding_spaces.npz",**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
