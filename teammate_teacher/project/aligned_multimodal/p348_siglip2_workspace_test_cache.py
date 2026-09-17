"""Exact Test counterpart of P335 SigLIP2 workspace cache."""
from __future__ import annotations
import csv,json,time
from pathlib import Path
import numpy as np,torch
from p335_siglip2_workspace_state_cache import H,POSITIONS,model
PIX=H/"runs/p87s_test_pixel_cache_t16_r160_v1";O=H/"runs/p348_siglip2_workspace_test_cache_v1";SHAPE=(405,2,4,768)
def main():
 print("P348 extracts the exact P335 SigLIP2 workspace contract for all 405 Test rows.",flush=True);O.mkdir(parents=True,exist_ok=True);images=np.load(PIX/"images.npy",mmap_mode="r");feat=np.lib.format.open_memmap(O/"features.npy",mode="w+",dtype=np.float16,shape=SHAPE);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");m=model(device);start=time.time()
 for lo in range(0,len(images),8):
  idx=np.arange(lo,min(lo+8,len(images)));raw=np.asarray(images[idx][:,:,POSITIONS,2],np.float32);x=torch.from_numpy(raw).reshape(-1,1,160,160).to(device)/255.;x=x.repeat(1,3,1,1);x=torch.nn.functional.interpolate(x,size=(256,256),mode="bicubic",align_corners=False);x=(x-.5)/.5
  with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):y=m(x)
  feat[idx]=y.reshape(len(idx),2,4,768).half().cpu().numpy()
 feat.flush()
 with (PIX/"rows.csv").open("r",encoding="utf-8-sig",newline="") as f:ids=np.asarray([r["sample_id"] for r in csv.DictReader(f)])
 np.save(O/"sample_ids.npy",ids);report={"stage":"P348_SigLIP2_workspace_Test_cache","status":"complete","rows":len(ids),"shape":list(feat.shape),"source":"P87S exact Test pixel cache","test_labels_read":False,"elapsed_seconds":time.time()-start};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
