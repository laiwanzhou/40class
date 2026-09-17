"""Frozen SigLIP2 scene/person/workspace state cache; reuses P335 workspace."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,torch
from p335_siglip2_workspace_state_cache import H,PIX,O as P335,POSITIONS,model
O=H/"runs/p340_siglip2_threeview_state_cache_v1";SHAPE=(2914,3,2,4,768)
def main():
 print("P340 extends the verified P335 SigLIP2 cache to scene and person with identical frames.",flush=True);O.mkdir(parents=True,exist_ok=True);images=np.load(PIX/"images.npy",mmap_mode="r");fp=O/"features.npy";dp=O/"done.npy";new=not dp.exists();feat=np.lib.format.open_memmap(fp,mode="r+" if fp.exists() else "w+",dtype=np.float16,shape=SHAPE);done=np.lib.format.open_memmap(dp,mode="r+" if dp.exists() else "w+",dtype=np.bool_,shape=(2914,2));
 if new:done[:]=False;feat[:,2]=np.load(P335/"features.npy",mmap_mode="r")
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");m=model(device);start=time.time()
 for view in (0,1):
  pending=np.flatnonzero(~np.asarray(done[:,view]))
  for lo in range(0,len(pending),8):
   idx=pending[lo:lo+8];raw=np.asarray(images[idx][:,:,POSITIONS,view],np.float32);x=torch.from_numpy(raw).reshape(-1,1,160,160).to(device)/255.;x=x.repeat(1,3,1,1);x=torch.nn.functional.interpolate(x,size=(256,256),mode="bicubic",align_corners=False);x=(x-.5)/.5
   with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):y=m(x)
   feat[idx,view]=y.reshape(len(idx),2,4,768).half().cpu().numpy();done[idx,view]=True
   if (lo//8+1)%25==0:feat.flush();done.flush();print(json.dumps({"view":view,"done":int(done[:,view].sum()),"total":2914,"seconds":round(time.time()-start,1)}),flush=True)
 feat.flush();done.flush();report={"stage":"P340_SigLIP2_threeview_state_cache","status":"complete","protocol":{"model":"ViT-B-16-SigLIP2-256","views":["scene","person","workspace"],"windows":2,"positions":list(POSITIONS),"workspace_reused_from_P335":True,"frozen":True,"labels_used":False,"test_rows_loaded":0},"shape":list(feat.shape),"complete":bool(done.all()),"elapsed_seconds":time.time()-start};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
