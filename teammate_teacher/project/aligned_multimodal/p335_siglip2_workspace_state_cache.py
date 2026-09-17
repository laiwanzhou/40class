"""Frozen SigLIP2-B/16 hand-workspace state cache from the Train pixel cache."""
from __future__ import annotations
import csv,hashlib,json,time
from pathlib import Path
import numpy as np,torch,timm
from safetensors import safe_open
H=Path(__file__).resolve().parent;PIX=H/"runs/p86_visual_pixel_cache_t16_r160_v12";O=H/"runs/p335_siglip2_workspace_state_cache_v1";W=Path.home()/".cache/modelscope/hub/models/timm/ViT-B-16-SigLIP2-256/open_clip_model.safetensors";POSITIONS=(0,5,10,15);SHAPE=(2914,2,4,768)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def model(device):
 m=timm.create_model("vit_base_patch16_siglip_256",pretrained=False,num_classes=0)
 with safe_open(W,framework="pt",device="cpu") as f:s={k.removeprefix("visual.trunk."):f.get_tensor(k) for k in f.keys() if k.startswith("visual.trunk.")}
 m.load_state_dict(s,strict=True);return m.eval().to(device)
def main():
 print("P335 extracts frozen SigLIP2 hand-workspace state embeddings from 8 frames per Train trial.",flush=True);O.mkdir(parents=True,exist_ok=True);images=np.load(PIX/"images.npy",mmap_mode="r");fp=O/"features.npy";dp=O/"done.npy";new=not dp.exists();feat=np.lib.format.open_memmap(fp,mode="r+" if fp.exists() else "w+",dtype=np.float16,shape=SHAPE);done=np.lib.format.open_memmap(dp,mode="r+" if dp.exists() else "w+",dtype=np.bool_,shape=(SHAPE[0],));
 if new:done[:]=False
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");m=model(device);pending=np.flatnonzero(~np.asarray(done));start=time.time()
 for lo in range(0,len(pending),8):
  idx=pending[lo:lo+8];raw=np.asarray(images[idx][:,:,POSITIONS,2],np.float32);x=torch.from_numpy(raw).reshape(-1,1,160,160).to(device)/255.;x=x.repeat(1,3,1,1);x=torch.nn.functional.interpolate(x,size=(256,256),mode="bicubic",align_corners=False);x=(x-.5)/.5
  with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):y=m(x)
  feat[idx]=y.reshape(len(idx),2,4,768).half().cpu().numpy();done[idx]=True
  if (lo//8+1)%20==0:feat.flush();done.flush();print(json.dumps({"stage":"siglip2_cache","done":int(np.sum(done)),"total":len(done),"seconds":round(time.time()-start,1)}),flush=True)
 feat.flush();done.flush();report={"stage":"P335_SigLIP2_workspace_state_cache","status":"complete","protocol":{"model":"timm/ViT-B-16-SigLIP2-256","weights_sha256":sha(W),"weights_expected_sha256":"97816a02002b8b0c41cb507e2cc250eb39ef4ea2a63d1b6e44490a2039cfcdda","input":"P86 hand-workspace grayscale repeated RGB","windows":2,"frame_positions":list(POSITIONS),"image_size":256,"frozen":True,"labels_used":False,"test_rows_loaded":0},"features":{"shape":list(feat.shape),"dtype":str(feat.dtype),"complete":bool(np.asarray(done).all())},"elapsed_seconds":time.time()-start};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
