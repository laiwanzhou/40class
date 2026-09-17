"""Extract early/late x scene/person/workspace Depth VideoMAE2 features."""
from __future__ import annotations
import csv,json,time
from pathlib import Path
import numpy as np,torch
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset,DataLoader
from transformers import VideoMAEImageProcessor
import p91_videomaev2_modality_teacher as p91
from p90_videomae_lora_teacher import VideoCollator,read_aligned_rows
from build_p46_videomae_cache import safe_relative,square_crop
from build_p30_shared_dir_roi_feature_cache import read_depth
from audit_yolo11_pose_skeleton import frame_map
H=Path(__file__).resolve().parent;TRAIN_ROI=H/"runs/p29_dir_multiscale_roi_full";TEST_ROI=H/"runs/p29_dir_multiscale_roi_test";TEST_MAN=H/"data/p46_test_union_manifest.csv";O=H/"runs/p281_depth_early_late_v1"
def test_rows():
 with TEST_MAN.open("r",encoding="utf-8-sig",newline="") as f:src=list(csv.DictReader(f))
 return [{**r,"sample_id":r["official_sample_id"],"source_id":r["sample_id"],"ir_dir":r["ir_path"],"depth_dir":r["depth_color_path"],"user_id":"anonymous"} for r in src]
def prepare(row,root):
 path=root/"trial_roi_cache"/safe_relative(row["source_id"]).with_suffix(".npz")
 if not path.is_file():
  black=np.zeros((224,224,3),np.uint8);return [[black]*16 for _ in range(6)],False
 with np.load(path) as z:ids=z["frame_ids"].astype(str);names=tuple(z["region_names"].astype(str));boxes=z["roi_boxes_xyxy"].astype(np.float32);valid=z["roi_valid"].astype(bool)
 paths=frame_map(Path(row["depth_dir"]),"depth");pi=names.index("full_body");wi=names.index("hand_workspace");clips=[]
 for lo,hi in ((0.,.7),(.3,1.)):
  ix=np.rint(np.linspace(lo*(len(ids)-1),hi*(len(ids)-1),16)).astype(int);scene=[];person=[];work=[]
  for i in ix:
   im=read_depth(paths[ids[i]]);h,w=im.shape[:2];scale=np.asarray([w/640,h/480,w/640,h/480],np.float32);pb=boxes[i,pi]*scale if valid[i,pi] else np.full(4,np.nan);wb=boxes[i,wi]*scale if valid[i,wi] else pb;scene.append(im);person.append(square_crop(im,pb,scale=1.15));work.append(square_crop(im,wb,scale=1.4))
  clips.extend((scene,person,work))
 return clips,True
class DS(Dataset):
 def __init__(self,rows,root):self.rows=rows;self.root=root
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  clips,av=prepare(self.rows[i],self.root);return {"videos":clips,"clip_indices":list(range(6)),"label":0,"sample_id":self.rows[i]["sample_id"],"available":av}
def collator(processor):
 base=VideoCollator(processor)
 def fn(items):out=base(items);out["available"]=[bool(x["available"]) for x in items];return out
 return fn
@torch.inference_mode()
def extract(rows,root,model,processor,name):
 partial=O/f"{name}_partial.npz";fs=[];acts=[];ids=[];av=[]
 if partial.exists():
  with np.load(partial,allow_pickle=False) as z:
   ids=z["sample_ids"].astype(str).tolist();fs=[z["features"].copy()];acts=[z["action_logits"].copy()];av=z["available"].astype(bool).tolist()
  expected=[row["sample_id"] for row in rows[:len(ids)]]
  if ids!=expected:raise RuntimeError(f"{name} partial order mismatch")
  print(json.dumps({"split":name,"resume_rows":len(ids),"total":len(rows)}),flush=True)
 loader=DataLoader(DS(rows[len(ids):],root),6,False,num_workers=0,collate_fn=collator(processor),pin_memory=True);start=time.time()
 for bi,b in enumerate(loader):
  pix=b["pixel_values"].permute(0,2,1,3,4).to("cuda",dtype=torch.float16);f=model.forward_features(pix);a=model.head(f);n=len(b["sample_ids"]);fs.append(f.reshape(n,6,768).half().cpu().numpy());acts.append(a.reshape(n,6,710).half().cpu().numpy());ids.extend(b["sample_ids"]);av.extend(b["available"])
  if (bi+1)%20==0:
   fnow=np.concatenate(fs);anow=np.concatenate(acts);np.savez(partial,sample_ids=np.asarray(ids),features=fnow,action_logits=anow,available=np.asarray(av,bool));fs=[fnow];acts=[anow];print(json.dumps({"split":name,"rows":len(ids),"total":len(rows),"minutes":round((time.time()-start)/60,2)}),flush=True)
 f=np.concatenate(fs);a=np.concatenate(acts);av=np.asarray(av,bool);f[~av]=0;a[~av]=0;return {"sample_ids":np.asarray(ids),"features":f,"action_logits":a,"available":av}
def main():
 model,_=p91.build_model(torch.device("cuda"));snap=Path(snapshot_download(p91.PROCESSOR_REPO,local_files_only=True));processor=VideoMAEImageProcessor.from_pretrained(snap,local_files_only=True);O.mkdir(parents=True,exist_ok=True);tr=extract(read_aligned_rows(),TRAIN_ROI,model,processor,"train");te=extract(test_rows(),TEST_ROI,model,processor,"test");np.savez_compressed(O/"train_features.npz",**tr);np.savez_compressed(O/"test_features.npz",**te);report={"stage":"P281_depth_early_late_extraction","status":"complete","protocol":{"windows":[[0,.7],[.3,1]],"views":["scene","person","workspace"],"clips":6,"frames_per_clip":16,"checkpoint":"VideoMAE2 K710 distilled","test_labels_read":False},"train":{"rows":len(tr["sample_ids"]),"available":int(tr["available"].sum())},"test":{"rows":len(te["sample_ids"]),"available":int(te["available"].sum())}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
