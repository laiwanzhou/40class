"""Extract exact early/late x scene/person/workspace Thermal VideoMAE2 features."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import cv2,numpy as np,torch
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset,DataLoader
from transformers import VideoMAEImageProcessor
import p91_videomaev2_modality_teacher as p91
from p90_videomae_lora_teacher import VideoCollator,read_aligned_rows
from build_p46_videomae_cache import safe_relative,square_crop
H=Path(__file__).resolve().parent;TRAIN_ROI=H/"runs/p29_dir_multiscale_roi_full";TEST_ROI=H/"runs/p29_dir_multiscale_roi_test";TEST_MAN=H/"data/p46_test_union_manifest.csv";O=H/"runs/p271_thermal_early_late_v1"
def test_rows():
 with TEST_MAN.open("r",encoding="utf-8-sig",newline="") as f:src=list(csv.DictReader(f))
 return [{**r,"sample_id":r["official_sample_id"],"source_id":r["sample_id"],"ir_dir":r["ir_path"],"user_id":"anonymous"} for r in src]
def thermal_read(path):
 b=cv2.imread(str(path),cv2.IMREAD_COLOR)
 if b is None:raise RuntimeError(path)
 return cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
def prepare(row,roi_root):
 roi=roi_root/"trial_roi_cache"/safe_relative(row["source_id"]).with_suffix(".npz")
 if not roi.is_file() or not any(p91.thermal_dir(row).glob("*.jpg")):
  black=np.zeros((224,224,3),np.uint8);return [[black]*16 for _ in range(6)],False
 with np.load(roi) as z:frame_ids=z["frame_ids"].astype(str);names=tuple(z["region_names"].astype(str));boxes=z["roi_boxes_xyxy"].astype(np.float32);valid=z["roi_valid"].astype(bool)
 files=sorted(p91.thermal_dir(row).glob("*.jpg"));pi=names.index("full_body");wi=names.index("hand_workspace");clips=[]
 for lo,hi in ((0.,.7),(.3,1.)):
  ti=np.rint(np.linspace(lo*(len(files)-1),hi*(len(files)-1),16)).astype(int);ri=np.rint(ti/max(len(files)-1,1)*(len(frame_ids)-1)).astype(int);scene=[];person=[];work=[]
  for fi,r in zip(ti,ri):
   im=thermal_read(files[fi]);h,w=im.shape[:2];scale=np.asarray([w/640,h/480,w/640,h/480],np.float32);pb=boxes[r,pi]*scale if valid[r,pi] else np.full(4,np.nan);wb=boxes[r,wi]*scale if valid[r,wi] else pb;scene.append(im);person.append(square_crop(im,pb,scale=1.15));work.append(square_crop(im,wb,scale=1.4))
  clips.extend((scene,person,work))
 return clips,True
class DS(Dataset):
 def __init__(self,rows,root):self.rows=rows;self.root=root
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  clips,av=prepare(self.rows[i],self.root);return {"videos":clips,"clip_indices":list(range(6)),"label":0,"sample_id":self.rows[i]["sample_id"],"available":av}
def collator(processor):
 base=VideoCollator(processor)
 def apply(items):
  out=base(items);out["available"]=[bool(item["available"]) for item in items];return out
 return apply
@torch.inference_mode()
def extract(rows,root,model,processor,name):
 loader=DataLoader(DS(rows,root),batch_size=6,shuffle=False,num_workers=0,collate_fn=collator(processor),pin_memory=True);fs=[];acts=[];ids=[];available=[];start=time.time()
 for bi,b in enumerate(loader):
  pix=b["pixel_values"].permute(0,2,1,3,4).to("cuda",dtype=torch.float16);f=model.forward_features(pix);a=model.head(f);n=len(b["sample_ids"]);fs.append(f.reshape(n,6,768).half().cpu().numpy());acts.append(a.reshape(n,6,710).half().cpu().numpy());ids.extend(b["sample_ids"]);available.extend(b["available"])
  if (bi+1)%20==0:print(json.dumps({"split":name,"rows":len(ids),"total":len(rows),"minutes":round((time.time()-start)/60,2)}),flush=True)
 f=np.concatenate(fs);a=np.concatenate(acts);av=np.asarray(available,bool);f[~av]=0;a[~av]=0;return {"sample_ids":np.asarray(ids),"features":f,"action_logits":a,"available":av}
def main():
 device=torch.device("cuda");model,_=p91.build_model(device);snap=Path(snapshot_download(p91.PROCESSOR_REPO,local_files_only=True));processor=VideoMAEImageProcessor.from_pretrained(snap,local_files_only=True);O.mkdir(parents=True,exist_ok=True);train=extract(read_aligned_rows(),TRAIN_ROI,model,processor,"train");test=extract(test_rows(),TEST_ROI,model,processor,"test");np.savez_compressed(O/"train_features.npz",**train);np.savez_compressed(O/"test_features.npz",**test);report={"stage":"P271_thermal_early_late_extraction","status":"complete","protocol":{"windows":[[0,.7],[.3,1]],"views":["scene","person","workspace"],"clips":6,"frames_per_clip":16,"checkpoint":"VideoMAE2 K710 distilled","test_labels_read":False},"train":{"rows":len(train["sample_ids"]),"available":int(train["available"].sum())},"test":{"rows":len(test["sample_ids"]),"available":int(test["available"].sum())}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
