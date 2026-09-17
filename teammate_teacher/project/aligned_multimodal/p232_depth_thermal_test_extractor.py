"""Extract the exact P91 VideoMAEv2 K710 Depth/Thermal contract on Test."""
from __future__ import annotations
import csv,json,time
from pathlib import Path
import numpy as np,torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader,Dataset
from transformers import VideoMAEImageProcessor
import p91_videomaev2_modality_teacher as p91
from p90_videomae_lora_teacher import VideoCollator
from build_p46_videomae_cache import safe_relative
H=Path(__file__).resolve().parent;R=H.parent;MAN=H/"data/p46_test_union_manifest.csv";P29=H/"runs/p29_dir_multiscale_roi_test";O=H/"runs/p232_depth_thermal_test_features_v1"
def rows():
 with MAN.open("r",encoding="utf-8-sig",newline="") as f:src=list(csv.DictReader(f))
 return [{**r,"sample_id":r["official_sample_id"],"source_id":r["sample_id"],"depth_dir":r["depth_color_path"],"ir_dir":r["ir_path"],"user_id":"anonymous"} for r in src]
def usable(row,modality):
 roi=P29/"trial_roi_cache"/safe_relative(row["source_id"]).with_suffix(".npz")
 return roi.is_file() and (modality=="depth" or any(p91.thermal_dir(row).glob("*.jpg")))
class TestDataset(Dataset):
 def __init__(self,source,modality):self.source=source;self.modality=modality
 def __len__(self):return len(self.source)
 def __getitem__(self,index):
  row=self.source[index]
  if usable(row,self.modality):clips,_=p91.prepare_modality_trial(row,P29,self.modality)
  else:
   black=np.zeros((224,224,3),np.uint8);clips=[[black]*16 for _ in range(3)]
  return {"videos":clips,"clip_indices":[0,1,2],"label":0,"sample_id":row["sample_id"]}
@torch.inference_mode()
def extract(modality,model,processor,device,source):
 dataset=TestDataset(source,modality);loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0,collate_fn=VideoCollator(processor),pin_memory=True);fp=[];ap=[];ids=[];started=time.time()
 for bi,b in enumerate(loader):
  pix=b["pixel_values"].permute(0,2,1,3,4).to(device=device,dtype=torch.float16,non_blocking=True);f=model.forward_features(pix);a=model.head(f);n=len(b["sample_ids"]);fp.append(f.reshape(n,3,768).half().cpu().numpy());ap.append(a.reshape(n,3,710).half().cpu().numpy());ids.extend(b["sample_ids"])
  if (bi+1)%10==0:print(json.dumps({"modality":modality,"rows":len(ids),"total":len(source),"seconds":round(time.time()-started,1)}),flush=True)
 available=np.asarray([int(usable(r,modality)) for r in source],np.uint8);features=np.concatenate(fp);actions=np.concatenate(ap);features[available==0]=0;actions[available==0]=0;return {"sample_ids":np.asarray(ids),"features":features,"action_logits":actions,"modality_available":available,"modality":np.asarray(modality)}
def main():
 source=rows();p91.P29_RUN=P29.resolve();device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model,_=p91.build_model(device);snap=Path(snapshot_download(p91.PROCESSOR_REPO,local_files_only=True));processor=VideoMAEImageProcessor.from_pretrained(snap,local_files_only=True);O.mkdir(parents=True,exist_ok=True);report={"stage":"P232_P91_depth_thermal_Test_extraction","status":"complete","test_labels_read":False,"modalities":{}}
 for m in ("depth","thermal"):
  v=extract(m,model,processor,device,source);path=O/f"{m}_features.npz";np.savez_compressed(path,**v);report["modalities"][m]={"rows":len(v["sample_ids"]),"available":int(v["modality_available"].sum()),"path":str(path.resolve())}
 (O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
