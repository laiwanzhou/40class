"""Extract ROI-independent early/late Thermal scene features for 3036 Train union + Test."""
from __future__ import annotations
import csv,json,time
from pathlib import Path
import cv2,numpy as np,torch
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset,DataLoader
from transformers import VideoMAEImageProcessor
import p91_videomaev2_modality_teacher as p91
from p90_videomae_lora_teacher import VideoCollator
H=Path(__file__).resolve().parent;UNION=H/"data/six_modality_audit/train_union_manifest.csv";MAIN=H/"data/manifest.csv";TEST=H/"data/p46_test_union_manifest.csv";O=H/"runs/p294_thermal_scene_union_v1"
def readcsv(p):
 with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def train_rows():
 main=readcsv(MAIN);lookup={(r["class_name"],r["user_id"],r["trial_id"]):r["sample_id"] for r in main};out=[]
 for r in readcsv(UNION):
  key=(r["class_name"],r["user_id"],r["trial_id"]);out.append({"sample_id":lookup.get(key,"extra__c"+r["class_id"].zfill(2)+"__"+r["user_id"]+"__"+r["trial_id"]),"user_id":r["user_id"],"class_id":r["class_id"],"thermal_path":r["thermal_path"]})
 return out
def test_rows():return [{"sample_id":r["official_sample_id"],"user_id":"anonymous","class_id":"-1","thermal_path":r["thermal_path"]} for r in readcsv(TEST)]
def readim(p):
 b=cv2.imread(str(p),cv2.IMREAD_COLOR)
 if b is None:raise RuntimeError(p)
 return cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
class DS(Dataset):
 def __init__(self,rows):self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];files=sorted(Path(r["thermal_path"]).glob("*.jpg"));av=bool(files)
  if av:
   clips=[]
   for lo,hi in ((0.,.7),(.3,1.)):clips.append([readim(files[j]) for j in np.rint(np.linspace(lo*(len(files)-1),hi*(len(files)-1),16)).astype(int)])
  else:
   black=np.zeros((224,224,3),np.uint8);clips=[[black]*16 for _ in range(2)]
  return {"videos":clips,"clip_indices":[0,1],"label":int(r["class_id"]),"sample_id":r["sample_id"],"user":r["user_id"],"available":av}
def coll(processor):
 base=VideoCollator(processor)
 def fn(items):out=base(items);out["users"]=[x["user"] for x in items];out["available"]=[x["available"] for x in items];return out
 return fn
@torch.inference_mode()
def extract(rows,model,processor,name):
 partial=O/f"{name}_partial.npz";ids=[];users=[];labels=[];av=[];fs=[];acts=[]
 if partial.exists():
  z=np.load(partial);ids=z["sample_ids"].astype(str).tolist();users=z["users"].astype(str).tolist();labels=z["labels"].astype(int).tolist();av=z["available"].astype(bool).tolist();fs=[z["features"].copy()];acts=[z["action_logits"].copy()]
  if ids!=[r["sample_id"] for r in rows[:len(ids)]]:raise RuntimeError(name+" partial order")
 loader=DataLoader(DS(rows[len(ids):]),batch_size=12,shuffle=False,num_workers=0,collate_fn=coll(processor),pin_memory=True);start=time.time()
 for bi,b in enumerate(loader):
  pix=b["pixel_values"].permute(0,2,1,3,4).to("cuda",dtype=torch.float16);f=model.forward_features(pix);a=model.head(f);n=len(b["sample_ids"]);fs.append(f.reshape(n,2,768).half().cpu().numpy());acts.append(a.reshape(n,2,710).half().cpu().numpy());ids.extend(b["sample_ids"]);users.extend(b["users"]);labels.extend(b["labels"].numpy().tolist());av.extend(b["available"])
  if (bi+1)%20==0:
   fv=np.concatenate(fs);aa=np.concatenate(acts);np.savez(partial,sample_ids=np.asarray(ids),users=np.asarray(users),labels=np.asarray(labels),available=np.asarray(av),features=fv,action_logits=aa);fs=[fv];acts=[aa];print(json.dumps({"split":name,"rows":len(ids),"total":len(rows),"seconds":round(time.time()-start,1)}),flush=True)
 return {"sample_ids":np.asarray(ids),"users":np.asarray(users),"labels":np.asarray(labels,int),"available":np.asarray(av,bool),"features":np.concatenate(fs),"action_logits":np.concatenate(acts)}
def main():
 model,_=p91.build_model(torch.device("cuda"));snap=Path(snapshot_download(p91.PROCESSOR_REPO,local_files_only=True));processor=VideoMAEImageProcessor.from_pretrained(snap,local_files_only=True);O.mkdir(parents=True,exist_ok=True);tr=extract(train_rows(),model,processor,"train");te=extract(test_rows(),model,processor,"test");np.savez_compressed(O/"train_features.npz",**tr);np.savez_compressed(O/"test_features.npz",**te);report={"stage":"P294_thermal_scene_union","status":"complete","protocol":{"roi_required":False,"windows":[[0,.7],[.3,1]],"scene_only":True,"test_labels_read":False},"train":{"rows":len(tr["sample_ids"]),"available":int(tr["available"].sum()),"extra_rows":int(np.sum(np.char.startswith(tr["sample_ids"].astype(str),"extra__")))},"test":{"rows":len(te["sample_ids"]),"available":int(te["available"].sum())}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
