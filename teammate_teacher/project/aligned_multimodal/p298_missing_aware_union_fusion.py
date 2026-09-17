"""Paired strict OOF missing-aware fusion using balanced extra Thermal-only rows."""
from __future__ import annotations
import collections,json,math
from pathlib import Path
import numpy as np,torch
import torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import class_weights,soft_cross_entropy,seed_everything
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p298_missing_aware_union_fusion_v1";P294=H/"runs/p294_thermal_scene_union_v1";IR=(R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz",R/"runs/p91_videomaev2_depth_fold0_v1/complete_features.npz");IRT=(R/"runs/p90_videomaev2_distilled_test_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_test_v1/complete_features.npz",H/"runs/p232_depth_thermal_test_features_v1/depth_features.npz");SEEDS=(29801,29817,29833);LENS=(6,6,3,2)
class Model(nn.Module):
 def __init__(self):
  super().__init__();h=192;self.proj=nn.ModuleList([nn.Sequential(nn.LayerNorm(768),nn.Linear(768,h),nn.GELU()) for _ in LENS]);self.cls=nn.Parameter(torch.randn(1,1,h)*.02);self.pos=nn.Parameter(torch.randn(1,5,h)*.02);layer=nn.TransformerEncoderLayer(h,6,h*2,.25,"gelu",batch_first=True,norm_first=True);self.enc=nn.TransformerEncoder(layer,2);self.head=nn.Sequential(nn.LayerNorm(h),nn.Dropout(.25),nn.Linear(h,40))
 def forward(self,x,mask):
  tok=[];off=0
  for n,p in zip(LENS,self.proj):tok.append(p(x[:,off:off+n]).mean(1));off+=n
  tok=torch.stack(tok,1);tok=tok.masked_fill(~mask[:,:,None],0);cls=self.cls.expand(len(x),-1,-1);pad=torch.cat((torch.zeros((len(x),1),dtype=torch.bool,device=x.device),~mask),1);return self.head(self.enc(torch.cat((cls,tok),1)+self.pos,src_key_padding_mask=pad)[:,0])
def balanced(ids,u,y,av):
 out=[];uc=collections.Counter();uu=collections.Counter()
 for i in np.flatnonzero(np.char.startswith(ids,"extra__")&av):
  k=(u[i],int(y[i]))
  if uc[k]<3 and uu[u[i]]<15:out.append(i);uc[k]+=1;uu[u[i]]+=1
 return np.asarray(out,int)
def train(x,mask,y,tr,va,seed,device):
 seed_everything(seed);m=Model().to(device);opt=torch.optim.AdamW(m.parameters(),3e-4,weight_decay=.05);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30*math.ceil(len(tr)/128),eta_min=1.5e-5);loader=DataLoader(TensorDataset(torch.from_numpy(tr),torch.from_numpy(y[tr])),128,shuffle=True,generator=torch.Generator().manual_seed(seed));w=torch.from_numpy(class_weights(y[tr])).to(device);sc=torch.amp.GradScaler("cuda",enabled=True)
 for ep in range(30):
  m.train()
  for ix,lab in loader:
   ni=ix.numpy();v=torch.from_numpy(np.asarray(x[ni],np.float32)).to(device);mk=torch.from_numpy(mask[ni]).to(device);target=F.one_hot(lab.to(device),40).float();opt.zero_grad(set_to_none=True)
   with torch.amp.autocast("cuda",dtype=torch.float16):log=m(v,mk);loss=soft_cross_entropy(log,target,w)
   sc.scale(loss).backward();sc.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),2);sc.step(opt);sc.update();sch.step()
 m.eval();out=[]
 with torch.inference_mode():
  for st in range(0,len(va),256):
   v=torch.from_numpy(np.asarray(x[va[st:st+256]],np.float32)).to(device);mk=torch.from_numpy(mask[va[st:st+256]]).to(device)
   with torch.amp.autocast("cuda",dtype=torch.float16):out.append(m(v,mk).float().cpu().numpy())
 return np.concatenate(out)
def fill(source,values,ids,out,slot,mask,mi):
 d={q:i for i,q in enumerate(ids)}
 for j,q in enumerate(source.astype(str)):
  if q in d:out[d[q],slot]=values[j];mask[d[q],mi]=True
def main():
 p=load_protocol();th=np.load(P294/"train_features.npz");ids=th["sample_ids"].astype(str);u=th["users"].astype(str);y=th["labels"].astype(int);n=len(ids);x=np.zeros((n,sum(LENS),768),np.float16);mask=np.zeros((n,4),bool);mainpos={q:i for i,q in enumerate(ids)};main=np.asarray([mainpos[q] for q in p.sample_ids]);off=0
 for mi,(path,leng) in enumerate(zip(IR,LENS[:3])):
  z=np.load(path);v=z["features"].astype(np.float16).reshape(len(z["features"]),leng,768);fill(z["sample_ids"],v,ids,x,slice(off,off+leng),mask,mi);off+=leng
 x[:,off:off+2]=th["features"];mask[:,3]=th["available"].astype(bool);extra=balanced(ids,u,y,mask[:,3]);device=torch.device("cuda");oof={k:np.zeros((len(p.labels),40),np.float32) for k in ("control","augmented")};folds={k:[] for k in oof}
 for f in range(3):
  held=p.val_indices(f);hc=main[held];base=main[p.fold_id!=f];held_users=set(p.users[held].tolist());add=extra[~np.isin(u[extra],list(held_users))]
  for name,tr in (("control",base),("augmented",np.concatenate((base,add)))):
   members=[train(x,mask,y,tr,hc,s+f*1000,device) for s in SEEDS];log=np.mean(members,0);pr=np.exp(log-log.max(1,keepdims=True));pr/=pr.sum(1,keepdims=True);oof[name][held]=pr;folds[name].append({"fold":f,"train_rows":len(tr),"extra_rows":0 if name=="control" else len(add),"correct":int(np.sum(pr.argmax(1)==p.labels[held])),"rows":len(held)})
 te=np.load(P294/"test_features.npz");tids=te["sample_ids"].astype(str);tx=np.zeros((len(tids),sum(LENS),768),np.float16);tm=np.zeros((len(tids),4),bool);off=0
 for mi,(path,leng) in enumerate(zip(IRT,LENS[:3])):
  z=np.load(path);v=z["features"].astype(np.float16).reshape(len(z["features"]),leng,768);fill(z["sample_ids"],v,tids,tx,slice(off,off+leng),tm,mi);off+=leng
  if mi==2 and "modality_available" in z.files:tm[:,mi]&=z["modality_available"].astype(bool)
 tx[:,off:]=te["features"];tm[:,3]=te["available"];values=np.concatenate((x,tx));masks=np.concatenate((mask,tm));labels=np.concatenate((y,np.zeros(len(tx),int)));tr=np.concatenate((main,extra));va=np.arange(len(x),len(values));members=[train(values,masks,labels,tr,va,s,device) for s in SEEDS];log=np.mean(members,0);tp=np.exp(log-log.max(1,keepdims=True));tp/=tp.sum(1,keepdims=True);report={"stage":"P298_missing_aware_union_fusion","status":"complete","protocol":{"modalities":["ir_vmae","ir_iv2","depth","thermal_scene"],"balanced_extra":len(extra),"missing_mask":True,"paired_control":True,"strict_subject_folds":True,"test_labels_read":False},"variants":{k:{"correct":int(np.sum(v.argmax(1)==p.labels)),"accuracy":float(np.mean(v.argmax(1)==p.labels)),"folds":folds[k]} for k,v in oof.items()},"test":{"rows":len(tids)}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"predictions.npz",sample_ids=p.sample_ids,labels=p.labels,control_probability=oof["control"],augmented_probability=oof["augmented"],test_sample_ids=tids,test_probability=tp.astype(np.float32));(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
