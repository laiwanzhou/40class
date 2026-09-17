"""Strict OOF physical Transformer with one projection per frozen backbone/modality."""
from __future__ import annotations
import json,math,random
from pathlib import Path
import numpy as np,torch
import torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import class_weights,soft_cross_entropy,seed_everything
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p266_modality_specific_physical_oof_v1";PATHS=(R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz",R/"runs/p91_videomaev2_depth_fold0_v1/complete_features.npz",R/"runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz");SEEDS=(26601,26617,26633);LENS=(6,6,3,3)
class Model(nn.Module):
 def __init__(self):
  super().__init__();h=192;self.proj=nn.ModuleList([nn.Sequential(nn.LayerNorm(768),nn.Linear(768,h),nn.GELU()) for _ in LENS]);self.cls=nn.Parameter(torch.randn(1,1,h)*.02);self.pos=nn.Parameter(torch.randn(1,19,h)*.02);self.mod=nn.Parameter(torch.randn(1,4,h)*.02);layer=nn.TransformerEncoderLayer(h,6,h*3,.2,"gelu",batch_first=True,norm_first=True);self.enc=nn.TransformerEncoder(layer,2);self.head=nn.Sequential(nn.LayerNorm(h),nn.Dropout(.2),nn.Linear(h,40))
 def forward(self,x):
  parts=[];off=0
  for i,(n,p) in enumerate(zip(LENS,self.proj)):
   parts.append(p(x[:,off:off+n])+self.mod[:,i:i+1]);off+=n
  tok=torch.cat(parts,1);cls=self.cls.expand(len(x),-1,-1);return self.head(self.enc(torch.cat((cls,tok),1)+self.pos)[:,0])
def train(x,y,tr,va,seed,device):
 seed_everything(seed);m=Model().to(device);opt=torch.optim.AdamW(m.parameters(),3e-4,weight_decay=.05);steps=35*math.ceil(len(tr)/128);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(steps,1),eta_min=1.5e-5);loader=DataLoader(TensorDataset(torch.from_numpy(tr),torch.from_numpy(y[tr])),128,shuffle=True,generator=torch.Generator().manual_seed(seed));w=torch.from_numpy(class_weights(y[tr])).to(device);sc=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
 for ep in range(35):
  m.train();loss_sum=0;rows=0
  for ix,lab in loader:
   ni=ix.numpy();v=torch.from_numpy(np.asarray(x[ni],np.float32)).to(device);target=F.one_hot(lab.to(device),40).float();lam=float(np.random.beta(.2,.2));order=torch.randperm(len(v),device=device);v=lam*v+(1-lam)*v[order];target=lam*target+(1-lam)*target[order];opt.zero_grad(set_to_none=True)
   with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.float16):log=m(v);loss=soft_cross_entropy(log,target,w)
   sc.scale(loss).backward();sc.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),2.);sc.step(opt);sc.update();sch.step();loss_sum+=float(loss)*len(v);rows+=len(v)
  if ep in (0,34) or (ep+1)%10==0:print(json.dumps({"seed":seed,"epoch":ep+1,"loss":loss_sum/rows}),flush=True)
 m.eval();out=[]
 with torch.inference_mode():
  for st in range(0,len(va),256):
   v=torch.from_numpy(np.asarray(x[va[st:st+256]],np.float32)).to(device)
   with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.float16):out.append(m(v).float().cpu().numpy())
 return np.concatenate(out)
def main():
 p=load_protocol();z=[np.load(q) for q in PATHS];x=np.concatenate([q["features"].astype(np.float16).reshape(len(p.labels),-1,768) for q in z],1);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");logits=np.zeros((len(p.labels),40),float);folds=[]
 for k in range(3):
  tr=p.train_indices(k);va=p.val_indices(k);members=[train(x,p.labels,tr,va,s+k*1000,device) for s in SEEDS];logits[va]=np.mean(members,0);folds.append({"fold":k,"rows":len(va),"correct":int(np.sum(logits[va].argmax(1)==p.labels[va]))})
 prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);report={"stage":"P266_modality_specific_physical_OOF","status":"complete","protocol":{"separate_projections":["ir_vmae","ir_iv2","depth_vmae","thermal_vmae"],"token_lengths":list(LENS),"hidden":192,"layers":2,"heads":6,"seeds":list(SEEDS),"strict_subject_folds":True,"test_rows_loaded":0},"metrics":{"correct":int(np.sum(prob.argmax(1)==p.labels)),"rows":len(p.labels),"accuracy":float(np.mean(prob.argmax(1)==p.labels)),"folds":folds}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",sample_ids=p.sample_ids,labels=p.labels,users=p.users,fold_id=p.fold_id,logits=logits.astype(np.float32),probability=prob.astype(np.float32));(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
