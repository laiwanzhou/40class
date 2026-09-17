"""Two-stage use of balanced extras: Thermal pretrain, then main-only fusion."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np,torch
import torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import class_weights,soft_cross_entropy,seed_everything
from p298_missing_aware_union_fusion import Model,balanced,LENS,IR,P294,SEEDS,fill
H=Path(__file__).resolve().parent;O=H/"runs/p299_union_thermal_pretrain_fusion_oof_v1";CONTROL=H/"runs/p298_missing_aware_union_fusion_v1/predictions.npz"
class Thermal(nn.Module):
 def __init__(self):super().__init__();self.proj=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,192),nn.GELU());self.head=nn.Sequential(nn.LayerNorm(192),nn.Dropout(.25),nn.Linear(192,40))
 def forward(self,x):return self.head(self.proj(x).mean(1))
def optimize(model,x,mask,y,tr,epochs,seed,device,thermal=False):
 opt=torch.optim.AdamW(model.parameters(),3e-4,weight_decay=.05);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs*math.ceil(len(tr)/128),eta_min=1.5e-5);loader=DataLoader(TensorDataset(torch.from_numpy(tr),torch.from_numpy(y[tr])),128,shuffle=True,generator=torch.Generator().manual_seed(seed));w=torch.from_numpy(class_weights(y[tr])).to(device);sc=torch.amp.GradScaler("cuda",enabled=True);model.train()
 for _ in range(epochs):
  for ix,lab in loader:
   ni=ix.numpy();target=F.one_hot(lab.to(device),40).float();opt.zero_grad(set_to_none=True)
   with torch.amp.autocast("cuda",dtype=torch.float16):
    if thermal:log=model(torch.from_numpy(np.asarray(x[ni,-2:],np.float32)).to(device))
    else:log=model(torch.from_numpy(np.asarray(x[ni],np.float32)).to(device),torch.from_numpy(mask[ni]).to(device))
    loss=soft_cross_entropy(log,target,w)
   sc.scale(loss).backward();sc.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),2);sc.step(opt);sc.update();sch.step()
def run(x,mask,y,pretrain,fusion,va,seed,device):
 seed_everything(seed);pre=Thermal().to(device);optimize(pre,x,mask,y,pretrain,15,seed,device,True);m=Model().to(device);m.proj[3].load_state_dict(pre.proj.state_dict());optimize(m,x,mask,y,fusion,30,seed+100,device,False);m.eval();out=[]
 with torch.inference_mode():
  for st in range(0,len(va),256):
   with torch.amp.autocast("cuda",dtype=torch.float16):out.append(m(torch.from_numpy(np.asarray(x[va[st:st+256]],np.float32)).to(device),torch.from_numpy(mask[va[st:st+256]]).to(device)).float().cpu().numpy())
 return np.concatenate(out)
def main():
 p=load_protocol();th=np.load(P294/"train_features.npz");ids=th["sample_ids"].astype(str);u=th["users"].astype(str);y=th["labels"].astype(int);x=np.zeros((len(ids),sum(LENS),768),np.float16);mask=np.zeros((len(ids),4),bool);pos={q:i for i,q in enumerate(ids)};main=np.asarray([pos[q] for q in p.sample_ids]);off=0
 for mi,(path,n) in enumerate(zip(IR,LENS[:3])):
  z=np.load(path);v=z["features"].astype(np.float16).reshape(len(z["features"]),n,768);fill(z["sample_ids"],v,ids,x,slice(off,off+n),mask,mi);off+=n
 x[:,off:]=th["features"];mask[:,3]=th["available"];extra=balanced(ids,u,y,mask[:,3]);device=torch.device("cuda");oof=np.zeros((len(p.labels),40),np.float32);folds=[]
 for f in range(3):
  held=p.val_indices(f);hc=main[held];fusion=main[p.fold_id!=f];held_users=set(p.users[held].tolist());add=extra[~np.isin(u[extra],list(held_users))];pre=np.concatenate((fusion[mask[fusion,3]],add));members=[run(x,mask,y,pre,fusion,hc,s+f*1000,device) for s in SEEDS];log=np.mean(members,0);pr=np.exp(log-log.max(1,keepdims=True));pr/=pr.sum(1,keepdims=True);oof[held]=pr;folds.append({"fold":f,"thermal_pretrain_rows":len(pre),"extra_rows":len(add),"fusion_rows":len(fusion),"correct":int(np.sum(pr.argmax(1)==p.labels[held])),"rows":len(held)})
 control=np.load(CONTROL)["control_probability"];report={"stage":"P299_union_Thermal_pretrain_then_fusion_OOF","status":"complete","protocol":{"balanced_extra":len(extra),"thermal_pretrain_epochs":15,"fusion_epochs":30,"fusion_train_main_only":True,"paired_control":"P298 same architecture/base seeds","strict_subject_folds":True},"control":{"correct":int(np.sum(control.argmax(1)==p.labels))},"two_stage":{"correct":int(np.sum(oof.argmax(1)==p.labels)),"accuracy":float(np.mean(oof.argmax(1)==p.labels)),"folds":folds}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",sample_ids=p.sample_ids,labels=p.labels,probability=oof);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
