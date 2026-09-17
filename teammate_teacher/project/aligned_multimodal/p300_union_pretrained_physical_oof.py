"""Pretrain P266 Thermal projection on balanced union extras, then main-only physical OOF."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np,torch
import torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import class_weights,soft_cross_entropy,seed_everything
from p266_modality_specific_physical_oof import Model,PATHS as PHYSICAL,SEEDS
from p298_missing_aware_union_fusion import balanced
H=Path(__file__).resolve().parent;O=H/"runs/p300_union_pretrained_physical_oof_v1";THERM=H/"runs/p294_thermal_scene_union_v1/train_features.npz";CONTROL=H/"runs/p266_modality_specific_physical_oof_v1/oof_predictions.npz"
class Pre(nn.Module):
 def __init__(self):super().__init__();self.proj=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,192),nn.GELU());self.head=nn.Sequential(nn.LayerNorm(192),nn.Dropout(.25),nn.Linear(192,40))
 def forward(self,x):return self.head(self.proj(x).mean(1))
def optimize(model,x,y,tr,epochs,seed,device,pre=False):
 opt=torch.optim.AdamW(model.parameters(),3e-4,weight_decay=.05);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs*math.ceil(len(tr)/128),eta_min=1.5e-5);loader=DataLoader(TensorDataset(torch.from_numpy(tr),torch.from_numpy(y[tr])),128,shuffle=True,generator=torch.Generator().manual_seed(seed));w=torch.from_numpy(class_weights(y[tr])).to(device);sc=torch.amp.GradScaler("cuda",enabled=True)
 for _ in range(epochs):
  model.train()
  for ix,lab in loader:
   ni=ix.numpy();v=torch.from_numpy(np.asarray(x[ni],np.float32)).to(device);target=F.one_hot(lab.to(device),40).float();opt.zero_grad(set_to_none=True)
   with torch.amp.autocast("cuda",dtype=torch.float16):log=model(v) if pre else model(v);loss=soft_cross_entropy(log,target,w)
   sc.scale(loss).backward();sc.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),2);sc.step(opt);sc.update();sch.step()
def run(physical,thermal,y,preidx,tr,va,seed,device):
 seed_everything(seed);pre=Pre().to(device);optimize(pre,thermal,y,preidx,15,seed,device,True);m=Model().to(device);m.proj[3].load_state_dict(pre.proj.state_dict());optimize(m,physical,y,tr,35,seed+100,device);m.eval();out=[]
 with torch.inference_mode():
  for st in range(0,len(va),256):
   v=torch.from_numpy(np.asarray(physical[va[st:st+256]],np.float32)).to(device)
   with torch.amp.autocast("cuda",dtype=torch.float16):out.append(m(v).float().cpu().numpy())
 return np.concatenate(out)
def main():
 p=load_protocol();sources=[np.load(q) for q in PHYSICAL];mainx=np.concatenate([q["features"].astype(np.float16).reshape(len(p.labels),-1,768) for q in sources],1);th=np.load(THERM);ids=th["sample_ids"].astype(str);u=th["users"].astype(str);y=th["labels"].astype(int);av=th["available"].astype(bool);thermal=th["features"].astype(np.float16);pos={q:i for i,q in enumerate(ids)};main=np.asarray([pos[q] for q in p.sample_ids]);extra=balanced(ids,u,y,av);physical=np.zeros((len(ids),18,768),np.float16);physical[main]=mainx;device=torch.device("cuda");oof=np.zeros((len(p.labels),40),np.float32);folds=[]
 for f in range(3):
  held=p.val_indices(f);hc=main[held];trmain=main[p.fold_id!=f];held_users=set(p.users[held].tolist());add=extra[~np.isin(u[extra],list(held_users))];pre=np.concatenate((trmain[av[trmain]],add));members=[run(physical,thermal,y,pre,trmain,hc,s+f*1000,device) for s in SEEDS];log=np.mean(members,0);pr=np.exp(log-log.max(1,keepdims=True));pr/=pr.sum(1,keepdims=True);oof[held]=pr;folds.append({"fold":f,"pretrain_rows":len(pre),"extra_rows":len(add),"physical_train_rows":len(trmain),"correct":int(np.sum(pr.argmax(1)==p.labels[held])),"rows":len(held)})
 control=np.load(CONTROL)["probability"];report={"stage":"P300_union_pretrained_physical_OOF","status":"complete","protocol":{"balanced_extra":len(extra),"thermal_projection_pretrain_epochs":15,"physical_train_main_only":True,"physical_epochs":35,"paired_control":"P266 modality-specific physical","strict_subject_folds":True},"control":{"correct":int(np.sum(control.argmax(1)==p.labels))},"pretrained":{"correct":int(np.sum(oof.argmax(1)==p.labels)),"accuracy":float(np.mean(oof.argmax(1)==p.labels)),"folds":folds}};O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"oof_predictions.npz",sample_ids=p.sample_ids,labels=p.labels,probability=oof);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
