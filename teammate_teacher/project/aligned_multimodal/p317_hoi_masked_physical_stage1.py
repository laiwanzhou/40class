"""Stage-1 paired fold-0 screen for entity-masked HOI physical learning."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np,torch
import torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import class_weights,soft_cross_entropy,seed_everything
from p266_modality_specific_physical_oof import PATHS,LENS
H=Path(__file__).resolve().parent;O=H/"runs/p317_hoi_masked_physical_stage1_v1";SEED=31701;ENTITY=np.asarray([1,2,4,5,7,8,10,11,13,14,16,17]);DYN=((2,5),(8,11))
class Model(nn.Module):
 def __init__(self):
  super().__init__();h=192;self.proj=nn.ModuleList([nn.Sequential(nn.LayerNorm(768),nn.Linear(768,h),nn.GELU()) for _ in LENS]);self.cls=nn.Parameter(torch.randn(1,1,h)*.02);self.mask=nn.Parameter(torch.randn(1,1,h)*.02);self.pos=nn.Parameter(torch.randn(1,19,h)*.02);self.mod=nn.Parameter(torch.randn(1,4,h)*.02);self.view=nn.Parameter(torch.randn(1,3,h)*.02);self.time=nn.Parameter(torch.randn(1,3,h)*.02);layer=nn.TransformerEncoderLayer(h,6,h*3,.2,"gelu",batch_first=True,norm_first=True);self.enc=nn.TransformerEncoder(layer,2);self.head=nn.Sequential(nn.LayerNorm(h),nn.Dropout(.2),nn.Linear(h,40));self.dyn=nn.Sequential(nn.LayerNorm(h*2),nn.Linear(h*2,h),nn.GELU(),nn.Dropout(.2),nn.Linear(h,40));self.recon=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h))
 def tokens(self,x):
  parts=[];off=0
  for i,(n,p) in enumerate(zip(LENS,self.proj)):
   q=p(x[:,off:off+n]);view=torch.arange(n,device=x.device)%3;q=q+self.mod[:,i:i+1]+self.view[:,view];time=torch.zeros(n,dtype=torch.long,device=x.device)
   if n==6:time[3:]=1
   else:time[:]=2
   parts.append(q+self.time[:,time]);off+=n
  return torch.cat(parts,1)
 def forward(self,x,mask=None):
  target=self.tokens(x);tok=target
  if mask is not None:tok=torch.where(mask[...,None],self.mask.expand(len(x),18,-1),tok)
  enc=self.enc(torch.cat((self.cls.expand(len(x),-1,-1),tok),1)+self.pos);main=self.head(enc[:,0]);diff=torch.cat((target[:,DYN[0][1]]-target[:,DYN[0][0]],target[:,DYN[1][1]]-target[:,DYN[1][0]]),1);dyn=self.dyn(diff);return main,dyn,self.recon(enc[:,1:]),target
def entity_mask(n,device):
 m=torch.zeros((n,18),dtype=torch.bool,device=device);score=torch.rand((n,len(ENTITY)),device=device);take=score.topk(len(ENTITY)//2,largest=False).indices;idx=torch.from_numpy(ENTITY).to(device)[None].expand(n,-1);m.scatter_(1,idx.gather(1,take),True);return m
def train(x,y,tr,va,seed,device,hoi):
 seed_everything(seed);m=Model().to(device);opt=torch.optim.AdamW(m.parameters(),3e-4,weight_decay=.05);steps=35*math.ceil(len(tr)/128);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(steps,1),eta_min=1.5e-5);loader=DataLoader(TensorDataset(torch.from_numpy(tr),torch.from_numpy(y[tr])),128,shuffle=True,generator=torch.Generator().manual_seed(seed));w=torch.from_numpy(class_weights(y[tr])).to(device);sc=torch.amp.GradScaler("cuda",enabled=device.type=="cuda");history=[]
 for ep in range(35):
  m.train();total=ce_sum=dyn_sum=rec_sum=0.;rows=0
  for ix,lab in loader:
   ni=ix.numpy();v=torch.from_numpy(np.asarray(x[ni],np.float32)).to(device);target=F.one_hot(lab.to(device),40).float();lam=float(np.random.beta(.2,.2));order=torch.randperm(len(v),device=device);v=lam*v+(1-lam)*v[order];target=lam*target+(1-lam)*target[order];mask=entity_mask(len(v),device) if hoi else None;opt.zero_grad(set_to_none=True)
   with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.float16):
    main,dyn,recon,original=m(v,mask);ce=soft_cross_entropy(main,target,w)
    if hoi:
     dl=soft_cross_entropy(dyn,target,w);cos=1-F.cosine_similarity(recon[mask],original.detach()[mask],dim=1).mean();loss=ce+.3*dl+.2*cos
    else:dl=torch.zeros((),device=device);cos=torch.zeros((),device=device);loss=ce
   sc.scale(loss).backward();sc.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),2.);sc.step(opt);sc.update();sch.step();n=len(v);total+=float(loss)*n;ce_sum+=float(ce)*n;dyn_sum+=float(dl)*n;rec_sum+=float(cos)*n;rows+=n
  row={"epoch":ep+1,"loss":total/rows,"classification":ce_sum/rows,"dynamics":dyn_sum/rows,"reconstruction":rec_sum/rows};history.append(row)
  if ep in (0,34) or (ep+1)%10==0:print(json.dumps({"variant":"hoi" if hoi else "control",**row}),flush=True)
 m.eval();out=[]
 with torch.inference_mode():
  for st in range(0,len(va),256):
   v=torch.from_numpy(np.asarray(x[va[st:st+256]],np.float32)).to(device)
   with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.float16):main,dyn,_,_=m(v,None);log=main+.3*dyn if hoi else main
   out.append(log.float().cpu().numpy())
 return np.concatenate(out),history
def main():
 print("P317 Stage 1 tests whether 50% entity masking, token reconstruction and early/late dynamics improve the same-seed fold-0 physical baseline.",flush=True);p=load_protocol();sources=[np.load(q) for q in PATHS];x=np.concatenate([q["features"].astype(np.float16).reshape(len(p.labels),-1,768) for q in sources],1);tr=p.train_indices(0);va=p.val_indices(0);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");control,ch=train(x,p.labels,tr,va,SEED,device,False);hoi,hh=train(x,p.labels,tr,va,SEED,device,True);cc=int(np.sum(control.argmax(1)==p.labels[va]));hc=int(np.sum(hoi.argmax(1)==p.labels[va]));O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"predictions.npz",sample_ids=p.sample_ids[va],labels=p.labels[va],control_logits=control.astype(np.float32),hoi_logits=hoi.astype(np.float32));(O/"history.json").write_text(json.dumps({"control":ch,"hoi":hh},indent=2)+"\n",encoding="utf-8");report={"stage":"P317_HOI_masked_physical_stage1","status":"complete","protocol":{"fold":0,"seed":SEED,"paired_same_architecture_seed_epochs":True,"entity_mask_ratio":.5,"reconstruction_weight":.2,"dynamics_weight":.3,"test_rows_loaded":0,"test_labels_read":False,"user_id_used_as_feature":False},"control":{"correct":cc,"rows":len(va),"accuracy":cc/len(va)},"hoi":{"correct":hc,"rows":len(va),"accuracy":hc/len(va),"net_vs_control":hc-cc},"gate":{"required_net":8,"pass":hc-cc>=8}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: paired fold-0 Stage-1 screen. Control and HOI variants share seed, optimizer, architecture, frozen features and 35 epochs. HOI adds 50% entity masking, masked-token cosine reconstruction, and early/late IR dynamics classification. Expansion gate: at least +8 correct.\nResult: "+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
