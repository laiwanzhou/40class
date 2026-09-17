"""Local SigLIP2 text-tower reconstruction and zero-shot action semantics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from safetensors import safe_open
from p90_teacher_common import load_protocol
from p117_transductive_multicandidate_router import load_candidate_splits
H=Path(__file__).resolve().parent;O=H/"runs/p339_siglip2_zeroshot_semantic_v1";ROOT=Path.home()/".cache/modelscope/hub/models/timm/ViT-B-16-SigLIP2-256";W=ROOT/"open_clip_model.safetensors";CACHE=H/"runs/p335_siglip2_workspace_state_cache_v1/features.npy";P310=H/"runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz";S=("H1_selection","H2_confirmation","H3_independent_fold0")
class Block(nn.Module):
 def __init__(self):
  super().__init__();self.ln_1=nn.LayerNorm(768,eps=1e-6);self.attn=nn.MultiheadAttention(768,12,batch_first=True);self.ln_2=nn.LayerNorm(768,eps=1e-6);self.mlp=nn.Module();self.mlp.c_fc=nn.Linear(768,3072);self.mlp.gelu=nn.GELU(approximate="tanh");self.mlp.c_proj=nn.Linear(3072,768)
 def forward(self,x):q=self.ln_1(x);x=x+self.attn(q,q,q,need_weights=False)[0];return x+self.mlp.c_proj(self.mlp.gelu(self.mlp.c_fc(self.ln_2(x))))
class Tower(nn.Module):
 def __init__(self):
  super().__init__();self.token_embedding=nn.Embedding(256000,768);self.positional_embedding=nn.Parameter(torch.empty(64,768));self.transformer=nn.Module();self.transformer.resblocks=nn.ModuleList([Block() for _ in range(12)]);self.ln_final=nn.LayerNorm(768,eps=1e-6);self.text_projection=nn.Linear(768,768,bias=True)
 def forward(self,ids,mask):
  x=self.token_embedding(ids)+self.positional_embedding
  for b in self.transformer.resblocks:x=b(x)
  x=self.ln_final(x);last=mask.sum(1).clamp_min(1)-1;x=x[torch.arange(len(x),device=x.device),last];return torch.nn.functional.normalize(self.text_projection(x),dim=1)
def text_model(device):
 m=Tower()
 with safe_open(W,framework="pt",device="cpu") as f:s={k.removeprefix("text."):f.get_tensor(k) for k in f.keys() if k.startswith("text.")}
 m.load_state_dict(s,strict=True);return m.eval().to(device=device,dtype=torch.float16)
def main():
 print("P339 tests zero-shot SigLIP2 action semantics from a strict local text-tower reconstruction.",flush=True);p=load_protocol();names=[]
 for k in range(40):
  raw=str(p.manifest.loc[p.manifest.class_id==k,"class_name"].iloc[0]);names.append(raw.split("_",1)[-1].replace("_"," ").lower())
 prompts=[]
 for x in names:prompts.extend((f"an image of the human action: {x}",f"hands performing the action: {x}",f"a video frame showing a person doing: {x}"))
 tok=PreTrainedTokenizerFast.from_pretrained(ROOT,local_files_only=True);enc=tok(prompts,padding="max_length",truncation=True,max_length=64,return_tensors="pt");device=torch.device("cuda" if torch.cuda.is_available() else "cpu");m=text_model(device)
 with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):t=m(enc.input_ids.to(device),enc.attention_mask.to(device)).reshape(40,3,768).mean(1);t=torch.nn.functional.normalize(t,dim=1).cpu().numpy()
 v=np.asarray(np.load(CACHE,mmap_mode="r"),np.float32);v/=np.clip(np.linalg.norm(v,axis=-1,keepdims=True),1e-6,None);frame=np.einsum("nwtf,kf->nwtk",v,t);scores={"mean8":frame.mean((1,2)),"max8":frame.max((1,2)),"late4":frame[:,1].mean(1)};splits=load_candidate_splits();basez=np.load(P310);base=basez["prediction"];labels=basez["labels"];pos={q:i for i,q in enumerate(p.sample_ids)};saved={"sample_ids":p.sample_ids,"labels":p.labels,"text_embeddings":t.astype(np.float32)};report={"stage":"P339_SigLIP2_zero_shot_semantic","status":"complete","protocol":{"model":"ViT-B-16-SigLIP2-256","prompt_templates":3,"text_tower_strict_state_load":True,"training_labels_used":False,"test_rows_loaded":0,"test_labels_read":False},"variants":{}}
 for name,logits in scores.items():
  pred=logits.argmax(1);saved[name+"_logits"]=logits.astype(np.float32);op=np.concatenate([pred[[pos[q] for q in splits[n].split.sample_ids.astype(str)]] for n in S]);q=op!=base;r=int(np.sum(q&(base!=labels)&(op==labels)));h=int(np.sum(q&(base==labels)&(op!=labels)));report["variants"][name]={"correct":int(np.sum(pred==p.labels)),"accuracy":float(np.mean(pred==p.labels)),"vs_p310":{"changed":int(q.sum()),"rescue":r,"harm":h,"oracle_correct":int(np.sum((base==labels)|(op==labels)))}}
 O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"predictions.npz",**saved);(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(O/"notes.txt").write_text("Run 1: strict local SigLIP2 text tower and three-template zero-shot action semantics.\n"+json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
