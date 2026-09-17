"""All-Train three-seed P238 Transformer inference on exact Test tokens."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import train_fold
H=Path(__file__).resolve().parent;R=H.parent;O=H/"runs/p239_physical_token_transformer_test_v1";TR=(R/"runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz",R/"runs/p91_videomaev2_depth_fold0_v1/complete_features.npz",R/"runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz");IR=(R/"runs/p90_videomaev2_distilled_test_v1/complete_features.npz",R/"runs/p90_internvideo2_l_k400_test_v1/complete_features.npz");D=H/"runs/p232_depth_thermal_test_features_v1/depth_features.npz";T=H/"runs/p232_depth_thermal_test_features_v1/thermal_features.npz";IDS=H/"runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz";SEEDS=(23801,23817,23833)
def cfg():return SimpleNamespace(hidden_dim=192,heads=6,layers=2,dropout=.2,view_dropout=.15,epochs=35,batch_size=128,learning_rate=3e-4,weight_decay=.05,mixup_alpha=.2,repeat_consistency_weight=0.,repeat_embedding_weight=0.,repeat_same_label_only=False,class_triplet_weight=0.,triplet_margin=.2,teacher_weight=0.,domain_adversarial_weight=0.)
def align_tokens(z,ids):
 v=z["features"].astype(np.float16).reshape(len(z["features"]),-1,768);out=np.zeros((len(ids),v.shape[1],768),np.float16);pos={q:i for i,q in enumerate(ids)};rows=np.asarray([pos[q] for q in z["sample_ids"].astype(str)]);out[rows]=v;mask=np.zeros(len(ids),bool);mask[rows]=True;return out,mask
def main():
 p=load_protocol();sources=[np.load(x) for x in TR];x=np.concatenate([z["features"].astype(np.float16).reshape(len(p.labels),-1,768) for z in sources],1);ids=np.load(IDS)["sample_ids"].astype(str);a,ma=align_tokens(np.load(IR[0]),ids);b,mb=align_tokens(np.load(IR[1]),ids);d,md=align_tokens(np.load(D),ids);t,mt=align_tokens(np.load(T),ids);tx=np.concatenate((a,b,d,t),1);available=ma&mb&md;values=np.concatenate((x,tx));labels=np.concatenate((p.labels,np.zeros(len(ids),np.int64)));domains=np.concatenate((p.fold_id,np.zeros(len(ids),np.int64)));tr=np.arange(len(p.labels));va=np.arange(len(p.labels),len(labels));device=torch.device("cuda" if torch.cuda.is_available() else "cpu");members=[]
 for seed in SEEDS:members.append(train_fold(values,labels,tr,va,domains,seed,cfg(),device,None,None))
 logits=np.mean(np.stack(members),0);prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"test_predictions.npz",sample_ids=ids,logits=logits.astype(np.float32),probability=prob.astype(np.float32),available=available,thermal_available=mt);report={"stage":"P239_P238_allTrain_to_Test","status":"complete","protocol":{"tokens":18,"seeds":list(SEEDS),"epochs":35,"features_frozen":True,"test_labels_read":False},"test":{"rows":len(ids),"available":int(available.sum()),"thermal_available":int(mt.sum()),"mean_confidence":float(prob[available].max(1).mean())}};(O/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
