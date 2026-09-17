"""All-Train/Test counterpart of P336 all_frames_a3000."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from p90_teacher_common import load_protocol,softmax
from p90_videomaev2_distilled_teacher import class_sample_weights
from train_p46_videomae_head import l2_normalize,make_model
H=Path(__file__).resolve().parent;O=H/"runs/p349_siglip2_ridge_test_v1";TR=H/"runs/p335_siglip2_workspace_state_cache_v1/features.npy";TE=H/"runs/p348_siglip2_workspace_test_cache_v1/features.npy";IDS=H/"runs/p348_siglip2_workspace_test_cache_v1/sample_ids.npy"
def feat(v):return np.concatenate([l2_normalize(v[:,w,t].astype(np.float32)) for w in range(2) for t in range(4)],1)
def main():
 print("P349 fits the frozen P336 all_frames alpha=3000 Ridge on all Train and infers Test.",flush=True);p=load_protocol();x=feat(np.load(TR,mmap_mode="r"));tx=feat(np.load(TE,mmap_mode="r"));m=make_model(3000.);m.fit(x,p.labels,ridge__sample_weight=class_sample_weights(p.labels,.75));logits=m.decision_function(tx);prob=softmax(logits);ids=np.load(IDS).astype(str);O.mkdir(parents=True,exist_ok=True);np.savez_compressed(O/"test_predictions.npz",sample_ids=ids,logits=logits.astype(np.float32),probability=prob.astype(np.float32));report={"stage":"P349_P336_all_frames_Ridge_Test","status":"complete","train_rows":len(x),"test_rows":len(tx),"alpha":3000.,"test_labels_read":False};(O/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
