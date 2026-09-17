"""Refit the fixed P90 MotionBERT front linear probe on all Train and infer Test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from p90_motionbert_teacher import LinearProbe, seed_everything
from p90_teacher_common import load_protocol


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN = REPO / "runs/p90_motionbert_teacher_v1/features_pretrain_front_t81.npz"
TEST = HERE / "runs/a18_test_features_v1/motionbert_pretrain_front_t81.npz"
OUTPUT = HERE / "runs/p184_motionbert_front_test_v1"


def main() -> None:
    protocol = load_protocol()
    with np.load(TRAIN, allow_pickle=False) as saved:
        train_ids = saved["sample_ids"].astype(str)
        train = saved["features"].astype(np.float32)
    with np.load(TEST, allow_pickle=False) as saved:
        test_ids = saved["sample_ids"].astype(str)
        test = saved["features"].astype(np.float32)
    if not np.array_equal(train_ids, protocol.sample_ids.astype(str)):
        raise RuntimeError("P184 Train MotionBERT order differs")
    if train.shape != (2914, 9216) or test.shape != (405, 9216):
        raise RuntimeError(f"P184 feature shapes differ: {train.shape}/{test.shape}")
    seed_everything(9001)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-4] = 1.0
    train = (train - mean) / std
    test = (test - mean) / std
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train), torch.from_numpy(protocol.labels.astype(np.int64))),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(9001),
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearProbe(train.shape[1], dropout=0.35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    for epoch in range(80):
        model.train(); loss_sum = 0.0
        for values, labels in loader:
            values=values.to(device); labels=labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(values),labels); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach())*len(labels)
        scheduler.step()
        if epoch in (0,19,39,59,79):
            print(json.dumps({"epoch":epoch+1,"train_loss":loss_sum/len(train)}),flush=True)
    model.eval(); chunks=[]
    with torch.inference_mode():
        for start in range(0,len(test),256):
            chunks.append(model(torch.from_numpy(test[start:start+256]).to(device)).float().cpu().numpy())
    logits=np.concatenate(chunks).astype(np.float64)
    probability=np.exp(logits-logits.max(axis=1,keepdims=True)); probability/=probability.sum(axis=1,keepdims=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    path=OUTPUT/"test_predictions.npz"
    np.savez_compressed(path,sample_ids=test_ids,logits=logits.astype(np.float32),probabilities=probability.astype(np.float32))
    report={"stage":"P184_MotionBERT_front_all2914_to_Test","status":"complete","train_rows":len(train),"test_rows":len(test),"epochs":80,"seed":9001,"mean_confidence":float(probability.max(axis=1).mean()),"test_labels_read":False,"output":str(path.resolve())}
    (OUTPUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
