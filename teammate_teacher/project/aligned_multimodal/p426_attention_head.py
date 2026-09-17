"""P426: fixed-capacity attention head over frozen 24x1024 VJEPA tokens.

This adapter deliberately contains no feature extraction, protocol loading, or
Test paths.  ``train_fn`` is injectable so contract tests need not run a model.
"""
from __future__ import annotations

import math
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SEEDS = (14201, 14217, 14233)
FIT_RECEIPTS: list[dict] = []
LAST_MEMBER_LOGITS: np.ndarray | None = None
LAST_TRAIN_DIAGNOSTIC: dict | None = None


def reset_fit_receipts() -> None:
    """Clear receipts between independent callers/tests."""
    FIT_RECEIPTS.clear()
    global LAST_MEMBER_LOGITS
    LAST_MEMBER_LOGITS = None


def last_member_logits():
    if LAST_MEMBER_LOGITS is None: raise RuntimeError("no completed attention fit")
    return LAST_MEMBER_LOGITS.copy()


def train_fixed_head(features, labels, train_indices, held_indices, domain_labels, seed, args, device,
                     repeat_pairs=None, teacher_probability=None):
    """P142 baseline loss/architecture, with AMP-aware optimizer scheduling."""
    from p142_vjepa_token_transformer_oof import TokenHead,seed_everything,class_weights,soft_cross_entropy
    if teacher_probability is not None or (repeat_pairs is not None and len(repeat_pairs)) or np.any(domain_labels):
        raise ValueError("P426 has no teacher/repeat/domain objective")
    seed_everything(seed)
    model=TokenHead(input_dim=1024,hidden_dim=args.hidden_dim,heads=args.heads,layers=args.layers,
                    dropout=args.dropout,view_dropout=args.view_dropout,num_tokens=24).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    total_steps=args.epochs*math.ceil(len(train_indices)/args.batch_size)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(total_steps,1),eta_min=args.learning_rate*.05)
    loader=DataLoader(TensorDataset(torch.from_numpy(np.asarray(train_indices,dtype=np.int64)),
                                    torch.from_numpy(np.asarray(labels)[train_indices].astype(np.int64))),
        batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0,drop_last=False)
    weights=torch.from_numpy(class_weights(np.asarray(labels)[train_indices])).to(device)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    successful=0;skipped=0
    for epoch in range(args.epochs):
        model.train();loss_sum=0.0;rows=0
        for indices,batch_labels in loader:
            values=torch.from_numpy(np.asarray(features[indices.numpy()],dtype=np.float32)).to(device)
            target=F.one_hot(batch_labels.to(device),num_classes=40).float()
            if args.mixup_alpha>0 and len(values)>1:
                lam=float(np.random.beta(args.mixup_alpha,args.mixup_alpha));order=torch.randperm(len(values),device=device)
                values=lam*values+(1-lam)*values[order];target=lam*target+(1-lam)*target[order]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.float16):
                loss=soft_cross_entropy(model(values),target,weights)
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite attention training loss")
            old_scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=old_scale:
                scheduler.step();successful+=1
            else:skipped+=1
            loss_sum+=float(loss.detach())*len(values);rows+=len(values)
        if epoch in (0,args.epochs-1) or (epoch+1)%10==0:
            print({"seed":seed,"epoch":epoch+1,"epochs":args.epochs,"train_loss":loss_sum/max(rows,1),
                   "optimizer_steps":successful,"amp_skips":skipped},flush=True)
    global LAST_TRAIN_DIAGNOSTIC
    LAST_TRAIN_DIAGNOSTIC={"planned_steps":total_steps,"optimizer_steps":successful,"amp_skips":skipped,
                           "final_lr":optimizer.param_groups[0]["lr"]}
    if successful<1 or (total_steps>=100 and successful<.9*total_steps):
        raise FloatingPointError("insufficient successful attention optimizer updates")
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(held_indices),args.batch_size*2):
            x=torch.from_numpy(np.asarray(features[held_indices[start:start+args.batch_size*2]],dtype=np.float32)).to(device)
            with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.float16):out.append(model(x).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def _args() -> Namespace:
    # Exactly P142's fixed recipe; optional objectives are explicitly disabled.
    return Namespace(
        hidden_dim=192, heads=6, layers=2, dropout=.20, view_dropout=.15,
        mixup_alpha=.20, learning_rate=3e-4, weight_decay=.05, epochs=35,
        repeat_consistency_weight=0.0, repeat_embedding_weight=0.0,
        repeat_same_label_only=False, class_triplet_weight=0.0,
        triplet_margin=.20, teacher_weight=0.0, teacher_confidence=.95,
        domain_adversarial_weight=0.0, batch_size=128,
    )


def _indices(value: np.ndarray, n: int, name: str) -> np.ndarray:
    x = np.asarray(value)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    x = x.astype(np.int64, copy=False)
    if len(np.unique(x)) != len(x) or np.any(x < 0) or np.any(x >= n):
        raise ValueError(f"{name} contains duplicate or out-of-range rows")
    return x


def fit_attention(features, labels, train_idx, held_idx, subjects, device="cuda",
                  train_fn: Callable | None = None):
    """Fit three fixed P142 heads and return mean-logit softmax probabilities.

    The return shape follows the family fitter contract: ``(None, None, p)``.
    ``train_fn`` must have the legacy ``train_fold`` signature and return held
    logits of shape ``(len(held_idx), 40)``.  Held labels are replaced by -1
    before it is called.
    """
    x = np.asarray(features)
    if x.ndim != 3 or x.shape[1:] != (24, 1024):
        raise ValueError(f"features must have shape [n,24,1024], got {x.shape}")
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("features must be finite numeric values")
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("labels must be a vector aligned to features")
    tr = _indices(train_idx, len(x), "train_idx")
    va = _indices(held_idx, len(x), "held_idx")
    if not tr.size or not va.size or np.intersect1d(tr, va).size:
        raise ValueError("train/held rows must be nonempty and disjoint")
    # Validate only labels that may reach optimization; non-training labels can
    # be unknown, nonsensical sentinels without affecting the fitted predictor.
    try: train_y=np.asarray(y[tr],dtype=np.float64)
    except (TypeError,ValueError) as exc: raise ValueError("training labels must be numeric class IDs") from exc
    if not np.isfinite(train_y).all() or np.any(train_y!=np.floor(train_y)) or np.any((train_y<0)|(train_y>=40)):
        raise ValueError("training labels must be integer class IDs in[0,39]")
    train_y=train_y.astype(np.int64)
    s = np.asarray(subjects)
    if s.ndim != 1 or len(s) != len(x):
        raise ValueError("subjects must align to features")
    if set(map(str, s[tr])) & set(map(str, s[va])):
        raise ValueError("train/held subjects overlap")
    try:
        dev = torch.device(device)
    except Exception as exc:
        raise ValueError(f"invalid device: {device}") from exc
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P426 requires available CUDA; pass device='cpu' explicitly for tests")
    fn = train_fixed_head if train_fn is None else train_fn
    # The legacy fitter receives labels only for the training partition.  Make
    # that guarantee explicit for every other row, including unassigned rows.
    clean_y = np.full(len(y), -1, dtype=np.int64)
    clean_y[tr] = train_y
    domains = np.zeros(len(x), dtype=np.int64)
    args = _args()
    logits = []
    for seed in SEEDS:
        global LAST_TRAIN_DIAGNOSTIC
        LAST_TRAIN_DIAGNOSTIC = None
        started = time.monotonic()
        out = fn(x, clean_y, tr, va, domains, seed, args, dev,
                  repeat_pairs=None, teacher_probability=None)
        out = np.asarray(out, dtype=np.float32)
        if out.shape != (len(va), 40) or not np.isfinite(out).all():
            raise ValueError("train_fn returned nonfinite or mis-shaped held logits")
        logits.append(out.astype(np.float64))
        FIT_RECEIPTS.append({"seed": seed, "train_indices": tr.tolist(),
                             "held_indices": va.tolist(),
                             "train_subjects": sorted(set(map(str, s[tr]))),
                             "held_subjects": sorted(set(map(str, s[va]))),
                             "train_class_counts": np.bincount(train_y, minlength=40).tolist(),
                             "recipe": {"epochs": 35, "batch_size": 128,
                                        "hidden_dim": 192, "heads": 6, "layers": 2,
                                        "dropout": .20, "view_dropout": .15,
                                        "mixup_alpha": .20, "learning_rate": 3e-4,
                                        "weight_decay": .05},
                             "seconds": time.monotonic() - started})
        FIT_RECEIPTS[-1]["training_diagnostics"] = dict(LAST_TRAIN_DIAGNOSTIC) if LAST_TRAIN_DIAGNOSTIC is not None else {"injected_test_fitter":True}
    mean = np.mean(logits, axis=0)
    global LAST_MEMBER_LOGITS
    LAST_MEMBER_LOGITS = np.stack(logits).astype(np.float32)
    mean -= mean.max(axis=1, keepdims=True)
    probability = np.exp(mean)
    probability /= probability.sum(axis=1, keepdims=True)
    if probability.shape != (len(va), 40) or not np.isfinite(probability).all():
        raise ValueError("nonfinite final probability")
    return None, None, probability.astype(np.float32)
