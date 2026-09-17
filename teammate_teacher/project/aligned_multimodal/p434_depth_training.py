"""Original P12 depth-only training kernel for the P434 rebuild."""
from __future__ import annotations

import copy
import json
import random
import time
from collections import Counter
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class _SourceLabels(Dataset):
    def __init__(self,dataset,labels):self.dataset,self.labels=dataset,labels
    def __len__(self):return len(self.labels)
    def __getitem__(self,index):
        return {'depth':self.dataset[index]['depth'],'label':int(self.labels[index])}


def _strict_labels(labels: Sequence[int], n: int) -> np.ndarray:
    raw = np.asarray(labels)
    if (raw.ndim != 1 or raw.shape != (n,) or raw.dtype.kind not in "iu"
            or raw.dtype.kind == "b" or np.any((raw < 0) | (raw >= 40))):
        raise ValueError("source_labels must be integer class IDs in [0,39]")
    return raw.astype(np.int64, copy=True)


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _loader(dataset: Dataset, labels: Sequence[int], *, seed: int, train: bool) -> DataLoader:
    sampler = None; shuffle = False
    if train:
        counts = Counter(int(x) for x in labels)
        if not counts:
            raise ValueError("cannot train on an empty source dataset")
        weights = torch.tensor([1.0 / counts[int(x)] for x in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset, batch_size=32, shuffle=shuffle, sampler=sampler,
                      num_workers=0, pin_memory=False, drop_last=train)


def _make_datasets(train_dataset, prediction_dataset, labels, *, cache_dir, sample_ids,
                   train_indices, predict_indices):
    if (train_dataset is None)!=(prediction_dataset is None):raise ValueError('provide both datasets or neither')
    if train_dataset is None or prediction_dataset is None:
        if cache_dir is None or sample_ids is None or train_indices is None or predict_indices is None:
            raise ValueError("provide datasets or cache_dir/sample_ids/indices")
        from .p434_depth_data import P434DepthDataset
        train_dataset = P434DepthDataset(cache_dir, sample_ids, train_indices, labels=labels, augment=True)
        prediction_dataset = P434DepthDataset(cache_dir, sample_ids, predict_indices, labels=None, augment=False)
    if len(train_dataset) != len(labels):
        raise ValueError("source_labels must align with train_dataset")
    return _SourceLabels(train_dataset,labels), prediction_dataset


@torch.no_grad()
def _predict(model, dataset, device, deadline=None) -> np.ndarray:
    loader = _loader(dataset, [], seed=0, train=False)
    model.eval(); outputs: list[np.ndarray] = []
    for batch in loader:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("P434 depth prediction deadline exceeded")
        depth = batch["depth"].to(device, non_blocking=False)
        logits = model({"depth": depth})
        if isinstance(logits, dict):
            logits = logits["logits"]
        if logits.ndim!=2 or logits.shape[1]!=40 or not torch.isfinite(logits).all():raise RuntimeError('invalid depth prediction logits')
        outputs.append(logits.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 40), np.float32)


def train_trajectory(train_dataset: Dataset | None, prediction_dataset: Dataset | None,
                     source_labels: Sequence[int], *, epochs: int = 15,
                     seed: int = 20260720, device: str | torch.device = "cuda",
                     deadline: float | None = None, collect_each_epoch: bool = True,
                     cache_dir: str | None = None, sample_ids: Sequence[str] | None = None,
                     train_indices: Sequence[int] | None = None,
                     predict_indices: Sequence[int] | None = None):
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or not 1 <= int(epochs) <= 15:
        raise ValueError("epochs must be an integer in [1,15]")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or not 0<=seed<2**32:raise ValueError('invalid seed')
    if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('P434 deadline before initialization')
    _seed(int(seed))
    labels = _strict_labels(source_labels, len(source_labels))
    train_dataset, prediction_dataset = _make_datasets(
        train_dataset, prediction_dataset, labels, cache_dir=cache_dir,
        sample_ids=sample_ids, train_indices=train_indices, predict_indices=predict_indices)
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for P434 depth training but unavailable")
    from .p434_depth_model import build_model
    model = build_model().to(requested)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = requested.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    diagnostics: dict[str, Any] = {"losses": [], "successful_amp_steps": [], "skipped_steps": [],
        "learning_rates": [], "epochs_completed": 0, "device": str(requested),
        "requested_device": str(requested), "batch_size": 32, "drop_last": True, "num_workers": 0}
    trajectory: list[np.ndarray] = []; started = time.monotonic()
    train_loader = _loader(train_dataset, labels, seed=int(seed), train=True)
    print(json.dumps({'event':'depth_fit_start','source_rows':len(train_dataset),'prediction_rows':len(prediction_dataset),'epochs':int(epochs)}),flush=True)
    for epoch in range(int(epochs)):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("P434 depth training deadline exceeded")
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []; good = skipped = 0
        for batch in train_loader:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("P434 depth training deadline exceeded")
            depth = batch["depth"].to(requested, non_blocking=False)
            target = torch.as_tensor(batch["label"], dtype=torch.long, device=requested)
            with torch.autocast(device_type=requested.type, enabled=use_amp):
                logits = model({"depth": depth})
                if isinstance(logits, dict): logits = logits["logits"]
                loss = criterion(logits, target)
            if not bool(torch.isfinite(loss).item()):
                skipped += 1; optimizer.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not bool(torch.isfinite(grad_norm).item()):
                skipped += 1; optimizer.zero_grad(set_to_none=True); scaler.update(); continue
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            good += 1; losses.append(float(loss.detach().cpu()))
        total = good + skipped
        if not good or (total and good / total < 0.9):
            raise RuntimeError(f"P434 epoch {epoch + 1} had insufficient successful steps: {good}/{total}")
        scheduler.step(); diagnostics["epochs_completed"] += 1
        diagnostics["losses"].append(float(np.mean(losses))); diagnostics["successful_amp_steps"].append(good)
        diagnostics["skipped_steps"].append(skipped); diagnostics["learning_rates"].append(float(optimizer.param_groups[0]["lr"]))
        with torch.random.fork_rng(devices=[requested.index or 0] if requested.type=='cuda' else []):
            trajectory.append(_predict(model, prediction_dataset, requested, deadline))
        print(json.dumps({'event':'depth_epoch_complete','epoch':epoch+1,'successful_steps':good,'skipped_steps':skipped,'seconds':time.monotonic()-started}),flush=True)
    if not collect_each_epoch:
        trajectory = [trajectory[-1]]
    diagnostics["elapsed_seconds"] = time.monotonic() - started
    diagnostics["finite_training"] = bool(np.isfinite(diagnostics["losses"]).all())
    state = {key: copy.deepcopy(value.detach().cpu()) for key, value in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):raise RuntimeError('nonfinite depth state')
    if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('P434 deadline after serialization')
    return np.stack(trajectory, axis=0).astype(np.float32, copy=False), state, diagnostics


train_model = train_trajectory

__all__ = ["train_trajectory", "train_model"]
