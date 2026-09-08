"""Training primitives; no formal experiment launch is exposed here."""
from __future__ import annotations

import torch
from torch.nn import functional as F
import math
import random
from pathlib import Path
import numpy as np

TRAIN_USERS={1,2,3,5,8,9,16,18,19,20,21,22}


def paired_epoch_indices(labels,users,seed,epoch):
    labels=np.asarray(labels);users=np.asarray(users)
    if len(labels)!=len(users) or not len(labels):raise ValueError('empty or mismatched sampler population')
    if not set(users.tolist())<=TRAIN_USERS:raise ValueError('nontraining user in sampler')
    if set(labels.tolist())!=set(range(40)):raise ValueError('sampler needs all40 classes')
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch]));result=[]
    for _ in range(math.ceil(len(labels)/2)):
        label=int(rng.integers(40));owners=np.unique(users[labels==label])
        if len(owners)>1:
            a,b=rng.choice(owners,2,replace=False)
            result.extend([int(rng.choice(np.flatnonzero((labels==label)&(users==u)))) for u in (a,b)])
        else:
            candidates=np.flatnonzero(labels==label)
            result.extend(rng.choice(candidates,2,replace=len(candidates)<2).tolist())
    return np.asarray(result,dtype=np.int64)


def learning_rate_at(epoch):
    if not 1<=epoch<=30:raise ValueError('epoch outside fixed30 protocol')
    if epoch<=2:return .0003*epoch/2
    return .0003*.5*(1+math.cos(math.pi*(epoch-2)/28))


def save_training_state(path,model,optimizer,epoch,identity,history):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'identity':identity,
        'history':history,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
    temp=path.with_suffix('.partial');torch.save(payload,temp);temp.replace(path)


def restore_training_state(path,model,optimizer,identity,device):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['identity']!=identity:raise ValueError('checkpoint identity mismatch')
    model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
    # torch optimizer load casts tensor state to the corresponding parameter device.
    random.setstate(payload['python_rng']);np.random.set_state(payload['numpy_rng']);torch.set_rng_state(payload['torch_rng'])
    if payload['cuda_rng'] is not None:torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['epoch'],payload['history']


def decide_results(a,b):
    net=b['correct']-a['correct']
    def level(n):return .90 if n>=350 else .85 if n>=330 else .82 if n>=319 else None
    return {'target_reached':max(a['correct'],b['correct'])>=350,'absolute_progress':{'A':level(a['correct']),'B':level(b['correct'])},
        'net':net,'appearance_increment':'positive' if net>=8 and b['worst_user_accuracy']>=a['worst_user_accuracy'] else 'regression' if net<0 else 'mixed_or_unproven'}


def cross_user_supcon(z, labels, users, sample_ids):
    if not (len(z) == len(labels) == len(users) == len(sample_ids)):
        raise ValueError('contrastive batch lengths differ')
    unique = {}
    positions = []
    for i, sid in enumerate(sample_ids):
        if sid in unique:
            j = unique[sid]
            if labels[i] != labels[j] or users[i] != users[j]:
                raise ValueError('duplicate ID has inconsistent ownership')
        else:
            unique[sid] = i
            positions.append(i)
    idx = torch.tensor(positions, dtype=torch.long, device=z.device)
    norm = F.normalize(z[idx].float(), dim=-1)
    y, u = labels[idx], users[idx]
    same_class = y[:, None] == y[None, :]
    positive = same_class & (u[:, None] != u[None, :])
    allowed = positive | ~same_class
    anchors = positive.any(-1)
    if not anchors.any():
        return z.sum() * 0
    sim = norm @ norm.T / .10
    denominator = torch.logsumexp(sim[anchors].masked_fill(~allowed[anchors], -torch.inf), dim=-1)
    mean_positive = (sim[anchors] * positive[anchors]).sum(-1) / positive[anchors].sum(-1)
    return (denominator - mean_positive).mean()
