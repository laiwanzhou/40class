"""Training primitives; no formal experiment launch is exposed here."""
from __future__ import annotations

import torch
from torch.nn import functional as F


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
