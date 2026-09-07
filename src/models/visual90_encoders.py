"""Strict frozen feature extraction, without a formal cache/training entrypoint."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import torch
from torch import nn
from torch.nn import functional as F

from src.models.ir_depth_videomaev2_teacher import build_official_videomaev2_vit_b

DINO_REVISION='7764ea0f912e53c92e82eb78a2a1631e92725fc8'


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def pool_video_tokens(tokens, norm):
    b,n,d=tokens.shape
    if n!=1568: raise ValueError('expected eight 14x14 video grids')
    grid=tokens.reshape(b*8,14,14,d).permute(0,3,1,2)
    pooled=F.adaptive_avg_pool2d(grid,2).flatten(2).transpose(1,2).reshape(b,8,4,d)
    return norm(pooled)


def pool_appearance_tokens(cls,patches):
    b,n,d=patches.shape
    if n!=256: raise ValueError('expected sixteen by sixteen DINO grid')
    pooled=F.adaptive_avg_pool2d(patches.reshape(b,16,16,d).permute(0,3,1,2),4)
    return torch.cat((cls[:,None],pooled.flatten(2).transpose(1,2)),dim=1)


class VideoEncoder(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.backbone,self.provenance=build_official_videomaev2_vit_b(checkpoint_path=Path(checkpoint),with_cp=False)
        self.backbone.requires_grad_(False); self.eval()

    @torch.inference_mode()
    def forward(self,x):
        m=self.backbone
        tokens=m.patch_embed(x)
        tokens=m.pos_drop(tokens+m.pos_embed.to(device=tokens.device,dtype=tokens.dtype))
        for block in m.blocks: tokens=block(tokens)
        return pool_video_tokens(tokens,m.fc_norm)


class AppearanceEncoder(nn.Module):
    def __init__(self,repository,checkpoint,expected_sha256):
        super().__init__()
        repository=Path(repository); checkpoint=Path(checkpoint)
        revision=subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(repository),'status','--porcelain'],text=True).strip()
        if revision!=DINO_REVISION or dirty: raise ValueError('DINO source revision/cleanliness mismatch')
        if file_hash(checkpoint)!=expected_sha256: raise ValueError('DINO checkpoint hash mismatch')
        sys.path.insert(0,str(repository))
        from dinov2.hub.backbones import dinov2_vitl14
        self.backbone=dinov2_vitl14(pretrained=False)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.backbone.load_state_dict(state,strict=True)
        self.backbone.requires_grad_(False); self.eval()
        self.provenance={'revision':revision,'checkpoint_sha256':expected_sha256,
                         'checkpoint_bytes':checkpoint.stat().st_size,'strict_load':True,
                         'parameter_count':sum(p.numel() for p in self.backbone.parameters())}

    @torch.inference_mode()
    def forward(self,x):
        out=self.backbone.forward_features(x)
        return pool_appearance_tokens(out['x_norm_clstoken'],out['x_norm_patchtokens'])
