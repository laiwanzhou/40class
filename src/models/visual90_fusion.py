"""Matched, trainable midfusion over frozen Visual90 regional features."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _safe_keys(values: torch.Tensor, mask: torch.Tensor):
    present = mask.any(-1)
    safe = mask.clone()
    safe[~present, 0] = True
    return values * mask[..., None], safe, present


class ResidualAttention(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0, batch_first=True)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, queries, keys, key_mask):
        keys, safe, present = _safe_keys(keys, key_mask)
        normalized = self.key_norm(keys)
        value = self.attention(self.query_norm(queries), normalized, normalized,
                               key_padding_mask=~safe, need_weights=False)[0]
        return 0.1 * torch.tanh(self.output(value)) * present[:, None, None]


class Visual90Fusion(nn.Module):
    def __init__(self, appearance: bool, seed: int = 20260715):
        super().__init__()
        self.use_appearance = appearance
        # Common initialization is independent of construction order and B-only RNG.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.video_projection = nn.Linear(768, 256)
            self.depth_residual = ResidualAttention()
            self.view_embedding = nn.Parameter(torch.randn(4, 256) * .02)
            self.space_embedding = nn.Parameter(torch.randn(4, 256) * .02)
            self.time_projection = nn.Linear(1, 256)
            self.roi_projection = nn.Linear(4, 256)
            self.local = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0,
                                                    batch_first=True, norm_first=True)
            self.pool_queries = nn.Parameter(torch.randn(4, 256) * .02)
            self.pool = nn.MultiheadAttention(256, 4, dropout=0, batch_first=True)
            self.trial_query = nn.Parameter(torch.randn(1, 1, 256) * .02)
            self.temporal = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0,
                                                       batch_first=True, norm_first=True)
            self.final_norm = nn.LayerNorm(256)
            self.classifier = nn.Linear(256, 40)
        if appearance:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed + 1)
                self.appearance_projection = nn.Linear(1024, 256)
                self.appearance_space = nn.Parameter(torch.randn(17, 256) * .02)
                self.appearance_residual = ResidualAttention()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        video = batch['video']
        b = video.shape[0]
        if video.shape[1:] != (2, 4, 4, 8, 4, 768):
            raise ValueError('video features must be [B,2,4,4,8,4,768]')
        for name in ('video', 'roi', 'times') + (('appearance',) if self.use_appearance else ()):
            if not torch.isfinite(batch[name]).all():
                raise ValueError(f'{name} must be finite')
        vm = batch['video_mask']
        if vm.shape != (b, 2, 4, 4) or vm.dtype != torch.bool:
            raise ValueError('video mask must be bool [B,2,4,4]')
        times = batch['times'].float()
        roi = batch['roi'].float()
        if times.shape != (b, 4, 8) or roi.shape != (b, 4, 4, 8, 4):
            raise ValueError('time/ROI layout mismatch')
        projected = self.video_projection(video.float())
        position = (self.view_embedding[None, None, :, None, None]
                    + self.space_embedding[None, None, None, None]
                    + self.time_projection(times[..., None])[:, :, None, :, None]
                    + self.roi_projection(roi)[:, :, :, :, None])
        ir = projected[:, 0] + position
        depth = projected[:, 1] + position
        im, dm = vm[:, 0], vm[:, 1]
        base = torch.where(im[..., None, None, None], ir, depth)
        valid = (im | dm)[..., None, None].expand(b, 4, 4, 8, 4).clone()
        delta = self.depth_residual(
            ir.reshape(b * 16, 32, 256), depth.reshape(b * 16, 32, 256),
            dm.reshape(b * 16, 1).expand(-1, 32),
        ).reshape(b, 4, 4, 8, 4, 256)
        base = (base + delta * im[..., None, None, None]) * valid[..., None]
        if self.training:
            retained = (torch.rand(valid.shape, device=valid.device) >= .1) & valid
            for row in range(b):
                if valid[row].any() and not retained[row].any():
                    retained[row].view(-1)[torch.nonzero(valid[row].flatten())[0, 0]] = True
            valid = retained
        base = base * valid[..., None]
        local = base.reshape(b * 4, 128, 256)
        local_mask = valid.reshape(b * 4, 128)
        appearance_delta = torch.zeros_like(local)
        if self.use_appearance:
            appearance = batch['appearance']
            am = batch['appearance_mask']
            if appearance.shape != (b, 4, 4, 4, 17, 1024) or am.shape != (b, 4, 4) or am.dtype != torch.bool:
                raise ValueError('appearance layout/mask mismatch')
            at = batch.get('appearance_times', times[..., [0, 2, 5, 7]]).float()
            ar = batch.get('appearance_roi', roi[:, :, :, [0, 2, 5, 7]]).float()
            keys = (self.appearance_projection(appearance.float())
                    + self.view_embedding[None, None, :, None, None]
                    + self.appearance_space[None, None, None, None]
                    + self.time_projection(at[..., None])[:, :, None, :, None]
                    + self.roi_projection(ar)[:, :, :, :, None])
            # An appearance view cannot resurrect an invalid IR clip-view.
            keys_mask = (am & im)[..., None, None].expand(b, 4, 4, 4, 17)
            appearance_delta = self.appearance_residual(
                local, keys.reshape(b * 4, 272, 256), keys_mask.reshape(b * 4, 272))
            local = local + appearance_delta
        local, safe, _ = _safe_keys(local, local_mask)
        local = self.local(local, src_key_padding_mask=~safe) * local_mask[..., None]
        group_values = local.reshape(b * 16, 32, 256)
        group_mask = local_mask.reshape(b * 16, 32)
        group_values, group_safe, group_present = _safe_keys(group_values, group_mask)
        queries = self.pool_queries[None, None].expand(b, 4, -1, -1).reshape(b * 16, 1, 256)
        pooled = self.pool(queries, group_values, group_values,
                           key_padding_mask=~group_safe, need_weights=False)[0]
        pooled = pooled[:, 0] * group_present[:, None]
        pooled = pooled.reshape(b, 16, 256)
        trial_mask = group_present.reshape(b, 16)
        tokens = torch.cat((self.trial_query.expand(b, -1, -1), pooled), dim=1)
        mask = torch.cat((torch.ones(b, 1, device=video.device, dtype=torch.bool), trial_mask), dim=1)
        feature = self.final_norm(self.temporal(tokens, src_key_padding_mask=~mask)[:, 0])
        supported = trial_mask.any(-1)
        feature = feature * supported[:, None]
        return {'logits': self.classifier(feature) * supported[:, None],
                'embedding': F.normalize(feature.float(), dim=-1), 'supported': supported,
                'depth_delta_mean': delta.float().abs().mean(),
                'appearance_delta_mean': appearance_delta.float().abs().mean()}
