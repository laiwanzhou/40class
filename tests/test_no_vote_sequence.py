import numpy as np
import torch


class NativeToy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.head=torch.nn.Linear(512,40)
    def encode_backbone_sequence(self,images):
        value=images.float().mean((-1,-2)).permute(0,1,3,2).unsqueeze(-1)
        return value.expand(-1,-1,-1,-1,512).contiguous()
    def forward_from_backbone_sequence(self,sequence,valid,quality,global_time_position=None):
        return {'logits':self.head(sequence.mean((1,2,3)))}


def test_native_sequence_and_frozen_anchor_are_real_time_features():
    from src.experiments.visual_student import sequence_and_anchor
    model=NativeToy();images=torch.arange(16,dtype=torch.uint8)[None,None,:,None,None,None].expand(1,2,16,3,8,8)
    batch=dict(images=images,view_valid=torch.ones(1,2,16,3,dtype=torch.bool),
        view_quality=torch.ones(1,2,16,3),global_time_position=torch.zeros(1,2,16))
    sequence,anchor=sequence_and_anchor(model,batch)
    assert sequence.shape==(1,2,3,16,512) and anchor.shape==(1,40)
    assert not np.array_equal(sequence[:,:,:,0],sequence[:,:,:,-1])
    saved=anchor.copy()
    with torch.no_grad():model.head.bias.add_(1)
    assert np.array_equal(saved,anchor)
    assert not np.array_equal(anchor,sequence_and_anchor(model,batch)[1])
