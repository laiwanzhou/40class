import numpy as np
import torch
from torch import nn

from src.models.visual90_encoders import pool_video_tokens, pool_appearance_tokens
from src.data.visual90_evidence import build_view_boxes
from src.data.visual90_dataset import select_continuous_clips


def test_video_pool_preserves_time_and_spatial_order():
    values = torch.arange(8.).view(1,8,1,1,1).expand(1,8,14,14,3).reshape(1,1568,3)
    result = pool_video_tokens(values, nn.Identity())
    assert result.shape == (1,8,4,3)
    torch.testing.assert_close(result[0,:,0,0], torch.arange(8.))


def test_appearance_cls_stays_separate_from_patches():
    cls = torch.full((1,3), 9.)
    patches = torch.ones(1,256,3)
    result = pool_appearance_tokens(cls, patches)
    assert result.shape == (1,17,3)
    assert (result[:,0]==9).all() and (result[:,1:]==1).all()


def test_no_pose_creates_only_real_global_view():
    selection = select_continuous_clips(np.arange(8,dtype=float))
    boxes, masks, diagnostics = build_view_boxes(
        np.full((8,4),np.nan), np.full((8,17,2),np.nan), np.zeros((8,17)),
        selection, 640,480, {})
    assert masks[:,0].all() and not masks[:,1:].any()
    assert (boxes[:,0,:,2:] == [640,480]).all()


def test_unselected_segment_pose_does_not_change_selected_roi():
    times = np.arange(16,dtype=float)*10; times[2:]+=1000
    selection=select_continuous_clips(times)
    person=np.tile([10,10,100,200],(16,1)).astype(float)
    points=np.full((16,17,2),np.nan); conf=np.zeros((16,17))
    before=build_view_boxes(person,points,conf,selection,640,480,{})[0][0]
    person[2:4]=[200,100,500,400]
    after=build_view_boxes(person,points,conf,selection,640,480,{})[0][0]
    np.testing.assert_array_equal(before,after)
