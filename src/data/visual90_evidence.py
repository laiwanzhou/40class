"""Small, training-only evidence loader used by Visual90 qualification."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.data.visual90_dataset import prepare_roi_track
from src.roi.object_interaction_builder import ObjectInteractionROIBuilder


def build_view_boxes(person, keypoints, confidence, selection, width, height, roi_config):
    boxes = np.zeros((4,4,16,4),dtype=np.float32)
    masks = np.zeros((4,4),dtype=bool)
    diagnostics = []
    builder = ObjectInteractionROIBuilder(**roi_config)
    for clip,(start,stop) in enumerate(selection.bounds):
        if not selection.valid[clip]:
            continue
        selected = selection.indices[clip]-start
        p=np.asarray(person[start:stop],dtype=np.float32).copy()
        full=np.array([0,0,width,height],dtype=np.float32)
        boxes[clip,0]=full; masks[clip,0]=True
        detected=np.isfinite(p).all(1)&(p[:,2]>p[:,0])&(p[:,3]>p[:,1])
        if not detected.any():
            diagnostics.append({'clip':clip,'view':1,'reason':'no_valid_person'})
            continue
        p[~detected]=np.nan
        interaction=builder.build(p,keypoints[start:stop],confidence[start:stop],width,height)
        candidates=[p,interaction.boxes[:,1].copy(),interaction.boxes[:,2].copy()]
        candidates[1][~interaction.valid_mask[:,1]]=np.nan
        candidates[2][~interaction.valid_mask[:,2]]=np.nan
        for view,track in enumerate(candidates,1):
            valid=np.isfinite(track).all(1)
            track[valid,:2]=np.maximum(track[valid,:2],0)
            track[valid,2:]=np.minimum(track[valid,2:],[width,height])
            result=prepare_roi_track(track,width,height)
            masks[clip,view]=result.eligible
            if result.eligible: boxes[clip,view]=result.boxes[selected]
            diagnostics.append({'clip':clip,'view':view,'reason':result.reason,
                                'interpolated':int(result.interpolated.sum())})
    return boxes,masks,diagnostics


def load_clip(paths, boxes, grayscale):
    frames=[]
    mean=torch.tensor([.485,.456,.406])[:,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None]
    for path,box in zip(paths,boxes,strict=True):
        with Image.open(path) as image:
            image=image.convert('L' if grayscale else 'RGB').crop(tuple(map(float,box)))
            value=TF.to_tensor(TF.resize(image,[224,224],antialias=True))
            if grayscale: value=value.repeat(3,1,1)
            frames.append((value-mean)/std)
    return torch.stack(frames,dim=1)


class RawClipDataset(torch.utils.data.Dataset):
    def __init__(self, records): self.records=records
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        r=self.records[index]
        return {'clip':load_clip(r['paths'],r['boxes'],r['modality']==0),'index':index}
