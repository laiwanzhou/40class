"""Pixel inference is independent of labels and teacher files."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .pixel_cache import validate_pixels


class NoVotePixelDataset(Dataset):
    def __init__(self,arrays,index,*,rows=None,labels=None,teacher=None,augmentation=None):
        validate_pixels(arrays,len(index.sample_ids))
        self.arrays=arrays;self.index=index;self.rows=rows or index
        lookup={sid:i for i,sid in enumerate(index.sample_ids)}
        if len(lookup)!=len(index.sample_ids) or not set(self.rows.sample_ids)<=set(lookup):
            raise ValueError('pixel sample IDs missing/duplicated')
        self.positions=[lookup[sid] for sid in self.rows.sample_ids]
        if tuple(index.user_ids[i] for i in self.positions)!=self.rows.user_ids:
            raise ValueError('pixel user/ID mismatch')
        self.labels=labels;self.teacher=teacher;self.augmentation=augmentation
        if labels is not None and set(labels)!=set(self.rows.sample_ids):
            raise ValueError('pixel supervised label IDs mismatch')
        if teacher is not None:
            ti=teacher.prediction.index;table={sid:i for i,sid in enumerate(ti.sample_ids)}
            if not set(self.rows.sample_ids)<=set(table) or ti.class_ids!=self.rows.class_ids or teacher.features is None:
                raise ValueError('pixel teacher IDs/features mismatch')
            self.teacher_positions=[table[sid] for sid in self.rows.sample_ids]
            if tuple(ti.user_ids[i] for i in self.teacher_positions)!=self.rows.user_ids:
                raise ValueError('pixel teacher user/ID mismatch')
            if teacher.features.shape!=(len(ti.sample_ids),2,3,1024):
                raise ValueError('pixel teacher six-clip features mismatch')

    def __len__(self):return len(self.positions)

    def __getitem__(self,item):
        pos=self.positions[item];a=self.arrays
        images=np.asarray(a['images'][pos]).copy()
        valid=np.asarray(a['view_valid'][pos]).copy();quality=np.asarray(a['view_quality'][pos]).copy()
        frame=np.asarray(a['source_frame_indices'][pos]).copy()
        times=np.asarray(a['source_time_seconds'][pos]).copy()
        count=int(frame.max(initial=0))+1
        temporal=np.arange(16,dtype=np.int64)
        if self.augmentation is not None:
            images,temporal=self.augmentation(images)
            valid=valid[:,temporal];quality=quality[:,temporal]
            frame=frame[:,temporal];times=times[:,temporal]
        result=dict(images=torch.from_numpy(np.ascontiguousarray(images)),
            view_valid=torch.from_numpy(valid),view_quality=torch.from_numpy(quality),
            temporal_source_index=torch.from_numpy(temporal),
            source_frame_index=torch.from_numpy(frame),source_time_seconds=torch.from_numpy(times),
            global_time_position=torch.from_numpy(frame.astype(np.float32)/max(count-1,1)),
            source_frame_count=torch.tensor(count),exact_source_time=torch.tensor(bool(np.isfinite(times).all())),
            sample_id=self.rows.sample_ids[item],user_id=self.rows.user_ids[item])
        if self.labels is not None:result['label']=torch.tensor(self.labels[result['sample_id']],dtype=torch.long)
        if self.teacher is not None:
            t=self.teacher_positions[item]
            result['teacher_logits']=torch.from_numpy(np.asarray(self.teacher.prediction.logits[t],dtype=np.float32).copy())
            result['teacher_features']=torch.from_numpy(np.asarray(self.teacher.features[t],dtype=np.float32).copy())
            result['teacher_valid']=torch.tensor(bool(self.teacher.prediction.valid[t]))
        return result
