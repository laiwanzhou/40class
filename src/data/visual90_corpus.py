"""Versioned corpus evidence and immutable per-trial feature cache contracts."""
from __future__ import annotations
from collections import OrderedDict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.data.visual90_dataset import select_continuous_clips
from src.data.visual90_evidence import build_view_boxes,load_clip
from src.data.pose_roi_dataset import paired_frame_paths,paired_frame_key,depth_frame_key
from src.models.visual90_encoders import file_hash

ROOT=Path(__file__).resolve().parents[2]


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');temp.replace(path)


def assert_identity(document,identity):
    if document.get('identity')!=identity: raise ValueError('cache/checkpoint identity mismatch')


def verify_files(files,root):
    for filename,expected in files.items():
        path=Path(root)/filename
        if not path.is_file() or file_hash(path)!=expected: raise ValueError(f'file hash mismatch: {path}')


def recipe_identity(config):
    files=['src/data/visual90_corpus.py','src/data/visual90_dataset.py','src/data/visual90_evidence.py',
        'src/models/visual90_encoders.py','src/models/visual90_fusion.py','src/training/visual90_training.py',
        'scripts/cache_visual90_features.py','scripts/run_visual90_experiment.py',
        'src/models/ir_depth_videomaev2_teacher.py','src/roi/object_interaction_builder.py',
        'src/data/pose_roi_dataset.py','configs/experiments/visual90_encoder_lock.json',
        'configs/experiments/ir_depth_videomaev2_vit_b_p0.yaml','metadata/manifest.csv',
        'metadata/splits/train12_val2_user6_user7_development.json',
        'reports/visual90_appearance_temporal_input_preflight_v3.json',
        'reports/visual90_appearance_temporal_geometry_corpus.json',
        'reports/visual90_appearance_temporal_smoke_measurements.json']
    hashes={name:file_hash(ROOT/name) for name in files}
    payload={'config':config,'file_sha256':hashes,'torch_version':torch.__version__}
    identity=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return identity,payload


def prepare_trial(row,pose,roi_config,verify_pixels=True):
    base={'sample_id':row['sample_id'],'user':int(row['user_id'].removeprefix('user')),
          'label':row['class_id'],'partition':row['partition'],'disposition':row['training_disposition']}
    if row['training_disposition']=='excluded_missing_timestamps' or row['ir_frames']==0:
        return {**base,'supported':False,'source_files':{}}
    if row['pairing_status']!='verified_keys': raise ValueError('unsupported unpaired visual source')
    dp,ip=paired_frame_paths(Path(row['depth_color_path']),Path(row['ir_path']))
    source_files={}
    for paths in (ip,dp):
        for path in paths:
            with Image.open(path) as image:
                if image.size!=(640,480): raise ValueError(f'unreviewed acquisition dimensions: {path}')
                if verify_pixels: image.verify()
            if verify_pixels: source_files[str(path)]=file_hash(path)
    sid=row['sample_id']
    for path in dp:
        if (sid,depth_frame_key(path)) not in pose.metadata_lookup: raise ValueError('pose metadata missing')
        pose.validate_frame(sid,path,640,480)
    person,keypoints,confidence=pose.trial_arrays(sid,[depth_frame_key(p) for p in dp])
    moments=[datetime.strptime(paired_frame_key(p,'IR')[0],'%Y-%m-%d_%H-%M-%S.%f') for p in ip]
    times=np.array([(t-moments[0]).total_seconds()*1000 for t in moments])
    selection=select_continuous_clips(times)
    boxes,mask,diagnostics=build_view_boxes(person,keypoints,confidence,selection,640,480,roi_config)
    return {**base,'supported':bool(mask.any()),'source_files':source_files,
        'paths':[[str(p) for p in ip],[str(p) for p in dp]],'mask':mask.tolist(),'boxes':boxes.tolist(),
        'indices':selection.indices.tolist(),'diagnostics':diagnostics,
        'frame_times':[[float(times[j]) if j>=0 else 0. for j in group] for group in selection.indices]}


def metadata_arrays(meta):
    raw=np.array(meta['boxes'],dtype=np.float32)/np.array([640,480,640,480],dtype=np.float32)
    coords=np.stack(((raw[...,0]+raw[...,2])/2,(raw[...,1]+raw[...,3])/2,
                     raw[...,2]-raw[...,0],raw[...,3]-raw[...,1]),axis=-1)
    t=np.array(meta['frame_times'],dtype=np.float32);t=t/max(float(t.max()),1.)
    mask=np.array(meta['mask'],dtype=bool)
    return {'roi':coords.reshape(4,4,8,2,4).mean(3),'times':t.reshape(4,8,2).mean(2),
        'appearance_roi':coords[:,:, [0,5,10,15]],'appearance_times':t[:,[0,5,10,15]],
        'video_mask':np.stack((mask,mask)),'appearance_mask':mask}


def empty_features():
    return {'video':torch.zeros(2,4,4,8,4,768,dtype=torch.float16),
        'appearance':torch.zeros(4,4,4,17,1024,dtype=torch.float16),
        'video_mask':torch.zeros(2,4,4,dtype=torch.bool),'appearance_mask':torch.zeros(4,4,dtype=torch.bool),
        'roi':torch.zeros(4,4,8,4),'times':torch.zeros(4,8),
        'appearance_roi':torch.zeros(4,4,4,4),'appearance_times':torch.zeros(4,4)}


class CorpusClipDataset(torch.utils.data.Dataset):
    def __init__(self,root,entries,stage):
        self.root=Path(root);self.stage=stage;self.records=[];self._metadata=OrderedDict()
        for index in entries:
            meta=json.loads((self.root/f'{index:05d}.meta.json').read_text(encoding='utf-8'))
            for clip in range(4):
                for view in range(4):
                    if meta['mask'][clip][view]:
                        for modality in ([0,1] if stage=='video' else [0]):
                            self.records.append((index,clip,view,modality))
    def __len__(self): return len(self.records)
    def __getitem__(self,item):
        index,clip,view,modality=self.records[item]
        if index not in self._metadata:
            if len(self._metadata)>=2:self._metadata.popitem(last=False)
            self._metadata[index]=json.loads((self.root/f'{index:05d}.meta.json').read_text(encoding='utf-8'))
        meta=self._metadata[index];positions=list(range(16)) if self.stage=='video' else [0,5,10,15]
        indices=[meta['indices'][clip][i] for i in positions]
        boxes=np.array([meta['boxes'][clip][view][i] for i in positions])
        paths=[meta['paths'][modality][i] for i in indices]
        return {'inputs':load_clip(paths,boxes,modality==0),'record':torch.tensor([index,clip,view,modality])}


class CorpusFeatureDataset(torch.utils.data.Dataset):
    def __init__(self,root,identity):
        self.root=Path(root);path=self.root/'complete.json'
        if not path.is_file(): raise ValueError('full corpus cache is incomplete')
        self.manifest=json.loads(path.read_text(encoding='utf-8'));assert_identity(self.manifest,identity)
        self.rows=self.manifest['rows']
        if len(self.rows)!=2427 or len({r['sample_id'] for r in self.rows})!=2427: raise ValueError('canonical cache population changed')
        for row in self.rows:
            for stage in ('video','appearance'):
                if row.get(stage+'_file'): verify_files({row[stage+'_file']:row[stage+'_sha256']},self.root)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        if row['supported']:
            with np.load(self.root/row['video_file'],allow_pickle=False) as z:
                data={key:torch.from_numpy(np.array(z[key],copy=True)) for key in z.files}
            data['appearance']=torch.from_numpy(np.load(self.root/row['appearance_file'],allow_pickle=False))
        else:data=empty_features()
        data.update(label=row['label'],user=row['user'],sample_id=row['sample_id'],index=index)
        return data
