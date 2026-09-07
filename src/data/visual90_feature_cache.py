"""Lazy per-worker mmap reader for smoke-scale frozen feature arrays."""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch

FEATURE_KEYS=('video','appearance','video_mask','appearance_mask','roi','times','appearance_times','appearance_roi')


def verify_array_hashes(root, manifest):
    expected=manifest.get('array_sha256',{})
    if set(expected)!=set(FEATURE_KEYS):
        raise ValueError('cache array hash manifest is incomplete')
    for key in FEATURE_KEYS:
        digest=hashlib.sha256()
        path=Path(root)/(key+'.npy')
        if not path.is_file(): raise ValueError(f'cache hash check: missing {key}')
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
        if digest.hexdigest()!=expected[key]: raise ValueError(f'cache array hash mismatch: {key}')


class FeatureCacheDataset(torch.utils.data.Dataset):
    def __init__(self,root,length=None):
        self.root=Path(root)
        marker=self.root/'complete.json'
        if not marker.is_file(): raise ValueError('feature cache is not complete')
        self.manifest=json.loads(marker.read_text(encoding='utf-8'))
        self.count=len(self.manifest['sample_ids'])
        if not self.count or len(set(self.manifest['sample_ids']))!=self.count:
            raise ValueError('invalid cache sample identities')
        if len(self.manifest['labels'])!=self.count or len(self.manifest['users'])!=self.count:
            raise ValueError('cache label/user count differs from sample count')
        verify_array_hashes(self.root,self.manifest)
        self.length=self.count if length is None else length
        self._arrays=None
    def __len__(self): return self.length
    def __getstate__(self):
        state=self.__dict__.copy(); state['_arrays']=None; return state
    def __getitem__(self,index):
        if self._arrays is None:
            self._arrays={key:np.load(self.root/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in FEATURE_KEYS}
        row=index%self.count
        result={key:torch.from_numpy(np.array(value[row],copy=True)) for key,value in self._arrays.items()}
        result.update(label=self.manifest['labels'][row],user=self.manifest['users'][row],
                      sample_id=self.manifest['sample_ids'][row])
        return result
