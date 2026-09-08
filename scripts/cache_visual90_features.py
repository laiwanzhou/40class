"""Resumable full-corpus feature extraction. Never fits a classifier."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import pandas as pd

from src.data.visual90_corpus import (ROOT,atomic_json,assert_identity,verify_files,
    prepare_trial,metadata_arrays,CorpusClipDataset)
from src.data.pose_roi_dataset import PoseTrackCache
from src.models.visual90_encoders import VideoEncoder,AppearanceEncoder,file_hash


def worker_init(_):torch.set_num_threads(1)


def save_array(path,value):
    temp=path.with_suffix(path.suffix+'.partial')
    with temp.open('wb') as f:
        if isinstance(value,dict):np.savez(f,**value)
        else:np.save(f,value,allow_pickle=False)
    temp.replace(path)


def build_full_cache(config,identity,provenance):
    run=ROOT/config['run_root'];root=run/'cache';root.mkdir(parents=True,exist_ok=True)
    lockpath=root/'identity.json'
    if lockpath.exists():assert_identity(json.loads(lockpath.read_text()),identity)
    else:
        if any(root.iterdir()):raise ValueError('unowned cache directory')
        if config['expected_cache_budget_gib']*2**30>shutil.disk_usage(root).free*.70:
            raise RuntimeError('cache disk budget gate failed')
        atomic_json(lockpath,{'identity':identity,'provenance':provenance})
    pf=json.loads((ROOT/config['preflight']).read_text(encoding='utf-8'))
    if pf['population']!={'train':2039,'validation':388} or pf['continuity_blockers']:
        raise ValueError('population or continuity gate changed')
    manifest_path=ROOT/'metadata/manifest.csv'
    split_path=ROOT/'metadata/splits/train12_val2_user6_user7_development.json'
    if file_hash(manifest_path)!=pf['manifest_sha256'] or file_hash(split_path)!=pf['split_sha256']:
        raise ValueError('preflight manifest/split is stale')
    source_manifest=pd.read_csv(manifest_path).set_index('sample_id')
    split=json.loads(split_path.read_text())
    for row in pf['rows']:
        source=source_manifest.loc[row['sample_id']]
        if int(source['class_id'])!=row['class_id'] or source['user_id']!=row['user_id']:
            raise ValueError('preflight row label/user differs from source manifest')
        if row['user_id'] not in split['train_user_ids' if row['partition']=='train' else 'validation_user_ids']:
            raise ValueError('preflight partition ownership differs from split')
    if file_hash(pf['pose_path'])!=pf['pose_sha256']:raise ValueError('pose cache changed')
    geometry=json.loads((ROOT/config['geometry']).read_text(encoding='utf-8'))
    if geometry.get('status')!='verified' or geometry.get('preflight_sha256')!=file_hash(ROOT/config['preflight']):
        raise ValueError('corpus geometry not qualified for this preflight')
    pose=PoseTrackCache(Path(pf['pose_path']))
    roi=yaml.safe_load((ROOT/'configs/experiments/ir_depth_videomaev2_vit_b_p0.yaml').read_text())['roi']
    rows=[]
    for i,row in enumerate(pf['rows']):
        path=root/f'{i:05d}.meta.json'
        if path.exists():
            meta=json.loads(path.read_text(encoding='utf-8'));assert_identity(meta,identity)
            if meta['sample_id']!=row['sample_id']:raise ValueError('metadata row reorder')
            verify_files(meta['source_files'],Path('/'))
        else:
            meta=prepare_trial(row,pose,roi,True);meta['identity']=identity;atomic_json(path,meta)
        rows.append({key:meta[key] for key in ('sample_id','label','user','partition','disposition','supported')})
        if i%50==0:
            atomic_json(run/'status.json',{'phase':'source_integrity','rows_done':i+1,'rows_total':len(pf['rows']),'identity':identity})
            print(f'source integrity {i+1}/{len(pf["rows"])}',flush=True)
    del pose;gc.collect()
    fit=[r for r in rows if r['partition']=='train' and r['disposition']=='eligible_pending_geometry' and r['supported']]
    if {r['label'] for r in fit}!=set(range(40)):raise ValueError('eligible cache loses classes')
    lock=json.loads((ROOT/config['encoder_lock']).read_text())
    for stage in ('video','appearance'):
        pending=[]
        for i,row in enumerate(rows):
            if not row['supported']:continue
            marker=root/f'{i:05d}.{stage}.json'
            if marker.exists():
                completed=json.loads(marker.read_text());assert_identity(completed,identity)
                if completed['meta_sha256']!=file_hash(root/f'{i:05d}.meta.json'):raise ValueError('stage metadata changed')
                verify_files({completed['file']:completed['sha256']},root)
                row[stage+'_file']=completed['file'];row[stage+'_sha256']=completed['sha256']
            else:pending.append(i)
        if not pending:continue
        print(f'loading {stage}; pending trials={len(pending)}',flush=True)
        if stage=='video':model=VideoEncoder(lock['videomae']['local_checkpoint'])
        else:
            d=lock['dinov2'];model=AppearanceEncoder(d['local_source'],d['local_checkpoint'],d['checkpoint_sha256'])
        model=model.cuda().eval();torch.cuda.reset_peak_memory_stats()
        dataset=CorpusClipDataset(root,pending,stage)
        dl=DataLoader(dataset,batch_size=1 if stage=='video' else 4,num_workers=4,prefetch_factor=1,
            persistent_workers=True,pin_memory=True,worker_init_fn=worker_init,timeout=120)
        current=None;buffer=None;done=0;started=time.perf_counter()
        def finish(index,array):
            nonlocal done
            if index is None:return
            if not np.isfinite(array).all():raise RuntimeError('nonfinite persisted cache')
            if stage=='video':
                meta=json.loads((root/f'{index:05d}.meta.json').read_text(encoding='utf-8'))
                values=metadata_arrays(meta);values['video']=array;extension='npz'
            else:values=array;extension='npy'
            filename=f'{index:05d}.{stage}.{extension}';save_array(root/filename,values)
            digest=file_hash(root/filename)
            atomic_json(root/f'{index:05d}.{stage}.json',{'identity':identity,'file':filename,'sha256':digest,
                'meta_sha256':file_hash(root/f'{index:05d}.meta.json'),'finite':True})
            rows[index][stage+'_file']=filename;rows[index][stage+'_sha256']=digest;done+=1
            if done%10==0 or done==len(pending):
                torch.cuda.synchronize()
                atomic_json(run/'status.json',{'phase':'cache_'+stage,'trials_done_this_run':done,'trials_pending_at_start':len(pending),
                    'elapsed_seconds':time.perf_counter()-started,'identity':identity,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
                print(f'{stage} {done}/{len(pending)} trials, {time.perf_counter()-started:.1f}s',flush=True)
        try:
            for batch in dl:
                x=batch['inputs'].cuda(non_blocking=True)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(x if stage=='video' else x.permute(0,2,1,3,4).reshape(-1,3,224,224))
                if torch.cuda.max_memory_allocated()/2**20>config['maximum_allocated_mib']:raise RuntimeError('GPU memory gate failed')
                if not torch.isfinite(output).all():raise RuntimeError('nonfinite encoder output')
                values=output.float().cpu().numpy()
                if stage=='appearance':values=values.reshape(len(x),4,17,1024)
                for value,record in zip(values,batch['record'].tolist(),strict=True):
                    index,clip,view,modality=record
                    if index!=current:
                        finish(current,buffer);current=index
                        buffer=np.zeros((2,4,4,8,4,768) if stage=='video' else (4,4,4,17,1024),dtype=np.float16)
                    if stage=='video':buffer[modality,clip,view]=value
                    else:buffer[clip,view]=value
            finish(current,buffer)
        finally:
            iterator=getattr(dl,'_iterator',None)
            if iterator is not None:iterator._shutdown_workers()
        del model,dl,dataset,buffer,output,x;gc.collect();torch.cuda.empty_cache()
    for row in rows:
        if row['supported'] and not all(row.get(s+'_file') for s in ('video','appearance')):raise ValueError('incomplete supported cache')
    atomic_json(root/'complete.json',{'identity':identity,'rows':rows,'fit_count':len(fit),'population':pf['population']})
    return root
