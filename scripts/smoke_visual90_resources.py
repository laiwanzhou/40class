"""Bounded resource probes only: this CLI cannot launch formal training."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.models.visual90_encoders import VideoEncoder,AppearanceEncoder,file_hash
from src.models.visual90_fusion import Visual90Fusion
from src.training.visual90_training import cross_user_supcon
from src.data.visual90_evidence import RawClipDataset
from src.data.visual90_feature_cache import FeatureCacheDataset,FEATURE_KEYS
from src.data.visual90_dataset import require_geometry

SMOKE=ROOT/'outputs/visual90_appearance_temporal/smoke'


def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['extract','benchmark','integrated','encoder-batches'],required=True)
    return p.parse_args(argv)


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8'); temp.replace(path)


def init_worker(worker):
    torch.set_num_threads(1)


def loader(dataset,batch,workers,pin=False):
    kwargs={'num_workers':workers,'batch_size':batch,'shuffle':False,'pin_memory':pin}
    if workers: kwargs.update(prefetch_factor=1,persistent_workers=True,worker_init_fn=init_worker,timeout=60)
    return DataLoader(dataset,**kwargs)


def close_loader(dl):
    it=getattr(dl,'_iterator',None)
    if it is not None: it._shutdown_workers()


def memory():
    proc=psutil.Process()
    children=proc.children(recursive=True)
    rss=proc.memory_info().rss
    for child in children:
        try: rss+=child.memory_info().rss
        except psutil.Error: pass
    return {'process_tree_rss_mib':rss/2**20,'system_available_mib':psutil.virtual_memory().available/2**20}


def evidence():
    path=SMOKE/'evidence.json'
    data=json.loads(path.read_text())
    geometry=json.loads((ROOT/'reports/visual90_appearance_temporal_geometry_smoke.json').read_text())
    require_geometry(geometry)
    if file_hash(path)!=geometry['source_sha256'] or len(data['trials'])!=8:
        raise ValueError('geometry/evidence identity mismatch')
    for trial in data['trials']:
        for filename,digest in trial['source_hashes']:
            if file_hash(filename)!=digest: raise ValueError('source image changed after review')
    return data


def sync(device):
    if device=='cuda': torch.cuda.synchronize()


def extract():
    data=evidence(); records=data['records']
    destination=SMOKE/'features'
    if destination.exists(): raise FileExistsError(destination)
    destination.mkdir()
    lock=json.loads((ROOT/'configs/experiments/visual90_encoder_lock.json').read_text())
    report={'scope':'eight_training_trials_only','formal_training_started':False,'encoders':{}}
    video=np.zeros((8,2,4,4,8,4,768),dtype=np.float16)
    appearance=np.zeros((8,4,4,4,17,1024),dtype=np.float16)
    for name in ['video','appearance']:
        if name=='video': model=VideoEncoder(lock['videomae']['local_checkpoint'])
        else:
            d=lock['dinov2']; model=AppearanceEncoder(d['local_source'],d['local_checkpoint'],d['checkpoint_sha256'])
        model=model.to('cuda').eval()
        subset=records if name=='video' else [r for r in records if r['modality']==0]
        dl=loader(RawClipDataset(subset),1,0)
        elapsed=[]; peak=0
        started=time.perf_counter()
        for j,batch in enumerate(dl):
            x=batch['clip'].to('cuda'); torch.cuda.reset_peak_memory_stats(); sync('cuda'); t=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                inp=x if name=='video' else x[:,:, [0,5,10,15]].permute(0,2,1,3,4).reshape(4,3,224,224)
                out=model(inp)
            sync('cuda'); elapsed.append(time.perf_counter()-t)
            peak=max(peak,torch.cuda.max_memory_allocated()/2**20)
            if not torch.isfinite(out).all(): raise RuntimeError('nonfinite encoder output')
            r=subset[j]; value=out.float().cpu().numpy().astype(np.float16)
            if name=='video': video[r['trial'],r['modality'],r['clip_index'],r['view']]=value[0]
            else: appearance[r['trial'],r['clip_index'],r['view']]=value
            if j%40==0: print(f'{name}: {j+1}/{len(subset)}',flush=True)
        report['encoders'][name]={'calls':len(subset),'wall_seconds':time.perf_counter()-started,
            'median_forward_seconds':float(np.median(elapsed[1:])),'peak_allocated_mib':peak,
            'provenance':model.provenance,'finite':True}
        np.save(destination/('video.npy' if name=='video' else 'appearance.npy'),video if name=='video' else appearance)
        del model,out,x,dl; gc.collect(); torch.cuda.empty_cache()
    roi=[]; times=[]; ar=[]; at=[]; masks=[]
    for trial in data['trials']:
        raw=np.array(trial['boxes'],dtype=np.float32)/[640,480,640,480]
        centers=np.stack(((raw[...,0]+raw[...,2])/2,(raw[...,1]+raw[...,3])/2,
                           raw[...,2]-raw[...,0],raw[...,3]-raw[...,1]),axis=-1)
        roi.append(centers.reshape(4,4,8,2,4).mean(3))
        ar.append(centers[:,:, [0,5,10,15]])
        t=np.array(trial['frame_times'],dtype=np.float32); t=t/max(float(t.max()),1.)
        times.append(t.reshape(4,8,2).mean(2)); at.append(t[:,[0,5,10,15]])
        masks.append(trial['mask'])
    mask=np.array(masks,dtype=bool)
    arrays={'roi':np.array(roi,dtype=np.float32),'times':np.array(times,dtype=np.float32),
            'appearance_roi':np.array(ar,dtype=np.float32),'appearance_times':np.array(at,dtype=np.float32),
            'video_mask':np.stack((mask,mask),axis=1),'appearance_mask':mask}
    for key,value in arrays.items(): np.save(destination/(key+'.npy'),value)
    manifest={'sample_ids':[t['sample_id'] for t in data['trials']],
        'labels':[t['class_id'] for t in data['trials']],
        'users':[int(t['user_id'].removeprefix('user')) for t in data['trials']],
        'evidence_sha256':file_hash(SMOKE/'evidence.json'),
        'array_sha256':{key:file_hash(destination/(key+'.npy')) for key in FEATURE_KEYS},
        'scope':'smoke_only_not_full_population'}
    write_json(destination/'complete.json',manifest)
    write_json(SMOKE/'encoder_report.json',report)
    print(json.dumps(report),flush=True)


def synthetic(batch):
    g=torch.Generator().manual_seed(20260715)
    data={'video':torch.randn(batch,2,4,4,8,4,768,generator=g,dtype=torch.float16),
        'appearance':torch.randn(batch,4,4,4,17,1024,generator=g,dtype=torch.float16),
        'video_mask':torch.ones(batch,2,4,4,dtype=torch.bool),
        'appearance_mask':torch.ones(batch,4,4,dtype=torch.bool),
        'roi':torch.rand(batch,4,4,8,4,generator=g),
        'times':torch.linspace(0,1,32).reshape(1,4,8).expand(batch,-1,-1),
        'label':torch.arange(batch)//2%40,'user':torch.arange(batch)%2,
        'sample_id':[f'synthetic_{i}' for i in range(batch)]}
    return data


def train_case(data,device,appearance,steps=3):
    model=Visual90Fusion(appearance).to(device).train()
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.05)
    batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in data.items()}
    timings=[]; losses=[]; before=model.classifier.weight.detach().clone()
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    for step in range(steps+1):
        sync(device); t=time.perf_counter(); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext():
            out=model(batch)
        ce=torch.nn.functional.cross_entropy(out['logits'].float(),batch['label'],label_smoothing=.05)
        loss=ce+.1*cross_user_supcon(out['embedding'],batch['label'],batch['user'],batch['sample_id'])
        if not torch.isfinite(loss): raise RuntimeError('nonfinite smoke loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(grad): raise RuntimeError('nonfinite smoke gradients')
        opt.step(); sync(device)
        if step: timings.append(time.perf_counter()-t)
        losses.append(float(loss.detach()))
    result={'status':'pass','batch':len(batch['label']),'device':device,'candidate':'B' if appearance else 'A',
        'median_step_seconds':float(np.median(timings)),'samples_per_second':len(batch['label'])/float(np.median(timings)),
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else None,
        'parameters_changed':not torch.equal(before,model.classifier.weight),'losses':losses,**memory()}
    del model,opt,batch,out,loss; gc.collect()
    if device=='cuda': torch.cuda.empty_cache()
    return result


def benchmark():
    path=SMOKE/'resource_report.json'
    if path.exists(): raise FileExistsError(path)
    dataset=FeatureCacheDataset(SMOKE/'features',length=256)
    for key,digest in dataset.manifest['array_sha256'].items():
        if file_hash(SMOKE/'features'/(key+'.npy'))!=digest: raise ValueError('smoke cache changed')
    report={'formal_training_started':False,'torch_version':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'cpu_threads':8,'hardware_start':memory(),
        'synthetic_pressure':[],'real_feature_smoke':[],'feature_workers':[],'raw_workers':[]}
    for device,batches in [('cuda',[8,16,32,64]),('cpu',[1,4,8,16,32])]:
        for appearance in [False,True]:
            for size in batches:
                try: result=train_case(synthetic(size),device,appearance)
                except torch.cuda.OutOfMemoryError as error:
                    gc.collect();torch.cuda.empty_cache()
                    result={'status':'oom','device':device,'candidate':'B' if appearance else 'A','batch':size,'error':str(error)}
                report['synthetic_pressure'].append(result);write_json(path,report)
                print('PRESSURE '+json.dumps(result),flush=True)
    real=next(iter(loader(dataset,32,0)))
    for appearance in [False,True]:
        result=train_case(real,'cuda',appearance)
        report['real_feature_smoke'].append(result);write_json(path,report)
        print('REAL '+json.dumps(result),flush=True)
    for workers in [0,2,4]:
        dl=loader(dataset,32,workers,pin=True)
        cold=time.perf_counter(); it=iter(dl); first=next(it); startup=time.perf_counter()-cold
        waits=[]; rss=[]; checks=[]; first_checksum=float(first['video'].float().sum())
        for epoch in range(2):
            it=iter(dl)
            while True:
                t=time.perf_counter()
                try: batch=next(it)
                except StopIteration: break
                moved={k:v.to('cuda',non_blocking=True) for k,v in batch.items() if torch.is_tensor(v)}
                torch.cuda.synchronize(); waits.append(time.perf_counter()-t)
                rss.append(memory()['process_tree_rss_mib']);checks.extend(batch['sample_id'])
                del moved
        result={'workers':workers,'batch':32,'prefetch_factor':1 if workers else None,
            'persistent_workers':bool(workers),'first_batch_seconds':startup,
            'median_load_and_transfer_seconds':float(np.median(waits)),
            'tree_rss_peak_observed_mib':max(rss),'sample_count':len(checks),'first_video_checksum':first_checksum,
            'scope':'hot eight-trial mmap cache repeated; not full-corpus cold I/O'}
        report['feature_workers'].append(result); close_loader(dl);del dl,it,first,batch;gc.collect()
        write_json(path,report);print('FEATURE_WORKERS '+json.dumps(result),flush=True)
    raw=RawClipDataset(json.loads((SMOKE/'evidence.json').read_text())['records'][:32])
    for workers in [0,2,4]:
        dl=loader(raw,1,workers);t=time.perf_counter();first=next(iter(dl));startup=time.perf_counter()-t
        timings=[];rss=[]
        for epoch in range(2):
            t=time.perf_counter();count=0
            for batch in dl: count+=len(batch['index']);rss.append(memory()['process_tree_rss_mib'])
            timings.append(time.perf_counter()-t)
        result={'workers':workers,'batch':1,'first_batch_seconds':startup,'epoch_seconds':timings,
                'clips_per_second':count/timings[-1],'tree_rss_peak_observed_mib':max(rss)}
        report['raw_workers'].append(result);close_loader(dl);del dl,first,batch;gc.collect()
        write_json(path,report);print('RAW_WORKERS '+json.dumps(result),flush=True)
    report['status']='smoke_completed_formal_training_not_started';write_json(path,report)


def integrated():
    path=SMOKE/'integrated_report.json'
    if path.exists(): raise FileExistsError(path)
    results=[]
    for workers in [0,2,4]:
        dataset=FeatureCacheDataset(SMOKE/'features',length=32*12)
        dl=loader(dataset,32,workers,pin=True)
        model=Visual90Fusion(True).cuda().train()
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.05)
        torch.cuda.reset_peak_memory_stats(); intervals=[]; waits=[];rss=[];losses=[]
        start=time.perf_counter(); it=iter(dl)
        for step in range(12):
            t=time.perf_counter(); batch=next(it); waits.append(time.perf_counter()-t)
            batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): out=model(batch)
            loss=torch.nn.functional.cross_entropy(out['logits'].float(),batch['label'],label_smoothing=.05)
            loss=loss+.1*cross_user_supcon(out['embedding'],batch['label'],batch['user'],batch['sample_id'])
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            if not torch.isfinite(loss) or not torch.isfinite(grad): raise RuntimeError('integrated nonfinite')
            opt.step();torch.cuda.synchronize()
            intervals.append(time.perf_counter()-t);rss.append(memory()['process_tree_rss_mib']);losses.append(float(loss.detach()))
        result={'workers':workers,'physical_batch':32,'candidate':'B','steps':12,
            'first_step_seconds':intervals[0],'median_warm_step_seconds':float(np.median(intervals[2:])),
            'median_warm_loader_wait_seconds':float(np.median(waits[2:])),
            'warm_samples_per_second':32/float(np.median(intervals[2:])),
            'wall_seconds_including_startup':time.perf_counter()-start,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'tree_rss_peak_observed_mib':max(rss),'finite':True,
            'scope':'8 real training examples repeated; no accuracy claim or retained optimizer'}
        results.append(result);write_json(path,{'formal_training_started':False,'results':results})
        print('INTEGRATED '+json.dumps(result),flush=True)
        close_loader(dl);del dl,it,model,opt,batch,out,loss;gc.collect();torch.cuda.empty_cache()


def encoder_batches():
    path=SMOKE/'encoder_batch_report.json'
    if path.exists(): raise FileExistsError(path)
    data=evidence(); source=next(r for r in data['records'] if r['modality']==0)
    clip=RawClipDataset([source])[0]['clip'][None]
    lock=json.loads((ROOT/'configs/experiments/visual90_encoder_lock.json').read_text());results=[]
    for name,batches in [('video',[1,2,4,8]),('appearance',[1,4,8,16])]:
        if name=='video': model=VideoEncoder(lock['videomae']['local_checkpoint']).cuda().eval()
        else:
            d=lock['dinov2'];model=AppearanceEncoder(d['local_source'],d['local_checkpoint'],d['checkpoint_sha256']).cuda().eval()
        for size in batches:
            x=(clip.repeat(size,1,1,1,1) if name=='video' else clip[:,:,0].repeat(size,1,1,1)).cuda()
            intervals=[];torch.cuda.reset_peak_memory_stats()
            for step in range(4):
                torch.cuda.synchronize();t=time.perf_counter()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): out=model(x)
                torch.cuda.synchronize()
                if step: intervals.append(time.perf_counter()-t)
                if not torch.isfinite(out).all(): raise RuntimeError('nonfinite encoder batch')
            result={'encoder':name,'batch':size,'median_forward_seconds':float(np.median(intervals)),
                    'items_per_second':size/float(np.median(intervals)),
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                    'scope':'repeated real clip/image, inference only; excludes decode'}
            results.append(result);write_json(path,{'formal_training_started':False,'results':results})
            print('ENCODER_BATCH '+json.dumps(result),flush=True);del x,out;torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()


def main(argv=None):
    args=parse_args(argv);torch.set_num_threads(8)
    torch.manual_seed(20260715)
    if args.phase=='extract': extract()
    elif args.phase=='benchmark': benchmark()
    elif args.phase=='integrated': integrated()
    else: encoder_batches()


if __name__=='__main__': main()
