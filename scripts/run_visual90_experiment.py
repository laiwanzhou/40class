"""Authorized fixed-split workflow: qualify -> cache -> A -> B -> final evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler
from sklearn.metrics import accuracy_score,f1_score,recall_score
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.cache_visual90_features import build_full_cache,worker_init,save_array
from src.data.visual90_corpus import atomic_json,recipe_identity,CorpusFeatureDataset
from src.models.visual90_fusion import Visual90Fusion
from src.models.visual90_encoders import file_hash
from src.training.visual90_training import (paired_epoch_indices,learning_rate_at,cross_user_supcon,
    save_training_state,restore_training_state,decide_results,TRAIN_USERS)


class EpochSampler(Sampler):
    def __init__(self,rows,fit,seed):
        self.fit=np.asarray(fit);self.labels=np.array([rows[i]['label'] for i in fit]);self.users=np.array([rows[i]['user'] for i in fit]);self.seed=seed;self.set_epoch(1)
    def set_epoch(self,epoch):self.indices=self.fit[paired_epoch_indices(self.labels,self.users,self.seed,epoch)]
    def __iter__(self):return iter(self.indices.tolist())
    def __len__(self):return len(self.indices)


def fit_indices(rows):
    chosen=[i for i,r in enumerate(rows) if r['partition']=='train' and r['supported'] and r['disposition']=='eligible_pending_geometry']
    if not chosen or {rows[i]['label'] for i in chosen}!=set(range(40)):raise ValueError('fit population lacks classes')
    if not {rows[i]['user'] for i in chosen}<=TRAIN_USERS:raise ValueError('validation user in fit')
    return chosen


def fit_candidate(dataset,candidate,config,identity,stop_after_epoch=None,model_factory=None,device='cuda'):
    run=ROOT/config['run_root'];folder=run/candidate;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'latest_checkpoint.pt';cid=identity+':'+candidate
    seed=config['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=(model_factory() if model_factory else Visual90Fusion(candidate=='appearance_temporal',seed)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.05)
    start=0;history=[]
    if checkpoint.exists():start,history=restore_training_state(checkpoint,model,optimizer,cid,torch.device(device))
    if not 0<=start<=30:raise ValueError('checkpoint epoch invalid')
    indices=fit_indices(dataset.rows);sampler=EpochSampler(dataset.rows,indices,seed)
    generator=torch.Generator().manual_seed(seed+99)
    kwargs={'batch_size':32,'sampler':sampler,'num_workers':config['num_workers'],
            'pin_memory':device=='cuda','generator':generator}
    if config['num_workers']:kwargs.update(prefetch_factor=1,persistent_workers=True,worker_init_fn=worker_init,timeout=120)
    loader=DataLoader(dataset,**kwargs)
    try:
        for epoch in range(start+1,31):
            if stop_after_epoch is not None and epoch>stop_after_epoch:break
            sampler.set_epoch(epoch)
            # Epoch-owned RNG makes dropout paired across A/B and resume independent of worker startup.
            torch.manual_seed(seed+epoch);np.random.seed(seed+epoch);random.seed(seed+epoch)
            for group in optimizer.param_groups:group['lr']=learning_rate_at(epoch)
            model.train();total=correct=0;loss_sum=0.;started=time.perf_counter()
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            for batch in loader:
                batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):out=model(batch)
                ce=torch.nn.functional.cross_entropy(out['logits'].float(),batch['label'],label_smoothing=.05)
                loss=ce+.10*cross_user_supcon(out['embedding'],batch['label'],batch['user'],batch['sample_id'])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(grad):raise RuntimeError('nonfinite training gradients')
                optimizer.step()
                n=len(batch['label']);total+=n;loss_sum+=float(loss.detach())*n
                correct+=int((out['logits'].argmax(-1)==batch['label']).sum())
                if device=='cuda' and torch.cuda.max_memory_allocated()/2**20>config['maximum_allocated_mib']:
                    raise RuntimeError('formal training GPU allocation exceeded gate')
            if device=='cuda':torch.cuda.synchronize()
            record={'epoch':epoch,'lr':learning_rate_at(epoch),'loss':loss_sum/total,'online_train_acc':correct/total,
                'sample_exposures':total,'fit_count':len(indices),'class_exposures':np.bincount([dataset.rows[i]['label'] for i in sampler.indices],minlength=40).tolist(),
                'sampler_sha256':hashlib.sha256(sampler.indices.tobytes()).hexdigest(),'seconds':time.perf_counter()-started}
            history.append(record);save_training_state(checkpoint,model,optimizer,epoch,cid,history)
            atomic_json(folder/'history.json',history)
            atomic_json(run/'status.json',{'phase':'training','candidate':candidate,'epoch':epoch,'epochs':30,'identity':identity,**record})
            print(candidate+' '+json.dumps(record),flush=True)
    finally:
        iterator=getattr(loader,'_iterator',None)
        if iterator is not None:iterator._shutdown_workers()
    return model,history


def metrics(logits,labels,users):
    pred=logits.argmax(1);correct=pred==labels;rank=np.argsort(-logits,axis=1)
    per_user={str(int(u)):float(correct[users==u].mean()) for u in np.unique(users)}
    logp=torch.log_softmax(torch.from_numpy(logits).float(),dim=1)
    recall=recall_score(labels,pred,labels=list(range(40)),average=None,zero_division=0)
    return {'samples':len(labels),'correct':int(correct.sum()),'accuracy':float(correct.mean()),
        'macro_f1':float(f1_score(labels,pred,labels=list(range(40)),average='macro',zero_division=0)),
        'worst_user_accuracy':min(per_user.values()),'per_user_accuracy':per_user,
        'top3_accuracy':float((rank[:,:3]==labels[:,None]).any(1).mean()),
        'top5_accuracy':float((rank[:,:5]==labels[:,None]).any(1).mean()),
        'nll':float(-logp[np.arange(len(labels)),labels].mean()),'zero_recall_classes':int((recall==0).sum()),'per_class_recall':recall.tolist()}


def evaluate(dataset,model,fit,device='cuda'):
    model.eval();prior=np.bincount([dataset.rows[i]['label'] for i in fit],minlength=40)+1
    prior=torch.tensor(np.log(prior/prior.sum()),device=device,dtype=torch.float32)
    values=[];labels=[];users=[];ids=[]
    loader=DataLoader(dataset,batch_size=32,num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):out=model(batch)
            logits=torch.where(out['supported'][:,None],out['logits'].float(),prior[None])
            values.append(logits.cpu().numpy());labels.extend(batch['label'].cpu().tolist());users.extend(batch['user'].cpu().tolist());ids.extend(batch['sample_id'])
    logits=np.concatenate(values);y=np.array(labels);u=np.array(users);ids=np.array(ids)
    train=np.array([r['partition']=='train' for r in dataset.rows]);val=~train
    if int(train.sum())!=2039 or int(val.sum())!=388 or set(u[val])!={6,7}:raise ValueError('evaluation population changed')
    return {'train_fit':metrics(logits[fit],y[fit],u[fit]),'train_canonical':metrics(logits[train],y[train],u[train]),
        'validation':metrics(logits[val],y[val],u[val])}, {'logits':logits,'labels':y,'users':u,'sample_ids':ids,'validation_mask':val}


def validate_config(config):
    required={'formal_authorization':'user_2026_09_08','epochs':30,'batch_size':32,'num_workers':4,
        'seed':20260715,'allow_threefold':False,'automatic_recipe_changes':False,
        'learning_rate':.0003,'weight_decay':.05,'label_smoothing':.05,'supcon_weight':.10,
        'supcon_temperature':.10,'warmup_epochs':2,'gradient_clip':5.,'prefetch_factor':1,
        'videomae_clip_batch':1,'dino_image_batch':16,'amp_dtype':'bfloat16',
        'validation_users':[6,7],'preflight_expected_train':2039,'preflight_expected_validation':388}
    if any(config.get(k)!=v for k,v in required.items()):raise ValueError('formal config contract changed')
    if config['candidates']!=['temporal_visual','appearance_temporal']:raise ValueError('candidate protocol changed')


def validate_resource_gates(config):
    bundle=json.loads((ROOT/'reports/visual90_appearance_temporal_smoke_measurements.json').read_text())['reports']
    cases=bundle['resource_report']['synthetic_pressure']
    for candidate in ('A','B'):
        matches=[x for x in cases if x.get('candidate')==candidate and x.get('device')=='cuda' and x.get('batch')==32]
        if len(matches)!=1 or matches[0]['status']!='pass' or not matches[0]['parameters_changed']:
            raise ValueError('batch32 backward gate missing')
        if matches[0]['peak_allocated_mib']>=config['maximum_allocated_mib']:raise ValueError('smoke allocation gate failed')
    real=bundle['resource_report']['real_feature_smoke']
    if {r['candidate'] for r in real if r['status']=='pass' and r['batch']==32}!={'A','B'}:
        raise ValueError('real feature backward gate missing')
    if not all(x['finite'] for x in bundle['encoder_report']['encoders'].values()):raise ValueError('encoder gate failed')
    seconds=sum(x['wall_seconds'] for x in bundle['encoder_report']['encoders'].values())/8*2427
    if seconds>12*3600:raise ValueError('extraction time estimate exceeds budget')
    return {'estimated_extraction_seconds_from_smoke':seconds,'source':'recorded_2026_09_06_smoke'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/experiments/visual90_appearance_temporal.yaml');p.add_argument('--check-only',action='store_true');args=p.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'));validate_config(config)
    resource_gates=validate_resource_gates(config)
    torch.set_num_threads(config['main_torch_threads'])
    identity,provenance=recipe_identity(config);run=ROOT/config['run_root']
    if args.check_only:
        print(json.dumps({'identity':identity,'config_valid':True,'formal_started':False}));return
    run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():raise FileExistsError('completed experiment exists')
    # OS byte-range lock is released even after a crash; the JSON is informational.
    import msvcrt
    guard=(run/'run.lock').open('a+b');guard.seek(0);guard.write(b'0');guard.flush();guard.seek(0)
    try:msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('another Visual90 workflow is already running')
    try:
        atomic_json(run/'process.json',{'pid':os.getpid(),'started_unix':time.time(),'identity':identity,'argv':sys.argv})
        atomic_json(run/'recipe.json',{'identity':identity,**provenance,'resource_gates':resource_gates,
            'device':torch.cuda.get_device_name(),'cuda_free_total_bytes':list(torch.cuda.mem_get_info())})
        cache=build_full_cache(config,identity,provenance);dataset=CorpusFeatureDataset(cache,identity)
        fit=fit_indices(dataset.rows)
        # Neither candidate evaluates user6/user7 until both fits are complete.
        for candidate in config['candidates']:
            model,history=fit_candidate(dataset,candidate,config,identity)
            del model;torch.cuda.empty_cache()
        results={};archives={}
        for candidate in config['candidates']:
            model=Visual90Fusion(candidate=='appearance_temporal',config['seed']).cuda()
            payload=torch.load(run/candidate/'latest_checkpoint.pt',map_location='cpu',weights_only=False)
            if payload['identity']!=identity+':'+candidate or payload['epoch']!=30:raise ValueError('final checkpoint ownership/epoch mismatch')
            model.load_state_dict(payload['model'],strict=True)
            result,archive=evaluate(dataset,model,fit);results[candidate]=result;archives[candidate]=archive
            save_array(run/candidate/'predictions.npz',archive)
            atomic_json(run/candidate/'metrics.json',result);del model;torch.cuda.empty_cache()
        a,b=[archives[c] for c in config['candidates']];mask=a['validation_mask'];y=a['labels'][mask]
        ac=a['logits'][mask].argmax(1)==y;bc=b['logits'][mask].argmax(1)==y
        paired={'rescue':int((~ac&bc).sum()),'harm':int((ac&~bc).sum()),'net':int(bc.sum()-ac.sum())}
        for c in range(40):
            cm=y==c;paired.setdefault('per_class',[]).append({'class_id':c,'support':int(cm.sum()),'rescue':int((~ac&bc&cm).sum()),'harm':int((ac&~bc&cm).sum())})
        report={'identity':identity,'results':results,'paired':paired,'decision':decide_results(*[results[c]['validation'] for c in config['candidates']]),'validation_is_development':True}
        atomic_json(run/'result.json',report);atomic_json(ROOT/'reports/visual90_appearance_temporal_result.json',report)
        atomic_json(run/'status.json',{'phase':'completed','identity':identity,'result':str(run/'result.json')})
        print(json.dumps(report),flush=True)
    except BaseException as error:
        atomic_json(run/'failure.json',{'time':time.time(),'identity':identity,'error':repr(error),'traceback':traceback.format_exc()})
        atomic_json(run/'status.json',{'phase':'failed','identity':identity,'error':repr(error)})
        raise
    finally:guard.close()


if __name__=='__main__':main()
