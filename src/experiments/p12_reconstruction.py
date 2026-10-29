"""Fresh phase-correct P12 teachers under the explicitly authorized reconstruction recipe."""
from collections import OrderedDict
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from scipy.special import softmax

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_types import ArtifactRef,RowIndex,Prediction,canonical_bytes,canonical_hash,write_json
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .no_vote_cache_import import immutable_bytes
from .teammate_source import verify_teammate_source,load_teammate_symbol

REVISION='task11_reconstruction_v2'


def ref(value):return ArtifactRef(Path(value['record_path']),value['sha256'])


def load_reconstruction(protocol,context):
    path=protocol.run_root/'protocol'/REVISION/'preparation.json';receipt=json.loads(path.read_text(encoding='utf-8'))
    file=Path(receipt['recipe']['path']);context.hash_file(file,receipt['recipe']['sha256']);body=json.loads(file.read_text(encoding='utf-8'))
    if body['base_protocol_sha256']!=protocol.identity() or body['revision']!=REVISION or body['historical_exact_reproduction'] is not False:
        raise ValueError('wrong reconstruction recipe/protocol')
    copied=protocol.run_root/'protocol'/REVISION/'recipe.json';immutable_bytes(copied,canonical_bytes(body)+b'\n')
    registry=ArtifactRegistry(protocol,context=context)
    reference=registry.register(stage='reconstruction_recipe',kind='raw_cache',phase='raw',files=[copied],
        config={'revision':REVISION,'recipe_sha256':receipt['recipe']['sha256'],'historical_exact_reproduction':False},source_files=[Path(__file__)])
    return body,reference,receipt


def source_p12_ops(protocol,context):
    key=(protocol.identity(),'reconstruction_p12_source')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        model=load_teammate_symbol(report,'aligned_model','AlignedMultimodalModel')
        train=load_teammate_symbol(report,'train','run_epoch');data=sys.modules['aligned_data'];trainer=sys.modules['train']
        context.public_rows[key]={'model_class':model,'run_epoch':train,'create_loader':trainer.create_loader,
            'seed':trainer.seed_everything,'frame_map':data.frame_map,'load_skeleton_sequence':data.load_skeleton_sequence,
            'flip_skeleton':data._flip_skeleton,'sample_positions':data.AlignedMultimodalDataset._sample_positions}
    return context.public_rows[key]


def initialize_skeleton(config,ops,*,device='cuda'):
    ops['seed'](config['seed']);return ops['model_class'](['skeleton'],dropout=float(config['dropout'])).to(device)


class SkeletonDataset(Dataset):
    def __init__(self,index,paths,config,ops,*,augment,labels=None):
        self.index=index;self.paths=paths;self.num_frames=config['num_frames'];self.augment=augment;self.temporal_view=.5
        self.ops=ops;self.labels=labels
        if set(paths)!=set(index.sample_ids):raise ValueError('Skeleton exact cache IDs required')
        if labels is not None and set(labels)!=set(index.sample_ids):raise ValueError('Skeleton label IDs mismatch')
        self.samples=[SimpleNamespace(class_id=labels[s]) for s in index.sample_ids] if labels is not None else []
        self.memory=OrderedDict()
    def __len__(self):return len(self.index.sample_ids)
    def __getitem__(self,i):
        sid=self.index.sample_ids[i]
        if sid not in self.memory:
            with np.load(self.paths[sid],allow_pickle=False) as z:self.memory[sid]=z['skeleton'].astype(np.float32)
            if len(self.memory)>128:self.memory.popitem(last=False)
        values=self.memory[sid];self.memory.move_to_end(sid)
        positions=self.ops['sample_positions'](self,len(values));flip=torch.rand(1).item()<.5 if self.augment else False
        skeleton=torch.from_numpy(values[positions].copy())
        if flip:skeleton=torch.stack([self.ops['flip_skeleton'](x) for x in skeleton])
        result={'skeleton':skeleton,'sample_id':sid,'user_id':self.index.user_ids[i]}
        if self.labels is not None:result['label']=int(self.labels[sid])
        return result


def complete_scores(scores,valid,prior):
    scores=np.asarray(scores,np.float32).copy();valid=np.asarray(valid,bool);prior=np.asarray(prior,np.float32)
    if scores.shape!=(len(valid),40) or prior.shape!=(40,) or not np.isclose(prior.sum(),1):raise ValueError('P12 score/prior shape mismatch')
    scores[~valid]=np.log(np.maximum(prior,1e-12));prob=softmax(scores,axis=1);prob[~valid]=prior
    return scores,prob


def build_skeleton_cache(protocol,partition,context):
    from .imu_rf_teacher import _write_npz
    recipe,rr,_=load_reconstruction(protocol,context);ops=source_p12_ops(protocol,context)
    inputs=load_stage_inputs(protocol,partition,'raw',context=context);rows=read_public_rows(inputs.public_manifest);index=row_index(rows)
    root=protocol.run_root/REVISION/'p12_skeleton_cache'/partition;registry=ArtifactRegistry(protocol,context=context)
    if (root/'artifact.json').exists():return ref(json.loads((root/'artifact.json').read_text(encoding='utf-8')))
    valid=[];digests={};started=time.perf_counter()
    for n,row in enumerate(rows):
        path=root/(row['sample_id']+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if str(z['sample_id'].item())!=row['sample_id']:raise ValueError('Skeleton cache identity drift')
                v=bool(z['valid']);a=z['skeleton'];assert a.ndim==3 and a.shape[1:]==(17,4)
        else:
            maps={};v=False;values=np.empty((0,17,4),np.float32)
            if all(row[k+'_available']=='1' and row[k+'_path'] and Path(row[k+'_path']).is_dir() for k in ['ir','depth','skeleton']):
                maps={k:ops['frame_map'](Path(row[k+'_path']),k) for k in ['ir','depth','skeleton']}
                shared=sorted(set.intersection(*(set(x) for x in maps.values())))
                if shared:
                    values=ops['load_skeleton_sequence']([maps['skeleton'][x] for x in shared],False,'first').numpy()
                    v=bool(np.any(values[:,:,3]>0))
            _write_npz(path,{'sample_id':np.asarray(row['sample_id']),'skeleton':values,'valid':np.asarray(v)})
        valid.append(v);digests[str(path.resolve())]=context.hash_file(path)
        if (n+1)%100==0:print(json.dumps({'event':'p12_skeleton_cache','partition':partition,'done':n+1,'total':len(rows),'seconds':round(time.perf_counter()-started,1)}),flush=True)
    meta=root/'index.json';immutable_bytes(meta,canonical_bytes({'sample_ids':index.sample_ids,'user_ids':index.user_ids,'valid':valid,
        'source_alignment':'source_IR_Depth_Skeleton_common_frame_ids','source_person_strategy':'first','source_representation':'frame_joint_4',
        'recipe_sha256':canonical_hash(recipe)})+b'\n');digests[str(meta.resolve())]=context.hash_file(meta)
    result=registry.register(stage='p12_skeleton_input',kind='raw_cache',phase='raw',files=list(digests),file_digests=digests,
        parents=[rr,inputs.parents['manifest']],rows=index,config={'partition':partition,'revision':REVISION,'valid_rows':sum(valid)},source_files=[Path(__file__)])
    write_json(root/'artifact.json',result);return result


def cache_index(reference,protocol,context):
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(reference,'p12_skeleton_input','raw')
    path=next(Path(x) for x in record.files if Path(x).name=='index.json');registry.verify_file(record,path)
    body=json.loads(path.read_text(encoding='utf-8'))
    if tuple(body['sample_ids'])!=record.rows.sample_ids or tuple(body['user_ids'])!=record.rows.user_ids:raise ValueError('Skeleton cache row index drift')
    return record,np.asarray(body['valid'],bool),{s:path.parent/(s+'.npz') for s in record.rows.sample_ids}


def allows_p12_reconstruction_stage(registry,record):
    if record.stage!='source_expert/p12_skeleton' or record.kind!='supervised_model' or record.phase not in {'select','refit'}:return False
    if record.config.get('revision')!=REVISION or record.config.get('budget_epochs')!=15 or len(record.parents)!=4:return False
    p=registry.protocol;c=registry.context or VerificationContext(p);recipe,rr,_=load_reconstruction(p,c)
    cfg=record.config;pop='train12' if record.phase=='select' else 'refit14'
    parents=[registry.read(x) for x in record.parents]
    expected=recipe['source_models']['skeleton']['configuration']
    return (cfg.get('revision')==REVISION and cfg.get('fit_partition')==pop and cfg.get('budget_epochs')==15
        and cfg.get('recipe_sha256')==canonical_hash(recipe) and canonical_bytes(cfg.get('source_config'))==canonical_bytes(expected)
        and rr in record.parents and len(parents)==4
        and sum(x.stage=='p12_skeleton_input' and x.kind=='raw_cache' for x in parents)==1
        and sum(x.stage=='labels_'+pop and x.kind=='supervised_labels' and x.phase==record.phase for x in parents)==1
        and sum(x.stage=='class_prior' and x.phase==record.phase for x in parents)==1)


def train_skeleton(protocol,phase,cache,context,*,device='cuda'):
    from .visual_teacher import _labels
    from .visual_student_v3 import prior,_label_record,_save_torch
    from .visual_student import phase_lock
    recipe,rr,_=load_reconstruction(protocol,context);cfg=dict(recipe['source_models']['skeleton']['configuration']);cfg.update(num_workers=0,val_num_workers=0)
    pop='train12' if phase=='select' else 'refit14';inputs=load_stage_inputs(protocol,pop,phase,context=context);index,y=_labels(inputs,phase,protocol)
    cr,cache_valid,paths=cache_index(cache,protocol,context);positions={s:i for i,s in enumerate(cr.rows.sample_ids)}
    valid=np.array([cache_valid[positions[s]] for s in index.sample_ids]);chosen=RowIndex(tuple(s for s,v in zip(index.sample_ids,valid) if v),tuple(u for u,v in zip(index.user_ids,valid) if v),index.class_ids)
    labels=dict(zip(index.sample_ids,y.tolist()));ops=source_p12_ops(protocol,context)
    dataset=SkeletonDataset(chosen,{s:paths[s] for s in chosen.sample_ids},cfg,ops,augment=True,labels={s:labels[s] for s in chosen.sample_ids})
    root=protocol.run_root/REVISION/'p12_skeleton'/phase;registry=ArtifactRegistry(protocol,context=context)
    identity=canonical_hash({'phase':phase,'recipe':rr,'cache':cache,'population':chosen,'labels':context.hash_file(inputs.labels),'producer':context.hash_file(Path(__file__))})
    with phase_lock(root):
        immutable_bytes(root/'identity.json',canonical_bytes({'identity':identity})+b'\n')
        if (root/'artifact.json').exists():return ref(json.loads((root/'artifact.json').read_text(encoding='utf-8')))
        model=initialize_skeleton(cfg,ops,device=device);loader=ops['create_loader'](dataset,cfg,True)
        enc=list(model.skeleton.parameters());enc_ids={id(x) for x in enc};head=[x for x in model.parameters() if id(x) not in enc_ids]
        optimizer=torch.optim.AdamW([{'params':enc,'lr':cfg['learning_rate'],'name':'encoders'},{'params':head,'lr':cfg['learning_rate'],'name':'fusion'}],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=15);scaler=torch.amp.GradScaler('cuda',enabled=torch.device(device).type=='cuda')
        criterion=torch.nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing']);history=[]
        for epoch in range(1,16):
            tick=time.perf_counter();lr=optimizer.param_groups[0]['lr']
            measured=ops['run_epoch'](model,loader,criterion,torch.device(device),['skeleton'],optimizer,scaler,
                torch.device(device).type=='cuda',float(cfg['max_grad_norm']))
            if not np.isfinite(measured['loss']):raise ValueError('non-finite source Skeleton loss')
            scheduler.step();entry={'epoch':epoch,'train_loss':measured['loss'],'train_accuracy':measured['accuracy'],
                'train_samples':len(measured['sample_ids']),'learning_rate_used':lr,'seconds':round(time.perf_counter()-tick,2)}
            history.append(entry);write_json(root/'history.json',history)
            _save_torch(root/'checkpoint.pt',{'model_state_dict':model.state_dict(),'config':cfg,'epoch':epoch,'reconstruction_revision':REVISION})
            print(json.dumps({'event':'p12_skeleton_epoch','phase':phase,**entry}),flush=True)
        pr,prob=prior(protocol,phase,context);lrref=_label_record(inputs,phase,index,protocol,context)
        metadata={'revision':REVISION,'role':'skeleton','fit_partition':pop,'budget_epochs':15,'checkpoint_selection':'fixed_last_epoch',
            'fit_rows':len(chosen.sample_ids),'sampled_rows_per_epoch':len(loader)*cfg['batch_size'],'source_drop_last':True,
            'source_config':recipe['source_models']['skeleton']['configuration'],'effective_workers':0,
            'worker_override_reason':'single_process_cache_and_source_function_imports_on_Windows','recipe_sha256':canonical_hash(recipe),
            'prior':prob.tolist(),'fit_ids_sha256':canonical_hash(chosen),'historical_exact_reproduction':False}
        write_json(root/'model.json',metadata)
        result=registry.register(stage='source_expert/p12_skeleton',kind='supervised_model',phase=phase,
            files=[root/'checkpoint.pt',root/'model.json',root/'history.json'],parents=[rr,cache,lrref,pr],rows=index,fit_users=inputs.partition.users,
            config=metadata,source_files=[Path(__file__)])
        write_json(root/'artifact.json',result);return result


def predict_skeleton(protocol,model,cache,partition,context,*,device='cuda'):
    from .imu_rf_teacher import _write_npz
    registry=ArtifactRegistry(protocol,context=context);mr=registry.verify(model,'source_expert/p12_skeleton');phase='select' if partition in {'train12','development2'} else 'refit'
    if mr.phase!=phase:raise ValueError('Skeleton model prediction phase mismatch')
    data,valid,paths=cache_index(cache,protocol,context);index=row_index(read_public_rows(protocol.public_manifests[partition]));lookup={s:i for i,s in enumerate(data.rows.sample_ids)}
    valid=np.array([valid[lookup[s]] for s in index.sample_ids]);chosen=RowIndex(tuple(s for s,v in zip(index.sample_ids,valid) if v),tuple(u for u,v in zip(index.user_ids,valid) if v),index.class_ids)
    ops=source_p12_ops(protocol,context);network=initialize_skeleton(dict(mr.config['source_config']),ops,device=device)
    file=next(Path(x) for x in mr.files if Path(x).name=='checkpoint.pt');registry.verify_file(mr,file)
    state=torch.load(file,map_location='cpu',weights_only=True);network.load_state_dict(state['model_state_dict']);network.eval()
    dataset=SkeletonDataset(chosen,{s:paths[s] for s in chosen.sample_ids},mr.config['source_config'],ops,augment=False)
    scores=np.zeros((len(index.sample_ids),40),np.float32);positions={s:i for i,s in enumerate(index.sample_ids)}
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=32,num_workers=0):
            with torch.autocast(device_type=torch.device(device).type,enabled=torch.device(device).type=='cuda'):
                values=network({'skeleton':batch['skeleton'].to(device)})
            scores[[positions[s] for s in batch['sample_id']]]=values.float().cpu().numpy()
    scores,prob=complete_scores(scores,valid,mr.config['prior']);path=protocol.run_root/REVISION/'p12_skeleton'/phase/(partition+'_targets.npz')
    digest=_write_npz(path,dict(sample_ids=np.asarray(index.sample_ids),class_ids=np.arange(40),logits=scores,probabilities=prob,valid=valid))
    inputs=load_stage_inputs(protocol,partition,'predict',context=context)
    result=registry.register(stage='source_expert/p12_skeleton',kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
        parents=[model,cache,inputs.parents['manifest']],rows=index,config={'partition':partition,'revision':REVISION,'fit_phase':phase,
            'historical_exact_reproduction':False,'invalid_policy':'same_phase_complete_fit_prior'},source_files=[Path(__file__)])
    write_json(path.with_suffix('.artifact.json'),result);return Prediction(index,scores,prob,valid,result)


def run_skeleton(protocol,*,device='cuda'):
    context=VerificationContext(protocol);recipe,rr,_=load_reconstruction(protocol,context)
    caches={pop:build_skeleton_cache(protocol,pop,context) for pop in ['refit14','final4']}
    models={phase:train_skeleton(protocol,phase,caches['refit14'],context,device=device) for phase in ['select','refit']}
    outputs={}
    for pop in protocol.partitions:
        phase='select' if pop in {'train12','development2'} else 'refit';cache=caches['final4' if pop=='final4' else 'refit14']
        prediction=predict_skeleton(protocol,models[phase],cache,pop,context,device=device);outputs[pop]=prediction.artifact
        print(json.dumps({'event':'p12_skeleton_predict','partition':pop,'rows':len(prediction.valid),'valid':int(prediction.valid.sum())}),flush=True)
    body={'revision':REVISION,'base_protocol_sha256':protocol.identity(),'skeleton_component_complete':True,'recipe':rr,
        'models':models,'caches':caches,'outputs':outputs,'final_labels_read':False,'task11_complete':False,'a8_generated':False}
    immutable_bytes(protocol.run_root/'protocol'/REVISION/'skeleton_component.json',canonical_bytes(body)+b'\n');return body
