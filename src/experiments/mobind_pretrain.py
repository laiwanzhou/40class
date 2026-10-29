"""Fixed24 source MoBind training with phase-bound A1/RF teachers."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import json
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from scipy.special import softmax

from .artifact_record import ArtifactRegistry,VerificationContext
from .motion_cache import MOTION_FIELDS,BOOLEAN_MOTION_FIELDS,motion_item,load_motion,_source_descriptor
from .no_vote_types import write_json,ArtifactRef,Prediction,TeacherTargets,RowIndex,canonical_bytes,canonical_hash
from .no_vote_cache_import import immutable_bytes
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .teammate_source import verify_teammate_source,load_teammate_symbol

SOURCE_FILES=('train_p86_mobind_pretrain.py','p86_mobind_lite_model.py',
    'p86_mobind_lite_data.py','p86_cached_motion_data.py','train_p86_visual_student_oof.py')
TASK8_REVISION='ir_supervision_v2'
TRAINING_POPULATION_POLICY='motion_or_valid_visual_supervision'


def motion_output_root(protocol,*,revision=TASK8_REVISION):
    if revision is None:return protocol.run_root/'A5'
    if revision!=TASK8_REVISION:raise ValueError('unknown Task8 revision')
    return protocol.run_root/'A5'/revision


def motion_capability_path(protocol,revision):
    motion_output_root(protocol,revision=revision)
    name='task8_motion_capability.json' if revision is None else 'task8_'+revision+'_motion_capability.json'
    return protocol.run_root/'protocol'/name


def motion_ready_path(protocol):
    return protocol.run_root/'protocol'/('task8_'+TASK8_REVISION+'_artifacts_ready.json')


def source_motion_math(protocol,context):
    key=(protocol.identity(),'task8_source_math')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        loss=load_teammate_symbol(report,'train_p86_mobind_pretrain','losses')
        trainer=sys.modules['train_p86_mobind_pretrain'];dataset=sys.modules['p86_mobind_lite_data']
        context.public_rows[key]={'losses':loss,'model_class':trainer.P86MoBindLite,'class_weights':trainer.class_weights,
            'seed_all':trainer.seed_all,'motion_fields':trainer.motion_fields,'to_device':trainer.to_device,
            'temporal_indices':dataset.P86MoBindMotionDataset.temporal_indices,'dataset_class':dataset.P86MoBindMotionDataset}
    return context.public_rows[key]


def motion_args(protocol):
    recipe=dict(protocol.recipe['motion_pretrain'])
    if recipe.get('fixed_epochs')!=24 or dict(recipe.get('epochs',{}))!={'min':24,'max':24,'select_on':'fixed_source_budget'}:
        raise ValueError('source motion requires fixed24')
    if recipe.get('imu_instance_normalization') or recipe.get('freeze_skeleton_teacher') or recipe.get('domain_adversarial_weight',0):
        raise ValueError('source motion flags drift')
    recipe.update(epochs=24,all_label_refit=True,final_refit=False,subject_holdout_users=None,max_train_batches=0,max_eval_batches=0)
    return SimpleNamespace(**recipe)


def initialize_motion(protocol,*,device='cuda',context=None):
    context=context or VerificationContext(protocol);ops=source_motion_math(protocol,context);args=motion_args(protocol)
    ops['seed_all'](args.seed)
    return ops['model_class'](width=args.width,alignment_width=args.alignment_width,dropout=args.dropout,
        classes=40,imu_instance_normalization=False,imu_event_feature_width=0,domain_classes=0,
        domain_reversal_scale=args.domain_reversal_scale).to(device)


def source_motion_losses(output,batch,weights,args,ops):
    # Preserve source CE/KD reductions for partially missing modalities.
    return ops['losses'](output,batch,weights,args)


class NoVoteMotionDataset(Dataset):
    """Source tensor casting/augmentation with explicit optional supervision."""
    def __init__(self,arrays,index,*,rows=None,labels=None,visual_teacher=None,rf_teacher=None,temporal_indices=None):
        self.arrays=arrays;self.index=index;self.rows=rows or index;self.labels=labels;self.temporal_indices=temporal_indices
        lookup={s:i for i,s in enumerate(index.sample_ids)}
        if any(s not in lookup or index.user_ids[lookup[s]]!=u for s,u in zip(self.rows.sample_ids,self.rows.user_ids)):
            raise ValueError('motion Dataset sample/user mismatch')
        self.positions=[lookup[s] for s in self.rows.sample_ids]
        if labels is not None and set(labels)!=set(self.rows.sample_ids):raise ValueError('motion Dataset label IDs mismatch')
        self.visual_teacher=visual_teacher;self.rf_teacher=rf_teacher
        for prefix,teacher in [('visual',visual_teacher),('rf',rf_teacher)]:
            if teacher is None:continue
            ti=teacher.prediction.index;mapping={s:i for i,s in enumerate(ti.sample_ids)}
            if any(s not in mapping or ti.user_ids[mapping[s]]!=u for s,u in zip(self.rows.sample_ids,self.rows.user_ids)):
                raise ValueError('motion teacher IDs/users mismatch')
            setattr(self,prefix+'_positions',[mapping[s] for s in self.rows.sample_ids])
        if visual_teacher is not None:
            if visual_teacher.features is None:raise ValueError('visual motion teacher features required')
            # Source Dataset casts before taking the view mean; FP16 mean is not equivalent.
            self.teacher_features=np.asarray(visual_teacher.features,dtype=np.float32).mean(axis=2)

    def __len__(self):return len(self.positions)

    def __getitem__(self,item):
        position=self.positions[item];index=self.temporal_indices(16) if self.temporal_indices else np.arange(16)
        motion=motion_item({k:self.arrays[k][position] for k in MOTION_FIELDS},temporal_index=index)
        result={k:torch.from_numpy(v) for k,v in motion.items()}
        sid=self.rows.sample_ids[item];result.update(sample_id=sid,user_id=self.rows.user_ids[item],cache_index=position)
        if self.labels is not None:result['label']=torch.tensor(self.labels[sid],dtype=torch.long)
        if self.visual_teacher is not None:
            i=self.visual_positions[item]
            result['teacher_logits']=torch.from_numpy(np.asarray(self.visual_teacher.prediction.logits[i],np.float32).copy())
            result['teacher_features']=torch.from_numpy(self.teacher_features[i].copy())
        if self.rf_teacher is not None:
            i=self.rf_positions[item];valid=bool(self.rf_teacher.prediction.valid[i])
            result['imu_teacher_valid']=torch.tensor(valid)
            values=np.asarray(self.rf_teacher.prediction.logits[i],np.float32).copy() if valid else np.zeros(40,np.float32)
            result['imu_teacher_logits']=torch.from_numpy(values)
        return result


def model_config(args):
    return {'width':args.width,'alignment_width':args.alignment_width,'classes':40,'dropout':args.dropout,
        'imu_instance_normalization':False,'imu_event_feature_width':0,'domain_classes':0,'domain_reversal_scale':args.domain_reversal_scale}


def run_motion_epochs(model,loader,labels,args,root,*,device,ops,identity,loss_function=source_motion_losses,epoch_callback=None):
    from .visual_student_v3 import _save_torch
    root=Path(root);root.mkdir(parents=True,exist_ok=True);device=torch.device(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    weights=ops['class_weights'](labels,args.class_weight_power,device);history=[];start=1
    resume=root/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise ValueError('motion epoch resume identity drift')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];start=saved['epoch']+1
        torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng'])
        nr=saved['numpy_rng'];np.random.set_state((nr[0],nr[1].numpy().astype(np.uint32),nr[2],nr[3],nr[4]))
        if saved['cuda_rng']:torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for epoch in range(start,args.epochs+1):
        model.train();ratio=args.minimum_learning_rate/args.learning_rate+.5*(1-args.minimum_learning_rate/args.learning_rate)*(1+math.cos(math.pi*(epoch-1)/max(args.epochs-1,1)))
        for group in optimizer.param_groups:group['lr']=args.learning_rate*ratio
        sums={};samples=0;tc=tn=lc=ln=0.;started=time.perf_counter()
        for batch in loader:
            batch=ops['to_device'](batch,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                output=model(ops['motion_fields'](batch),mask_ratio=args.mask_ratio)
                values=loss_function(output,batch,weights,args,ops)
            if not torch.isfinite(values['loss']):raise ValueError('non-finite source motion loss')
            scaler.scale(values['loss']).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.);scaler.step(optimizer);scaler.update()
            count=len(batch['label']);samples+=count
            for k,v in values.items():
                if k not in {'token_correct','token_count','local_correct','local_count'}:sums[k]=sums.get(k,0.)+float(v.detach())*count
            tc+=float(values.get('token_correct',0));tn+=float(values.get('token_count',0))
            lc+=float(values.get('local_correct',0));ln+=float(values.get('local_count',0))
        if samples!=len(loader.dataset):raise ValueError('motion fixed budget dropped training rows')
        record={'epoch':epoch,'learning_rate':optimizer.param_groups[0]['lr'],**{'train_'+k:v/max(samples,1) for k,v in sums.items()},
            'train_samples':samples,'train_token_retrieval':tc/max(tn,1.),'train_part_retrieval':lc/max(ln,1.),'seconds':round(time.perf_counter()-started,2)}
        history.append(record);write_json(root/'history.json',history)
        _save_torch(root/'checkpoint.pt',{'stage':'A5_fixed24','model_state':model.state_dict(),'model_config':model_config(args),'fixed_epochs':24})
        nr=np.random.get_state();_save_torch(resume,{'identity':identity,'epoch':epoch,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'history':history,'torch_rng':torch.get_rng_state(),
            'python_rng':random.getstate(),'numpy_rng':(nr[0],torch.from_numpy(nr[1].astype(np.int64)),nr[2],nr[3],nr[4]),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []})
        print(json.dumps({'event':'motion_epoch','phase':root.name,**record}),flush=True)
        if epoch_callback:epoch_callback(epoch)
    if len(history)!=24 or history[-1]['epoch']!=24:raise ValueError('motion fixed24 budget incomplete')
    return history


def _ref(value):
    if isinstance(value,(str,Path)):value=json.loads(Path(value).read_text(encoding='utf-8'))
    return ArtifactRef(Path(value['record_path']),value['sha256'])


def _pins(protocol,context):
    key=(protocol.identity(),'task8_source_pins')
    if key not in context.public_rows:
        manifest=json.loads((protocol.source_root.parent/'source_manifest.json').read_text(encoding='utf-8'))
        pins={x['path']:x['sha256'] for x in manifest['files']}
        context.public_rows[key]={'aligned_multimodal/'+name:context.hash_file(protocol.source_root/'aligned_multimodal'/name,pins['aligned_multimodal/'+name]) for name in SOURCE_FILES}
    return context.public_rows[key]


def motion_references(protocol):
    body=json.loads((protocol.run_root/'protocol/task6_artifacts_ready.json').read_text(encoding='utf-8'))
    return {pop:{k:_ref(v) for k,v in values.items()} for pop,values in body['partitions'].items()}


def load_motion_teachers(protocol,partition,context):
    from .visual_student_v3 import load_teacher
    visual=load_teacher(protocol,partition,context)
    body=json.loads((protocol.run_root/'protocol/task7_artifacts_ready.json').read_text(encoding='utf-8'))
    ref=_ref(body['outputs'][partition]['fit_teacher_target']);registry=ArtifactRegistry(protocol,context=context)
    index=row_index(read_public_rows(protocol.public_manifests[partition]));record=registry.verify(ref,'source_expert/imu_rf','predict',index)
    phase='select' if partition in {'train12','development2'} else 'refit'
    if record.config.get('consumer_profile')!='fit_teacher_target' or record.config.get('fit_phase')!=phase:
        raise ValueError('motion RF requires own-phase fit_teacher_target profile')
    files=[Path(x) for x in record.files if Path(x).suffix=='.npz']
    if len(files)!=1:raise ValueError('one RF motion teacher NPZ required')
    arrays=context.load_arrays(registry.verify_file(record,files[0]))
    rf=TeacherTargets(Prediction(index,arrays['logits'],arrays['probabilities'],arrays['valid'],ref),None)
    return visual,rf


def validate_transform(protocol,motion,transform,context):
    record=ArtifactRegistry(protocol,context=context).verify(transform,'source_transform','public')
    body=json.loads(next(Path(x) for x in record.files).read_text(encoding='utf-8'))
    if (body.get('external_transform')!='identity' or body.get('cross_sample_fit') is not False
            or body.get('imu_instance_normalization') is not False or _ref(body['motion_record'])!=motion):
        raise ValueError('motion source transform binding mismatch')


def prepare_motion_capability(protocol,context):
    source_motion_math(protocol,context);motion_args(protocol);refs=motion_references(protocol);proofs={}
    registry=ArtifactRegistry(protocol,context=context)
    for task in [4,6,7]:
        path=protocol.run_root/f'protocol/task{task}_complete.json';body=json.loads(path.read_text(encoding='utf-8'))
        if body.get('complete') is not True or body.get('protocol_sha256')!=protocol.identity():raise ValueError('motion prerequisite completion mismatch')
        proofs[str(task)]=context.hash_file(path)
    teachers={}
    for phase,pop in [('select','train12'),('refit','refit14')]:
        visual,rf=load_motion_teachers(protocol,pop,context)
        teachers[phase]={'visual':visual.prediction.artifact,'rf':rf.prediction.artifact}
    for pop in ['refit14','final4']:
        registry.verify(refs[pop]['windows'],'motion_windows','raw',row_index(read_public_rows(protocol.public_manifests[pop])))
        validate_transform(protocol,refs[pop]['windows'],refs[pop]['normalization'],context)
    body={'schema_version':3,'scope':'task8_motion_pretrain_only','protocol_sha256':protocol.identity(),'stages':['A5'],
        'budget_epochs':24,'source_files':_pins(protocol,context),'completion_digests':proofs,'teachers':teachers,
        'public_motion':{pop:{k:refs[pop][k] for k in ['windows','normalization']} for pop in ['refit14','final4']},
        'global_formal_generation_ready':False,'oof_enabled':False,'training_executed':False,
        'task8_revision':TASK8_REVISION,'training_population_policy':TRAINING_POPULATION_POLICY}
    path=motion_capability_path(protocol,TASK8_REVISION);immutable_bytes(path,canonical_bytes(body)+b'\n')
    ref=registry.register(stage='motion_readiness',kind='descriptor',phase='public',files=[path],
        parents=[_source_descriptor(protocol,context),refs['refit14']['windows'],refs['refit14']['normalization'],refs['final4']['windows'],refs['final4']['normalization']],
        config={'source_contract_version':3,'task8_revision':TASK8_REVISION},source_files=[Path(__file__)])
    write_json(path.with_suffix('.artifact.json'),ref);return ref


def validate_motion_capability(registry,record):
    p=registry.protocol;revision=record.config.get('task8_revision');path=motion_capability_path(p,revision)
    if list(record.files)!=[str(path)] or len(record.parents)!=5:raise ValueError('motion readiness payload/parents mismatch')
    parents=[registry.read(x) for x in record.parents]
    if [x.stage for x in parents]!=['source_ops','motion_windows','source_transform','motion_windows','source_transform'] or any(x.kind not in {'descriptor','raw_cache'} for x in parents):
        raise ValueError('motion readiness cannot share learned ancestors')
    body=json.loads(path.read_text(encoding='utf-8'));c=registry.context or VerificationContext(p)
    expected={'schema_version':3,'scope':'task8_motion_pretrain_only','protocol_sha256':p.identity(),'stages':['A5'],
        'budget_epochs':24,'source_files':_pins(p,c),'completion_digests':{str(t):c.hash_file(p.run_root/f'protocol/task{t}_complete.json') for t in [4,6,7]},
        'public_motion':{pop:{'windows':record.parents[1+2*i],'normalization':record.parents[2+2*i]} for i,pop in enumerate(['refit14','final4'])},
        'global_formal_generation_ready':False,'oof_enabled':False,'training_executed':False}
    if revision is not None:expected.update(task8_revision=revision,training_population_policy=TRAINING_POPULATION_POLICY)
    if any(canonical_bytes(body.get(k))!=canonical_bytes(v) for k,v in expected.items()):raise ValueError('motion readiness binding drift')
    if set(body.get('teachers',{}))!={'select','refit'}:raise ValueError('motion teacher bindings incomplete')
    for phase,pop in [('select','train12'),('refit','refit14')]:
        teachers=body['teachers'][phase]
        for key,stage in [('visual','A1'),('rf','source_expert/imu_rf')]:
            r=registry.read(_ref(teachers[key]))
            if r.stage!=stage or r.kind!='predictions' or r.config.get('partition')!=pop:raise ValueError('motion bound teacher phase mismatch')


def allows_motion_stage(registry,record):
    if record.stage!='A5' or record.kind!='supervised_model' or record.config.get('budget_epochs')!=24 or record.phase not in {'select','refit'}:return False
    caps=[x for x in record.parents if registry.read(x).stage=='motion_readiness']
    if len(caps)!=1:return False
    cap=registry.read(caps[0]);validate_motion_capability(registry,cap)
    body=json.loads(next(Path(x) for x in cap.files).read_text(encoding='utf-8'))
    expected=[_ref(body['teachers'][record.phase][k]) for k in ['visual','rf']]
    expected+=[_ref(body['public_motion']['refit14'][k]) for k in ['windows','normalization']]
    if record.config.get('task8_revision')!=body.get('task8_revision'):return False
    if body.get('task8_revision') is not None and record.config.get('training_population_policy')!=TRAINING_POPULATION_POLICY:return False
    return all(x in record.parents for x in expected)


def motion_validity(arrays):
    n=len(arrays['skeleton_joint_mask'])
    sk=np.asarray(arrays['skeleton_joint_mask']).reshape(n,-1).any(1)
    imu=np.asarray(arrays['imu_global_mask'])[:,:,0].any(1)|np.asarray(arrays['imu_sequence_mask']).reshape(n,-1).any(1)
    return sk,imu


def motion_training_mask(skeleton_valid,imu_valid,visual_teacher_valid):
    masks=[np.asarray(x) for x in [skeleton_valid,imu_valid,visual_teacher_valid]]
    if any(x.ndim!=1 or x.dtype!=np.bool_ or x.shape!=masks[0].shape for x in masks):raise ValueError('training mask axes/dtype mismatch')
    return masks[0]|masks[1]|masks[2]


def train_motion_student(phase,fit,motion,transform,visual_teacher,rf_teacher,capability,*,protocol,device='cuda',context=None):
    required={'select':'train12','refit':'refit14'}
    if phase not in required or fit.partition.name!=required[phase]:raise ValueError('motion fit phase/population mismatch')
    from .visual_teacher import _labels
    from .visual_student import phase_lock
    context=context or VerificationContext(protocol);registry=ArtifactRegistry(protocol,context=context)
    index,y=_labels(fit,phase,protocol);cap=registry.verify(capability,'motion_readiness','public')
    body=json.loads(next(Path(x) for x in cap.files).read_text(encoding='utf-8'))
    if body.get('task8_revision')!=TASK8_REVISION:raise ValueError('new training requires corrected Task8 revision')
    if (visual_teacher.prediction.artifact!=_ref(body['teachers'][phase]['visual']) or rf_teacher.prediction.artifact!=_ref(body['teachers'][phase]['rf'])
            or motion!=_ref(body['public_motion']['refit14']['windows']) or transform!=_ref(body['public_motion']['refit14']['normalization'])):
        raise ValueError('motion inputs differ from own-phase bound source teachers')
    validate_transform(protocol,motion,transform,context);record,arrays=load_motion(motion,protocol=protocol,context=context)
    lookup={s:i for i,s in enumerate(record.rows.sample_ids)}
    if any(s not in lookup or record.rows.user_ids[lookup[s]]!=u for s,u in zip(index.sample_ids,index.user_ids)):raise ValueError('motion fit exact ID/user mismatch')
    positions=np.array([lookup[s] for s in index.sample_ids]);sk,imu=motion_validity(arrays)
    if visual_teacher.prediction.index!=index or rf_teacher.prediction.index!=index:raise ValueError('motion supervision population/index mismatch')
    valid=motion_training_mask(sk[positions],imu[positions],visual_teacher.prediction.valid)
    restored=visual_teacher.prediction.valid & ~(sk|imu)[positions]
    selected=RowIndex(tuple(s for s,v in zip(index.sample_ids,valid) if v),tuple(u for u,v in zip(index.user_ids,valid) if v),index.class_ids)
    if not len(selected.sample_ids):raise ValueError('no effective motion fit inputs')
    labels=dict(zip(index.sample_ids,y.tolist()));ops=source_motion_math(protocol,context);args=motion_args(protocol)
    dataset=NoVoteMotionDataset(arrays,record.rows,rows=selected,labels={s:labels[s] for s in selected.sample_ids},
        visual_teacher=visual_teacher,rf_teacher=rf_teacher,temporal_indices=ops['temporal_indices'])
    root=motion_output_root(protocol)/phase;identity=canonical_hash({'protocol':protocol.identity(),'phase':phase,'fit':fit,
        'motion':motion,'transform':transform,'visual':visual_teacher.prediction.artifact,'rf':rf_teacher.prediction.artifact,
        'capability':capability,'producer':context.hash_file(Path(__file__)),'labels':context.hash_file(fit.labels)})
    with phase_lock(root):
        immutable_bytes(root/'identity.json',canonical_bytes({'identity':identity})+b'\n')
        if (root/'artifact.json').exists():
            ref=_ref(root/'artifact.json');registry.verify(ref,'A5',phase,index);return ref
        model=initialize_motion(protocol,device=device,context=context)
        loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=torch.device(device).type=='cuda',drop_last=False)
        history=run_motion_epochs(model,loader,y[valid],args,root,device=device,ops=ops,identity=identity)
        config={'budget_epochs':24,'checkpoint_epoch':24,'checkpoint_selection':'fixed_last_epoch','partition':required[phase],
            'model_config':model_config(args),'valid_fit_rows':len(selected.sample_ids),'unavailable_fit_rows_excluded':int((~valid).sum()),
            'fit_ids_sha256':canonical_hash(selected),'prior':(np.bincount(y,minlength=40)/len(y)).tolist(),
            'teacher_origin':'own_phase_calibrated_early_late_and_rf_fit_teacher_target','fresh_random_initialization':True,
            'external_transform':'identity','imu_instance_normalization':False,'task8_revision':TASK8_REVISION,
            'training_population_policy':TRAINING_POPULATION_POLICY,'restored_visual_only_rows':int(restored.sum()),
            'restored_visual_only_ids_sha256':canonical_hash(tuple(s for s,v in zip(index.sample_ids,restored) if v)),
            'class_weight_population_ids_sha256':canonical_hash(selected)}
        write_json(root/'model.json',config)
        label_ref=registry.register(stage='labels_'+fit.partition.name,kind='supervised_labels',phase=phase,files=[fit.labels],
            parents=[fit.parents['manifest']],rows=index,config={'partition':fit.partition.name})
        ref=registry.register(stage='A5',kind='supervised_model',phase=phase,files=[root/'checkpoint.pt',root/'model.json',root/'history.json'],
            parents=[motion,transform,visual_teacher.prediction.artifact,rf_teacher.prediction.artifact,label_ref,capability],
            fit_users=fit.partition.users,rows=index,config=config,source_files=[Path(__file__)])
        write_json(root/'artifact.json',ref);return ref


def combine_motion_probabilities(skeleton,imu,sk_valid,imu_valid,prior):
    count=sk_valid.astype(np.int64)+imu_valid.astype(np.int64)
    probabilities=(skeleton*sk_valid[:,None]+imu*imu_valid[:,None])/np.maximum(count[:,None],1)
    probabilities[count==0]=prior;return probabilities


def predict_motion_student(model,motion,transform,partition,*,protocol,context,device='cuda'):
    from .imu_rf_teacher import _write_npz
    registry=ArtifactRegistry(protocol,context=context);mr=registry.verify(model,'A5')
    phase='select' if partition in {'train12','development2'} else 'refit'
    if partition not in protocol.partitions or phase!=mr.phase:raise ValueError('motion prediction phase/population mismatch')
    validate_transform(protocol,motion,transform,context);record,arrays=load_motion(motion,protocol=protocol,context=context)
    inputs=load_stage_inputs(protocol,partition,'predict',context=context);index=row_index(read_public_rows(inputs.public_manifest))
    root=motion_output_root(protocol,revision=mr.config.get('task8_revision'))/phase
    paths={kind:root/(partition+'_'+kind+'_targets.npz') for kind in ['skeleton','imu','combined']}
    if all(path.with_suffix('.artifact.json').exists() for path in paths.values()):
        results={}
        for kind,path in paths.items():
            stage={'skeleton':'A5-S','imu':'A5-I','combined':'A5'}[kind];ref=_ref(path.with_suffix('.artifact.json'))
            r=registry.verify(ref,stage,'predict',index)
            if model not in r.parents or motion not in r.parents:raise ValueError('motion prediction parents mismatch')
            a=context.load_arrays(registry.verify_file(r,path));results[kind]=Prediction(index,a['logits'],a['probabilities'],a['valid'],ref)
        return results
    if any(path.exists() for path in paths.values()):raise ValueError('partial motion predictions require explicit recovery')
    key=(protocol.identity(),'motion_inference',model.sha256,device)
    if key not in context.public_rows:
        network=initialize_motion(protocol,device='cpu',context=context)
        path=next(Path(x) for x in mr.files if Path(x).name=='checkpoint.pt');registry.verify_file(mr,path)
        state=torch.load(path,map_location='cpu',weights_only=True);network.load_state_dict(state['model_state'],strict=True)
        network.eval().to(device);context.public_rows[key]=network
    network=context.public_rows[key];dataset=NoVoteMotionDataset(arrays,record.rows,rows=index);prior=np.asarray(mr.config['prior'],np.float32)
    sk_logits=np.tile(np.log(np.maximum(prior,1e-12)),(len(dataset),1));imu_logits=sk_logits.copy();sk_valid=np.zeros(len(dataset),bool);imu_valid=sk_valid.copy();offset=0
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=protocol.recipe['motion_pretrain']['batch_size'],shuffle=False,num_workers=0):
            batch=source_motion_math(protocol,context)['to_device'](batch,torch.device(device));n=len(batch['sample_id'])
            s=batch['skeleton_joint_mask'].flatten(1).any(1);i=batch['imu_global_mask'][:,:,0].any(1)|batch['imu_sequence_mask'].flatten(1).any(1)
            active=s|i;positions=torch.nonzero(active,as_tuple=False).flatten()
            if len(positions):
                fields={k:batch[k].index_select(0,positions) for k in MOTION_FIELDS}
                with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,enabled=torch.device(device).type=='cuda'):
                    out=network(fields,mask_ratio=0.)
                ids=offset+positions.cpu().numpy();ss=s.index_select(0,positions).cpu().numpy();ii=i.index_select(0,positions).cpu().numpy()
                sk_logits[ids[ss]]=out['skeleton_logits'].float().cpu().numpy()[ss];imu_logits[ids[ii]]=out['imu_logits'].float().cpu().numpy()[ii]
            sk_valid[offset:offset+n]=s.cpu().numpy();imu_valid[offset:offset+n]=i.cpu().numpy();offset+=n
    sk_probs=softmax(sk_logits,axis=1);imu_probs=softmax(imu_logits,axis=1);sk_probs[~sk_valid]=prior;imu_probs[~imu_valid]=prior
    combined=combine_motion_probabilities(sk_probs,imu_probs,sk_valid,imu_valid,prior);results={}
    for kind,scores,prob,valid in [('skeleton',sk_logits,sk_probs,sk_valid),('imu',imu_logits,imu_probs,imu_valid),('combined',np.log(np.maximum(combined,1e-12)),combined,sk_valid|imu_valid)]:
        path=paths[kind];digest=_write_npz(path,{'sample_ids':np.asarray(index.sample_ids),'class_ids':np.arange(40),'logits':scores,'probabilities':prob,'valid':valid})
        stage={'skeleton':'A5-S','imu':'A5-I','combined':'A5'}[kind]
        ref=registry.register(stage=stage,kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
            parents=[model,motion,transform,inputs.parents['manifest']],rows=index,
            config={'partition':partition,'branch':kind,'phase':phase,'combined_role':'diagnostic_available_mean_not_source_learned_head' if kind=='combined' else 'source_branch',
                'invalid_policy':'same_phase_fit_prior','task8_revision':mr.config.get('task8_revision')},source_files=[Path(__file__)])
        write_json(path.with_suffix('.artifact.json'),ref);results[kind]=Prediction(index,scores,prob,valid,ref)
    return results


def run_motion_pipeline(protocol,*,device='cuda'):
    from .visual_teacher import _labels
    context=VerificationContext(protocol);cap=prepare_motion_capability(protocol,context);refs=motion_references(protocol);models={}
    for phase,pop in [('select','train12'),('refit','refit14')]:
        visual,rf=load_motion_teachers(protocol,pop,context);fit=load_stage_inputs(protocol,pop,phase,context=context)
        models[phase]=train_motion_student(phase,fit,refs['refit14']['windows'],refs['refit14']['normalization'],visual,rf,cap,
            protocol=protocol,device=device,context=context)
    outputs={};metrics={}
    for pop in protocol.partitions:
        phase='select' if pop in {'train12','development2'} else 'refit';cache='final4' if pop=='final4' else 'refit14'
        outputs[pop]=predict_motion_student(models[phase],refs[cache]['windows'],refs[cache]['normalization'],pop,protocol=protocol,context=context,device=device)
        if pop!='final4':
            inputs=load_stage_inputs(protocol,pop,phase,context=context);index,y=_labels(inputs,phase,protocol);metrics[pop]={}
            for kind,prediction in outputs[pop].items():
                correct=prediction.logits.argmax(1)==y;metrics[pop][kind]={'correct':int(correct.sum()),'total':len(y),
                    'accuracy':float(correct.mean()),'valid':int(prediction.valid.sum()),'valid_correct':int(correct[prediction.valid].sum())}
    write_json(motion_output_root(protocol)/'development_metrics.json',metrics)
    body={'protocol_sha256':protocol.identity(),'complete':True,'budget_epochs':24,'models':models,'readiness':cap,
        'outputs':{pop:{kind:pred.artifact for kind,pred in results.items()} for pop,results in outputs.items()},
        'final_labels_read':False,'task9_started':False,'combined_role':'diagnostic_available_mean_not_source_learned_head',
        'task8_revision':TASK8_REVISION,'training_population_policy':TRAINING_POPULATION_POLICY}
    immutable_bytes(motion_ready_path(protocol),canonical_bytes(body)+b'\n')
    print(json.dumps({'event':'motion_pretrain_pipeline_complete','metrics':metrics,'final_accuracy':None}),flush=True)
    return body
