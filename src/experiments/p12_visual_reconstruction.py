"""Source image teachers with public, label-optional input adapters."""
import ast
import importlib.util
import inspect
import csv
import json
from pathlib import Path
import sys
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .p12_reconstruction import source_p12_ops,load_reconstruction,REVISION,ref,complete_scores
from .source_teacher_bank import pinned_source
from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_types import RowIndex,Prediction,canonical_hash,canonical_bytes,write_json
from .no_vote_cache_import import immutable_bytes
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index


def _without_label(function):
    """Remove the one output label expression; keep source pixel operations intact."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)));removed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Dict):
            keep=[]
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='label':
                    value=node.values[i]
                    if not isinstance(value,ast.Attribute) or value.attr!='class_id':raise ValueError('unexpected source label expression')
                    removed+=1
                else:keep.append(i)
            node.keys=[node.keys[i] for i in keep];node.values=[node.values[i] for i in keep]
    if removed!=1:raise ValueError('source public adapter must remove exactly one label expression')
    namespace=dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree),function.__code__.co_filename,'exec',flags=__import__('__future__').annotations.compiler_flag),namespace)
    return namespace[function.__name__]


def visual_source_ops(protocol,context):
    key=(protocol.identity(),'p12_visual_reconstruction_source')
    if key in context.public_rows:return context.public_rows[key]
    base=source_p12_ops(protocol,context);modules={};pins=[]
    for name in ['thermal_model','thermal_oof_data','thermal_tsm_model','train_imagenet_oof']:
        _,pin=pinned_source(protocol,'thermal_baseline/'+name+'.py',context);pins.append(pin)
        qualified='_p12_reconstruction_'+name+'_'+pin['sha256'][:12]
        if qualified in sys.modules:module=sys.modules[qualified]
        else:
            spec=importlib.util.spec_from_file_location(qualified,protocol.source_root/pin['path'])
            module=importlib.util.module_from_spec(spec);sys.modules[qualified]=module
            previous={n:sys.modules.get(n) for n in modules}
            old_bytecode=sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode=True
                sys.modules.update(modules);spec.loader.exec_module(module)
            finally:
                sys.dont_write_bytecode=old_bytecode
                for n,value in previous.items():
                    if value is None:sys.modules.pop(n,None)
                    else:sys.modules[n]=value
        modules[name]=module
    aligned=sys.modules['aligned_data'].AlignedMultimodalDataset
    thermal=modules['thermal_oof_data'].ThermalOOFDataset
    ops={**base,'aligned_model_class':base['model_class'],'aligned_dataset_class':aligned,
        'thermal_dataset_class':thermal,'thermal_model_class':modules['thermal_tsm_model'].ThermalResNetTSM,
        'thermal_train':modules['train_imagenet_oof'],
        'public_depth_getitem':_without_label(aligned._getitem_from_cache),
        'public_thermal_getitem':_without_label(thermal.__getitem__),'source_pins':pins}
    context.public_rows[key]=ops;return ops


class _DepthFrames:
    def __init__(self,paths,height,width):self.paths=paths;self.height=height;self.width=width
    def __getitem__(self,positions):
        frames={}
        for i in set(np.asarray(positions).tolist()):
            with Image.open(self.paths[i]) as image:
                image=image.convert('RGB').resize((self.width,self.height),resample=Image.Resampling.BILINEAR)
                frames[i]=np.asarray(image).copy()
        return np.stack([frames[i] for i in positions])


class SourceImageDataset(Dataset):
    def __init__(self,rows,descriptors,role,config,ops,*,augment,labels=None):
        if role not in {'depth','thermal'}:raise ValueError('unsupported P12 image role')
        self.rows=rows;self.descriptors=descriptors;self.role=role;self.ops=ops;self.labels=labels
        ids=[r['sample_id'] for r in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(descriptors):raise ValueError('exact unique image IDs required')
        if labels is not None and set(labels)!=set(ids):raise ValueError('image label IDs mismatch')
        self.samples=[]
        for row in rows:
            sid=row['sample_id'];attrs={'sample_id':sid,'user_id':row['user_id']}
            if labels is not None:attrs['class_id']=int(labels[sid])
            if role=='thermal':attrs['trial_dir']=Path(descriptors[sid]['thermal_dir'])
            self.samples.append(SimpleNamespace(**attrs))
        reader=object.__new__(ops['thermal_dataset_class' if role=='thermal' else 'aligned_dataset_class'])
        reader.samples=self.samples;reader.num_frames=int(config['num_frames']);reader.image_height=int(config['image_height']);reader.image_width=int(config['image_width'])
        reader.augment=bool(augment);reader.normalization=config.get('normalization','legacy')
        reader.temporal_view=.5;reader.force_horizontal_flip=False;reader.temporal_sampling='uniform'
        reader.modalities=('depth',);reader.depth_representation=config.get('depth_representation','jet_rgb');reader.visual_normalization='legacy'
        if role=='depth':
            reader.cache_locations={s:(0,len(d['depth_frames'])) for s,d in descriptors.items()}
            reader._cache_array=lambda modality: self.current_frames
        self.reader=reader
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        sample=self.samples[i]
        if self.role=='depth':
            self.current_frames=_DepthFrames(self.descriptors[sample.sample_id]['depth_frames'],self.reader.image_height,self.reader.image_width)
            fn=self.ops['public_depth_getitem'] if self.labels is None else self.ops['aligned_dataset_class']._getitem_from_cache
            result=fn(self.reader,sample)
        else:
            fn=self.ops['public_thermal_getitem'] if self.labels is None else self.ops['thermal_dataset_class'].__getitem__
            result=fn(self.reader,i)
        result['user_id']=sample.user_id
        return result


def initialize_image_model(role,config,ops,*,device='cuda',initializer=None):
    ops['seed'](int(config['seed']))
    if role=='depth':
        model=ops['aligned_model_class'](['depth'],dropout=float(config['dropout']),
            use_layer3_spatial=bool(config.get('use_layer3_spatial',False)),imagenet_pretrained=False)
    elif role=='thermal':
        model=ops['thermal_model_class'](num_classes=40,dropout=float(config['dropout']),imagenet_pretrained=False)
        if initializer is not None:
            state=torch.load(initializer,map_location='cpu',weights_only=True)
            model.stem[0].load_state_dict({'weight':state['conv1.weight']})
            model.stem[1].load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('bn1.')})
            for name in ['layer1','layer2','layer3','layer4']:
                getattr(model,name).load_state_dict({k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+'.')})
    else:raise ValueError('unsupported P12 image role')
    return model.to(device)


def build_image_index(protocol,partition,role,context):
    recipe,rr,receipt=load_reconstruction(protocol,context);ops=visual_source_ops(protocol,context)
    inputs=load_stage_inputs(protocol,partition,'raw',context=context);rows=read_public_rows(inputs.public_manifest);index=row_index(rows)
    root=protocol.run_root/REVISION/('p12_'+role+'_input')/partition;alias=root/'artifact.json'
    registry=ArtifactRegistry(protocol,context=context)
    if alias.exists():return ref(json.loads(alias.read_text(encoding='utf-8')))
    descriptors={};valid=[];parents=[rr,inputs.parents['manifest']];extra={};tick=time.perf_counter()
    if role=='thermal':
        table=receipt['thermal_public_sidecars'][partition];path=Path(table['path']);context.hash_file(path,table['sha256'])
        with path.open(encoding='utf-8-sig',newline='') as stream:thermal=list(csv.DictReader(stream))
        if [(r['sample_id'],r['user_id']) for r in thermal]!=[(r['sample_id'],r['user_id']) for r in rows]:raise ValueError('Thermal public input IDs differ')
        copied=root/'thermal_public.csv';immutable_bytes(copied,path.read_bytes());extra={'sidecar_sha256':table['sha256'],'observations_sha256':table['observations_sha256']}
    for n,row in enumerate(rows):
        sid=row['sample_id'];descriptor=None
        if role=='depth':
            if all(row[k+'_available']=='1' and row[k+'_path'] and Path(row[k+'_path']).is_dir() for k in ['ir','depth','skeleton']):
                maps={k:ops['frame_map'](Path(row[k+'_path']),k) for k in ['ir','depth','skeleton']}
                common=sorted(set.intersection(*(set(x) for x in maps.values())))
                if common:descriptor={'depth_frames':[str(maps['depth'][k]) for k in common]}
        elif role=='thermal':
            entry=thermal[n]
            if entry['thermal_available']=='1':descriptor={'thermal_dir':entry['thermal_path']}
        else:raise ValueError('unsupported image role')
        valid.append(descriptor is not None)
        if descriptor is not None:descriptors[sid]=descriptor
        if (n+1)%200==0:print(json.dumps({'event':'p12_image_index','role':role,'partition':partition,'done':n+1,'seconds':round(time.perf_counter()-tick,1)}),flush=True)
    payload=root/'index.json';immutable_bytes(payload,canonical_bytes({'sample_ids':index.sample_ids,'user_ids':index.user_ids,
        'valid':valid,'descriptors':descriptors,'role':role,'recipe_sha256':canonical_hash(recipe),**extra})+b'\n')
    files=[payload]+([copied] if role=='thermal' else [])
    result=registry.register(stage='p12_'+role+'_input',kind='raw_cache',phase='raw',files=files,parents=parents,rows=index,
        config={'partition':partition,'revision':REVISION,'valid_rows':sum(valid),'role':role,'raw_payload_full_scan':False},source_files=[Path(__file__)])
    write_json(alias,result);return result


def image_index(protocol,reference,role,context):
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(reference,'p12_'+role+'_input','raw')
    path=next(Path(x) for x in record.files if Path(x).name=='index.json');registry.verify_file(record,path)
    body=json.loads(path.read_text(encoding='utf-8'));valid=np.asarray(body['valid'],bool)
    if (tuple(body['sample_ids'])!=record.rows.sample_ids or tuple(body['user_ids'])!=record.rows.user_ids
        or valid.shape!=(len(record.rows.sample_ids),) or set(body['descriptors'])!={s for s,v in zip(record.rows.sample_ids,valid) if v}):
        raise ValueError('image index/valid/descriptor mismatch')
    return record,valid,body['descriptors']


def image_initializer(protocol,context):
    """Bind the cached public torchvision initializer once, without changing old YAML."""
    recipe,rr,_=load_reconstruction(protocol,context);root=protocol.run_root/REVISION/'p12_thermal_initializer';alias=root/'artifact.json'
    registry=ArtifactRegistry(protocol,context=context)
    if alias.exists():
        reference=ref(json.loads(alias.read_text(encoding='utf-8')));record=registry.verify(reference,'reconstructed_imagenet_initializer','raw')
        path=next(Path(x) for x in record.files if Path(x).suffix=='.pth');registry.verify_file(record,path);return reference,path
    source=Path(torch.hub.get_dir())/'checkpoints'/'resnet18-f37072fd.pth';digest=context.hash_file(source)
    if not digest.startswith('f37072fd'):raise ValueError('unexpected official ResNet18 V1 weight digest')
    root.mkdir(parents=True,exist_ok=True);dest=root/source.name
    if not dest.exists():
        if source.drive.lower()==dest.drive.lower():__import__('os').link(source,dest)
        else:__import__('shutil').copyfile(source,dest)
    context.hash_file(dest,digest)
    receipt=root/'receipt.json';immutable_bytes(receipt,canonical_bytes({'enum':'ResNet18_Weights.IMAGENET1K_V1',
        'origin':'https://download.pytorch.org/models/resnet18-f37072fd.pth','sha256':digest,
        'local_source':str(source),'recipe':rr,'historical_training_initializer_verified':False})+b'\n')
    result=registry.register(stage='reconstructed_imagenet_initializer',kind='raw_cache',phase='raw',files=[dest,receipt],parents=[rr],
        config={'revision':REVISION,'role':'thermal','enum':'ResNet18_Weights.IMAGENET1K_V1','sha256':digest},source_files=[Path(__file__)])
    write_json(alias,result);return result,dest


def allows_image_reconstruction_stage(registry,record):
    if record.stage not in {'source_expert/p12_depth','source_expert/p12_thermal'} or record.kind!='supervised_model' or record.phase not in {'select','refit'}:return False
    role=record.stage.removeprefix('source_expert/p12_');cfg=record.config
    if cfg.get('revision')!=REVISION or cfg.get('budget_epochs')!=15:return False
    if cfg.get('storage_export')=='float16_roundtrip':
        if len(record.parents)!=2 or cfg.get('new_training_executed') is not False:return False
        parents=[registry.read(x) for x in record.parents];masters=[x for x in parents if x.stage==record.stage and x.kind==record.kind and x.phase==record.phase and not x.config.get('storage_export')]
        return (len(masters)==1 and any(x.stage=='reconstruction_recipe' for x in parents) and masters[0].rows==record.rows
            and masters[0].config['recipe_sha256']==cfg.get('recipe_sha256') and masters[0].config['fit_partition']==cfg.get('fit_partition'))
    pop='train12' if record.phase=='select' else 'refit14'
    if cfg.get('fit_partition')!=pop or len(record.parents)!=(5 if role=='thermal' else 4):return False
    p=registry.protocol;c=registry.context or VerificationContext(p);recipe,rr,_=load_reconstruction(p,c)
    if cfg.get('recipe_sha256')!=canonical_hash(recipe) or canonical_bytes(cfg.get('source_config'))!=canonical_bytes(recipe['source_models'][role]['configuration']):return False
    parents=[registry.read(x) for x in record.parents]
    required=[('reconstruction_recipe','raw_cache','raw'),('p12_'+role+'_input','raw_cache','raw'),('labels_'+pop,'supervised_labels',record.phase),('class_prior','statistics',record.phase)]
    if role=='thermal':required.append(('reconstructed_imagenet_initializer','raw_cache','raw'))
    return rr in record.parents and all(sum((x.stage,x.kind,x.phase)==r for x in parents)==1 for r in required)


def _image_dataset(protocol,reference,pop,role,cfg,ops,context,*,augment,labels=None):
    record,valid,descriptors=image_index(protocol,reference,role,context);rows=read_public_rows(protocol.public_manifests[pop]);lookup=dict(zip(record.rows.sample_ids,valid))
    if any(r['sample_id'] not in lookup for r in rows):raise ValueError('population missing from image cache')
    kept=[r for r in rows if lookup[r['sample_id']]]
    ds=SourceImageDataset(kept,{r['sample_id']:descriptors[r['sample_id']] for r in kept},role,cfg,ops,augment=augment,
        labels=None if labels is None else {r['sample_id']:labels[r['sample_id']] for r in kept})
    return ds,np.array([lookup[r['sample_id']] for r in rows],bool)


def fit_image_epochs(network,loader,optimizer,scheduler,scaler,criterion,source,role,phase,cfg,root,identity,*,device='cuda',fail_after=None):
    from .visual_student_v3 import _save_torch
    from .mobind_fusion import _rng_state,_restore_rng
    root=Path(root);root.mkdir(parents=True,exist_ok=True);history=[];resume=root/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise ValueError('image resume identity mismatch')
        network.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];loader.sampler.generator.set_state(saved['sampler_rng']);_restore_rng(saved)
        if [x['epoch'] for x in history]!=list(range(1,len(history)+1)) or len(history)>15:raise ValueError('image resume budget/history drift')
        print(json.dumps({'event':'p12_image_resume','role':role,'phase':phase,'completed_epochs':len(history)}),flush=True)
    for epoch in range(len(history)+1,16):
        tick=time.perf_counter();lr=optimizer.param_groups[0]['lr']
        args=[network,loader,criterion,torch.device(device)]
        if role=='depth':args.append(['depth'])
        measured=source.run_epoch(*args,optimizer,scaler,torch.device(device).type=='cuda',float(cfg['max_grad_norm']))
        if not np.isfinite(measured['loss']):raise ValueError('non-finite source image loss')
        scheduler.step();entry={'epoch':epoch,'train_loss':measured['loss'],'train_accuracy':measured['accuracy'],
            'train_samples':len(measured['sample_ids']),'learning_rate_used':lr,'seconds':round(time.perf_counter()-tick,2)}
        history.append(entry)
        _save_torch(resume,{'identity':identity,'model_state':network.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'history':history,'sampler_rng':loader.sampler.generator.get_state(),**_rng_state()})
        write_json(root/'history.json',history)
        print(json.dumps({'event':'p12_image_epoch','role':role,'phase':phase,**entry}),flush=True)
        if fail_after==epoch:raise RuntimeError('simulated interruption after complete image epoch')
    return history


def train_image(protocol,phase,reference,role,context,*,device='cuda'):
    from .visual_teacher import _labels
    from .visual_student_v3 import prior,_label_record,_save_torch
    from .visual_student import phase_lock
    from .mobind_fusion import _rng_state,_restore_rng
    recipe,rr,_=load_reconstruction(protocol,context);source_cfg=dict(recipe['source_models'][role]['configuration']);cfg={**source_cfg,'num_workers':0,'val_num_workers':0}
    pop='train12' if phase=='select' else 'refit14';inputs=load_stage_inputs(protocol,pop,phase,context=context);index,y=_labels(inputs,phase,protocol)
    ops=visual_source_ops(protocol,context);ds,valid=_image_dataset(protocol,reference,pop,role,cfg,ops,context,augment=True,labels=dict(zip(index.sample_ids,y.tolist())))
    root=protocol.run_root/REVISION/('p12_'+role)/phase;registry=ArtifactRegistry(protocol,context=context)
    initializer=None;init_ref=None
    if role=='thermal':init_ref,initializer=image_initializer(protocol,context)
    identity=canonical_hash({'role':role,'phase':phase,'recipe':rr,'cache':reference,'labels':context.hash_file(inputs.labels),
        'producer':context.hash_file(Path(__file__)),'initializer':init_ref,'fit_ids':[r['sample_id'] for r in ds.rows]})
    with phase_lock(root):
        immutable_bytes(root/'identity.json',canonical_bytes({'identity':identity})+b'\n')
        if (root/'artifact.json').exists():
            result=ref(json.loads((root/'artifact.json').read_text(encoding='utf-8')));registry.verify(result);return result
        network=initialize_image_model(role,cfg,ops,device=device,initializer=initializer)
        if torch.device(device).type=='cuda':torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
        source=ops['thermal_train'] if role=='thermal' else sys.modules['train'];loader=source.create_loader(ds,cfg,True)
        if role=='depth':
            enc=list(network.visual.parameters());ids={id(v) for v in enc};head=[v for v in network.parameters() if id(v) not in ids]
            params=[{'params':enc,'lr':cfg['learning_rate'],'name':'encoders'},{'params':head,'lr':cfg['learning_rate'],'name':'fusion'}]
        else:params=network.parameters()
        optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=15);scaler=torch.amp.GradScaler('cuda',enabled=torch.device(device).type=='cuda')
        criterion=torch.nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
        history=fit_image_epochs(network,loader,optimizer,scheduler,scaler,criterion,source,role,phase,cfg,root,identity,device=device)
        _save_torch(root/'checkpoint.pt',{'model_state_dict':network.state_dict(),'config':cfg,'epoch':15,'reconstruction_revision':REVISION})
        pr,prob=prior(protocol,phase,context);lrref=_label_record(inputs,phase,index,protocol,context)
        meta={'revision':REVISION,'role':role,'fit_partition':pop,'budget_epochs':15,'checkpoint_selection':'fixed_last_epoch',
            'fit_rows':len(ds),'sampled_rows_per_epoch':len(loader)*cfg['batch_size'],'source_drop_last':True,'source_config':source_cfg,
            'effective_workers':0,'worker_override_reason':'single_process_source_adapter_on_Windows','recipe_sha256':canonical_hash(recipe),
            'prior':prob.tolist(),'fit_ids_sha256':canonical_hash([r['sample_id'] for r in ds.rows]),'historical_exact_reproduction':False,
            'initializer':init_ref,'source_pins':ops['source_pins']}
        write_json(root/'model.json',meta);parents=[rr,reference,lrref,pr]+([init_ref] if init_ref else [])
        result=registry.register(stage='source_expert/p12_'+role,kind='supervised_model',phase=phase,files=[root/'checkpoint.pt',root/'model.json',root/'history.json'],
            parents=parents,rows=index,fit_users=inputs.partition.users,config=meta,source_files=[Path(__file__)])
        write_json(root/'artifact.json',result);return result


def export_image_model(protocol,master,role,context):
    from .visual_student_v3 import _save_torch
    from .p12_skeleton_export import fp16_state
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(master,'source_expert/p12_'+role)
    _,rr,_=load_reconstruction(protocol,context);root=protocol.run_root/REVISION/('p12_'+role+'_fp16')/record.phase
    if (root/'artifact.json').exists():
        result=ref(json.loads((root/'artifact.json').read_text(encoding='utf-8')));registry.verify(result);return result
    file=next(Path(x) for x in record.files if Path(x).name=='checkpoint.pt');registry.verify_file(record,file)
    state=torch.load(file,map_location='cpu',weights_only=True);state['model_state_dict']=fp16_state(state['model_state_dict'])
    state.update(storage_dtype='float16',execution_dtype='float32_with_source_amp');root.mkdir(parents=True,exist_ok=True);_save_torch(root/'checkpoint.pt',state)
    cfg={**record.config,'storage_export':'float16_roundtrip','new_training_executed':False,'master_model':master,'master_checkpoint_preserved':True}
    write_json(root/'model.json',cfg)
    result=registry.register(stage=record.stage,kind='supervised_model',phase=record.phase,files=[root/'checkpoint.pt',root/'model.json'],parents=[rr,master],
        rows=record.rows,fit_users=record.fit_users,config=cfg,source_files=[Path(__file__)])
    write_json(root/'artifact.json',result);return result


def predict_image(protocol,model,reference,partition,role,context,*,device='cuda'):
    from .imu_rf_teacher import _write_npz
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(model,'source_expert/p12_'+role)
    phase='select' if partition in {'train12','development2'} else 'refit'
    if record.phase!=phase:raise ValueError('image prediction phase mismatch')
    ops=visual_source_ops(protocol,context);ds,valid=_image_dataset(protocol,reference,partition,role,record.config['source_config'],ops,context,augment=False)
    network=initialize_image_model(role,record.config['source_config'],ops,device=device)
    file=next(Path(x) for x in record.files if Path(x).name=='checkpoint.pt');registry.verify_file(record,file)
    network.load_state_dict(torch.load(file,map_location='cpu',weights_only=True)['model_state_dict']);network.eval()
    index=row_index(read_public_rows(protocol.public_manifests[partition]));scores=np.zeros((len(index.sample_ids),40),np.float32);positions={s:i for i,s in enumerate(index.sample_ids)}
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(ds,batch_size=32,num_workers=0):
            with torch.autocast(device_type=torch.device(device).type,enabled=torch.device(device).type=='cuda'):
                value=network(batch['clip'].to(device) if role=='thermal' else {'depth':batch['depth'].to(device)})
            scores[[positions[s] for s in batch['sample_id']]]=value.float().cpu().numpy()
    scores,prob=complete_scores(scores,valid,record.config['prior']);path=protocol.run_root/REVISION/('p12_'+role+'_fp16')/phase/(partition+'_targets.npz')
    digest=_write_npz(path,dict(sample_ids=np.asarray(index.sample_ids),class_ids=np.arange(40),logits=scores,probabilities=prob,valid=valid))
    inputs=load_stage_inputs(protocol,partition,'predict',context=context)
    result=registry.register(stage=record.stage,kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},parents=[model,reference,inputs.parents['manifest']],rows=index,
        config={'partition':partition,'revision':REVISION,'fit_phase':phase,'historical_exact_reproduction':False,'invalid_policy':'same_phase_complete_fit_prior'},source_files=[Path(__file__)])
    write_json(path.with_suffix('.artifact.json'),result);return Prediction(index,scores,prob,valid,result)


def run_images(protocol,*,device='cuda',roles=('depth','thermal')):
    torch.set_num_threads(4)
    context=VerificationContext(protocol);_,rr,_=load_reconstruction(protocol,context)
    for role in roles:
        caches={pop:build_image_index(protocol,pop,role,context) for pop in ['refit14','final4']};masters={};models={};outputs={}
        for phase in ['select','refit']:
            masters[phase]=train_image(protocol,phase,caches['refit14'],role,context,device=device)
            models[phase]=export_image_model(protocol,masters[phase],role,context)
        for pop in protocol.partitions:
            phase='select' if pop in {'train12','development2'} else 'refit';cache=caches['final4' if pop=='final4' else 'refit14']
            output=predict_image(protocol,models[phase],cache,pop,role,context,device=device);outputs[pop]=output.artifact
            print(json.dumps({'event':'p12_image_predict','role':role,'partition':pop,'rows':len(output.valid),'valid':int(output.valid.sum())}),flush=True)
        body={'revision':REVISION,'role':role,'base_protocol_sha256':protocol.identity(),'recipe':rr,'masters':masters,'models':models,'caches':caches,'outputs':outputs,
            'component_complete':True,'task11_complete':False,'final_labels_read':False,'a8_generated':False}
        path=protocol.run_root/'protocol'/REVISION/(role+'_fp16_component.json');immutable_bytes(path,canonical_bytes(body)+b'\n')
        write_json(path.parent/('p12_'+role+'_current.json'),{'component_path':str(path),'sha256':context.hash_file(path)})
