"""Source-recipe MC3, fixed 16 epochs and same-phase cached sequence export."""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_cache_import import immutable_bytes
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .no_vote_types import ArtifactRef,Prediction,TeacherTargets,Selection,RowIndex,canonical_bytes,canonical_hash,write_json
from .teammate_source import verify_teammate_source,load_teammate_symbol,sha256_file
from .visual_teacher_v3 import read_ref,_table

SOURCE_FILES=('train_p86_visual_pixel_oof.py','train_p86_visual_student_oof.py','p86_mc3_visual_model.py','p86_visual_pixel_data.py')


def source_student_math(protocol,context):
    key=(protocol.identity(),'source_student_math')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        symbols={'class_weights':('train_p86_visual_student_oof','class_weights'),
            'relation_loss':('train_p86_visual_student_oof','relation_loss'),
            'learning_rate_scale':('train_p86_visual_pixel_oof','learning_rate_scale'),
            'optimizer_for':('train_p86_visual_pixel_oof','optimizer_for'),
            'seed_all':('train_p86_visual_pixel_oof','seed_all'),
            'model_class':('p86_mc3_visual_model','P86MC3VisualStudent'),
            'dataset_class':('p86_visual_pixel_data','P86VisualPixelDataset')}
        context.public_rows[key]={k:load_teammate_symbol(report,*v) for k,v in symbols.items()}
    return context.public_rows[key]


def fixed_budget(recipe):
    if (recipe.get('fixed_epochs')!=16 or dict(recipe.get('epochs',{}))!=
            {'min':16,'max':16,'select_on':'fixed_source_budget'}):
        raise ValueError('source student requires the fixed 16 epoch budget')
    return 16


def checkpoint_epoch(validation_best_epoch,budget):
    # Validation ranking never controls the saved state or refit budget.
    if budget!=16:raise ValueError('student budget must be 16')
    return budget


def source_hybrid_loss(output,batch,weights,recipe,relation_loss):
    if (recipe['mode']!='hybrid' or recipe['feature_weight_effective']!=0
            or recipe['stage_distillation_weight']!=0):raise ValueError('source hybrid loss flags mismatch')
    if not bool(batch['teacher_valid'].all()):raise ValueError('source student trains only usable IR rows')
    ce=F.cross_entropy(output['logits'],batch['label'],weight=weights,label_smoothing=recipe['label_smoothing'])
    temperature=recipe['distillation_temperature']
    kd=F.kl_div(F.log_softmax(output['logits']/temperature,dim=1),
        F.softmax(batch['teacher_logits']/temperature,dim=1),reduction='batchmean')*temperature**2
    relation=relation_loss(output['clip_embeddings'],batch['teacher_features'],output['clip_mask'])
    zero=output['clip_embeddings'].sum()*0.
    return {'loss':ce+recipe['distillation_weight']*kd+recipe['relation_weight']*relation,
        'ce':ce,'kd':kd,'relation':relation,'feature':zero,'stage_kd':output['logits'].sum()*0.}


def pixel_refs(protocol):
    body=json.loads((protocol.run_root/'protocol/task2_prepared.json').read_text(encoding='utf-8'))
    return {x['partition']:ArtifactRef(Path(x['record_path']),x['sha256']) for x in body['imports'] if x['stage']=='pixels'}


def teacher_ref(protocol,partition):
    phase='select' if partition in {'train12','development2'} else 'refit'
    return read_ref(protocol.run_root/'A1'/phase/(partition+'_targets.artifact.json'))


def require_teacher(registry,ref,partition):
    p=registry.protocol
    if p.recipe.get('source_contract_version')!=3:raise ValueError('student requires v3 Task4 targets')
    record=registry.verify(ref,'A1','predict',row_index(read_public_rows(p.public_manifests[partition])))
    phase='select' if partition in {'train12','development2'} else 'refit'
    population='train12' if phase=='select' else 'refit14'
    models=[registry.read(x) for x in record.parents if registry.read(x).stage=='A1' and registry.read(x).kind=='supervised_model']
    if (record.config.get('partition')!=partition or len(models)!=1 or models[0].phase!=phase
            or models[0].config.get('primary_family')!='early_late'
            or models[0].config.get('calibration_partition')!=population
            or models[0].config.get('target_origin')!='fixed_split_fit_predictions_not_oof'
            or not float(models[0].config.get('head_temperature',0))>0):
        raise ValueError('student requires own-phase calibrated v3 Task4 early_late targets')
    return record


def public_initializer(protocol,context):
    path=protocol.weights['mc3'];receipt_path=protocol.run_root/'protocol/weights_manifest.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))['weights']['mc3']
    digests={str(Path(x['path']).resolve()):x['sha256'] for x in receipt['files']}
    return ArtifactRegistry(protocol,context=context).register(stage='visual_student_initializer',kind='public_weights',phase='public',
        files=[path],file_digests=digests,config={'role':'mc3','validation':'inherited_pinned_public_receipt',
            'receipt_sha256':context.hash_file(receipt_path)},source_files=[Path(__file__)])


def prepare_student_capability(protocol,context):
    if protocol.recipe.get('source_contract_version')!=3:raise ValueError('Task5 requires v3 Task4')
    complete=protocol.run_root/'protocol/task4_complete.json'
    if not complete.exists():raise ValueError('Task4 complete marker required')
    proof=json.loads(complete.read_text(encoding='utf-8'))
    if (not proof.get('complete') or proof.get('protocol_sha256')!=protocol.identity()
            or proof.get('independent_artifact_review',{}).get('heads')!='GO'
            or proof.get('independent_artifact_review',{}).get('fullwindow')!='GO'):
        raise ValueError('Task4 complete/independent review mismatch')
    fixed_budget(protocol.recipe['visual_student']);registry=ArtifactRegistry(protocol,context=context)
    refs=pixel_refs(protocol);targets={}
    for phase,partition in [('select','train12'),('refit','refit14')]:
        ref=teacher_ref(protocol,partition);require_teacher(registry,ref,partition)
        targets[phase]={'record_path':str(ref.record_path),'sha256':ref.sha256}
    for partition in ['refit14','final4']:
        registry.verify(refs[partition],'pixels','raw',row_index(read_public_rows(protocol.public_manifests[partition])))
    source_student_math(protocol,context)
    source_manifest=json.loads((protocol.source_root.parent/'source_manifest.json').read_text(encoding='utf-8'))
    pins={x['path']:x['sha256'] for x in source_manifest['files']}
    hashes={}
    for name in SOURCE_FILES:
        rel='aligned_multimodal/'+name;hashes[rel]=context.hash_file(protocol.source_root/rel,pins[rel])
    from .visual_teacher_v3 import capability as visual_capability
    initializer=public_initializer(protocol,context)
    body={'schema_version':3,'protocol_sha256':protocol.identity(),'scope':'task5_student_only','stages':['A2'],
        'fixed_epochs':16,'training_executed':False,'oof_enabled':False,'task4_complete_sha256':context.hash_file(complete),
        'source_files':hashes,'teacher_targets':targets,'global_formal_generation_ready':False}
    path=protocol.run_root/'protocol/task5_student_capability.json'
    immutable_bytes(path,canonical_bytes(body)+b'\n')
    # Learned targets are phase-specific direct parents of each actual A2 model,
    # not shared descriptor ancestors (which would mix select/refit lineages).
    ref=registry.register(stage='student_readiness',kind='descriptor',phase='public',files=[path],
        parents=[visual_capability(protocol,context),refs['refit14'],refs['final4'],initializer],
        config={'source_contract_version':3},source_files=[Path(__file__)])
    immutable_bytes(path.with_suffix('.artifact.json'),canonical_bytes(ref)+b'\n');return ref


def validate_student_capability(registry,record):
    p=registry.protocol;path=p.run_root/'protocol/task5_student_capability.json'
    if list(record.files)!=[str(path)] or len(record.parents)!=4:raise ValueError('invalid student capability payload/parents')
    parents=[registry.read(x) for x in record.parents]
    if ([x.stage for x in parents]!=['visual_readiness','pixels','pixels','visual_student_initializer']
            or any(x.kind not in {'descriptor','raw_cache','public_weights'} for x in parents)):
        raise ValueError('student capability cannot share learned ancestors')
    body=json.loads(path.read_text(encoding='utf-8'));complete=p.run_root/'protocol/task4_complete.json'
    digest=registry.context.hash_file(complete) if registry.context else sha256_file(complete)
    if (body.get('schema_version')!=3 or body.get('protocol_sha256')!=p.identity() or body.get('scope')!='task5_student_only'
            or body.get('stages')!=['A2'] or body.get('fixed_epochs')!=16 or body.get('training_executed') is not False
            or body.get('oof_enabled') is not False or body.get('task4_complete_sha256')!=digest
            or body.get('global_formal_generation_ready') is not False):raise ValueError('student readiness binding drift')
    manifest=json.loads((p.source_root.parent/'source_manifest.json').read_text(encoding='utf-8'))
    pins={x['path']:x['sha256'] for x in manifest['files']}
    if body.get('source_files')!={'aligned_multimodal/'+x:pins['aligned_multimodal/'+x] for x in SOURCE_FILES}:
        raise ValueError('student source pins mismatch')
    if set(body.get('teacher_targets',{}))!={'select','refit'}:raise ValueError('student teacher bindings missing')


def allows_student_stage(registry,record):
    if record.stage!='A2' or record.kind!='supervised_model' or record.config.get('budget_epochs')!=16:return False
    caps=[x for x in record.parents if registry.read(x).stage=='student_readiness']
    if len(caps)!=1:return False
    cap=registry.read(caps[0]);validate_student_capability(registry,cap)
    body=json.loads(next(Path(x) for x in cap.files).read_text(encoding='utf-8'))
    target=body['teacher_targets'].get(record.phase,{})
    return any(x.sha256==target.get('sha256') and x.record_path==Path(target.get('record_path','')).resolve() for x in record.parents)


def capability(protocol,context):
    ref=read_ref(protocol.run_root/'protocol/task5_student_capability.artifact.json')
    ArtifactRegistry(protocol,context=context).verify(ref,'student_readiness','public');return ref


def initialize_student(protocol,*,device='cuda',public_init=True,context=None):
    context=context or VerificationContext(protocol);recipe=protocol.recipe['visual_student'];ops=source_student_math(protocol,context)
    ops['seed_all'](recipe['seed'])
    kwargs={k:recipe[k] for k in ['width','frames','dropout','fusion_mode','temporal_modeling','exact_time_modeling',
        'cross_view_time_modeling','spatial_region_modeling','region_temporal_modeling','structured_region_modeling','enable_distillation_projection']}
    model=ops['model_class'](classes=40,kinetics_pretrained=False,**kwargs);initializer=None
    if public_init:
        initializer=public_initializer(protocol,context)
        state=torch.load(protocol.weights['mc3'],map_location='cpu',weights_only=True)
        for name in ['stem','layer1','layer2','layer3','layer4']:
            block={key[len(name)+1:]:value for key,value in state.items() if key.startswith(name+'.')}
            getattr(model,name).load_state_dict(block,strict=True)
    model.freeze_low_level(recipe['freeze_through']);model.to(device)
    return model,initializer


def load_teacher(protocol,partition,context):
    registry=ArtifactRegistry(protocol,context=context);ref=teacher_ref(protocol,partition);record=require_teacher(registry,ref,partition)
    paths=[Path(x) for x in record.files if Path(x).suffix=='.npz']
    if len(paths)!=1:raise ValueError('one student teacher NPZ required')
    arrays=context.load_arrays(registry.verify_file(record,paths[0]))
    feature_refs=[x for x in record.parents if registry.read(x).stage=='visual_features']
    if len(feature_refs)!=1:raise ValueError('teacher feature source missing')
    _,features=_table(feature_refs[0],record.rows,protocol,context)
    for array in features.values():array.flags.writeable=False
    prediction=Prediction(record.rows,arrays['logits'],arrays['probabilities'],arrays['valid'],ref)
    return TeacherTargets.from_validated_features(prediction,features['features'])


def guard_teacher(teacher,inputs,*,protocol,context):
    expected=load_teacher(protocol,inputs.partition.name,context)
    if (teacher.prediction.artifact!=expected.prediction.artifact or teacher.prediction.index!=expected.prediction.index
            or teacher.features is not expected.features):raise ValueError('v3 teacher features/identity differ from Task4')
    for name in ['logits','probabilities','valid']:
        if not np.array_equal(getattr(teacher.prediction,name),getattr(expected.prediction,name)):
            raise ValueError('v3 teacher scores differ from Task4')


def _label_record(inputs,phase,index,protocol,context):
    return ArtifactRegistry(protocol,context=context).register(stage='labels_'+inputs.partition.name,kind='supervised_labels',phase=phase,
        files=[inputs.labels],parents=[inputs.parents['manifest']],rows=index,config={'partition':inputs.partition.name})


def prior(protocol,phase,context):
    ref=read_ref(protocol.run_root/'A1'/phase/'prior_artifact.json')
    record=ArtifactRegistry(protocol,context=context).verify(ref,'class_prior',phase)
    data=json.loads(next(Path(x) for x in record.files).read_text(encoding='utf-8'));values=np.asarray(data['probabilities'],np.float32)
    if values.shape!=(40,) or not np.isclose(values.sum(),1) or np.any(values<0):raise ValueError('student prior mismatch')
    return ref,values


def _save_torch(path,body):
    temp=path.with_suffix(path.suffix+'.building');torch.save(body,temp);temp.replace(path)


def _read_selection(path):
    body=json.loads(path.read_text(encoding='utf-8'))
    body['fit_artifact']=ArtifactRef(Path(body['fit_artifact']['record_path']),body['fit_artifact']['sha256'])
    return Selection(**body)


def train_visual_student(phase,fit,development,pixels,teacher,selection,*,protocol,device='cuda',context=None):
    from . import visual_student as base
    from .pixel_cache import load_pixels
    from .no_vote_datasets import NoVotePixelDataset
    from .visual_teacher import _labels
    context=context or VerificationContext(protocol);registry=ArtifactRegistry(protocol,context=context)
    recipe=protocol.recipe['visual_student'];epochs=fixed_budget(recipe);expected='train12' if phase=='select' else 'refit14'
    if phase not in {'select','refit'} or fit.partition.name!=expected:raise ValueError('student phase/population mismatch')
    if phase=='select' and (development is None or development.partition.name!='development2' or selection is not None):
        raise ValueError('source student select requires development2 metadata')
    if phase=='refit':
        if development is not None or selection is None or selection.budget.get('epochs')!=epochs:raise ValueError('student refit fixed budget mismatch')
        selected=registry.verify(selection.fit_artifact,'A2','select')
        dev=row_index(read_public_rows(protocol.public_manifests['development2']))
        if (dict(selection.config)!=dict(recipe) or selected.config.get('budget_epochs')!=epochs
                or selection.development_ids_sha256!=canonical_hash(dev)):raise ValueError('student selection/config mismatch')
    cap=capability(protocol,context);guard_teacher(teacher,fit,protocol=protocol,context=context)
    index,y=_labels(fit,phase,protocol);pixel_record,arrays=load_pixels(pixels,protocol=protocol,context=context)
    positions={sid:i for i,sid in enumerate(pixel_record.rows.sample_ids)}
    if not set(index.sample_ids)<=set(positions):raise ValueError('student fit pixels missing IDs')
    valid=np.array([arrays['view_valid'][positions[sid]].any() for sid in index.sample_ids])
    if not np.array_equal(valid,teacher.prediction.valid) or set(y[valid])!=set(range(40)):
        raise ValueError('student teacher/IR availability or class support mismatch')
    rows=RowIndex(tuple(s for s,v in zip(index.sample_ids,valid) if v),tuple(u for u,v in zip(index.user_ids,valid) if v),index.class_ids)
    labels=dict(zip(index.sample_ids,y.tolist()));ops=source_student_math(protocol,context)
    dataset=NoVotePixelDataset(arrays,pixel_record.rows,rows=rows,labels={s:labels[s] for s in rows.sample_ids},
        teacher=teacher,augmentation=ops['dataset_class']._subject_robust_augment)
    root=protocol.run_root/'A2'/phase
    identity=canonical_hash({'protocol':protocol.identity(),'phase':phase,'fit':fit,'pixels':pixels,'teacher':teacher.prediction.artifact,
        'selection':selection,'producer':sha256_file(Path(__file__)),'dataset_source':sha256_file(Path(__file__).with_name('no_vote_datasets.py')),
        'labels_sha256':context.hash_file(fit.labels),'student_readiness':cap})
    with base.phase_lock(root):
        immutable_bytes(root/'identity.json',canonical_bytes({'identity':identity})+b'\n')
        if (root/'artifact.json').exists():
            ref=read_ref(root/'artifact.json');registry.verify(ref,'A2',phase)
            return ref,_read_selection(root/'selection.json') if phase=='select' else None
        model,initializer=initialize_student(protocol,device=device,context=context)
        optimizer=ops['optimizer_for'](model,SimpleNamespace(**recipe))
        scaler=torch.amp.GradScaler('cuda',enabled=torch.device(device).type=='cuda')
        loader=DataLoader(dataset,batch_size=recipe['batch_size'],shuffle=True,num_workers=0,
            drop_last=len(dataset)%recipe['batch_size']==1)
        weights=ops['class_weights'](y[valid],recipe['class_weight_power'],torch.device(device))
        prior_ref,prob=prior(protocol,phase,context);history=[];start_epoch=1
        resume=root/'resume.pt'
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=True)
            if saved['identity']!=identity:raise ValueError('student complete-epoch resume identity drift')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
            history=saved['history'];start_epoch=saved['epoch']+1
            torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng'])
            nr=saved['numpy_rng'];np.random.set_state((nr[0],nr[1].numpy().astype(np.uint32),nr[2],nr[3],nr[4]))
            if saved['cuda_rng']:torch.cuda.set_rng_state_all(saved['cuda_rng'])
            _save_torch(root/'checkpoint.pt',model.state_dict())
        state_path=protocol.run_root/'run_state.json';state=json.loads(state_path.read_text(encoding='utf-8'))
        if state['state'] in {'frozen','revealed'}:raise ValueError('student generation after freeze/reveal')
        state['state']='generating';write_json(state_path,state)
        loss_fn=partial(source_hybrid_loss,relation_loss=ops['relation_loss'])
        for epoch in range(start_epoch,epochs+1):
            scale=ops['learning_rate_scale'](epoch,epochs,recipe['minimum_learning_rate']/recipe['head_learning_rate'])
            for group in optimizer.param_groups:group['lr']=group['initial_lr']*scale
            training=base.train_epoch(model,loader,optimizer,weights,recipe,device=device,scaler=scaler,loss_function=loss_fn)
            current={'epoch':epoch,'training':training,'backbone_learning_rate':optimizer.param_groups[0]['lr'],
                'head_learning_rate':optimizer.param_groups[1]['lr']}
            history.append(current);write_json(root/'history.json',history)
            _save_torch(root/'checkpoint.pt',model.state_dict())
            nr=np.random.get_state()
            saved={'identity':identity,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scaler':scaler.state_dict(),'history':history,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                'numpy_rng':(nr[0],torch.from_numpy(nr[1].astype(np.int64)),nr[2],nr[3],nr[4]),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.device(device).type=='cuda' else []}
            _save_torch(resume,saved)
            print(json.dumps({'event':'source_student_epoch','phase':phase,**current}),flush=True)
        if len(history)!=epochs or history[-1]['epoch']!=epochs:raise ValueError('student fixed budget did not complete')
        measured={}
        if phase=='select':
            dev,dy=_labels(development,'select',protocol)
            evaluation=NoVotePixelDataset(arrays,pixel_record.rows,rows=dev)
            scores,_,_,_=base.infer_arrays(model,evaluation,prob,device=device,batch_size=recipe['batch_size'])
            measured=base.metrics(dev,dy,scores)
            write_json(root/'development_metrics.json',{'epoch':16,**measured})
        config={'model_recipe':recipe,'budget_epochs':epochs,'checkpoint_epoch':16,'checkpoint_selection':'fixed_last_epoch',
            'partition':expected,'valid_fit_rows':len(rows.sample_ids),'valid_fit_ids_sha256':canonical_hash(rows),
            'unavailable_rows_excluded':int((~valid).sum()),'drop_singleton_per_epoch':len(dataset)%recipe['batch_size']==1,
            'teacher_target_origin':'v3_calibrated_early_late_in_sample_non_oof','public_reinitialization':True}
        write_json(root/'model.json',config)
        ref=registry.register(stage='A2',kind='supervised_model',phase=phase,files=[root/'checkpoint.pt',root/'model.json',root/'history.json'],
            parents=[initializer,pixels,teacher.prediction.artifact,prior_ref,_label_record(fit,phase,index,protocol,context),cap],
            fit_users=fit.partition.users,select_users=development.partition.users if phase=='select' else (),rows=index,config=config,
            source_files=[Path(__file__),Path(__file__).with_name('no_vote_datasets.py')])
        chosen=Selection('A2',recipe,{'epochs':16},measured,ref,canonical_hash(dev)) if phase=='select' else None
        if chosen is not None:chosen.write(root/'selection.json')
        write_json(root/'completion.json',{'identity':identity,'artifact':ref,'selection':chosen})
        write_json(root/'artifact.json',ref)
        return ref,chosen


def load_student(ref,*,protocol,device,context):
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(ref,'A2')
    if (record.phase not in {'select','refit'} or record.config.get('budget_epochs')!=16
            or dict(record.config.get('model_recipe',{}))!=dict(protocol.recipe['visual_student'])):
        raise ValueError('source student checkpoint phase/recipe mismatch')
    key=(protocol.identity(),'student_inference_network',ref.sha256,device)
    if key not in context.public_rows:
        network,_=initialize_student(protocol,device='cpu',public_init=False,context=context)
        checkpoint=next(Path(x) for x in record.files if Path(x).name=='checkpoint.pt')
        network.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        network.eval().to(device);context.public_rows[key]=network
    return record,context.public_rows[key]


def export_population(model,pixels,rows,*,protocol,device='cuda',context=None,sequence_output=None):
    from . import visual_student as base
    from .pixel_cache import load_pixels
    from .no_vote_datasets import NoVotePixelDataset
    from .pose_roi_adapter import _atomic_npz
    context=context or VerificationContext(protocol);registry=ArtifactRegistry(protocol,context=context)
    mr,network=load_student(model,protocol=protocol,device=device,context=context)
    population=base._student_population(mr,rows,protocol)
    inputs=load_stage_inputs(protocol,population,'predict',context=context)
    output=protocol.run_root/'A2'/mr.phase;path=output/(population+'_targets.npz')
    sequence_root=Path(sequence_output).resolve() if sequence_output else protocol.run_root/'sequence'/mr.phase/population
    if not sequence_root.is_relative_to(protocol.run_root):raise ValueError('source sequence output outside run')
    target_ref=path.with_suffix('.artifact.json');sequence_ref=sequence_root/'artifact.json'
    if target_ref.exists() and sequence_ref.exists():
        ref=read_ref(target_ref);sr=read_ref(sequence_ref)
        registry.verify(ref,'A2','predict',rows);registry.verify(sr,'visual_sequence',mr.phase,rows)
        a=context.load_arrays(registry.verify_file(registry.read(ref),path))
        return Prediction(rows,a['logits'],a['probabilities'],a['valid'],ref),sr
    if path.exists() or (sequence_root/'sequence.npy').exists():
        raise ValueError('partial export without final registration; use explicit recovery, never overwrite')
    pr,arrays=load_pixels(pixels,protocol=protocol,context=context);prior_ref,prob=prior(protocol,mr.phase,context)
    dataset=NoVotePixelDataset(arrays,pr.rows,rows=rows);started=time.perf_counter()
    print(json.dumps({'event':'source_student_export_start','partition':population,'phase':mr.phase,'rows':len(rows.sample_ids)}),flush=True)
    logits,probabilities,valid,sequence=base.infer_arrays(network,dataset,prob,device=device,
        batch_size=protocol.recipe['visual_student']['batch_size'],with_sequence=True)
    _atomic_npz(path,dict(sample_ids=np.asarray(rows.sample_ids),class_ids=np.arange(40),logits=logits,
        probabilities=probabilities,valid=valid))
    pred=registry.register(stage='A2',kind='predictions',phase='predict',files=[path],
        parents=[model,pixels,prior_ref,inputs.parents['manifest']],rows=rows,
        config={'partition':population,'source_fixed_epoch':16,'anchor_phase':mr.phase},source_files=[Path(__file__)])
    write_json(target_ref,pred)
    sequence_root.mkdir(parents=True,exist_ok=True)
    values={'sequence':sequence,'anchor_logits':logits,'anchor_valid':valid,'completed':np.ones(len(rows.sample_ids),bool)}
    digests={}
    for name,array in values.items():
        dest=sequence_root/(name+'.npy');digests[str(dest)]=base.save_array_with_digest(dest,array)
    from .no_vote_manifest import csv_bytes,PUBLIC_COLUMNS
    table={x['sample_id']:x for x in read_public_rows(inputs.public_manifest)}
    content=csv_bytes(PUBLIC_COLUMNS,[table[s] for s in rows.sample_ids]);rp=sequence_root/'rows.csv';rp.write_bytes(content)
    import hashlib
    digests[str(rp)]=hashlib.sha256(content).hexdigest()
    sr=registry.register(stage='visual_sequence',kind='raw_cache',phase=mr.phase,files=list(digests),file_digests=digests,
        parents=[model,pixels,prior_ref,inputs.parents['manifest']],rows=rows,
        config={'partition':population,'model_sha256':model.sha256,'pixels_sha256':pixels.sha256,
            'native_backbone_sequence':True,'anchor_phase':mr.phase,'checkpoint_epoch':16},source_files=[Path(__file__)])
    write_json(sequence_ref,sr)
    print(json.dumps({'event':'source_student_export_complete','partition':population,'valid':int(valid.sum()),
        'seconds':round(time.perf_counter()-started,2)}),flush=True)
    return Prediction(rows,logits,probabilities,valid,pred),sr


def run_pipeline(protocol,*,device='cuda'):
    context=VerificationContext(protocol);prepare_student_capability(protocol,context)
    refs=pixel_refs(protocol);selected=None;models={}
    for phase,partition in [('select','train12'),('refit','refit14')]:
        fit=load_stage_inputs(protocol,partition,phase,context=context)
        development=load_stage_inputs(protocol,'development2','select',context=context) if phase=='select' else None
        teacher=load_teacher(protocol,partition,context)
        model,chosen=train_visual_student(phase,fit,development,refs['refit14'],teacher,selected,
            protocol=protocol,device=device,context=context)
        if chosen is not None:selected=chosen
        models[phase]=model
    outputs={}
    for phase,parts in [('select',['train12','development2']),('refit',['refit14','final4'])]:
        for partition in parts:
            rows=row_index(read_public_rows(protocol.public_manifests[partition]))
            pred,sequence=export_population(models[phase],refs['final4' if partition=='final4' else 'refit14'],rows,
                protocol=protocol,device=device,context=context)
            outputs[partition]={'prediction':pred.artifact,'sequence':sequence,'rows':len(rows.sample_ids),'valid':int(pred.valid.sum())}
    write_json(protocol.run_root/'protocol/task5_artifacts_ready.json',{'protocol_sha256':protocol.identity(),
        'fixed_epochs':16,'partitions':outputs,'final_labels_read':False,'independent_review':'pending'})
    return outputs
