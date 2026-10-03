"""Fixed-split MC3 student, single-teacher hybrid losses and native sequences."""
import json
import math
import hashlib
import os
from contextlib import contextmanager
from filelock import FileLock,Timeout
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from .artifact_record import ArtifactRegistry
from .no_vote_types import ArtifactRef,Prediction,TeacherTargets,Selection,RowIndex,canonical_hash,write_json
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .pixel_cache import load_pixels
from .no_vote_datasets import NoVotePixelDataset
from .visual_teacher import _labels,_label_record,_load_features,_subset
from .teammate_source import verify_teammate_source,load_teammate_symbol,sha256_file


def source_symbol(protocol,module,name):
    report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
        expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
    return load_teammate_symbol(report,module,name)


def initialize_student(protocol,*,device='cuda',public_init=True):
    started=time.perf_counter();recipe=protocol.recipe['visual_student']
    seed=recipe['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    cls=source_symbol(protocol,'p86_mc3_visual_model','P86MC3VisualStudent')
    model=cls(classes=40,width=512,frames=16,dropout=.18,fusion_mode='gated',
        temporal_modeling=True,kinetics_pretrained=False,enable_distillation_projection=False)
    ref=None
    if public_init:
        registry=ArtifactRegistry(protocol);path=protocol.weights['mc3']
        receipt=json.loads((protocol.run_root/'protocol/weights_manifest.json').read_text())['weights']['mc3']
        pinned={str(Path(item['path']).resolve()):item['sha256'] for item in receipt['files']}
        ref=registry.register(stage='visual_student_initializer',kind='public_weights',phase='public',
            files=[path],file_digests=pinned,config={'role':'mc3','asset_identity':protocol.recipe['asset_bindings']},
            source_files=[Path(__file__)])
        registry.verify_file(registry.read(ref),path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        for name in ('stem','layer1','layer2','layer3','layer4'):
            block={key[len(name)+1:]:value for key,value in state.items() if key.startswith(name+'.')}
            getattr(model,name).load_state_dict(block,strict=True)
    model.freeze_low_level(recipe['freeze_through']);model.to(device)
    print(json.dumps({'event':'student_initialized','device':str(device),'public_init':public_init,
        'seconds':round(time.perf_counter()-started,2)}),flush=True)
    return model,ref


def epoch_key(metrics):
    return (-metrics['accuracy'],-metrics['macro_f1'],-metrics['worst_user_accuracy'],metrics['epoch'])


def relation_loss(student,teacher,mask):
    """P86 relation_loss algebra, with six clips and availability mask."""
    s=F.normalize(student.float(),dim=-1).flatten(1,2)
    t=F.normalize(teacher.float(),dim=-1).flatten(1,2)
    sg=torch.einsum('bwd,bvd->bwv',s,s);tg=torch.einsum('bwd,bvd->bwv',t,t)
    flat=mask.flatten(1);pairs=flat.unsqueeze(1)&flat.unsqueeze(2)
    return ((sg-tg).square()*pairs).sum()/pairs.sum().clamp_min(1)


def hybrid_loss(output,batch,weights,recipe):
    if recipe['feature_weight_effective']!=0 or recipe['stage_distillation_weight']!=0:
        raise ValueError('hybrid cannot enable direct feature/stage distillation')
    ce=F.cross_entropy(output['logits'].float(),batch['label'],weight=weights,
        label_smoothing=recipe['label_smoothing'])
    temperature=recipe['distillation_temperature'];valid=batch['teacher_valid'].bool()
    each=F.kl_div(F.log_softmax(output['logits'].float()/temperature,dim=1),
        F.softmax(batch['teacher_logits'].float()/temperature,dim=1),reduction='none').sum(1)*temperature**2
    kd=(each*valid).sum()/valid.sum().clamp_min(1)
    mask=output['clip_mask'].bool()&valid[:,None,None]
    relation=relation_loss(output['clip_embeddings'],batch['teacher_features'],mask)
    zero=output['logits'].sum()*0.
    return dict(loss=ce+recipe['distillation_weight']*kd+recipe['relation_weight']*relation,
        ce=ce,kd=kd,relation=relation,feature=zero,stage_kd=zero)


def sequence_and_anchor(model,batch):
    model.eval()
    with torch.inference_mode():
        sequence=model.encode_backbone_sequence(batch['images'])
        if sequence.shape!=(len(batch['images']),2,3,16,512):raise ValueError('native sequence shape mismatch')
        output=model.forward_from_backbone_sequence(sequence,batch['view_valid'],batch['view_quality'],
            batch.get('global_time_position'))
    return sequence.detach().float().cpu().numpy().copy(),output['logits'].detach().float().cpu().numpy().copy()


def read_ref(path):
    body=json.loads(Path(path).read_text(encoding='utf-8'))
    return ArtifactRef(Path(body['record_path']),body['sha256'])


class StageVerificationContext:
    def __init__(self,protocol):self.protocol_sha256=protocol.identity();self.features={};self.records={}
    def feature_table(self,ref,protocol):
        if self.protocol_sha256!=protocol.identity():raise ValueError('verification context protocol mismatch')
        if ref not in self.features:self.features[ref]=_load_features(ref,protocol)
        return self.features[ref]


def guard_teacher(teacher,inputs,*,protocol,verification_context=None):
    registry=ArtifactRegistry(protocol);record=registry.verify(teacher.prediction.artifact,'A1','predict')
    if record.config['partition']!=inputs.partition.name or record.rows!=teacher.prediction.index:
        raise ValueError('teacher population/index mismatch')
    expected=row_index(read_public_rows(inputs.public_manifest))
    if expected!=record.rows or teacher.features is None or teacher.features.shape!=(len(expected.sample_ids),2,3,1024):
        raise ValueError('teacher population/features mismatch')
    with np.load(next(Path(p) for p in record.files if p.endswith('.npz')),allow_pickle=False) as z:
        for name,value in [('logits',teacher.prediction.logits),('probabilities',teacher.prediction.probabilities),('valid',teacher.prediction.valid)]:
            if not np.array_equal(z[name],value):raise ValueError('teacher targets differ from registered file')
    parents=[parent for parent in record.parents if registry.read(parent).stage=='visual_features']
    if len(parents)!=1:raise ValueError('teacher features require one registered source')
    fr,arrays=verification_context.feature_table(parents[0],protocol) if verification_context else _load_features(parents[0],protocol)
    arrays=_subset(fr,arrays,record.rows)
    if not np.array_equal(arrays['features'],teacher.features):raise ValueError('teacher features differ from registered source')
    return record


def load_teacher(protocol,partition,verification_context=None):
    phase='select' if partition=='train12' else 'refit'
    path=protocol.run_root/'A1'/phase/(partition+'_targets.npz')
    ref=read_ref(path.with_suffix('.artifact.json'))
    record=ArtifactRegistry(protocol).verify(ref,'A1','predict')
    with np.load(path,allow_pickle=False) as z:
        pred=Prediction(record.rows,z['logits'],z['probabilities'],z['valid'],ref)
    registry=ArtifactRegistry(protocol)
    parents=[parent for parent in record.parents if registry.read(parent).stage=='visual_features']
    if len(parents)!=1:raise ValueError('teacher requires one feature source')
    fr,arrays=verification_context.feature_table(parents[0],protocol) if verification_context else _load_features(parents[0],protocol)
    arrays=_subset(fr,arrays,record.rows)
    return TeacherTargets(pred,arrays['features'])


def to_device(batch,device):
    return {key:value.to(device) if torch.is_tensor(value) else value for key,value in batch.items()}


def train_epoch(model,loader,optimizer,weights,recipe,*,device='cuda',scaler=None):
    device=torch.device(device);model.train();accum=recipe['gradient_accumulation']
    if scaler is None:scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    optimizer.zero_grad(set_to_none=True)
    sums={key:0. for key in ('loss','ce','kd','relation','feature','stage_kd')};samples=steps=batches=0
    started=time.perf_counter()
    for batches,batch in enumerate(loader,1):
        batch=to_device(batch,device)
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            output=model(batch['images'],batch['view_valid'],batch['view_quality'],batch.get('global_time_position'))
            parts=hybrid_loss(output,batch,weights,recipe)
        if not torch.isfinite(parts['loss']):raise ValueError('non-finite student loss')
        scaler.scale(parts['loss']/accum).backward()
        if batches%accum==0:
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);steps+=1
        n=len(batch['label']);samples+=n
        for key in sums:sums[key]+=float(parts[key].detach())*n
        if batches%100==0:print(json.dumps({'event':'student_train_batch','batches':batches,'samples':samples}),flush=True)
    if batches%accum:
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);steps+=1
    return {**{key:value/max(samples,1) for key,value in sums.items()},'samples':samples,'optimizer_steps':steps,
        'seconds':round(time.perf_counter()-started,2)}


def infer_arrays(model,dataset,prior,*,device='cuda',batch_size=4,with_sequence=False):
    device=torch.device(device);n=len(dataset);logits=np.tile(np.log(np.maximum(prior,1e-12)),(n,1)).astype(np.float32)
    valid=np.zeros(n,bool);sequence=np.zeros((n,2,3,16,512),np.float16) if with_sequence else None
    model.eval();offset=0
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):
            batch=to_device(batch,device);count=len(batch['images']);usable=batch['view_valid'].flatten(1).any(1)
            positions=torch.nonzero(usable,as_tuple=False).flatten()
            if len(positions):
                selected={key:value.index_select(0,positions) if torch.is_tensor(value) and value.ndim and len(value)==count else value for key,value in batch.items()}
                with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    if with_sequence:scores,anchors=sequence_and_anchor(model,selected)
                    else:anchors=model(selected['images'],selected['view_valid'],selected['view_quality'],selected['global_time_position'])['logits'].float().cpu().numpy()
                rows=offset+positions.cpu().numpy();logits[rows]=anchors;valid[rows]=True
                if with_sequence:sequence[rows]=scores.astype(np.float16)
            offset+=count
    from scipy.special import softmax
    probabilities=softmax(logits,axis=1);probabilities[~valid]=prior
    return logits,probabilities,valid,sequence


def metrics(index,y,logits):
    prediction=logits.argmax(1);users=np.asarray(index.user_ids)
    return dict(accuracy=float(np.mean(prediction==y)),macro_f1=float(f1_score(y,prediction,labels=np.arange(40),average='macro',zero_division=0)),
        worst_user_accuracy=float(min(np.mean(prediction[users==u]==y[users==u]) for u in set(users))))


def phase_prior(protocol,phase):
    ref=read_ref(protocol.run_root/'A1'/phase/'prior_artifact.json')
    record=ArtifactRegistry(protocol).verify(ref,'class_prior',phase)
    body=json.loads(next(Path(p) for p in record.files if p.endswith('.json')).read_text(encoding='utf-8'))
    prior=np.asarray(body['probabilities'],np.float32)
    if prior.shape!=(40,) or not np.isfinite(prior).all() or not np.isclose(prior.sum(),1):raise ValueError('student prior invalid')
    return ref,prior


@contextmanager
def phase_lock(root):
    root.mkdir(parents=True,exist_ok=True)
    lock=FileLock(root/'training.lock',timeout=0)
    try:lock.acquire()
    except Timeout as error:raise RuntimeError('student phase already running') from error
    try:yield
    finally:lock.release()


def train_visual_student(phase,fit,development,pixels,teacher,selection,*,protocol,device='cuda',verification_context=None):
    if phase not in ('select','refit'):raise ValueError('student phase mismatch')
    registry=ArtifactRegistry(protocol)
    selected=None
    if phase=='refit':
        if selection is None:raise ValueError('student refit needs selection')
        selected=registry.verify(selection.fit_artifact,'A2','select')
        if selected.kind!='supervised_model' or selected.config['budget_epochs']!=selection.budget['epochs']:
            raise ValueError('student selected epoch budget mismatch')
    root=protocol.run_root/'A2'/phase
    identity=canonical_hash({'protocol':protocol.identity(),'phase':phase,'fit':fit,'pixels':pixels,
        'teacher':teacher.prediction.artifact,'selection':selection,'source':sha256_file(Path(__file__)),
        'labels_sha256':sha256_file(fit.labels)})
    with phase_lock(root):
        path=root/'identity.json'
        if path.exists() and json.loads(path.read_text())!={'identity':identity}:raise ValueError('student phase identity drift')
        write_json(path,{'identity':identity})
        completion=root/'completion.json'
        if not (root/'artifact.json').exists() and completion.exists():
            done=json.loads(completion.read_text(encoding='utf-8'))
            if done['identity']!=identity:raise ValueError('student completion identity mismatch')
            ref=ArtifactRef(Path(done['artifact']['record_path']),done['artifact']['sha256'])
            registry.verify(ref,'A2',phase)
            write_json(root/'artifact.json',done['artifact'])
            if done['selection'] is not None:write_json(root/'selection.json',done['selection'])
        if (root/'artifact.json').exists():
            ref=read_ref(root/'artifact.json');registry.verify(ref,'A2',phase)
            if phase=='refit':return ref,None
            body=json.loads((root/'selection.json').read_text(encoding='utf-8'))
            body['fit_artifact']=ArtifactRef(Path(body['fit_artifact']['record_path']),body['fit_artifact']['sha256'])
            return ref,Selection(**body)
        ref,chosen=_train_visual_student(phase,fit,development,pixels,teacher,selection,protocol=protocol,
            device=device,verification_context=verification_context,selected_record=selected)
        write_json(completion,{'identity':identity,'artifact':ref,'selection':chosen})
        write_json(root/'artifact.json',{'record_path':str(ref.record_path),'sha256':ref.sha256})
        return ref,chosen


def _train_visual_student(phase,fit,development,pixels,teacher,selection,*,protocol,device='cuda',verification_context=None,selected_record=None):
    expected='train12' if phase=='select' else 'refit14'
    if phase not in ('select','refit') or fit.partition.name!=expected:raise ValueError('student fit phase/population mismatch')
    if phase=='select' and (development is None or development.partition.name!='development2' or selection is not None):
        raise ValueError('student select needs development2 only')
    if phase=='refit' and (development is not None or selection is None or selection.stage!='A2'):
        raise ValueError('student refit needs fixed A2 selection without development')
    recipe=protocol.recipe['visual_student'];registry=ArtifactRegistry(protocol)
    if phase=='refit':
        # Verify selection metadata; never initialize from selected weights.
        selected=selected_record or registry.verify(selection.fit_artifact,'A2','select')
        if selected.kind!='supervised_model' or dict(selection.config)!=dict(recipe) or dict(selected.config['model_recipe'])!=dict(recipe):raise ValueError('student selection/config phase mismatch')
        if selected.config['budget_epochs']!=selection.budget['epochs']:raise ValueError('student selected epoch budget mismatch')
        dev=row_index(read_public_rows(protocol.public_manifests['development2']))
        if selection.development_ids_sha256!=canonical_hash(dev):raise ValueError('student selection development IDs mismatch')
    tr,y=_labels(fit,phase,protocol);guard_teacher(teacher,fit,protocol=protocol,verification_context=verification_context)
    pixel_record,arrays=load_pixels(pixels,protocol=protocol)
    lookup={sid:i for i,sid in enumerate(pixel_record.rows.sample_ids)}
    if not set(tr.sample_ids)<=set(lookup):raise ValueError('student pixels missing fit IDs')
    positions=[lookup[sid] for sid in tr.sample_ids]
    valid=arrays['view_valid'][positions].reshape(len(tr.sample_ids),-1).any(1)
    if not np.array_equal(valid,teacher.prediction.valid):raise ValueError('student pixel/teacher availability mismatch')
    if set(y[valid])!=set(range(40)):raise ValueError('student valid fit population lacks 40 classes')
    rows=RowIndex(tuple(sid for sid,v in zip(tr.sample_ids,valid) if v),tuple(u for u,v in zip(tr.user_ids,valid) if v),tr.class_ids)
    labels=dict(zip(tr.sample_ids,y.tolist()))
    augment_cls=source_symbol(protocol,'p86_visual_pixel_data','P86VisualPixelDataset')
    dataset=NoVotePixelDataset(arrays,pixel_record.rows,rows=rows,labels={sid:labels[sid] for sid in rows.sample_ids},
        teacher=teacher,augmentation=augment_cls._subject_robust_augment)
    loader=DataLoader(dataset,batch_size=recipe['batch_size'],shuffle=True,num_workers=0,
        drop_last=len(dataset)%recipe['batch_size']==1)
    epochs=recipe['epochs']['max'] if phase=='select' else selection.budget['epochs']
    if not recipe['epochs']['min']<=epochs<=recipe['epochs']['max']:raise ValueError('student epoch budget outside frozen range')
    prior_ref,prior=phase_prior(protocol,phase)
    if phase=='select':
        dev,dy=_labels(development,'select',protocol)
        dev_dataset=NoVotePixelDataset(arrays,pixel_record.rows,rows=dev)
    model,initializer=initialize_student(protocol,device=device)
    optimizer=torch.optim.AdamW([
        {'params':[x for x in model.backbone_parameters() if x.requires_grad],'lr':recipe['backbone_learning_rate'],'initial_lr':recipe['backbone_learning_rate']},
        {'params':[x for x in model.head_parameters() if x.requires_grad],'lr':recipe['head_learning_rate'],'initial_lr':recipe['head_learning_rate']}],
        weight_decay=recipe['weight_decay'],foreach=False)
    counts=np.bincount(y[valid],minlength=40).astype(float)
    values=(counts.mean()/counts)**recipe['class_weight_power'];values/=values.mean()
    weights=torch.tensor(values,dtype=torch.float32,device=device)
    root=protocol.run_root/'A2'/phase;root.mkdir(parents=True,exist_ok=True)
    state=json.loads((protocol.run_root/'run_state.json').read_text(encoding='utf-8'))
    if state['state'] in ('frozen','revealed'):raise ValueError('student generation forbidden after freeze/reveal')
    state['state']='generating';write_json(protocol.run_root/'run_state.json',state)
    history=[];best=None;best_state=None;start_epoch=1
    scaler=torch.amp.GradScaler('cuda',enabled=torch.device(device).type=='cuda')
    resume_path=root/'resume.pt'
    if resume_path.exists():
        saved=torch.load(resume_path,map_location=device,weights_only=True)
        if saved['identity']!=json.loads((root/'identity.json').read_text())['identity']:raise ValueError('student resume identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];best=saved['best'];best_state={k:v.cpu() for k,v in saved['best_state'].items()};start_epoch=saved['epoch']+1
        torch.save(best_state,root/'checkpoint.pt')
        torch.set_rng_state(saved['torch_rng'].cpu());random.setstate(saved['python_rng'])
        nr=saved['numpy_rng'];np.random.set_state((nr[0],nr[1].cpu().numpy().astype(np.uint32),nr[2],nr[3],nr[4]))
        if saved['cuda_rng']:torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    for epoch in range(start_epoch,epochs+1):
        minimum=recipe['minimum_learning_rate']/recipe['head_learning_rate']
        scale=epoch/2 if epoch<=2 else minimum+.5*(1-minimum)*(1+math.cos(math.pi*(epoch-2)/max(epochs-2,1)))
        for group in optimizer.param_groups:group['lr']=group['initial_lr']*scale
        training=train_epoch(model,loader,optimizer,weights,recipe,device=device,scaler=scaler)
        current={'epoch':epoch,'training':training}
        if phase=='select':
            logits,_,_,_=infer_arrays(model,dev_dataset,prior,device=device,batch_size=recipe['batch_size'])
            current.update(metrics(dev,dy,logits))
        if phase=='refit' or best is None or epoch_key(current)<epoch_key(best):
            torch.save(model.state_dict(),root/'checkpoint.tmp.pt');(root/'checkpoint.tmp.pt').replace(root/'checkpoint.pt')
            best=current;best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        history.append(current);write_json(root/'history.json',history)
        nr=np.random.get_state()
        saved={'identity':json.loads((root/'identity.json').read_text())['identity'],'epoch':epoch,
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'history':history,'best':best,'best_state':best_state,'torch_rng':torch.get_rng_state(),
            'python_rng':random.getstate(),'numpy_rng':(nr[0],torch.from_numpy(nr[1].astype(np.int64)),nr[2],nr[3],nr[4]),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.device(device).type=='cuda' else []}
        torch.save(saved,root/'resume.tmp.pt');(root/'resume.tmp.pt').replace(resume_path)
        print(json.dumps({'event':'student_epoch','phase':phase,**current}),flush=True)
    config={'model_recipe':recipe,'budget_epochs':best['epoch'] if phase=='select' else epochs,
        'partition':expected,'valid_fit_ids_sha256':canonical_hash(rows),'valid_fit_rows':len(rows.sample_ids)}
    write_json(root/'model.json',config)
    parents=[initializer,pixels,teacher.prediction.artifact,prior_ref,_label_record(fit,phase,protocol,tr)]
    ref=registry.register(stage='A2',kind='supervised_model',phase=phase,files=[root/'checkpoint.pt',root/'model.json'],
        parents=parents,fit_users=fit.partition.users,select_users=development.partition.users if phase=='select' else (),
        rows=tr,config=config,source_files=[Path(__file__),Path(__file__).with_name('no_vote_datasets.py')])
    if phase=='select':
        chosen=Selection('A2',recipe,{'epochs':best['epoch']},{k:best[k] for k in ('accuracy','macro_f1','worst_user_accuracy')},ref,canonical_hash(dev))
        chosen.write(root/'selection.json');return ref,chosen
    return ref,None


def load_student(ref,*,protocol,device='cuda'):
    record=ArtifactRegistry(protocol).verify(ref,'A2')
    if record.phase not in ('select','refit') or dict(record.config['model_recipe'])!=dict(protocol.recipe['visual_student']):
        raise ValueError('student checkpoint phase/recipe mismatch')
    model,_=initialize_student(protocol,device=device,public_init=False)
    path=next(Path(p) for p in record.files if p.endswith('.pt'))
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True);model.eval()
    return record,model


def _student_population(record,rows,protocol):
    allowed=('train12','development2') if record.phase=='select' else ('refit14','final4')
    population=next((part for part in allowed if row_index(read_public_rows(protocol.public_manifests[part]))==rows),None)
    if population is None:raise ValueError('student prediction phase/population mismatch')
    return population


def predict_visual_student(model,pixels,rows,*,protocol,device='cuda'):
    from .pose_roi_adapter import _atomic_npz
    registry=ArtifactRegistry(protocol);mr,network=load_student(model,protocol=protocol,device=device)
    population=_student_population(mr,rows,protocol);inputs=load_stage_inputs(protocol,population,'predict')
    pr,arrays=load_pixels(pixels,protocol=protocol);prior_ref,prior=phase_prior(protocol,mr.phase)
    dataset=NoVotePixelDataset(arrays,pr.rows,rows=rows)
    logits,prob,valid,_=infer_arrays(network,dataset,prior,device=device,batch_size=protocol.recipe['visual_student']['batch_size'])
    path=protocol.run_root/'A2'/mr.phase/(population+'_targets.npz')
    _atomic_npz(path,dict(sample_ids=np.asarray(rows.sample_ids),class_ids=np.arange(40),logits=logits,probabilities=prob,valid=valid))
    ref=registry.register(stage='A2',kind='predictions',phase='predict',files=[path],
        parents=[model,pixels,prior_ref,inputs.parents['manifest']],rows=rows,config={'partition':population},source_files=[Path(__file__)])
    write_json(path.with_suffix('.artifact.json'),{'record_path':str(ref.record_path),'sha256':ref.sha256})
    return Prediction(rows,logits,prob,valid,ref)


def save_array_with_digest(path,array):
    class Writer:
        def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256()
        def write(self,data):self.digest.update(data);return self.stream.write(data)
        def flush(self):return self.stream.flush()
    with Path(path).open('wb') as stream:
        writer=Writer(stream);np.save(writer,array,allow_pickle=False);writer.flush()
    return writer.digest.hexdigest()


def build_sequence(model,pixels,rows,output,*,protocol,device='cuda'):
    registry=ArtifactRegistry(protocol);mr,network=load_student(model,protocol=protocol,device=device)
    population=_student_population(mr,rows,protocol);inputs=load_stage_inputs(protocol,population,'predict')
    pr,arrays=load_pixels(pixels,protocol=protocol);prior_ref,prior=phase_prior(protocol,mr.phase)
    output=Path(output).resolve()
    if not output.is_relative_to(protocol.run_root):raise ValueError('sequence output outside run')
    output.mkdir(parents=True,exist_ok=True)
    config={'partition':population,'model_sha256':model.sha256,'pixels_sha256':pixels.sha256,
        'native_backbone_sequence':True,'anchor_phase':mr.phase}
    if (output/'artifact.json').exists():
        ref=read_ref(output/'artifact.json');record=registry.verify(ref,'visual_sequence',mr.phase,rows)
        if dict(record.config)!=config:raise ValueError('sequence checkpoint/pixel identity drift')
        return ref
    dataset=NoVotePixelDataset(arrays,pr.rows,rows=rows);started=time.perf_counter()
    print(json.dumps({'event':'sequence_start','phase':mr.phase,'partition':population,'rows':len(rows.sample_ids)}),flush=True)
    logits,_,valid,sequence=infer_arrays(network,dataset,prior,device=device,
        batch_size=protocol.recipe['visual_student']['batch_size'],with_sequence=True)
    values={'sequence':sequence,'anchor_logits':logits,'anchor_valid':valid,'completed':np.ones(len(rows.sample_ids),bool)}
    hashes={}
    for name,array in values.items():
        path=output/(name+'.npy');hashes[str(path)]=save_array_with_digest(path,array)
    from .no_vote_manifest import csv_bytes,PUBLIC_COLUMNS
    table={row['sample_id']:row for row in read_public_rows(inputs.public_manifest)}
    content=csv_bytes(PUBLIC_COLUMNS,[table[sid] for sid in rows.sample_ids]);path=output/'rows.csv';path.write_bytes(content)
    hashes[str(path)]=hashlib.sha256(content).hexdigest()
    ref=registry.register(stage='visual_sequence',kind='raw_cache',phase=mr.phase,files=list(hashes),file_digests=hashes,
        parents=[model,pixels,prior_ref,inputs.parents['manifest']],rows=rows,config=config,source_files=[Path(__file__)])
    write_json(output/'artifact.json',{'record_path':str(ref.record_path),'sha256':ref.sha256})
    print(json.dumps({'event':'sequence_registered','seconds':round(time.perf_counter()-started,2)}),flush=True)
    return ref
