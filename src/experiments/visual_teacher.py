"""Frozen public six-clip VideoMAE and one fixed-split Ridge classifier."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .artifact_record import ArtifactRegistry,verify_public_file
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .no_vote_types import ArtifactRef,RowIndex,Selection,StageInputs,Prediction,TeacherTargets,canonical_hash,write_json
from .no_vote_weights import validate_attention_biases
from .pose_roi_adapter import compatible_frame_map,snapshot_raw_files,_atomic_npz
from .teammate_source import verify_teammate_source,load_teammate_symbol,sha256_file

FEATURE_KEYS=frozenset({'sample_ids','class_ids','features','kinetics_logits','valid'})


def validate_feature_arrays(arrays,index: RowIndex):
    if set(arrays)!=FEATURE_KEYS:raise ValueError('visual feature schema mismatch')
    if tuple(arrays['sample_ids'].astype(str))!=index.sample_ids:raise ValueError('visual feature IDs/order mismatch')
    if not np.array_equal(arrays['class_ids'],np.arange(40)):raise ValueError('visual feature class order mismatch')
    n=len(index.sample_ids)
    for key,shape in [('features',(n,2,3,1024)),('kinetics_logits',(n,2,3,400))]:
        if arrays[key].shape!=shape or not np.isfinite(arrays[key]).all():raise ValueError('visual feature shape/value mismatch')
    if arrays['valid'].shape!=(n,) or arrays['valid'].dtype!=np.bool_:raise ValueError('visual valid must be bool[N]')


def window_indices(n,low,high):
    if n<1 or not 0<=low<high<=1:raise ValueError('invalid temporal window')
    return np.rint(np.linspace(low*(n-1),high*(n-1),16)).astype(np.int64)


def feature_sets(features,kinetics_logits):
    """Exact P85 transforms; StandardScaler is fitted later on fit rows only."""
    values=np.asarray(features,dtype=np.float32);logits=np.asarray(kinetics_logits,dtype=np.float32)
    if values.shape[1:]!=(2,3,1024) or logits.shape!=(len(values),2,3,400):raise ValueError('six-clip shape mismatch')
    l2=lambda x:x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
    values=l2(values);early=values[:,0];late=values[:,1];mean=l2(values.mean(axis=1))
    kinetics=logits.reshape(len(values),-1);kinetics=kinetics-kinetics.mean(axis=-1,keepdims=True)
    kinetics=kinetics/np.maximum(kinetics.std(axis=-1,keepdims=True),1e-6)
    return dict(early=early.reshape(len(values),-1),late=late.reshape(len(values),-1),
        window_mean=mean.reshape(len(values),-1),early_late=values.reshape(len(values),-1),
        temporal_delta=np.concatenate((mean,late-early),axis=1).reshape(len(values),-1),kinetics=kinetics)


def candidate_grid(recipe):
    return [dict(family=f,class_weight_power=float(p),alpha=float(a),temperature=1.,
                 candidate_id=f'{f}:p={float(p):g}:a={float(a):g}')
            for f in recipe['families'] for p in recipe['class_weight_powers'] for a in recipe['alphas']]


def choose_candidate(results):
    return min(results,key=lambda r:(-r['accuracy'],-r['macro_f1'],-r['worst_user_accuracy'],
                                     r['dimensions'],-r['alpha'],r['candidate_id']))


def _weights(y,power):
    if power==0:return np.ones(len(y),np.float64)
    counts=np.bincount(y,minlength=40).astype(float);weights=np.zeros(40)
    present=counts>0;weights[present]=(counts[present].mean()/counts[present])**power
    values=weights[y];return values/values.mean()


def _make_model(alpha):
    return Pipeline([('scale',StandardScaler()),('ridge',RidgeClassifier(alpha=alpha,
        solver='lsqr',tol=1e-5,max_iter=5000))])


def _labels(inputs,phase,protocol):
    fresh=load_stage_inputs(protocol,inputs.partition.name,phase)
    if inputs!=fresh or inputs.labels is None:raise ValueError('supervised input ownership mismatch')
    rows=read_public_rows(inputs.public_manifest);index=row_index(rows)
    with inputs.labels.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if tuple(reader.fieldnames or ())!=('sample_id','class_id'):raise ValueError('label table schema mismatch')
        table=list(reader)
    lookup={r['sample_id']:int(r['class_id']) for r in table}
    if len(lookup)!=len(table) or set(lookup)!=set(index.sample_ids):raise ValueError('supervised label IDs mismatch')
    y=np.asarray([lookup[sid] for sid in index.sample_ids],np.int64)
    if np.any((y<0)|(y>=40)):raise ValueError('invalid class labels')
    return index,y


def _load_features(ref,protocol,*,complete=True):
    registry=ArtifactRegistry(protocol);record=registry.verify(ref,'visual_features','raw')
    if complete and not record.complete:raise ValueError('partial features cannot enter complete teacher fitting')
    parents=[registry.read(parent) for parent in record.parents]
    rois=[parent for parent in parents if parent.stage=='p29' and parent.kind=='raw_cache']
    if protocol.recipe['execution_kind']=='formal':
        if len(rois)!=1 or not any(p.stage=='visual_initializer' and p.kind=='public_weights' for p in parents):
            raise ValueError('formal visual features require own P29/public initializer ancestors')
    for roi in rois:
        inputs=load_stage_inputs(protocol,record.config['partition'],'raw')
        public=read_public_rows(inputs.public_manifest);lookup={row['sample_id']:row for row in public}
        if not set(roi.rows.sample_ids)<=set(lookup):raise ValueError('ROI public IDs mismatch')
        source_rows=[lookup[sid] for sid in roi.rows.sample_ids]
        if row_index(source_rows)!=roi.rows:raise ValueError('ROI public user/order mismatch')
        # Downstream heads consume registered features, not raw files. Their
        # input inventory was recorded at generation and is not rescanned.
    files=[Path(f) for f in record.files if Path(f).name=='features.npz' or Path(f).name.endswith('_features.npz')]
    if len(files)!=1 or record.rows is None:raise ValueError('one feature table and indexed population required')
    registry.verify_file(record,files[0])
    with np.load(files[0],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    validate_feature_arrays(arrays,record.rows)
    return record,arrays


def _subset(record,arrays,index):
    lookup={sid:i for i,sid in enumerate(record.rows.sample_ids)}
    if not set(index.sample_ids)<=set(lookup):raise ValueError('feature IDs missing from exact partition')
    order=np.asarray([lookup[sid] for sid in index.sample_ids])
    if tuple(record.rows.user_ids[i] for i in order)!=index.user_ids:raise ValueError('feature users/IDs mismatch')
    result={k:v[order] if k!='class_ids' else v for k,v in arrays.items()}
    validate_feature_arrays(result,index);return result


def _label_record(inputs,phase,protocol,index):
    return ArtifactRegistry(protocol).register(stage='labels_'+inputs.partition.name,kind='supervised_labels',
        phase=phase,files=[inputs.labels],parents=[inputs.parents['manifest']],rows=index,
        config={'partition':inputs.partition.name},source_files=[Path(__file__)])


def fit_class_prior(inputs: StageInputs,phase: str,*,protocol):
    index,y=_labels(inputs,phase,protocol)
    required='train12' if phase=='select' else 'refit14'
    if inputs.partition.name!=required:raise ValueError('prior fit population mismatch')
    counts=np.bincount(y,minlength=40);probabilities=counts/counts.sum()
    out=protocol.run_root/'A1'/phase/'prior.json'
    write_json(out,{'class_ids':list(range(40)),'probabilities':probabilities.tolist(),
        'counts':counts.tolist(),'fit_ids_sha256':canonical_hash(index)})
    ref=ArtifactRegistry(protocol).register(stage='class_prior',kind='statistics',phase=phase,files=[out],
        parents=[_label_record(inputs,phase,protocol,index)],fit_users=inputs.partition.users,rows=index,
        config={'partition':required},source_files=[Path(__file__)])
    write_json(out.parent/'prior_artifact.json',{'record_path':str(ref.record_path),'sha256':ref.sha256})
    return ref


def _prior_logits(probabilities):
    values=np.maximum(probabilities,1e-12);return np.log(values/values.sum())


def _save_model(model,cfg,inputs,index,features,phase,protocol,*,selection=None):
    root=protocol.run_root/'A1'/phase;root.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,root/'head.joblib',compress=3)
    record,_=_load_features(features,protocol)
    config={**cfg,'feature_signature':record.config['feature_signature'],'partition':inputs.partition.name,
        'standardizer_fit':'valid_fit_rows_only','head_temperature':1.}
    write_json(root/'head.json',config)
    return ArtifactRegistry(protocol).register(stage='A1',kind='supervised_model',phase=phase,
        files=[root/'head.joblib',root/'head.json'],parents=[features,_label_record(inputs,phase,protocol,index)],
        fit_users=inputs.partition.users,select_users=protocol.partitions['development2'].users if selection else (),
        rows=index,config=config,source_files=[Path(__file__)])


def select_visual_head(train: StageInputs,development: StageInputs,features: ArtifactRef,*,protocol)->Selection:
    if train.partition.name!='train12' or development.partition.name!='development2':raise ValueError('select population mismatch')
    tr,y=_labels(train,'select',protocol);dev,dy=_labels(development,'select',protocol)
    record,arrays=_load_features(features,protocol)
    ta,da=_subset(record,arrays,tr),_subset(record,arrays,dev);valid=ta['valid']
    if set(y[valid])!=set(range(40)):raise ValueError('valid fit IR rows must cover all 40 classes')
    tm=feature_sets(ta['features'],ta['kinetics_logits']);dm=feature_sets(da['features'],da['kinetics_logits'])
    candidates=candidate_grid(protocol.recipe['visual_teacher'])
    if protocol.recipe['execution_kind']=='formal' and len(candidates)!=72:raise ValueError('formal A1 must use frozen 72 grid')
    prior=np.bincount(y,minlength=40)/len(y);results=[]
    root=protocol.run_root/'A1/select';root.mkdir(parents=True,exist_ok=True)
    for cfg in candidates:
        model=_make_model(cfg['alpha'])
        with threadpool_limits(limits=4):model.fit(tm[cfg['family']][valid],y[valid],ridge__sample_weight=_weights(y[valid],cfg['class_weight_power']))
        scores=np.tile(_prior_logits(prior),(len(dy),1))
        if da['valid'].any():scores[da['valid']]=model.decision_function(dm[cfg['family']][da['valid']])
        pred=scores.argmax(1);user=np.asarray(dev.user_ids)
        result={**cfg,'accuracy':float(np.mean(pred==dy)),
            'macro_f1':float(f1_score(dy,pred,labels=np.arange(40),average='macro',zero_division=0)),
            'worst_user_accuracy':float(min(np.mean(pred[user==u]==dy[user==u]) for u in set(user))),
            'dimensions':tm[cfg['family']].shape[1]}
        results.append(result);print(json.dumps({'candidate':len(results),'total':len(candidates),**result}),flush=True)
    best=choose_candidate(results);cfg={k:best[k] for k in candidates[0]}
    model=_make_model(cfg['alpha'])
    with threadpool_limits(limits=4):model.fit(tm[cfg['family']][valid],y[valid],ridge__sample_weight=_weights(y[valid],cfg['class_weight_power']))
    ref=_save_model(model,cfg,train,tr,features,'select',protocol,selection=True)
    selection=Selection('A1',cfg,{'candidates':len(candidates),'valid_fit_rows':int(valid.sum())},
        {k:best[k] for k in ('accuracy','macro_f1','worst_user_accuracy')},ref,canonical_hash(dev))
    write_json(root/'grid.json',results);selection.write(root/'selection.json')
    return selection


def fit_visual_head(refit: StageInputs,features: ArtifactRef,selection: Selection,*,protocol)->ArtifactRef:
    if refit.partition.name!='refit14' or selection.stage!='A1':raise ValueError('refit population/stage mismatch')
    registry=ArtifactRegistry(protocol);selected=registry.verify(selection.fit_artifact,'A1','select')
    cfg=dict(selection.config)
    if cfg not in candidate_grid(protocol.recipe['visual_teacher']):raise ValueError('selection recipe not frozen')
    expected_dev=row_index(read_public_rows(protocol.public_manifests['development2']))
    if selection.development_ids_sha256!=canonical_hash(expected_dev):raise ValueError('selection development IDs drift')
    if any(selected.config[k]!=v for k,v in cfg.items()):raise ValueError('selected head/config mismatch')
    index,y=_labels(refit,'refit',protocol);record,arrays=_load_features(features,protocol)
    if selected.config['feature_signature']!=record.config['feature_signature']:raise ValueError('public feature identity mismatch')
    arrays=_subset(record,arrays,index);valid=arrays['valid']
    if set(y[valid])!=set(range(40)):raise ValueError('valid refit IR rows must cover all 40 classes')
    values=feature_sets(arrays['features'],arrays['kinetics_logits'])[cfg['family']]
    model=_make_model(cfg['alpha'])
    with threadpool_limits(limits=4):model.fit(values[valid],y[valid],ridge__sample_weight=_weights(y[valid],cfg['class_weight_power']))
    return _save_model(model,cfg,refit,index,features,'refit',protocol)


def predict_visual_teacher(model: ArtifactRef,features: ArtifactRef,rows: RowIndex,prior: ArtifactRef,*,protocol)->TeacherTargets:
    registry=ArtifactRegistry(protocol);mr=registry.verify(model,'A1');pr=registry.verify(prior,'class_prior',mr.phase)
    allowed=('train12','development2') if mr.phase=='select' else ('refit14','final4')
    population=next((p for p in allowed if rows==row_index(read_public_rows(protocol.public_manifests[p]))),None)
    if population is None:raise ValueError('prediction phase/population mismatch')
    inputs=load_stage_inputs(protocol,population,'predict')
    record,arrays=_load_features(features,protocol);arrays=_subset(record,arrays,rows)
    if mr.config['feature_signature']!=record.config['feature_signature']:raise ValueError('public feature identity mismatch')
    prior_body=json.loads(next(Path(f) for f in pr.files if f.endswith('.json')).read_text())
    prob=np.asarray(prior_body['probabilities'],np.float64)
    if prob.shape!=(40,) or np.any(prob<0) or not np.isclose(prob.sum(),1):raise ValueError('invalid prior')
    logits=np.tile(_prior_logits(prob),(len(rows.sample_ids),1));valid=arrays['valid']
    head=joblib.load(next(Path(f) for f in mr.files if f.endswith('.joblib')))
    if not np.array_equal(head.named_steps['ridge'].classes_,np.arange(40)):raise ValueError('head class order mismatch')
    if valid.any():
        values=feature_sets(arrays['features'],arrays['kinetics_logits'])[mr.config['family']]
        logits[valid]=head.decision_function(values[valid])
    probabilities=softmax(logits,axis=1);probabilities[~valid]=prob
    path=protocol.run_root/'A1'/mr.phase/(population+'_targets.npz')
    _atomic_npz(path,dict(sample_ids=np.asarray(rows.sample_ids),class_ids=np.arange(40),
        logits=logits.astype(np.float32),probabilities=probabilities.astype(np.float32),valid=valid))
    ref=registry.register(stage='A1',kind='predictions',phase='predict',files=[path],
        parents=[model,features,prior,inputs.parents['manifest']],rows=rows,
        config={'partition':population,'head_temperature':1.},source_files=[Path(__file__)])
    return TeacherTargets(Prediction(rows,logits,probabilities,valid,ref),arrays['features'])


def _encoding_ops(protocol):
    report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
        expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
    return tuple(load_teammate_symbol(report,m,s) for m,s in [
        ('build_p46_videomae_cache','square_crop'),('build_p30_shared_dir_roi_feature_cache','read_ir'),
        ('build_p46_videomae_cache','encode')])


def _prepare_clips(row,cache,crop,read_ir):
    with np.load(cache,allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str);acquired=z['acquisition_ids'].astype(str)
        names=tuple(z['region_names'].astype(str));boxes=z['roi_boxes_xyxy'];valid=z['roi_valid']
        if str(z['sample_id'].item())!=row['sample_id'] or not bool(z['completed']):raise ValueError('ROI trial identity/incomplete')
    if acquired.shape!=ids.shape or boxes.shape!=(len(ids),7,4) or valid.shape!=(len(ids),7):raise ValueError('ROI schema mismatch')
    actual=compatible_frame_map(Path(row['ir_path']),'ir') if row['ir_path'] else {}
    if tuple(ids)!=tuple(sorted(actual)):raise ValueError('ROI/IR frame axis mismatch')
    if not len(ids):return [],{'frame_indices':np.zeros((2,16),np.int64),'acquisition_ids':np.full((2,16),'',dtype='U1')}
    person=names.index('full_body');workspace=names.index('hand_workspace');clips=[];chosen_all=[]
    for low,high in ((0.,.70),(.30,1.)):
        chosen=window_indices(len(ids),low,high);chosen_all.append(chosen);views=[[],[],[]]
        for i in chosen:
            image=read_ir(actual[ids[i]])
            pbox=boxes[i,person] if valid[i,person] else np.full(4,np.nan)
            wbox=boxes[i,workspace] if valid[i,workspace] else pbox
            for view,img in zip(views,(image,crop(image,pbox,1.15),crop(image,wbox,1.40))):view.append(img)
        clips.extend(views)
    indices=np.stack(chosen_all)
    return clips,{'frame_indices':indices,'acquisition_ids':acquired[indices]}


def extract_visual_features(inputs: StageInputs,roi: ArtifactRef,weights: ArtifactRef,output: Path,*,
                            protocol,device='cuda',max_trials=0,clip_batch=6)->ArtifactRef:
    registry=ArtifactRegistry(protocol)
    fresh=load_stage_inputs(protocol,inputs.partition.name,'raw')
    if inputs.public_manifest!=fresh.public_manifest or inputs.parents!=fresh.parents:raise ValueError('raw input ownership mismatch')
    rows=read_public_rows(inputs.public_manifest)
    if max_trials<0 or clip_batch not in (1,2,3,6):raise ValueError('invalid extraction batch/population limit')
    selected=rows[:max_trials] if max_trials else rows;index=row_index(selected)
    rr=registry.verify(roi,'p29','raw')
    if rr.rows not in (index,row_index(rows)):raise ValueError('ROI input population/IDs mismatch')
    # Extraction consumes raw data. This producer-specific inventory check
    # is separate from downstream cached fitting's metadata-only ancestry.
    parent_rows=selected if rr.rows==index else rows
    inventory={key:value for row in parent_rows for key,value in snapshot_raw_files(row).items()}
    if inventory!=dict(rr.raw_input_hashes):raise ValueError('ROI raw input membership drift before extraction/resume')
    wr=registry.verify(weights,'visual_initializer','public')
    if not max_trials and not rr.complete:raise ValueError('partial ROI cannot enter full extraction')
    expected={str((protocol.weights['videomae']/n).resolve()) for n in ('config.json','preprocessor_config.json','model.safetensors')}
    if set(wr.files)!=expected:raise ValueError('visual public weight identity mismatch')
    receipt=json.loads((protocol.run_root/'protocol/weights_manifest.json').read_text(encoding='utf-8'))
    pinned={str(Path(f['path']).resolve()):f['sha256'] for f in receipt['weights']['videomae']['files']}
    if dict(wr.files)!=pinned:raise ValueError('visual public weight receipt mismatch')
    signature=canonical_hash({'weights':wr.files,'pose_recipe':rr.config,'windows':[[0.,.70],[.30,1.]],
        'views':['scene','person','workspace'],'crops':[1.15,1.40],'clip_batch':clip_batch,
        'producer':sha256_file(Path(__file__)),'device':device,'head_temperature':1.})
    config={'partition':inputs.partition.name,'feature_signature':signature,'max_trials':max_trials,
        'clip_batch':clip_batch,'backbone_frozen':True,'schema':1}
    identity=canonical_hash({'config':config,'roi':roi.sha256,'weights':weights.sha256,'rows':index})
    output=Path(output).resolve()
    if not output.is_relative_to(protocol.run_root):raise ValueError('feature output must be within run')
    output.mkdir(parents=True,exist_ok=True);identity_path=output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!={'identity':identity}:raise ValueError('feature resume identity drift')
    write_json(identity_path,{'identity':identity})
    if (output/'artifact.json').exists():
        body=json.loads((output/'artifact.json').read_text());ref=ArtifactRef(Path(body['record_path']),body['sha256'])
        registry.verify(ref,'visual_features','raw',index);return ref
    roi_files={Path(f).stem:Path(f) for f in rr.files if f.endswith('.npz')}
    if set(roi_files)!=set(rr.rows.sample_ids):raise ValueError('ROI cache IDs mismatch')
    model=processor=ops=None;started=time.perf_counter();peak=0.;all_features=[];all_logits=[];all_valid=[];trial_files=[]
    for number,row in enumerate(selected,1):
        sid=row['sample_id'];cache=output/'trial_cache'/(sid+'.npz');meta=cache.with_suffix('.json')
        if cache.exists() and meta.exists():
            body=json.loads(meta.read_text())
            if body['identity']!=identity or body['cache_sha256']!=sha256_file(cache):raise ValueError('feature trial resume hash/identity drift')
            verify_public_file(cache)
        else:
            registry.verify_file(rr,roi_files[sid])
            if ops is None:ops=_encoding_ops(protocol)
            crop,read_ir,encode=ops;clips,metadata=_prepare_clips(row,roi_files[sid],crop,read_ir)
            features=np.zeros((2,3,1024),np.float16);logits=np.zeros((2,3,400),np.float16)
            if clips:
                import torch
                if model is None:
                    from transformers import VideoMAEForVideoClassification,VideoMAEImageProcessor
                    local=protocol.weights['videomae']
                    for file in wr.files:registry.verify_file(wr,Path(file))
                    model=VideoMAEForVideoClassification.from_pretrained(local,local_files_only=True)
                    processor=VideoMAEImageProcessor.from_pretrained(local,local_files_only=True)
                    bias=validate_attention_biases(model,local/'model.safetensors')
                    if model.config.hidden_size!=1024 or model.config.num_labels!=400:raise ValueError('public model architecture mismatch')
                    model.eval().to(torch.device(device))
                    for parameter in model.parameters():parameter.requires_grad_(False)
                    write_json(output/'model_validation.json',{'attention_biases':bias,'frozen':all(not p.requires_grad for p in model.parameters())})
                encoded=[];kinetics=[]
                for start in range(0,6,clip_batch):
                    f,k,p=encode(model,processor,clips[start:start+clip_batch],torch.device(device))
                    encoded.append(f);kinetics.append(k);peak=max(peak,p)
                features=np.concatenate(encoded).reshape(2,3,1024).astype(np.float16)
                logits=np.concatenate(kinetics).reshape(2,3,400).astype(np.float16)
            _atomic_npz(cache,dict(sample_id=np.asarray(sid),features=features,kinetics_logits=logits,
                valid=np.asarray(bool(clips)),**metadata))
            write_json(meta,{'identity':identity,'cache_sha256':sha256_file(cache)})
        with np.load(cache,allow_pickle=False) as z:
            if str(z['sample_id'].item())!=sid:raise ValueError('feature trial ID mismatch')
            all_features.append(z['features']);all_logits.append(z['kinetics_logits']);all_valid.append(bool(z['valid']))
        trial_files.extend([cache,meta])
        if number%20==0 or number==len(selected):print(json.dumps({'extracted':number,'total':len(selected),
            'elapsed_seconds':round(time.perf_counter()-started,1),'peak_cuda_gib':round(peak,3)}),flush=True)
    arrays=dict(sample_ids=np.asarray(index.sample_ids),class_ids=np.arange(40),features=np.stack(all_features),
        kinetics_logits=np.stack(all_logits),valid=np.asarray(all_valid,bool))
    validate_feature_arrays(arrays,index);_atomic_npz(output/'features.npz',arrays)
    ref=registry.register(stage='visual_features',kind='raw_cache',phase='raw',files=[output/'features.npz',*trial_files],
        parents=[roi,weights,inputs.parents['manifest']],rows=index,config=config,
        source_files=[Path(__file__)],complete=rr.complete and len(selected)==len(rows))
    write_json(output/'artifact.json',{'record_path':str(ref.record_path),'sha256':ref.sha256})
    return ref
