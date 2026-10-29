"""Source RF over original 32-point IMU statistics; fixed-split, never old OOF."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from io import BytesIO
import hashlib

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_cache_import import immutable_bytes
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .no_vote_types import ArtifactRef,Prediction,TeacherTargets,canonical_bytes,canonical_hash,write_json
from .teammate_source import verify_teammate_source,load_teammate_symbol

RF_SEED=20260723
SOURCE_FILES=('build_imu_cache.py','imu_data.py','run_imu_stat_baseline.py','train_final_imu_rf.py')


def source_rf_ops(protocol,context):
    if protocol.recipe.get('source_contract_version')!=3:raise ValueError('RF requires source contract v3')
    key=(protocol.identity(),'source_rf_math')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        feature=load_teammate_symbol(report,'run_imu_stat_baseline','feature_vector')
        load=load_teammate_symbol(report,'build_imu_cache','load_trial')
        baseline=sys.modules['run_imu_stat_baseline'];cache=sys.modules['build_imu_cache']
        context.public_rows[key]={'feature_vector':feature,'load_trial':load,'resample_trial':cache.resample_trial,
            'random_present_devices':baseline.random_present_devices,'drop_devices':baseline.drop_devices,
            'devices':tuple(sys.modules['imu_data'].DEVICES)}
    return context.public_rows[key]


def make_rf(protocol):
    expected=dict(n_estimators=400,max_depth=18,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,seed=RF_SEED)
    recipe=protocol.recipe.get('imu_teacher',{})
    if any(recipe.get(k,v)!=v for k,v in expected.items()):raise ValueError('source RF recipe drift')
    return RandomForestClassifier(**{k:v for k,v in expected.items() if k!='seed'},random_state=RF_SEED)


def dropout_training_data(features,masks,labels,ops):
    features=np.asarray(features,np.float32);masks=np.asarray(masks,np.float32);labels=np.asarray(labels,np.int64)
    if features.shape!=(len(labels),240) or masks.shape!=(len(labels),10):raise ValueError('RF feature/mask axes mismatch')
    chosen=ops['random_present_devices'](masks,np.random.default_rng(RF_SEED))
    dropped,dm=ops['drop_devices'](features,masks,chosen)
    return np.concatenate([np.c_[features,masks],np.c_[dropped,dm]]),np.concatenate([labels,labels]),chosen


def aligned_rf_logits(probabilities,classes,profile):
    if profile not in {'fit_teacher_target','terminal_export'}:raise ValueError('unknown RF consumer profile')
    classes=np.asarray(classes);probabilities=np.asarray(probabilities,np.float64)
    if (classes.ndim!=1 or not np.issubdtype(classes.dtype,np.integer) or len(set(classes.tolist()))!=len(classes)
            or np.any((classes<0)|(classes>=40)) or probabilities.shape!=(len(probabilities),len(classes))):
        raise ValueError('RF classes/probability columns mismatch')
    if not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):raise ValueError('RF probabilities invalid')
    floor=-1e6 if profile=='fit_teacher_target' else float(np.log(1e-12))
    scores=np.full((len(probabilities),40),floor,np.float64)
    scores[:,classes]=np.log(np.clip(probabilities,1e-12,1.))
    return scores


def statistics_for_row(row,ops):
    folder=Path(row['imu_path']) if row.get('imu_path') else Path('__missing_task7_imu__')
    if row['imu_available']=='1' and not folder.is_dir():raise ValueError('declared IMU folder missing')
    by_device=ops['load_trial'](folder)
    values,time_mask,device_mask,duration=ops['resample_trial'](by_device,32)
    features,masks=ops['feature_vector'](values,time_mask,device_mask)
    if features.shape!=(240,) or masks.shape!=(10,) or not np.isfinite(features).all() or not np.isfinite(masks).all():
        raise ValueError('source32 feature axes/nonfinite values')
    return features,masks,{'usable':bool(device_mask.any()),'device_count':int(device_mask.sum()),'duration_seconds':float(duration)}


def _ref(body):
    if isinstance(body,(str,Path)):body=json.loads(Path(body).read_text(encoding='utf-8'))
    return ArtifactRef(Path(body['record_path']),body['sha256'])


def _write_npz(path,arrays):
    stream=BytesIO();np.savez_compressed(stream,**arrays);content=stream.getvalue();immutable_bytes(path,content)
    return hashlib.sha256(content).hexdigest()


def build_rf_statistics(inputs,*,protocol,context):
    from .motion_cache import _source_descriptor,_stamp
    if inputs!=load_stage_inputs(protocol,inputs.partition.name,'raw',context=context) or inputs.labels is not None:
        raise ValueError('source32 statistics require current label-free inputs')
    registry=ArtifactRegistry(protocol,context=context);ops=source_rf_ops(protocol,context)
    rows=read_public_rows(inputs.public_manifest);index=row_index(rows);root=protocol.run_root/'imu_statistics32'/inputs.partition.name
    immutable_bytes(root/'identity.json',canonical_bytes({'protocol':protocol.identity(),'manifest':inputs.parents['manifest'],
        'source_files':_source_pins(protocol,context),'producer':context.hash_file(Path(__file__))})+b'\n')
    if (root/'artifact.json').exists():
        ref=_ref(root/'artifact.json');registry.verify(ref,'imu_statistics32','raw',index);return ref
    n=len(rows);features=np.zeros((n,240),np.float32);masks=np.zeros((n,10),np.float32)
    valid=np.zeros(n,bool);counts=np.zeros(n,np.int64);duration=np.zeros(n,np.float64);started=time.perf_counter()
    for i,row in enumerate(rows):
        features[i],masks[i],meta=statistics_for_row(row,ops)
        valid[i]=meta['usable'];counts[i]=meta['device_count'];duration[i]=meta['duration_seconds']
        if (i+1)%200==0 or i+1==n:print(json.dumps({'event':'rf_statistics_progress','partition':inputs.partition.name,
            'completed':i+1,'total':n,'seconds':round(time.perf_counter()-started,2)}),flush=True)
    path=root/'statistics.npz';digest=_write_npz(path,{'sample_ids':np.asarray(index.sample_ids),'features':features,
        'masks':masks,'valid':valid,'device_count':counts,'duration_seconds':duration})
    ref=registry.register(stage='imu_statistics32',kind='raw_cache',phase='raw',files=[path],file_digests={str(path.resolve()):digest},
        parents=[inputs.parents['manifest'],_source_descriptor(protocol,context)],rows=index,
        config={'partition':inputs.partition.name,'bins':32,'features':240,'mask_features':10,'source_loader':'build_imu_cache.load_trial',
            'not_p31_motion_statistics':True,'payload_stamps':{path.name:_stamp(path)}},source_files=[Path(__file__)])
    write_json(root/'artifact.json',ref);return ref


def _source_pins(protocol,context):
    key=(protocol.identity(),'rf_source_pins')
    if key not in context.public_rows:
        manifest=json.loads((protocol.source_root.parent/'source_manifest.json').read_text(encoding='utf-8'))
        pins={x['path']:x['sha256'] for x in manifest['files']};result={}
        for name in SOURCE_FILES:
            rel='aligned_multimodal/'+name;result[rel]=context.hash_file(protocol.source_root/rel,pins[rel])
        context.public_rows[key]=result
    return context.public_rows[key]


def load_rf_statistics(ref,index,*,protocol,context):
    from .motion_cache import _bounded_payload
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(ref,'imu_statistics32','raw')
    if not record.complete or record.rows is None:raise ValueError('RF statistics must be complete')
    if record.config.get('bins')!=32 or record.config.get('not_p31_motion_statistics') is not True:raise ValueError('wrong RF statistic source')
    files=[Path(x) for x in record.files if Path(x).name=='statistics.npz']
    if len(files)!=1:raise ValueError('one source32 statistics payload required')
    arrays=context.load_arrays(_bounded_payload(record,files[0],context))
    if tuple(arrays['sample_ids'].astype(str))!=record.rows.sample_ids:raise ValueError('RF statistic IDs mismatch')
    lookup={s:i for i,s in enumerate(record.rows.sample_ids)}
    if any(s not in lookup or record.rows.user_ids[lookup[s]]!=u for s,u in zip(index.sample_ids,index.user_ids)):
        raise ValueError('RF statistic population/users mismatch')
    positions=np.array([lookup[s] for s in index.sample_ids],np.int64)
    selected={k:v[positions] for k,v in arrays.items()}
    n=len(index.sample_ids)
    if (selected['features'].shape!=(n,240) or selected['masks'].shape!=(n,10) or selected['valid'].shape!=(n,)
            or selected['valid'].dtype!=np.bool_ or not np.array_equal(selected['valid'],selected['masks'][:,::2].any(1))):
        raise ValueError('RF statistic axes/valid mask mismatch')
    if not np.isfinite(selected['features']).all() or not np.isfinite(selected['masks']).all():raise ValueError('nonfinite RF statistics')
    return selected


def prepare_rf_capability(protocol,statistics,context):
    from .motion_cache import _source_descriptor
    source_rf_ops(protocol,context);make_rf(protocol)
    marker=protocol.run_root/'protocol/task6_complete.json'
    proof=json.loads(marker.read_text(encoding='utf-8'))
    if proof.get('complete') is not True or proof.get('protocol_sha256')!=protocol.identity():raise ValueError('Task6 completion binding mismatch')
    registry=ArtifactRegistry(protocol,context=context)
    for pop in ['refit14','final4']:
        registry.verify(statistics[pop],'imu_statistics32','raw',row_index(read_public_rows(protocol.public_manifests[pop])))
    body={'schema_version':3,'scope':'task7_imu_rf_only','protocol_sha256':protocol.identity(),'stages':['source_expert/imu_rf'],
        'task6_complete_sha256':context.hash_file(marker),'source_files':_source_pins(protocol,context),
        'statistics':statistics,'seed':RF_SEED,'global_formal_generation_ready':False,'oof_enabled':False,'training_executed':False}
    path=protocol.run_root/'protocol/task7_rf_capability.json';immutable_bytes(path,canonical_bytes(body)+b'\n')
    ref=registry.register(stage='imu_rf_readiness',kind='descriptor',phase='public',files=[path],
        parents=[_source_descriptor(protocol,context),statistics['refit14'],statistics['final4']],
        config={'source_contract_version':3},source_files=[Path(__file__)])
    write_json(path.with_suffix('.artifact.json'),ref);return ref


def validate_rf_capability(registry,record):
    p=registry.protocol;path=p.run_root/'protocol/task7_rf_capability.json'
    if list(record.files)!=[str(path)] or len(record.parents)!=3:raise ValueError('RF readiness payload/parents mismatch')
    parents=[registry.read(x) for x in record.parents]
    if [x.stage for x in parents]!=['source_ops','imu_statistics32','imu_statistics32'] or any(x.kind not in {'descriptor','raw_cache'} for x in parents):
        raise ValueError('RF readiness cannot inherit learned models')
    body=json.loads(path.read_text(encoding='utf-8'));context=registry.context or VerificationContext(p)
    expected={'schema_version':3,'scope':'task7_imu_rf_only','protocol_sha256':p.identity(),'stages':['source_expert/imu_rf'],
        'task6_complete_sha256':context.hash_file(p.run_root/'protocol/task6_complete.json'),'source_files':_source_pins(p,context),
        'statistics':{k:record.parents[i+1] for i,k in enumerate(['refit14','final4'])},'seed':RF_SEED,
        'global_formal_generation_ready':False,'oof_enabled':False,'training_executed':False}
    if canonical_bytes(body)!=canonical_bytes(expected):raise ValueError('RF readiness source/completion binding drift')
    for pop,parent in zip(['refit14','final4'],parents[1:]):
        if parent.kind!='raw_cache' or not parent.complete or parent.config.get('partition')!=pop:raise ValueError('RF readiness statistics phase mismatch')


def allows_rf_stage(registry,record):
    if record.stage!='source_expert/imu_rf' or record.kind!='supervised_model' or record.phase not in {'select','refit'}:return False
    caps=[x for x in record.parents if registry.read(x).stage=='imu_rf_readiness']
    if len(caps)!=1:return False
    cap=registry.read(caps[0]);validate_rf_capability(registry,cap);make_rf(registry.protocol)
    body=json.loads(next(Path(x) for x in cap.files).read_text(encoding='utf-8'))
    cache=_ref(body['statistics']['refit14'])
    return (cache in record.parents and record.config.get('seed')==RF_SEED
        and record.config.get('n_estimators')==400 and record.config.get('target_origin')=='in_sample_non_oof')


def train_imu_rf(phase,fit,statistics,capability,*,protocol,context):
    from .visual_teacher import _labels
    required={'select':'train12','refit':'refit14'}
    if phase not in required or fit.partition.name!=required[phase]:raise ValueError('RF fit phase/population mismatch')
    index,labels=_labels(fit,phase,protocol);registry=ArtifactRegistry(protocol,context=context)
    registry.verify(capability,'imu_rf_readiness','public');arrays=load_rf_statistics(statistics,index,protocol=protocol,context=context)
    ops=source_rf_ops(protocol,context);valid=arrays['valid']
    if not valid.any():raise ValueError('no usable fit IMU samples')
    root=protocol.run_root/'A4'/phase
    identity={'protocol':protocol.identity(),'phase':phase,'fit':fit,'statistics':statistics,'readiness':capability,
        'producer':context.hash_file(Path(__file__)),'labels_sha256':context.hash_file(fit.labels)}
    immutable_bytes(root/'identity.json',canonical_bytes(identity)+b'\n')
    if (root/'artifact.json').exists():ref=_ref(root/'artifact.json');registry.verify(ref,'source_expert/imu_rf',phase,index);return ref
    source,y,chosen=dropout_training_data(arrays['features'][valid],arrays['masks'][valid],labels[valid],ops)
    model=make_rf(protocol);started=time.perf_counter()
    print(json.dumps({'event':'rf_fit_start','phase':phase,'valid_rows':int(valid.sum()),'augmented_rows':len(y),'trees':400}),flush=True)
    model.fit(source,y)
    checkpoint=root/'imu_random_forest.joblib';temp=checkpoint.with_suffix('.joblib.building');joblib.dump(model,temp,compress=3);temp.replace(checkpoint)
    prior=np.bincount(labels,minlength=40)/len(labels)
    config={'phase':phase,'partition':required[phase],'seed':RF_SEED,'n_estimators':400,'max_depth':18,'min_samples_leaf':2,
        'max_features':'sqrt','class_weight':'balanced_subsample','n_jobs':-1,'source_bins':32,'feature_width':250,
        'target_origin':'in_sample_non_oof','dropout_rng':'source_seed_no_fold','valid_fit_rows':int(valid.sum()),
        'augmented_fit_rows':len(y),'classes':model.classes_.astype(int).tolist(),'prior':prior.tolist(),
        'fit_ids_sha256':canonical_hash(index),'dropped_devices':chosen.tolist(),'fit_seconds':round(time.perf_counter()-started,2)}
    path=root/'model.json';write_json(path,config)
    label=registry.register(stage='labels_'+fit.partition.name,kind='supervised_labels',phase=phase,files=[fit.labels],
        parents=[fit.parents['manifest']],rows=index,config={'partition':fit.partition.name})
    ref=registry.register(stage='source_expert/imu_rf',kind='supervised_model',phase=phase,files=[checkpoint,path],
        parents=[statistics,label,capability],fit_users=fit.partition.users,rows=index,config=config,source_files=[Path(__file__)])
    write_json(root/'artifact.json',ref)
    print(json.dumps({'event':'rf_fit_complete','phase':phase,'seconds':config['fit_seconds'],'classes':len(config['classes'])}),flush=True)
    return ref


def predict_imu_teacher(model,statistics,partition,*,protocol,context,profile='terminal_export'):
    registry=ArtifactRegistry(protocol,context=context);mr=registry.verify(model,'source_expert/imu_rf')
    expected='select' if partition in {'train12','development2'} else 'refit'
    if partition not in protocol.partitions or mr.phase!=expected:raise ValueError('RF prediction phase/population mismatch')
    inputs=load_stage_inputs(protocol,partition,'predict',context=context);index=row_index(read_public_rows(inputs.public_manifest))
    arrays=load_rf_statistics(statistics,index,protocol=protocol,context=context);root=protocol.run_root/'A4'/mr.phase/profile
    path=root/(partition+'_targets.npz');refpath=path.with_suffix('.artifact.json')
    if refpath.exists():
        ref=_ref(refpath);record=registry.verify(ref,'source_expert/imu_rf','predict',index)
        if record.config.get('consumer_profile')!=profile or model not in record.parents or statistics not in record.parents:raise ValueError('RF output binding drift')
        data=context.load_arrays(registry.verify_file(record,path))
        return TeacherTargets(Prediction(index,data['logits'],data['probabilities'],data['valid'],ref),None)
    prior=np.asarray(mr.config['prior'],np.float64);valid=arrays['valid'];scores=np.tile(np.log(np.maximum(prior,1e-12)),(len(valid),1))
    key=(protocol.identity(),'rf_model',model.sha256)
    if key not in context.public_rows:
        file=next(Path(x) for x in mr.files if Path(x).suffix=='.joblib');registry.verify_file(mr,file)
        context.public_rows[key]=joblib.load(file)
    network=context.public_rows[key]
    if network.classes_.astype(int).tolist()!=list(mr.config['classes']):raise ValueError('RF stored classes mismatch')
    if valid.any():
        probabilities=network.predict_proba(np.c_[arrays['features'][valid],arrays['masks'][valid]])
        scores[valid]=aligned_rf_logits(probabilities,network.classes_,profile)
    probs=softmax(scores,axis=1);probs[~valid]=prior
    digest=_write_npz(path,{'sample_ids':np.asarray(index.sample_ids),'class_ids':np.arange(40),'logits':scores,'probabilities':probs,'valid':valid})
    ref=registry.register(stage='source_expert/imu_rf',kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
        parents=[model,statistics,inputs.parents['manifest']],rows=index,
        config={'partition':partition,'consumer_profile':profile,'logit_temperature':1.,'target_origin':'in_sample_non_oof' if partition in {'train12','refit14'} else 'fixed_held_prediction',
            'fit_phase':mr.phase,'invalid_policy':'same_phase_fit_prior'},source_files=[Path(__file__)])
    write_json(refpath,ref);return TeacherTargets(Prediction(index,scores,probs,valid,ref),None)


def run_rf_pipeline(protocol):
    context=VerificationContext(protocol);statistics={}
    for partition in ['refit14','final4']:
        inputs=load_stage_inputs(protocol,partition,'raw',context=context)
        statistics[partition]=build_rf_statistics(inputs,protocol=protocol,context=context)
    cap=prepare_rf_capability(protocol,statistics,context);models={}
    for phase,partition in [('select','train12'),('refit','refit14')]:
        fit=load_stage_inputs(protocol,partition,phase,context=context)
        models[phase]=train_imu_rf(phase,fit,statistics['refit14'],cap,protocol=protocol,context=context)
    outputs={}
    for partition in protocol.partitions:
        phase='select' if partition in {'train12','development2'} else 'refit';pop='final4' if partition=='final4' else 'refit14'
        outputs[partition]={profile:predict_imu_teacher(models[phase],statistics[pop],partition,
            protocol=protocol,context=context,profile=profile) for profile in ['fit_teacher_target','terminal_export']}
    from .visual_teacher import _labels
    measurements={}
    for partition in ['train12','development2','refit14']:
        phase='refit' if partition=='refit14' else 'select'
        inputs=load_stage_inputs(protocol,partition,phase,context=context);index,y=_labels(inputs,phase,protocol)
        pred=outputs[partition]['terminal_export'].prediction;correct=pred.logits.argmax(1)==y
        measurements[partition]={'correct':int(correct.sum()),'total':len(y),'accuracy':float(correct.mean()),
            'valid':int(pred.valid.sum()),'valid_correct':int(correct[pred.valid].sum())}
    write_json(protocol.run_root/'A4/development_metrics.json',measurements)
    body={'protocol_sha256':protocol.identity(),'complete':True,'models':models,'statistics':statistics,'readiness':cap,
        'outputs':{pop:{key:target.prediction.artifact for key,target in profiles.items()} for pop,profiles in outputs.items()},
        'final_labels_read':False,'oof_enabled':False,'task8_started':False}
    immutable_bytes(protocol.run_root/'protocol/task7_artifacts_ready.json',canonical_bytes(body)+b'\n')
    print(json.dumps({'event':'rf_pipeline_complete','measurements':measurements,'final_accuracy':None}),flush=True)
    return {'models':models,'statistics':statistics,'outputs':outputs,'measurements':measurements,'readiness':cap}
