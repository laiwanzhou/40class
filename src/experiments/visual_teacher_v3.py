"""Fixed-split source visual heads. No legacy folds, predictions or label maps."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from threadpoolctl import threadpool_limits

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_cache_import import immutable_bytes
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .no_vote_types import ArtifactRef,Prediction,TeacherTargets,Selection,canonical_bytes,canonical_hash,write_json
from .teammate_source import verify_teammate_source,load_teammate_symbol,sha256_file

FAMILIES=('early','late','window_mean','early_late','temporal_delta','kinetics')
VISUAL_STAGES=frozenset({'A1','class_prior','visual_mechanism',*('visual_head/'+x for x in FAMILIES)})
SOURCE_FILES=('train_p46_videomae_head.py','train_p85_videomae_full40_head.py',
    'audit_p86_teacher_mechanisms.py','p89_build_multiexpert_test_logits.py','build_p85_videomae_fullwindow_cache.py')


def source_math(protocol,context):
    key=(protocol.identity(),'source_visual_math')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        symbols={'feature_sets':('train_p85_videomae_full40_head','feature_sets'),
            'aligned_scores':('train_p85_videomae_full40_head','aligned_scores_40'),
            'sample_weights':('train_p85_videomae_full40_head','sample_weights'),
            'make_model':('train_p46_videomae_head','make_model'),
            'fit_temperature':('train_p46_videomae_head','fit_temperature'),
            'perturbations':('audit_p86_teacher_mechanisms','fixed_model_perturbations')}
        context.public_rows[key]={k:load_teammate_symbol(report,*v) for k,v in symbols.items()}
    return context.public_rows[key]


def choose_source_candidate(rows):
    return max(rows,key=lambda r:(r['accuracy'],r['balanced_accuracy'],r['macro_f1'],
        -r['class_weight_power'],-r['alpha']))


def fit_calibrated_head(values,labels,alpha,power,*,protocol,context):
    ops=source_math(protocol,context);model=ops['make_model'](alpha)
    with threadpool_limits(limits=4):model.fit(values,labels,ridge__sample_weight=ops['sample_weights'](labels,power))
    temperature=ops['fit_temperature'](ops['aligned_scores'](model,values),labels)
    return model,temperature


def fit_mechanism_arrays(features,labels,*,protocol,context):
    ops=source_math(protocol,context);values=np.asarray(features,np.float32)
    norm=values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-8)
    model=ops['make_model'](3000.)
    with threadpool_limits(limits=4):model.fit(norm.reshape(len(norm),-1),labels,
        ridge__sample_weight=ops['sample_weights'](labels,.75))
    return model,norm.mean(axis=0)


def prepare_visual_capability(protocol,context):
    bindings=protocol.recipe.get('source_ops_bindings',{})
    if not bindings.get('manifest_path'):raise ValueError('visual source binding missing')
    proof_path=protocol.run_root/'protocol/task3_geometry_parity.json'
    if not proof_path.exists():raise ValueError('Task3 geometry proof missing')
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    if (proof.get('protocol_sha256')!=protocol.identity() or not proof.get('pixel_crop_quality_times_exact')
            or not proof.get('retry_reason_bits_exact') or proof.get('detector_executed') is not False):
        raise ValueError('Task3 geometry proof is invalid')
    from . import no_vote_source_ops
    registry=ArtifactRegistry(protocol,context=context)
    source=registry.register(stage='source_ops',kind='descriptor',phase='public',
        files=[Path(bindings['manifest_path'])],config={'source_contract_version':3},
        source_files=[Path(no_vote_source_ops.__file__)])
    source_math(protocol,context)
    manifest=json.loads((protocol.source_root.parent/'source_manifest.json').read_text(encoding='utf-8'))
    pinned={x['path']:x['sha256'] for x in manifest['files']};hashes={}
    for name in SOURCE_FILES:
        relative='aligned_multimodal/'+name;path=protocol.source_root/relative
        digest=context.hash_file(path,pinned[relative]);hashes[relative]=digest
    body={'schema_version':3,'training_executed':False,'oof_enabled':False,
        'protocol_sha256':protocol.identity(),'scope':'task4_visual_only','stages':sorted(VISUAL_STAGES),
        'source_ops_sha256':bindings['manifest_sha256'],'task3_proof_sha256':context.hash_file(proof_path),
        'source_files':hashes,'global_formal_generation_ready':False}
    path=protocol.run_root/'protocol/task4_visual_capability.json'
    immutable_bytes(path,canonical_bytes(body)+b'\n')
    ref=registry.register(stage='visual_readiness',kind='descriptor',phase='public',files=[path],parents=[source],
        config={'source_contract_version':3},source_files=[Path(__file__)])
    immutable_bytes(path.with_suffix('.artifact.json'),canonical_bytes(ref)+b'\n')
    return ref


def validate_visual_capability(registry,record):
    p=registry.protocol;bindings=p.recipe.get('source_ops_bindings',{})
    path=p.run_root/'protocol/task4_visual_capability.json'
    if list(record.files)!=[str(path)] or len(record.parents)!=1:raise ValueError('invalid visual source capability')
    parent=registry.read(record.parents[0])
    if (parent.kind!='descriptor' or parent.stage!='source_ops'
            or list(parent.files.values())!=[bindings.get('manifest_sha256')]):
        raise ValueError('visual source capability needs bound source_ops parent')
    body=json.loads(path.read_text(encoding='utf-8'))
    proof=p.run_root/'protocol/task3_geometry_parity.json'
    digest=registry.context.hash_file(proof) if registry.context else sha256_file(proof)
    expected={'schema_version':3,'training_executed':False,'oof_enabled':False,'protocol_sha256':p.identity(),
        'scope':'task4_visual_only','stages':sorted(VISUAL_STAGES),'source_ops_sha256':bindings.get('manifest_sha256'),
        'task3_proof_sha256':digest,'global_formal_generation_ready':False}
    if any(body.get(k)!=v for k,v in expected.items()):raise ValueError('visual source capability binding drift')
    manifest=json.loads((p.source_root.parent/'source_manifest.json').read_text(encoding='utf-8'))
    pinned={x['path']:x['sha256'] for x in manifest['files']}
    if body.get('source_files')!={'aligned_multimodal/'+x:pinned['aligned_multimodal/'+x] for x in SOURCE_FILES}:
        raise ValueError('visual source capability source pins mismatch')


def allows_visual_stage(registry,record):
    if record.stage not in VISUAL_STAGES or record.kind not in {'supervised_model','statistics','calibration'}:return False
    capabilities=[x for x in record.parents if registry.read(x).stage=='visual_readiness']
    if len(capabilities)!=1:return False
    cap=registry.read(capabilities[0])
    if cap.kind!='descriptor' or cap.phase!='public':return False
    validate_visual_capability(registry,cap)
    return True


def capability(protocol,context):
    path=protocol.run_root/'protocol/task4_visual_capability.artifact.json'
    if not path.exists():raise ValueError('visual source capability not prepared')
    body=json.loads(path.read_text(encoding='utf-8'));ref=ArtifactRef(Path(body['record_path']),body['sha256'])
    ArtifactRegistry(protocol,context=context).verify(ref,'visual_readiness','public')
    return ref


def _table(features,index,protocol,context):
    key=(protocol.identity(),'visual_subset',features.sha256,canonical_hash(index))
    if key not in context.public_rows:
        from .visual_teacher import _load_features,validate_feature_arrays
        record,arrays=_load_features(features,protocol,context=context)
        lookup={sid:i for i,sid in enumerate(record.rows.sample_ids)}
        if not set(index.sample_ids)<=set(lookup):raise ValueError('visual feature IDs missing')
        positions=np.array([lookup[sid] for sid in index.sample_ids])
        if tuple(record.rows.user_ids[i] for i in positions)!=index.user_ids:raise ValueError('visual feature users mismatch')
        selected={k:v[positions] if k!='class_ids' else v for k,v in arrays.items()}
        validate_feature_arrays(selected,index,bounded=True)
        context.public_rows[key]=(record,selected)
    return context.public_rows[key]


def _matrices(features,index,protocol,context):
    key=(protocol.identity(),'visual_matrices',features.sha256,canonical_hash(index))
    if key not in context.public_rows:
        _,arrays=_table(features,index,protocol,context)
        context.public_rows[key]=source_math(protocol,context)['feature_sets'](arrays['features'],arrays['kinetics_logits'])
    return context.public_rows[key]


def _label_ref(inputs,index,phase,protocol,context):
    return ArtifactRegistry(protocol,context=context).register(stage='labels_'+inputs.partition.name,kind='supervised_labels',
        phase=phase,files=[inputs.labels],parents=[inputs.parents['manifest']],rows=index,config={'partition':inputs.partition.name})


def _labels(inputs,phase,protocol):
    from .visual_teacher import _labels as read
    required={'select':'train12','refit':'refit14'}[phase]
    if inputs.partition.name!=required:raise ValueError('visual fit population/phase mismatch')
    return read(inputs,phase,protocol)


def _save(model,cfg,stage,inputs,index,features,phase,protocol,context,*,extra_files=()):
    registry=ArtifactRegistry(protocol,context=context);cap=capability(protocol,context)
    root=protocol.run_root/('A1' if stage=='A1' else 'visual_heads'/Path(stage.split('/',1)[-1]))/phase
    root.mkdir(parents=True,exist_ok=True)
    if (root/'head.joblib').exists():raise ValueError('visual model is immutable; load existing artifact instead')
    joblib.dump(model,root/'head.joblib',compress=3);write_json(root/'head.json',cfg)
    ref=registry.register(stage=stage,kind='supervised_model',phase=phase,files=[root/'head.joblib',root/'head.json',*extra_files],
        parents=[features,_label_ref(inputs,index,phase,protocol,context),cap],fit_users=inputs.partition.users,
        select_users=protocol.partitions['development2'].users if phase=='select' and stage.startswith('visual_head/') else (),
        rows=index,config=cfg,source_files=[Path(__file__)])
    write_json(root/'artifact.json',ref);return ref


def _alias(primary,family_configs,inputs,index,protocol,context):
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(primary)
    cfg={**record.config,'primary_family':'early_late','family_configs':family_configs}
    ref=registry.register(stage='A1',kind='supervised_model',phase=record.phase,files=record.files,
        parents=[primary,capability(protocol,context)],fit_users=inputs.partition.users,rows=index,config=cfg,
        source_files=[Path(__file__)])
    write_json(protocol.run_root/'A1'/record.phase/'artifact.json',ref)
    return ref


def fit_prior(inputs,phase,*,protocol,context):
    index,y=_labels(inputs,phase,protocol);counts=np.bincount(y,minlength=40);prob=counts/counts.sum()
    path=protocol.run_root/'A1'/phase/'prior.json'
    immutable_bytes(path,canonical_bytes({'class_ids':list(range(40)),'counts':counts.tolist(),'probabilities':prob.tolist()})+b'\n')
    ref=ArtifactRegistry(protocol,context=context).register(stage='class_prior',kind='statistics',phase=phase,files=[path],
        parents=[_label_ref(inputs,index,phase,protocol,context),capability(protocol,context)],fit_users=inputs.partition.users,
        rows=index,config={'partition':inputs.partition.name},source_files=[Path(__file__)])
    write_json(path.parent/'prior_artifact.json',ref);return ref


def select_heads(train,development,features,*,protocol,context):
    capability(protocol,context)
    from .visual_teacher import _labels as labels
    index,y=_labels(train,'select',protocol);dev,dy=labels(development,'select',protocol)
    if development.partition.name!='development2':raise ValueError('visual selection requires fixed development2')
    record,arrays=_table(features,index,protocol,context);_,da=_table(features,dev,protocol,context)
    tm=_matrices(features,index,protocol,context);dm=_matrices(features,dev,protocol,context)
    valid=arrays['valid'];dv=da['valid'];ops=source_math(protocol,context)
    if set(y[valid])!=set(range(40)):raise ValueError('visual fit IR rows lack 40-class support')
    prior=np.bincount(y,minlength=40)/len(y);results=[];models={};configs={}
    recipe=protocol.recipe['visual_teacher']
    if tuple(recipe['families'])!=FAMILIES:raise ValueError('source visual family order mismatch')
    for family in FAMILIES:
        candidates=[];best_model=None
        for power in recipe['class_weight_powers']:
            for alpha in recipe['alphas']:
                model=ops['make_model'](alpha)
                with threadpool_limits(limits=4):model.fit(tm[family][valid],y[valid],ridge__sample_weight=ops['sample_weights'](y[valid],power))
                scores=np.tile(np.log(np.maximum(prior,1e-12)),(len(dy),1))
                if dv.any():scores[dv]=ops['aligned_scores'](model,dm[family][dv])
                pred=scores.argmax(1)
                result=dict(family=family,alpha=float(alpha),class_weight_power=float(power),
                    accuracy=float(accuracy_score(dy,pred)),balanced_accuracy=float(balanced_accuracy_score(dy,pred)),
                    macro_f1=float(f1_score(dy,pred,average='macro',zero_division=0)))
                candidates.append(result)
                if choose_source_candidate(candidates) is result:best_model=model
                print(json.dumps({'event':'head_candidate','candidate':len(results)+len(candidates),'total':72,**result}),flush=True)
        best=choose_source_candidate(candidates);results+=candidates
        temperature=ops['fit_temperature'](ops['aligned_scores'](best_model,tm[family][valid]),y[valid])
        hyper={k:best[k] for k in ['family','alpha','class_weight_power']};configs[family]=hyper
        cfg={**hyper,'head_temperature':temperature,'feature_signature':record.config['feature_signature'],
            'partition':'train12','calibration_partition':'train12','calibration_origin':'in_sample_non_oof',
            'target_origin':'fixed_split_fit_predictions_not_oof','fit_ids_sha256':canonical_hash(index)}
        models[family]=_save(best_model,cfg,'visual_head/'+family,train,index,features,'select',protocol,context)
    primary=_alias(models['early_late'],configs,train,index,protocol,context)
    best=choose_source_candidate([r for r in results if r['family']=='early_late'])
    selection=Selection('A1',{'family':'early_late','family_configs':configs},{'candidates':72,'families':6},
        {k:best[k] for k in ['accuracy','balanced_accuracy','macro_f1']},primary,canonical_hash(dev))
    write_json(protocol.run_root/'A1/select/grid.json',results);selection.write(protocol.run_root/'A1/select/selection.json')
    return selection


def refit_heads(inputs,features,selection,*,protocol,context):
    registry=ArtifactRegistry(protocol,context=context);chosen=registry.verify(selection.fit_artifact,'A1','select')
    if selection.stage!='A1' or dict(selection.config['family_configs'])!=dict(chosen.config['family_configs']):
        raise ValueError('visual selection config mismatch')
    if selection.development_ids_sha256!=canonical_hash(row_index(read_public_rows(protocol.public_manifests['development2']))):
        raise ValueError('visual selection development IDs drift')
    index,y=_labels(inputs,'refit',protocol);record,arrays=_table(features,index,protocol,context)
    if record.config['feature_signature']!=chosen.config['feature_signature']:raise ValueError('visual feature signature mismatch')
    matrices=_matrices(features,index,protocol,context);valid=arrays['valid'];models={}
    if set(y[valid])!=set(range(40)):raise ValueError('refit visual IR rows lack classes')
    configs=selection.config['family_configs']
    for family in FAMILIES:
        hyper=dict(configs[family]);model,t=fit_calibrated_head(matrices[family][valid],y[valid],hyper['alpha'],
            hyper['class_weight_power'],protocol=protocol,context=context)
        cfg={**hyper,'head_temperature':t,'feature_signature':record.config['feature_signature'],
            'partition':'refit14','calibration_partition':'refit14','calibration_origin':'in_sample_non_oof',
            'target_origin':'fixed_split_fit_predictions_not_oof','fit_ids_sha256':canonical_hash(index)}
        models[family]=_save(model,cfg,'visual_head/'+family,inputs,index,features,'refit',protocol,context)
        print(json.dumps({'event':'head_refit','family':family,'temperature':t}),flush=True)
    return _alias(models['early_late'],configs,inputs,index,protocol,context)


def _predict_inputs(model,rows,prior,protocol,context):
    registry=ArtifactRegistry(protocol,context=context);mr=registry.verify(model)
    pr=registry.verify(prior,'class_prior',mr.phase)
    allowed=['train12','development2'] if mr.phase=='select' else ['refit14','final4']
    population=next((x for x in allowed if rows==row_index(read_public_rows(protocol.public_manifests[x]))),None)
    if population is None:raise ValueError('visual prediction phase/population mismatch')
    body=json.loads(next(Path(x) for x in pr.files if x.endswith('.json')).read_text(encoding='utf-8'))
    return registry,mr,np.asarray(body['probabilities'],np.float64),population


def predict_head(model,features,rows,prior,*,protocol,context):
    registry,mr,prob,pop=_predict_inputs(model,rows,prior,protocol,context)
    if mr.stage!='A1' and not mr.stage.startswith('visual_head/'):raise ValueError('not a visual head')
    record,arrays=_table(features,rows,protocol,context)
    if record.config['feature_signature']!=mr.config['feature_signature']:raise ValueError('visual features signature drift')
    head=joblib.load(next(Path(x) for x in mr.files if x.endswith('.joblib')))
    raw=np.tile(np.log(np.maximum(prob,1e-12)),(len(rows.sample_ids),1));valid=arrays['valid']
    if valid.any():raw[valid]=source_math(protocol,context)['aligned_scores'](head,_matrices(features,rows,protocol,context)[mr.config['family']][valid])
    logits=raw.copy();logits[valid]/=mr.config['head_temperature'];probabilities=softmax(logits,axis=1);probabilities[~valid]=prob
    root=protocol.run_root/('A1' if mr.stage=='A1' else 'visual_heads'/Path(mr.stage.split('/',1)[1]))/mr.phase
    return _write_prediction(root,pop,mr.stage,model,features,prior,rows,raw,logits,probabilities,valid,protocol,context,
        arrays['features'] if mr.stage=='A1' else None)


def _write_prediction(root,pop,stage,model,features,prior,rows,raw,logits,prob,valid,protocol,context,teacher_features=None,*,extra_config=None):
    from .pose_roi_adapter import _atomic_npz
    path=root/(pop+'_targets.npz');raw_path=root/(pop+'_raw_scores.npy')
    if path.exists() or raw_path.exists():raise ValueError('visual predictions immutable; use existing artifact')
    root.mkdir(parents=True,exist_ok=True);np.save(raw_path,raw.astype(np.float32))
    _atomic_npz(path,dict(sample_ids=np.asarray(rows.sample_ids),class_ids=np.arange(40),logits=logits.astype(np.float32),
        probabilities=prob.astype(np.float32),valid=valid))
    inputs=load_stage_inputs(protocol,pop,'predict',context=context)
    ref=ArtifactRegistry(protocol,context=context).register(stage=stage,kind='predictions',phase='predict',files=[path,raw_path],
        parents=[model,features,prior,inputs.parents['manifest']],rows=rows,
        config={'partition':pop,'target_origin':'fixed_split_fit_predictions_not_oof',**(extra_config or {})},source_files=[Path(__file__)])
    write_json(path.with_suffix('.artifact.json'),ref)
    prediction=Prediction(rows,logits,prob,valid,ref)
    return TeacherTargets.from_validated_features(prediction,teacher_features) if teacher_features is not None else prediction


def fit_visual_mechanism_head(inputs,features,phase,*,protocol,context):
    capability(protocol,context);index,y=_labels(inputs,phase,protocol);record,arrays=_table(features,index,protocol,context)
    valid=arrays['valid'];model,mean=fit_mechanism_arrays(arrays['features'][valid],y[valid],protocol=protocol,context=context)
    root=protocol.run_root/'visual_heads/visual_mechanism'/phase;root.mkdir(parents=True,exist_ok=True)
    mean_file=root/'fit_feature_mean.npy'
    if mean_file.exists():raise ValueError('mechanism mean immutable')
    np.save(mean_file,mean)
    cfg={'alpha':3000.,'class_weight_power':.75,'head_temperature':1.,'partition':inputs.partition.name,
        'feature_signature':record.config['feature_signature'],'mean_origin':'valid_fit_l2_features_only',
        'target_origin':'fixed_split_fit_predictions_not_oof','backbone_root':'videomae_large'}
    return _save(model,cfg,'visual_mechanism',inputs,index,features,phase,protocol,context,extra_files=[mean_file])


def predict_visual_mechanisms(model,features,rows,prior,*,protocol,context):
    registry,mr,prob,pop=_predict_inputs(model,rows,prior,protocol,context)
    if mr.stage!='visual_mechanism':raise ValueError('mechanism requires own fixed Ridge')
    record,arrays=_table(features,rows,protocol,context)
    if record.config['feature_signature']!=mr.config['feature_signature']:raise ValueError('mechanism signature drift')
    head=joblib.load(next(Path(x) for x in mr.files if x.endswith('.joblib')))
    mean=np.load(next(Path(x) for x in mr.files if x.endswith('.npy')),allow_pickle=False)
    values=np.asarray(arrays['features'],np.float32);values=values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-8)
    ops=source_math(protocol,context);result={};valid=arrays['valid']
    for name,perturbed in ops['perturbations'](values,mean).items():
        logits=np.tile(np.log(np.maximum(prob,1e-12)),(len(rows.sample_ids),1))
        if valid.any():logits[valid]=ops['aligned_scores'](head,perturbed[valid].reshape(valid.sum(),-1))
        probs=softmax(logits,axis=1);probs[~valid]=prob
        root=protocol.run_root/'visual_mechanisms'/mr.phase/name
        result[name]=_write_prediction(root,pop,'visual_mechanism',model,features,prior,rows,logits,logits,probs,valid,
            protocol,context,extra_config={'perturbation':name,'head_temperature':1.})
    return result


def read_ref(path):
    body=json.loads(Path(path).read_text(encoding='utf-8'))
    return ArtifactRef(Path(body['record_path']),body['sha256'])


def run_head_pipeline(protocol,context):
    """One stage context for six families, two fit phases and all predictions."""
    prepare_visual_capability(protocol,context)
    registry=ArtifactRegistry(protocol,context=context)
    receipt=json.loads((protocol.run_root/'protocol/task2_prepared.json').read_text(encoding='utf-8'))
    features={item['partition']:ArtifactRef(Path(item['record_path']),item['sha256'])
        for item in receipt['imports'] if item['stage']=='visual_features'}
    path=protocol.run_root/'A1/select/selection.json'
    if path.exists():
        body=json.loads(path.read_text(encoding='utf-8'))
        body['fit_artifact']=ArtifactRef(Path(body['fit_artifact']['record_path']),body['fit_artifact']['sha256'])
        selection=Selection(**body);registry.verify(selection.fit_artifact,'A1','select')
    else:
        selection=select_heads(load_stage_inputs(protocol,'train12','select',context=context),
            load_stage_inputs(protocol,'development2','select',context=context),features['refit14'],protocol=protocol,context=context)
    refit=protocol.run_root/'A1/refit/artifact.json'
    if refit.exists():registry.verify(read_ref(refit),'A1','refit')
    else:refit_heads(load_stage_inputs(protocol,'refit14','refit',context=context),features['refit14'],selection,
        protocol=protocol,context=context)
    outputs={}
    for phase,fit_partition,populations in [('select','train12',['train12','development2']),('refit','refit14',['refit14','final4'])]:
        inputs=load_stage_inputs(protocol,fit_partition,phase,context=context)
        prior_path=protocol.run_root/'A1'/phase/'prior_artifact.json'
        prior=read_ref(prior_path) if prior_path.exists() else fit_prior(inputs,phase,protocol=protocol,context=context)
        registry.verify(prior,'class_prior',phase)
        mechanism_path=protocol.run_root/'visual_heads/visual_mechanism'/phase/'artifact.json'
        mechanism=read_ref(mechanism_path) if mechanism_path.exists() else fit_visual_mechanism_head(inputs,features['refit14'],phase,protocol=protocol,context=context)
        for population in populations:
            rows=row_index(read_public_rows(load_stage_inputs(protocol,population,'predict',context=context).public_manifest))
            feature=features['final4' if population=='final4' else 'refit14']
            root=protocol.run_root/'A1'/phase
            if not (root/(population+'_targets.artifact.json')).exists():
                result=predict_head(read_ref(root/'artifact.json'),feature,rows,prior,protocol=protocol,context=context)
            for family in FAMILIES:
                root=protocol.run_root/'visual_heads'/family/phase
                if not (root/(population+'_targets.artifact.json')).exists():
                    predict_head(read_ref(root/'artifact.json'),feature,rows,prior,protocol=protocol,context=context)
            mechanism_root=protocol.run_root/'visual_mechanisms'/phase
            if not all((mechanism_root/name/(population+'_targets.artifact.json')).exists()
                    for name in source_math(protocol,context)['perturbations'](np.zeros((1,2,3,1024),np.float32),np.zeros((2,3,1024),np.float32))):
                predict_visual_mechanisms(mechanism,feature,rows,prior,protocol=protocol,context=context)
            target_path=protocol.run_root/'A1'/phase/(population+'_targets.npz')
            target=read_ref(target_path.with_suffix('.artifact.json'));registry.verify(target,'A1','predict',rows)
            with np.load(target_path,allow_pickle=False) as data:
                probabilities=data['probabilities'];valid=data['valid']
                confidence=float(probabilities[valid].max(1).mean())
                entropy=float(-(probabilities[valid]*np.log(np.maximum(probabilities[valid],1e-30))).sum(1).mean())
                correct=None
                if population!='final4':
                    from .visual_teacher import _labels as read_labels
                    idx,y=read_labels(load_stage_inputs(protocol,population,phase if population==fit_partition else 'select'),
                        phase if population==fit_partition else 'select',protocol)
                    correct=int((probabilities.argmax(1)==y).sum())
            outputs[population]={'rows':len(rows.sample_ids),'valid':int(valid.sum()),'correct':correct,
                'mean_confidence_valid':confidence,'entropy_valid':entropy,'score_role':'fit' if population==fit_partition else 'development' if population=='development2' else 'unlabeled'}
            print(json.dumps({'event':'visual_targets_complete','partition':population,**outputs[population]}),flush=True)
    write_json(protocol.run_root/'protocol/task4_heads_complete.json',{'protocol_sha256':protocol.identity(),
        'primary_family':'early_late','head_families':6,'mechanisms':10,'partitions':outputs,
        'global_formal_generation_ready':False,'fullwindow_completed':False,'oof_enabled':False})
    return outputs
