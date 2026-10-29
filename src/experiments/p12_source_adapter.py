"""Fresh fixed-split P12 -> global mixture -> per-population KNN producers."""
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import log_softmax,softmax
from threadpoolctl import threadpool_limits

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_types import Prediction,canonical_hash,canonical_bytes,write_json
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .no_vote_cache_import import immutable_bytes
from .p12_reconstruction import REVISION,ref,load_reconstruction
from .p12_fusion import (p12_fusion_source_ops,sd_scores,fit_imu,apply_imu,fit_thermal,apply_thermal,
    fit_thermal_router,apply_thermal_router,apply_global,apply_knn)
from .visual_teacher_v3 import FAMILIES
from . import p12_fusion

P12_NAMES=('skeleton','depth','sd','imu','sd_imu','thermal','thermal_candidate','final')
CONTRACT='p12_global_fixed_fit_v1'
PRODUCER_FILES=(Path(__file__),Path(p12_fusion.__file__))


def _population(phase):
    if phase not in {'select','refit'}:raise ValueError('P12/global fit phase must be select or refit')
    return 'train12' if phase=='select' else 'refit14'


def compose_p12(scores,masks,device_counts,prior,state,ops):
    prior=np.asarray(prior,np.float64)
    if prior.shape!=(40,) or np.any(prior<0) or not np.isclose(prior.sum(),1):raise ValueError('P12 complete prior required')
    keys=('skeleton','depth','imu','thermal');values={k:np.asarray(scores[k],np.float64).copy() for k in keys}
    n=len(values['skeleton']);valid={k:np.asarray(masks[k]) for k in keys};log_prior=np.log(np.maximum(prior,1e-12))
    for k in keys:
        if values[k].shape!=(n,40) or not np.isfinite(values[k]).all() or valid[k].shape!=(n,) or valid[k].dtype!=np.bool_:
            raise ValueError('P12 source axes/finite/mask mismatch')
        values[k][~valid[k]]=log_prior
    sd=sd_scores(values['skeleton'],values['depth']);sd_valid=valid['skeleton']|valid['depth'];sd[~sd_valid]=log_prior
    base=apply_imu(sd,values['imu'],valid['imu'],device_counts,state['imu']);base_valid=sd_valid|valid['imu'];base[~base_valid]=log_prior
    candidate=apply_thermal(base,values['thermal'],valid['thermal'],state['thermal'],ops)
    final_valid=base_valid|valid['thermal'];candidate[~final_valid]=log_prior
    routed=apply_thermal_router(base,candidate,valid['thermal'],state['router'],ops)
    final=routed['logits'];final[~final_valid]=log_prior
    expert=np.stack([values['skeleton'],values['depth'],sd,values['imu'],base,values['thermal'],candidate,final],axis=1)
    expert_valid=np.column_stack([valid['skeleton'],valid['depth'],sd_valid,valid['imu'],base_valid,valid['thermal'],final_valid,final_valid])
    probability=softmax(final,axis=1);probability[~final_valid]=prior
    return {'expert_names':P12_NAMES,'expert_logits':expert,'expert_valid':expert_valid,'logits':final,
        'probabilities':probability,'valid':final_valid,'route':routed['route'],'route_probability':routed['route_probability']}


def _read_prediction(protocol,reference,stage,partition,context):
    registry=ArtifactRegistry(protocol,context=context);index=row_index(read_public_rows(protocol.public_manifests[partition]))
    record=registry.verify(reference,stage,'predict',index)
    if record.config.get('partition')!=partition:raise ValueError('source prediction partition mismatch')
    paths=[Path(x) for x in record.files if Path(x).suffix=='.npz']
    if len(paths)!=1:raise ValueError('one prediction table required')
    arrays=context.load_arrays(paths[0])
    if tuple(arrays['sample_ids'].tolist())!=index.sample_ids or not np.array_equal(arrays['class_ids'],np.arange(40)):
        raise ValueError('source table exact ID/class order mismatch')
    if arrays['logits'].shape!=(len(index.sample_ids),40) or arrays['valid'].shape!=(len(index.sample_ids),):raise ValueError('source prediction axes mismatch')
    return record,arrays


def _component_inputs(protocol,partition,context):
    index=row_index(read_public_rows(protocol.public_manifests[partition]));root=protocol.run_root/'protocol'/REVISION
    recipe,rr,_=load_reconstruction(protocol,context);scores={};masks={};parents=[]
    for role in ['skeleton','depth','thermal']:
        pointer_path=root/('p12_'+role+'_current.json')
        if not pointer_path.exists():raise ValueError('fresh '+role+' component is not complete')
        pointer=json.loads(pointer_path.read_text(encoding='utf-8'));path=Path(pointer['component_path']);context.hash_file(path,pointer['sha256'])
        body=json.loads(path.read_text(encoding='utf-8'))
        if body['base_protocol_sha256']!=protocol.identity() or ref(body['recipe'])!=rr:raise ValueError('P12 component recipe mismatch')
        reference=ref(body['outputs'][partition]);_,a=_read_prediction(protocol,reference,'source_expert/p12_'+role,partition,context)
        scores[role]=a['logits'];masks[role]=a['valid'];parents.append(reference)
    rf=json.loads((protocol.run_root/'protocol/task7_artifacts_ready.json').read_text(encoding='utf-8'))
    profile='fit_teacher_target' if partition in {'train12','refit14'} else 'terminal_export'
    reference=ref(rf['outputs'][partition][profile]);_,a=_read_prediction(protocol,reference,'source_expert/imu_rf',partition,context)
    scores['imu']=a['logits'];masks['imu']=a['valid'];parents.append(reference)
    from .imu_rf_teacher import load_rf_statistics
    statistic=ref(rf['statistics']['final4' if partition=='final4' else 'refit14'])
    arrays=load_rf_statistics(statistic,index,protocol=protocol,context=context)
    if not np.array_equal(masks['imu'],arrays['valid']):raise ValueError('RF prediction/device-count presence mismatch')
    return index,scores,masks,arrays['device_count'],parents,statistic


def _validated_components(protocol,partition,bundle,context):
    index,scores,masks,counts,parents,statistic=bundle
    if index!=row_index(read_public_rows(protocol.public_manifests[partition])) or len(parents)!=4:raise ValueError('P12 supplied input identity mismatch')
    for role,parent in zip(['skeleton','depth','thermal','imu'],parents):
        stage='source_expert/'+('imu_rf' if role=='imu' else 'p12_'+role)
        _,a=_read_prediction(protocol,parent,stage,partition,context)
        if not np.array_equal(scores[role],a['logits']) or not np.array_equal(masks[role],a['valid']):raise ValueError('P12 supplied arrays differ from parent payload')
    from .imu_rf_teacher import load_rf_statistics
    a=load_rf_statistics(statistic,index,protocol=protocol,context=context)
    if not np.array_equal(counts,a['device_count']) or not np.array_equal(masks['imu'],a['valid']):raise ValueError('P12 supplied device counts differ from parent payload')
    return bundle


def allows_p12_global_stage(registry,record):
    if record.stage not in {'p12_fusion','global_mixture'} or record.kind!='statistics' or record.phase not in {'select','refit'}:return False
    cfg=record.config;expected_count=7 if record.stage=='p12_fusion' else 10
    if cfg.get('revision')!=REVISION or cfg.get('operator_contract')!=CONTRACT or len(record.parents)!=expected_count:return False
    if record.stage=='p12_fusion' and tuple(cfg.get('p12_experts',()))!=P12_NAMES:return False
    if record.stage=='global_mixture' and (tuple(cfg.get('expert_names',()))!=P12_NAMES+FAMILIES
        or cfg.get('knn_neighbors')!=3 or cfg.get('knn_weight')!=.4 or cfg.get('knn_exclude_self') is not True):return False
    pop=_population(record.phase)
    if cfg.get('fit_partition')!=pop:return False
    p=registry.protocol;c=registry.context or VerificationContext(p);recipe,rr,_=load_reconstruction(p,c)
    index=row_index(read_public_rows(p.public_manifests[pop]));parents=[registry.read(x) for x in record.parents]
    if (record.rows!=index or rr not in record.parents or cfg.get('recipe_sha256')!=canonical_hash(recipe)
        or cfg.get('historical_exact_reproduction') is not False):return False
    required=[('reconstruction_recipe','raw_cache','raw'),('labels_'+pop,'supervised_labels',record.phase),('class_prior','statistics',record.phase)]
    stages=(['source_expert/p12_'+x for x in ['skeleton','depth','thermal']]+['source_expert/imu_rf'] if record.stage=='p12_fusion'
        else ['p12_fusion']+['visual_head/'+x for x in FAMILIES])
    required.extend((s,'predictions','predict') for s in stages)
    if not all(sum((x.stage,x.kind,x.phase)==r for x in parents)==1 for r in required):return False
    predictions=[x for x in parents if x.kind=='predictions']
    return all(x.rows==index and x.config.get('partition')==pop for x in predictions)


def build_p12_source_chain(protocol,phase,context,*,components=None):
    pop=_population(phase)
    from .visual_teacher import _labels
    from .visual_student_v3 import prior,_label_record
    recipe,rr,_=load_reconstruction(protocol,context);registry=ArtifactRegistry(protocol,context=context)
    root=protocol.run_root/REVISION/'p12_fusion'/phase;alias=root/'artifact.json'
    if alias.exists():
        result=ref(json.loads(alias.read_text(encoding='utf-8')));registry.verify(result,'p12_fusion',phase);return result
    index,scores,masks,counts,parents,_=(_validated_components(protocol,pop,components,context)
        if components is not None else _component_inputs(protocol,pop,context))
    inputs=load_stage_inputs(protocol,pop,phase,context=context);li,y=_labels(inputs,phase,protocol)
    if li!=index:raise ValueError('P12 fit labels order mismatch')
    pr,prob=prior(protocol,phase,context);ops=p12_fusion_source_ops(protocol,context)
    sd=sd_scores(scores['skeleton'],scores['depth']);common=(masks['skeleton']|masks['depth'])&masks['imu']
    state={'imu':fit_imu(sd,scores['imu'],y,common,ops)}
    base=apply_imu(sd,scores['imu'],masks['imu'],counts,state['imu']);base_valid=masks['skeleton']|masks['depth']|masks['imu']
    base[~base_valid]=np.log(np.maximum(prob,1e-12))
    state['thermal']=fit_thermal(base,scores['thermal'],y,masks['thermal'],ops)
    candidate=apply_thermal(base,scores['thermal'],masks['thermal'],state['thermal'],ops)
    state['router']=fit_thermal_router(base,candidate,y,masks['thermal'],ops)
    root.mkdir(parents=True,exist_ok=True);file=root/'parameters.joblib';joblib.dump(state,file,compress=3)
    cfg={'revision':REVISION,'operator_contract':CONTRACT,'fit_partition':pop,'recipe_sha256':canonical_hash(recipe),
        'historical_exact_reproduction':False,'p12_experts':P12_NAMES,'fit_rows':len(y),'imu_common_rows':int(common.sum()),
        'thermal_present_rows':int(masks['thermal'].sum()),'router_kind':state['router']['kind'],'sensitive_rows':state['router']['sensitive_rows'],
        'source_single_class_fallback':state['router'].get('source_degenerate_rule'),'fit_origin':'in_sample_non_oof',
        'missing_expert_policy':'same_phase_prior_keep_invalid_mask','fit_export_dtype':'source_float16','prediction_export_dtype':'float32'}
    write_json(root/'parameters.json',{**cfg,'imu':state['imu'],'thermal':state['thermal'],
        'router':{k:v for k,v in state['router'].items() if k!='model'}})
    lr=_label_record(inputs,phase,index,protocol,context)
    result=registry.register(stage='p12_fusion',kind='statistics',phase=phase,files=[file,root/'parameters.json'],
        parents=[rr,*parents,lr,pr],rows=index,fit_users=inputs.partition.users,config=cfg,source_files=PRODUCER_FILES)
    write_json(alias,result);return result


def predict_p12(protocol,model,partition,context,*,components=None):
    from .visual_student_v3 import prior
    from .imu_rf_teacher import _write_npz
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(model,'p12_fusion')
    phase='select' if partition in {'train12','development2'} else 'refit'
    if record.phase!=phase:raise ValueError('P12 prediction own-phase model required')
    index,scores,masks,counts,parents,statistics=(_validated_components(protocol,partition,components,context)
        if components is not None else _component_inputs(protocol,partition,context))
    root=protocol.run_root/REVISION/'p12_fusion'/phase;dest=root/(partition+'_targets.npz');alias=dest.with_suffix('.artifact.json')
    if alias.exists():
        result=ref(json.loads(alias.read_text(encoding='utf-8')));registry.verify(result,'p12_fusion','predict',index);return result
    _,prob=prior(protocol,phase,context);ops=p12_fusion_source_ops(protocol,context)
    file=next(Path(x) for x in record.files if Path(x).suffix=='.joblib');state=joblib.load(file)
    output=compose_p12(scores,masks,counts,prob,state,ops)
    table=output['expert_logits'].astype(np.float16 if partition==_population(phase) else np.float32)
    if not np.isfinite(table).all():raise ValueError('source P12 storage overflow; do not silently clip missing-class sentinels')
    logits=table[:,-1,:].astype(np.float32);probabilities=softmax(logits,axis=1);probabilities[~output['valid']]=prob
    payload={'sample_ids':np.asarray(index.sample_ids),'class_ids':np.arange(40),'expert_names':np.asarray(P12_NAMES),
        'expert_logits':table,'expert_valid':output['expert_valid'],'logits':logits,'probabilities':probabilities,
        'valid':output['valid'],'route':output['route'],'route_probability':output['route_probability']}
    digest=_write_npz(dest,payload);inputs=load_stage_inputs(protocol,partition,'predict',context=context)
    result=registry.register(stage='p12_fusion',kind='predictions',phase='predict',files=[dest],file_digests={str(dest.resolve()):digest},
        parents=[model,*parents,statistics,inputs.parents['manifest']],rows=index,config={'partition':partition,'revision':REVISION,
            'operator_contract':CONTRACT,'fit_phase':phase,'expert_names':P12_NAMES,'historical_exact_reproduction':False},source_files=PRODUCER_FILES)
    write_json(alias,result);return result


def _global_inputs(protocol,partition,context):
    phase='select' if partition in {'train12','development2'} else 'refit';root=protocol.run_root/REVISION/'p12_fusion'/phase
    p12=ref(json.loads((root/(partition+'_targets.artifact.json')).read_text(encoding='utf-8')))
    _,a=_read_prediction(protocol,p12,'p12_fusion',partition,context)
    if tuple(a['expert_names'].tolist())!=P12_NAMES:raise ValueError('source P12 expert column order changed')
    tables=[a['expert_logits'].astype(np.float64)];parents=[p12];valid=None
    for family in FAMILIES:
        file=protocol.run_root/'visual_heads'/family/phase/(partition+'_targets.artifact.json')
        reference=ref(json.loads(file.read_text(encoding='utf-8')));_,b=_read_prediction(protocol,reference,'visual_head/'+family,partition,context)
        if valid is None:valid=b['valid'].copy()
        elif not np.array_equal(valid,b['valid']):raise ValueError('visual family availability drift')
        tables.append(b['logits'][:,None,:].astype(np.float64));parents.append(reference)
    return np.concatenate(tables,axis=1),valid,a['probabilities'],parents


def fit_global_mixture(protocol,phase,context,*,components=None):
    pop=_population(phase)
    from .visual_teacher import _labels
    from .visual_student_v3 import prior,_label_record
    recipe,rr,_=load_reconstruction(protocol,context);registry=ArtifactRegistry(protocol,context=context)
    root=protocol.run_root/REVISION/'global_mixture'/phase;alias=root/'artifact.json'
    if alias.exists():
        result=ref(json.loads(alias.read_text(encoding='utf-8')));registry.verify(result,'global_mixture',phase);return result
    table,valid,_,parents=components if components is not None else _global_inputs(protocol,pop,context)
    if components is not None:
        if len(parents)!=7:raise ValueError('global requires P12 plus six visual parent predictions')
        source=[];source_valid=None
        for stage,parent in zip(['p12_fusion',*('visual_head/'+f for f in FAMILIES)],parents):
            _,a=_read_prediction(protocol,parent,stage,pop,context)
            source.append(a['expert_logits'].astype(np.float64) if stage=='p12_fusion' else a['logits'][:,None,:].astype(np.float64))
            if stage!='p12_fusion':
                if source_valid is None:source_valid=a['valid']
                elif not np.array_equal(source_valid,a['valid']):raise ValueError('global visual parent mask mismatch')
        if not np.array_equal(table,np.concatenate(source,axis=1)) or not np.array_equal(valid,source_valid):raise ValueError('global supplied arrays differ from parent payload')
    inputs=load_stage_inputs(protocol,pop,phase,context=context);index,y=_labels(inputs,phase,protocol)
    if len(table)!=len(y) or not valid.any():raise ValueError('global fit input rows mismatch or unavailable')
    ops=p12_fusion_source_ops(protocol,context)
    with threadpool_limits(limits=4):weights=ops['fit_simplex'](log_softmax(table[valid],axis=2),y[valid])
    pr,_=prior(protocol,phase,context);lr=_label_record(inputs,phase,index,protocol,context)
    cfg={'revision':REVISION,'operator_contract':CONTRACT,'recipe_sha256':canonical_hash(recipe),'fit_partition':pop,
        'historical_exact_reproduction':False,'expert_names':P12_NAMES+FAMILIES,'weights':weights.tolist(),
        'fit_rows':int(valid.sum()),'fit_policy':'available_visual_details_only','fit_origin':'in_sample_non_oof',
        'knn_neighbors':3,'knn_weight':.4,'knn_exclude_self':True,'knn_pool':'each_prediction_population_separately'}
    write_json(root/'parameters.json',cfg)
    result=registry.register(stage='global_mixture',kind='statistics',phase=phase,files=[root/'parameters.json'],
        parents=[rr,*parents,lr,pr],rows=index,fit_users=inputs.partition.users,config=cfg,source_files=PRODUCER_FILES)
    write_json(alias,result);return result


def predict_global(protocol,model,partition,context):
    from .imu_rf_teacher import _write_npz
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(model,'global_mixture')
    phase='select' if partition in {'train12','development2'} else 'refit'
    if record.phase!=phase:raise ValueError('global prediction own-phase model required')
    table,valid,fallback,parents=_global_inputs(protocol,partition,context)
    index=row_index(read_public_rows(protocol.public_manifests[partition]));root=protocol.run_root/REVISION/'global_mixture'/phase
    path=root/(partition+'_targets.npz');alias=path.with_suffix('.artifact.json')
    if alias.exists():
        result=ref(json.loads(alias.read_text(encoding='utf-8')));_,a=_read_prediction(protocol,result,'global_mixture',partition,context)
        return Prediction(index,a['logits'],a['probabilities'],a['valid'],result)
    scores,prob=apply_global(table,record.config['weights'],valid,fallback)
    # The source KNN is a separate script: it sees saved FP32 global scores.
    # Return that same persisted representation on first generation and resume.
    scores=scores.astype(np.float32);prob=softmax(scores.astype(np.float64),axis=1).astype(np.float32)
    prob[~valid]=fallback[~valid]
    digest=_write_npz(path,{'sample_ids':np.asarray(index.sample_ids),'class_ids':np.arange(40),'logits':scores,
        'probabilities':prob,'valid':valid})
    inputs=load_stage_inputs(protocol,partition,'predict',context=context)
    result=registry.register(stage='global_mixture',kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
        parents=[model,*parents,inputs.parents['manifest']],rows=index,config={'partition':partition,'revision':REVISION,
            'operation':'source_global_simplex','fallback':'p12_final_without_visual_details'},source_files=PRODUCER_FILES)
    write_json(alias,result);return Prediction(index,scores,prob,valid,result)


def apply_fullwindow_knn(protocol,prediction,partition,context):
    from .imu_rf_teacher import _write_npz
    registry=ArtifactRegistry(protocol,context=context);base=registry.verify(prediction.artifact,'global_mixture','predict',prediction.index)
    if base.config.get('operation')!='source_global_simplex' or base.config.get('partition')!=partition:raise ValueError('KNN must consume original same-population global mixture')
    path=protocol.run_root/REVISION/'global_knn'/('select' if partition in {'train12','development2'} else 'refit')/(partition+'_targets.npz')
    alias=path.with_suffix('.artifact.json')
    if alias.exists():
        output=ref(json.loads(alias.read_text(encoding='utf-8')));record,a=_read_prediction(protocol,output,'global_mixture',partition,context)
        if record.config.get('operation')!='source_fullwindow_knn' or prediction.artifact not in record.parents:raise ValueError('KNN cached prediction producer mismatch')
        return Prediction(prediction.index,a['logits'],a['probabilities'],a['valid'],output)
    population='final4' if partition=='final4' else 'refit14'
    reference=ref(json.loads((protocol.run_root/'fullwindow_features'/population/'full/artifact.json').read_text(encoding='utf-8')))
    record=registry.verify(reference,'fullwindow_features','raw');file=next(Path(x) for x in record.files if Path(x).suffix=='.npz')
    if (not record.complete or record.rows!=row_index(read_public_rows(protocol.public_manifests[population]))
        or record.config.get('partition')!=population or record.config.get('windows')!='full_0_1'
        or record.config.get('frames')!=16 or tuple(record.config.get('views',()))!=('scene','person','workspace')
        or record.config.get('backbone_frozen') is not True):raise ValueError('KNN requires complete source fullwindow feature contract')
    # Read actual needed frozen features; no whole-cache SHA or unrelated logits.
    key=(protocol.identity(),'consumed_fullwindow',reference.sha256);stamp=context.stamp(file)
    if key not in context.public_rows:
        with np.load(file,allow_pickle=False) as z:ids=z['sample_ids'];features=z['features'];valid=z['valid'];columns=z['class_ids']
        if (tuple(ids.tolist())!=record.rows.sample_ids or features.shape!=(len(ids),3,1024) or features.dtype!=np.float16
            or valid.shape!=(len(ids),) or valid.dtype!=np.bool_ or not np.array_equal(columns,np.arange(40))):raise ValueError('fullwindow ID/axis drift')
        features.flags.writeable=False;valid.flags.writeable=False;context.public_rows[key]=(stamp,features,valid)
    saved,features,valid=context.public_rows[key]
    if saved!=stamp:raise ValueError('consumed fullwindow feature stat drift')
    lookup={s:i for i,s in enumerate(record.rows.sample_ids)}
    positions=np.asarray([lookup[s] for s in prediction.index.sample_ids]);valid=valid[positions]&prediction.valid
    if tuple(record.rows.user_ids[i] for i in positions)!=prediction.index.user_ids:raise ValueError('fullwindow subset user mismatch')
    emission=softmax(prediction.logits.astype(np.float64),axis=1);emission[~prediction.valid]=prediction.probabilities[~prediction.valid]
    result=apply_knn(emission,features[positions],valid,p12_fusion_source_ops(protocol,context))
    probability=result['probability'].astype(np.float32);scores=np.log(np.clip(result['probability'],1e-8,1.)).astype(np.float32)
    digest=_write_npz(path,{'sample_ids':np.asarray(prediction.index.sample_ids),'class_ids':np.arange(40),'logits':scores,
        'probabilities':probability,'valid':valid,'neighbor_indices':result['neighbor_indices'],'neighbor_similarities':result['neighbor_similarities']})
    output=registry.register(stage='global_mixture',kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
        parents=[prediction.artifact,reference],rows=prediction.index,config={'partition':partition,'revision':REVISION,
            'operation':'source_fullwindow_knn','neighbors':3,'neighbor_weight':.4,'exclude_self':True,
            'pool':partition,'cache_verification':'metadata_ID_axes_and_actual_consumed_features_no_full_SHA'},source_files=PRODUCER_FILES)
    write_json(alias,output);return Prediction(prediction.index,scores,probability,valid,output)


def run_p12_global(protocol):
    context=VerificationContext(protocol);models={};outputs={};global_models={};global_outputs={};knn_outputs={}
    for phase in ['select','refit']:
        models[phase]=build_p12_source_chain(protocol,phase,context)
        for pop in (['train12','development2'] if phase=='select' else ['refit14','final4']):
            outputs[pop]=predict_p12(protocol,models[phase],pop,context)
        global_models[phase]=fit_global_mixture(protocol,phase,context)
        for pop in (['train12','development2'] if phase=='select' else ['refit14','final4']):
            prediction=predict_global(protocol,global_models[phase],pop,context);global_outputs[pop]=prediction.artifact
            knn_outputs[pop]=apply_fullwindow_knn(protocol,prediction,pop,context).artifact
            print(json.dumps({'event':'reconstructed_p12_global_knn','partition':pop,'rows':len(prediction.valid),'visual_valid':int(prediction.valid.sum())}),flush=True)
    body={'revision':REVISION,'base_protocol_sha256':protocol.identity(),'p12_models':models,'p12_outputs':outputs,
        'global_models':global_models,'global_outputs':global_outputs,'knn_outputs':knn_outputs,
        'p12_global_knn_component_complete':True,'final_labels_read':False,'task11_complete':False,'a8_generated':False}
    immutable_bytes(protocol.run_root/'protocol'/REVISION/'p12_global_knn_component.json',canonical_bytes(body)+b'\n');return body
