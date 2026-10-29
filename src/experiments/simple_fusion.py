"""A6 diagnostic controls only; never source-main supervision or expert-bank input."""
from pathlib import Path
import csv
import json
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp,softmax
from sklearn.metrics import f1_score

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .no_vote_types import ArtifactRef,Prediction,Selection,canonical_bytes,canonical_hash,write_json
from .no_vote_cache_import import immutable_bytes

ROLE='diagnostic_not_source_main'
BRANCHES=('visual','skeleton','imu')
CANDIDATES={'A6-VS':('visual','skeleton'),'A6-VI':('visual','imu'),'A6-VSI':BRANCHES}


def ref(value):
    return ArtifactRef(Path(value['record_path']),value['sha256'])


def aligned(prediction,index):
    source=prediction.index
    if source.class_ids!=index.class_ids or set(source.sample_ids)!=set(index.sample_ids):
        raise ValueError('fusion ID/class sets mismatch')
    lookup={s:i for i,s in enumerate(source.sample_ids)};order=np.array([lookup[s] for s in index.sample_ids])
    if tuple(source.user_ids[i] for i in order)!=index.user_ids:raise ValueError('fusion ID/user mismatch')
    return prediction.logits[order],prediction.valid[order]


def combine_available(branches,prior,*,index,temperatures=None):
    if not branches:raise ValueError('at least one fusion branch required')
    prior=np.asarray(prior)
    if prior.shape!=(40,) or not np.isfinite(prior).all() or np.any(prior<0) or not np.isclose(prior.sum(),1):
        raise ValueError('invalid same-phase prior')
    temperatures=[1.]*len(branches) if temperatures is None else temperatures
    if len(temperatures)!=len(branches) or any(not np.isfinite(t) or not .25<=t<=4 for t in temperatures):
        raise ValueError('diagnostic temperatures must be within .25..4')
    total=np.zeros((len(index.sample_ids),40),np.float64);count=np.zeros(len(total),np.int64)
    for prediction,t in zip(branches,temperatures):
        scores,valid=aligned(prediction,index)
        total+=softmax(scores/float(t),axis=1)*valid[:,None];count+=valid
    valid=count>0;total[valid]/=count[valid,None];total[~valid]=prior
    return total,valid


def read_labels(path,index):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if tuple(reader.fieldnames or ())!=('sample_id','class_id'):raise ValueError('diagnostic label schema mismatch')
        rows=list(reader)
    values={x['sample_id']:int(x['class_id']) for x in rows}
    if len(values)!=len(rows) or set(values)!=set(index.sample_ids):raise ValueError('diagnostic label IDs mismatch')
    y=np.array([values[s] for s in index.sample_ids],np.int64)
    if np.any((y<0)|(y>=40)):raise ValueError('invalid diagnostic target')
    return y


def fit_temperature(prediction,y):
    y=np.asarray(y)
    if y.shape!=(len(prediction.index.sample_ids),) or np.any((y<0)|(y>=40)):raise ValueError('temperature target shape/range mismatch')
    valid=prediction.valid
    if not valid.any():return 1.
    x=prediction.logits[valid].astype(np.float64);labels=y[valid]
    def objective(log_t):
        z=x/np.exp(log_t)
        return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(z)),labels]))
    lo,hi=np.log(.25),np.log(4.)
    result=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-8})
    if not result.success:raise ValueError('diagnostic temperature optimization failed')
    # Include boundary and identity explicitly: never worsen the fit NLL by solver tolerance.
    best=min([lo,hi,0.,float(result.x)],key=objective)
    return float(np.exp(best))


def metrics(probabilities,y,index):
    guessed=probabilities.argmax(1);correct=guessed==y;users=np.asarray(index.user_ids)
    return {'accuracy':float(correct.mean()),'macro_f1':float(f1_score(y,guessed,labels=np.arange(40),average='macro',zero_division=0)),
        'worst_user_accuracy':min(float(correct[users==u].mean()) for u in set(users))}


def select_calibration(fit_predictions,fit_labels,dev_predictions,dev_labels,*,prior):
    if set(fit_predictions)!=set(BRANCHES) or set(dev_predictions)!=set(BRANCHES):raise ValueError('requires V/S/I diagnostics')
    fit_index=fit_predictions['visual'].index;dev_index=dev_predictions['visual'].index
    if set(fit_index.user_ids)&set(dev_index.user_ids) or set(fit_index.sample_ids)&set(dev_index.sample_ids):
        raise ValueError('fit and development users overlap')
    y=read_labels(fit_labels,fit_index);dy=read_labels(dev_labels,dev_index)
    fitted={}
    for branch in BRANCHES:
        scores,valid=aligned(fit_predictions[branch],fit_index)
        p=Prediction(fit_index,scores,softmax(scores,axis=1),valid,fit_predictions[branch].artifact)
        fitted[branch]=fit_temperature(p,y)
    candidates=[]
    for policy in ['identity','fitted']:
        temperatures={k:1. if policy=='identity' else fitted[k] for k in BRANCHES}
        prob,_=combine_available([dev_predictions[k] for k in BRANCHES],prior,index=dev_index,temperatures=[temperatures[k] for k in BRANCHES])
        candidates.append({'policy':policy,'temperatures':temperatures,**metrics(prob,dy,dev_index)})
    best=min(candidates,key=lambda x:(-x['accuracy'],-x['macro_f1'],-x['worst_user_accuracy'],x['policy']!='identity'))
    config={'role':ROLE,'policy':best['policy'],'temperatures':best['temperatures'],'fitted_temperatures':fitted,
        'candidates':candidates,'selection_candidate':'A6-VSI','fit_ids_sha256':canonical_hash(fit_index),
        'temperature_bounds':[.25,4.],'target_origin':'fixed_split_fit_predictions_not_oof'}
    return Selection('A6-VSI',config,{}, {k:best[k] for k in ['accuracy','macro_f1','worst_user_accuracy']},
        fit_predictions['visual'].artifact,canonical_hash(dev_index))


def current_inputs(protocol,context):
    """Bind completed A2 and reviewed corrected A5, not legacy root-directory guesses."""
    root=protocol.run_root;proofs={}
    def read(path,expected=None):
        path=Path(path).resolve()
        if not path.is_relative_to(root/'protocol'):raise ValueError('diagnostic prerequisite outside protocol')
        digest=context.hash_file(path,expected);proofs[str(path)]=digest
        return json.loads(path.read_text(encoding='utf-8'))
    task5=read(root/'protocol/task5_complete.json')
    pointer=read(root/'protocol/task8_current.json')
    complete=read(pointer['complete']['path'],pointer['complete']['sha256'])
    review=read(pointer['independent_review']['path'],pointer['independent_review']['sha256'])
    if (not task5.get('complete') or task5.get('independent_review',{}).get('status')!='GO'
            or not complete.get('complete') or pointer.get('verdict')!='GO'
            or pointer.get('task8_revision')!='ir_supervision_v2' or review.get('verdict')!='GO'
            or review.get('unresolved_p1_p2') or review.get('task8_revision')!=pointer['task8_revision']
            or any(x.get('protocol_sha256')!=protocol.identity() for x in [task5,pointer,complete])):
        raise ValueError('diagnostics require completed reviewed same-protocol A2/current A5')
    student=read(root/'protocol/task5_artifacts_ready.json',task5['artifacts_ready_sha256'])
    motion=read(review['ready']['path'],review['ready']['sha256'])
    if (canonical_bytes(complete.get('outputs'))!=canonical_bytes(pointer['outputs'])
            or canonical_bytes(motion.get('outputs'))!=canonical_bytes(pointer['outputs'])
            or canonical_bytes(motion.get('models'))!=canonical_bytes(pointer['models'])
            or motion.get('protocol_sha256')!=protocol.identity()
            or canonical_bytes(student['partitions'])!=canonical_bytes(task5['partitions'])):
        raise ValueError('current diagnostic inputs differ from completed/reviewed references')
    outputs={pop:{'visual':ref(student['partitions'][pop]['prediction']),
        **{k:ref(pointer['outputs'][pop][k]) for k in ['skeleton','imu']}} for pop in protocol.partitions}
    return outputs,proofs


def load_prediction(protocol,reference,stage,partition,context):
    registry=ArtifactRegistry(protocol,context=context);index=row_index(read_public_rows(protocol.public_manifests[partition]))
    record=registry.verify(reference,stage,'predict',index)
    if record.config.get('partition')!=partition:raise ValueError('prediction partition mismatch')
    paths=[Path(x) for x in record.files if Path(x).suffix=='.npz']
    if len(paths)!=1:raise ValueError('one diagnostic input NPZ required')
    a=context.load_arrays(registry.verify_file(record,paths[0]))
    if tuple(a['sample_ids'].astype(str))!=index.sample_ids or not np.array_equal(a['class_ids'],np.arange(40)):
        raise ValueError('prediction payload ID/class mismatch')
    return Prediction(index,a['logits'],a['probabilities'],a['valid'],reference)


def allows_diagnostic_stage(registry,record):
    """Only A6 local fitting is ready. Never enables P12, A7, or a teacher bank."""
    if record.stage not in CANDIDATES or record.kind!='supervised_model' or record.phase not in {'select','refit'}:return False
    cfg=record.config;pop='train12' if record.phase=='select' else 'refit14'
    if (cfg.get('role')!=ROLE or cfg.get('fit_partition')!=pop or cfg.get('task8_revision')!='ir_supervision_v2'
            or cfg.get('policy') not in {'identity','fitted'} or set(cfg.get('temperatures',{}))!=set(BRANCHES)):
        return False
    if any(not .25<=t<=4 for t in cfg['temperatures'].values()):return False
    if cfg['policy']=='identity' and any(t!=1 for t in cfg['temperatures'].values()):return False
    c=registry.context or VerificationContext(registry.protocol)
    inputs,proofs=current_inputs(registry.protocol,c)
    if canonical_bytes(cfg.get('prerequisites'))!=canonical_bytes(proofs):return False
    parents=[registry.read(x) for x in record.parents]
    expected=[inputs[pop][k] for k in BRANCHES]
    from .visual_student_v3 import prior
    prior_ref,_=prior(registry.protocol,record.phase,c)
    labels=[p for p in parents if p.kind=='supervised_labels' and p.stage=='labels_'+pop and p.phase==record.phase]
    if len(labels)!=1 or len(parents)!=5 or not all(x in record.parents for x in [*expected,prior_ref]):return False
    if len(record.files)!=1:return False
    path=Path(next(iter(record.files)))
    return path.suffix=='.json' and canonical_bytes(json.loads(path.read_text(encoding='utf-8')))==canonical_bytes(cfg)


def run_pipeline(protocol):
    from .imu_rf_teacher import _write_npz
    from .visual_student_v3 import prior,_label_record
    started=time.perf_counter();context=VerificationContext(protocol);registry=ArtifactRegistry(protocol,context=context)
    refs,proofs=current_inputs(protocol,context);loaded={}
    for pop in protocol.partitions:
        loaded[pop]={k:load_prediction(protocol,refs[pop][k],{'visual':'A2','skeleton':'A5-S','imu':'A5-I'}[k],pop,context) for k in BRANCHES}
    fit=load_stage_inputs(protocol,'train12','select',context=context);dev=load_stage_inputs(protocol,'development2','select',context=context)
    _,select_prior=prior(protocol,'select',context)
    selection=select_calibration(loaded['train12'],fit.labels,loaded['development2'],dev.labels,prior=select_prior)
    selection_path=protocol.run_root/'A6/selection.json';immutable_bytes(selection_path,canonical_bytes(selection)+b'\n')
    models={};phase_config={}
    for phase,pop in [('select','train12'),('refit','refit14')]:
        inputs=load_stage_inputs(protocol,pop,phase,context=context);index=loaded[pop]['visual'].index
        y=read_labels(inputs.labels,index);pr,prob=prior(protocol,phase,context)
        temperatures=dict(selection.config['temperatures']) if phase=='select' else {k:1. if selection.config['policy']=='identity' else fit_temperature(loaded[pop][k],y) for k in BRANCHES}
        cfg={'role':ROLE,'fit_partition':pop,'policy':selection.config['policy'],'temperatures':temperatures,
            'temperature_bounds':[.25,4.],'target_origin':'fixed_split_fit_predictions_not_oof','task8_revision':'ir_supervision_v2',
            'fit_ids_sha256':canonical_hash(index),'prerequisites':proofs,
            'selection_sha256':context.hash_file(selection_path),'selection_candidate':'A6-VSI',
            'invalid_policy':'same_phase_complete_fit_prior','prior':prob.tolist()}
        path=protocol.run_root/'A6'/phase/'model.json';immutable_bytes(path,canonical_bytes(cfg)+b'\n');phase_config[phase]=cfg;models[phase]={}
        label_ref=_label_record(inputs,phase,index,protocol,context)
        for candidate in CANDIDATES:
            models[phase][candidate]=registry.register(stage=candidate,kind='supervised_model',phase=phase,files=[path],
                parents=[*[refs[pop][k] for k in BRANCHES],pr,label_ref],fit_users=inputs.partition.users,
                select_users=protocol.partitions['development2'].users if phase=='select' else (),config=cfg,source_files=[Path(__file__)])
    outputs={};results={}
    for pop in protocol.partitions:
        phase='select' if pop in {'train12','development2'} else 'refit';cfg=phase_config[phase];index=loaded[pop]['visual'].index
        inputs=load_stage_inputs(protocol,pop,'predict',context=context);outputs[pop]={};results[pop]={}
        y=read_labels(protocol.supervised_labels[pop],index) if pop!='final4' else None
        for candidate,keys in CANDIDATES.items():
            probabilities,valid=combine_available([loaded[pop][k] for k in keys],cfg['prior'],index=index,temperatures=[cfg['temperatures'][k] for k in keys])
            logits=np.log(np.maximum(probabilities,1e-12));path=protocol.run_root/'A6'/phase/(pop+'_'+candidate+'_targets.npz')
            digest=_write_npz(path,dict(sample_ids=np.asarray(index.sample_ids),class_ids=np.arange(40),logits=logits,probabilities=probabilities,valid=valid))
            reference=registry.register(stage=candidate,kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
                parents=[models[phase][candidate],*[refs[pop][k] for k in keys],inputs.parents['manifest']],rows=index,
                config={'partition':pop,'fit_phase':phase,'role':ROLE,'policy':cfg['policy'],'branches':keys,
                    'invalid_policy':'same_phase_complete_fit_prior','task8_revision':'ir_supervision_v2'},source_files=[Path(__file__)])
            write_json(path.with_suffix('.artifact.json'),reference);outputs[pop][candidate]=reference
            result={'rows':len(index.sample_ids),'valid':int(valid.sum()),'prior_rows':int((~valid).sum()),'accuracy':None}
            if y is not None:result.update(metrics(probabilities,y,index),correct=int((probabilities.argmax(1)==y).sum()))
            results[pop][candidate]=result
    write_json(protocol.run_root/'A6/development_metrics.json',results)
    body={'protocol_sha256':protocol.identity(),'complete':True,'role':ROLE,'models':models,'outputs':outputs,
        'selection':str(selection_path),'selection_sha256':context.hash_file(selection_path),'input_refs':refs,'prerequisites':proofs,
        'task8_revision':'ir_supervision_v2','final_labels_read':False,'task10_started':False,'seconds':round(time.perf_counter()-started,2)}
    immutable_bytes(protocol.run_root/'protocol/task9_artifacts_ready.json',canonical_bytes(body)+b'\n')
    print(json.dumps({'event':'diagnostic_mean_pipeline_complete','policy':selection.config['policy'],'metrics':results,'seconds':body['seconds']}),flush=True)
    return body
