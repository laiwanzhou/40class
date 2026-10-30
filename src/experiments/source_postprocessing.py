"""Reconstructed safe-base -> retained bank -> source group/sequence -> P310/A8."""
import json
from pathlib import Path
from dataclasses import asdict
import joblib
import numpy as np
from scipy.special import softmax

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_types import canonical_bytes,canonical_hash,write_json
from .no_vote_manifest import read_public_rows,row_index,load_stage_inputs
from .no_vote_cache_import import immutable_bytes
from .p12_reconstruction import REVISION,ref,load_reconstruction
from .p12_source_adapter import _read_prediction
from .source_base_targets import _metadata,source_base_target_ops
from .source_group_ops import group_feature_blocks,pruned_group_equivalence
from .session_metadata import load_recording_rows,session_blocks
from .source_post_math import (post_ops,safe_probability,safe_decision,bank_arrays,fit_group,group_probability,group_threshold,
    sequence_gate,sequence_prediction)
from .source_teacher_bank import pinned_source,precedence_targets
from .visual_student_v3 import prior,_label_record
from . import source_post_math as _math,source_group_ops as _group,session_metadata as _session

OWN_FILES=(Path(__file__),Path(_math.__file__),Path(_group.__file__),Path(_session.__file__))

CONTRACT='reconstructed_source_post_v1'
SOURCE_FILES=('aligned_multimodal/p89_imu_probability_blend.py','aligned_multimodal/p89_global_joint_repeat_decoder.py',
 'aligned_multimodal/freeze_p87s_tiny_decoder.py','aligned_multimodal/p165_deployable_group_teacher.py',
 'aligned_multimodal/p134_frozen_repeat_consensus.py','aligned_multimodal/p137_group_classifier_selector.py',
 'aligned_multimodal/p270_fixed_emission065_transition045.py','aligned_multimodal/p139_soft_sequence_gate.py',
 'aligned_multimodal/p309_union_repeat_group_test.py','aligned_multimodal/p310_union_repeat_precedence_teacher.py')

def phase_for(pop):return 'select' if pop in {'train12','development2'} else 'refit'
def fit_population(phase):
    if phase not in {'select','refit'}:raise ValueError('postprocessor fit phase mismatch')
    return 'train12' if phase=='select' else 'refit14'
def alias(path):return ref(json.loads(Path(path).read_text(encoding='utf-8')))
def _root(p):return p.run_root/REVISION/'postprocessing'
def _registry(p,c):return ArtifactRegistry(p,context=c)
def _pins(p,c):return {name:pinned_source(p,name,c)[1]['sha256'] for name in SOURCE_FILES}
def _component(p,name):
    body=json.loads((p.run_root/'protocol'/REVISION/name).read_text(encoding='utf-8'))
    if body['base_protocol_sha256']!=p.identity() or body['revision']!=REVISION:raise ValueError('postprocessing component identity mismatch')
    return body

def column_stage(name):
    if name=='p89_safe_probability':return 'safe_base'
    if name=='p85_teacher':return 'global_mixture'
    if name.startswith('p85_head_') and name.endswith('_logits'):
        family=name[len('p85_head_'):-len('_logits')]
        if family in {'early','late','window_mean','early_late','temporal_delta','kinetics'}:return 'visual_head/'+family
    if name.startswith('p86_mechanism_') and name.endswith('_logits'):return 'visual_mechanism'
    if name in {'p12_skeleton','p12_thermal'}:return 'source_expert/'+name
    raise ValueError('removed or unknown reconstructed teacher column: '+name)

def _transition(p,phase,c):
    reference=ref(_component(p,'source_base_decoder_fit.json')['models'][phase]);record=_registry(p,c).verify(reference,'source_base_decoder',phase)
    path=next(Path(x) for x in record.files if Path(x).suffix=='.npz');ops=source_base_target_ops(p,c)
    model=ops['transition_class'](**c.load_arrays(path));cfg=record.config['decoder'];math=post_ops(p,c)
    decoder=math['DecoderConfig'](cfg['gap_seconds'],cfg['transition_weight'],cfg['trigram_backoff'],cfg['beam_width'])
    return reference,model,decoder

def allows_post_stage(registry,r):
    counts={'safe_base':5,'group_old':3,'group_new':3,'sequence':2,'gate':5}
    if r.stage not in counts or r.phase not in {'select','refit'} or r.kind!={'safe_base':'statistics','sequence':'transition'}.get(r.stage,'supervised_model'):return False
    cfg=r.config
    if cfg.get('revision')!=REVISION or cfg.get('post_contract')!=CONTRACT or len(r.parents)!=counts[r.stage]:return False
    p=registry.protocol;c=registry.context or VerificationContext(p);recipe,rr,_=load_reconstruction(p,c);pop=fit_population(r.phase)
    if (cfg.get('recipe_sha256')!=canonical_hash(recipe) or cfg.get('fit_partition')!=pop or rr not in r.parents
        or r.rows!=row_index(read_public_rows(p.public_manifests[pop])) or cfg.get('source_files')!=_pins(p,c)):return False
    parents=[registry.read(x) for x in r.parents]
    expected={'safe_base':['source_base_adapt12_reconstructed','source_base_decoder','source_expert/imu_rf','class_prior'],
        'group_old':['teacher_bank','labels_'+pop],'group_new':['group_old','teacher_bank'],
        'sequence':['source_base_decoder'],'gate':['group_old','sequence','teacher_bank','labels_'+pop]}[r.stage]
    if not all(sum(x.stage==s for x in parents)==1 for s in expected):return False
    if r.stage=='safe_base' and canonical_bytes(cfg.get('safe_recipe'))!=canonical_bytes(recipe['safe_base']):return False
    if r.stage in {'group_old','group_new','gate'} and tuple(cfg.get('columns',()))!=tuple(recipe['roster_view']['columns']):return False
    if r.stage=='group_old' and cfg.get('classifier_recipe')!={'C':.03,'peer_weight':2.,'balanced':False,'frequency_power':0.}:return False
    if r.stage=='group_new' and (cfg.get('equivalent_alias') is not True or not pruned_group_equivalence(p,c,recipe)['same_group_inputs_and_recipe']):return False
    if r.stage=='sequence' and cfg.get('sequence_recipe')!={'emission':.65,'transition':.45,'gap':30.,'trigram':1.,'beam':50}:return False
    return True

def _config(p,phase,c):
    recipe,rr,_=load_reconstruction(p,c)
    return recipe,rr,{'revision':REVISION,'post_contract':CONTRACT,'fit_partition':fit_population(phase),
        'recipe_sha256':canonical_hash(recipe),'source_files':_pins(p,c),'historical_exact_reproduction':False}

def fit_safe(p,phase,c):
    recipe,rr,cfg=_config(p,phase,c);pop=fit_population(phase);r=_registry(p,c);root=_root(p)/'safe_base'/phase
    if (root/'artifact.json').exists():result=alias(root/'artifact.json');r.verify(result,'safe_base',phase);return result
    base=ref(_component(p,'source_base_adapt12_component.json')['models'][phase]);r.verify(base,'source_base_adapt12_reconstructed',phase)
    decoder,_,_= _transition(p,phase,c)
    rf=ref(json.loads((p.run_root/'protocol/task7_artifacts_ready.json').read_text(encoding='utf-8'))['models'][phase]);r.verify(rf,'source_expert/imu_rf',phase)
    pr,_=prior(p,phase,c);ops=post_ops(p,c)
    if asdict(ops['safe_repeat'])!=dict(recipe['safe_base']['repeat_config']):raise ValueError('safe repeat source differs from frozen recipe')
    cfg.update(safe_recipe=dict(recipe['safe_base']),transition_origin='own_phase_frozen_tiny_decoder_same_as_source_intermediate_decoder',
        initial_prediction='decode_unadjusted_source_base_before_RF_joint_decode')
    write_json(root/'parameters.json',cfg)
    result=r.register(stage='safe_base',kind='statistics',phase=phase,files=[root/'parameters.json'],parents=[rr,base,decoder,rf,pr],
        rows=row_index(read_public_rows(p.public_manifests[pop])),fit_users=p.partitions[pop].users,config=cfg,source_files=OWN_FILES)
    write_json(root/'artifact.json',result);return result

def predict_safe(p,model,pop,c):
    from .imu_rf_teacher import _write_npz
    r=_registry(p,c);record=r.verify(model,'safe_base',phase_for(pop));path=_root(p)/'safe_base'/record.phase/(pop+'_targets.npz')
    if path.with_suffix('.artifact.json').exists():return alias(path.with_suffix('.artifact.json'))
    source=ref(_component(p,'source_base_adapt12_component.json')['outputs'][pop]);_,a=_read_prediction(p,source,'source_base_adapt12_reconstructed',pop,c)
    rf=ref(json.loads((p.run_root/'protocol/task7_artifacts_ready.json').read_text(encoding='utf-8'))['outputs'][pop]['terminal_export']);_,b=_read_prediction(p,rf,'source_expert/imu_rf',pop,c)
    metadata=_metadata(p,pop);index,rows=load_recording_rows(p,pop,metadata,c);_,transition,decoder=_transition(p,record.phase,c);_,pr=prior(p,record.phase,c)
    valid=a['valid']|b['valid'];raw=softmax(a['logits'].astype(np.float64),axis=1);raw[~a['valid']]=a['probabilities'][~a['valid']]
    probability=safe_probability(raw,b['logits'],valid,pr);decision=safe_decision(index,rows,raw,probability,transition,decoder,post_ops(p,c))
    decision[~valid]=pr.argmax();probability=probability.astype(np.float32);probability[~valid]=pr
    digest=_write_npz(path,{'sample_ids':np.asarray(index.sample_ids),'class_ids':np.arange(40),'logits':np.log(np.maximum(probability,1e-12)),
        'probabilities':probability,'valid':valid,'decision':decision})
    result=r.register(stage='safe_base',kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
        parents=[model,source,rf,metadata],rows=index,config={'partition':pop,'revision':REVISION,'post_contract':CONTRACT,
            'operation':'source_safe_probability_and_joint_decision'},source_files=OWN_FILES)
    write_json(path.with_suffix('.artifact.json'),result);return result

def _expert(p,name,pop,c):
    phase=phase_for(pop)
    if name=='p85_teacher':return ref(_component(p,'p12_global_knn_component.json')['knn_outputs'][pop])
    if name.startswith('p85_head_'):
        family=name[len('p85_head_'):-len('_logits')];return alias(p.run_root/'visual_heads'/family/phase/(pop+'_targets.artifact.json'))
    if name.startswith('p86_mechanism_'):
        perturbation=name[len('p86_mechanism_'):-len('_logits')];return alias(p.run_root/'visual_mechanisms'/phase/perturbation/(pop+'_targets.artifact.json'))
    role=name.removeprefix('p12_');return ref(_component(p,role+'_fp16_component.json')['outputs'][pop])

def validate_reconstructed_bank(r,record):
    p=r.protocol;c=r.context or VerificationContext(p);recipe,rr,_=load_reconstruction(p,c);names=list(recipe['roster_view']['columns'])
    parents=[r.read(x) for x in record.parents];pred=[x for x in parents if x.kind=='predictions'];pop=record.config.get('partition')
    if (record.stage!='teacher_bank' or record.kind!='teacher_bank' or record.config.get('post_contract')!=CONTRACT
        or list(record.config.get('experts',()))!=names or len(record.parents)!=len(names)+1 or rr not in record.parents
        or len(pred)!=len(names) or len(set(x.sha256 for x in record.parents))!=len(record.parents)
        or record.rows!=row_index(read_public_rows(p.public_manifests[pop]))):raise ValueError('reconstructed bank roster/parents/ID mismatch')
    for name,parent in zip(names,pred):
        if parent.stage!=column_stage(name) or parent.rows!=record.rows or parent.config.get('partition')!=pop:raise ValueError('reconstructed bank exact producer order mismatch')
        if name=='p85_teacher' and parent.config.get('operation')!='source_fullwindow_knn':raise ValueError('bank requires actual P85 KNN')
        if name.startswith('p86_mechanism_') and parent.config.get('perturbation')!=name[len('p86_mechanism_'):-len('_logits')]:raise ValueError('bank mechanism identity mismatch')
    if len(record.files)!=1:raise ValueError('one reconstructed bank payload required')
    path=Path(next(iter(record.files)));a=c.load_arrays(path)
    if (set(a)!= {'sample_ids','class_ids','probabilities','availability','fallback'} or tuple(a['sample_ids'].tolist())!=record.rows.sample_ids
        or not np.array_equal(a['class_ids'],np.arange(40)) or a['probabilities'].shape!=(len(record.rows.sample_ids),len(names),40)
        or a['availability'].shape!=a['probabilities'].shape[:2] or a['availability'].dtype!=np.bool_
        or not np.array_equal(a['fallback'],~a['availability']) or not np.isfinite(a['probabilities']).all()
        or np.any(a['probabilities']<0) or not np.allclose(a['probabilities'].sum(2),1.,atol=1e-5)):raise ValueError('reconstructed bank payload schema')
    for j in range(1,len(names)):
        mask=~a['availability'][:,j]
        if not np.array_equal(a['probabilities'][mask,j],a['probabilities'][mask,0]):raise ValueError('bank unavailable fallback must equal safe first column')

def build_source_bank(p,pop,safe,c):
    from .imu_rf_teacher import _write_npz
    recipe,rr,_=load_reconstruction(p,c);names=list(recipe['roster_view']['columns']);r=_registry(p,c);root=_root(p)/'bank'/pop
    if (root/'artifact.json').exists():result=alias(root/'artifact.json');r.verify(result,'teacher_bank','predict');return result
    sr,a=_read_prediction(p,safe,'safe_base',pop,c);columns=[];masks=[];refs=[safe]
    for name in names[1:]:
        reference=_expert(p,name,pop,c);_,values=_read_prediction(p,reference,column_stage(name),pop,c)
        columns.append(values['probabilities']);masks.append(values['valid']);refs.append(reference)
    bank,availability=bank_arrays(a['probabilities'],columns,masks,a['valid']);path=root/'bank.npz'
    digest=_write_npz(path,dict(sample_ids=np.asarray(sr.rows.sample_ids),class_ids=np.arange(40),probabilities=bank,availability=availability,fallback=~availability))
    result=r.register(stage='teacher_bank',kind='teacher_bank',phase='predict',files=[path],file_digests={str(path.resolve()):digest},parents=[rr,*refs],rows=sr.rows,
        config={'partition':pop,'revision':REVISION,'post_contract':CONTRACT,'experts':names,'recipe_sha256':canonical_hash(recipe),
            'columns_are_actual_fresh_producers':True,'historical_exact_reproduction':False},source_files=OWN_FILES)
    write_json(root/'artifact.json',result);return result

def _bank_data(p,reference,c):
    r=_registry(p,c);record=r.verify(reference,'teacher_bank','predict');a=c.load_arrays(Path(next(iter(record.files))))
    pop=record.config['partition'];safe=next(x for x in record.parents if r.read(x).stage=='safe_base');_,s=_read_prediction(p,safe,'safe_base',pop,c)
    nn=next(x for x in record.parents if r.read(x).stage=='global_mixture');_,v=_read_prediction(p,nn,'global_mixture',pop,c)
    detail_ids=np.asarray(record.rows.sample_ids)[v['valid']];visual=np.isin(np.asarray(record.rows.sample_ids),detail_ids)
    metadata=_metadata(p,pop);index,rows=load_recording_rows(p,pop,metadata,c)
    return record,a,s,visual,metadata,rows

def fit_postprocessor(p,phase,bank,c):
    from .visual_teacher import _labels
    recipe,rr,cfg=_config(p,phase,c);pop=fit_population(phase);r=_registry(p,c);br,a,s,visual,meta,rows=_bank_data(p,bank,c)
    if br.config['partition']!=pop:raise ValueError('postprocessing fit must use own training bank')
    root=_root(p);existing=root/'gate'/phase/'artifact.json'
    if existing.exists():return {key:alias(root/key/phase/'artifact.json') for key in ['group_old','group_new','sequence','gate']}
    inputs=load_stage_inputs(p,pop,phase,context=c);index,y=_labels(inputs,phase,p);labels=_label_record(inputs,phase,index,p,c)
    ops=post_ops(p,c);x=group_feature_blocks(index,rows,s['decision'],a['probabilities'],ops)
    model,prob=fit_group(x,y,x,ops);threshold=group_threshold(s['decision'],prob.argmax(1),prob,y,ops)
    gc={**cfg,'columns':recipe['roster_view']['columns'],'classifier_recipe':{'C':.03,'peer_weight':2.,'balanced':False,'frequency_power':0.},
        'group_config':asdict(ops['group_config']),'group_threshold':threshold,'fit_rows':len(y),'features':x.shape[1],
        'threshold_origin':'own_phase_fit_in_sample_non_oof','unknown_time_output':'safe_base','bank':bank}
    old_root=root/'group_old'/phase;old_root.mkdir(parents=True,exist_ok=True);joblib.dump(model,old_root/'model.joblib',compress=3);write_json(old_root/'parameters.json',gc)
    old=r.register(stage='group_old',kind='supervised_model',phase=phase,files=[old_root/'model.joblib',old_root/'parameters.json'],
        parents=[rr,bank,labels],rows=index,fit_users=inputs.partition.users,config=gc,source_files=OWN_FILES)
    write_json(old_root/'artifact.json',old)
    proof=pruned_group_equivalence(p,c,recipe)
    if not proof['same_group_inputs_and_recipe']:raise ValueError('new group needs a separate producer after roster change')
    nc={**cfg,'columns':recipe['roster_view']['columns'],'equivalent_alias':True,'old_group_model':old,'same_bank':bank,
        'equivalence_evidence':proof,'no_extra_teacher_column':True}
    nr=root/'group_new'/phase;write_json(nr/'parameters.json',nc)
    new=r.register(stage='group_new',kind='supervised_model',phase=phase,files=[old_root/'model.joblib',nr/'parameters.json'],
        parents=[rr,old,bank],rows=index,fit_users=inputs.partition.users,config=nc,source_files=OWN_FILES)
    write_json(nr/'artifact.json',new)
    decoder,transition,_=_transition(p,phase,c);dr=r.read(decoder);file=next(Path(x) for x in dr.files if Path(x).suffix=='.npz')
    sc={**cfg,'sequence_recipe':{'emission':.65,'transition':.45,'gap':30.,'trigram':1.,'beam':50},
        'transition_counts_reused':'same_fit_labels_same_gap30_trigram1_alpha025_as_frozen_tiny_decoder'}
    seqroot=root/'sequence'/phase;write_json(seqroot/'parameters.json',sc)
    seq=r.register(stage='sequence',kind='transition',phase=phase,files=[file,seqroot/'parameters.json'],parents=[rr,decoder],rows=index,
        fit_users=inputs.partition.users,config=sc,source_files=OWN_FILES)
    write_json(seqroot/'artifact.json',seq)
    known=np.array([x['timestamp_available']=='1' for x in rows]);proposal=prob.argmax(1);score=prob[np.arange(len(y)),proposal]-prob[np.arange(len(y)),s['decision']]
    route=(proposal!=s['decision'])&(score>=threshold['threshold'])&visual&known;base=s['decision'].copy();base[route]=proposal[route]
    # Source P270 persists group probabilities as FP32 before its sequence step.
    prob=prob.astype(np.float32).astype(np.float64)
    predicted=sequence_prediction(index,rows,prob,base,transition,ops);features=ops['gate_features'](prob,base,predicted)
    gate=sequence_gate(base,predicted,features,y,ops)
    gcfg={**cfg,'columns':recipe['roster_view']['columns'],'sequence_gate':gate,'fit_rows':len(y),'threshold_origin':'own_phase_fit_in_sample_non_oof',
        'sequence_visual_guard':False,'unknown_time_policy':'no_edges_keep_base'}
    gr=root/'gate'/phase;write_json(gr/'parameters.json',gcfg)
    gated=r.register(stage='gate',kind='supervised_model',phase=phase,files=[gr/'parameters.json'],parents=[rr,old,seq,bank,labels],rows=index,
        fit_users=inputs.partition.users,config=gcfg,source_files=OWN_FILES)
    write_json(gr/'artifact.json',gated);print(json.dumps({'event':'source_post_fit','phase':phase,'rows':len(y),'feature_dim':x.shape[1],
        'group_threshold':threshold['threshold'],'sequence_gate':gate,'new_group_equivalent':True}),flush=True)
    return {'group_old':old,'group_new':new,'sequence':seq,'gate':gated}

def validate_reconstructed_a8(r,record):
    p=r.protocol;c=r.context or VerificationContext(p);recipe,rr,_=load_reconstruction(p,c);pop=record.config['partition'];phase=phase_for(pop)
    if record.config.get('post_contract')!=CONTRACT or record.config.get('recipe_sha256')!=canonical_hash(recipe):raise ValueError('A8 reconstruction contract mismatch')
    parents=[r.read(x) for x in record.parents]
    for stage,kind in [('teacher_bank','teacher_bank'),('safe_base','predictions'),('group_old','supervised_model'),('group_new','supervised_model'),('sequence','transition'),('gate','supervised_model'),('class_prior','statistics')]:
        matches=[x for x in parents if x.stage==stage and x.kind==kind]
        if len(matches)!=1 or matches[0].phase!=('predict' if kind in {'teacher_bank','predictions'} else phase):raise ValueError('A8 requires full own-phase bank/group/gate/sequence chain')
    if rr not in record.parents:raise ValueError('A8 needs frozen reconstruction recipe')

def apply_postprocessor(p,pop,bank,models,c):
    from .imu_rf_teacher import _write_npz
    recipe,rr,_=load_reconstruction(p,c);r=_registry(p,c);phase=phase_for(pop);br,a,s,visual,meta,rows=_bank_data(p,bank,c);index=br.rows
    old=r.verify(models['group_old'],'group_old',phase);new=r.verify(models['group_new'],'group_new',phase)
    seq=r.verify(models['sequence'],'sequence',phase);gate=r.verify(models['gate'],'gate',phase)
    if new.config.get('old_group_model')!=json.loads(canonical_bytes(models['group_old'])):raise ValueError('new group alias must use exact old group model')
    ops=post_ops(p,c);x=group_feature_blocks(index,rows,s['decision'],a['probabilities'],ops)
    model=joblib.load(next(Path(x) for x in old.files if Path(x).suffix=='.joblib'));prob=group_probability(model,x)
    known=np.array([v['timestamp_available']=='1' for v in rows]);proposal=prob.argmax(1);score=prob[np.arange(len(prob)),proposal]-prob[np.arange(len(prob)),s['decision']]
    route=(proposal!=s['decision'])&(score>=old.config['group_threshold']['threshold'])&visual&known
    old_prediction=s['decision'].copy();old_prediction[route]=proposal[route]
    new_prediction=old_prediction.copy() # Proven identical bank/config, one fitted classifier, not two fake experts.
    _,transition,_=_transition(p,phase,c);stored_prob=prob.astype(np.float32);prob=stored_prob.astype(np.float64)
    raw_sequence=sequence_prediction(index,rows,prob,old_prediction,transition,ops)
    features=ops['gate_features'](prob,old_prediction,raw_sequence);g=gate.config['sequence_gate']
    seqroute=(raw_sequence!=old_prediction)&(features[:,g['score_index']]>=g['threshold'])
    sequence=old_prediction.copy();sequence[seqroute]=raw_sequence[seqroute]
    pr,prior_prob=prior(p,phase,c);targets=precedence_targets(old_prediction,sequence,new_prediction,s['valid'],prior_prob)
    if targets['precedence_route'].any():raise ValueError('equivalent old/new group unexpectedly creates precedence changes')
    payload={**targets,'sample_ids':np.asarray(index.sample_ids),'class_ids':np.arange(40),
        'logits':np.log(np.maximum(targets['structured_distillation_probability'],1e-12)),
        'probabilities':targets['structured_distillation_probability'],'valid':targets['target_mask'],
        'safe_decision':s['decision'],'old_group_prediction':old_prediction,'new_group_prediction':new_prediction,
        'group_probability':stored_prob,'group_route':route,'sequence_prediction':sequence,'sequence_route':seqroute,
        'raw_sequence_prediction':raw_sequence,'visual_available':visual,'known_time':known}
    path=_root(p)/'A8'/phase/(pop+'_targets.npz');digest=_write_npz(path,payload)
    safe=next(x for x in br.parents if r.read(x).stage=='safe_base')
    result=r.register(stage='A8',kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
        parents=[rr,bank,safe,*models.values(),pr,meta],rows=index,config={'partition':pop,'revision':REVISION,'post_contract':CONTRACT,
            'recipe_sha256':canonical_hash(recipe),'fit_phase':phase,'target_kind':'source_P310_precedence_sharpened',
            'precedence_changes':int(targets['precedence_route'].sum()),'group_changes':int(route.sum()),'sequence_changes':int(seqroute.sum()),
            'unknown_time_policy':'no_edges_keep_safe_base','final_labels_read':False},source_files=OWN_FILES)
    write_json(path.with_suffix('.artifact.json'),result);print(json.dumps({'event':'A8_source_targets','partition':pop,'rows':len(index.sample_ids),
        'valid':int(s['valid'].sum()),'group_changes':int(route.sum()),'sequence_changes':int(seqroute.sum()),'precedence_changes':0}),flush=True)
    return result

def run_postprocessing(p):
    c=VerificationContext(p);models={};safe_models={};safes={};banks={};outputs={}
    for phase,populations in [('select',['train12','development2']),('refit',['refit14','final4'])]:
        safe_models[phase]=fit_safe(p,phase,c)
        for pop in populations:
            safes[pop]=predict_safe(p,safe_models[phase],pop,c);banks[pop]=build_source_bank(p,pop,safes[pop],c)
        models[phase]=fit_postprocessor(p,phase,banks[fit_population(phase)],c)
        for pop in populations:outputs[pop]=apply_postprocessor(p,pop,banks[pop],models[phase],c)
    body={'revision':REVISION,'base_protocol_sha256':p.identity(),'safe_models':safe_models,'safe_outputs':safes,'banks':banks,
        'post_models':models,'outputs':outputs,'source_postprocessing_generated':True,'task11_complete':False,
        'independent_review':'pending','final_labels_read':False,'task12_started':False}
    immutable_bytes(p.run_root/'protocol'/REVISION/'source_postprocessing_component.json',canonical_bytes(body)+b'\n');return body
