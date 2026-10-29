"""Source P87 posterior targets for the explicitly fixed intermediate fit pools."""
import json
from pathlib import Path
import numpy as np

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_types import canonical_hash,canonical_bytes,write_json
from .no_vote_cache_import import immutable_bytes
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .p12_reconstruction import REVISION,ref,load_reconstruction
from .session_metadata import source_session_ops,session_blocks,load_recording_rows
from .teammate_source import verify_teammate_source,load_teammate_symbol
from .source_teacher_bank import pinned_source
from . import session_metadata

STAGE='source_base_decoder'
SOURCE_FILES=('aligned_multimodal/audit_p87_sequence_decoder.py','aligned_multimodal/build_p87s_structured_targets.py')


def source_base_target_ops(protocol,context):
    key=(protocol.identity(),'source_base_target_math')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        names={'fit_transition_model':('audit_p87_sequence_decoder','fit_transition_model'),
            'decode_unique_beam_posterior':('audit_p87_sequence_decoder','decode_unique_beam_posterior'),
            'transition_class':('audit_p87_sequence_decoder','TransitionModel'),
            'backed_off_structured_probability':('build_p87s_structured_targets','backed_off_structured_probability')}
        context.public_rows[key]={**source_session_ops(protocol,context),**{k:load_teammate_symbol(report,*x) for k,x in names.items()}}
    return context.public_rows[key]


def structured_source_base(index,rows,emission,emission_log,valid,state,cfg,ops):
    probability=np.asarray(emission,np.float64);logs=np.asarray(emission_log,np.float64);valid=np.asarray(valid)
    if (probability.shape!=(len(index.sample_ids),40) or logs.shape!=probability.shape or valid.shape!=(len(probability),)
        or valid.dtype!=np.bool_ or np.any(probability<0) or not np.isfinite(probability).all()
        or not np.isfinite(logs).all() or not np.allclose(probability.sum(1),1,atol=1e-5)):
        raise ValueError('intermediate source probability/log/mask mismatch')
    blocks=session_blocks(index,rows,ops,gap_seconds=cfg['gap_seconds'])
    posterior=probability.copy();distillation=probability.copy();prediction=probability.argmax(1)
    session_id=np.full(len(probability),-1,np.int64);weights=np.zeros(len(probability),np.float64)
    for number,session in enumerate(blocks.sessions):
        selected=session[valid[session]]
        if not len(selected):continue
        result=ops['decode_unique_beam_posterior'](logs[selected],state,transition_weight=cfg['transition_weight'],
            beam_width=cfg['beam_width'],posterior_temperature=cfg['posterior_temperature'])
        posterior[selected]=result.marginals;prediction[selected]=result.paths[0];session_id[selected]=number
        distillation[selected],weights[selected]=ops['backed_off_structured_probability'](probability[selected],result.marginals,cfg['beam_width'])
    confidence=1.-(-(posterior*np.log(np.maximum(posterior,1e-300))).sum(1)/np.log(40))
    return {'emission_probability':probability,'structured_probability':posterior,'structured_distillation_probability':distillation,
        'structured_confidence':confidence,'structured_prediction':prediction,'structured_weight':weights,
        'sequence_session_id':session_id,'decoded_mask':session_id>=0,'target_mask':valid.copy()}


def _population(phase):
    if phase not in {'select','refit'}:raise ValueError('intermediate source decoder phase mismatch')
    return 'train12' if phase=='select' else 'refit14'


def _metadata(protocol,partition):
    body=json.loads((protocol.run_root/'protocol/task2_prepared.json').read_text(encoding='utf-8'))
    if body['protocol_sha256']!=protocol.identity():raise ValueError('source metadata preparation protocol mismatch')
    return ref(body['recording_metadata'][partition])


def _pins(protocol,context):return {name:pinned_source(protocol,name,context)[1]['sha256'] for name in SOURCE_FILES}


def allows_source_base_decoder(registry,record):
    if record.stage!=STAGE or record.kind!='transition' or record.phase not in {'select','refit'}:return False
    cfg=record.config;pop=_population(record.phase)
    if cfg.get('revision')!=REVISION or cfg.get('purpose')!='intermediate_source_base_fit_only' or cfg.get('fit_partition')!=pop or len(record.parents)!=3:return False
    p=registry.protocol;c=registry.context or VerificationContext(p);recipe,rr,_=load_reconstruction(p,c)
    if (cfg.get('recipe_sha256')!=canonical_hash(recipe) or canonical_bytes(cfg.get('decoder'))!=canonical_bytes(recipe['intermediate_source_base']['decoder'])
        or cfg.get('source_files')!=_pins(p,c) or record.rows!=row_index(read_public_rows(p.public_manifests[pop]))):return False
    parents=[registry.read(x) for x in record.parents]
    return rr in record.parents and _metadata(p,pop) in record.parents and sum(x.stage=='labels_'+pop and x.kind=='supervised_labels' and x.phase==record.phase for x in parents)==1


def fit_source_base_decoder(protocol,phase,context):
    pop=_population(phase)
    from .visual_teacher import _labels
    from .visual_student_v3 import _label_record
    from .imu_rf_teacher import _write_npz
    recipe,rr,_=load_reconstruction(protocol,context);cfg=dict(recipe['intermediate_source_base']['decoder'])
    registry=ArtifactRegistry(protocol,context=context);root=protocol.run_root/REVISION/STAGE/phase;alias=root/'artifact.json'
    if alias.exists():
        result=ref(json.loads(alias.read_text(encoding='utf-8')));registry.verify(result,STAGE,phase);return result
    metadata=_metadata(protocol,pop);index,rows=load_recording_rows(protocol,pop,metadata,context)
    inputs=load_stage_inputs(protocol,pop,phase,context=context);li,y=_labels(inputs,phase,protocol)
    if li!=index:raise ValueError('source decoder fit label IDs mismatch')
    ops=source_base_target_ops(protocol,context);blocks=session_blocks(index,rows,ops,gap_seconds=cfg['gap_seconds'])
    model=ops['fit_transition_model'](y,list(blocks.sessions),num_classes=40,trigram_backoff=cfg['trigram_backoff'])
    path=root/'transition.npz';digest=_write_npz(path,{name:getattr(model,name) for name in ['start_log_probability','end_log_probability','bigram_log_probability','trigram_log_probability']})
    body={'revision':REVISION,'purpose':'intermediate_source_base_fit_only','recipe_sha256':canonical_hash(recipe),'fit_partition':pop,
        'decoder':cfg,'source_alpha_default':.25,'source_files':_pins(protocol,context),'fit_rows':len(y),
        'sessions':len(blocks.sessions),'unknown_time_rows':len(blocks.unknown),'fit_session_ids_sha256':canonical_hash([x.tolist() for x in blocks.sessions]),
        'development_target_adaptation':False,'final_target_adaptation':False,'historical_exact_reproduction':False}
    write_json(root/'parameters.json',body);labels=_label_record(inputs,phase,index,protocol,context)
    result=registry.register(stage=STAGE,kind='transition',phase=phase,files=[path,root/'parameters.json'],
        file_digests={str(path.resolve()):digest,str((root/'parameters.json').resolve()):context.hash_file(root/'parameters.json')},
        parents=[rr,metadata,labels],rows=index,fit_users=inputs.partition.users,config=body,source_files=[Path(__file__),Path(session_metadata.__file__)])
    write_json(alias,result);print(json.dumps({'event':'source_base_decoder_fit','phase':phase,'fit_rows':len(y),'sessions':len(blocks.sessions)}),flush=True)
    return result


def make_source_base_targets(protocol,decoder,knn,partition,context):
    if partition not in {'train12','refit14'}:raise ValueError('intermediate source targets restricted to own phase fit pool')
    from .imu_rf_teacher import _write_npz
    from .p12_source_adapter import _read_prediction
    phase='select' if partition=='train12' else 'refit';registry=ArtifactRegistry(protocol,context=context)
    record=registry.verify(decoder,STAGE,phase);kr,arrays=_read_prediction(protocol,knn,'global_mixture',partition,context)
    if kr.config.get('operation')!='source_fullwindow_knn':raise ValueError('source-base structured teacher must be P85 global then fullwindow KNN')
    index=kr.rows;metadata=_metadata(protocol,partition);mi,rows=load_recording_rows(protocol,partition,metadata,context)
    if mi!=index:raise ValueError('source target public metadata IDs differ')
    root=protocol.run_root/REVISION/STAGE/phase;path=root/(partition+'_targets.npz');alias=path.with_suffix('.artifact.json')
    if alias.exists():
        result=ref(json.loads(alias.read_text(encoding='utf-8')));registry.verify(result,STAGE,'predict',index);return result
    ops=source_base_target_ops(protocol,context);file=next(Path(x) for x in record.files if Path(x).suffix=='.npz')
    saved=context.load_arrays(file);model=ops['transition_class'](**saved)
    values=structured_source_base(index,rows,arrays['probabilities'],arrays['logits'],arrays['valid'],model,record.config['decoder'],ops)
    probability=values['structured_distillation_probability'].astype(np.float32)
    payload={name:value.astype(np.float32) if value.dtype.kind=='f' else value for name,value in values.items()}
    payload.update(sample_ids=np.asarray(index.sample_ids),class_ids=np.arange(40),logits=np.log(np.clip(probability,1e-8,1.)),probabilities=probability,valid=values['target_mask'])
    digest=_write_npz(path,payload)
    result=registry.register(stage=STAGE,kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
        parents=[decoder,knn,metadata],rows=index,config={'partition':partition,'revision':REVISION,'purpose':'intermediate_source_base_fit_only',
            'unknown_time_policy':'identity_emission_no_session','label_free_application':True},source_files=[Path(__file__),Path(session_metadata.__file__)])
    write_json(alias,result);return result


def run_source_base_decoder_fit(protocol):
    context=VerificationContext(protocol);models={phase:fit_source_base_decoder(protocol,phase,context) for phase in ['select','refit']}
    body={'revision':REVISION,'base_protocol_sha256':protocol.identity(),'models':models,'decoder_fit_complete':True,
        'structured_targets_generated':False,'source_base_adapt12_complete':False,'task11_complete':False,'task12_started':False,'final_labels_read':False}
    immutable_bytes(protocol.run_root/'protocol'/REVISION/'source_base_decoder_fit.json',canonical_bytes(body)+b'\n');return body
