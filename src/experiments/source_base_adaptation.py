"""Own-phase intermediate adaptation; distinct from final Task12 target fitting."""
import ast
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace
import numpy as np
import torch
from scipy.special import softmax

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_types import RowIndex,Prediction,canonical_hash,canonical_bytes,write_json
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .no_vote_cache_import import immutable_bytes
from .p12_reconstruction import REVISION,ref,load_reconstruction
from .source_teacher_bank import pinned_source
from .teammate_source import verify_teammate_source,load_teammate_symbol

STAGE='source_base_adapt12_reconstructed'
SOURCE_FILES=('aligned_multimodal/adapt_p87s_structured_student.py','aligned_multimodal/adapt_p87s_test_student.py')


def source_adaptation_ops(protocol,context):
    key=(protocol.identity(),'source_base_adaptation_ops')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        names=['train_label_free','LabelFreePseudoDataset','make_loader','seed_all','configure_adaptation_parameters']
        context.public_rows[key]={name:load_teammate_symbol(report,'adapt_p87s_structured_student',name) for name in names}
    return context.public_rows[key]


def train_source_with_resume(model,loader,args,ops,path,identity,*,device='cuda',fail_after=None):
    """Keep every original loss/update; add complete-epoch restore/save hooks only."""
    from .visual_student_v3 import _save_torch
    from .mobind_fusion import _rng_state,_restore_rng
    path=Path(path);args_hash=canonical_hash(vars(args));original=ops['train_label_free']
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)));function=tree.body[0]
    histories=[i for i,x in enumerate(function.body) if isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name) and x.target.id=='history']
    loops=[x for x in function.body if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='epoch']
    if len(histories)!=1 or len(loops)!=1 or ast.unparse(loops[0].iter)!='range(1, args.epochs + 1)':raise ValueError('source label-free epoch structure changed')
    function.body[histories[0]]=ast.parse('history, start_epoch = _no_vote_resume_initialize(model, optimizer, scaler)').body[0]
    loops[0].iter.args[0]=ast.Name(id='start_epoch',ctx=ast.Load())
    loops[0].body.append(ast.parse('_no_vote_resume_checkpoint(model, optimizer, scaler, history, epoch)').body[0])
    def initialize(network,optimizer,scaler):
        if not path.exists():return [],1
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['identity']!=identity or saved['args_sha256']!=args_hash:raise ValueError('intermediate adaptation resume identity/recipe mismatch')
        history=saved['history']
        if [x['epoch'] for x in history]!=list(range(1,len(history)+1)) or len(history)>args.epochs:raise ValueError('intermediate adaptation resume budget/history mismatch')
        network.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);_restore_rng(saved)
        print(json.dumps({'event':'source_base_resume','completed_epochs':len(history),'next_epoch':len(history)+1}),flush=True)
        return history,len(history)+1
    def checkpoint(network,optimizer,scaler,history,epoch):
        _save_torch(path,{'identity':identity,'args_sha256':args_hash,'model_state':network.state_dict(),
            'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'history':history,**_rng_state()})
        if fail_after==epoch:raise RuntimeError('simulated complete-epoch source-base interruption')
    namespace={**original.__globals__,'_no_vote_resume_initialize':initialize,'_no_vote_resume_checkpoint':checkpoint}
    exec(compile(ast.fix_missing_locations(tree),'<no_vote_resumable_label_free>','exec'),namespace)
    return namespace['train_label_free'](model,loader,args,torch.device(device))


def train_source_base(protocol,phase,targets,context,*,device='cuda'):
    if phase not in {'select','refit'}:raise ValueError('intermediate source-base phase must be select or refit')
    from .mobind_fusion import fusion_inputs,load_fusion_data,dataset_for
    from .visual_student_v3 import prior,_save_torch
    from .visual_student import phase_lock
    from .p12_source_adapter import _read_prediction
    recipe,rr,_=load_reconstruction(protocol,context);cfg=dict(recipe['intermediate_source_base']['recipe']);args=adaptation_args(cfg)
    pop='train12' if phase=='select' else 'refit14';registry=ArtifactRegistry(protocol,context=context)
    target_record,arrays=_read_prediction(protocol,targets,'source_base_decoder',pop,context)
    if target_record.config.get('purpose')!='intermediate_source_base_fit_only':raise ValueError('wrong intermediate target purpose')
    index=target_record.rows;refs=fusion_inputs(protocol,context);data=load_fusion_data(protocol,pop,context,refs)
    valid=arrays['target_mask']
    if data['index']!=index or not np.array_equal(valid,arrays['valid']) or np.any(valid&~data['observed_valid']):raise ValueError('source-base target/student exact input coverage mismatch')
    chosen=RowIndex(tuple(s for s,v in zip(index.sample_ids,valid) if v),tuple(u for u,v in zip(index.user_ids,valid) if v),index.class_ids)
    if not chosen.sample_ids:raise ValueError('source-base has no fit-pool pseudo rows')
    from .mobind_fusion import source_fusion_math
    ops=source_adaptation_ops(protocol,context);fops=source_fusion_math(protocol,context)
    base=ref(json.loads((protocol.run_root/'A7'/phase/'artifact.json').read_text(encoding='utf-8')));registry.verify(base,'A7',phase)
    pr,prob=prior(protocol,phase,context);parents=[rr,base,targets,*data['parents'],pr]
    root=protocol.run_root/REVISION/STAGE/phase
    identity=canonical_hash({'phase':phase,'parents':parents,'rows':chosen,'recipe':cfg,'producer':context.hash_file(Path(__file__))})
    with phase_lock(root):
        immutable_bytes(root/'identity.json',canonical_bytes({'identity':identity})+b'\n')
        if (root/'artifact.json').exists():
            result=ref(json.loads((root/'artifact.json').read_text(encoding='utf-8')));registry.verify(result,STAGE,phase);return result
        network=initialize_source_base(protocol,phase,base,refs,context,seed=cfg['seed'],device=device)
        dataset=dataset_for(data,rows=chosen,temporal_indices=fops['temporal_indices'])
        pseudo=ops['LabelFreePseudoDataset'](dataset,dict(zip(chosen.sample_ids,arrays['structured_distillation_probability'][valid])),
            dict(zip(chosen.sample_ids,arrays['structured_confidence'][valid].tolist())))
        loader=ops['make_loader'](pseudo,args.batch_size,args.workers,shuffle=True)
        history=train_source_with_resume(network,loader,args,ops,root/'resume.pt',identity,device=device)
        if len(history)!=12 or any(x['train_samples']!=len(chosen.sample_ids) or x['ground_truth_fields_seen']!=0 for x in history):raise ValueError('source-base fixed twelve epochs/label-free population mismatch')
        write_json(root/'history.json',history)
        _save_torch(root/'checkpoint.pt',{'stage':STAGE,'model_state':network.state_dict(),'source_phase':phase,'epochs':12})
        metadata={'revision':REVISION,'purpose':'intermediate_source_base_fit_only','fit_partition':pop,'budget_epochs':12,
            'adaptation_recipe':cfg,'recipe_sha256':canonical_hash(recipe),'fit_rows':len(chosen.sample_ids),'fit_ids_sha256':canonical_hash(chosen),
            'supervision':'own_phase_structured_pseudo_no_ground_truth_batch','base_model':base,'targets':targets,'source_files':_pins(protocol,context),
            'prior':prob.tolist(),'source_scope':'heads','development_target_adaptation':False,'final_target_adaptation':False,
            'historical_exact_reproduction':False,'teacher_or_labels_needed_at_inference':False,'complete_epoch_resume':True}
        write_json(root/'model.json',metadata)
        result=registry.register(stage=STAGE,kind='supervised_model',phase=phase,files=[root/'checkpoint.pt',root/'model.json',root/'history.json'],
            parents=parents,rows=index,fit_users=protocol.partitions[pop].users,config=metadata,source_files=[Path(__file__)])
        write_json(root/'artifact.json',result);return result


def adaptation_args(config):
    if config['epochs']!=12 or config['scope']!='heads' or config['confidence_power']!=0.:raise ValueError('intermediate adaptation requires source twelve epochs/heads/no confidence filtering')
    return SimpleNamespace(**{k:v for k,v in config.items() if k!='scope'},adaptation_scope=config['scope'],target='structured',
        emission_warmup_epochs=0,max_train_batches=0)


def _pins(protocol,context):return {name:pinned_source(protocol,name,context)[1]['sha256'] for name in SOURCE_FILES}


def initialize_source_base(protocol,phase,base,refs,context,*,seed,device='cuda',load_base=True):
    from .mobind_fusion import source_fusion_math,fusion_args,assemble_fusion,motion_model_config,motion_args
    from .visual_student_v3 import source_student_math
    ops=source_fusion_math(protocol,context);source_adaptation_ops(protocol,context)['seed_all'](seed)
    recipe=protocol.recipe['visual_student'];kwargs={k:recipe[k] for k in ['width','frames','dropout','fusion_mode','temporal_modeling',
        'exact_time_modeling','cross_view_time_modeling','spatial_region_modeling','region_temporal_modeling','structured_region_modeling','enable_distillation_projection']}
    visual=source_student_math(protocol,context)['model_class'](classes=40,kinetics_pretrained=False,**kwargs)
    config=motion_model_config(motion_args(protocol));motion=ops['module'].P86MoBindLite(**config)
    network=assemble_fusion(visual,motion,config,fusion_args(protocol),ops)
    if load_base:
        registry=ArtifactRegistry(protocol,context=context);record=registry.verify(base,'A7',phase)
        file=next(Path(x) for x in record.files if Path(x).name=='checkpoint.pt');registry.verify_file(record,file)
        network.load_state_dict(torch.load(file,map_location='cpu',weights_only=True)['model_state'])
    return network.to(device)


def allows_source_base_adaptation(registry,record):
    if record.stage!=STAGE or record.kind!='supervised_model' or record.phase not in {'select','refit'}:return False
    cfg=record.config;pop='train12' if record.phase=='select' else 'refit14'
    if (cfg.get('revision')!=REVISION or cfg.get('purpose')!='intermediate_source_base_fit_only' or cfg.get('fit_partition')!=pop
        or cfg.get('budget_epochs')!=12 or cfg.get('source_scope')!='heads' or len(record.parents)!=8):return False
    p=registry.protocol;c=registry.context or VerificationContext(p);recipe,rr,_=load_reconstruction(p,c)
    if (cfg.get('recipe_sha256')!=canonical_hash(recipe) or canonical_bytes(cfg.get('adaptation_recipe'))!=canonical_bytes(recipe['intermediate_source_base']['recipe'])
        or cfg.get('source_files')!=_pins(p,c) or record.rows!=row_index(read_public_rows(p.public_manifests[pop]))):return False
    parents=[registry.read(x) for x in record.parents]
    base=[x for x in parents if x.stage=='A7' and x.kind=='supervised_model' and x.phase==record.phase and x.config.get('variant')=='full']
    targets=[x for x in parents if x.stage=='source_base_decoder' and x.kind=='predictions' and x.config.get('partition')==pop]
    return (rr in record.parents and len(base)==len(targets)==1 and base[0].rows==targets[0].rows==record.rows
        and targets[0].config.get('purpose')=='intermediate_source_base_fit_only'
        and all(sum(x.stage==stage for x in parents)==1 for stage in ['motion_windows','source_transform','pixels','visual_sequence','class_prior'])
        and cfg.get('development_target_adaptation') is False and cfg.get('final_target_adaptation') is False)


def predict_source_base(protocol,model,partition,refs,context,*,device='cuda'):
    from .mobind_fusion import load_fusion_data,dataset_for,source_fusion_math,fusion_args
    from .imu_rf_teacher import _write_npz
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(model,STAGE)
    phase='select' if partition in {'train12','development2'} else 'refit'
    if record.phase!=phase:raise ValueError('source-base prediction own-phase mismatch')
    data=load_fusion_data(protocol,partition,context,refs);index=data['index'];ops=source_fusion_math(protocol,context)
    root=protocol.run_root/REVISION/STAGE/phase;path=root/(partition+'_targets.npz');alias=path.with_suffix('.artifact.json')
    if alias.exists():
        from .p12_source_adapter import _read_prediction
        reference=ref(json.loads(alias.read_text(encoding='utf-8')));_,a=_read_prediction(protocol,reference,STAGE,partition,context)
        return Prediction(index,a['logits'],a['probabilities'],a['valid'],reference)
    network=initialize_source_base(protocol,phase,None,refs,context,seed=record.config['adaptation_recipe']['seed'],device='cpu',load_base=False)
    file=next(Path(x) for x in record.files if Path(x).name=='checkpoint.pt');registry.verify_file(record,file)
    network.load_state_dict(torch.load(file,map_location='cpu',weights_only=True)['model_state']);network.eval().to(device)
    dataset=dataset_for(data);scores=np.tile(np.log(np.maximum(record.config['prior'],1e-12)),(len(index.sample_ids),1)).astype(np.float32)
    offset=0
    with torch.inference_mode():
        for batch in ops['module'].loader(dataset,fusion_args(protocol),False):
            n=len(batch['sample_id']);active=np.flatnonzero(data['observed_valid'][offset:offset+n])
            if len(active):
                batch=ops['to_device'](batch,torch.device(device));take=torch.as_tensor(active,device=device)
                fields={k:v.index_select(0,take) if torch.is_tensor(v) else [v[i] for i in active] for k,v in batch.items()}
                with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,enabled=torch.device(device).type=='cuda'):
                    values=ops['model_forward'](network,fields)
                scores[offset+active]=values['logits'].float().cpu().numpy()
            offset+=n
    probability=softmax(scores,axis=1);probability[~data['observed_valid']]=record.config['prior']
    digest=_write_npz(path,dict(sample_ids=np.asarray(index.sample_ids),class_ids=np.arange(40),logits=scores,probabilities=probability,valid=data['observed_valid']))
    inputs=load_stage_inputs(protocol,partition,'predict',context=context)
    reference=registry.register(stage=STAGE,kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
        parents=[model,*data['parents'],inputs.parents['manifest']],rows=index,
        config={'partition':partition,'revision':REVISION,'fit_phase':phase,'operation':'source_base_adapt12_raw_probability'},source_files=[Path(__file__)])
    write_json(alias,reference);return Prediction(index,scores,probability,data['observed_valid'],reference)


def run_source_base_adaptation(protocol,*,device='cuda'):
    from .mobind_fusion import fusion_inputs
    context=VerificationContext(protocol);refs=fusion_inputs(protocol,context);root=protocol.run_root/'protocol'/REVISION
    targets=json.loads((root/'source_base_structured_targets.json').read_text(encoding='utf-8'))
    if targets['base_protocol_sha256']!=protocol.identity():raise ValueError('source-base fit targets protocol mismatch')
    models={};outputs={}
    for phase,pop in [('select','train12'),('refit','refit14')]:models[phase]=train_source_base(protocol,phase,ref(targets['outputs'][pop]),context,device=device)
    for pop in protocol.partitions:
        phase='select' if pop in {'train12','development2'} else 'refit'
        prediction=predict_source_base(protocol,models[phase],pop,refs,context,device=device);outputs[pop]=prediction.artifact
        print(json.dumps({'event':'source_base_raw_prediction','partition':pop,'rows':len(prediction.valid),'valid':int(prediction.valid.sum())}),flush=True)
    body={'revision':REVISION,'base_protocol_sha256':protocol.identity(),'models':models,'outputs':outputs,
        'source_base_adapt12_complete':True,'budget_epochs_each_phase':12,'development_target_adaptation':False,
        'final_target_adaptation':False,'final_labels_read':False,'task11_complete':False,'a8_generated':False,'task12_started':False}
    immutable_bytes(root/'source_base_adapt12_component.json',canonical_bytes(body)+b'\n');return body
