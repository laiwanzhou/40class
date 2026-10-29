"""Task10 source separate/additive fusion and explicitly independent controls."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import random
import hashlib
import time
import ast
import inspect
import textwrap

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from scipy.special import softmax

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_types import ArtifactRef,RowIndex,Prediction,canonical_bytes,canonical_hash,write_json
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .no_vote_cache_import import immutable_bytes
from .motion_cache import MOTION_FIELDS,motion_item,load_motion
from .mobind_pretrain import motion_references,motion_validity,validate_transform,model_config as motion_model_config,motion_args
from .teammate_source import verify_teammate_source,load_teammate_symbol

VARIANTS={'full':'A7','mask_motion':'A7-mask','shuffle_s':'A7-shuffle-S','shuffle_i':'A7-shuffle-I'}
SOURCE_FILES=('train_p86_mobind_fusion_proxy.py','p86_mobind_lite_model.py','p86_cached_motion_data.py',
    'p86_mc3_visual_model.py','train_p86_cached_motion_proxy.py','train_p86_visual_student_oof.py')


def compatible_residual_class(source_class):
    """Keep the pinned source forward; rewrite its one invalid rank3 mask assignment.

    This only makes the first padding token visible in all-empty windows. It
    changes no attention values, availability masks, loss or trainable parameter.
    The source module/class on disk and in sys.modules remain untouched.
    """
    original=source_class.forward;tree=ast.parse(textwrap.dedent(inspect.getsource(original)));count=0
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal count
            if ast.unparse(node.targets[0])=='safe_mask[empty, 0]':
                count+=1
                return ast.parse('safe_mask[..., 0] = safe_mask[..., 0] | empty').body[0]
            return node
    tree=ast.fix_missing_locations(Rewrite().visit(tree))
    if count!=1:raise ValueError('pinned source residual padding compatibility expression changed')
    namespace=dict(original.__globals__);exec(compile(tree,'<no_vote_source_padding_compat>','exec'),namespace)
    return type('NoVoteCompatibleResidual',(source_class,),{'forward':namespace['forward']})


def source_fusion_math(protocol,context):
    key=(protocol.identity(),'task10_source_math')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        function=load_teammate_symbol(report,'train_p86_mobind_fusion_proxy','train_stage')
        source=sys.modules['train_p86_mobind_fusion_proxy'];data=sys.modules['p86_cached_motion_data']
        context.public_rows[key]={'module':source,'train_stage':function,'losses':source.losses,
            'residual_class':compatible_residual_class(source.P86MoBindMotionResidual),
            'set_stage':source.set_stage,'model_forward':source.model_forward,'class_weights':source.class_weights,
            'seed_all':source.seed_all,'temporal_indices':data.P86CachedSequenceMotionDataset.temporal_indices,
            'collate':data.collate_p86_cached_motion,'to_device':source.to_device}
    return context.public_rows[key]


def fusion_args(protocol):
    recipe=dict(protocol.recipe['fusion'])
    if (recipe.get('stage_a_epochs')!=4 or recipe.get('stage_b_epochs')!=20
            or recipe.get('global_fusion_mode')!='additive' or recipe.get('fusion_position')!='clip'
            or recipe.get('live_visual_anchor') or not recipe.get('joint_freeze_pretrained_encoders')):
        raise ValueError('source fusion requires separate/additive fixed4+20 and frozen same-phase anchor')
    recipe.update(modality='separate',stage_b_fusion_learning_rate=None,imu_event_features=None,imu_teacher_logits=None,
        all_label_refit=True,final_refit=False,subject_holdout_users=None,max_train_batches=0,max_eval_batches=0,
        initial_fusion_checkpoint=None,temporal_anchor_checkpoint=None,compact_sequence_cache=None)
    return SimpleNamespace(**recipe)


def assemble_fusion(visual,pretrained,config,args,ops):
    """Exact source build_model's active separate/clip branch; input models already loaded."""
    source=ops['module'];state=torch.random.get_rng_state()
    encoder=source.P86SeparateMotionEncoder(pretrained.skeleton_encoder,pretrained.imu_encoder,
        width=int(config['width']),modality_dropout=args.separate_modality_dropout)
    torch.random.set_rng_state(state)
    residual=ops['residual_class']('separate',encoder,pretrained.skeleton_head,pretrained.skeleton_teacher_projection,
        visual_width=512,motion_width=int(config['width']),dropout=float(config['dropout']),
        initial_residual_strength=args.initial_residual_strength,reliability_event_feature_width=0,
        global_fusion_mode=args.global_fusion_mode,reliability_groups=args.reliability_groups)
    return source.P86UnifiedMoBindStudent(visual,residual)


def fusion_losses(output,batch,weights,args,ops):
    return ops['losses'](output,batch,weights,args)


def run_source_stage(model,loader,labels,args,device,stage,epochs,ops):
    # Source owns freeze flags, training modes, groups, optimizer, cosine, AMP,
    # clipping and every loss reduction. No custom missing-modality gates.
    return ops['train_stage'](model,loader,labels,args,torch.device(device),stage,epochs)


def fit_shuffle(rows,*,seed):
    return np.random.default_rng(seed).permutation(len(rows.sample_ids))


class FusionDataset(Dataset):
    def __init__(self,motion,motion_index,sequence,sequence_index,pixels,pixel_index,anchor,*,rows,
                 labels=None,teacher=None,temporal_indices=None,variant='full',shuffle=None):
        if variant not in VARIANTS:raise ValueError('unknown fusion control')
        self.motion=motion;self.sequence=sequence;self.pixels=pixels;self.anchor=anchor;self.rows=rows
        self.labels=labels;self.teacher=teacher;self.temporal_indices=temporal_indices;self.variant=variant
        self.positions=[]
        for index in [motion_index,sequence_index,pixel_index]:
            lookup={s:i for i,s in enumerate(index.sample_ids)}
            if any(s not in lookup or index.user_ids[lookup[s]]!=u for s,u in zip(rows.sample_ids,rows.user_ids)):
                raise ValueError('fusion Dataset ID/user mismatch')
            self.positions.append([lookup[s] for s in rows.sample_ids])
        if labels is not None and set(labels)!=set(rows.sample_ids):raise ValueError('fusion labels ID mismatch')
        if teacher is not None:
            index=teacher.prediction.index;lookup={s:i for i,s in enumerate(index.sample_ids)}
            if teacher.features is None or any(s not in lookup or index.user_ids[lookup[s]]!=u for s,u in zip(rows.sample_ids,rows.user_ids)):
                raise ValueError('fusion teacher ID/user/features mismatch')
            self.teacher_positions=[lookup[s] for s in rows.sample_ids]
        if shuffle is not None and (labels is None or sorted(np.asarray(shuffle).tolist())!=list(range(len(self)))):
            raise ValueError('shuffle donors must be an exact permutation of labeled fit rows')
        if variant.startswith('shuffle') and labels is not None and shuffle is None:raise ValueError('fit shuffle requires bound permutation')
        self.shuffle=shuffle

    def __len__(self):return len(self.rows.sample_ids)

    def __getitem__(self,item):
        mi,si,pi=(v[item] for v in self.positions);indices=self.temporal_indices(16) if self.temporal_indices else np.arange(16)
        raw={k:self.motion[k][mi] for k in MOTION_FIELDS}
        if self.shuffle is not None:
            donor=self.positions[0][int(self.shuffle[item])];prefix='skeleton_' if self.variant=='shuffle_s' else 'imu_'
            raw={k:self.motion[k][donor] if k.startswith(prefix) else value for k,value in raw.items()}
        raw=motion_item(raw,temporal_index=indices)
        if self.variant=='mask_motion':raw={k:np.zeros_like(v) for k,v in raw.items()}
        result={k:torch.from_numpy(v) for k,v in raw.items()}
        frame=np.asarray(self.pixels['source_frame_indices'][pi]);count=int(frame.max(initial=0))+1
        result.update(backbone_sequence=torch.from_numpy(np.asarray(self.sequence[si],np.float32)[:,:,indices].copy()),
            view_valid=torch.from_numpy(np.asarray(self.pixels['view_valid'][pi],bool)[:,indices].copy()),
            view_quality=torch.from_numpy(np.asarray(self.pixels['view_quality'][pi],np.float16).astype(np.float32)[:,indices].copy()),
            global_time_position=torch.from_numpy(frame[:,indices].astype(np.float32)/max(count-1,1)),
            anchor_logits=torch.from_numpy(np.asarray(self.anchor[si],np.float16).astype(np.float32)),
            sample_id=self.rows.sample_ids[item],user_id=self.rows.user_ids[item],cache_index=mi)
        if self.labels is not None:result['label']=torch.tensor(self.labels[result['sample_id']],dtype=torch.long)
        if self.teacher is not None:
            t=self.teacher_positions[item]
            result.update(teacher_logits=torch.from_numpy(np.asarray(self.teacher.prediction.logits[t],np.float32).copy()),
                teacher_features=torch.from_numpy(np.asarray(self.teacher.features[t],np.float32).copy()))
        return result


def ref(value):
    return ArtifactRef(Path(value['record_path']),value['sha256'])


def fusion_inputs(protocol,context):
    from .simple_fusion import current_inputs
    _,proofs=current_inputs(protocol,context)
    root=protocol.run_root;registry=ArtifactRegistry(protocol,context=context)
    student=json.loads((root/'protocol/task5_artifacts_ready.json').read_text(encoding='utf-8'))
    motion=json.loads((root/'protocol/task8_current.json').read_text(encoding='utf-8'))
    result={'proofs':proofs,'motion':motion_references(protocol),'phases':{},'sequences':{}}
    for phase in ['select','refit']:
        a2=ref(json.loads((root/'A2'/phase/'artifact.json').read_text(encoding='utf-8')))
        a5=ref(motion['models'][phase]);registry.verify(a2,'A2',phase);registry.verify(a5,'A5',phase)
        result['phases'][phase]={'visual':a2,'motion':a5}
    for pop in protocol.partitions:
        result['sequences'][pop]=ref(student['partitions'][pop]['sequence'])
    return result


def load_fusion_data(protocol,pop,context,refs):
    from .visual_student_v3 import pixel_refs
    key=(protocol.identity(),'task10_data',pop)
    if key in context.public_rows:return context.public_rows[key]
    registry=ArtifactRegistry(protocol,context=context);phase='select' if pop in {'train12','development2'} else 'refit'
    index=row_index(read_public_rows(protocol.public_manifests[pop]));cache='final4' if pop=='final4' else 'refit14'
    mref=refs['motion'][cache]['windows'];transform=refs['motion'][cache]['normalization']
    validate_transform(protocol,mref,transform,context);mr,ma=load_motion(mref,protocol=protocol,context=context)
    pixels=pixel_refs(protocol)[cache];pr=registry.verify(pixels,'pixels','raw')
    files={Path(x).name:Path(x) for x in pr.files};pa={}
    for name in ['view_valid','view_quality','source_frame_indices']:
        pa[name]=context.load_arrays(registry.verify_file(pr,files[name+'.npy']))
    sr=registry.verify(refs['sequences'][pop],'visual_sequence',phase,index)
    if (sr.config.get('model_sha256')!=refs['phases'][phase]['visual'].sha256
            or sr.config.get('pixels_sha256')!=pixels.sha256 or sr.config.get('checkpoint_epoch')!=16
            or refs['phases'][phase]['visual'] not in sr.parents or pixels not in sr.parents
            or not sr.config.get('native_backbone_sequence')):
        raise ValueError('fusion requires native sequence from exact own-phase A2/pixels')
    files={Path(x).name:Path(x) for x in sr.files};sequence=context.load_arrays(files['sequence.npy'])
    if sequence.shape!=(len(index.sample_ids),2,3,16,512) or sequence.dtype!=np.float16 or sequence.flags.writeable:
        raise ValueError('fusion sequence header/read-only mismatch')
    anchor=context.load_arrays(registry.verify_file(sr,files['anchor_logits.npy']))
    anchor_valid=context.load_arrays(registry.verify_file(sr,files['anchor_valid.npy']))
    done=context.load_arrays(registry.verify_file(sr,files['completed.npy']))
    if anchor.shape!=(len(index.sample_ids),40) or not np.isfinite(anchor).all() or not done.all():raise ValueError('fusion anchor/incomplete sequence')
    mi={s:i for i,s in enumerate(mr.rows.sample_ids)};pi={s:i for i,s in enumerate(pr.rows.sample_ids)}
    if any(mi.get(s) is None or pi.get(s) is None or mr.rows.user_ids[mi[s]]!=u or pr.rows.user_ids[pi[s]]!=u for s,u in zip(index.sample_ids,index.user_ids)):
        raise ValueError('fusion motion/pixels exact ID/user mismatch')
    positions=[mi[s] for s in index.sample_ids];p_positions=[pi[s] for s in index.sample_ids]
    if not np.array_equal(ma['source_frame_indices'][positions],pa['source_frame_indices'][p_positions]):raise ValueError('fusion source-frame alignment drift')
    visible=pa['view_valid'][p_positions].reshape(len(positions),-1).any(1)
    if not np.array_equal(anchor_valid,visible):raise ValueError('fusion anchor validity mismatch')
    sk,imu=motion_validity(ma);valid=visible|sk[positions]|imu[positions]
    result=dict(index=index,motion_index=mr.rows,motion=ma,pixel_index=pr.rows,pixels=pa,sequence=sequence,anchor=anchor,
        observed_valid=valid,parents=[mref,transform,pixels,refs['sequences'][pop]])
    context.public_rows[key]=result;return result


def dataset_for(data,*,rows=None,**kwargs):
    return FusionDataset(data['motion'],data['motion_index'],data['sequence'],data['index'],data['pixels'],data['pixel_index'],
        data['anchor'],rows=rows or data['index'],**kwargs)


def initialize_fusion(protocol,phase,refs,context,*,device='cuda',load_parents=True):
    from .visual_student_v3 import source_student_math
    ops=source_fusion_math(protocol,context);args=fusion_args(protocol);ops['seed_all'](args.seed)
    recipe=protocol.recipe['visual_student'];kwargs={k:recipe[k] for k in ['width','frames','dropout','fusion_mode','temporal_modeling',
        'exact_time_modeling','cross_view_time_modeling','spatial_region_modeling','region_temporal_modeling','structured_region_modeling','enable_distillation_projection']}
    visual=source_student_math(protocol,context)['model_class'](classes=40,kinetics_pretrained=False,**kwargs)
    config=motion_model_config(motion_args(protocol));base=ops['module'].P86MoBindLite(**config)
    if load_parents:
        registry=ArtifactRegistry(protocol,context=context)
        for key,target in [('visual',visual),('motion',base)]:
            reference=refs['phases'][phase][key];record=registry.verify(reference,'A2' if key=='visual' else 'A5',phase)
            ck=(protocol.identity(),'task10_parent_state',reference.sha256)
            if ck not in context.public_rows:
                path=next(Path(x) for x in record.files if Path(x).name=='checkpoint.pt');registry.verify_file(record,path)
                context.public_rows[ck]=torch.load(path,map_location='cpu',weights_only=True)
            state=context.public_rows[ck]
            target.load_state_dict(state if key=='visual' else state['model_state'],strict=True)
    network=assemble_fusion(visual,base,config,args,ops)
    return network.to(device)


def _source_pins(protocol,context):
    key=(protocol.identity(),'task10_pins')
    if key not in context.public_rows:
        table=json.loads((protocol.source_root.parent/'source_manifest.json').read_text(encoding='utf-8'))
        pins={x['path']:x['sha256'] for x in table['files']}
        context.public_rows[key]={name:context.hash_file(protocol.source_root/'aligned_multimodal'/name,pins['aligned_multimodal/'+name]) for name in SOURCE_FILES}
    return context.public_rows[key]


def allows_fusion_stage(registry,record):
    if record.stage not in VARIANTS.values() or record.kind!='supervised_model' or record.phase not in {'select','refit'}:return False
    cfg=record.config;pop='train12' if record.phase=='select' else 'refit14';p=registry.protocol;c=registry.context or VerificationContext(p)
    if (cfg.get('variant') not in VARIANTS or VARIANTS[cfg['variant']]!=record.stage or cfg.get('fit_partition')!=pop
            or cfg.get('task8_revision')!='ir_supervision_v2' or cfg.get('stage_epochs')!={'A':4,'B':20}
            or canonical_bytes(cfg.get('source_recipe'))!=canonical_bytes(p.recipe['fusion'])):return False
    refs=fusion_inputs(p,c)
    if canonical_bytes(cfg.get('prerequisites'))!=canonical_bytes(refs['proofs']) or dict(cfg.get('source_files',{}))!=_source_pins(p,c):return False
    from .visual_student_v3 import teacher_ref,prior
    expected=[refs['phases'][record.phase][k] for k in ['visual','motion']]
    expected += [refs['motion']['refit14'][k] for k in ['windows','normalization']]
    expected += [refs['sequences'][pop],teacher_ref(p,pop),prior(p,record.phase,c)[0]]
    labels=[registry.read(x) for x in record.parents if registry.read(x).kind=='supervised_labels']
    return (all(x in record.parents for x in expected) and len(record.parents)==9 and len(labels)==1
        and labels[0].stage=='labels_'+pop and labels[0].phase==record.phase)


class TracedLoader:
    def __init__(self,loader):self.loader=loader;self.dataset=loader.dataset;self.orders=[]
    def __iter__(self):
        ids=[]
        for batch in self.loader:
            ids.extend(batch['sample_id']);yield batch
        self.orders.append(canonical_hash(tuple(ids)))


def _head_digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        if k.startswith(tuple('visual.'+x+'.' for x in ['stem','layer1','layer2','layer3','layer4'])):continue
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _rng_state():
    nr=np.random.get_state()
    return {'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
        'numpy_rng':(nr[0],torch.from_numpy(nr[1].astype(np.int64)),nr[2],nr[3],nr[4]),
        'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);nr=state['numpy_rng']
    np.random.set_state((nr[0],nr[1].numpy().astype(np.uint32),nr[2],nr[3],nr[4]))
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])


def train_fusion(variant,phase,*,protocol,refs,context,device='cuda'):
    from .visual_student_v3 import load_teacher,prior,_label_record,_save_torch
    from .visual_teacher import _labels
    from .visual_student import phase_lock
    if variant not in VARIANTS or phase not in {'select','refit'}:raise ValueError('unknown fusion variant/phase')
    pop='train12' if phase=='select' else 'refit14';inputs=load_stage_inputs(protocol,pop,phase,context=context)
    index,y=_labels(inputs,phase,protocol);data=load_fusion_data(protocol,pop,context,refs);valid=data['observed_valid']
    chosen=RowIndex(tuple(s for s,v in zip(index.sample_ids,valid) if v),tuple(u for u,v in zip(index.user_ids,valid) if v),index.class_ids)
    labels=dict(zip(index.sample_ids,y.tolist()));teacher=load_teacher(protocol,pop,context);args=fusion_args(protocol);ops=source_fusion_math(protocol,context)
    donors=fit_shuffle(chosen,seed=args.seed) if variant.startswith('shuffle') else None
    dataset=dataset_for(data,rows=chosen,labels={s:labels[s] for s in chosen.sample_ids},teacher=teacher,
        temporal_indices=ops['temporal_indices'],variant=variant,shuffle=donors)
    root=protocol.run_root/VARIANTS[variant]/phase;registry=ArtifactRegistry(protocol,context=context)
    prior_ref,prob=prior(protocol,phase,context);label_ref=_label_record(inputs,phase,index,protocol,context)
    parents=[refs['phases'][phase]['visual'],refs['phases'][phase]['motion'],*data['parents'],teacher.prediction.artifact,prior_ref,label_ref]
    identity=canonical_hash({'protocol':protocol.identity(),'variant':variant,'phase':phase,'parents':parents,
        'producer':context.hash_file(Path(__file__)),'fit':canonical_hash(chosen),'recipe':protocol.recipe['fusion']})
    with phase_lock(root):
        immutable_bytes(root/'identity.json',canonical_bytes({'identity':identity})+b'\n')
        if (root/'artifact.json').exists():
            result=ref(json.loads((root/'artifact.json').read_text(encoding='utf-8')));registry.verify(result,VARIANTS[variant],phase,index);return result
        network=initialize_fusion(protocol,phase,refs,context,device=device);initial=_head_digest(network)
        histories=[];orders={};stage_a=root/'stage_A_resume.pt'
        if stage_a.exists():
            state=torch.load(stage_a,map_location='cpu',weights_only=True)
            if state['identity']!=identity:raise ValueError('fusion stage resume identity mismatch')
            network.load_state_dict(state['model_state']);histories=state['history'];orders=state['orders'];_restore_rng(state)
        for stage,epochs in [('A',4),('B',20)]:
            if stage=='A' and histories:continue
            loader=TracedLoader(ops['module'].loader(dataset,args,True))
            print(json.dumps({'event':'fusion_stage_start','variant':variant,'phase':phase,'stage':stage,'epochs':epochs,'rows':len(dataset)}),flush=True)
            history=run_source_stage(network,loader,y[valid],args,device,stage,epochs,ops)
            if len(history)!=epochs or any(x['train_samples']!=len(dataset) for x in history):raise ValueError('incomplete source fusion stage/population')
            histories+=history;orders[stage]=loader.orders;write_json(root/'history.json',histories)
            if stage=='A':_save_torch(stage_a,{'identity':identity,'model_state':network.state_dict(),'history':histories,'orders':orders,**_rng_state()})
        if variant!='full':
            full=json.loads((protocol.run_root/'A7'/phase/'model.json').read_text(encoding='utf-8'))
            if full['initial_head_sha256']!=initial or full['epoch_order_digests']!=orders:raise ValueError('fusion controls must share initial weights and every source epoch order')
        cfg={'variant':variant,'role':'source_main' if variant=='full' else 'diagnostic_training_control',
            'fit_partition':pop,'source_recipe':dict(protocol.recipe['fusion']),'stage_epochs':{'A':4,'B':20},
            'checkpoint_selection':'fixed_A4_B20_last','fit_rows':len(dataset),'excluded_all_unavailable':int((~valid).sum()),
            'fit_ids_sha256':canonical_hash(chosen),'class_weight_population_ids_sha256':canonical_hash(chosen),
            'task8_revision':'ir_supervision_v2','prerequisites':refs['proofs'],'source_files':_source_pins(protocol,context),
            'prior':prob.tolist(),'initial_head_sha256':initial,'epoch_order_digests':orders,
            'shuffle_fit_only':donors is not None,'shuffle_donors_sha256':canonical_hash(donors.tolist()) if donors is not None else None,
            'control_applied_at_inference':variant=='mask_motion','anchor_consumer_dtype':'float16_then_float32',
            'quality_consumer_dtype':'float16_then_float32','padding_compatibility':'safe_mask_first_column_or_empty',
            'fresh_same_phase_initializers':True,'external_transform':'identity'}
        write_json(root/'model.json',cfg)
        _save_torch(root/'checkpoint.pt',{'stage':VARIANTS[variant],'model_state':network.state_dict(),
            'visual_config':dict(protocol.recipe['visual_student']),'pretrain_config':motion_model_config(motion_args(protocol)),
            'modality':'separate','fusion_position':'clip','fixed_stage_epochs':{'A':4,'B':20}})
        result=registry.register(stage=VARIANTS[variant],kind='supervised_model',phase=phase,
            files=[root/'checkpoint.pt',root/'model.json',root/'history.json'],parents=parents,rows=index,
            fit_users=inputs.partition.users,config=cfg,source_files=[Path(__file__)])
        write_json(root/'artifact.json',result)
        del network
        if torch.device(device).type=='cuda':torch.cuda.empty_cache()
        return result


def predict_fusion(model,partition,*,protocol,refs,context,device='cuda',zero=None):
    from .imu_rf_teacher import _write_npz
    if zero not in {None,'skeleton','imu'}:raise ValueError('unknown zero control')
    registry=ArtifactRegistry(protocol,context=context);mr=registry.verify(model);phase='select' if partition in {'train12','development2'} else 'refit'
    if mr.phase!=phase or mr.stage not in VARIANTS.values() or zero and mr.stage!='A7':raise ValueError('fusion inference phase/model mismatch')
    data=load_fusion_data(protocol,partition,context,refs);index=data['index'];ops=source_fusion_math(protocol,context)
    network=initialize_fusion(protocol,phase,refs,context,device='cpu',load_parents=False)
    path=next(Path(x) for x in mr.files if Path(x).name=='checkpoint.pt');registry.verify_file(mr,path)
    state=torch.load(path,map_location='cpu',weights_only=True);network.load_state_dict(state['model_state']);network.eval().to(device)
    variant=mr.config['variant'];dataset=dataset_for(data,variant='mask_motion' if variant=='mask_motion' else 'full')
    scores=np.tile(np.log(np.maximum(mr.config['prior'],1e-12)),(len(dataset),1)).astype(np.float32);offset=0
    with torch.inference_mode():
        for batch in ops['module'].loader(dataset,fusion_args(protocol),False):
            n=len(batch['sample_id']);active=np.flatnonzero(data['observed_valid'][offset:offset+n])
            if len(active):
                batch=ops['to_device'](batch,torch.device(device));take=torch.as_tensor(active,device=device)
                fields={k:v.index_select(0,take) if torch.is_tensor(v) else [v[i] for i in active] for k,v in batch.items()}
                if zero:
                    for k in MOTION_FIELDS:
                        if k.startswith(zero+'_'):fields[k]=torch.zeros_like(fields[k])
                with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,enabled=torch.device(device).type=='cuda'):
                    out=ops['model_forward'](network,fields)
                scores[offset+active]=out['logits'].float().cpu().numpy()
            offset+=n
    probabilities=softmax(scores,axis=1);probabilities[~data['observed_valid']]=mr.config['prior']
    stage={'skeleton':'A7-zero-S','imu':'A7-zero-I'}[zero] if zero else mr.stage
    path=protocol.run_root/stage/phase/(partition+'_targets.npz')
    digest=_write_npz(path,dict(sample_ids=np.asarray(index.sample_ids),class_ids=np.arange(40),logits=scores,probabilities=probabilities,valid=data['observed_valid']))
    inputs=load_stage_inputs(protocol,partition,'predict',context=context)
    result=registry.register(stage=stage,kind='predictions',phase='predict',files=[path],file_digests={str(path.resolve()):digest},
        parents=[model,*data['parents'],inputs.parents['manifest']],rows=index,config={'partition':partition,'fit_phase':phase,
            'variant':variant,'zero_modality':zero,'role':'source_main' if stage=='A7' else 'diagnostic_control',
            'task8_revision':'ir_supervision_v2','invalid_policy':'same_phase_complete_fit_prior'},source_files=[Path(__file__)])
    write_json(path.with_suffix('.artifact.json'),result);del network,state
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    return Prediction(index,scores,probabilities,data['observed_valid'],result)


def run_pipeline(protocol,*,device='cuda'):
    from .simple_fusion import read_labels,metrics
    context=VerificationContext(protocol);refs=fusion_inputs(protocol,context);models={};outputs={};results={};started=time.perf_counter()
    for phase in ['select','refit']:
        models[phase]={}
        for variant in VARIANTS:models[phase][variant]=train_fusion(variant,phase,protocol=protocol,refs=refs,context=context,device=device)
    for pop in protocol.partitions:
        phase='select' if pop in {'train12','development2'} else 'refit';outputs[pop]={};results[pop]={}
        jobs=[(stage,models[phase][v],None) for v,stage in VARIANTS.items()]+[('A7-zero-S',models[phase]['full'],'skeleton'),('A7-zero-I',models[phase]['full'],'imu')]
        for stage,model,zero in jobs:
            prediction=predict_fusion(model,pop,protocol=protocol,refs=refs,context=context,device=device,zero=zero)
            outputs[pop][stage]=prediction.artifact;result={'rows':len(prediction.valid),'valid':int(prediction.valid.sum()),'correct':None}
            if pop!='final4':
                y=read_labels(protocol.supervised_labels[pop],prediction.index);result.update(metrics(prediction.probabilities,y,prediction.index),correct=int((prediction.logits.argmax(1)==y).sum()))
            results[pop][stage]=result
            print(json.dumps({'event':'fusion_population_export','partition':pop,'stage':stage,**result}),flush=True)
    write_json(protocol.run_root/'A7/development_metrics.json',results)
    body={'protocol_sha256':protocol.identity(),'complete':True,'models':models,'outputs':outputs,'prerequisites':refs['proofs'],
        'task8_revision':'ir_supervision_v2','fixed_stage_epochs':{'A':4,'B':20},'final_labels_read':False,
        'task11_started':False,'independent_review':'pending','seconds':round(time.perf_counter()-started,2)}
    immutable_bytes(protocol.run_root/'protocol/task10_artifacts_ready.json',canonical_bytes(body)+b'\n')
    print(json.dumps({'event':'source_fusion_pipeline_complete','seconds':body['seconds'],'final_accuracy':None}),flush=True)
    return body
