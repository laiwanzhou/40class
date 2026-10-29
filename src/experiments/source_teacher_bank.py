"""Task11 evidence/preflight and pure P310 rule; not a completed teacher bank."""
import ast
import json
from pathlib import Path

import numpy as np

from .no_vote_types import canonical_bytes
from .no_vote_cache_import import immutable_bytes


def precedence_targets(old_group,sequence,new_group,target_mask,prior):
    old_group=np.asarray(old_group);sequence=np.asarray(sequence);new_group=np.asarray(new_group)
    mask=np.asarray(target_mask);prior=np.asarray(prior,np.float32)
    if (old_group.ndim!=1 or any(x.shape!=old_group.shape or not np.issubdtype(x.dtype,np.integer)
            or np.any((x<0)|(x>=40)) for x in [old_group,sequence,new_group])
            or mask.shape!=old_group.shape or mask.dtype!=np.bool_):raise ValueError('P310 prediction/mask schema mismatch')
    if prior.shape!=(40,) or not np.isfinite(prior).all() or np.any(prior<0) or not np.isclose(prior.sum(),1):
        raise ValueError('P310 complete-fit prior required')
    route=new_group!=old_group;prediction=sequence.copy();prediction[route]=new_group[route]
    prob=np.full((len(prediction),40),.0005,np.float32);prob[np.arange(len(prob)),prediction]=.9805
    prob[~mask]=prior;confidence=np.full(len(prob),.9805,np.float32);confidence[~mask]=prior.max()
    return {'precedence_route':route,'precedence_prediction':prediction,'target_mask':mask.copy(),
        'emission_probability':prob.copy(),'structured_distillation_probability':prob,
        'structured_confidence':confidence,'emission_prediction':prob.argmax(1),
        'structured_distillation_prediction':prob.argmax(1)}


def validate_producer_roster(slots,producers):
    if any(x['status']=='unresolved' for x in slots):raise ValueError('unresolved teacher lineage blocks bank')
    names=[x['name'] for x in slots if x['status']=='retained']
    if len(names)!=len(set(names)) or set(producers)!=set(names):raise ValueError('exact retained producer coverage required')
    for name in names:
        value=producers[name]
        if value.get('fresh') is not True or not set(value.get('roots',()))<={'videomae_large'}:
            raise ValueError('historical prediction or removed large root cannot fill source bank')
    # This is only a roster preflight. Actual artifact phase/IDs and source
    # parameter evidence remain mandatory at the eventual bank producer.


def pinned_source(protocol,relative,context):
    key=(protocol.identity(),'task11_source_manifest')
    if key not in context.public_rows:
        path=protocol.source_root.parent/'source_manifest.json'
        context.hash_file(path,protocol.recipe['asset_bindings']['source_manifest_sha256'])
        context.public_rows[key]={x['path']:x['sha256'] for x in json.loads(path.read_text(encoding='utf-8'))['files']}
    path=protocol.source_root/relative;digest=context.hash_file(path,context.public_rows[key][relative])
    text=path.read_text(encoding='utf-8')
    return text,{'path':relative,'sha256':digest}


def a18_lineage(protocol,context):
    data,data_pin=pinned_source(protocol,'aligned_multimodal/a18_full_teacher_data.py',context)
    base,base_pin=pinned_source(protocol,'aligned_multimodal/p100a_global_teacher_data.py',context)
    tree=ast.parse(data);base_tree=ast.parse(base)
    imports={alias.name for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)
        and node.module=='p100a_global_teacher_data' for alias in node.names}
    if not {'VMAE_PATH','IV2_PATH'}<=imports:raise ValueError('A18 source feature imports changed')
    roots=[];expressions=[]
    for name,marker,root in [('VMAE_PATH','p90_videomaev2_distilled_teacher_v1','videomaev2'),
                              ('IV2_PATH','p90_internvideo2_l_k400_teacher_v1','internvideo2')]:
        assignments=[x for x in base_tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets)]
        calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and any(isinstance(a,ast.Name) and a.id==name for a in x.args)]
        if len(assignments)!=1 or not calls or not any(isinstance(x,ast.Constant) and isinstance(x.value,str) and marker in x.value for x in ast.walk(assignments[0])):
            raise ValueError('A18 consumed large-root lineage is not the pinned expected chain')
        roots.append(root);expressions.append({'path':base_pin['path'],'line':assignments[0].lineno,
            'expression':ast.unparse(assignments[0]),'consumer_path':data_pin['path'],'consumer_line':calls[0].lineno})
    return {'name':'a18_best_session','status':'excluded','roots':roots,'removed_large_roots':roots,
        'actual_consumption_confirmed':True,'evidence':[data_pin,base_pin],'expressions':expressions,
        'reason':'Actual A18 VS consumes VideoMAEv2/InternVideo2; never substitute A5 or drop only the declared modality label.'}


def task11_preflight(protocol,context):
    """Versioned evidence only: preserve Task1's identity, never silently unseal A8."""
    root=protocol.run_root;bindings=protocol.recipe['source_ops_bindings'];roster_path=Path(bindings['roster_path'])
    old_hash=context.hash_file(roster_path,bindings['roster_sha256']);roster=json.loads(roster_path.read_text(encoding='utf-8'))
    a18=a18_lineage(protocol,context);slots=[]
    for slot in roster['slots']:
        slots.append({**slot,**a18} if slot['name']=='a18_best_session' else dict(slot))
    evidence=[]
    for name in ['aligned_multimodal/build_complete_p11_oof.py','aligned_multimodal/train.py',
                 'thermal_baseline/run_imagenet_oof_folds.ps1','thermal_baseline/configs/resnet18_tsm_imagenet_12f.json',
                 'thermal_baseline/train_imagenet_oof.py','aligned_multimodal/p165_deployable_group_teacher.py',
                 'aligned_multimodal/p89_build_final_test_submissions.py','aligned_multimodal/run_p87s_final_pipeline.py',
                 'aligned_multimodal/build_p87s_test_structured_targets.py','aligned_multimodal/p310_union_repeat_precedence_teacher.py']:
        _,pin=pinned_source(protocol,name,context);evidence.append(pin)
    blockers=[
        {'code':'p12_sd_selected_config_and_receipt','finding':'P39-M3',
         'need':'Actual p11_fp16_oof selected config_used/invocation + checkpoint conversion ancestry + public initializer receipt; candidate JSON alone is insufficient.'},
        {'code':'p12_thermal_selected_fp16_binding','finding':'P39-M3',
         'need':'Bind known resnet18_tsm_imagenet_12f invocation/p11_thermal_imagenet to the selected p11_thermal_imagenet_fp16 checkpoints, including initializer receipt.'},
        {'code':'thermal_public_input_sidecar','finding':'P39-M3',
         'need':'Exact sample_id,user_id,thermal_path,thermal_available for all four public populations; absence means unknown, never all-false.'},
        {'code':'safe_base_intermediate_adaptation_producer','finding':'P39-M1',
         'need':'P85 KNN -> original intermediate structured target/decoder -> independent 12-epoch adapted source-base -> P87S raw -> P89 safe probability/decision; no A7 or final A9-12 substitution.'},
        {'code':'retained_producers_and_routing_closure','finding':'P39-M1',
         'need':'Fresh same-phase retained producer graph, P89 joint grouping/threshold provenance, actual public asset bindings; old runs/output references are not producers.'}]
    unresolved=[x['name'] for x in slots if x['status']=='unresolved']
    if unresolved:blockers.append({'code':'unresolved_teacher_lineage','slots':unresolved,
        'need':'Close actual roots and producers in a new evidence version before bank generation.'})
    parent={'path':str(roster_path.resolve()),'sha256':old_hash}
    body={'schema_version':3,'protocol_sha256':protocol.identity(),'revision':'task11_source_closure_v1',
        'scope':'preflight_evidence_only_not_formal_bank','parent_roster':parent,'source_evidence':evidence,
        'retained_large_roots':['videomae_large'],'slots':slots,'corrections':[a18],
        'source_base_chain':['P85_KNN','intermediate_structured_targets','independent_source_base_adapt12',
            'P87S_student_raw','P89_base_probability_and_safe_decision','source_bank'],
        'formal_generation_ready':False,'blocking_requirements':blockers,'original_roster_modified':False,
        'training_executed':False,'final_labels_read':False,'bank_generated':False,'a8_generated':False,
        'known_thermal_runner_output':'thermal_baseline/runs/p11_thermal_imagenet',
        'required_thermal_historical_output':'thermal_baseline/runs/p11_thermal_imagenet_fp16',
        'known_public_initializer':'ResNet18_Weights.IMAGENET1K_V1',
        'status':'BLOCKED_FORMAL_SOURCE_BANK'}
    path=root/'protocol/task11_source_closure_v1.json';immutable_bytes(path,canonical_bytes(body)+b'\n')
    return body


def require_task11_formal_ready(body):
    # v1 is deliberately an evidence-only closure. There is not yet a supported
    # formal-ready schema backed by registered P12/safe-base producers.
    if (body.get('schema_version')!=3 or body.get('revision')!='task11_source_closure_v1'
            or body.get('scope')!='preflight_evidence_only_not_formal_bank'):
        raise ValueError('unsupported Task11 readiness proof; registered source producers are required')
    if (body.get('scope')=='preflight_evidence_only_not_formal_bank' or body.get('formal_generation_ready') is not True
            or body.get('blocking_requirements') or any(x['status']=='unresolved' for x in body.get('slots',()))):
        codes=[x['code'] for x in body.get('blocking_requirements',())]
        raise ValueError('Task11 formal source bank blocked: '+', '.join(codes))
