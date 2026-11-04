"""Task12 fixed final-input pseudo adaptation, with no truth or epoch selection."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import softmax

from .artifact_record import ArtifactRegistry, VerificationContext, FORBIDDEN_LABEL_FIELDS
from .no_vote_types import AdaptationInputs, RowIndex, Prediction, canonical_bytes, canonical_hash, write_json
from .no_vote_manifest import read_public_rows, row_index, load_stage_inputs
from .no_vote_cache_import import immutable_bytes
from .p12_reconstruction import ref
from .source_base_adaptation import source_adaptation_ops, initialize_source_base, train_source_with_resume, _pins
from .visual_student_v3 import prior, _save_torch
from .visual_student import phase_lock
from .mobind_fusion import fusion_inputs, load_fusion_data, dataset_for, source_fusion_math, fusion_args

CONTRACT = 'task12_final_target_v1'
ENDPOINTS = {
    12: dict(epochs=12, scope='heads', seed=20260812, fusion_learning_rate=5e-5,
             visual_head_learning_rate=1e-5, minimum_learning_rate=2e-6),
    40: dict(epochs=40, scope='heads_motion_encoder', seed=20260826, fusion_learning_rate=1e-4,
             visual_head_learning_rate=5e-5, minimum_learning_rate=5e-6),
}
COMMON = dict(batch_size=64, workers=0, weight_decay=.02, temperature=1., confidence_power=0.)


def target_args(protocol, endpoint):
    if type(endpoint) is not int or endpoint not in ENDPOINTS:
        raise ValueError('target endpoint must be independent 12 or 40')
    expected = {**ENDPOINTS[endpoint], **COMMON}
    key = 'primary40' if endpoint == 40 else 'diagnostic12'
    configured = protocol.recipe['adaptation']['endpoints'][key]
    if dict(configured) != expected:
        raise ValueError('target recipe differs from approved fixed source endpoint')
    return SimpleNamespace(**{k:v for k,v in expected.items() if k != 'scope'},
                           adaptation_scope=expected['scope'], target='structured',
                           emission_warmup_epochs=0, max_train_batches=0)


class RejectLabelsDataset(torch.utils.data.Dataset):
    """Reject upstream label keys before the original pseudo wrapper can delete them."""
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, index):
        item = self.base[index]
        if any(str(key).lower() in FORBIDDEN_LABEL_FIELDS for key in item):
            raise ValueError('label field in target input Dataset')
        return item


def train_fixed_budget(network, loader, args, ops, resume, identity, *, expected_rows, device='cuda'):
    history = train_source_with_resume(network, loader, args, ops, resume, identity, device=device)
    if ([x['epoch'] for x in history] != list(range(1, args.epochs + 1))
            or any(x['train_samples'] != expected_rows or x['ground_truth_fields_seen'] != 0 for x in history)):
        raise ValueError('fixed adaptation budget or label-free population mismatch')
    return history


def _completion(protocol, context):
    path = protocol.run_root/'protocol/task11_complete.json'
    body = json.loads(path.read_text(encoding='utf-8'))
    if (body.get('protocol_sha256') != protocol.identity() or body.get('complete') is not True
            or body.get('task11_complete') is not True or body.get('a8_generated') is not True
            or body.get('independent_review', {}).get('status') != 'GO'):
        raise ValueError('Task12 requires complete independently reviewed Task11')
    review = body['independent_review']
    context.hash_file(Path(review['path']), review['sha256'])
    reviewed = json.loads(Path(review['path']).read_text(encoding='utf-8'))
    if reviewed.get('verdict') != 'GO' or reviewed.get('unresolved_p1_p2'):
        raise ValueError('Task11 independent review is not GO')
    return body, context.hash_file(path)


def target_inputs(protocol, context):
    refs = fusion_inputs(protocol, context)
    body, _ = _completion(protocol, context)
    base = ref(json.loads((protocol.run_root/'A7/refit/artifact.json').read_text(encoding='utf-8')))
    motion = refs['motion']['final4']
    from .visual_student_v3 import pixel_refs
    return AdaptationInputs(row_index(read_public_rows(protocol.public_manifests['final4'])),
        base, ref(body['outputs']['final4']), pixel_refs(protocol)['final4'],
        refs['sequences']['final4'], motion['windows'], motion['normalization'])


def _validate_inputs(protocol, inputs, context):
    expected = target_inputs(protocol, context)
    if inputs != expected:
        raise ValueError('Task12 requires exact completed final4 A8 and own refit A7/input references')
    r = ArtifactRegistry(protocol, context=context)
    base = r.verify(inputs.base, 'A7', 'refit')
    if base.config.get('variant') != 'full': raise ValueError('target initialization must be full A7 refit')
    target = r.verify(inputs.targets, 'A8', 'predict', inputs.rows)
    if target.config.get('partition') != 'final4': raise ValueError('target population must be final4')
    arrays = context.load_arrays(r.verify_file(target, next(Path(x) for x in target.files if Path(x).suffix == '.npz')))
    mask = arrays['target_mask']
    n = len(inputs.rows.sample_ids)
    if (tuple(arrays['sample_ids'].tolist()) != inputs.rows.sample_ids
            or not np.array_equal(arrays['class_ids'], np.arange(40))
            or mask.dtype != np.bool_ or mask.shape != (n,) or not np.array_equal(mask, arrays['valid'])
            or not np.array_equal(arrays['probabilities'], arrays['structured_distillation_probability'])):
        raise ValueError('A8 exact target IDs/mask/probability mismatch')
    cfg = protocol.recipe['adaptation']
    if n != cfg['expected_rows'] or int(mask.sum()) != cfg['expected_loss_rows'] or int((~mask).sum()) != cfg['expected_prior_rows']:
        raise ValueError('final target fixed population/mask mismatch')
    pr, prior_value = prior(protocol, 'refit', context)
    sharp = np.full((int(mask.sum()), 40), .0005, np.float32)
    sharp[np.arange(len(sharp)), arrays['probabilities'][mask].argmax(1)] = .9805
    np.testing.assert_array_equal(arrays['probabilities'][mask], sharp)
    np.testing.assert_array_equal(arrays['probabilities'][~mask], np.tile(prior_value, ((~mask).sum(), 1)))
    return arrays, pr, prior_value


def _model_parents(inputs, prior_ref):
    return [inputs.base, inputs.targets, inputs.pixels, inputs.sequence, inputs.motion, inputs.normalization, prior_ref]


def allows_target_stage(registry, record):
    if record.stage not in {'A9-12', 'A9-40'} or record.kind != 'adapted_model' or record.phase != 'adapt':
        return False
    cfg = record.config
    if cfg.get('target_contract') != CONTRACT: return False
    p = registry.protocol; c = registry.context or VerificationContext(p)
    endpoint = 40 if record.stage == 'A9-40' else 12
    args = target_args(p, endpoint)
    inputs = target_inputs(p, c)
    body, complete_sha = _completion(p, c)
    pr = ref(json.loads((p.run_root/'A1/refit/prior_artifact.json').read_text(encoding='utf-8')))
    return (cfg.get('endpoint') == endpoint and cfg.get('partition') == 'final4'
        and cfg.get('fit_phase') == 'refit' and cfg.get('budget_epochs') == endpoint
        and canonical_bytes(cfg.get('adaptation_recipe')) == canonical_bytes(vars(args))
        and cfg.get('task11_complete_sha256') == complete_sha and cfg.get('source_files') == _pins(p,c)
        and record.rows == inputs.rows and set(record.adaptation_users) == set(p.partitions['final4'].users)
        and tuple(record.parents) == tuple(_model_parents(inputs, pr))
        and cfg.get('base_model_sha256') == inputs.base.sha256 and cfg.get('target_sha256') == inputs.targets.sha256
        and cfg.get('loss_rows') == p.recipe['adaptation']['expected_loss_rows']
        and cfg.get('independent_initialization') is True and cfg.get('output_mode') == 'raw')


def adapt_target(inputs: AdaptationInputs, endpoint: int, output: Path, *, protocol, context=None, device='cuda'):
    context = context or VerificationContext(protocol)
    args = target_args(protocol, endpoint); stage = f'A9-{endpoint}'
    output = Path(output).resolve()
    if output != (protocol.run_root/stage/'adapt').resolve(): raise ValueError('endpoint output must be isolated within its own run stage')
    arrays, pr, prior_value = _validate_inputs(protocol, inputs, context)
    _, complete_sha = _completion(protocol, context)
    registry = ArtifactRegistry(protocol, context=context); parents = _model_parents(inputs, pr)
    identity = canonical_hash(dict(stage=stage, parents=parents, rows=inputs.rows,
        recipe=vars(args), producer=context.hash_file(Path(__file__)), task11_complete_sha256=complete_sha))
    with phase_lock(output):
        immutable_bytes(output/'identity.json', canonical_bytes({'identity':identity}) + b'\n')
        if (output/'artifact.json').exists():
            result = ref(json.loads((output/'artifact.json').read_text(encoding='utf-8')))
            registry.verify(result, stage, 'adapt', inputs.rows)
            return result
        refs = fusion_inputs(protocol, context)
        data = load_fusion_data(protocol, 'final4', context, refs)
        mask = arrays['target_mask']
        if data['index'] != inputs.rows or not np.array_equal(mask, data['observed_valid']):
            raise ValueError('target and cached student input availability mismatch')
        chosen = RowIndex(tuple(s for s,v in zip(inputs.rows.sample_ids,mask) if v),
            tuple(u for u,v in zip(inputs.rows.user_ids,mask) if v), inputs.rows.class_ids)
        network = initialize_source_base(protocol, 'refit', inputs.base, refs, context, seed=args.seed, device=device)
        fops = source_fusion_math(protocol, context); ops = source_adaptation_ops(protocol, context)
        base_data = RejectLabelsDataset(dataset_for(data, rows=chosen, temporal_indices=fops['temporal_indices']))
        dataset = ops['LabelFreePseudoDataset'](base_data,
            dict(zip(chosen.sample_ids, arrays['structured_distillation_probability'][mask])),
            dict(zip(chosen.sample_ids, arrays['structured_confidence'][mask].tolist())))
        loader = ops['make_loader'](dataset, args.batch_size, args.workers, shuffle=True)
        history = train_fixed_budget(network, loader, args, ops, output/'resume.pt', identity,
                                    expected_rows=len(chosen.sample_ids), device=device)
        write_json(output/'history.json', history)
        _save_torch(output/'checkpoint.pt', dict(stage=stage, endpoint=endpoint, model_state=network.state_dict(), base_model_sha256=inputs.base.sha256))
        cfg = dict(target_contract=CONTRACT, endpoint=endpoint, partition='final4', fit_phase='refit',
            budget_epochs=endpoint, adaptation_recipe=vars(args), task11_complete_sha256=complete_sha,
            source_files=_pins(protocol,context), loss_rows=len(chosen.sample_ids), excluded_prior_rows=int((~mask).sum()),
            loss_ids_sha256=canonical_hash(chosen), base_model_sha256=inputs.base.sha256, target_sha256=inputs.targets.sha256,
            prior=prior_value.tolist(), independent_initialization=True, output_mode='raw',
            no_ground_truth_batch=True, no_agreement_selection=True, evaluation_mode='transductive_final_inputs',
            teacher_or_labels_needed_at_inference=False)
        write_json(output/'model.json', cfg)
        result = registry.register(stage=stage, kind='adapted_model', phase='adapt',
            files=[output/'checkpoint.pt',output/'history.json',output/'model.json'], parents=parents,
            rows=inputs.rows, adaptation_users=protocol.partitions['final4'].users, config=cfg, source_files=[Path(__file__)])
        write_json(output/'artifact.json', result)
        del network
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return result


def require_prediction_binding(prediction_record, model_ref, model_record):
    cfg=prediction_record.config; expected=model_record.config
    if (prediction_record.stage!=model_record.stage or model_ref not in prediction_record.parents
            or cfg.get('partition')!='final4' or cfg.get('fit_phase')!='refit' or cfg.get('output_mode')!='raw'
            or cfg.get('endpoint')!=expected['endpoint'] or cfg.get('target_sha256')!=expected['target_sha256']):
        raise ValueError('target prediction model/endpoint/target binding mismatch')


def predict_target(protocol, model, inputs, context, *, device='cuda'):
    from .p12_source_adapter import _read_prediction
    from .imu_rf_teacher import _write_npz
    registry = ArtifactRegistry(protocol, context=context)
    if inputs!=target_inputs(protocol,context):raise ValueError('prediction exact target input binding mismatch')
    record = registry.verify(model)
    if not allows_target_stage(registry, record): raise ValueError('unbound target model')
    root = protocol.run_root/record.stage/'adapt'; path = root/'final4_targets.npz'; alias = path.with_suffix('.artifact.json')
    if alias.exists():
        reference = ref(json.loads(alias.read_text(encoding='utf-8')))
        prediction_record, a = _read_prediction(protocol, reference, record.stage, 'final4', context)
        require_prediction_binding(prediction_record,model,record)
        return Prediction(inputs.rows,a['logits'],a['probabilities'],a['valid'],reference)
    refs = fusion_inputs(protocol,context); data = load_fusion_data(protocol,'final4',context,refs)
    network = initialize_source_base(protocol,'refit',None,refs,context,seed=record.config['adaptation_recipe']['seed'],device='cpu',load_base=False)
    file = next(Path(x) for x in record.files if Path(x).name == 'checkpoint.pt')
    registry.verify_file(record,file)
    network.load_state_dict(torch.load(file,map_location='cpu',weights_only=True)['model_state']);network.eval().to(device)
    ops = source_fusion_math(protocol,context); dataset = RejectLabelsDataset(dataset_for(data))
    valid = data['observed_valid']; scores = np.tile(np.log(np.maximum(record.config['prior'],1e-12)),(len(valid),1)).astype(np.float32)
    offset = 0
    with torch.inference_mode():
        for batch in ops['module'].loader(dataset,fusion_args(protocol),False):
            n = len(batch['sample_id']); active = np.flatnonzero(valid[offset:offset+n])
            if len(active):
                batch = ops['to_device'](batch,torch.device(device)); take = torch.as_tensor(active,device=device)
                fields = {k:v.index_select(0,take) if torch.is_tensor(v) else [v[i] for i in active] for k,v in batch.items()}
                with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,enabled=torch.device(device).type=='cuda'):
                    values = ops['model_forward'](network,fields)
                scores[offset+active] = values['logits'].float().cpu().numpy()
            offset += n
    if offset != len(valid): raise ValueError('target inference row coverage mismatch')
    probability = softmax(scores,axis=1);probability[~valid] = record.config['prior']
    digest = _write_npz(path,dict(sample_ids=np.asarray(inputs.rows.sample_ids),class_ids=np.arange(40),logits=scores,probabilities=probability,valid=valid))
    public = load_stage_inputs(protocol,'final4','predict',context=context)
    reference = registry.register(stage=record.stage,kind='predictions',phase='predict',files=[path],file_digests={str(path):digest},
        parents=[model,*data['parents'],public.parents['manifest']],rows=inputs.rows,
        config=dict(partition='final4',fit_phase='refit',endpoint=record.config['endpoint'],output_mode='raw',target_sha256=inputs.targets.sha256),source_files=[Path(__file__)])
    write_json(alias,reference)
    del network
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return Prediction(inputs.rows,scores,probability,valid,reference)


def teacher_student_agreement(target,before,after,mask,*,phase,endpoint,base_model,adapted_model):
    if phase != 'refit':raise ValueError('agreement requires same refit phase')
    if endpoint not in ENDPOINTS:raise ValueError('agreement endpoint mismatch')
    if target.index != before.index or target.index != after.index:raise ValueError('agreement exact ID/class order mismatch')
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (len(target.index.sample_ids),) or not np.array_equal(mask,target.valid):
        raise ValueError('agreement target mask mismatch')
    def counts(prediction):
        same = prediction.probabilities.argmax(1) == target.probabilities.argmax(1)
        return {name:dict(same=int(same[part].sum()),total=int(part.sum()),disagreement=int((~same[part]).sum()))
                for name,part in [('all',np.ones(len(mask),bool)),('adapted',mask),('excluded_prior',~mask)]}
    return dict(endpoint=endpoint,fit_phase=phase,sample_ids_sha256=canonical_hash(target.index),
        target_mask_sha256=canonical_hash(mask.tolist()),target=target.artifact,base_model=base_model,adapted_model=adapted_model,
        before_prediction=before.artifact,after_prediction=after.artifact,before_adaptation=counts(before),
        after_adaptation=counts(after),is_selection_gate=False,output_mode='raw_eval_no_augmentation')


def run_target_adaptation(protocol, *, device='cuda'):
    from .p12_source_adapter import _read_prediction
    context = VerificationContext(protocol); inputs = target_inputs(protocol,context)
    arrays, _, _ = _validate_inputs(protocol,inputs,context)
    ready = json.loads((protocol.run_root/'protocol/task10_artifacts_ready.json').read_text(encoding='utf-8'))
    before_ref = ref(ready['outputs']['final4']['A7'])
    br,ba = _read_prediction(protocol,before_ref,'A7','final4',context)
    if inputs.base not in br.parents:raise ValueError('before prediction not from exact base A7')
    before = Prediction(inputs.rows,ba['logits'],ba['probabilities'],ba['valid'],before_ref)
    target = Prediction(inputs.rows,arrays['logits'],arrays['probabilities'],arrays['valid'],inputs.targets)
    result = {}
    for endpoint in [40,12]:
        root = protocol.run_root/f'A9-{endpoint}'/'adapt'
        model = adapt_target(inputs,endpoint,root,protocol=protocol,context=context,device=device)
        prediction = predict_target(protocol,model,inputs,context,device=device)
        agreement = teacher_student_agreement(target,before,prediction,arrays['target_mask'],phase='refit',endpoint=endpoint,base_model=inputs.base,adapted_model=model)
        immutable_bytes(root/'teacher_student_agreement.json',canonical_bytes(agreement)+b'\n')
        result[str(endpoint)] = dict(model=model,prediction=prediction.artifact,agreement=agreement)
        print(json.dumps(dict(event='target_endpoint_complete',endpoint=endpoint,rows=len(prediction.valid),valid=int(prediction.valid.sum()),agreement=agreement['after_adaptation'])),flush=True)
    body = dict(protocol_sha256=protocol.identity(),task12_generation_complete=True,task12_complete=False,
        independent_review='pending',task13_started=False,
        target_contract=CONTRACT,base=inputs.base,targets=inputs.targets,endpoints=result,final_labels_read=False)
    immutable_bytes(protocol.run_root/'protocol/task12_artifacts_ready.json',canonical_bytes(body)+b'\n')
    return body
