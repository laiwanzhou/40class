"""Fixed target recipes, source parity, no-label agreement and endpoint isolation."""
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml


def module():
    assert importlib.util.find_spec('src.experiments.target_adaptation'), 'Task12 producer missing'
    return importlib.import_module('src.experiments.target_adaptation')


def recipe():
    return yaml.safe_load((Path(__file__).parents[1]/'configs/experiments/teammate_single_teacher_fixed_split_v3.yaml').read_text(encoding='utf-8'))['recipe']['adaptation']


def test_final40_recipe():
    args=module().target_args(SimpleNamespace(recipe={'adaptation':recipe()}),40)
    assert (args.epochs,args.adaptation_scope,args.seed)==(40,'heads_motion_encoder',20260826)
    assert (args.fusion_learning_rate,args.visual_head_learning_rate,args.minimum_learning_rate)==(1e-4,5e-5,5e-6)
    assert (args.batch_size,args.workers,args.temperature,args.confidence_power,args.emission_warmup_epochs)==(64,0,1.,0.,0)
    changed=recipe();changed['endpoints']['primary40']['epochs']=12
    with pytest.raises(ValueError,match='recipe'):module().target_args(SimpleNamespace(recipe={'adaptation':changed}),40)


def test_base12_recipe():
    args=module().target_args(SimpleNamespace(recipe={'adaptation':recipe()}),12)
    assert (args.epochs,args.adaptation_scope,args.seed)==(12,'heads',20260812)
    assert (args.fusion_learning_rate,args.visual_head_learning_rate,args.minimum_learning_rate)==(5e-5,1e-5,2e-6)
    with pytest.raises(ValueError,match='endpoint'):module().target_args(SimpleNamespace(recipe={'adaptation':recipe()}),17)


def predictions(tmp_path):
    from src.experiments.no_vote_types import RowIndex,ArtifactRef,Prediction
    index=RowIndex(('a','b','c'),('target',)*3,tuple(range(40)))
    valid=np.array([True,True,False]); refs=[ArtifactRef(tmp_path/(name+'.json'),str(i)*64) for i,name in enumerate(['target','before','after'],1)]
    result=[]
    for guesses,ref in zip([[0,1,2],[0,0,2],[0,1,2]],refs):
        prob=np.full((3,40),.0005,np.float32);prob[np.arange(3),guesses]=.9805
        result.append(Prediction(index,np.log(prob),prob,valid.copy(),ref))
    return result,valid


def test_teacher_student_agreement_label_free(tmp_path,monkeypatch):
    m=module();ps,mask=predictions(tmp_path)
    def forbid(*a,**kw):raise AssertionError('agreement must not open files or labels')
    monkeypatch.setattr(Path,'open',forbid)
    report=m.teacher_student_agreement(*ps,mask,phase='refit',endpoint=40,base_model=ps[1].artifact,adapted_model=ps[2].artifact)
    assert report['before_adaptation']['all']=={'same':2,'total':3,'disagreement':1}
    assert report['after_adaptation']['all']=={'same':3,'total':3,'disagreement':0}
    assert report['is_selection_gate'] is False
    assert 'accuracy' not in repr(report) and 'ground_truth' not in repr(report)


def test_agreement_mask_denominators(tmp_path):
    from src.experiments.no_vote_types import RowIndex,Prediction
    m=module();ps,mask=predictions(tmp_path)
    kw=dict(phase='refit',endpoint=12,base_model=ps[1].artifact,adapted_model=ps[2].artifact)
    report=m.teacher_student_agreement(*ps,mask,**kw)
    assert report['before_adaptation']['adapted']=={'same':1,'total':2,'disagreement':1}
    assert report['after_adaptation']['excluded_prior']=={'same':1,'total':1,'disagreement':0}
    with pytest.raises(ValueError,match='phase'):m.teacher_student_agreement(*ps,mask,**dict(kw,phase='select'))
    with pytest.raises(ValueError,match='mask'):m.teacher_student_agreement(*ps,~mask,**kw)
    wrong=RowIndex(('c','b','a'),('target',)*3,tuple(range(40)))
    other=Prediction(wrong,ps[2].logits,ps[2].probabilities,ps[2].valid,ps[2].artifact)
    with pytest.raises(ValueError,match='ID'):m.teacher_student_agreement(ps[0],ps[1],other,mask,**kw)
    with pytest.raises(ValueError,match='unique'):RowIndex(('a','a'),('target',)*2,tuple(range(40)))


@pytest.mark.parametrize('endpoint',[12,40])
def test_agreement_is_not_a_selection_gate_and_source_update_is_exact(no_vote_fixture,tmp_path,endpoint):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from test_no_vote_source_base_adaptation import ToyModel
    from test_no_vote_mobind_pretrain import synthetic_batch
    from src.experiments.artifact_record import VerificationContext
    from src.experiments.source_base_adaptation import source_adaptation_ops
    p=v3_protocol(no_vote_fixture);ops=source_adaptation_ops(p,VerificationContext(p));torch.set_num_threads(2)
    args=m.target_args(SimpleNamespace(recipe={'adaptation':recipe()}),endpoint)
    batch=synthetic_batch(2);batch.pop('label');batch['sample_id']=['a','b']
    batch.update(backbone_sequence=torch.randn(2,2,3,16,512),view_valid=torch.ones(2,2,16,3,dtype=torch.bool),
        view_quality=torch.ones(2,2,16,3),global_time_position=torch.zeros(2,2,16),
        pseudo_probability=torch.softmax(torch.randn(2,40),1),pseudo_confidence=torch.ones(2))
    torch.manual_seed(51);initial=ToyModel();expected=deepcopy(initial);actual=deepcopy(initial)
    ops['seed_all'](args.seed);reference=ops['train_label_free'](expected,[batch],args,torch.device('cpu'))
    ops['seed_all'](args.seed)
    history=m.train_fixed_budget(actual,[batch],args,ops,tmp_path/'resume.pt','same-base-independent-endpoint',expected_rows=2,device='cpu')
    assert len(history)==endpoint and history[-1]['student_target_agreement']<1.
    assert all(x['ground_truth_fields_seen']==0 for x in history)
    for k,v in expected.state_dict().items():torch.testing.assert_close(v,actual.state_dict()[k],atol=0,rtol=0)
    assert [{k:v for k,v in x.items() if k!='seconds'} for x in history]==[{k:v for k,v in x.items() if k!='seconds'} for x in reference]
    assert actual.motion_residual.encoder.weight.requires_grad==(endpoint==40)
    with pytest.raises(ValueError,match='identity|recipe'):
        m.train_fixed_budget(deepcopy(initial),[batch],m.target_args(SimpleNamespace(recipe={'adaptation':recipe()}),40 if endpoint==12 else 12),ops,tmp_path/'resume.pt','other-endpoint',expected_rows=2,device='cpu')


def test_target_roles_reject_unbound_adaptation(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
    p=v3_protocol(no_vote_fixture);r=ArtifactRegistry(p,context=VerificationContext(p))
    for stage in ['A9-12','A9-40']:
        record=SimpleNamespace(stage=stage,kind='adapted_model',phase='adapt',config={},parents=())
        assert m.allows_target_stage(r,record) is False


def test_label_keys_rejected_before_pseudo_wrapper_strips_them(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.source_base_adaptation import source_adaptation_ops
    from src.experiments.artifact_record import VerificationContext
    ops=source_adaptation_ops(v3_protocol(no_vote_fixture),VerificationContext(v3_protocol(no_vote_fixture)))
    class Labeled(torch.utils.data.Dataset):
        def __len__(self):return 1
        def __getitem__(self,i):return {'sample_id':'a','label':0}
    pseudo=ops['LabelFreePseudoDataset'](m.RejectLabelsDataset(Labeled()),{'a':np.full(40,.025)},{'a':1.})
    with pytest.raises(ValueError,match='label'):pseudo[0]


def test_exact_model_contract_rejects_other_endpoint_target_and_missing_inputs(monkeypatch,tmp_path):
    m=module()
    from src.experiments.no_vote_types import AdaptationInputs,ArtifactRef,RowIndex,canonical_bytes
    rows=RowIndex(('a',),('target',),tuple(range(40)))
    rr=[ArtifactRef(tmp_path/f'{i}.json',str(i)*64) for i in range(1,8)]
    inputs=AdaptationInputs(rows,*rr[:6]);p=SimpleNamespace(recipe={'adaptation':dict(recipe(),expected_loss_rows=1)},partitions={'final4':SimpleNamespace(users=('target',))})
    registry=SimpleNamespace(protocol=p,context=object())
    monkeypatch.setattr(m,'target_inputs',lambda *a:inputs)
    monkeypatch.setattr(m,'_completion',lambda *a:({},'complete-sha'))
    monkeypatch.setattr(m,'_pins',lambda *a:{'source':'sha'})
    p.run_root=tmp_path;path=tmp_path/'A1/refit/prior_artifact.json';path.parent.mkdir(parents=True)
    path.write_bytes(canonical_bytes(rr[6]))
    cfg=dict(target_contract=m.CONTRACT,endpoint=40,partition='final4',fit_phase='refit',budget_epochs=40,
        adaptation_recipe=vars(m.target_args(p,40)),task11_complete_sha256='complete-sha',source_files={'source':'sha'},
        base_model_sha256=inputs.base.sha256,target_sha256=inputs.targets.sha256,loss_rows=1,independent_initialization=True,output_mode='raw')
    record=SimpleNamespace(stage='A9-40',kind='adapted_model',phase='adapt',config=cfg,parents=tuple(rr),rows=rows,adaptation_users=('target',))
    assert m.allows_target_stage(registry,record)
    cfg['target_sha256']='wrong';assert not m.allows_target_stage(registry,record);cfg['target_sha256']=inputs.targets.sha256
    cfg['endpoint']=12;assert not m.allows_target_stage(registry,record);cfg['endpoint']=40
    record.parents=tuple(rr[:-1]);assert not m.allows_target_stage(registry,record)


def test_cached_prediction_requires_exact_model_and_endpoint_target(tmp_path):
    m=module();ps,_=predictions(tmp_path)
    model=SimpleNamespace(stage='A9-40',config={'endpoint':40,'target_sha256':ps[0].artifact.sha256})
    cached=SimpleNamespace(stage='A9-40',parents=(ps[2].artifact,),config=dict(partition='final4',endpoint=40,
        target_sha256=ps[0].artifact.sha256,fit_phase='refit',output_mode='raw'))
    m.require_prediction_binding(cached,ps[2].artifact,model)
    with pytest.raises(ValueError,match='model|binding'):m.require_prediction_binding(cached,ps[1].artifact,model)
    cached.config['target_sha256']='other-target'
    with pytest.raises(ValueError,match='target|binding'):m.require_prediction_binding(cached,ps[2].artifact,model)
