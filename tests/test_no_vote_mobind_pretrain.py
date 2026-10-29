"""Task8 source losses and exact motion/teacher seams."""
from dataclasses import replace
import importlib

import numpy as np
import pytest
import torch

from src.experiments.artifact_record import VerificationContext
from src.experiments.motion_cache import MOTION_FIELDS,window_schema,BOOLEAN_MOTION_FIELDS
from src.experiments.no_vote_types import RowIndex,ArtifactRef,Prediction,TeacherTargets
from test_no_vote_source_ops import v3_protocol,SOURCE_ROOT


def module():
    assert importlib.util.find_spec('src.experiments.mobind_pretrain'), 'Task8 MoBind pretrain is missing'
    return importlib.import_module('src.experiments.mobind_pretrain')


def motion_protocol(fixture):
    from src.experiments.no_vote_source_ops import compile_source_ops,runtime_recipe_values
    p=v3_protocol(fixture);recipe=runtime_recipe_values(compile_source_ops(SOURCE_ROOT))['motion_pretrain']
    return replace(p,recipe={**p.recipe,'motion_pretrain':recipe})


def synthetic_batch(n=3):
    rng=np.random.default_rng(17);batch={}
    for k,(dtype,shape) in window_schema(n).items():
        value=np.ones(shape,bool) if k in BOOLEAN_MOTION_FIELDS else rng.normal(0,.05,shape).astype(np.float32)
        if k=='skeleton_frame_quality':value[:]=.9
        if k=='imu_global_mask':value[:]=1
        batch[k]=torch.from_numpy(value)
    batch.update(label=torch.arange(n)%40,teacher_logits=torch.from_numpy(rng.normal(size=(n,40)).astype(np.float32)),
        teacher_features=torch.from_numpy(rng.normal(size=(n,2,1024)).astype(np.float32)),
        imu_teacher_logits=torch.from_numpy(rng.normal(size=(n,40)).astype(np.float32)),imu_teacher_valid=torch.ones(n,dtype=torch.bool))
    return batch


@pytest.mark.parametrize('missing',['none','imu','skeleton'])
def test_source_motion_losses_parity(no_vote_fixture,missing):
    m=module();p=motion_protocol(no_vote_fixture);context=VerificationContext(p);ops=m.source_motion_math(p,context)
    args=m.motion_args(p);batch=synthetic_batch()
    if missing=='imu':
        for k in MOTION_FIELDS:
            if k.startswith('imu_'):batch[k][0]=0
        batch['imu_teacher_valid'][0]=False
    if missing=='skeleton':
        for k in MOTION_FIELDS:
            if k.startswith('skeleton_'):batch[k][0]=0
    torch.set_num_threads(2)
    model=m.initialize_motion(p,device='cpu',context=context);weights=ops['class_weights'](np.arange(3),.35,torch.device('cpu'))
    torch.manual_seed(121);out=model({k:batch[k] for k in MOTION_FIELDS},mask_ratio=.35)
    actual=m.source_motion_losses(out,batch,weights,args,ops)
    model.zero_grad(set_to_none=True);actual['loss'].backward()
    grads={k:v.grad.clone() for k,v in model.named_parameters() if v.grad is not None}
    torch.manual_seed(121);out=model({k:batch[k] for k in MOTION_FIELDS},mask_ratio=.35)
    expected=ops['losses'](out,batch,weights,args)
    model.zero_grad(set_to_none=True);expected['loss'].backward()
    assert actual.keys()==expected.keys()
    for k in actual:torch.testing.assert_close(actual[k],expected[k],atol=0,rtol=0)
    for k,v in model.named_parameters():
        if k in grads:torch.testing.assert_close(grads[k],v.grad,atol=0,rtol=0)
    assert float(actual['skeleton_ce'])>0 and float(actual['imu_ce'])>0 and float(actual['distillation'])>0


def test_fixed24_fresh_source_model_config(no_vote_fixture):
    m=module();p=motion_protocol(no_vote_fixture);args=m.motion_args(p)
    assert args.epochs==24 and args.batch_size==64 and args.imu_instance_normalization is False
    c=VerificationContext(p);a=m.initialize_motion(p,device='cpu',context=c);b=m.initialize_motion(p,device='cpu',context=c)
    assert a.width==96 and a.alignment_width==64 and a.domain_classes==0
    assert not a.imu_encoder.instance_normalization
    for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)


def test_motion_dataset_temporal_and_teacher_mapping(no_vote_fixture):
    m=module();assert hasattr(m,'NoVoteMotionDataset'), 'phase-bound motion Dataset is missing'
    p=motion_protocol(no_vote_fixture);ops=m.source_motion_math(p,VerificationContext(p))
    batch=synthetic_batch();arrays={k:v.numpy() for k,v in batch.items() if k in MOTION_FIELDS}
    index=RowIndex(('a','b','c'),('fit1',)*3,tuple(range(40)));ref=ArtifactRef(__import__('pathlib').Path('fixture.json'),'a'*64)
    logits=np.arange(120,dtype=np.float32).reshape(3,40);features=np.arange(3*2*3*1024,dtype=np.float32).reshape(3,2,3,1024).astype(np.float16)
    from scipy.special import softmax
    visual=TeacherTargets(Prediction(index,logits,softmax(logits,axis=1),np.ones(3,bool),ref),features)
    rf=TeacherTargets(Prediction(index,logits,softmax(logits,axis=1),np.array([True,False,True]),ref),None)
    subset=RowIndex(('c','b'),('fit1',)*2,tuple(range(40)))
    labels={'c':39,'b':5};dataset=m.NoVoteMotionDataset(arrays,index,rows=subset,labels=labels,visual_teacher=visual,rf_teacher=rf,temporal_indices=lambda n:np.arange(n)[::-1])
    item=dataset[0]
    np.testing.assert_array_equal(item['teacher_logits'].numpy(),logits[2])
    np.testing.assert_array_equal(item['teacher_features'].numpy(),features[2].astype(np.float32).mean(axis=1))
    for k in MOTION_FIELDS:
        expected=arrays[k][2][:,::-1] if k in MOTION_FIELDS[:10] else arrays[k][2]
        np.testing.assert_array_equal(item[k].numpy(),expected)
    assert int(item['label'])==39 and item['sample_id']=='c'
    assert not bool(dataset[1]['imu_teacher_valid']) and not dataset[1]['imu_teacher_logits'].any()
    inference=m.NoVoteMotionDataset(arrays,index,rows=subset)
    assert 'label' not in inference[0] and 'teacher_logits' not in inference[0]


def test_fixed24_training_engine_resume_and_no_validation(no_vote_fixture,tmp_path):
    m=module();assert hasattr(m,'run_motion_epochs'), 'fixed24 resumable source training engine is missing'
    p=motion_protocol(no_vote_fixture);ops=m.source_motion_math(p,VerificationContext(p));args=m.motion_args(p)
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor([1.,-1.]))
        def forward(self,motion,mask_ratio=0):return self.weight
    dataset=[{'label':torch.tensor(i%2),**{k:torch.tensor(0.) for k in MOTION_FIELDS}} for i in range(3)]
    loader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True)
    # One pure source-like loss boundary; the actual source losses have separate numerical/gradient tests.
    def loss(output,batch,weights,args,ops):return {'loss':output.square().sum(),'semantic':output.square().sum()}
    root=tmp_path/'epochs';model=Tiny()
    class Interrupted(Exception):pass
    def stop(epoch):
        if epoch==2:raise Interrupted()
    with pytest.raises(Interrupted):m.run_motion_epochs(model,loader,np.array([0,1,0]),args,root,device='cpu',ops=ops,identity='fixture',loss_function=loss,epoch_callback=stop)
    history=m.run_motion_epochs(Tiny(),loader,np.array([0,1,0]),args,root,device='cpu',ops=ops,identity='fixture',loss_function=loss)
    assert len(history)==24 and [x['epoch'] for x in history]==list(range(1,25))
    assert all(x['train_samples']==3 for x in history)
    assert history[0]['learning_rate']==pytest.approx(4e-4) and history[-1]['learning_rate']==pytest.approx(2e-5)
    assert all(not any(k.startswith(('val_','validation','development')) for k in row) for row in history)


def test_motion_phase_boundary_rejects_held_users_before_fit(no_vote_fixture):
    m=module();assert hasattr(m,'train_motion_student'), 'Task8 phase training API is missing'
    from types import SimpleNamespace
    p=motion_protocol(no_vote_fixture);inputs=SimpleNamespace(partition=SimpleNamespace(name='development2'))
    with pytest.raises(ValueError,match='phase|population'):
        m.train_motion_student('select',inputs,None,None,None,None,None,protocol=p,device='cpu')


def test_combined_motion_output_is_diagnostic_and_masks_missing_branches():
    m=module();assert hasattr(m,'combine_motion_probabilities'), 'motion combined diagnostic is missing'
    sk=np.tile(np.eye(40,dtype=np.float32)[1],(3,1));imu=np.tile(np.eye(40,dtype=np.float32)[5],(3,1));prior=np.full(40,.025)
    result=m.combine_motion_probabilities(sk,imu,np.array([1,1,0],bool),np.array([1,0,0],bool),prior)
    np.testing.assert_array_equal(result[0],(sk[0]+imu[0])*.5)
    np.testing.assert_array_equal(result[1],sk[1]);np.testing.assert_array_equal(result[2],prior)
