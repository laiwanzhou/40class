"""Task10: source fusion math/modes and synchronized no-vote adapters."""
from copy import deepcopy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.experiments.artifact_record import VerificationContext
from src.experiments.no_vote_types import RowIndex
from src.experiments.motion_cache import MOTION_FIELDS,window_schema,BOOLEAN_MOTION_FIELDS
from test_no_vote_mobind_pretrain import motion_protocol


def module():
    assert importlib.util.find_spec('src.experiments.mobind_fusion'), 'Task10 source fusion is missing'
    return importlib.import_module('src.experiments.mobind_fusion')


def protocol(fixture):
    from dataclasses import replace
    from src.experiments.no_vote_source_ops import compile_source_ops,runtime_recipe_values
    from test_no_vote_source_ops import SOURCE_ROOT
    p=motion_protocol(fixture)
    return replace(p,recipe={**p.recipe,'fusion':runtime_recipe_values(compile_source_ops(SOURCE_ROOT))['fusion']})


class ToyVisual(torch.nn.Module):
    temporal_modeling=True
    def __init__(self):
        super().__init__();self.classifier=torch.nn.Linear(512,40)
        for name in ['temporal_encoder','temporal_fusion','token_encoder','quality_gate','stage_fusion']:
            setattr(self,name,torch.nn.Linear(1,1))
        for name in ['time_position','view_embedding','window_embedding']:setattr(self,name,torch.nn.Parameter(torch.zeros(1)))
    def encode_clips_from_backbone_sequence(self,x,valid,time):return x.mean(3)
    def _fuse_clips(self,x,valid,quality):
        pooled=x.mean((1,2));return dict(logits=self.classifier(pooled),visual_embedding=pooled,clip_embeddings=x,clip_mask=valid.any(2))


def test_factory_matches_source_build_model_and_rng(no_vote_fixture,tmp_path,monkeypatch):
    m=module();p=protocol(no_vote_fixture);c=VerificationContext(p);ops=m.source_fusion_math(p,c);args=m.fusion_args(p)
    torch.set_num_threads(2);visual=ToyVisual();cfg=dict(width=96,alignment_width=64,classes=40,dropout=.12)
    base=ops['module'].P86MoBindLite(**cfg);path=tmp_path/'motion.pt'
    torch.save({'model_config':cfg,'model_state':base.state_dict()},path)
    args.pretrain_checkpoint=path;args.visual_checkpoint=tmp_path/'unused.pt'
    monkeypatch.setattr(ops['module'],'load_visual',lambda _: (deepcopy(visual),{}))
    torch.manual_seed(91);fresh=ops['module'].P86MoBindLite(**cfg);fresh.load_state_dict(base.state_dict())
    actual=m.assemble_fusion(deepcopy(visual),fresh,cfg,args,ops);actual_rng=torch.get_rng_state()
    torch.manual_seed(91);expected,_,_=ops['module'].build_model(args)
    assert actual.state_dict().keys()==expected.state_dict().keys()
    for k,v in actual.state_dict().items():torch.testing.assert_close(v,expected.state_dict()[k],atol=0,rtol=0)
    assert torch.equal(actual_rng,torch.get_rng_state())


@pytest.mark.parametrize('missing',[False,True])
def test_full_source_losses_and_gradients_without_new_gating(no_vote_fixture,missing):
    m=module();p=protocol(no_vote_fixture);ops=m.source_fusion_math(p,VerificationContext(p));args=m.fusion_args(p)
    torch.manual_seed(7);out={k:torch.randn(3,40,requires_grad=True) for k in ['logits','motion_logits','unfused_visual_logits']}
    out.update(clip_embeddings=torch.randn(3,2,3,512,requires_grad=True),clip_mask=torch.ones(3,2,3,dtype=torch.bool),
        motion_available=torch.full((3,2),not missing),motion_global_reliability=torch.full((3,1),.7,requires_grad=True),
        motion_reliability=torch.full((3,2,3),.7,requires_grad=True))
    batch=dict(label=torch.tensor([0,1,2]),user_id=['fit1','fit2','fit1'],teacher_logits=torch.randn(3,40),
        teacher_features=torch.randn(3,2,3,1024),anchor_logits=torch.randn(3,40))
    weights=torch.ones(40);a=m.fusion_losses(out,batch,weights,args,ops);a['loss'].backward()
    grads={k:v.grad.clone() for k,v in out.items() if v.requires_grad and v.grad is not None}
    expected={k:v.detach().clone().requires_grad_(v.requires_grad) for k,v in out.items()}
    b=ops['losses'](expected,batch,weights,args);b['loss'].backward()
    for k in a:torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
    for k in grads:torch.testing.assert_close(grads[k],expected[k].grad,atol=0,rtol=0)
    assert float(a['motion_aux'])>0 and args.stage_a_epochs==4 and args.stage_b_epochs==20


def test_frozen_motion_dropout_and_stage_groups_match_source(no_vote_fixture):
    m=module();p=protocol(no_vote_fixture);ops=m.source_fusion_math(p,VerificationContext(p));args=m.fusion_args(p)
    torch.set_num_threads(2);v=ToyVisual();base=ops['module'].P86MoBindLite(width=96,alignment_width=64,dropout=.12)
    net=m.assemble_fusion(v,base,dict(width=96,dropout=.12),args,ops)
    class Empty:
        dataset=[]
        def __iter__(self):return iter(())
    for stage in ['A','B']:
        m.run_source_stage(net,Empty(),np.arange(40),args,'cpu',stage,1,ops)
        assert all(not x.requires_grad for x in net.motion_residual.encoder.skeleton_encoder.parameters())
        assert all(not x.requires_grad for x in net.motion_residual.encoder.imu_encoder.parameters())
        assert net.motion_residual.encoder.skeleton_encoder.training
        assert any(x.training for x in net.motion_residual.encoder.modules() if isinstance(x,torch.nn.Dropout))
        assert v.training==(stage=='B') and v.classifier.weight.requires_grad==(stage=='B')


def test_dataset_temporal_join_precision_and_no_teacher_inference(no_vote_fixture):
    m=module();n=4;idx=RowIndex(tuple('abcd'),('fit1',)*4,tuple(range(40)));rng=np.random.default_rng(3)
    arrays={k:(np.ones(shape,bool) if k in BOOLEAN_MOTION_FIELDS else np.full(shape,.125,np.float32)) for k,(_,shape) in window_schema(n).items()}
    for j in range(n):arrays['skeleton_features'][j]=j+1;arrays['imu_global_statistics'][j]=j+10
    seq=np.broadcast_to(np.arange(16,dtype=np.float16)[None,None,None,:,None],(n,2,3,16,512)).copy()
    pixel=dict(view_valid=np.ones((n,2,16,3),bool),view_quality=np.full((n,2,16,3),.1234567,np.float32),
        source_frame_indices=np.broadcast_to(np.arange(16),(n,2,16)).copy())
    anchor=np.full((n,40),.2345678,np.float32);reverse=lambda _:np.arange(15,-1,-1)
    ds=m.FusionDataset(arrays,idx,seq,idx,pixel,idx,anchor,rows=idx,temporal_indices=reverse)
    item=ds[1];assert 'label' not in item and 'teacher_logits' not in item
    np.testing.assert_array_equal(item['backbone_sequence'][0,0,:,0],np.arange(15,-1,-1))
    np.testing.assert_array_equal(item['global_time_position'][0],np.arange(15,-1,-1,dtype=np.float32)/np.float32(15))
    assert float(item['view_quality'][0,0,0])==float(np.float16(.1234567))
    assert float(item['anchor_logits'][0])==float(np.float16(.2345678))
    bad=RowIndex(tuple('abcz'),('fit1',)*4,tuple(range(40)))
    with pytest.raises(ValueError,match='ID'):m.FusionDataset(arrays,idx,seq,bad,pixel,idx,anchor,rows=idx)
    donors=m.fit_shuffle(idx,seed=19);assert sorted(donors)==list(range(4))
    shuffled=m.FusionDataset(arrays,idx,seq,idx,pixel,idx,anchor,rows=idx,labels=dict.fromkeys(idx.sample_ids,0),variant='shuffle_s',shuffle=donors)
    assert float(shuffled[1]['skeleton_features'].flatten()[0])==donors[1]+1
    assert float(shuffled[1]['imu_global_statistics'].flatten()[0])==11
    masked=m.FusionDataset(arrays,idx,seq,idx,pixel,idx,anchor,rows=idx,variant='mask_motion')
    assert all(not masked[1][k].any() for k in MOTION_FIELDS)


def test_source_residual_padding_index_compatibility(no_vote_fixture):
    m=module();assert hasattr(m,'compatible_residual_class'), 'source 3D boolean padding needs equivalent indexing'
    p=protocol(no_vote_fixture);ops=m.source_fusion_math(p,VerificationContext(p))
    # PyTorch 2.7 rejects this source expression for rank3 even if no row is empty.
    x=torch.zeros(2,2,5,dtype=torch.bool);empty=~x.any(-1)
    with pytest.raises(IndexError):x[empty,0]=True
    fn=m.compatible_residual_class(ops['module'].P86MoBindMotionResidual).forward
    import inspect,ast,textwrap
    # Independent reference replaces just the offending assignment by explicit row flattening.
    original=ops['module'].P86MoBindMotionResidual.forward
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0])=='safe_mask[empty, 0]':
                return ast.parse('safe_mask.reshape(-1, safe_mask.shape[-1])[empty.reshape(-1), 0] = True').body[0]
            return node
    tree=ast.fix_missing_locations(Rewrite().visit(tree));namespace=dict(original.__globals__)
    exec(compile(tree,'<independent_flat_padding>','exec'),namespace);reference=namespace['forward']
    from test_no_vote_mobind_pretrain import synthetic_batch
    torch.manual_seed(3);base=ops['module'].P86MoBindLite(width=96,alignment_width=64,dropout=.12)
    net=m.assemble_fusion(ToyVisual(),base,dict(width=96,dropout=.12),m.fusion_args(p),ops)
    batch=synthetic_batch();motion={k:batch[k] for k in MOTION_FIELDS}
    for k in MOTION_FIELDS:motion[k][0]=0
    clips=torch.randn(3,2,3,512);times=torch.linspace(0,1,16).expand(3,2,-1)
    torch.manual_seed(17);a=fn(net.motion_residual,clips,times,motion)
    torch.manual_seed(17);b=reference(net.motion_residual,clips,times,motion)
    torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
    for k in a[1]:torch.testing.assert_close(a[1][k],b[1][k],atol=0,rtol=0)


def test_controls_cannot_become_main_supervision(no_vote_fixture):
    module()
    from test_no_vote_v3_roles import registry,model
    r=registry(no_vote_fixture);control=model(r,'A7-mask');file=r.root/'main.bin';file.write_bytes(b'x')
    with pytest.raises(ValueError,match='diagnostic'):
        r.register(stage='A7',kind='supervised_model',phase='select',files=[file],parents=[control],
            fit_users=r.protocol.partitions['train12'].users)


def test_unbound_fusion_cannot_bypass_global_gate(no_vote_fixture):
    module()
    from dataclasses import replace
    from test_no_vote_v3_roles import registry
    from src.experiments.artifact_record import ArtifactRegistry
    r=registry(no_vote_fixture);p=replace(r.protocol,recipe={**r.protocol.recipe,'source_ops_bindings':{'formal_generation_ready':False}})
    r=ArtifactRegistry(p);file=r.root/'main.bin';file.write_bytes(b'x')
    with pytest.raises(ValueError,match='not ready'):
        r.register(stage='A7',kind='supervised_model',phase='select',files=[file],fit_users=p.partitions['train12'].users)


def test_checkpoint_saves_frozen_recipe_without_changing_tensor_state(tmp_path):
    from collections import OrderedDict
    from src.experiments.no_vote_types import freeze
    from src.experiments.visual_student_v3 import _save_torch
    state=OrderedDict(weight=torch.arange(4));state._metadata={'':{'version':1}}
    body={'visual_config':freeze({'epochs':{'max':16},'parts':[{'width':512}]}),'model_state':state}
    path=tmp_path/'checkpoint.pt';_save_torch(path,body)
    actual=torch.load(path,weights_only=True)
    assert actual['visual_config']['epochs']=={'max':16} and actual['model_state']._metadata==state._metadata
    torch.testing.assert_close(actual['model_state']['weight'],state['weight'],atol=0,rtol=0)
    with pytest.raises(TypeError):body['visual_config']['epochs']['max']=20
