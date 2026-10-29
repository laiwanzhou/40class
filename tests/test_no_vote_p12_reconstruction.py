"""P12 reconstructed Skeleton teacher must use source pose math and label-free prediction."""
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments.no_vote_types import RowIndex


def module():
    assert importlib.util.find_spec('src.experiments.p12_reconstruction'), 'P12 reconstructed source trainer is missing'
    return importlib.import_module('src.experiments.p12_reconstruction')


def test_skeleton_dataset_uses_source_sampling_flip_and_optional_labels(no_vote_fixture,tmp_path):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.source_p12_ops(p,VerificationContext(p));idx=RowIndex(('a',),('fit1',),tuple(range(40)))
    value=np.arange(20*17*4,dtype=np.float32).reshape(20,17,4)/100
    path=tmp_path/'a.npz';np.savez(path,skeleton=value)
    cfg=dict(num_frames=12);ds=m.SkeletonDataset(idx,{'a':path},cfg,ops,augment=False)
    x=ds[0];assert 'label' not in x
    positions=ops['sample_positions'](ds,20)
    np.testing.assert_array_equal(x['skeleton'],value[positions])
    fit=m.SkeletonDataset(idx,{'a':path},cfg,ops,augment=True,labels={'a':7})
    torch.manual_seed(19);actual=fit[0];torch.manual_seed(19);positions=ops['sample_positions'](fit,20);flip=torch.rand(1).item()<.5
    expected=torch.from_numpy(value[positions].copy())
    if flip:expected=torch.stack([ops['flip_skeleton'](row) for row in expected])
    torch.testing.assert_close(actual['skeleton'],expected,rtol=0,atol=0);assert int(actual['label'])==7


def test_invalid_skeleton_rows_keep_full_ids_and_fit_prior():
    m=module();prior=np.arange(1,41,dtype=np.float32);prior/=prior.sum()
    scores=np.full((3,40),-2.,np.float32);valid=np.array([True,False,True])
    logits,prob=m.complete_scores(scores,valid,prior)
    assert logits.shape==prob.shape==(3,40);np.testing.assert_array_equal(prob[1],prior)
    np.testing.assert_allclose(prob.sum(1),1,atol=1e-6)


def test_p12_skeleton_model_uses_original_class(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.source_p12_ops(p,VerificationContext(p))
    cfg={'dropout':.3,'seed':20260720};a=m.initialize_skeleton(cfg,ops,device='cpu')
    ops['seed'](20260720);b=ops['model_class'](['skeleton'],dropout=.3)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)


def test_unbound_reconstructed_teacher_does_not_unlock_legacy_global_gate(no_vote_fixture):
    module()
    from dataclasses import replace
    from test_no_vote_v3_roles import registry
    from src.experiments.artifact_record import ArtifactRegistry
    r=registry(no_vote_fixture);p=replace(r.protocol,recipe={**r.protocol.recipe,'source_ops_bindings':{'formal_generation_ready':False}})
    r=ArtifactRegistry(p);file=r.root/'bad.pt';file.write_bytes(b'not a recipe-bound source teacher')
    with pytest.raises(ValueError,match='not ready'):
        r.register(stage='source_expert/p12_skeleton',kind='supervised_model',phase='select',files=[file],fit_users=p.partitions['train12'].users)
