"""Original label-free training and exact complete-epoch resume for source-base."""
import importlib
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def module():
    assert importlib.util.find_spec('src.experiments.source_base_adaptation'),'intermediate source-base adaptation is missing'
    return importlib.import_module('src.experiments.source_base_adaptation')


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from test_no_vote_mobind_fusion import ToyVisual
        self.visual=ToyVisual();self.dropout=torch.nn.Dropout(.25)
        self.motion_residual=torch.nn.Module();self.motion_residual.encoder=torch.nn.Linear(4,4);self.motion_residual.head=torch.nn.Linear(4,40)
    def forward_from_backbone_sequence(self,sequence,valid,quality,time,motion):
        pooled=self.dropout(sequence.mean((1,2,3)))
        value=motion['imu_global_statistics'].flatten(1)[:,:4]
        return {'logits':self.visual.classifier(pooled)+self.motion_residual.head(self.motion_residual.encoder(value))}


def test_source_train12_and_interrupted_resume_are_identical(no_vote_fixture,tmp_path):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from test_no_vote_mobind_pretrain import synthetic_batch
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.source_adaptation_ops(p,VerificationContext(p));torch.set_num_threads(2)
    batch=synthetic_batch(3);batch.pop('label');batch['sample_id']=['a','b','c']
    batch.update(backbone_sequence=torch.randn(3,2,3,16,512),view_valid=torch.ones(3,2,16,3,dtype=torch.bool),
        view_quality=torch.ones(3,2,16,3),global_time_position=torch.zeros(3,2,16),
        pseudo_probability=torch.softmax(torch.randn(3,40),dim=1),pseudo_confidence=torch.ones(3))
    args=SimpleNamespace(epochs=12,adaptation_scope='heads',fusion_learning_rate=5e-5,visual_head_learning_rate=1e-5,
        minimum_learning_rate=2e-6,weight_decay=.02,temperature=1.,confidence_power=0.,emission_warmup_epochs=0,
        target='structured',max_train_batches=0)
    torch.manual_seed(17);initial=ToyModel();reference=deepcopy(initial);actual=deepcopy(initial)
    class RandomInputDataset(torch.utils.data.Dataset):
        def __len__(self):return 3
        def __getitem__(self,i):
            import random
            item={k:(v[i].clone() if torch.is_tensor(v) else v[i]) for k,v in batch.items()}
            item['backbone_sequence']+=float(random.random()+np.random.random()+torch.rand(()).item())*.01
            return item
    def loader():return ops['make_loader'](RandomInputDataset(),2,0,shuffle=True)
    ops['seed_all'](31);expected=ops['train_label_free'](reference,loader(),args,torch.device('cpu'))
    ops['seed_all'](31)
    with pytest.raises(RuntimeError,match='simulated'):
        m.train_source_with_resume(actual,loader(),args,ops,tmp_path/'resume.pt','identity',device='cpu',fail_after=4)
    resumed=deepcopy(initial)
    history=m.train_source_with_resume(resumed,loader(),args,ops,tmp_path/'resume.pt','identity',device='cpu')
    for k,v in reference.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],atol=0,rtol=0)
    assert [{k:v for k,v in row.items() if k!='seconds'} for row in expected]==[{k:v for k,v in row.items() if k!='seconds'} for row in history]
    assert len(history)==12 and history[-1]['fusion_learning_rate']==2e-6 and history[-1]['visual_head_learning_rate']==2e-6
    assert not resumed.motion_residual.encoder.weight.requires_grad and resumed.motion_residual.head.weight.requires_grad
    assert all(row['ground_truth_fields_seen']==0 for row in history)
    assert not any(x.requires_grad for name,x in resumed.named_parameters() if name.startswith('motion_residual.encoder.'))


def test_intermediate_scope_and_label_guard(no_vote_fixture,tmp_path):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.source_adaptation_ops(p,VerificationContext(p));torch.set_num_threads(2)
    args=SimpleNamespace(epochs=1,adaptation_scope='heads',fusion_learning_rate=5e-5,visual_head_learning_rate=1e-5,
        minimum_learning_rate=2e-6,weight_decay=.02,temperature=1.,confidence_power=0.,emission_warmup_epochs=0,target='structured',max_train_batches=0)
    with pytest.raises(RuntimeError,match='Ground-truth label'):
        m.train_source_with_resume(ToyModel(),[{'label':torch.tensor([0])}],args,ops,tmp_path/'guard.pt','guard',device='cpu')
    with pytest.raises(ValueError,match='phase'):m.train_source_base(p,'final4',None,VerificationContext(p))


def test_intermediate_recipe_and_unbound_model_are_rejected(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
    p=v3_protocol(no_vote_fixture);r=ArtifactRegistry(p,context=VerificationContext(p));file=p.run_root/'unbound_source_base.bin';file.write_bytes(b'unbound')
    with pytest.raises(ValueError,match='not ready'):
        r.register(stage=m.STAGE,kind='supervised_model',phase='select',files=[file],fit_users=p.partitions['train12'].users)
    for changed in [{'epochs':40,'scope':'heads','confidence_power':0.},{'epochs':12,'scope':'heads_motion_encoder','confidence_power':0.}]:
        with pytest.raises(ValueError,match='twelve epochs'):m.adaptation_args(changed)
