import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.visual90_corpus import assert_identity, verify_files
from src.training.visual90_training import paired_epoch_indices, learning_rate_at, save_training_state, restore_training_state, decide_results


def test_resume_rejects_changed_identity():
    with pytest.raises(ValueError, match='identity'):
        assert_identity({'identity':'old'},'new')


def test_source_or_cache_corruption_is_rejected(tmp_path):
    p=tmp_path/'cache.bin';p.write_bytes(b'a')
    with pytest.raises(ValueError,match='hash'):
        verify_files({'cache.bin':'not-the-hash'},tmp_path)


def test_sampler_is_deterministic_and_never_accepts_validation_users():
    y=np.repeat(np.arange(40),2);u=np.tile([1,2],40)
    indices=paired_epoch_indices(y,u,20260715,1)
    np.testing.assert_array_equal(indices,paired_epoch_indices(y,u,20260715,1))
    for a,b in indices.reshape(-1,2):
        assert y[a]==y[b] and u[a]!=u[b]
    u[0]=6
    with pytest.raises(ValueError,match='user'):
        paired_epoch_indices(y,u,20260715,1)


def test_lr_warmup_and_decay_are_epoch_owned():
    assert learning_rate_at(1)==.00015
    assert learning_rate_at(2)==.0003
    assert learning_rate_at(30)<learning_rate_at(15)<learning_rate_at(2)


def test_checkpoint_resume_restores_next_randomness_and_optimizer_update(tmp_path):
    torch.manual_seed(8)
    m=torch.nn.Linear(3,2);o=torch.optim.AdamW(m.parameters(),lr=.001)
    def step():
        o.zero_grad(); m(torch.rand(4,3)).square().mean().backward();o.step()
    step();p=tmp_path/'latest.pt'
    save_training_state(p,m,o,1,'fixed',{'loss':1.})
    step();expected={k:v.clone() for k,v in m.state_dict().items()}
    restore_training_state(p,m,o,'fixed',torch.device('cpu'))
    step()
    for key,value in expected.items(): torch.testing.assert_close(m.state_dict()[key],value,rtol=0,atol=0)
    with pytest.raises(ValueError,match='identity'):
        restore_training_state(p,m,o,'different',torch.device('cpu'))


def test_absolute_target_is_separate_from_appearance_increment():
    result=decide_results({'correct':350,'worst_user_accuracy':.9},{'correct':354,'worst_user_accuracy':.9})
    assert result['target_reached']
    assert result['appearance_increment']=='mixed_or_unproven'


def test_full_training_loop_resume_matches_uninterrupted_and_never_reads_validation(tmp_path):
    from scripts.run_visual90_experiment import fit_candidate
    class TinyData(torch.utils.data.Dataset):
        def __init__(self):
            self.rows=[{'label':i//2,'user':1+i%2,'partition':'train','supported':True,
                        'disposition':'eligible_pending_geometry','sample_id':str(i)} for i in range(80)]
            self.rows.append({'label':0,'user':6,'partition':'validation','supported':True,
                              'disposition':'validation_only','sample_id':'val'})
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            assert i<80,'training accessed validation'
            r=self.rows[i]
            return {'features':torch.tensor([i/80.,(i%5)/5.,1.]),'label':r['label'],'user':r['user'],'sample_id':r['sample_id']}
    class TinyModel(torch.nn.Module):
        def __init__(self):super().__init__();self.fc=torch.nn.Linear(3,40)
        def forward(self,b):
            out=self.fc(b['features']);return {'logits':out,'embedding':torch.nn.functional.normalize(out[:,:8],dim=-1)}
    torch.set_num_threads(1)
    ds=TinyData();config={'seed':20260715,'num_workers':0,'run_root':str(tmp_path/'continuous')}
    full,_=fit_candidate(ds,'temporal_visual',config,'fixture',stop_after_epoch=2,model_factory=TinyModel,device='cpu')
    config['run_root']=str(tmp_path/'resumed')
    fit_candidate(ds,'temporal_visual',config,'fixture',stop_after_epoch=1,model_factory=TinyModel,device='cpu')
    resumed,_=fit_candidate(ds,'temporal_visual',config,'fixture',stop_after_epoch=2,model_factory=TinyModel,device='cpu')
    for key,value in full.state_dict().items():torch.testing.assert_close(resumed.state_dict()[key],value,rtol=0,atol=0)
