"""Reconstructed source image teachers: original dataset math without fake labels."""
import importlib
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image


def module():
    assert importlib.util.find_spec('src.experiments.p12_visual_reconstruction'), 'P12 visual bridge is missing'
    return importlib.import_module('src.experiments.p12_visual_reconstruction')


def test_public_source_getitem_removes_label_dependency_only(no_vote_fixture,tmp_path):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.visual_source_ops(p,VerificationContext(p))
    cfg={'num_frames':12,'image_height':144,'image_width':192,'normalization':'legacy'}
    folder=tmp_path/'thermal';folder.mkdir()
    for i in range(2):Image.fromarray(np.full((16,20,3),60+i*30,np.uint8)).save(folder/f'frame_{i}.png')
    row={'sample_id':'a'*64,'user_id':'fit1'}
    actual=m.SourceImageDataset([row],{'a'*64:{'thermal_dir':str(folder)}},'thermal',cfg,ops,augment=False)[0]
    assert 'label' not in actual and actual['clip'].shape==(12,3,144,192)
    original=object.__new__(ops['thermal_dataset_class'])
    original.num_frames=12;original.image_height=144;original.image_width=192;original.augment=False;original.normalization='legacy'
    original.samples=[SimpleNamespace(sample_id='a'*64,class_id=3,trial_dir=folder)]
    expected=original[0]
    torch.testing.assert_close(actual['clip'],expected['clip'],atol=0,rtol=0)


def test_depth_cached_consumer_matches_original_source(no_vote_fixture,tmp_path):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.visual_source_ops(p,VerificationContext(p));files=[]
    for i in range(3):
        path=tmp_path/f'depth_{i}.png';Image.fromarray(np.full((20,24,3),50+i*40,np.uint8)).save(path);files.append(str(path))
    cfg={'num_frames':12,'image_height':144,'image_width':192,'depth_representation':'jet_rgb'}
    ds=m.SourceImageDataset([{'sample_id':'b'*64,'user_id':'fit1'}],{'b'*64:{'depth_frames':files}},'depth',cfg,ops,augment=True)
    torch.manual_seed(7);actual=ds[0];assert 'label' not in actual
    public=ds.reader;sample=SimpleNamespace(sample_id='b'*64,user_id='fit1',class_id=4)
    torch.manual_seed(7);expected=ops['aligned_dataset_class']._getitem_from_cache(public,sample)
    torch.testing.assert_close(actual['depth'],expected['depth'],rtol=0,atol=0)


def test_image_models_keep_source_initialization(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.visual_source_ops(p,VerificationContext(p));torch.set_num_threads(2)
    cfg={'seed':20260720,'dropout':.3,'use_layer3_spatial':True}
    a=m.initialize_image_model('depth',cfg,ops,device='cpu');ops['seed'](20260720)
    b=ops['aligned_model_class'](['depth'],dropout=.3,use_layer3_spatial=True,imagenet_pretrained=False)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)


def test_image_gate_rejects_unbound_or_wrong_population(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
    p=v3_protocol(no_vote_fixture);registry=ArtifactRegistry(p,context=VerificationContext(p))
    record=SimpleNamespace(stage='source_expert/p12_depth',kind='supervised_model',phase='select',
        config={'revision':'task11_reconstruction_v2','budget_epochs':15,'fit_partition':'refit14'},parents=())
    assert m.allows_image_reconstruction_stage(registry,record) is False


def test_fixed_image_training_resume_preserves_sampler_rng_optimizer_and_budget(no_vote_fixture,tmp_path):
    m=module();assert hasattr(m,'fit_image_epochs'),'resumable image training seam is missing'
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    from src.experiments.mobind_fusion import _rng_state
    import random
    p=v3_protocol(no_vote_fixture);ops=m.visual_source_ops(p,VerificationContext(p));source=ops['thermal_train'];torch.set_num_threads(2)
    class SmallDataset(torch.utils.data.Dataset):
        samples=[SimpleNamespace(class_id=i%2) for i in range(32)]
        def __len__(self):return 32
        def __getitem__(self,i):return {'clip':torch.tensor([i/32,random.random(),np.random.random(),torch.rand(()).item()]),'label':i%2,'sample_id':str(i)}
    cfg={'seed':19,'batch_size':16,'balanced_sampling':True,'num_workers':0,'val_num_workers':0,'max_grad_norm':5.}
    def run(path,fail_after=None):
        source.seed_everything(19);network=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.3),torch.nn.Linear(8,40))
        loader=source.create_loader(SmallDataset(),cfg,True);opt=torch.optim.AdamW(network.parameters(),lr=.0007,weight_decay=.0002)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=15);scaler=torch.amp.GradScaler('cuda',enabled=False)
        history=m.fit_image_epochs(network,loader,opt,schedule,scaler,torch.nn.CrossEntropyLoss(label_smoothing=.1),source,
            'thermal','select',cfg,path,'fixed_identity',device='cpu',fail_after=fail_after)
        return network.state_dict(),opt.state_dict(),schedule.state_dict(),_rng_state(),history
    complete=run(tmp_path/'complete')
    with pytest.raises(RuntimeError,match='simulated'):run(tmp_path/'resume',fail_after=4)
    resumed=run(tmp_path/'resume')
    def same(a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    for a,b in zip(complete[:-1],resumed[:-1]):same(a,b)
    assert len(resumed[-1])==15 and resumed[-1][-1]['epoch']==15
    same([{k:v for k,v in h.items() if k!='seconds'} for h in complete[-1]],
         [{k:v for k,v in h.items() if k!='seconds'} for h in resumed[-1]])
