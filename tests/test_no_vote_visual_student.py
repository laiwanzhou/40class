import torch
import pytest


def test_hybrid_active_losses_and_missing_teacher_mask():
    from src.experiments.visual_student import hybrid_loss
    torch.manual_seed(9)
    output={'logits':torch.randn(2,40,requires_grad=True),
        'clip_embeddings':torch.randn(2,2,3,512,requires_grad=True),'clip_mask':torch.ones(2,2,3,dtype=torch.bool)}
    batch={'label':torch.tensor([0,1]),'teacher_logits':torch.randn(2,40),
        'teacher_features':torch.randn(2,2,3,1024),'teacher_valid':torch.tensor([True,False])}
    recipe=dict(label_smoothing=.1,distillation_temperature=2.,distillation_weight=1.,relation_weight=.2,
        feature_weight_configured=.5,feature_weight_effective=0.,stage_distillation_weight=0.)
    parts=hybrid_loss(output,batch,torch.ones(40),recipe)
    expected=torch.nn.functional.kl_div(torch.log_softmax(output['logits'][:1]/2,1),
        torch.softmax(batch['teacher_logits'][:1]/2,1),reduction='batchmean')*4
    assert torch.allclose(parts['kd'],expected)
    assert parts['feature'].item()==0 and parts['stage_kd'].item()==0
    assert torch.allclose(parts['loss'],parts['ce']+parts['kd']+.2*parts['relation'])
    parts['loss'].backward()
    assert torch.count_nonzero(output['clip_embeddings'].grad[1])==0


def test_epoch_tie_break_prefers_earlier_epoch():
    from src.experiments.visual_student import epoch_key
    a=dict(accuracy=.7,macro_f1=.6,worst_user_accuracy=.5,epoch=3)
    b=dict(a,epoch=4)
    assert epoch_key(a)<epoch_key(b)


def test_local_mc3_initializer_and_frozen_batchnorm():
    from pathlib import Path
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.visual_student import initialize_student
    p=load_protocol(Path(__file__).resolve().parents[1]/'configs/experiments/teammate_single_teacher_fixed_split.yaml')
    model,ref=initialize_student(p,device='cpu')
    assert ref is not None and model.distillation_projection is None
    assert not any(x.requires_grad for block in (model.stem,model.layer1,model.layer2) for x in block.parameters())
    assert any(x.requires_grad for x in model.layer3.parameters())
    model.train()
    assert all(not block.training for block in model.modules() if isinstance(block,torch.nn.BatchNorm3d))


def test_teacher_guard_rejects_refit_target_for_select(no_vote_fixture,tmp_path):
    import csv
    import numpy as np
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_manifest import prepare_inputs,read_public_rows,row_index,load_stage_inputs
    from src.experiments.no_vote_types import Prediction,TeacherTargets
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.visual_student import guard_teacher
    p=load_protocol(no_vote_fixture[0]);source=tmp_path/'canonical.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','class_id','user_id']);w.writeheader()
        for user in ('fit1','dev1','target1'):
            for c in range(40):w.writerow(dict(sample_id=f'{user}-{c}',class_id=c,user_id=user))
    prepare_inputs(source,p,p.run_root/'protocol',tmp_path/'private')
    r=ArtifactRegistry(p);file=p.run_root/'teacher.json';file.write_text('{}')
    model=r.register(stage='A1',kind='supervised_model',phase='refit',files=[file],fit_users=p.partitions['refit14'].users)
    index=row_index(read_public_rows(p.public_manifests['refit14']));n=len(index.sample_ids)
    path=p.run_root/'targets.npz';np.savez(path,sample_ids=index.sample_ids,class_ids=np.arange(40),
        logits=np.zeros((n,40)),probabilities=np.full((n,40),1/40),valid=np.ones(n,bool))
    ref=r.register(stage='A1',kind='predictions',phase='predict',files=[path],parents=[model],rows=index,
        config={'partition':'refit14'})
    target=TeacherTargets(Prediction(index,np.zeros((n,40)),np.full((n,40),1/40),np.ones(n,bool),ref),
        np.zeros((n,2,3,1024)))
    with pytest.raises(ValueError,match='teacher.*population|phase'):
        guard_teacher(target,load_stage_inputs(p,'train12','select'),protocol=p)


def test_training_epoch_updates_last_partial_accumulation_group():
    from src.experiments.visual_student import train_epoch
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias=torch.nn.Parameter(torch.randn(40));self.clips=torch.nn.Parameter(torch.randn(2,3,512))
        def forward(self,images,valid,quality,position=None):
            n=len(images)
            return dict(logits=self.bias.expand(n,-1),clip_embeddings=self.clips.expand(n,-1,-1,-1),
                clip_mask=torch.ones(n,2,3,dtype=torch.bool))
    model=Tiny();before=model.bias.detach().clone()
    batch=dict(images=torch.zeros(2,2,16,3,2,2,dtype=torch.uint8),view_valid=torch.ones(2,2,16,3,dtype=torch.bool),
        view_quality=torch.ones(2,2,16,3),label=torch.tensor([0,1]),teacher_valid=torch.ones(2,dtype=torch.bool),
        teacher_logits=torch.zeros(2,40),teacher_features=torch.randn(2,2,3,1024))
    recipe=dict(gradient_accumulation=4,label_smoothing=.1,distillation_temperature=2.,distillation_weight=1.,
        relation_weight=.2,feature_weight_effective=0.,stage_distillation_weight=0.)
    report=train_epoch(model,[batch],torch.optim.SGD(model.parameters(),lr=.1),torch.ones(40),recipe,device='cpu')
    assert report['optimizer_steps']==1 and report['samples']==2
    assert not torch.equal(before,model.bias.detach())


def test_fixed_split_train_refit_and_unlabeled_prediction(no_vote_fixture,tmp_path,monkeypatch):
    import csv,json
    from pathlib import Path
    import numpy as np,yaml
    from src.experiments import visual_student as vs
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_manifest import prepare_inputs,read_public_rows,row_index,load_stage_inputs
    from src.experiments.no_vote_types import Prediction,TeacherTargets
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.visual_teacher import fit_class_prior
    config,payload=no_vote_fixture
    recipe=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/experiments/teammate_single_teacher_fixed_split.yaml').read_text(encoding='utf-8'))['recipe']['visual_student']
    recipe['epochs']={'min':1,'max':2,'select_on':'development2'};recipe['batch_size']=8
    payload['recipe']['visual_student']=recipe;config.write_text(yaml.safe_dump(payload));p=load_protocol(config)
    source=tmp_path/'canonical.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','class_id','user_id']);w.writeheader()
        for user in ('fit1','dev1','target1'):
            for c in range(40):w.writerow(dict(sample_id=f'{user}-{c}',class_id=c,user_id=user))
    prepare_inputs(source,p,p.run_root/'protocol',tmp_path/'private');r=ArtifactRegistry(p)
    index=row_index(read_public_rows(p.public_manifests['refit14']));folder=p.run_root/'pixels';folder.mkdir()
    arrays=dict(images=np.zeros((80,2,16,3,160,160),np.uint8),view_valid=np.ones((80,2,16,3),bool),
        view_quality=np.ones((80,2,16,3),np.float32),source_frame_indices=np.tile(np.arange(16),(80,2,1)),
        source_time_seconds=np.tile(np.arange(16,dtype=float),(80,2,1)),completed=np.ones(80,bool))
    for name,value in arrays.items():np.save(folder/(name+'.npy'),value)
    (folder/'rows.csv').write_bytes(p.public_manifests['refit14'].read_bytes())
    pixels=r.register(stage='pixels',kind='raw_cache',phase='raw',files=list(folder.iterdir()),rows=index)
    targets={};target_paths=[]
    for phase,part in [('select','train12'),('refit','refit14')]:
        fit=load_stage_inputs(p,part,phase);fit_class_prior(fit,phase,protocol=p)
        idx=row_index(read_public_rows(fit.public_manifest));n=len(idx.sample_ids)
        file=p.run_root/(phase+'_teacher.json');file.write_text('{}')
        model=r.register(stage='A1',kind='supervised_model',phase=phase,files=[file],fit_users=p.partitions[part].users)
        path=p.run_root/(phase+'_target.npz');target_paths.append(path)
        logits=np.zeros((n,40),np.float32);prob=np.full((n,40),1/40,np.float32);valid=np.ones(n,bool)
        np.savez(path,sample_ids=idx.sample_ids,class_ids=np.arange(40),logits=logits,probabilities=prob,valid=valid)
        features=p.run_root/(phase+'_features.npz')
        np.savez(features,sample_ids=idx.sample_ids,class_ids=np.arange(40),features=np.ones((n,2,3,1024),np.float32),
            kinetics_logits=np.zeros((n,2,3,400),np.float32),valid=valid)
        fr=r.register(stage='visual_features',kind='raw_cache',phase='raw',files=[features],rows=idx,
            config={'partition':part,'feature_signature':'fixture'})
        ref=r.register(stage='A1',kind='predictions',phase='predict',files=[path],parents=[model,fr],rows=idx,config={'partition':part})
        targets[phase]=TeacherTargets(Prediction(idx,logits,prob,valid,ref),np.ones((n,2,3,1024),np.float32))
    initializations=[]
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias=torch.nn.Parameter(torch.zeros(40));self.clips=torch.nn.Parameter(torch.ones(2,3,512))
        def head_parameters(self):return [self.bias]
        def backbone_parameters(self):return [self.clips]
        def encode_backbone_sequence(self,images):
            value=images.float().mean((-1,-2)).permute(0,1,3,2).unsqueeze(-1)
            return value.expand(-1,-1,-1,-1,512)+self.clips[None,:,:,None,:]
        def forward_from_backbone_sequence(self,sequence,valid,quality,position=None):
            n=len(sequence);return dict(logits=self.bias.expand(n,-1),clip_embeddings=self.clips.expand(n,-1,-1,-1),
                clip_mask=valid.any(2))
        def forward(self,images,valid,quality,position=None):return self.forward_from_backbone_sequence(self.encode_backbone_sequence(images),valid,quality,position)
    def initialize(protocol,device='cpu',public_init=True):
        initializations.append(public_init);model=Tiny()
        ref=r.register(stage='visual_student_initializer',kind='public_weights',phase='public',files=[p.weights['mc3']]) if public_init else None
        return model,ref
    class Augment:
        @staticmethod
        def _subject_robust_augment(images):return images,np.arange(16)
    monkeypatch.setattr(vs,'initialize_student',initialize);monkeypatch.setattr(vs,'source_symbol',lambda *args:Augment)
    from dataclasses import replace
    bad=replace(targets['select'],features=np.zeros_like(targets['select'].features))
    with pytest.raises(ValueError,match='teacher.*features'):
        vs.guard_teacher(bad,load_stage_inputs(p,'train12','select'),protocol=p)
    actual_epoch=vs.train_epoch;completed=[]
    def interrupted(*args,**kwargs):
        if completed:raise RuntimeError('simulated interruption before second epoch')
        result=actual_epoch(*args,**kwargs);completed.append(1);return result
    monkeypatch.setattr(vs,'train_epoch',interrupted)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        vs.train_visual_student('select',load_stage_inputs(p,'train12','select'),
            load_stage_inputs(p,'development2','select'),pixels,targets['select'],None,protocol=p,device='cpu')
    monkeypatch.setattr(vs,'train_epoch',actual_epoch)
    selected,selection=vs.train_visual_student('select',load_stage_inputs(p,'train12','select'),
        load_stage_inputs(p,'development2','select'),pixels,targets['select'],None,protocol=p,device='cpu')
    repeated,_=vs.train_visual_student('select',load_stage_inputs(p,'train12','select'),
        load_stage_inputs(p,'development2','select'),pixels,targets['select'],None,protocol=p,device='cpu')
    assert repeated==selected and initializations==[True,True]
    # Removing only the registration marker must recover the last complete
    # epoch (optimizer/scaler/RNG included), not silently train again.
    (p.run_root/'A2/select/artifact.json').unlink()
    recovered,_=vs.train_visual_student('select',load_stage_inputs(p,'train12','select'),
        load_stage_inputs(p,'development2','select'),pixels,targets['select'],None,protocol=p,device='cpu')
    assert recovered==selected
    wrong_budget=replace(selection,budget={'epochs':2 if selection.budget['epochs']==1 else 1})
    with pytest.raises(ValueError,match='budget|epoch'):
        vs.train_visual_student('refit',load_stage_inputs(p,'refit14','refit'),None,pixels,targets['refit'],
            wrong_budget,protocol=p,device='cpu')
    refit,_=vs.train_visual_student('refit',load_stage_inputs(p,'refit14','refit'),None,pixels,
        targets['refit'],selection,protocol=p,device='cpu')
    assert initializations==[True,True,True] and selection.budget['epochs']==1
    assert selected not in r.verify(refit).parents
    for path in [*p.supervised_labels.values(),*target_paths]:path.unlink()
    prediction=vs.predict_visual_student(refit,pixels,index,protocol=p,device='cpu')
    assert prediction.probabilities.shape==(80,40) and np.isfinite(prediction.probabilities).all()
    sequence=vs.build_sequence(refit,pixels,index,p.run_root/'sequence',protocol=p,device='cpu')
    assert r.verify(sequence).phase=='refit'
    stored=np.load(p.run_root/'sequence/sequence.npy',mmap_mode='r')
    assert stored.shape==(80,2,3,16,512)
    anchor=np.load(p.run_root/'sequence/anchor_logits.npy')
    assert np.allclose(anchor,prediction.logits,atol=1e-6)


def test_phase_lock_rejects_an_active_duplicate(tmp_path):
    from src.experiments.visual_student import phase_lock
    with phase_lock(tmp_path):
        with pytest.raises(RuntimeError,match='already running'):
            with phase_lock(tmp_path):pass
    assert not (tmp_path/'training.lock.json').exists()
