"""Task5 source loss, fixed budget and v3-only ancestry contracts."""
import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.experiments.artifact_record import VerificationContext
from test_no_vote_source_ops import v3_protocol
from test_no_vote_visual_teacher_source_parity import source_population


def module():
    assert importlib.util.find_spec('src.experiments.visual_student_v3'), 'Task5 source student missing'
    return importlib.import_module('src.experiments.visual_student_v3')


def test_source_fixed_budget_16():
    m=module();recipe={'fixed_epochs':16,'epochs':{'min':16,'max':16,'select_on':'fixed_source_budget'}}
    assert m.fixed_budget(recipe)==16
    for key in ['min','max']:
        bad={**recipe,'epochs':{**recipe['epochs'],key:18}}
        with pytest.raises(ValueError,match='fixed|budget'):m.fixed_budget(bad)
    assert m.checkpoint_epoch(3,16)==16


def test_source_hybrid_loss_parity(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);context=VerificationContext(p);ops=m.source_student_math(p,context)
    torch.manual_seed(31)
    logits=torch.randn(3,40);emb=torch.randn(3,2,3,512)
    mask=torch.ones(3,2,3,dtype=torch.bool);mask[1,0,2]=False
    batch={'label':torch.tensor([0,7,15]),'teacher_logits':torch.randn(3,40),
        'teacher_features':torch.randn(3,2,3,1024),'teacher_valid':torch.ones(3,dtype=torch.bool)}
    recipe=dict(mode='hybrid',feature_weight_effective=0.,stage_distillation_weight=0.,label_smoothing=.1,
        distillation_temperature=2.,distillation_weight=1.,relation_weight=.2)
    weights=ops['class_weights'](np.tile(np.arange(40),2),.35,torch.device('cpu'))
    a={'logits':logits.clone().requires_grad_(),'clip_embeddings':emb.clone().requires_grad_(),'clip_mask':mask}
    b={'logits':logits.clone().requires_grad_(),'clip_embeddings':emb.clone().requires_grad_(),'clip_mask':mask}
    parts=m.source_hybrid_loss(a,batch,weights,recipe,ops['relation_loss'])
    ce=F.cross_entropy(b['logits'],batch['label'],weight=weights,label_smoothing=.1)
    kd=F.kl_div(F.log_softmax(b['logits']/2,dim=1),F.softmax(batch['teacher_logits']/2,dim=1),reduction='batchmean')*4
    relation=ops['relation_loss'](b['clip_embeddings'],batch['teacher_features'],mask)
    expected=ce+kd+.2*relation
    torch.testing.assert_close(parts['loss'],expected,rtol=0,atol=0)
    parts['loss'].backward();expected.backward()
    torch.testing.assert_close(a['logits'].grad,b['logits'].grad,rtol=0,atol=0)
    torch.testing.assert_close(a['clip_embeddings'].grad,b['clip_embeddings'].grad,rtol=0,atol=0)
    assert parts['feature'].item()==0 and parts['stage_kd'].item()==0


def test_v2_targets_rejected(no_vote_fixture):
    from src.experiments.no_vote_protocol import load_protocol
    m=module();p=load_protocol(no_vote_fixture[0])
    with pytest.raises(ValueError,match='v3|Task4'):
        m.prepare_student_capability(p,VerificationContext(p))


def test_student_stage_scope_does_not_change_global_gate(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);before=p.identity()
    with pytest.raises(ValueError,match='Task4|complete'):
        m.prepare_student_capability(p,VerificationContext(p))
    assert p.identity()==before


def test_source_weight_and_schedule_parity(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);ops=m.source_student_math(p,VerificationContext(p))
    labels=np.array([0,0,1,2,2,2,7])
    actual=ops['class_weights'](labels,.35,torch.device('cpu'))
    counts=np.bincount(labels,minlength=40);present=counts>0
    expected=np.zeros(40,np.float32);expected[present]=(counts[present].mean()/counts[present])**.35
    torch.testing.assert_close(actual,torch.from_numpy(expected))
    assert ops['learning_rate_scale'](1,16,.05)==.5
    assert ops['learning_rate_scale'](2,16,.05)==1
    assert ops['learning_rate_scale'](16,16,.05)==pytest.approx(.05)


def test_inference_protocol_needs_receipts_not_teacher_assets(no_vote_fixture):
    from src.experiments.no_vote_protocol import load_protocol
    p=load_protocol(no_vote_fixture[0]);identity=p.identity()
    for name,path in p.weights.items():
        if name=='videomae':
            for file in path.iterdir():file.unlink()
        else:path.unlink()
    loaded=load_protocol(no_vote_fixture[0],inference_only=True)
    assert loaded.identity()==identity
    with pytest.raises(ValueError,match='weight file'):load_protocol(no_vote_fixture[0])


def test_v3_fixed_last_epoch_resume_and_refit_without_validation(source_population,monkeypatch):
    import json
    from src.experiments import visual_student as base,visual_student_v3 as source,visual_teacher_v3 as teacher
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows,row_index
    from src.experiments.no_vote_types import canonical_bytes,canonical_hash
    p,c,features=source_population
    formal=__import__('yaml').safe_load((Path(__file__).resolve().parents[1]/'configs/experiments/teammate_single_teacher_fixed_split_v3.yaml').read_text(encoding='utf-8'))
    import os,hashlib
    initializer=p.run_root/'weights/mc3/mc3.pth';initializer.parent.mkdir(parents=True)
    os.link(p.weights['mc3'],initializer)
    receipt_path=p.run_root/'protocol/weights_manifest.json';weight_receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    weight_receipt['weights']['mc3']['local_path']=str(initializer)
    weight_receipt['weights']['mc3']['files'][0]['path']=str(initializer)
    receipt_path.write_text(json.dumps(weight_receipt),encoding='utf-8')
    # Rebind the fixture before producing any stage records; source recipe fixed16.
    p=replace(p,weights={**p.weights,'mc3':initializer},recipe={**p.recipe,'visual_student':formal['recipe']['visual_student'],
        'asset_bindings':{**p.recipe['asset_bindings'],'weights_manifest_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest()}})
    c=VerificationContext(p)
    # Fixture's old preparation was a different protocol. Its payloads are unchanged;
    # make fresh registry/receipt records instead of mutating records.
    r=ArtifactRegistry(p,context=c)
    receipt={'protocol_sha256':p.identity(),'partitions':{},'supervised_label_hashes':{}}
    from src.experiments.teammate_source import sha256_file
    for part in p.partitions:
        rows=row_index(read_public_rows(p.public_manifests[part]))
        ref=r.register(stage='manifest_'+part,kind='public_manifest',phase='raw',files=[p.public_manifests[part]],rows=rows)
        receipt['partitions'][part]={'record_path':str(ref.record_path),'sha256':ref.sha256}
        if part!='final4':receipt['supervised_label_hashes'][part]=sha256_file(p.supervised_labels[part])
    from src.experiments.no_vote_types import write_json
    write_json(p.run_root/'protocol/prepared_inputs.json',receipt)
    write_json(p.run_root/'run_state.json',{'protocol_sha256':p.identity(),'state':'prepared',
        'prepared_inputs_sha256':sha256_file(p.run_root/'protocol/prepared_inputs.json')})
    new_features={}
    for part,ref in features.items():
        # Same public fixture payload, new protocol record, no old ancestry.
        body=json.loads(ref.record_path.read_text(encoding='utf-8'))
        rows=row_index(read_public_rows(p.public_manifests[part]))
        new_features[part]=r.register(stage='visual_features',kind='raw_cache',phase='raw',files=list(body['files']),
            file_digests=body['files'],rows=rows,config=body['config'])
    write_json(p.run_root/'protocol/task3_geometry_parity.json',{'protocol_sha256':p.identity(),
        'pixel_crop_quality_times_exact':True,'retry_reason_bits_exact':True,'detector_executed':False})
    teacher.prepare_visual_capability(p,c)
    train=load_stage_inputs(p,'train12','select');dev=load_stage_inputs(p,'development2','select');refit=load_stage_inputs(p,'refit14','refit')
    selected=teacher.select_heads(train,dev,new_features['refit14'],protocol=p,context=c)
    train_prior=teacher.fit_prior(train,'select',protocol=p,context=c)
    teacher.predict_head(selected.fit_artifact,new_features['refit14'],row_index(read_public_rows(train.public_manifest)),train_prior,protocol=p,context=c)
    refit_head=teacher.refit_heads(refit,new_features['refit14'],selected,protocol=p,context=c)
    refit_prior=teacher.fit_prior(refit,'refit',protocol=p,context=c)
    teacher.predict_head(refit_head,new_features['refit14'],row_index(read_public_rows(refit.public_manifest)),refit_prior,protocol=p,context=c)
    # Private synthetic pixel cache, small disk-backed arrays, no real data.
    pixels={}
    shapes={'images':((2,16,3,160,160),np.uint8),'view_valid':((2,16,3),bool),'view_quality':((2,16,3),np.float32),
        'source_frame_indices':((2,16),np.int64),'source_time_seconds':((2,16),np.float64),'completed':((),bool)}
    for part in ['refit14','final4']:
        rows=row_index(read_public_rows(p.public_manifests[part]));folder=p.run_root/'pixel-fixture'/part;folder.mkdir(parents=True)
        for name,(tail,dtype) in shapes.items():
            a=np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype=dtype,shape=(len(rows.sample_ids),*tail))
            a[:]=1 if name in {'view_valid','view_quality','completed'} else 0;a.flush();del a
        (folder/'rows.csv').write_bytes(p.public_manifests[part].read_bytes())
        pixels[part]=r.register(stage='pixels',kind='raw_cache',phase='raw',files=list(folder.iterdir()),rows=rows,
            config={'partition':part})
    write_json(p.run_root/'protocol/task2_prepared.json',{'imports':[{'stage':'pixels','partition':k,
        'record_path':str(v.record_path),'sha256':v.sha256} for k,v in pixels.items()]})
    write_json(p.run_root/'protocol/task4_complete.json',{'complete':True,'protocol_sha256':p.identity(),
        'independent_artifact_review':{'heads':'GO','fullwindow':'GO'}})
    source.prepare_student_capability(p,c)
    inits=[]
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.backbone=torch.nn.Parameter(torch.zeros(2));self.head=torch.nn.Parameter(torch.zeros(40))
        def backbone_parameters(self):return [self.backbone]
        def head_parameters(self):return [self.head]
        def encode_backbone_sequence(self,images):
            return self.backbone[0].expand(len(images),2,3,16,512)
        def forward_from_backbone_sequence(self,sequence,valid,quality,position=None):
            return {'logits':self.head.expand(len(sequence),-1)}
        def forward(self,images,valid,quality,position=None):
            return self.forward_from_backbone_sequence(self.encode_backbone_sequence(images),valid,quality,position)
    def initialize(protocol,device='cpu',public_init=True,context=None):
        inits.append(public_init);return Tiny(),source.public_initializer(protocol,context) if public_init else None
    monkeypatch.setattr(source,'initialize_student',initialize)
    epochs=[];evaluations=[]
    def fake_epoch(model,*args,**kwargs):
        if len(epochs)==2 and not (p.run_root/'interrupted').exists():
            (p.run_root/'interrupted').write_text('yes');raise RuntimeError('fixture interruption')
        with torch.no_grad():model.head.add_(1)
        epochs.append(1);return {'loss':1.,'samples':40,'optimizer_steps':3,'seconds':0.}
    def evaluate(model,dataset,prior,**kwargs):
        evaluations.append(float(model.head[0].detach()));n=len(dataset)
        return np.zeros((n,40)),np.full((n,40),.025),np.ones(n,bool),None
    original_infer=base.infer_arrays
    monkeypatch.setattr(base,'train_epoch',fake_epoch);monkeypatch.setattr(base,'infer_arrays',evaluate)
    target=source.load_teacher(p,'train12',c)
    with pytest.raises(RuntimeError,match='fixture interruption'):
        source.train_visual_student('select',train,dev,pixels['refit14'],target,None,protocol=p,device='cpu',context=c)
    model,selection=source.train_visual_student('select',train,dev,pixels['refit14'],target,None,protocol=p,device='cpu',context=c)
    assert len(epochs)==16 and evaluations==[16.] and selection.budget['epochs']==16
    assert r.verify(model).config['checkpoint_epoch']==16
    refit_model,_=source.train_visual_student('refit',refit,None,pixels['refit14'],source.load_teacher(p,'refit14',c),
        selection,protocol=p,device='cpu',context=c)
    assert len(epochs)==32 and evaluations==[16.] and inits==[True,True,True]
    assert selection.fit_artifact not in r.verify(refit_model).parents
    monkeypatch.setattr(base,'infer_arrays',original_infer)
    # Generation should need no teacher/label/initializer payload at inference.
    # Guard opens rather than deleting real data or mutating registered fixture files.
    original_open=Path.open
    def label_free(path,*args,**kwargs):
        mode=args[0] if args else kwargs.get('mode','r')
        if 'r' in mode or mode=='rb':
            assert not path.name.endswith('_labels.csv')
            assert not (path.suffix=='.npz' and ('A1' in path.parts or path.name.endswith('_features.npz')))
            assert path!=p.weights['mc3']
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',label_free)
    prediction,sequence=source.export_population(refit_model,pixels['final4'],row_index(read_public_rows(p.public_manifests['final4'])),
        protocol=p,device='cpu',context=c)
    assert prediction.probabilities.shape==(40,40)
    record=r.verify(sequence,'visual_sequence','refit');paths={Path(x).name:Path(x) for x in record.files}
    assert np.load(paths['sequence.npy'],mmap_mode='r').shape==(40,2,3,16,512)
    np.testing.assert_array_equal(np.load(paths['anchor_logits.npy']),prediction.logits)
